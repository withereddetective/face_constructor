import sys
import threading
import time
from pathlib import Path
from typing import Any
import cv2
from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QHBoxLayout,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


FACE_VISIBLE_SECONDS = 3.0
RESULT_DISPLAY_SECONDS = 4.0
DETECTION_INTERVAL_SECONDS = 0.1


class FaceReconstructionWorker(QThread):
    hold_started = Signal(object)
    result_ready = Signal(str, str)
    preview_resumed = Signal()
    error_occurred = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._frame_lock = threading.Lock()
        self._boxes_lock = threading.Lock()
        self._latest_frame: Any = None
        self._latest_boxes: list[tuple[int, int, int, int]] = []

    def update_frame(self, frame: Any) -> None:
        with self._frame_lock:
            self._latest_frame = frame

    def latest_boxes(self) -> list[tuple[int, int, int, int]]:
        with self._boxes_lock:
            return list(self._latest_boxes)

    def _set_boxes(self, boxes: list[tuple[int, int, int, int]]) -> None:
        with self._boxes_lock:
            self._latest_boxes = boxes

    def _current_frame(self) -> Any:
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    @staticmethod
    def _face_box(
        landmarks: Any,
        frame_shape: Any,
    ) -> tuple[int, int, int, int] | None:
        frame_height, frame_width = frame_shape[:2]
        left = max(0, int(float(landmarks[:, 0].min()) * frame_width))
        top = max(0, int(float(landmarks[:, 1].min()) * frame_height))
        right = min(frame_width, int(float(landmarks[:, 0].max()) * frame_width))
        bottom = min(frame_height, int(float(landmarks[:, 1].max()) * frame_height))
        if right <= left or bottom <= top:
            return None
        return left, top, right - left, bottom - top

    @staticmethod
    def _intersection_over_union(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        left = max(ax, bx)
        top = max(ay, by)
        right = min(ax + aw, bx + bw)
        bottom = min(ay + ah, by + bh)
        intersection = max(0, right - left) * max(0, bottom - top)
        union = aw * ah + bw * bh - intersection
        return intersection / union if union else 0.0

    @staticmethod
    def _create_pipeline() -> Any:
        project_dir = Path(__file__).resolve().parent / "3D-FaceReconstruction"
        if not (project_dir / "main.py").is_file():
            raise FileNotFoundError(
                f"3D-FaceReconstruction was not found at {project_dir}"
            )

        project_path = str(project_dir)
        if project_path not in sys.path:
            sys.path.insert(0, project_path)

        from main import FaceReconstruction3D

        pipeline = FaceReconstruction3D(config_path=str(project_dir / "config.yaml"))
        output_dir = Path(__file__).resolve().parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        pipeline.mesh_exporter.output_dir = output_dir
        return pipeline

    def run(self) -> None:
        pipeline = None
        try:
            pipeline = self._create_pipeline()
            visible_since: float | None = None
            tracked_box: tuple[int, int, int, int] | None = None

            while not self.isInterruptionRequested():
                frame = self._current_frame()
                if frame is None:
                    self.msleep(50)
                    continue

                _, mesh = pipeline.process_frame(frame)
                boxes: list[tuple[int, int, int, int]] = []
                if mesh is not None and pipeline.current_landmarks is not None:
                    box = self._face_box(pipeline.current_landmarks, frame.shape)
                    if box is not None:
                        boxes.append(box)
                self._set_boxes(boxes)

                now = time.monotonic()
                candidate = max(boxes, key=lambda box: box[2] * box[3]) if boxes else None
                if candidate is None:
                    visible_since = None
                    tracked_box = None
                elif tracked_box is None:
                    tracked_box = candidate
                    visible_since = now
                else:
                    best_match = max(
                        boxes,
                        key=lambda box: self._intersection_over_union(tracked_box, box),
                    )
                    if self._intersection_over_union(tracked_box, best_match) < 0.15:
                        tracked_box = candidate
                        visible_since = now
                    else:
                        tracked_box = best_match

                if (
                    visible_since is not None
                    and now - visible_since >= FACE_VISIBLE_SECONDS
                ):
                    frozen_frame = frame.copy()
                    if tracked_box is not None:
                        x, y, width, height = tracked_box
                        cv2.rectangle(
                            frozen_frame,
                            (x, y),
                            (x + width, y + height),
                            (70, 220, 120),
                            2,
                        )
                    self.hold_started.emit(frozen_frame)
                    self._set_boxes([])
                    filename = f"face_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
                    obj_path = pipeline.mesh_exporter.export_mesh(
                        mesh,
                        filename,
                        format="obj",
                        include_texture=False,
                    )
                    stl_path = pipeline.mesh_exporter.export_mesh(
                        mesh,
                        filename,
                        format="stl",
                        include_texture=False,
                    )
                    self.result_ready.emit(obj_path, stl_path)
                    self.msleep(int(RESULT_DISPLAY_SECONDS * 1000))
                    if self.isInterruptionRequested():
                        break
                    visible_since = None
                    tracked_box = None
                    self.preview_resumed.emit()

                self.msleep(int(DETECTION_INTERVAL_SECONDS * 1000))
        except Exception as exc:
            self.error_occurred.emit(f"3D face reconstruction failed: {exc}")
        finally:
            if pipeline is not None:
                try:
                    pipeline.cleanup()
                except Exception as exc:
                    self.error_occurred.emit(
                        f"Could not close the reconstruction pipeline: {exc}"
                    )


class FaceWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("3D Face Reconstruction")
        self.resize(960, 640)
        self.setMinimumSize(640, 480)
        self.setStyleSheet(
            "QWidget { background: #101114; color: white; }"
            "QLabel { color: white; }"
        )

        self._camera = cv2.VideoCapture(0)
        self._holding = False
        self._failed = False
        self._error_label = QLabel()
        self._error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._error_label.setWordWrap(True)

        self._video_label = QLabel("Starting camera…")
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setStyleSheet("QLabel { background: black; }")

        self._result_page = self._make_result_page()
        self._pages = QStackedWidget()
        self._pages.addWidget(self._video_label)
        self._pages.addWidget(self._result_page)
        self._pages.addWidget(self._error_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._pages)

        self._reconstruction = FaceReconstructionWorker()
        self._reconstruction.hold_started.connect(self._begin_reconstruction)
        self._reconstruction.result_ready.connect(self._show_result)
        self._reconstruction.preview_resumed.connect(self._resume_preview)
        self._reconstruction.error_occurred.connect(self._show_error)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_preview)

        if not self._camera.isOpened():
            self._show_error("Could not open the camera. Check that it is connected and available.")
        else:
            self._reconstruction.start()
            self._timer.start(30)

    def _make_result_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(40)

        self._still_label = QLabel()
        self._still_label.setFixedSize(320, 240)
        self._still_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._still_label, 0, Qt.AlignmentFlag.AlignTop)

        message_layout = QVBoxLayout()
        message_layout.addStretch(1)
        self._result_title = QLabel()
        self._result_title.setStyleSheet("font-size: 36px; font-weight: 600;")
        self._result_message = QLabel()
        self._result_message.setWordWrap(True)
        self._result_message.setStyleSheet("font-size: 22px; color: #c9cbd1;")
        message_layout.addWidget(self._result_title)
        message_layout.addWidget(self._result_message)
        message_layout.addStretch(1)
        layout.addLayout(message_layout, 1)
        return page

    def _update_preview(self) -> None:
        ok, frame = self._camera.read()
        if not ok:
            self._timer.stop()
            self._show_error("The camera stopped providing frames.")
            return

        self._reconstruction.update_frame(frame)
        if self._holding:
            return

        display_frame = frame.copy()
        for x, y, width, height in self._reconstruction.latest_boxes():
            cv2.rectangle(
                display_frame,
                (x, y),
                (x + width, y + height),
                (70, 220, 120),
                2,
            )
        self._video_label.setPixmap(self._to_pixmap(display_frame, self._video_label.size()))

    @staticmethod
    def _to_pixmap(frame: Any, size: Any) -> QPixmap:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb_frame.shape
        image = QImage(
            rgb_frame.data,
            width,
            height,
            rgb_frame.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        return QPixmap.fromImage(image).scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _begin_reconstruction(self, frame: Any) -> None:
        if self._failed:
            return
        self._holding = True
        self._still_label.setPixmap(self._to_pixmap(frame, self._still_label.size()))
        self._result_title.setText("Reconstructing face…")
        self._result_message.setText("Saving OBJ and STL files…")
        self._pages.setCurrentWidget(self._result_page)

    def _show_result(self, obj_path: str, stl_path: str) -> None:
        if self._failed:
            return
        self._result_title.setText("3D face model created")
        self._result_message.setText(f"OBJ file:\n{obj_path}\n\nSTL file:\n{stl_path}")

    def _resume_preview(self) -> None:
        if self._failed:
            return
        self._holding = False
        self._pages.setCurrentWidget(self._video_label)

    def _show_error(self, message: str) -> None:
        self._failed = True
        self._timer.stop()
        if self._reconstruction.isRunning():
            self._reconstruction.requestInterruption()
        self._error_label.setText(message)
        self._pages.setCurrentWidget(self._error_label)

    def closeEvent(self, event: Any) -> None:
        self._timer.stop()
        if self._reconstruction.isRunning():
            self._reconstruction.requestInterruption()
            self._reconstruction.wait()
        self._camera.release()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = FaceWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())