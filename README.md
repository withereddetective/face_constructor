# Face Constructor

The app uses the webcam and the bundled `3D-FaceReconstruction` project to
reconstruct a face that remains in view, then saves the mesh as OBJ and STL
files in the `output` folder.

## Setup

Install the app and reconstruction dependencies:

```powershell
python -m pip install PySide6 opencv-python numpy mediapipe open3d scipy PyYAML
```

Run the app from this repository:

```powershell
python app.py
```

The reconstruction model and mesh exporter are loaded from
`3D-FaceReconstruction`. The app displays the paths to the generated OBJ and
