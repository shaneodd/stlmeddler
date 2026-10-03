# STL Meddler

![CI](https://github.com/shaneodd/stlmeddler/actions/workflows/ci.yml/badge.svg?branch=master)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **cross-platform desktop GUI** for loading, visualising, and manipulating STL/OBJ meshes.
Built with **PyQt5** for the interface, **Open3D** for mesh I/O, **trimesh** for boolean
operations, and **PyOpenGL** for rendering.

## Screenshots

<!-- Add a screenshot at docs/screenshot.png and uncomment:
![Screenshot](docs/screenshot.png)
-->

## Features

- Load STL/OBJ meshes and generate primitives (cube, sphere, cylinder, cone)
- Multiple objects per scene, each with its own transform and colour
- Move, rotate and resize (target width/depth/height in mm, with optional proportion lock)
- 3-D click-to-select, face-pick alignment, and measure-between-two-points tools
- Boolean union/difference between any two objects in the scene
- Horizontal cut plane with build-plate alignment
- Scene persistence (`.stlproj` project bundles) and merged single-mesh export (STL/OBJ/3MF)

## Installation

Requires **Python 3.12** (for full Open3D compatibility).

```bash
python3 -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate.bat
pip install -e .
```

Or with Anaconda:

```bash
conda create -n stlmeddler python=3.12
conda activate stlmeddler
pip install -e .
```

For development (tests, linting, type checking):

```bash
pip install -e ".[dev]"   # or: pip install -r requirements.txt -r requirements-dev.txt
```

## Quick Start

```bash
python src/main.py
```

## Building a Stand-alone Executable

To distribute the tool without requiring Python on the target machine, use **PyInstaller**:

```bash
pip install pyinstaller
pyinstaller --onefile src/main.py
```

The generated binary will be placed in `dist/` (e.g., `dist/main`).

## Running Tests

All tests are headless (no Qt/GL context required) and run on Python 3.12:

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_load_save.py

# With coverage report
pytest --cov=src --cov-report=term-missing
```

### Test Structure

- `tests/fixtures/` - Test data files (STL, OBJ meshes)
- `tests/test_load_save.py` - Tests for mesh loading and saving
- `tests/test_save.py` - Tests for STL/OBJ/3MF export
- `tests/test_primitives.py` - Tests for primitive shape generation
- `tests/test_boolean_ops.py` - Tests for boolean operations
- `tests/test_mesh_cutting.py` - Tests for mesh cutting functionality
- `tests/test_mesh_ops_basic.py` - Tests for core mesh_ops helpers
- `tests/test_picking.py` - Pure-maths tests for camera modelview, ray casting and triangle intersection
- `tests/test_scene.py` - Headless tests for the multi-object SceneModel (object management, transform isolation, face-to-plate alignment, cut integration, export consistency)
- `tests/test_workflow.py` - End-to-end operation workflows

## Project Structure

```
stlmeddler/
├─ pyproject.toml            # Packaging, ruff/mypy/pytest configuration
├─ requirements.txt          # pip package list
├─ README.md                 # This file
├─ src/                      # Source code
│   ├─ __init__.py           # Package init
│   ├─ main.py               # Entry point / GUI launch
│   ├─ mesh_ops.py           # Mesh operations (load, save, boolean ops, primitives, cut)
│   └─ gui/                  # GUI package
│       ├─ __init__.py       # Init for GUI subpackage
│       ├─ picking.py        # Pure-maths camera/ray-picking helpers (no Qt/GL)
│       ├─ scene.py          # Pure-Python multi-object SceneModel (transforms, alignment, export)
│       ├─ viewer.py         # OpenGL mesh viewer widget (Qt facade over SceneModel)
│       ├─ primitive_dialog.py # Primitive creation dialog (shape + dimensions)
│       └─ window.py         # MainWindow UI definition / controller
└─ tests/                    # Unit / integration tests
```

The viewer supports **independent manipulation of multiple objects**: each loaded or
generated mesh is a separate selectable object with its own transform (rotation +
translation) and colour; the active object is highlighted, and the camera
automatically frames it when objects are opened, selected or modified. Shapes are
shaded with per-face normals so geometry reads as 3-D from any angle. Operations
such as cut replace an object with new parts, and export merges every object's
transform into a single saved mesh.

## Architecture

The code is organised in thin, testable layers:

```
mesh_ops.py        Pure mesh operations (load/save, primitives, booleans, cut).
                   No Qt/GL dependency - the foundation everything builds on.
       |
gui/scene.py       SceneModel: owns the ordered object list (base geometry + a
                   per-object transform and colour) plus transforms, build-plate
                   alignment (including face-to-plate), cut and merged-export logic.
                   Pure Python/numpy; tested headlessly.
       |
gui/picking.py     Camera modelview + ray-cast maths shared by rendering and
                   selection, so clicks always match what is drawn. Pure numpy.
       |
gui/viewer.py      MeshViewer: QOpenGLWidget facade that draws every object lit and
                   shaded (per-face normals), frames the active object by orbit
                   pivot, highlights it with a glow, and handles click-to-select,
                   face-pick mode, camera orbit/zoom. Delegates all state to SceneModel.
       |
gui/window.py      MainWindow: Qt controller - menus, object list, transform spin boxes
                   and operation handlers. Drives the viewer; holds no geometry itself.
```

**Data flow.** Opening a file or creating a primitive calls `viewer.add_object`, which
adds to the `SceneModel` and emits `active_changed`; the window mirrors that into the
"Loaded Objects" list. Selecting a row or clicking an object in the 3-D view calls
`viewer.set_active`, which reframes the camera on that object and repopulates the
transform spin boxes; editing a spin box writes back to the active object's transform.
The same panel resizes objects: the Width/Depth/Height boxes take target sizes in mm
(intrinsic bounding-box dimensions, so they are unaffected by rotation), and a
"Lock proportions" checkbox scales all three axes together when one is edited.
Scaling is applied about the object's local origin, so re-use Move or
**Center on Build Plate** afterwards to reposition.
**Align Face to Build Plate** (Operations menu) arms face-pick mode: the next face
clicked in the viewer is rotated flat onto the plate. Cut replaces an object with new
parts, and export merges every object's transform into one mesh via `save_mesh`.

## Additional Notes

**Note:** The project requires Python 3.12 for full Open3D compatibility. Use `conda create -n stlmeddler python=3.12` to create a compatible environment.

### OpenGL Requirements

The GUI relies on **PyQt5** and **PyOpenGL**; ensure your system has the appropriate OpenGL drivers. On Linux, you may need:
- `mesa-libGL` (or equivalent for your distribution)
- Proper GPU drivers (NVIDIA/AMD)

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
