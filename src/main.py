"""
Entry point for the STL/OBJ Processing Tool desktop application.
Launches the Qt GUI with a desktop OpenGL compatibility context.
"""

import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QSurfaceFormat
from PyQt5.QtWidgets import QApplication

# Ensure src package can be imported when running this script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gui.window import MainWindow


def _configure_opengl() -> None:
    """Request a desktop OpenGL compatibility context before any widget exists.

    MeshViewer uses the fixed-function pipeline (glBegin/glEnd, glMultMatrixf).
    Under Wayland/EGL some drivers default to a core or ES profile where those
    entry points are absent, which segfaults on first paint ("Wayland connection
    broke"). Forcing desktop OpenGL + CompatibilityProfile keeps them available.
    """
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)


def main():
    # Recommended for QOpenGLWidget on Wayland/macOS; must be set before QApplication.
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    _configure_opengl()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
