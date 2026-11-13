"""Animated busy dialog for long-running operations.

``run_busy()`` executes a callable in a worker thread while showing a small
modal dialog with an animated spinner, so the UI never looks hung during slow
mesh work (health checks, repairs, booleans, STEP export...).  The callable
must be pure computation (numpy / Open3D / trimesh) with no Qt access; its
result or exception is delivered back on the calling thread.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from PyQt5.QtCore import QEventLoop, QPointF, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QLabel, QWidget

# Only one busy operation may run at a time (module-level guard).
_active: _BusyDialog | None = None


class _SpinnerWidget(QWidget):
    """A rotating ring of fading dots painted with QPainter (no image assets)."""

    def __init__(self, parent: QWidget | None = None, size: int = 40) -> None:
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._angle = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def start(self) -> None:
        self._timer.start(40)

    def stop(self) -> None:
        self._timer.stop()

    def _tick(self) -> None:
        self._angle = (self._angle + 12.0) % 360.0
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        radius = max(4.0, min(cx, cy) - 5.0)
        dot = max(1.5, radius * 0.16)
        ticks = 12
        for i in range(ticks):
            angle = math.radians(self._angle + i * 360.0 / ticks)
            alpha = int(40 + 215 * (i / (ticks - 1)))  # leading dot brightest
            painter.setBrush(QColor(70, 130, 220, alpha))
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            painter.drawEllipse(QPointF(x, y), dot, dot)


class _Worker(QThread):
    """Runs a plain callable on a background thread and reports the outcome."""

    done = pyqtSignal(object)
    failed = pyqtSignal(object)

    def __init__(self, fn: Callable[[], Any]) -> None:
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            result = self._fn()
        except BaseException as exc:  # surface everything to the caller
            self.failed.emit(exc)
            return
        self.done.emit(result)


class _BusyDialog(QDialog):
    """Frameless modal dialog: spinner + task label. Cannot be dismissed."""

    def __init__(self, parent: QWidget | None, label: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("Working\u2026")
        # Title only - no close button; keyPressEvent swallows Esc as well.
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        self.setModal(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)
        self.spinner = _SpinnerWidget()
        self.spinner.start()
        text = QLabel(label)
        layout.addWidget(self.spinner)
        layout.addWidget(text)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        event.ignore()  # the task cannot be cancelled; ignore Esc


def _center_on_parent(dialog: QDialog, parent: QWidget | None) -> None:
    dialog.adjustSize()
    if parent is None:
        return
    frame = parent.window().frameGeometry()
    size = dialog.sizeHint()
    center = frame.center()
    dialog.move(center.x() - size.width() // 2, center.y() - size.height() // 2)


def run_busy(parent: QWidget | None, label: str, fn: Callable[[], Any]) -> Any:
    """Run *fn* in a worker thread while showing an animated busy dialog.

    Returns the result of ``fn()`` or re-raises its exception on the calling
    thread so existing error dialogs keep working unchanged.  Only one busy
    operation may be active at a time; nested calls raise RuntimeError.
    """
    global _active
    if _active is not None:
        raise RuntimeError("Another operation is already running")

    dialog = _BusyDialog(parent, label)
    worker = _Worker(fn)
    loop = QEventLoop()
    state: dict = {"result": None, "error": None}

    def finish(result: Any = None, error: BaseException | None = None) -> None:
        state["result"] = result
        state["error"] = error
        loop.quit()

    worker.done.connect(finish)
    worker.failed.connect(lambda exc: finish(error=exc))

    _center_on_parent(dialog, parent)
    _active = dialog
    try:
        dialog.show()
        worker.start()
        loop.exec_()
    finally:
        dialog.spinner.stop()
        worker.wait(30_000)  # run() already emitted; just reap it
        dialog.close()
        _active = None

    if state["error"] is not None:
        raise state["error"]
    return state["result"]
