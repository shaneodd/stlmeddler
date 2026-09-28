"""Headless tests for face-pick mode UI state: banner, hover, cancellation.

Runs Qt on the offscreen platform and never shows/paints the widget, so no
OpenGL context is needed.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

BOX_TRIS = np.array(
    [
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ]
)


def _box():
    verts = np.array(
        [
            [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],
            [0, 0, 3], [2, 0, 3], [2, 2, 3], [0, 2, 3],
        ],
        dtype=float,
    )
    return SimpleNamespace(vertices=verts, triangles=BOX_TRIS)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _viewer(qapp):
    from src.gui.viewer import MeshViewer

    viewer = MeshViewer()
    viewer.resize(400, 400)
    return viewer


def test_banner_shows_only_while_armed(qapp):
    viewer = _viewer(qapp)
    assert not viewer._help_label.isVisibleTo(viewer)
    viewer.start_face_pick_mode()
    assert viewer._help_label.isVisibleTo(viewer)
    viewer.cancel_face_pick_mode()
    assert not viewer._help_label.isVisibleTo(viewer)


def test_hover_records_face_and_clears_on_exit(qapp):
    viewer = _viewer(qapp)
    viewer.add_object(_box(), name="box", auto_align=False)
    viewer.start_face_pick_mode()
    viewer._last_hover_at = 0.0
    viewer._update_hover(200, 200)  # centre of the framed box
    assert viewer._hover_face is not None
    assert viewer._hover_face[0] == 0
    assert 0 <= viewer._hover_face[1] < len(BOX_TRIS)
    viewer.cancel_face_pick_mode()
    assert viewer._hover_face is None


def test_hover_misses_empty_space(qapp):
    viewer = _viewer(qapp)
    viewer.add_object(_box(), name="box", auto_align=False)
    viewer.start_face_pick_mode()
    viewer._last_hover_at = 0.0
    viewer._update_hover(2, 2)  # far corner: no geometry behind the cursor
    assert viewer._hover_face is None


def test_hover_throttled_within_min_interval(qapp):
    viewer = _viewer(qapp)
    viewer.add_object(_box(), name="box", auto_align=False)
    viewer.start_face_pick_mode()
    viewer._last_hover_at = 0.0
    viewer._update_hover(200, 200)
    assert viewer._hover_face is not None
    viewer._hover_face = None
    viewer._hover_pos_px = (0, 0)
    viewer._last_hover_at = float("inf")  # pretend the interval has not elapsed
    viewer._update_hover(200, 200)
    assert viewer._hover_face is None  # skipped by the throttle
