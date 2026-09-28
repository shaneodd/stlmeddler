"""Headless tests for the measure-mode helpers and interaction state machine."""

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
BOX_VERTS = np.array(
    [
        [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],
        [0, 0, 3], [2, 0, 3], [2, 2, 3], [0, 2, 3],
    ],
    dtype=float,
)


def _box():
    return SimpleNamespace(vertices=BOX_VERTS.copy(), triangles=BOX_TRIS)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _viewer(qapp):
    from src.gui.viewer import MeshViewer

    viewer = MeshViewer()
    viewer.resize(400, 400)
    return viewer


def test_world_per_pixel_scales_with_distance():
    from src.gui.viewer import MeshViewer

    near = MeshViewer._world_per_pixel(3.0, 400)
    far = MeshViewer._world_per_pixel(30.0, 400)
    assert far == pytest.approx(near * 10)
    assert MeshViewer._world_per_pixel(3.0, 0) == 0.0


def test_snap_to_vertex_snaps_near_corner():
    from src.gui.viewer import MeshViewer

    tri = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=float)
    snapped = MeshViewer._snap_to_vertex(np.array([0.05, 0.05, 0.0]), tri, threshold=0.2)
    assert np.allclose(snapped, [0, 0, 0])


def test_snap_to_vertex_keeps_raw_point_when_far():
    from src.gui.viewer import MeshViewer

    tri = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=float)
    raw = MeshViewer._snap_to_vertex(np.array([0.7, 0.7, 0.0]), tri, threshold=0.2)
    assert np.allclose(raw, [0.7, 0.7, 0.0])


def test_measure_state_machine(qapp):
    viewer = _viewer(qapp)
    viewer.add_object(_box(), name="box", auto_align=False)

    signals = []
    viewer.measure_done.connect(lambda: signals.append(1))

    viewer.start_measure_mode()
    assert viewer.measure_mode
    assert "first point" in viewer._help_label.text()

    # First click near a top corner -> sets A (snapped to a vertex).
    viewer._measure_click(200, 200)
    assert viewer._measure_a is not None
    assert viewer._measure_b is None
    assert "second point" in viewer._help_label.text()

    # Second click on a different corner -> sets B and reports a distance in mm.
    viewer._measure_click(230, 200)
    assert viewer._measure_b is not None
    assert "mm" in viewer._help_label.text()

    # A third click starts a fresh measurement (B cleared back to pending).
    viewer._measure_click(180, 210)
    assert viewer._measure_b is None
    assert "second point" in viewer._help_label.text()

    viewer.cancel_measure_mode()
    assert not viewer.measure_mode
    assert viewer._measure_a is None
    assert signals == [1]


def test_measure_click_ignores_empty_space(qapp):
    viewer = _viewer(qapp)
    viewer.add_object(_box(), name="box", auto_align=False)
    viewer.start_measure_mode()
    viewer._measure_click(3, 3)  # empty corner: no geometry
    assert viewer._measure_a is None


def test_starting_measure_cancels_face_pick(qapp):
    viewer = _viewer(qapp)
    viewer.add_object(_box(), name="box", auto_align=False)
    viewer.start_face_pick_mode()
    viewer.start_measure_mode()
    assert viewer.measure_mode
    assert not viewer.face_pick_mode
