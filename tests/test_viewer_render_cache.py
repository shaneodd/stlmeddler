"""Headless tests for viewer helpers: render-array cache and plate metrics."""

import math

import numpy as np


def _obj():
    verts = np.array(
        [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float
    )
    tris = np.array([[0, 2, 1], [1, 2, 3]])
    return {"verts": verts, "tris": tris}


def test_render_cache_shapes_and_normals():
    from src.gui.viewer import MeshViewer

    obj = _obj()
    pos, nor = MeshViewer._render_cache(obj)
    assert pos.shape == (6, 3) and nor.shape == (6, 3)
    assert pos.dtype == np.float32 and nor.dtype == np.float32
    assert pos.flags["C_CONTIGUOUS"] and nor.flags["C_CONTIGUOUS"]
    # Row 0 of tris is [[0,2,1],[1,2,3]] -> first expanded triangle.
    assert np.allclose(pos[:3], obj["verts"][[0, 2, 1]], atol=1e-7)
    # Flat shading: the three normals of a triangle are identical and unit length.
    assert np.allclose(nor[0], nor[1]) and np.allclose(nor[1], nor[2])
    assert abs(np.linalg.norm(nor[0]) - 1.0) < 1e-6
    # Face (0,2,1) has winding normal -Z; cached normal should match its direction.
    assert nor[0][2] < -0.99


def test_render_cache_hit_and_invalidation():
    from src.gui.viewer import MeshViewer

    obj = _obj()
    first = MeshViewer._render_cache(obj)
    assert MeshViewer._render_cache(obj) is first  # cache hit, no rebuild
    obj["tris"] = np.array([[0, 2, 1]])  # geometry change invalidates by length
    second = MeshViewer._render_cache(obj)
    assert second is not first
    assert second[0].shape == (3, 3)


def test_plate_metrics_default_when_empty():
    from src.gui.viewer import MeshViewer

    assert MeshViewer._plate_metrics(0.0) == (10.0, 1.0)
    assert MeshViewer._plate_metrics(float("nan")) == (10.0, 1.0)


def test_plate_metrics_covers_scene_with_round_steps():
    from src.gui.viewer import MeshViewer

    for extent in (0.5, 2.0, 37.0, 500.0, 12345.0):
        half, step = MeshViewer._plate_metrics(extent)
        assert half >= extent, extent  # plate must cover the scene
        assert step > 0 and half % step < 1e-9 * half  # grid lands on round steps
        lines = half / step
        assert 5 <= lines <= 14, (extent, half, step)  # neither sparse nor dense
        # step is one of 1/2/5 x 10^n
        mag = 10.0 ** math.floor(math.log10(step))
        assert round(step / mag, 6) in (1.0, 2.0, 5.0, 10.0), (extent, step)
