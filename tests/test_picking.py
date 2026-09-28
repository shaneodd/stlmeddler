import math

import numpy as np

from src.gui.picking import (
    modelview_matrix,
    ray_aabb,
    ray_triangles_nearest,
    screen_to_world_ray,
)


def _frustum(w: int = 400, h: int = 400):
    """Mirror the viewer's frustum so tests exercise a representative camera."""
    aspect = w / max(h, 1)
    fov = 45.0
    near = 0.1
    far = 500.0
    fovy = math.tan(math.radians(fov) / 2.0) * near
    return (-fovy * aspect, fovy * aspect, -fovy, fovy, near, far)


def _ray(x_px: float, y_px: float, **overrides):
    left, right, bottom, top, near, far = _frustum()
    params = dict(
        width=400, height=400, zoom=1.0, rot_x_deg=0.0, rot_y_deg=0.0, rot_z_deg=0.0,
        left=left, right=right, bottom=bottom, top=top, near=near, far=far,
    )
    params.update(overrides)
    return screen_to_world_ray(x_px=x_px, y_px=y_px, **params)


def test_center_ray_points_down_minus_z():
    ray = _ray(200.0, 200.0)
    assert ray is not None
    origin, direction = ray
    assert direction[2] < -0.99 and abs(direction[0]) < 1e-6 and abs(direction[1]) < 1e-6
    assert origin[2] > 0


def test_offcenter_ray_tilts():
    _, d_right = _ray(380.0, 200.0)
    assert d_right[0] > 0  # right of centre -> +X (no camera rotation)
    _, d_up = _ray(200.0, 20.0)
    assert d_up[1] > 0  # near top of widget -> +Y


def test_degenerate_viewport_returns_none():
    assert _ray(0.0, 0.0, width=0) is None


def test_modelview_default_translation():
    m = modelview_matrix(zoom=1.0, rot_x_deg=0.0, rot_y_deg=0.0, rot_z_deg=0.0)
    eye = m @ np.array([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(eye[:3], [0.0, 0.0, -3.0])  # camera pushed back by 3 * zoom


def test_modelview_center_is_orbit_pivot():
    m = modelview_matrix(
        zoom=1.0, rot_x_deg=0.0, rot_y_deg=0.0, rot_z_deg=0.0, center=(2.0, 3.0, 4.0)
    )
    eye = m @ np.array([2.0, 3.0, 4.0, 1.0])
    assert np.allclose(eye[:3], [0.0, 0.0, -3.0])  # pivot maps to the screen centre


def test_center_ray_passes_through_pivot():
    ray = _ray(200.0, 200.0, zoom=10.0, center=(2.0, 3.0, 4.0))
    assert ray is not None
    origin, direction = ray
    t = (4.0 - origin[2]) / direction[2]  # straight ahead, no camera rotation
    hit = origin + t * direction
    assert np.allclose(hit, [2.0, 3.0, 4.0], atol=1e-6)


def test_ray_triangles_hit_and_miss():
    verts = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=float)
    tris = np.array([[0, 1, 2]])
    hit = ray_triangles_nearest(
        np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, -1.0]), verts, tris
    )
    assert hit is not None
    t, tri = hit
    assert abs(t - 5.0) < 1e-6 and tri == 0
    miss = ray_triangles_nearest(
        np.array([10.0, 10.0, 5.0]), np.array([0.0, 0.0, -1.0]), verts, tris
    )
    assert miss is None


def test_ray_triangles_nearest_returns_triangle_row():
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 2], [1, 0, 2], [0, 1, 2]], dtype=float
    )
    tris = np.array([[0, 1, 2], [3, 4, 5]])
    hit = ray_triangles_nearest(
        np.array([0.1, 0.1, 5.0]), np.array([0.0, 0.0, -1.0]), verts, tris
    )
    assert hit is not None
    t, tri = hit
    assert abs(t - 3.0) < 1e-6 and tri == 1  # upper triangle (z=2) is hit first
    assert ray_triangles_nearest(
        np.array([9.0, 9.0, 5.0]), np.array([0.0, 0.0, -1.0]), verts, tris
    ) is None


def test_ray_aabb_hit_and_miss():
    bmin = np.array([-1.0, -1.0, -1.0])
    bmax = np.array([1.0, 1.0, 1.0])
    assert ray_aabb(np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, -1.0]), bmin, bmax)
    assert not ray_aabb(np.array([9.0, 9.0, 5.0]), np.array([0.0, 0.0, -1.0]), bmin, bmax)
