"""Pure-Python ray-picking maths for 3D click-to-select.

No Qt/OpenGL here: given a camera description (the same parameters the viewer uses
to render) and a pixel, :func:`screen_to_world_ray` produces a world-space ray that
matches what is drawn; :func:`ray_aabb` and :func:`ray_triangles_nearest` intersect
it against geometry. Keeping this display-free makes selection testable headlessly.

Matrices use the column-vector convention (``clip = P @ M @ v``) to match OpenGL's
fixed-function pipeline exactly, so picking stays aligned with rendering.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

EPS = 1e-9


def perspective_from_frustum(
    left: float, right: float, bottom: float, top: float, near: float, far: float
) -> np.ndarray:
    """Build the OpenGL projection matrix for ``glFrustum(left,right,bottom,top,near,far)``."""
    rl = right - left
    tb = top - bottom
    fn = far - near
    p = np.zeros((4, 4), dtype=float)
    p[0, 0] = 2.0 * near / rl
    p[0, 2] = (right + left) / rl
    p[1, 1] = 2.0 * near / tb
    p[1, 2] = (top + bottom) / tb
    p[2, 2] = -(far + near) / fn
    p[2, 3] = -2.0 * far * near / fn
    p[3, 2] = -1.0
    return p


def _rot_x(a_deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(a_deg)), math.sin(math.radians(a_deg))
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=float)


def _rot_y(a_deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(a_deg)), math.sin(math.radians(a_deg))
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=float)


def _rot_z(a_deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(a_deg)), math.sin(math.radians(a_deg))
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)


def modelview_matrix(
    zoom: float,
    rot_x_deg: float,
    rot_y_deg: float,
    rot_z_deg: float,
    center: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Replicate the viewer's modelview: ``T(0,0,-3*zoom) @ Rx @ Ry @ Rz @ T(-center)``.

    ``center`` is the orbit pivot: the world point the camera looks at, which the
    view maps onto the eye-space origin before being pushed back by ``3 * zoom``.
    """
    t = np.eye(4, dtype=float)
    t[2, 3] = -3.0 * zoom
    t_center = np.eye(4, dtype=float)
    t_center[:3, 3] = -np.asarray(center, dtype=float)
    return t @ _rot_x(rot_x_deg) @ _rot_y(rot_y_deg) @ _rot_z(rot_z_deg) @ t_center


def _unproject(inv_mvp: np.ndarray, ndc_x: float, ndc_y: float, ndc_z: float) -> np.ndarray | None:
    clip = np.array([ndc_x, ndc_y, ndc_z, 1.0], dtype=float)
    p = inv_mvp @ clip
    if abs(p[3]) < EPS:
        return None
    return p[:3] / p[3]


def screen_to_world_ray(
    *,
    width: int,
    height: int,
    x_px: float,
    y_px: float,
    zoom: float,
    rot_x_deg: float,
    rot_y_deg: float,
    rot_z_deg: float,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    left: float,
    right: float,
    bottom: float,
    top: float,
    near: float,
    far: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(origin, direction)`` of the world-space ray through pixel (x_px, y_px).

    ``direction`` is unit length. ``center`` is the orbit pivot used by the viewer.
    Returns None for a degenerate viewport or ray.
    """
    if width <= 0 or height <= 0:
        return None
    proj = perspective_from_frustum(left, right, bottom, top, near, far)
    modelview = modelview_matrix(zoom, rot_x_deg, rot_y_deg, rot_z_deg, center)
    inv_mvp = np.linalg.inv(proj @ modelview)

    ndc_x = 2.0 * x_px / width - 1.0
    ndc_y = 1.0 - 2.0 * y_px / height  # Qt's y is top-down; GL's is bottom-up

    near_pt = _unproject(inv_mvp, ndc_x, ndc_y, -1.0)
    far_pt = _unproject(inv_mvp, ndc_x, ndc_y, 1.0)
    if near_pt is None or far_pt is None:
        return None

    direction = far_pt - near_pt
    norm = float(np.linalg.norm(direction))
    if norm < EPS:
        return None
    return near_pt, direction / norm


def ray_aabb(
    origin: np.ndarray, direction: np.ndarray, bmin: np.ndarray, bmax: np.ndarray
) -> bool:
    """Slab test: True if the ray (t >= 0) intersects the axis-aligned box."""
    tmin = -np.inf
    tmax = np.inf
    for i in range(3):
        d = direction[i]
        if abs(d) < EPS:
            if origin[i] < bmin[i] or origin[i] > bmax[i]:
                return False
        else:
            ta = (bmin[i] - origin[i]) / d
            tb = (bmax[i] - origin[i]) / d
            if ta > tb:
                ta, tb = tb, ta
            tmin = max(tmin, ta)
            tmax = min(tmax, tb)
            if tmin > tmax:
                return False
    return tmax >= 0.0


def ray_triangles_nearest(
    origin: np.ndarray, direction: np.ndarray, verts: np.ndarray, tris: np.ndarray
) -> tuple[float, int] | None:
    """Nearest positive intersection of a ray with triangles (two-sided).

    Vectorised Moeller-Trumbore. Returns ``(t, triangle_row)`` for the closest hit
    or None if no hit.
    """
    if len(tris) == 0 or len(verts) == 0:
        return None
    v0 = verts[tris[:, 0]]
    e1 = verts[tris[:, 1]] - v0
    e2 = verts[tris[:, 2]] - v0

    h = np.cross(direction, e2)
    a = np.einsum("ij,ij->i", e1, h)
    valid = np.abs(a) > EPS
    if not np.any(valid):
        return None

    s = origin - v0
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.einsum("ij,ij->i", s, h) / a
        q = np.cross(s, e1)
        v = np.einsum("j,ij->i", direction, q) / a
        t = np.einsum("ij,ij->i", e2, q) / a
        hit = (
            valid & (u >= -EPS) & (u <= 1.0 + EPS) & (v >= -EPS)
            & ((u + v) <= 1.0 + EPS) & (t > EPS)
        )

    if not np.any(hit):
        return None
    hit_indices = np.flatnonzero(hit)
    best = hit_indices[np.argmin(t[hit_indices])]
    return float(t[best]), int(best)
