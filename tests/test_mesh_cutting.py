"""Test mesh cutting operations."""

import numpy as np


def test_cut_mesh_below():
    """Test cutting when mesh is entirely below plane (all vertices at z < cut_z)."""
    from src.mesh_ops import create_primitive, cut_mesh_at_z

    # Cube size=1 goes from z=[0,1]. Cut at z=5 means entire cube is below.
    cube = create_primitive("cube", size=1.0)

    bottom, top = cut_mesh_at_z(cube, z_height=5.0)

    assert bottom is not None
    assert top is None


def test_cut_mesh_above():
    """Test cutting when mesh is entirely above plane (all vertices at z > cut_z)."""
    from src.mesh_ops import create_primitive, cut_mesh_at_z

    # Cube size=1 goes from z=[0,1]. Cut at z=-5 means entire cube is above.
    cube = create_primitive("cube", size=1.0)

    bottom, top = cut_mesh_at_z(cube, z_height=-5.0)

    assert bottom is None
    assert top is not None


def test_cut_mesh_intersecting():
    """Test cutting when mesh intersects plane."""
    from src.mesh_ops import create_primitive, cut_mesh_at_z

    # Cube size=2 goes from z=[0,2]. Cut at z=1 splits it in half.
    cube = create_primitive("cube", size=2.0)

    bottom, top = cut_mesh_at_z(cube, z_height=1.0)

    assert bottom is not None
    assert top is not None

    # Verify the split is correct
    bottom_verts = np.asarray(bottom.vertices)[:, 2]
    top_verts = np.asarray(top.vertices)[:, 2]

    assert bottom_verts.max() <= 1.0  # Bottom should be at or below cut
    assert top_verts.min() >= 1.0     # Top should be at or above cut


def test_cut_mesh_empty():
    """Test cutting with empty mesh."""
    import open3d as o3d

    from src.mesh_ops import cut_mesh_at_z

    mesh = o3d.geometry.TriangleMesh()

    bottom, top = cut_mesh_at_z(mesh, z_height=0.0)

    assert bottom is None or len(np.asarray(bottom.vertices)) == 0
    assert top is None or len(np.asarray(top.vertices)) == 0


def _torus(R: float = 20.0, r: float = 8.0, n_u: int = 96, n_v: int = 48):
    """Parametric torus centred on the origin (tube centre circle at z=0)."""
    u = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
    v = np.linspace(0, 2 * np.pi, n_v, endpoint=False)
    U, V = np.meshgrid(u, v, indexing="ij")
    verts = np.stack(
        [(R + r * np.cos(V)) * np.cos(U), (R + r * np.cos(V)) * np.sin(U), r * np.sin(V)],
        axis=-1,
    ).reshape(-1, 3)
    i = np.arange(n_u * n_v).reshape(n_u, n_v)
    a = i.ravel()
    b = np.roll(i, -1, axis=0).ravel()
    c = np.roll(np.roll(i, -1, axis=0), -1, axis=1).ravel()
    d = np.roll(i, -1, axis=1).ravel()
    tris = np.vstack([np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)])
    return verts, tris


def test_cut_torus_caps_annulus_without_filling_hole():
    """A cut through a torus has a two-loop (annulus) cross-section.

    The caps must close the cut while leaving the central hole open, and each
    half must pass the full health check. Regression test: building caps with
    nested loops goes through trimesh's enclosure_tree, which needs rtree.
    """
    import open3d as o3d

    from src.mesh_ops import cut_mesh_at_z, mesh_health

    R, r = 20.0, 8.0
    verts, tris = _torus(R, r)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)

    bottom, top = cut_mesh_at_z(mesh, z_height=0.0)

    assert bottom is not None and top is not None
    expected_half_volume = np.pi**2 * R * r**2  # half of the 2*pi^2*R*r^2 torus
    for half in (bottom, top):
        report = mesh_health(half)
        assert report.ok, report.problems
        assert abs(report.volume - expected_half_volume) < 0.02 * expected_half_volume

        # No cap face may sit inside the central hole radius (R - r).
        v = np.asarray(half.vertices)
        t = np.asarray(half.triangles)
        on_plane = np.all(np.abs(v[t][:, :, 2]) < 1e-9, axis=1)
        assert on_plane.sum() > 0  # caps exist
        radii = np.hypot(*v[t[on_plane]].mean(axis=1)[:, :2].T)
        assert np.all(radii > R - r)


def test_cut_washer_has_no_sliver_faces():
    """Cutting the washer prism at 0.7 mm must leave clean, healthy halves.

    Regression test: its walls are diagonally triangulated quads, so the
    trimesh slicer produced 22+ zero-area sliver caps plus phantom
    self-intersections at the cut plane. The boolean-based cut removes them.
    """
    from pathlib import Path

    from src.mesh_ops import cut_mesh_at_z, load_mesh, mesh_health

    mesh = load_mesh(str(Path(__file__).parent / "fixtures" / "washer.stl"))
    total = mesh_health(mesh).volume

    bottom, top = cut_mesh_at_z(mesh, z_height=0.7)

    assert bottom is not None and top is not None
    rep_b, rep_t = mesh_health(bottom), mesh_health(top)
    assert rep_b.ok, rep_b.problems
    assert rep_t.ok, rep_t.problems
    assert abs(rep_b.volume + rep_t.volume - total) < 0.005 * total
    height = float(np.ptp(np.asarray(mesh.vertices)[:, 2]))
    fraction = rep_b.volume / (rep_b.volume + rep_t.volume)
    assert abs(fraction - 0.7 / height) < 0.02
