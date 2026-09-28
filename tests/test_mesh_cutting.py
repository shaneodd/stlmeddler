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
