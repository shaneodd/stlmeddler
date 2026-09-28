"""Test ensure_normals function."""

import numpy as np


def test_ensure_normals_adds_normals():
    """Test that normals are added when missing."""
    import open3d as o3d

    from src.mesh_ops import ensure_normals

    verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0]
    ], dtype=np.float64)

    tris = np.array([[0, 1, 2]], dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)

    assert not mesh.has_vertex_normals()

    result = ensure_normals(mesh)

    assert result.has_vertex_normals()


def test_ensure_normals_preserves_existing():
    """Test that existing normals are preserved."""
    import open3d as o3d

    from src.mesh_ops import ensure_normals

    verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0]
    ], dtype=np.float64)

    tris = np.array([[0, 1, 2]], dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    mesh.compute_vertex_normals()

    original_normals = np.array(mesh.vertex_normals).copy()

    result = ensure_normals(mesh)

    assert result.has_vertex_normals()
    assert np.allclose(result.vertex_normals, original_normals)


# ---------------------------------------------------------------------- #
# mesh_health / is_watertight
# ---------------------------------------------------------------------- #
def _broken_fixture():
    from pathlib import Path

    from src.mesh_ops import load_mesh

    return load_mesh(str(Path(__file__).parent / "fixtures" / "test_cube_broken.stl"))


def test_health_valid_primitive_has_no_problems():
    from src.mesh_ops import create_primitive, mesh_health

    report = mesh_health(create_primitive("cube", size=1.0))
    assert report.ok
    assert report.problems == []
    assert report.volume > 0


def test_health_detects_broken_fixture():
    from src.mesh_ops import mesh_health

    report = mesh_health(_broken_fixture())
    assert not report.ok
    assert not report.watertight
    joined = " ".join(report.problems)
    assert "watertight" in joined.lower()


def test_health_counts_duplicate_and_degenerate_faces():
    from types import SimpleNamespace

    import numpy as np

    from src.mesh_ops import mesh_health

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    tris = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 3], [0, 0, 1]])  # dup, ok, ok, degenerate
    report = mesh_health(SimpleNamespace(vertices=verts, triangles=tris))
    assert report.duplicate_faces == 1
    assert report.degenerate_faces == 1
    assert not report.ok


def test_health_empty_mesh():
    from types import SimpleNamespace

    import numpy as np

    from src.mesh_ops import mesh_health

    empty = SimpleNamespace(vertices=np.empty((0, 3)), triangles=np.empty((0, 3), dtype=int))
    report = mesh_health(empty)
    assert not report.ok
    assert report.problems


def test_health_components():
    from types import SimpleNamespace

    import numpy as np

    from src.mesh_ops import mesh_health

    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
         [5, 0, 0], [6, 0, 0], [5, 1, 0], [5, 0, 1]], dtype=float
    )
    tris = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3],
                     [4, 5, 6], [4, 5, 7], [4, 6, 7], [5, 6, 7]])
    report = mesh_health(SimpleNamespace(vertices=verts, triangles=tris))
    assert report.components == 2


def test_health_self_intersection_skip():
    from src.mesh_ops import create_primitive, mesh_health

    cube = create_primitive("cube", size=1.0)
    skipped = mesh_health(cube, check_self_intersections=False)
    assert not skipped.self_intersection_checked
    assert not skipped.self_intersecting  # neutral, not a failure
    assert skipped.ok  # skipping must not fail an otherwise-valid mesh
    checked = mesh_health(cube, check_self_intersections=True)
    assert checked.self_intersection_checked
    assert checked.ok


def test_is_watertight_helper():
    from src.mesh_ops import create_primitive, is_watertight

    assert is_watertight(create_primitive("cube", size=1.0))
    assert not is_watertight(_broken_fixture())
