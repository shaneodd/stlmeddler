"""Tests for repair_mesh (hole capping, branch loops, face cleanup)."""

import numpy as np


def _cube_arrays():
    """Closed unit cube: 8 welded vertices, 12 outward-wound triangles."""
    verts = np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom 0-3
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # top 4-7
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],  # bottom (z=0)
            [4, 5, 6], [4, 6, 7],  # top (z=1)
            [0, 1, 5], [0, 5, 4],  # front (y=0)
            [1, 2, 6], [1, 6, 5],  # right (x=1)
            [3, 7, 6], [3, 6, 2],  # back (y=1)
            [0, 4, 7], [0, 7, 3],  # left (x=0)
        ],
        dtype=np.int64,
    )
    return verts, faces


def _as_o3d(verts: np.ndarray, faces: np.ndarray):
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.uint32))
    return mesh


def test_repair_fills_single_hole():
    """Removing one cube face leaves a triangular hole; repair must re-seal it."""
    from src.mesh_ops import mesh_health, repair_mesh

    verts, faces = _cube_arrays()
    broken = _as_o3d(verts, np.delete(faces, 2, axis=0))  # drop [4,5,6] -> top hole
    assert not mesh_health(broken).watertight

    repaired = repair_mesh(broken)
    report = mesh_health(repaired)
    assert report.ok, report.problems
    assert np.isclose(report.volume, 1.0)


def test_repair_splits_bowtie_holes_at_shared_vertex():
    """Two holes sharing exactly one corner vertex (the beanpod defect shape).

    Dropping the top triangle [4,5,6] and the front triangle [0,1,5] makes two
    triangular boundary loops that meet at vertex 5 - a degree-4 branch vertex.
    Repair must cap them as two separate holes and restore the closed cube.
    """
    from src.mesh_ops import mesh_health, repair_mesh

    verts, faces = _cube_arrays()
    dropped = ([4, 5, 6], [0, 1, 5])  # top + front triangles sharing vertex 5
    keep = [i for i, f in enumerate(faces) if not any((f == d).all() for d in dropped)]
    broken = _as_o3d(verts, faces[keep])

    from src.mesh_ops import _boundary_loops

    loops = _boundary_loops(verts, faces[keep])
    assert len(loops) == 2  # split at the shared vertex, not one tangled ring

    repaired = repair_mesh(broken)
    report = mesh_health(repaired)
    assert report.ok, report.problems
    assert np.isclose(report.volume, 1.0)
    assert len(repaired.triangles) == len(faces)  # two caps restore the 12-face cube


def test_repair_valid_mesh_is_a_no_op():
    """An already-valid mesh passes through with identical geometry."""
    from src.mesh_ops import create_primitive, repair_mesh

    cube = create_primitive("cube", size=2.0)
    repaired = repair_mesh(cube)
    assert len(repaired.triangles) == len(cube.triangles)
    assert np.allclose(np.asarray(repaired.vertices), np.asarray(cube.vertices))


def test_repair_removes_degenerate_and_duplicate_faces():
    """Zero-area and duplicated faces are dropped while the holes get capped."""
    from src.mesh_ops import mesh_health, repair_mesh

    verts, faces = _cube_arrays()
    keep = [i for i, f in enumerate(faces) if not (f == [4, 5, 6]).all()]
    dirty = np.vstack(
        [
            faces[keep],
            [[0, 1, 2]],       # duplicate of an existing face
            [[0, 0, 1]],       # degenerate: two identical vertices -> zero area
        ]
    )
    repaired = repair_mesh(_as_o3d(verts, dirty))
    report = mesh_health(repaired)
    assert report.ok, report.problems
    assert report.degenerate_faces == 0
    assert report.duplicate_faces == 0


def test_repair_empty_mesh_does_not_crash():
    from src.mesh_ops import repair_mesh

    empty = _as_o3d(np.empty((0, 3)), np.empty((0, 3), dtype=np.uint32))
    repaired = repair_mesh(empty)
    assert len(repaired.vertices) == 0
    assert len(repaired.triangles) == 0


def test_repair_flips_inverted_shell():
    """A closed but inside-out cube gets positive volume without new faces."""
    from src.mesh_ops import mesh_health, repair_mesh

    verts, faces = _cube_arrays()
    inverted = _as_o3d(verts, faces[:, ::-1])
    before = mesh_health(inverted)
    assert before.volume < 0.0  # sanity: the shell really is inside-out

    repaired = repair_mesh(inverted)
    report = mesh_health(repaired)
    assert report.ok, report.problems
    assert np.isclose(report.volume, 1.0)
