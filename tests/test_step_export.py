"""Headless tests for the mesh -> STEP export pipeline (src/step_ops.py).

Exercises solid construction, open-shell handling, decimation and STEP
re-import without touching Qt. Skips cleanly when cadquery-ocp is absent.
"""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")


@pytest.fixture(scope="module")
def ocp_available():
    from src.step_ops import _load_ocp

    try:
        _load_ocp()
    except RuntimeError as e:
        pytest.skip(str(e))


def _open_tube():
    cyl = trimesh.creation.cylinder(radius=5.0, height=10.0, sections=32)
    z = cyl.vertices[:, 2]
    caps = (z[cyl.faces].max(axis=1) - z[cyl.faces].min(axis=1)) < 1e-9
    return cyl.vertices, cyl.faces[~caps]


# --------------------------------------------------------------------- #
# Closed meshes -> solids
# --------------------------------------------------------------------- #
def test_box_exports_as_valid_solid(tmp_path, ocp_available):
    from src.step_ops import mesh_to_step

    box = trimesh.creation.box(extents=[10.0, 20.0, 30.0])
    result = mesh_to_step(box.vertices, box.faces, str(tmp_path / "box.step"))

    assert result.solids == 1
    assert result.watertight
    assert result.valid
    assert result.volume == pytest.approx(6000.0, rel=1e-6)
    assert result.faces_written == 12
    assert result.decimated_from is None
    assert "solid" in result.summary


def test_unshared_vertices_are_welded_into_solid(tmp_path, ocp_available):
    from src.step_ops import mesh_to_step

    box = trimesh.creation.box(extents=[10.0, 20.0, 30.0])
    base = box.vertices[box.faces].reshape(-1, 3)
    faces = np.arange(len(base)).reshape(-1, 3)

    result = mesh_to_step(base, faces, str(tmp_path / "exploded.step"))
    assert result.watertight
    assert result.solids == 1
    assert result.volume == pytest.approx(6000.0, rel=1e-6)


def test_disconnected_mesh_yields_two_solids(tmp_path, ocp_available):
    from src.step_ops import mesh_to_step

    a = trimesh.creation.box(extents=[10.0, 10.0, 10.0])
    b = a.copy()
    b.apply_translation([50.0, 0.0, 0.0])
    both = trimesh.util.concatenate([a, b])

    result = mesh_to_step(both.vertices, both.faces, str(tmp_path / "two.step"))
    assert result.solids == 2
    assert result.watertight
    assert result.volume == pytest.approx(2000.0, rel=1e-6)


# --------------------------------------------------------------------- #
# Open meshes -> shells, never fake solids
# --------------------------------------------------------------------- #
def test_open_mesh_exports_shell_not_solid(tmp_path, ocp_available):
    from src.step_ops import mesh_to_step, step_info

    verts, faces = _open_tube()
    out = tmp_path / "tube.step"
    result = mesh_to_step(verts, faces, str(out))

    assert not result.watertight
    assert result.solids == 0
    assert result.shells >= 1
    assert result.valid  # a genuine shell must stay valid, not a broken solid
    assert "open shell" in result.summary

    info = step_info(str(out))
    assert info.solids == 0
    assert info.faces == result.faces_written


# --------------------------------------------------------------------- #
# Decimation
# --------------------------------------------------------------------- #
def test_decimation_reduces_face_count(tmp_path, ocp_available):
    from src.step_ops import mesh_to_step

    ico = trimesh.creation.icosphere(subdivisions=3)  # 1280 triangles
    result = mesh_to_step(
        ico.vertices, ico.faces, str(tmp_path / "ico.step"), decimate_to=500
    )

    assert result.decimated_from == 1280
    assert result.triangles_in == 1280
    assert result.faces_written < 1280
    assert result.watertight
    assert "decimated" in result.summary


# --------------------------------------------------------------------- #
# step_info
# --------------------------------------------------------------------- #
def test_step_info_reports_volume_and_bbox(tmp_path, ocp_available):
    from src.step_ops import mesh_to_step, step_info

    box = trimesh.creation.box(extents=[10.0, 20.0, 30.0])
    out = tmp_path / "info.step"
    mesh_to_step(box.vertices, box.faces, str(out))

    info = step_info(str(out))
    assert info.solids == 1
    assert info.volume == pytest.approx(6000.0, rel=1e-3)
    assert info.bbox_min == pytest.approx((-5.0, -10.0, -15.0), abs=1e-3)
    assert info.bbox_max == pytest.approx((5.0, 10.0, 15.0), abs=1e-3)


# --------------------------------------------------------------------- #
# Error paths
# --------------------------------------------------------------------- #
def test_rejects_non_step_extension(tmp_path, ocp_available):
    from src.step_ops import mesh_to_step

    box = trimesh.creation.box()
    with pytest.raises(ValueError, match=r"\.step or \.stp"):
        mesh_to_step(box.vertices, box.faces, str(tmp_path / "out.txt"))


def test_rejects_bad_vertex_arrays(ocp_available):
    from src.step_ops import mesh_to_step

    with pytest.raises(ValueError, match=r"\(N,3\)"):
        mesh_to_step(np.zeros((2, 3)), np.array([[0, 1, 2]]), "x.step")

    with pytest.raises(ValueError, match="outside verts"):
        mesh_to_step(np.zeros((3, 3)), np.array([[0, 1, 9]]), "x.step")


def test_rejects_empty_faces(ocp_available):
    from src.step_ops import mesh_to_step

    with pytest.raises(ValueError, match=r"\(M,3\)"):
        mesh_to_step(np.zeros((4, 3)), np.zeros((0, 3), dtype=int), "x.step")
