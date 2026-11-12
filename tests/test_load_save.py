"""Test mesh loading and saving functions."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_load_mesh_valid_stl():
    """Test loading a valid STL file."""
    from src.mesh_ops import load_mesh

    stl_path = FIXTURES_DIR / "test_cube.stl"

    mesh = load_mesh(str(stl_path))

    assert mesh.has_vertices()
    assert len(mesh.vertices) > 0
    assert mesh.has_triangles()


def test_load_mesh_invalid_file():
    """Test loading a nonexistent file raises error."""
    from src.mesh_ops import load_mesh

    with pytest.raises(ValueError, match="Failed to load mesh"):
        load_mesh("/nonexistent/path/file.stl")


def test_load_mesh_valid_obj():
    """Test loading a valid OBJ file."""
    from src.mesh_ops import load_mesh

    obj_path = FIXTURES_DIR / "test_cube.obj"

    mesh = load_mesh(str(obj_path))

    assert mesh.has_vertices()
    assert len(mesh.vertices) > 0
    assert mesh.has_triangles()


def test_load_mesh_broken_still_loads():
    """The loader stays lenient: an invalid cube loads, health checks flag it."""
    from src.mesh_ops import load_mesh, mesh_health

    mesh = load_mesh(str(FIXTURES_DIR / "test_cube_broken.stl"))
    assert len(mesh.triangles) == 6
    assert not mesh_health(mesh).ok


def test_load_mesh_cube_fixture_is_valid():
    from src.mesh_ops import load_mesh, mesh_health

    mesh = load_mesh(str(FIXTURES_DIR / "test_cube.stl"))
    assert mesh_health(mesh).ok


def test_load_mesh_rejects_oversized_file(tmp_path, monkeypatch):
    """Files beyond the on-disk budget are refused before parsing."""
    from src import mesh_ops

    big = tmp_path / "big.stl"
    big.write_bytes(b"\0" * 2048)
    monkeypatch.setattr(mesh_ops, "_MAX_MESH_FILE_BYTES", 1024)
    with pytest.raises(ValueError, match="too large"):
        mesh_ops.load_mesh(str(big))


def test_load_mesh_rejects_too_many_vertices(tmp_path, monkeypatch):
    """Parsed meshes beyond the vertex budget are refused."""
    from src import mesh_ops

    monkeypatch.setattr(mesh_ops, "_MAX_MESH_VERTICES", 2)
    with pytest.raises(ValueError, match="too large"):
        mesh_ops.load_mesh(str(FIXTURES_DIR / "test_cube.stl"))
