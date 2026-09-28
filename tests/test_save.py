"""Test mesh saving function."""

from pathlib import Path


def test_save_mesh_stl():
    """Test saving a mesh to STL format."""
    from src.mesh_ops import create_primitive, save_mesh

    cube = create_primitive("cube", size=1.0)

    output_path = Path("/tmp/test_cube_output.stl")

    save_mesh(cube, str(output_path))

    assert output_path.exists()

    output_path.unlink()


def test_save_mesh_obj():
    """Test saving a mesh to OBJ format."""
    from src.mesh_ops import create_primitive, save_mesh

    sphere = create_primitive("sphere", radius=1.0)

    output_path = Path("/tmp/test_sphere_output.obj")

    save_mesh(sphere, str(output_path))

    assert output_path.exists()

    output_path.unlink()


def test_save_mesh_3mf():
    """Test saving a mesh to 3MF format."""
    from src.mesh_ops import create_primitive, save_mesh

    cylinder = create_primitive("cylinder", radius=0.5, height=2.0)

    output_path = Path("/tmp/test_cylinder_output.3mf")

    save_mesh(cylinder, str(output_path))

    assert output_path.exists()

    output_path.unlink()


def test_save_mesh_missing_normals():
    """Test that saving auto-computes missing normals."""
    from src.mesh_ops import create_primitive, save_mesh

    cube = create_primitive("cube", size=1.0)
    cube.vertex_normals.clear()

    output_path = Path("/tmp/test_cube_no_normals.stl")

    save_mesh(cube, str(output_path))

    assert output_path.exists()

    output_path.unlink()
