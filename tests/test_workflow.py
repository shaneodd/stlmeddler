"""Test GUI workflows."""



def test_primitive_creation_workflow():
    """Test creating a primitive through the workflow."""
    from src.mesh_ops import create_primitive

    # Create all primitive types
    cube = create_primitive("cube", size=1.0)
    sphere = create_primitive("sphere", radius=1.0)
    cylinder = create_primitive("cylinder", radius=0.5, height=2.0)
    cone = create_primitive("cone", radius=1.0, height=2.0)

    assert cube.has_vertices()
    assert sphere.has_vertices()
    assert cylinder.has_vertices()
    assert cone.has_vertices()


def test_boolean_workflow():
    """Test boolean operations workflow."""
    from src.mesh_ops import boolean_difference, boolean_union, create_primitive

    # Create meshes
    cube1 = create_primitive("cube", size=2.0)
    cube2 = create_primitive("cube", size=1.0)

    # Test difference
    result_diff = boolean_difference(cube1, cube2)
    assert result_diff.has_vertices()

    # Test union
    result_union = boolean_union(cube1, cube2)
    assert result_union.has_vertices()


def test_file_operations_workflow():
    """Test file save/load workflow."""
    import tempfile

    from src.mesh_ops import create_primitive, load_mesh, save_mesh

    # Create a mesh and save it
    cube = create_primitive("cube", size=1.0)

    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        temp_path = f.name

    try:
        save_mesh(cube, temp_path)

        # Load it back
        loaded = load_mesh(temp_path)
        assert loaded.has_vertices()
    finally:
        import os
        if os.path.exists(temp_path):
            os.unlink(temp_path)
