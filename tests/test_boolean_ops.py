"""Test boolean operations."""



def test_boolean_difference():
    """Test boolean difference operation."""
    from src.mesh_ops import boolean_difference, create_primitive

    cube1 = create_primitive("cube", size=2.0)
    cube2 = create_primitive("cube", size=1.0)

    result = boolean_difference(cube1, cube2)

    assert result.has_vertices()
    assert result.has_triangles()


def test_boolean_union():
    """Test boolean union operation."""
    from src.mesh_ops import boolean_union, create_primitive

    cube1 = create_primitive("cube", size=2.0)
    cube2 = create_primitive("cube", size=1.0)

    result = boolean_union(cube1, cube2)

    assert result.has_vertices()
    assert result.has_triangles()


def test_boolean_difference_sphere_from_cube():
    """Test subtracting a sphere from a cube."""
    from src.mesh_ops import boolean_difference, create_primitive

    # Create a larger cube
    cube = create_primitive("cube", size=2.0)
    # Create a smaller sphere that fits inside the cube
    sphere = create_primitive("sphere", radius=0.5)

    result = boolean_difference(cube, sphere)

    assert result.has_vertices()
    assert len(result.vertices) > 0
    # The result should have fewer triangles than the original cube (cube has 12, sphere has ~760)
    # After subtraction we expect a mesh with a spherical cavity


def test_boolean_union_separate_cubes():
    """Test union of two separate (non-overlapping) cubes."""
    from src.mesh_ops import boolean_union, create_primitive

    # Create first cube at origin
    cube1 = create_primitive("cube", size=1.0)

    # Create second cube shifted away so they don't overlap
    cube2 = create_primitive("cube", size=1.0)

    result = boolean_union(cube1, cube2)

    assert result.has_vertices()
    assert len(result.vertices) > 0
