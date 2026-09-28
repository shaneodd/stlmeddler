"""Test primitive shape creation."""

import pytest


def test_create_cube():
    """Test cube primitive generation."""
    from src.mesh_ops import create_primitive

    mesh = create_primitive("cube", size=2.0)

    assert mesh.has_vertices()
    assert len(mesh.vertices) == 8
    assert len(mesh.triangles) == 12


def test_create_sphere():
    """Test sphere primitive generation."""
    from src.mesh_ops import create_primitive

    mesh = create_primitive("sphere", radius=1.0)

    assert mesh.has_vertices()
    assert len(mesh.triangles) > 0


def test_create_cylinder():
    """Test cylinder primitive generation."""
    from src.mesh_ops import create_primitive

    mesh = create_primitive("cylinder", radius=0.5, height=2.0)

    assert mesh.has_vertices()
    assert len(mesh.triangles) > 0


def test_create_cone():
    """Test cone primitive generation."""
    from src.mesh_ops import create_primitive

    mesh = create_primitive("cone", radius=1.0, height=2.0)

    assert mesh.has_vertices()
    assert len(mesh.triangles) > 0


def test_create_invalid_shape():
    """Test invalid shape raises error."""
    from src.mesh_ops import create_primitive

    with pytest.raises(ValueError, match="Unsupported primitive shape"):
        create_primitive("invalid_shape")


# --------------------------------------------------------------------- #
# primitive_kwargs (shape -> create_primitive parameters mapping)
# --------------------------------------------------------------------- #
def test_primitive_kwargs_mapping():
    from src.mesh_ops import primitive_kwargs

    assert primitive_kwargs("cube", size=3.0) == {"size": 3.0}
    assert primitive_kwargs("sphere", radius=2.5) == {"radius": 2.5}
    assert primitive_kwargs("cylinder", radius=1.5, height=4.0) == {"radius": 1.5, "height": 4.0}
    assert primitive_kwargs("cone", radius=1.0, height=3.0) == {"radius": 1.0, "height": 3.0}


def test_primitive_kwargs_ignores_irrelevant_dims():
    from src.mesh_ops import primitive_kwargs

    # A cube only uses size; a zero/absent radius must not affect the result.
    assert primitive_kwargs("cube", size=2.0, radius=99.0) == {"size": 2.0}


def test_primitive_kwargs_rejects_bad_input():
    from src.mesh_ops import primitive_kwargs

    with pytest.raises(ValueError):
        primitive_kwargs("tetrahedron")
    with pytest.raises(ValueError):
        primitive_kwargs("cube", size=0)
    with pytest.raises(ValueError):
        primitive_kwargs("cylinder", radius=-1.0, height=2.0)


def test_primitive_kwargs_feeds_create_primitive():
    from src.mesh_ops import create_primitive, primitive_kwargs

    kwargs = primitive_kwargs("cylinder", radius=0.5, height=3.0)
    mesh = create_primitive("cylinder", **kwargs)
    assert mesh.has_vertices()
    assert len(mesh.triangles) > 0
