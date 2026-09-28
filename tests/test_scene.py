"""Headless tests for the multi-object SceneModel backing the viewer.

These exercise object management, per-object transform isolation, cut
integration and export consistency without instantiating any Qt/OpenGL widget.
"""

import os
import tempfile

import numpy as np
import pytest


def _cube(size=1.0):
    from src.mesh_ops import create_primitive
    return create_primitive("cube", size=size)


# --------------------------------------------------------------------- #
# Object management
# --------------------------------------------------------------------- #
def test_add_get_set_active_roundtrip():
    from src.gui.scene import SceneModel

    s = SceneModel()
    i0 = s.add_object(_cube(), name="A")
    i1 = s.add_object(_cube(), name="B")

    assert (i0, i1) == (0, 1)
    assert len(s.get_objects()) == 2
    assert s.active_index == i1  # last added becomes active
    assert s.get_index("A") == 0 and s.get_index("B") == 1

    s.set_active(0)
    assert s.active_index == 0


def test_remove_object_updates_selection():
    from src.gui.scene import SceneModel

    s = SceneModel()
    s.add_object(_cube(), name="A")
    s.add_object(_cube(), name="B")
    s.add_object(_cube(), name="C")
    s.set_active(2)

    s.remove_object(2)
    assert len(s.get_objects()) == 2
    assert s.active_index == 1
    assert s.get_index("C") == -1

    # remove_by_name path
    assert s.remove_by_name("A") is True
    assert s.get_index("A") == -1
    assert s.remove_by_name("missing") is False


def test_add_object_rejects_non_mesh():
    from src.gui.scene import SceneModel

    s = SceneModel()
    with pytest.raises(TypeError):
        s.add_object(object())


# --------------------------------------------------------------------- #
# Transform isolation
# --------------------------------------------------------------------- #
def test_translate_only_affects_active_object():
    from src.gui.scene import SceneModel

    s = SceneModel()
    s.add_object(_cube(), name="A")
    s.add_object(_cube(), name="B")  # active

    before_a = s.transformed_vertices(0).copy()
    s.translate([5.0, 0.0, 0.0])  # applies to active (B)

    after_a = s.transformed_vertices(0)
    assert np.allclose(before_a, after_a), "object A must not move"

    moved_x = s.transformed_vertices(1)[:, 0] - before_a[:, 0]
    assert np.allclose(moved_x, 5.0)


def test_rotate_only_affects_active_object():
    from src.gui.scene import SceneModel

    s = SceneModel()
    s.add_object(_cube(), name="A")
    s.add_object(_cube(size=1.5), name="B")  # active

    before_a = s.transformed_vertices(0).copy()
    s.rotate([0, 0, 1], np.pi / 2)  # rotate B by 90 deg about Z

    assert np.allclose(before_a, s.transformed_vertices(0)), "A must be untouched"
    # A rotated cube's world vertices differ from its unrotated ones.
    assert not np.allclose(s.transformed_vertices(1), before_a)


def test_align_active_to_plate_sets_min_z_zero():
    import open3d as o3d

    from src.gui.scene import SceneModel

    s = SceneModel()
    obj = _cube(size=2.0)
    # Lift the mesh so it floats above the plate.
    v = np.asarray(obj.vertices).copy()
    v[:, 2] += 10.0
    obj.vertices = o3d.utility.Vector3dVector(v)

    s.add_object(obj, auto_align=False)
    assert s.transformed_vertices(0)[:, 2].min() == pytest.approx(10.0)

    s.align_active_to_plate()
    assert s.transformed_vertices(0)[:, 2].min() == pytest.approx(0.0)


# --------------------------------------------------------------------- #
# Cut integration
# --------------------------------------------------------------------- #
def test_cut_produces_two_distinct_named_objects():
    from src.gui.scene import SceneModel

    s = SceneModel()
    s.add_object(_cube(size=2.0), name="block")  # spans z[0,2] after align
    added = s.cut_active_at_z(1.0)

    assert len(added) == 2
    names = {o["name"] for o in s.get_objects()}
    assert "block_bottom" in names and "block_top" in names
    assert "block" not in names  # original replaced


def test_cut_outside_bounds_yields_single_part():
    from src.gui.scene import SceneModel

    s = SceneModel()
    s.add_object(_cube(size=2.0), name="block")
    added = s.cut_active_at_z(100.0)  # plane above the whole mesh

    assert len(added) == 1
    assert s.get_index("block_bottom") >= 0


# --------------------------------------------------------------------- #
# Export consistency
# --------------------------------------------------------------------- #
def test_export_reflects_transformed_arrangement():
    from src.gui.scene import SceneModel
    from src.mesh_ops import load_mesh, save_mesh

    s = SceneModel()
    s.add_object(_cube(size=1.0), name="A")
    world_before = s.transformed_vertices(0)

    path = tempfile.mktemp(suffix=".stl")
    try:
        save_mesh(s.merged_mesh(), path)
        reloaded = np.asarray(load_mesh(path).vertices)
        # Reloaded bounds match the on-screen (transformed) geometry.
        assert reloaded.min(axis=0) == pytest.approx(world_before.min(axis=0), abs=1e-3)
        assert reloaded.max(axis=0) == pytest.approx(world_before.max(axis=0), abs=1e-3)

        # Move the object and confirm the exported geometry follows it.
        s.translate([5.0, 0.0, 0.0])
        save_mesh(s.merged_mesh(), path)
        reloaded2 = np.asarray(load_mesh(path).vertices)
        assert reloaded2.max(axis=0)[0] == pytest.approx(
            world_before.max(axis=0)[0] + 5.0, abs=1e-3
        )
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_merged_mesh_combines_all_objects():
    from src.gui.scene import SceneModel

    s = SceneModel()
    s.add_object(_cube(), name="A")   # 8 verts
    s.add_object(_cube(), name="B")   # 8 verts
    merged = s.merged_mesh()
    assert len(merged.vertices) == 16
    assert len(merged.triangles) == 24


# --------------------------------------------------------------------- #
# Project persistence (.stlproj)
# --------------------------------------------------------------------- #
def test_project_roundtrip_preserves_state(tmp_path):
    from src.gui.scene import SceneModel

    m = SceneModel()
    m.add_object(_cube(), name="A")
    m.add_object(_cube(size=1.5), name="B")  # active
    m.rotate([0, 0, 1], np.pi / 4)
    m.translate([3.0, -2.0, 5.0])
    m.set_active(0)

    snap = [
        (o["name"], tuple(o["color"]), o["rot_x"], o["trans_z"], np.asarray(o["verts"]).copy())
        for o in m.objects
    ]
    path = str(tmp_path / "proj.stlproj")
    m.save_project(path)
    loaded = SceneModel.load_project(path)

    assert len(loaded.objects) == 2
    assert loaded.active_index == 0
    for (name, color, rx, tz, verts), obj in zip(snap, loaded.objects, strict=True):
        assert obj["name"] == name
        assert np.allclose(color, obj["color"])
        assert obj["rot_x"] == pytest.approx(rx)
        assert obj["trans_z"] == pytest.approx(tz)
        assert np.allclose(verts, obj["verts"])


def test_project_roundtrip_with_cut_parts(tmp_path):
    from src.gui.scene import SceneModel

    m = SceneModel()
    m.add_object(_cube(size=2.0), name="block")  # spans z[0,2] after align
    added = m.cut_active_at_z(1.0)
    assert len(added) == 2

    before_bounds = np.asarray(m.merged_mesh().vertices).max(axis=0)
    path = str(tmp_path / "cut.stlproj")
    m.save_project(path)
    loaded = SceneModel.load_project(path)

    names = {o["name"] for o in loaded.objects}
    assert {"block_bottom", "block_top"} == names
    after_bounds = np.asarray(loaded.merged_mesh().vertices).max(axis=0)
    assert np.allclose(before_bounds, after_bounds, atol=1e-6)


def test_empty_project_roundtrip(tmp_path):
    from src.gui.scene import SceneModel

    m = SceneModel()
    path = str(tmp_path / "empty.stlproj")
    m.save_project(path)
    loaded = SceneModel.load_project(path)
    assert len(loaded.objects) == 0
    assert loaded.active_index == -1


def test_matrix_override_baked_on_save(tmp_path):
    from src.gui.scene import SceneModel

    m = SceneModel()
    m.add_object(_cube(), name="C")
    M = np.eye(4)
    M[0, 3] = 10.0  # translate +X via override
    m.apply_transform(M)

    world_before = m.transformed_vertices(0).copy()
    path = str(tmp_path / "override.stlproj")
    m.save_project(path)
    loaded = SceneModel.load_project(path)

    # Override baked into base geometry; transform emitted as identity.
    assert np.allclose(loaded.objects[0]["verts"], world_before)
    assert loaded.objects[0]["trans_x"] == pytest.approx(0.0)


# --------------------------------------------------------------------- #
# Object-to-object booleans
# --------------------------------------------------------------------- #
def _two_overlapping_boxes():
    from src.gui.scene import SceneModel

    m = SceneModel()
    # Native-cornered boxes overlap; auto_align=False keeps their coordinates.
    m.add_object(_cube(size=2.0), name="A", auto_align=False)
    m.add_object(_cube(size=1.0), name="B", auto_align=False)
    m.set_active(0)  # A is operand A, B (index 1) is the other operand
    return m


def test_boolean_union_consumes_both_operands():
    m = _two_overlapping_boxes()
    result = m.boolean_active_with(1, "union")

    assert result == "A"
    assert len(m.objects) == 1
    assert m.objects[0]["name"] == "A"
    assert len(m.objects[0]["verts"]) > 0


def test_boolean_difference_consumes_both_operands():
    m = _two_overlapping_boxes()
    result = m.boolean_active_with(1, "difference")

    assert result == "A"
    assert len(m.objects) == 1
    assert len(m.objects[0]["verts"]) > 0


def test_boolean_removal_is_index_order_independent():
    # Active is the LAST object (B); operand A has a lower index. Removal by name
    # must still consume both and name the result after the active operand (B).
    from src.gui.scene import SceneModel

    m = SceneModel()
    m.add_object(_cube(size=2.0), name="A", auto_align=False)
    m.add_object(_cube(size=1.0), name="B", auto_align=False)
    m.set_active(1)  # B active, A (index 0) is the other operand

    result = m.boolean_active_with(0, "union")
    assert result == "B"
    assert len(m.objects) == 1


def test_boolean_invalid_operand_index_is_noop():
    m = _two_overlapping_boxes()
    before = [o["name"] for o in m.objects]

    # self index (active is 0) and out-of-range both return None, scene unchanged.
    assert m.boolean_active_with(0, "union") is None
    assert m.boolean_active_with(99, "difference") is None
    assert [o["name"] for o in m.objects] == before


def test_cut_pieces_are_watertight_volumes():
    """Regression: cut halves must be proper volumes (consistent winding) so they can
    feed booleans/slicers. The old per-face flip heuristic broke winding."""
    import trimesh

    from src.gui.scene import SceneModel

    m = SceneModel()
    m.add_object(_cube(size=2.0), name="A")
    added = m.cut_active_at_z(1.0)
    assert len(added) == 2
    for i, o in enumerate(m.objects):
        tri = trimesh.Trimesh(vertices=m.transformed_vertices(i), faces=o["tris"], process=False)
        assert tri.is_watertight and tri.is_volume, (o["name"], tri.is_winding_consistent)


def test_boolean_after_cut_and_move_succeeds():
    """Regression: cutting then moving a part and booleaning it with another object
    used to raise 'Not all meshes are volumes!'."""
    from src.gui.scene import SceneModel

    for kind in ("union", "difference"):
        m = SceneModel()
        m.add_object(_cube(size=2.0), name="A")
        added = m.cut_active_at_z(1.0)
        m.add_object(_cube(size=3.0), name="B", auto_align=False)

        part = added[0]
        m.set_active(m.get_index(part))
        m.translate([4.0, -2.0, 1.5])
        result = m.boolean_active_with(m.get_index("B"), kind)

        names = [o["name"] for o in m.objects]
        assert result == part, (kind, result)
        assert len(names) == 2 and "B" not in names, (kind, names)


# --------------------------------------------------------------------- #
# Click-to-select ray casting (SceneModel.hit_test)
# --------------------------------------------------------------------- #
def test_hit_test_selects_nearest_object():
    from src.gui.scene import SceneModel

    m = SceneModel()
    m.add_object(_cube(size=2.0), name="A", auto_align=False)  # near: z 0..2
    m.add_object(_cube(size=2.0), name="B", auto_align=False)
    m.set_active(1)
    m.translate([0, 0, -5])  # B far: z -5..-3

    origin = np.array([1.0, 1.0, 5.0])
    direction = np.array([0.0, 0.0, -1.0])
    hit = m.hit_test(origin, direction)  # A is nearer along the ray
    assert hit is not None and hit[0] == 0

    m.remove_by_name("A")
    hit = m.hit_test(origin, direction)  # now B (index 0)
    assert hit is not None and hit[0] == 0


def test_hit_test_miss_and_empty():
    from src.gui.scene import SceneModel

    m = SceneModel()
    m.add_object(_cube(size=2.0), name="A", auto_align=False)
    assert m.hit_test(np.array([50.0, 50.0, 5.0]), np.array([0.0, 0.0, -1.0])) is None
    assert SceneModel().hit_test(np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, -1.0])) is None


def test_hit_test_degenerate_direction_is_none():
    from src.gui.scene import SceneModel

    m = SceneModel()
    m.add_object(_cube(size=2.0), name="A", auto_align=False)
    assert m.hit_test(np.array([1.0, 1.0, 5.0]), np.array([0.0, 0.0, 0.0])) is None


# --------------------------------------------------------------------- #
# Face-to-plate alignment (SceneModel.align_face_to_plate)
# --------------------------------------------------------------------- #
def _box():
    """2(x) x 2(y) x 3(z) axis-aligned box, z 0..3, outward winding.

    Triangle rows: 0-1 bottom (-Z), 2-3 top (+Z), 4-5 y=0 (-Y),
    6-7 y=2 (+Y), 8-9 x=0 (-X), 10-11 x=2 (+X).
    """
    from types import SimpleNamespace

    verts = np.array(
        [
            [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],
            [0, 0, 3], [2, 0, 3], [2, 2, 3], [0, 2, 3],
        ], dtype=float
    )
    tris = np.array(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6],
            [0, 4, 7], [0, 7, 3],
            [1, 2, 6], [1, 6, 5],
        ]
    )
    return SimpleNamespace(vertices=verts, triangles=tris)


def _face_world_normal(m, obj_index, tri_index):
    from src.gui.scene import model_matrix

    obj = m.objects[obj_index]
    M = obj.get("matrix_override", model_matrix(obj))
    tri = obj["tris"][tri_index]
    v = obj["verts"]
    n = np.cross(v[tri[1]] - v[tri[0]], v[tri[2]] - v[tri[0]])
    n = (M[:3, :3] @ n)
    return n / np.linalg.norm(n)


def test_align_face_to_plate_top_face_flips_box_flat():
    from src.gui.scene import SceneModel

    m = SceneModel()
    m.add_object(_box(), name="box", auto_align=False)
    assert m.align_face_to_plate(2)  # top face (normal +Z) becomes the bottom

    world = m.transformed_vertices(0)
    assert np.allclose(_face_world_normal(m, 0, 2), [0.0, 0.0, -1.0], atol=1e-9)
    tri_verts = world[m.objects[0]["tris"][2]]
    assert np.allclose(tri_verts[:, 2], 0.0, atol=1e-9)  # clicked face on the plate
    assert world[:, 2].min() >= -1e-9  # nothing below the plate
    extents = sorted(world.max(axis=0) - world.min(axis=0))
    assert np.allclose(extents, [2.0, 2.0, 3.0])  # box still axis-aligned


def test_align_face_to_plate_side_face_lays_box_down():
    from src.gui.scene import SceneModel

    m = SceneModel()
    m.add_object(_box(), name="box", auto_align=False)
    assert m.align_face_to_plate(10)  # x=2 side face

    world = m.transformed_vertices(0)
    assert np.allclose(_face_world_normal(m, 0, 10), [0.0, 0.0, -1.0], atol=1e-9)
    tri_verts = world[m.objects[0]["tris"][10]]
    assert np.allclose(tri_verts[:, 2], 0.0, atol=1e-9)
    assert world[:, 2].min() >= -1e-9
    extents = sorted(world.max(axis=0) - world.min(axis=0))
    assert np.allclose(extents, [2.0, 2.0, 3.0])
    # longest face edge snapped to an axis -> rotation columns are +-unit axes
    from src.gui.scene import model_matrix

    R = model_matrix(m.objects[0])[:3, :3]
    assert np.allclose(np.sort(np.abs(R), axis=0)[-1], 1.0, atol=1e-9)
    assert np.allclose(np.sort(np.abs(R), axis=0)[:-1], 0.0, atol=1e-9)


def test_align_face_to_plate_snaps_yaw_of_rotated_object():
    from src.gui.scene import SceneModel

    m = SceneModel()
    m.add_object(_box(), name="box", auto_align=False)
    a = np.pi / 4.0
    yaw = np.eye(4)
    yaw[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    m.apply_transform(yaw)

    assert m.align_face_to_plate(10)
    world = m.transformed_vertices(0)
    extents = sorted(world.max(axis=0) - world.min(axis=0))
    assert np.allclose(extents, [2.0, 2.0, 3.0])  # snapped back to axis-aligned


def test_align_face_to_plate_view_dir_fixes_inside_out_winding():
    from types import SimpleNamespace

    from src.gui.scene import SceneModel

    box = _box()
    inside_out = SimpleNamespace(
        vertices=box.vertices,
        triangles=box.triangles[:, ::-1],  # flip winding: normals point inward
    )
    m = SceneModel()
    m.add_object(inside_out, name="box", auto_align=False)

    # Top face clicked from above; without the hint its (inward) normal already
    # matches the target, so the box stays put with the face on top.
    assert m.align_face_to_plate(2)
    world = m.transformed_vertices(0)
    assert np.allclose(world[m.objects[0]["tris"][2]][:, 2], 3.0, atol=1e-9)

    # With the view direction (looking down), the normal is treated as inside-out.
    assert m.align_face_to_plate(2, view_dir=(0.0, 0.0, -1.0))
    world = m.transformed_vertices(0)
    assert np.allclose(world[m.objects[0]["tris"][2]][:, 2], 0.0, atol=1e-9)
    assert world[:, 2].min() >= -1e-9


def test_align_face_to_plate_invalid_inputs_return_false():
    from src.gui.scene import SceneModel

    m = SceneModel()
    m.add_object(_box(), name="box", auto_align=False)
    assert not m.align_face_to_plate(999)
    assert not m.align_face_to_plate(0, index=5)
    assert SceneModel().align_face_to_plate(0) is False


def test_align_face_to_plate_keeps_euler_transform_in_sync():
    from src.gui.scene import SceneModel

    m = SceneModel()
    m.add_object(_box(), name="box", auto_align=False)
    m.apply_transform(np.eye(4))  # force a matrix_override first
    assert "matrix_override" in m.objects[0]
    assert m.align_face_to_plate(10)
    assert "matrix_override" not in m.objects[0]  # decomposed back to euler + translation
    # Euler + translation still reproduce the aligned pose: clicked face on the plate.
    world = m.transformed_vertices(0)
    tri_verts = world[m.objects[0]["tris"][10]]
    assert np.allclose(tri_verts[:, 2], 0.0, atol=1e-9)


# --------------------------------------------------------------------- #
# Center-on-plate (SceneModel.center_active_on_plate)
# --------------------------------------------------------------------- #
def _centered_bounds(m, idx=0):
    world = m.transformed_vertices(idx)
    bmin, bmax = world.min(axis=0), world.max(axis=0)
    return (bmin + bmax) / 2.0, float(bmin[2])


def test_center_on_plate_moves_offcenter_object_to_origin():
    from src.gui.scene import SceneModel

    m = SceneModel()
    m.add_object(_box(), name="box", auto_align=False)
    m.translate([10.0, 20.0, 5.0])
    assert m.center_active_on_plate()

    center, min_z = _centered_bounds(m)
    assert np.allclose(center[:2], [0.0, 0.0], atol=1e-9)
    assert abs(min_z) < 1e-9  # resting on the plate
    # Box spans 0..2 in its local frame, so its centre sits at local (1, 1).
    t = m.get_transform()
    assert abs(t["trans_x"] - (-1.0)) < 1e-9 and abs(t["trans_y"] - (-1.0)) < 1e-9


def test_center_on_plate_keeps_rotation():
    from src.gui.scene import SceneModel

    m = SceneModel()
    m.add_object(_box(), name="box", auto_align=False)
    m.rotate([0.0, 0.0, 1.0], np.pi / 4.0)
    rot_before = (m.objects[0]["rot_x"], m.objects[0]["rot_y"], m.objects[0]["rot_z"])
    m.translate([7.0, -3.0, 0.0])

    assert m.center_active_on_plate()
    assert (m.objects[0]["rot_x"], m.objects[0]["rot_y"], m.objects[0]["rot_z"]) == rot_before
    center, min_z = _centered_bounds(m)
    assert np.allclose(center[:2], [0.0, 0.0], atol=1e-9)
    assert abs(min_z) < 1e-9


def test_center_on_plate_decomposes_matrix_override():
    from src.gui.scene import SceneModel

    m = SceneModel()
    m.add_object(_box(), name="box", auto_align=False)
    a = np.pi / 4.0
    yaw = np.eye(4)
    yaw[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    yaw[0, 3] = 50.0
    m.apply_transform(yaw)  # forces matrix_override
    assert "matrix_override" in m.objects[0]

    assert m.center_active_on_plate()
    assert "matrix_override" not in m.objects[0]
    center, min_z = _centered_bounds(m)
    assert np.allclose(center[:2], [0.0, 0.0], atol=1e-9)
    assert abs(min_z) < 1e-9


def test_center_on_plate_only_moves_active_object():
    from src.gui.scene import SceneModel

    m = SceneModel()
    m.add_object(_box(), name="A", auto_align=False)
    m.add_object(_box(), name="B", auto_align=False)
    m.translate([9.0, 9.0, 0.0])  # applies to active (B)
    before_a = m.transformed_vertices(0).copy()

    assert m.center_active_on_plate()
    assert np.allclose(m.transformed_vertices(0), before_a)  # A untouched
    center, _ = _centered_bounds(m, 1)
    assert np.allclose(center[:2], [0.0, 0.0], atol=1e-9)


def test_center_on_plate_invalid_returns_false():
    from src.gui.scene import SceneModel

    assert SceneModel().center_active_on_plate() is False
    m = SceneModel()
    m.add_object(_box(), name="box", auto_align=False)
    assert m.center_active_on_plate(index=7) is False


# --------------------------------------------------------------------- #
# pick_surface (exact hit point for measuring)
# --------------------------------------------------------------------- #
def test_pick_surface_returns_exact_world_point():
    from src.gui.scene import SceneModel

    m = SceneModel()
    m.add_object(_box(), name="box", auto_align=False)  # spans z 0..3
    hit = m.pick_surface(np.array([1.0, 1.0, 10.0]), np.array([0.0, 0.0, -1.0]))
    assert hit is not None
    obj_index, tri_row, point = hit
    assert obj_index == 0 and 0 <= tri_row < len(m.objects[0]["tris"])
    assert np.allclose(point, [1.0, 1.0, 3.0], atol=1e-9)  # lands on the top face


def test_pick_surface_miss_returns_none():
    from src.gui.scene import SceneModel

    m = SceneModel()
    m.add_object(_box(), name="box", auto_align=False)
    assert m.pick_surface(np.array([99.0, 99.0, 5.0]), np.array([0.0, 0.0, -1.0])) is None
    assert m.pick_surface(np.array([1.0, 1.0, 5.0]), np.array([0.0, 0.0, 0.0])) is None
