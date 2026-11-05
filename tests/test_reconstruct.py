"""Headless tests for Tier 2 scan reverse-engineering (src/reconstruct.py).

Feature detection (planes, cylindrical holes), nominal-diameter snapping and
snap-list JSON round-trips, exercised on synthetic parts with and without
scan noise. No Qt involved.
"""

from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _part_with_holes():
    """40x30x10 block: Ø6 through hole at (10,5) and Ø8 blind hole (depth 6).

    The blind hole opens on the bottom face (z=-5) and stops at z=1.
    """
    import trimesh

    part = trimesh.creation.box(extents=[40.0, 30.0, 10.0])
    through = trimesh.creation.cylinder(radius=3.0, height=40.0, sections=64)
    through.apply_translation([10.0, 5.0, 0.0])
    blind = trimesh.creation.cylinder(radius=4.0, height=6.0, sections=64)
    blind.apply_translation([-10.0, -5.0, -2.0])
    return part.difference(through).difference(blind)


def _noise(part, sigma, seed):
    rng = np.random.default_rng(seed)
    return part.vertices + rng.normal(0.0, sigma, part.vertices.shape)


def _hole_at(holes, xy, atol=1.5):
    for h in holes:
        if np.linalg.norm(np.array([h.center[0], h.center[1]]) - np.array(xy)) < atol:
            return h
    raise AssertionError(f"no hole detected near {xy}; got {[h.center for h in holes]}")


# --------------------------------------------------------------------- #
# Clean synthetic part
# --------------------------------------------------------------------- #
def test_clean_block_detects_planes_and_two_holes():
    from src.reconstruct import detect_features

    part = _part_with_holes()
    feats = detect_features(part.vertices, part.faces)

    assert feats.triangles == len(part.faces)
    assert len(feats.planes) >= 3  # top, bottom and the blind-hole floor

    thru = _hole_at(feats.holes, (10.0, 5.0))
    assert thru.concave
    assert thru.is_through
    assert thru.raw_diameter == pytest.approx(6.0, abs=0.05)
    assert thru.diameter == pytest.approx(6.0)  # snapped exactly
    assert thru.depth == pytest.approx(10.0, abs=0.2)

    blind = _hole_at(feats.holes, (-10.0, -5.0))
    assert blind.concave
    assert not blind.is_through
    assert blind.raw_diameter == pytest.approx(8.0, abs=0.05)
    assert blind.diameter == pytest.approx(8.0)
    assert blind.depth == pytest.approx(6.0, abs=0.3)

    assert "2" in feats.summary()


# --------------------------------------------------------------------- #
# Scan noise
# --------------------------------------------------------------------- #
def test_light_noise_still_snaps_nominal_diameters():
    from src.reconstruct import DetectOptions, detect_features

    part = _part_with_holes()
    verts = _noise(part, 0.05, seed=42)
    opts = DetectOptions(plane_distance=0.2, cylinder_distance=0.2)
    feats = detect_features(verts, part.faces, opts=opts)

    thru = _hole_at(feats.holes, (10.0, 5.0))
    blind = _hole_at(feats.holes, (-10.0, -5.0))
    assert thru.diameter == pytest.approx(6.0)
    assert blind.diameter == pytest.approx(8.0)
    assert thru.is_through and not blind.is_through


def test_heavier_noise_finds_holes_within_half_mm():
    from src.reconstruct import DetectOptions, detect_features

    part = _part_with_holes()
    verts = _noise(part, 0.15, seed=7)
    opts = DetectOptions(plane_distance=0.45, cylinder_distance=0.45)
    feats = detect_features(verts, part.faces, opts=opts)

    thru = _hole_at(feats.holes, (10.0, 5.0))
    blind = _hole_at(feats.holes, (-10.0, -5.0))
    assert thru.raw_diameter == pytest.approx(6.0, abs=0.5)
    assert blind.raw_diameter == pytest.approx(8.0, abs=0.5)


# --------------------------------------------------------------------- #
# Coaxial counterbore: one component, two stacked cylinders (peeling)
# --------------------------------------------------------------------- #
def test_counterbore_peels_two_diameters():
    import trimesh

    from src.reconstruct import detect_features

    part = trimesh.creation.box(extents=[40.0, 30.0, 10.0])
    through = trimesh.creation.cylinder(radius=3.0, height=40.0, sections=64)
    counter = trimesh.creation.cylinder(radius=5.0, height=6.0, sections=64)
    counter.apply_translation([0.0, 0.0, 2.0])  # top 3 mm at r=5
    part = part.difference(through).difference(counter)

    feats = detect_features(part.vertices, part.faces)
    diameters = sorted(h.diameter or h.raw_diameter for h in feats.holes)
    assert any(abs(d - 6.0) < 0.1 for d in diameters)
    assert any(abs(d - 10.0) < 0.1 for d in diameters)


# --------------------------------------------------------------------- #
# Concavity: real scan part (washer) and a solid round bar
# --------------------------------------------------------------------- #
def test_washer_bore_concave_rim_convex():
    from src.reconstruct import detect_features

    mesh = trimesh.load(str(FIXTURES_DIR / "washer.stl"))
    feats = detect_features(mesh.vertices, mesh.faces)

    assert len(feats.holes) >= 2
    bore = min(feats.holes, key=lambda h: h.raw_diameter)
    rim = max(feats.holes, key=lambda h: h.raw_diameter)
    assert bore.concave and bore.is_through
    assert bore.raw_diameter == pytest.approx(17.5, abs=0.2)
    assert not rim.concave
    assert rim.raw_diameter == pytest.approx(25.5, abs=0.2)
    assert "did not snap" in feats.summary()  # 17.5/25.5 are not drill sizes


def test_solid_bar_wall_is_convex():
    from src.reconstruct import detect_features

    bar = trimesh.creation.cylinder(radius=10.0, height=30.0, sections=64)
    feats = detect_features(bar.vertices, bar.faces)

    assert len(feats.holes) == 1
    assert not feats.holes[0].concave


# --------------------------------------------------------------------- #
# Negative cases
# --------------------------------------------------------------------- #
def test_sphere_has_no_features():
    from src.reconstruct import detect_features

    ico = trimesh.creation.icosphere(subdivisions=2)
    feats = detect_features(ico.vertices, ico.faces)
    assert feats.planes == []
    assert feats.holes == []


def test_plain_box_has_planes_no_holes():
    from src.reconstruct import detect_features

    box = trimesh.creation.box(extents=[20.0, 15.0, 4.0])
    feats = detect_features(box.vertices, box.faces)
    assert len(feats.planes) == 6
    assert feats.holes == []


# --------------------------------------------------------------------- #
# Snap list
# --------------------------------------------------------------------- #
def test_snap_diameter_within_tolerance():
    from src.reconstruct import SnapList

    snap = SnapList.default()
    snapped, delta = snap.snap_diameter(4.97)
    assert snapped == pytest.approx(5.0)
    assert delta == pytest.approx(0.03)

    snapped, delta = snap.snap_diameter(5.30)
    assert snapped is None
    assert delta == pytest.approx(0.2, abs=0.06)


def test_snap_list_json_roundtrip(tmp_path):
    from src.reconstruct import SnapList

    snap = SnapList(hole_diameters=(3.2, 6.6, 12.7), tolerance=0.25)
    path = tmp_path / "snap.json"
    snap.save(path)

    loaded = SnapList.load(path)
    assert loaded.hole_diameters == (3.2, 6.6, 12.7)
    assert loaded.tolerance == pytest.approx(0.25)

    path.write_text('{"hole_diameters_mm": []}')
    with pytest.raises(ValueError, match="No usable"):
        SnapList.load(path)


# --------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------- #
def test_detect_features_rejects_bad_arrays():
    from src.reconstruct import detect_features

    with pytest.raises(ValueError, match=r"\(N,3\)"):
        detect_features(np.zeros((2, 3)), np.array([[0, 1, 2]]))

    with pytest.raises(ValueError, match="outside verts"):
        detect_features(np.zeros((3, 3)), np.array([[0, 1, 9]]))


# --------------------------------------------------------------------- #
# FeatureSet serialisation
# --------------------------------------------------------------------- #
def test_feature_set_to_dict_round_trips_values():
    from src.reconstruct import detect_features

    part = _part_with_holes()
    feats = detect_features(part.vertices, part.faces)
    data = feats.to_dict()

    assert data["triangles"] == len(part.faces)
    assert len(data["holes"]) == 2
    bbox_size = [mx - mn for mn, mx in zip(data["bbox_min"], data["bbox_max"], strict=True)]
    assert bbox_size[0] == pytest.approx(40.0, abs=0.1)
    by_d = {h["diameter"]: h for h in data["holes"]}
    assert by_d[6.0]["is_through"] is True
    assert by_d[8.0]["is_through"] is False
