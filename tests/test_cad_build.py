"""Headless tests for feature-driven CAD rebuilding (src/cad_build.py).

Detection runs on synthetic scans, the builder reconstructs a kernel-native
solid and the exported STEP is re-imported and checked against analytic
volumes. Skips cleanly when cadquery-ocp is absent.
"""

import math

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


def _part_with_holes():
    part = trimesh.creation.box(extents=[40.0, 30.0, 10.0])
    h1 = trimesh.creation.cylinder(radius=3.0, height=40.0, sections=64)
    h1.apply_translation([10.0, 5.0, 0.0])
    h2 = trimesh.creation.cylinder(radius=4.0, height=6.0, sections=64)
    h2.apply_translation([-10.0, -5.0, -2.0])  # blind: z -5..1
    return part.difference(h1).difference(h2)


# --------------------------------------------------------------------- #
# Rebuild accuracy
# --------------------------------------------------------------------- #
def test_holed_block_rebuilt_to_nominal_volume(tmp_path, ocp_available):
    from src.cad_build import features_to_step
    from src.reconstruct import SnapList, detect_features

    part = _part_with_holes()
    features = detect_features(part.vertices, part.faces, snap=SnapList.default())
    result = features_to_step(features, str(tmp_path / "rebuilt.step"), verts=part.vertices)

    assert result.base in ("hull+planes", "planes")
    assert result.holes_cut == 2
    assert result.solids == 1
    assert result.valid and result.watertight
    expected = 40 * 30 * 10 - math.pi * 3.0**2 * 10 - math.pi * 4.0**2 * 6
    assert result.volume == pytest.approx(expected, rel=2e-3)


def test_plain_box_rebuilt_exactly(tmp_path, ocp_available):
    from src.cad_build import features_to_step
    from src.reconstruct import detect_features

    box = trimesh.creation.box(extents=[20.0, 15.0, 4.0])
    features = detect_features(box.vertices, box.faces)
    result = features_to_step(features, str(tmp_path / "box.step"), verts=box.vertices)

    assert result.holes_cut == 0
    assert result.walls_rebuilt == 0
    assert result.watertight
    assert result.volume == pytest.approx(1200.0, rel=1e-4)


def test_washer_gets_round_wall_and_bore(tmp_path, ocp_available):
    from pathlib import Path

    from src.cad_build import features_to_step
    from src.reconstruct import SnapList, detect_features

    washer = trimesh.load(str(Path(__file__).parent / "fixtures" / "washer.stl"))
    features = detect_features(washer.vertices, washer.faces, snap=SnapList.default())
    result = features_to_step(features, str(tmp_path / "washer.step"), verts=washer.vertices)

    assert result.walls_rebuilt == 1  # outer rim rebuilt as a nominal cylinder
    assert result.holes_cut == 1  # bore
    assert result.watertight
    nominal = math.pi * (12.75**2 - 8.725**2) * 3.4
    assert result.volume == pytest.approx(nominal, rel=0.02)


def test_counterbore_peels_to_two_cuts(tmp_path, ocp_available):
    from src.cad_build import features_to_step
    from src.reconstruct import SnapList, detect_features

    part = trimesh.creation.box(extents=[30.0, 20.0, 10.0])
    part = part.difference(trimesh.creation.cylinder(radius=3.0, height=40.0, sections=64))
    cbore = trimesh.creation.cylinder(radius=5.0, height=6.0, sections=64)
    cbore.apply_translation([0.0, 0.0, 2.0])  # Ø10 blind from top down to z=-1
    part = part.difference(cbore)

    features = detect_features(part.vertices, part.faces, snap=SnapList.default())
    result = features_to_step(features, str(tmp_path / "cbore.step"), verts=part.vertices)

    assert result.holes_cut == 2
    assert result.watertight
    expected = 30 * 20 * 10 - math.pi * 5.0**2 * 6 - math.pi * 3.0**2 * 4
    assert result.volume == pytest.approx(expected, rel=2e-3)


def test_round_bar_wall_rebuilt(tmp_path, ocp_available):
    from src.cad_build import features_to_step
    from src.reconstruct import detect_features

    bar = trimesh.creation.cylinder(radius=10.0, height=30.0, sections=48)
    features = detect_features(bar.vertices, bar.faces)
    result = features_to_step(features, str(tmp_path / "bar.step"), verts=bar.vertices)

    assert result.walls_rebuilt == 1
    assert result.watertight
    assert result.volume == pytest.approx(math.pi * 100.0 * 30.0, rel=0.01)


# --------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------- #
def test_rejects_non_step_extension(tmp_path, ocp_available):
    from src.cad_build import features_to_step
    from src.reconstruct import detect_features

    box = trimesh.creation.box()
    features = detect_features(box.vertices, box.faces)
    with pytest.raises(ValueError, match=r"\.step or \.stp"):
        features_to_step(features, str(tmp_path / "out.txt"))


def test_bbox_fallback_without_source_vertices(tmp_path, ocp_available):
    from src.cad_build import features_to_step
    from src.reconstruct import detect_features

    box = trimesh.creation.box(extents=[20.0, 15.0, 4.0])
    features = detect_features(box.vertices, box.faces)
    result = features_to_step(features, str(tmp_path / "noverts.step"))

    assert result.base in ("planes", "bbox", "hull+planes", "hull")
    assert result.watertight
    assert result.volume == pytest.approx(1200.0, rel=1e-4)


def test_noisy_scan_rebuilds_to_nominal(tmp_path, ocp_available):
    from src.cad_build import features_to_step
    from src.reconstruct import DetectOptions, SnapList, detect_features

    part = _part_with_holes()
    rng = np.random.default_rng(42)
    noisy = np.asarray(part.vertices) + rng.normal(0.0, 0.05, part.vertices.shape)
    opts = DetectOptions(plane_distance=0.2, cylinder_distance=0.2)
    features = detect_features(noisy, part.faces, snap=SnapList.default(), opts=opts)
    result = features_to_step(features, str(tmp_path / "noisy.step"), verts=noisy)

    assert result.holes_cut == 2
    assert result.watertight
    expected = 40 * 30 * 10 - math.pi * 3.0**2 * 10 - math.pi * 4.0**2 * 6
    # scan noise perturbs hole positions slightly; nominal diameters are exact
    assert result.volume == pytest.approx(expected, rel=0.01)
