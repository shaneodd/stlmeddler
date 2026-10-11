"""
Rebuild clean CAD solids from detected features and export STEP.

Consumes the :class:`~src.reconstruct.FeatureSet` produced by the detection
stage and constructs a kernel-native solid instead of a faceted scan:

* the base body is the intersection of the half-spaces kept by every detected
  plane (a clean prism), falling back to the bounding box when that fails;
* concave (drilled) holes are cut as exact cylinders at their nominal -- or,
  when unsnapped, raw -- diameter; through holes clear the part at both ends,
  blind holes end at the detected floor;
* convex detections spanning >= 270 degrees (round outer walls such as washer
  rims or bar stock) intersect the base with a nominal cylinder, restoring a
  true revolved profile.

The heavy Open Cascade bindings are shared with :mod:`src.step_ops` and loaded
lazily, so importing this module stays cheap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .reconstruct import DetectedHole, FeatureSet
from .step_ops import (
    _build_shape,
    _count_subshapes,
    _is_valid,
    _load_ocp,
    _read_step,
    _shape_volume,
    _write_step,
)

CONVEX_ARC_MIN_DEG = 270.0  # full round outer wall worth rebuilding as cylinder
_FLOOR_EPS = 0.005  # overshoot past a blind hole floor for a clean cut
_PLANE_SAFE_EPS = 1.0  # a hull-bound plane may trail the farthest point by this


@dataclass(frozen=True)
class BuildResult:
    """Summary of a features -> STEP rebuild; see :func:`features_to_step`."""

    path: str
    base: str  # "planes" (half-space intersection) or "bbox" fallback
    holes_cut: int
    walls_rebuilt: int
    solids: int
    valid: bool
    watertight: bool
    volume: float

    @property
    def summary(self) -> str:
        state = f"solid ({self.volume:.2f} mm3)" if self.watertight else "invalid shape"
        parts = [
            f"base from {self.base}",
            f"{self.holes_cut} hole cut(s)",
            f"{self.walls_rebuilt} round wall(s)",
            state,
        ]
        if not self.valid:
            parts.append("warning: kernel flagged the shape as invalid")
        return ", ".join(parts)


class _Fallback(Exception):
    """Internal signal: half-space base construction failed, use bbox."""


def _bbox_corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.array(list(product(*zip(lo, hi, strict=True))))


def _rotation_z_to(normal: np.ndarray, ocp: SimpleNamespace) -> Any:
    """Rotation taking +Z onto ``normal`` (gp_Quaternion chokes on 180 deg)."""
    trs = ocp.gp.gp_Trsf()
    dot = float(np.dot([0.0, 0.0, 1.0], normal))
    if dot < -0.9999:
        perp = np.cross([0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-9:  # pragma: no cover - defensive
            perp = np.cross([0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
        perp = perp / np.linalg.norm(perp)
        trs.SetRotation(
            ocp.gp.gp_Ax1(
                ocp.gp.gp_Pnt(0.0, 0.0, 0.0),
                ocp.gp.gp_Dir(float(perp[0]), float(perp[1]), float(perp[2])),
            ),
            math.pi,
        )
    elif dot < 0.9999:
        trs.SetRotation(
            ocp.gp.gp_Quaternion(ocp.gp.gp_Vec(0.0, 0.0, 1.0), ocp.gp.gp_Vec(*normal))
        )
    return trs


def _half_space(normal: np.ndarray, offset: float, diag: float, ocp: SimpleNamespace) -> Any:
    """A big box occupying the half-space ``normal . p <= offset``."""
    length = 10.0 * diag
    box = ocp.prim.BRepPrimAPI_MakeBox(
        ocp.gp.gp_Pnt(-length, -length, -2.0 * length),
        ocp.gp.gp_Pnt(length, length, 0.0),
    ).Shape()
    move = ocp.gp.gp_Trsf()
    move.SetTranslationPart(
        ocp.gp.gp_Vec(
            float(offset * normal[0]),
            float(offset * normal[1]),
            float(offset * normal[2]),
        )
    )
    combined = move.Multiplied(_rotation_z_to(normal, ocp))
    return ocp.builder.BRepBuilderAPI_Transform(box, combined).Shape()


def _hull_solid(verts: np.ndarray, ocp: SimpleNamespace) -> Any:
    """Kernel solid bounded by the mesh's convex hull."""
    import trimesh

    hull = trimesh.convex.convex_hull(np.asarray(verts, dtype=float))
    shape, _shells, solids, closed = _build_shape(
        np.asarray(hull.vertices), np.asarray(hull.faces), 1e-4, ocp
    )
    if not closed or solids < 1:
        raise _Fallback
    return shape


def _base_solid(
    features: FeatureSet, ocp: SimpleNamespace, verts: np.ndarray | None = None
) -> tuple[Any, str]:
    """Convex hull intersected with the detected planes, with fallbacks.

    Only planes that keep the entire mesh on one side are used: a plane
    through a concave feature (e.g. a blind-hole floor) is not a hull bound
    and slicing the solid with it would gut the part. Without source vertices
    the mesh's bbox corners stand in for the same safety test.
    """
    lo = np.asarray(features.bbox_min, dtype=float)
    hi = np.asarray(features.bbox_max, dtype=float)
    center = (lo + hi) / 2.0
    diag = max(float(np.linalg.norm(hi - lo)), 1e-6)
    probe = verts if verts is not None and len(verts) else _bbox_corners(lo, hi)

    safe: list[tuple[np.ndarray, float]] = []
    for plane in features.planes:
        normal = np.asarray(plane.normal, dtype=float)
        offset = float(plane.offset)
        if float(normal @ center) > offset:  # keep the centre's side
            normal, offset = -normal, -offset
        if float((probe @ normal).max()) <= offset + _PLANE_SAFE_EPS:
            safe.append((normal, offset))

    def intersect(start: Any) -> Any:
        solid = start
        for normal, offset in safe:
            op = ocp.algo.BRepAlgoAPI_Common(solid, _half_space(normal, offset, diag, ocp))
            solid = op.Shape()  # IsDone unreliable in this binding; validate below
        if not _is_valid(solid, ocp) or _shape_volume(solid, ocp) <= 0.0:
            raise _Fallback
        return solid

    bbox_box = ocp.prim.BRepPrimAPI_MakeBox(
        ocp.gp.gp_Pnt(float(lo[0]), float(lo[1]), float(lo[2])),
        ocp.gp.gp_Pnt(float(hi[0]), float(hi[1]), float(hi[2])),
    ).Shape()

    if verts is not None and len(verts):
        try:
            return intersect(_hull_solid(verts, ocp)), ("hull+planes" if safe else "hull")
        except _Fallback:
            pass
    if safe:
        try:
            return intersect(bbox_box), "planes"
        except _Fallback:
            pass
    return bbox_box, "bbox"


def _hole_cylinder(
    hole: DetectedHole,
    lo: np.ndarray,
    hi: np.ndarray,
    margin: float,
    ocp: SimpleNamespace,
) -> Any | None:
    """Nominal cylinder spanning the correct axial range for this hole."""
    axis = np.asarray(hole.axis, dtype=float)
    center = np.asarray(hole.center, dtype=float)
    diameter = hole.diameter if hole.diameter is not None else hole.raw_diameter
    radius = float(diameter) / 2.0
    if radius <= 0.0:
        return None

    projections = _bbox_corners(lo, hi) @ axis
    t_min, t_max = float(projections.min()), float(projections.max())
    t_mid = float(center @ axis)  # cylinder axis passes through q at t = 0
    q = center - axis * t_mid

    if hole.concave and not hole.is_through:
        t_lo = t_mid - hole.depth / 2.0
        t_hi = t_mid + hole.depth / 2.0
        if abs(t_lo - t_min) <= abs(t_hi - t_max):  # opening faces t_min side
            start, end = t_min - margin, t_hi + _FLOOR_EPS
        else:
            start, end = t_lo - _FLOOR_EPS, t_max + margin
    else:
        start, end = t_min - margin, t_max + margin

    base_point = q + axis * start
    ax2 = ocp.gp.gp_Ax2(
        ocp.gp.gp_Pnt(float(base_point[0]), float(base_point[1]), float(base_point[2])),
        ocp.gp.gp_Dir(float(axis[0]), float(axis[1]), float(axis[2])),
    )
    try:
        # IsDone lies False in this binding even when Shape() is a good
        # cylinder (same trap as MakeSolid); validate by use instead.
        return ocp.prim.BRepPrimAPI_MakeCylinder(ax2, radius, end - start).Shape()
    except Exception:
        return None


def features_to_step(
    features: FeatureSet,
    path: str,
    verts: np.ndarray | None = None,
    margin: float = 1.0,
    convex_arc_min: float = CONVEX_ARC_MIN_DEG,
    verify: bool = True,
) -> BuildResult:
    """Build a kernel-native STEP file from a detected feature set.

    Args:
        features: Output of :func:`src.reconstruct.detect_features`.
        path: Output ``.step`` / ``.stp`` path.
        verts: Source mesh vertices; when given, the base body starts from the
            convex hull (faithful to non-prismatic outlines) instead of the
            bounding box.
        margin: Overshoot (mm) with which through cuts clear the part.
        convex_arc_min: Minimum angular span (deg) for a convex detection to
            rebuild the outer wall as a nominal cylinder.
        verify: Re-read the written file and count the solids found there.

    Returns:
        :class:`BuildResult` describing the built solid.

    Raises:
        ValueError: On a non-STEP output extension.
        RuntimeError: When cadquery-ocp is missing or the write fails.
    """
    ocp = _load_ocp()
    out = Path(path)
    if out.suffix.lower() not in (".step", ".stp"):
        raise ValueError(f"Output path must end in .step or .stp, got {out.suffix!r}")

    lo = np.asarray(features.bbox_min, dtype=float)
    hi = np.asarray(features.bbox_max, dtype=float)
    solid, base = _base_solid(features, ocp, verts)

    holes_cut = 0
    walls = 0
    for hole in features.holes:
        if not hole.concave and hole.arc_deg < convex_arc_min:
            continue  # partial outer arc: the planar base already bounds it
        cyl = _hole_cylinder(hole, lo, hi, margin, ocp)
        if cyl is None:
            continue
        if hole.concave:
            op = ocp.algo.BRepAlgoAPI_Cut(solid, cyl)
        else:
            op = ocp.algo.BRepAlgoAPI_Common(solid, cyl)
        try:
            # IsDone is unreliable in this binding; keep the result only when
            # the kernel validates it and the volume stayed sane.
            candidate = op.Shape()
        except Exception:
            continue
        if not _is_valid(candidate, ocp) or _shape_volume(candidate, ocp) <= 0.0:
            continue  # boolean went wrong: keep the solid before this hole
        solid = candidate
        if hole.concave:
            holes_cut += 1
        else:
            walls += 1

    valid = _is_valid(solid, ocp)
    volume = _shape_volume(solid, ocp)
    _write_step(solid, str(out), ocp)

    check_shape = _read_step(str(out), ocp) if verify else solid
    solids = _count_subshapes(check_shape, ocp, ocp.topabs.TopAbs_ShapeEnum.TopAbs_SOLID)
    watertight = valid and solids >= 1 and volume > 0.0

    return BuildResult(
        path=str(out),
        base=base,
        holes_cut=holes_cut,
        walls_rebuilt=walls,
        solids=solids,
        valid=valid,
        watertight=watertight,
        volume=volume,
    )
