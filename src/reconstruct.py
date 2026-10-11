"""
Reverse-engineering of scanned/scan-derived meshes into CAD features.

Scans of machined (prismatic) parts carry the design intent implicitly:
flat faces, drilled holes of standard diameters. This module detects that
structure from a triangle mesh - planar faces (iterative RANSAC) and
cylindrical holes (face-normal axis estimation refined by least-squares
circle fitting) - and snaps the noisy measurements to nominal dimensions
taken from a JSON snap list (standard drill diameters by default, editable
by the user).

The result is a :class:`FeatureSet`: planes with normals/offsets and holes
with axis, centre, snapped diameter and depth, ready to drive clean B-rep
reconstruction or a review dialog.

Pure numpy/scipy/trimesh/open3d - no Qt, fully testable headlessly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Standard metric drill diameters (mm) used when no snap list is given.
DEFAULT_DRILL_DIAMETERS: tuple[float, ...] = (
    2.0, 2.5, 3.0, 3.2, 3.5, 4.0, 4.2, 4.5, 5.0, 5.5, 6.0, 6.35, 6.5, 6.6,
    6.8, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 12.0, 12.7, 13.0,
    14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 22.0, 25.0,
)


def _xyz(v: np.ndarray) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


@dataclass(frozen=True)
class SnapList:
    """Nominal dimensions that detected features snap to.

    Diameters are in millimetres; a detected value within ``tolerance`` of a
    nominal one is reported as that nominal dimension.
    """

    hole_diameters: tuple[float, ...] = DEFAULT_DRILL_DIAMETERS
    tolerance: float = 0.15

    @staticmethod
    def default() -> SnapList:
        return SnapList()

    @classmethod
    def load(cls, path: str | Path) -> SnapList:
        """Read ``{"hole_diameters_mm": [...], "tolerance_mm": 0.15}`` from JSON."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        diams = raw.get("hole_diameters_mm")
        tol = raw.get("tolerance_mm")
        if diams is not None:
            diams = tuple(sorted(float(d) for d in diams if float(d) > 0.0))
            if not diams:
                raise ValueError(f"No usable hole diameters in {path}")
        else:
            diams = DEFAULT_DRILL_DIAMETERS
        return cls(hole_diameters=diams, tolerance=float(tol) if tol else 0.15)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "hole_diameters_mm": list(self.hole_diameters),
                    "tolerance_mm": self.tolerance,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def snap_diameter(self, measured: float) -> tuple[float | None, float]:
        """Snap ``measured`` to the nearest nominal diameter within tolerance.

        Returns ``(snapped_or_None, delta)`` where delta is the distance to
        the nearest nominal whether or not it was close enough to snap.
        """
        arr = np.asarray(self.hole_diameters)
        deltas = np.abs(arr - float(measured))
        i = int(np.argmin(deltas))
        if deltas[i] <= self.tolerance:
            return float(arr[i]), float(deltas[i])
        return None, float(deltas[i])


@dataclass(frozen=True)
class DetectedPlane:
    """A planar face region: unit normal, signed offset, centroid and area."""

    normal: tuple[float, float, float]
    offset: float  # plane equation: normal . x == offset
    centroid: tuple[float, float, float]
    area: float
    inlier_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "normal": list(self.normal),
            "offset": self.offset,
            "centroid": list(self.centroid),
            "area": self.area,
            "inlier_fraction": self.inlier_fraction,
        }


@dataclass(frozen=True)
class DetectedHole:
    """A cylindrical hole: axis line, diameter (raw and snapped), depth."""

    center: tuple[float, float, float]  # point on axis at mid-axial extent
    axis: tuple[float, float, float]  # unit direction
    raw_diameter: float
    diameter: float | None  # snapped nominal; None when nothing is close
    depth: float  # axial extent of the detected cylinder surface
    is_through: bool
    concave: bool  # True for drilled holes, False for convex outer walls
    inlier_fraction: float
    arc_deg: float  # angular span of the detected surface

    def to_dict(self) -> dict[str, Any]:
        return {
            "center": list(self.center),
            "axis": list(self.axis),
            "raw_diameter": self.raw_diameter,
            "diameter": self.diameter,
            "depth": self.depth,
            "is_through": self.is_through,
            "concave": self.concave,
            "inlier_fraction": self.inlier_fraction,
            "arc_deg": self.arc_deg,
        }


@dataclass(frozen=True)
class FeatureSet:
    """All features detected in one mesh, plus the raw measurements."""

    planes: list[DetectedPlane]
    holes: list[DetectedHole]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    triangles: int

    @property
    def bbox_size(self) -> tuple[float, float, float]:
        return _xyz(np.subtract(self.bbox_max, self.bbox_min))

    def summary(self) -> str:
        holes = [h for h in self.holes if h.concave]
        parts = [f"{len(self.planes)} planes", f"{len(holes)} holes"]
        parts.append("bbox " + " x ".join(f"{v:.2f}" for v in self.bbox_size) + " mm")
        unsnapped = sum(1 for h in holes if h.diameter is None)
        if unsnapped:
            parts.append(f"{unsnapped} hole(s) did not snap to a nominal diameter")
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "planes": [p.to_dict() for p in self.planes],
            "holes": [h.to_dict() for h in self.holes],
            "bbox_min": list(self.bbox_min),
            "bbox_max": list(self.bbox_max),
            "triangles": self.triangles,
        }


@dataclass(frozen=True)
class DetectOptions:
    """Tuning knobs for :func:`detect_features` (all distances in mm)."""

    plane_distance: float = 0.15  # RANSAC inlier distance for planes
    plane_min_seed_faces: int = 12  # min face centres for a RANSAC plane seed
    plane_min_area_fraction: float = 0.01  # ignore tiny planar regions
    plane_min_width: float = 0.5  # reject needle-thin tangent strips (mm)
    plane_max_planes: int = 12
    cylinder_distance: float = 0.15  # radial inlier distance for cylinders
    cylinder_min_radius: float = 0.75
    cylinder_max_radius: float = 60.0
    cylinder_min_triangles: int = 30  # ignore tiny curved patches
    cylinder_min_area: float = 30.0  # min on-surface patch area (mm2)
    through_span_fraction: float = 0.9  # depth vs part extent => through hole


def _weld(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    tri = trimesh.Trimesh(
        vertices=np.asarray(verts, dtype=float), faces=np.asarray(faces), process=True
    )
    return np.asarray(tri.vertices), np.asarray(tri.faces)


def _tri_areas(tris: np.ndarray) -> np.ndarray:
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def _extract_planes(
    verts: np.ndarray, faces: np.ndarray, opts: DetectOptions
) -> tuple[list[DetectedPlane], np.ndarray]:
    """Iteratively RANSAC planes; returns them plus the non-planar face indices.

    RANSAC can lock onto degenerate supports (e.g. two offset circular rims
    whose union beats any real face), so each seed is grown and refit: faces
    aligned with the seed and close to it are owned, the plane is refit by
    least squares to the owned vertices, and ownership recomputed. A strip of
    a real face therefore grows into the whole face, while a spurious
    tangent plane collapses onto whichever real face it grazed.
    """
    import open3d as o3d

    mesh_area = float(_tri_areas(verts[faces]).sum())
    total_area = max(mesh_area, 1e-12)
    fn = _face_normals(verts[faces])
    centres = verts[faces].mean(axis=1)
    # ``pool`` are faces still eligible as RANSAC seed centres; ``remaining``
    # are faces not yet owned by a plane. A degenerate seed shrinks only the
    # pool - its faces stay candidates for later seeds and cylinder fitting.
    pool = np.ones(len(faces), dtype=bool)
    remaining = np.ones(len(faces), dtype=bool)
    planes: list[DetectedPlane] = []
    # Low-poly CAD meshes (a primitive box face is 2 triangles) could never
    # satisfy the scan-oriented seed floor, so scale it to tiny inputs; the
    # width and area gates still reject spurious planes on curved surfaces.
    min_seed = min(opts.plane_min_seed_faces, max(2, len(faces) // 6))

    def grow(
        normal: np.ndarray, offset: float, min_faces: int
    ) -> tuple[np.ndarray, float, np.ndarray, float] | None:
        """Grow a seed plane to owned faces; None when a gate rejects it."""
        owned = np.zeros(len(faces), dtype=bool)
        for _ in range(4):  # own -> refit -> re-own until the set settles
            grown = _own_faces(verts, faces, fn, pool, normal, offset, opts)
            if not grown.any() or np.array_equal(grown, owned):
                break
            owned = grown
            refit = _refine_plane_to(verts[np.unique(faces[owned].ravel())])
            if refit is None:
                break
            normal, offset = refit
            owned = _own_faces(verts, faces, fn, pool, normal, offset, opts)
        if owned.sum() < min_faces:
            return None
        if not _region_has_width(verts[np.unique(faces[owned].ravel())], opts):
            return None  # tangent sliver on a curved surface, not a real face
        area = float(_tri_areas(verts[faces[owned]]).sum())
        if area / total_area < opts.plane_min_area_fraction:
            return None
        return normal, offset, owned, area

    def record(normal: np.ndarray, offset: float, owned: np.ndarray, area: float) -> None:
        cen = verts[np.unique(faces[owned].ravel())].mean(axis=0)
        planes.append(
            DetectedPlane(
                normal=_xyz(normal),
                offset=float(cen @ normal),
                centroid=_xyz(cen),
                area=area,
                inlier_fraction=float(owned.sum()) / len(faces),
            )
        )
        np.logical_and(remaining, ~owned, out=remaining)
        np.logical_and(pool, ~owned, out=pool)

    def seed_from_faces(seeds: np.ndarray) -> int:
        """Grow each seed face's own plane; returns how many planes got recorded."""
        found = 0
        for f in seeds:
            if not pool[f]:
                continue  # consumed by an earlier seed this pass
            result = grow(fn[f].copy(), float(centres[f] @ fn[f]), min_seed)
            if result is not None:
                record(*result)
                found += 1
        return found

    for _ in range(opts.plane_max_planes):
        idx = np.nonzero(pool)[0]
        if len(idx) < 2:
            break
        # Seed from face CENTRES: removed faces' vertices linger (shared with
        # walls) and would drag RANSAC back onto planes already taken.
        model = None
        inl: np.ndarray = np.empty(0, dtype=int)
        if len(idx) >= max(3, min_seed):
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centres[idx]))
            try:
                model, inl = pc.segment_plane(
                    distance_threshold=opts.plane_distance,
                    ransac_n=3,
                    num_iterations=1000,
                )
            except Exception:
                model = None
            if model is not None and len(inl) >= min_seed:
                normal = np.asarray(model[:3], dtype=float)
                seed_norm = float(np.linalg.norm(normal))
                offset = 0.0
                if seed_norm >= 1e-12:
                    normal /= seed_norm
                    offset = -float(model[3]) / seed_norm
                    result = grow(normal, offset, 1)
                    if result is not None:
                        record(*result)
                        continue
                # Degenerate support - the RANSAC plane holds no face at all
                # (e.g. on a box the four side-face CENTRES are coplanar).
                # The support faces themselves are real; seed their own planes.
                support = idx[np.asarray(inl, dtype=int)]
                if seed_from_faces(support):
                    continue
                gone = np.zeros(len(faces), dtype=bool)
                gone[support] = True
                gone &= pool
                if not gone.any():
                    break
                pool &= ~gone
                continue
        if len(idx) > max(2 * min_seed, 64):
            break  # dense leftover (curved walls): exhaustive seeding too costly
        if not seed_from_faces(idx):
            break

    # Faces aligned with a found plane but missed by ownership (noisy fringe
    # beyond the strict tolerance) still belong to that plane, not to any
    # cylinder - prune them from the curved remainder.
    for p in planes:
        n = np.asarray(p.normal)
        aligned = np.abs(fn @ n) >= 0.98
        near = np.all(
            np.abs(verts[faces] @ n - p.offset) <= max(3.0 * opts.plane_distance, 0.5),
            axis=1,
        )
        remaining &= ~(aligned & near)

    return planes, np.nonzero(remaining)[0]


def _region_has_width(pts: np.ndarray, opts: DetectOptions) -> bool:
    """False when the owned points form a needle-thin strip (tangent plane)."""
    if len(pts) < 4:
        return False
    c = pts - pts.mean(axis=0)
    try:
        s = np.linalg.svd(c, compute_uv=False)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        return False
    std_minor = s[1] / np.sqrt(max(len(pts) - 1, 1))
    return bool(std_minor >= opts.plane_min_width)


def _own_faces(
    verts: np.ndarray,
    faces: np.ndarray,
    fnormals: np.ndarray,
    candidate: np.ndarray,
    normal: np.ndarray,
    offset: float,
    opts: DetectOptions,
) -> np.ndarray:
    """Faces whose three vertices lie on the plane and whose normal aligns with it."""
    on_plane = np.all(np.abs(verts[faces] @ normal - offset) <= opts.plane_distance, axis=1)
    aligned = np.abs(fnormals @ normal) >= 0.98
    return on_plane & aligned & candidate


def _refine_plane_to(pts: np.ndarray) -> tuple[np.ndarray, float] | None:
    if len(pts) < 3:
        return None
    cen = pts.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(pts - cen, full_matrices=False)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        return None
    normal = vt[2]
    return normal, float(cen @ normal)


def _component_indices(faces: np.ndarray) -> list[np.ndarray]:
    """Face-connected components of the given face list (union-find over edges)."""
    n = len(faces)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edge_map: dict[tuple[int, int], int] = {}
    for fi, tri in enumerate(faces):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for e in ((a, b), (b, c), (c, a)):
            key = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
            if key in edge_map:
                ra, rb = find(edge_map[key]), find(fi)
                if ra != rb:
                    parent[rb] = ra
            else:
                edge_map[key] = fi

    groups: dict[int, list[int]] = {}
    for fi in range(n):
        groups.setdefault(find(fi), []).append(fi)
    return [np.asarray(g) for g in groups.values()]


def _face_normals(tris: np.ndarray) -> np.ndarray:
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(fn, axis=1)
    fn[ln > 1e-12] /= ln[ln > 1e-12][:, None]
    return fn


def _cylinder_axis_votes(
    fnormals: np.ndarray, areas: np.ndarray, k: int = 4
) -> list[tuple[np.ndarray, float]]:
    """Top-k cylinder-axis candidates with their area-explained fractions.

    A cylinder's face normals are all perpendicular to its axis, so the axis
    is the direction with maximal area-weighted perpendicularity. Candidates
    come from cross products of sampled normal pairs (each perpendicular to
    two observed normals, exactly as a true axis must be) plus a PCA
    fallback. Voting by area beats PCA alone because boolean rims leave
    tilted fringe faces glued to the wall that smear the covariance into
    near-isotropy while still being a minority of the area. Several
    candidates are kept because a heavy-noise component can put a spurious
    patch above a real bore in the ranking.
    """
    w = np.clip(areas, 0.0, None)
    total = float(w.sum())
    if total <= 0.0:
        return []

    cands: list[np.ndarray] = []
    cov = (fnormals * w[:, None]).T @ fnormals
    try:
        evals, evecs = np.linalg.eigh(cov)
        if evals[2] > 1e-10:
            cands.append(evecs[:, 0])
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        pass
    step = max(len(fnormals) // 60, 1)
    sub = fnormals[::step]
    for i in range(len(sub) - 1):
        cross = np.cross(sub[i], sub[i + 1 :])
        norms = np.linalg.norm(cross, axis=1)
        keep = norms > 0.2  # near-parallel pairs give unstable axes
        if keep.any():
            cands.extend(list(cross[keep] / norms[keep][:, None]))
    if not cands:
        return []

    A = np.asarray(cands)
    scores = np.empty(len(A))
    for lo in range(0, len(A), 512):  # chunk to bound the (faces x cands) matrix
        block = A[lo : lo + 512]
        scores[lo : lo + 512] = (np.abs(fnormals @ block.T) <= 0.2).T @ w

    out: list[tuple[np.ndarray, float]] = []
    for idx in np.argsort(scores)[::-1]:
        a = A[idx] / np.linalg.norm(A[idx])
        if all(abs(a @ b) < 0.98 for b, _ in out):  # skip parallel duplicates
            out.append((a, float(scores[idx]) / total))
        if len(out) >= k:
            break
    return out


def _fit_cylinder(
    faces: np.ndarray,
    centres: np.ndarray,
    fnormals: np.ndarray,
    areas: np.ndarray,
    opts: DetectOptions,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray] | None:
    """Fit a cylinder to a curved patch of faces (area-weighted, sliver-proof).

    Returns ``(axis, point_on_axis, radius, inlier_face_mask)`` or None when
    the patch is not cylindrical. The axis is voted from face normals, then
    ownership (normal perpendicular to the axis AND centre at the fitted
    radius) and the fit alternate until the owned set settles - the same
    grow-and-refit trick that keeps RANSAC planes from latching onto
    degenerate supports. Area weighting makes rim slivers negligible.
    """
    from scipy.optimize import least_squares

    if len(centres) < 12:
        return None
    w = np.clip(areas, 0.0, None)
    if w.sum() <= 0.0:
        return None

    votes = _cylinder_axis_votes(fnormals, w)

    def radial_fit(
        pts: np.ndarray, pts_w: np.ndarray, d: np.ndarray, q0: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        ref = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(d, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(d, e1)
        step = max(len(pts) // 2000, 1)
        s, sw = pts[::step], np.sqrt(pts_w[::step])

        def direction(p: np.ndarray) -> np.ndarray:
            v = d + np.sin(p[0]) * e1 + np.sin(p[1]) * e2
            return v / np.linalg.norm(v)

        def residuals(p: np.ndarray) -> np.ndarray:
            dd = direction(p)
            q = q0 + p[2] * e1 + p[3] * e2
            v = s - q
            v -= (v @ dd)[:, None] * dd
            return (np.linalg.norm(v, axis=1) - p[4]) * sw

        dd0 = direction(np.zeros(5))
        v = s - q0
        v -= (v @ dd0)[:, None] * dd0
        r0 = float(np.average(np.linalg.norm(v, axis=1), weights=pts_w[::step]))
        if not opts.cylinder_min_radius <= r0 <= opts.cylinder_max_radius:
            return None
        try:
            fit = least_squares(residuals, np.zeros(5), max_nfev=200)
        except Exception:  # pragma: no cover - scipy failure on degenerate patches
            return None
        d2 = direction(fit.x)
        q = q0 + fit.x[2] * e1 + fit.x[3] * e2
        r = abs(float(fit.x[4]))
        if not opts.cylinder_min_radius <= r <= opts.cylinder_max_radius:
            return None
        return d2, q, r

    def try_axis(
        axis: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray] | None:
        owned = np.abs(fnormals @ axis) <= 0.35
        d, q, r = axis, centres.mean(axis=0), 0.0
        for _ in range(3):
            if owned.sum() < 6:
                return None
            q0 = np.average(centres[owned], axis=0, weights=w[owned])
            q0 = q0 - (q0 @ axis) * axis
            fit = radial_fit(centres[owned], w[owned], axis, q0)
            if fit is None:
                return None
            d, q, r = fit
            v = centres - q
            v -= (v @ d)[:, None] * d
            new_owned = (
                np.abs(np.linalg.norm(v, axis=1) - r) <= opts.cylinder_distance
            ) & (np.abs(fnormals @ d) <= 0.35)
            if new_owned.sum() < 6:
                return None
            cov = (fnormals[new_owned] * w[new_owned, None]).T @ fnormals[new_owned]
            try:
                evals, evecs = np.linalg.eigh(cov)
                if evals[2] > 1e-10 and evals[0] / max(evals[1], 1e-18) <= 0.2:
                    axis = _orient(evecs[:, 0])
            except np.linalg.LinAlgError:  # pragma: no cover - defensive
                pass
            if np.array_equal(new_owned, owned):
                break
            owned = new_owned

        inlier_area = float(w[owned].sum())
        if inlier_area < opts.cylinder_min_area:
            return None
        # A real bore wall is one contiguous band; a noise artifact gathers a
        # scatter of tiny unrelated patches from all over the part.
        if len(_component_indices(faces[owned])) > max(2, len(owned) // 10):
            return None
        return d, q, r, owned

    for axis, cover in votes:
        if cover < 0.25:
            break  # remaining candidates explain too little area to be axes
        out = try_axis(_orient(np.asarray(axis, dtype=float)))
        if out is not None:
            return out
    return None


def _arc_deg(pts: np.ndarray, axis: np.ndarray, center: np.ndarray) -> float:
    """Angular span (degrees) of the points around the cylinder axis."""
    v = pts - center
    v -= (v @ axis)[:, None] * axis
    norms = np.linalg.norm(v, axis=1)
    ok = norms > 1e-9
    if not ok.any():
        return 0.0
    ref = v[ok][0] / norms[ok][0]
    e = np.cross(axis, ref)
    ang = np.sort(np.arctan2(v[ok] @ e, v[ok] @ ref))
    gaps = np.append(np.diff(ang), 2 * np.pi - (ang[-1] - ang[0]))
    return float(np.degrees(2 * np.pi - gaps.max()))


def _axis_extent(verts: np.ndarray, axis: np.ndarray) -> float:
    t = verts @ axis
    return float(t.max() - t.min())


def _orient(axis: np.ndarray) -> np.ndarray:
    """Deterministic sign: make the largest-magnitude component positive."""
    a = np.asarray(axis, dtype=float)
    if a[int(np.argmax(np.abs(a)))] < 0:
        a = -a
    return a


def detect_features(
    verts: np.ndarray,
    faces: np.ndarray,
    snap: SnapList | None = None,
    opts: DetectOptions | None = None,
) -> FeatureSet:
    """Detect planar faces and cylindrical holes, snapping hole diameters.

    Args:
        verts: (N,3) vertex coordinates in millimetres.
        faces: (M,3) triangle indices.
        snap: Nominal diameter list to snap to; defaults to standard drills.
        opts: Detection tuning; see :class:`DetectOptions`.

    Returns:
        :class:`FeatureSet` (planes by area, holes by diameter).
        ``DetectedHole.diameter`` holds the snapped nominal, or ``None``
        when the measurement is farther than the snap tolerance from every
        nominal value.
    """
    opts = opts or DetectOptions()
    snap = snap or SnapList.default()
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces)
    if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) < 3:
        raise ValueError("verts must be an (N,3) array with N >= 3")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("faces must be a non-empty (M,3) index array")
    if faces.min() < 0 or faces.max() >= len(verts):
        raise ValueError("faces reference vertex indices outside verts")

    verts, faces = _weld(verts, faces)
    planes, curved = _extract_planes(verts, faces, opts)

    holes: list[DetectedHole] = []
    if len(curved) >= opts.cylinder_min_triangles:
        tris = verts[faces]
        fn = _face_normals(tris)
        areas = _tri_areas(tris)
        curved_faces = faces[curved]

        for comp in _component_indices(curved_faces):
            # A component can stack several coaxial bores (e.g. a counterbore);
            # peel one cylinder at a time and refit what the inliers leave.
            alive = np.ones(len(comp), dtype=bool)
            for _ in range(4):
                pos = np.nonzero(alive)[0]
                if len(pos) < opts.cylinder_min_triangles:
                    break
                f = curved_faces[comp[pos]]
                comp_fn = fn[curved][comp[pos]]
                comp_areas = areas[curved][comp[pos]]
                comp_centres = verts[f].mean(axis=1)
                fit = _fit_cylinder(f, comp_centres, comp_fn, comp_areas, opts)
                if fit is None:
                    break
                axis, q, r, inliers = fit
                if not inliers.any():
                    break
                alive[pos[~inliers]] = False
                axis = _orient(axis)
                # Concavity: hole walls face the axis, convex outer walls away.
                radial = comp_centres - q
                radial -= (radial @ axis)[:, None] * axis
                rl = np.linalg.norm(radial, axis=1)
                ok = inliers & (rl > 1e-9)
                concavity = (
                    float(np.mean((comp_fn[ok] * (radial[ok] / rl[ok, None])).sum(axis=1)))
                    if ok.any()
                    else 0.0
                )
                # Depth from the inlier wall faces' vertices so rim slivers and
                # planar fringe cannot stretch the span.
                wall_pts = verts[np.unique(f[inliers].ravel())]
                t = (wall_pts - q) @ axis
                span = float(t.max() - t.min())
                extent = _axis_extent(verts, axis)
                if span > extent + 2.0 * opts.cylinder_distance:
                    continue  # longer than the part along its own axis: noise artifact
                cen = q + 0.5 * (t.min() + t.max()) * axis
                raw_d = 2.0 * r
                # Fringe left by a previous peel can refit onto the same bore.
                if any(
                    abs(float(axis @ np.asarray(h.axis))) >= 0.99
                    and abs(raw_d - h.raw_diameter) <= max(0.5, 2 * opts.cylinder_distance)
                    and np.linalg.norm(cen - np.asarray(h.center))
                    <= max(0.5, 2 * opts.cylinder_distance)
                    for h in holes
                ):
                    continue
                is_through = span >= opts.through_span_fraction * extent
                snapped, _delta = snap.snap_diameter(raw_d)
                holes.append(
                    DetectedHole(
                        center=_xyz(cen),
                        axis=_xyz(axis),
                        raw_diameter=raw_d,
                        diameter=snapped,
                        depth=span,
                        is_through=is_through,
                        concave=concavity < 0.0,
                        inlier_fraction=float(
                            comp_areas[inliers].sum() / max(comp_areas.sum(), 1e-12)
                        ),
                        arc_deg=_arc_deg(comp_centres[inliers], axis, cen),
                    )
                )
                if inliers.all():  # whole patch consumed; no second bore in it
                    break

    planes.sort(key=lambda p: p.area, reverse=True)
    holes.sort(key=lambda h: h.raw_diameter, reverse=True)
    bmin, bmax = verts.min(axis=0), verts.max(axis=0)
    return FeatureSet(
        planes=planes,
        holes=holes,
        bbox_min=_xyz(bmin),
        bbox_max=_xyz(bmax),
        triangles=len(faces),
    )
