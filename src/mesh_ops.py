"""
Mesh operations module for STL/OBJ processing.
Provides functions for loading and saving meshes, boolean difference/union using
trimesh, primitive generation via Open3D, and mesh validity reporting.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:  # imported for static analysis only, never at runtime here
    import open3d as o3d

    type TriangleMesh = "o3d.geometry.TriangleMesh"


class MeshLike(Protocol):
    """Minimal structural contract consumed by the boolean helpers.

    Anything exposing ``.vertices`` / ``.triangles`` arrays (an Open3D
    TriangleMesh, or a lightweight adapter) is a valid boolean operand.
    """

    vertices: Any
    triangles: Any

# Heavy third-party imports are loaded lazily via the _load_* helpers below so
# importing this module stays cheap and optional dependencies stay optional.


def _load_open3d() -> ModuleType:
    try:
        import open3d as o3d
        return o3d
    except ImportError as e:
        raise RuntimeError("Open3D is required for mesh operations") from e

# Helper to ensure vertex normals are present
def ensure_normals(mesh: TriangleMesh) -> TriangleMesh:
    """Compute vertex normals if they are missing.
    Open3D STL export requires normals; this function makes sure they exist.
    Returns the same mesh (modified in-place).
    """
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh

def _load_trimesh() -> ModuleType:
    try:
        import trimesh
        return trimesh
    except ImportError as e:
        raise RuntimeError("trimesh is required for mesh operations") from e

def _load_mapbox_earcut() -> ModuleType:
    try:
        import mapbox_earcut
        return mapbox_earcut
    except ImportError as e:
        raise RuntimeError("mapbox-earcut is required to fill mesh holes") from e

def _as_volume(
    trimesh_module: ModuleType, vertices: np.ndarray, faces: np.ndarray
) -> Any:
    """Build a trimesh mesh with consistent outward winding so it qualifies as a volume.

    Boolean engines (manifold3d) reject operands that are not watertight volumes; this
    welds duplicate vertices and repairs winding, raising a clear error if still unusable.
    """
    tri = trimesh_module.Trimesh(vertices=vertices, faces=faces, process=False)
    tri.merge_vertices()
    trimesh_module.repair.fix_normals(tri)
    if not tri.is_volume:
        raise ValueError(
            "Boolean operands must be watertight volumes; a mesh could not be repaired "
            f"(watertight={tri.is_watertight}, winding_consistent={tri.is_winding_consistent})."
        )
    return tri

# Budgets for untrusted files handed to load_mesh; see its docstring.
_MAX_MESH_FILE_BYTES = 1024 * 1024 * 1024   # 1 GiB on disk (binary STL is ~50 B/tri)
_MAX_MESH_VERTICES = 5_000_000              # generous vs. real scans (~1e6), bounds DoS
_MAX_MESH_TRIANGLES = 10_000_000

def load_mesh(file_path: str) -> TriangleMesh:
    """Load a mesh from STL or OBJ using Open3D.

    Returns an Open3D TriangleMesh object. The file is untrusted input, so it
    is bounded twice: on disk before parsing (a multi-gigabyte file would be
    swallowed whole by the native reader) and on vertex/triangle counts after
    (huge meshes wedge rendering and boolean ops in C++ code that cannot be
    interrupted). Both limits sit far above any realistic scan.
    """
    try:
        size = Path(file_path).stat().st_size
    except OSError as e:
        raise ValueError(f"Failed to load mesh from {file_path}: {e}") from e
    if size > _MAX_MESH_FILE_BYTES:
        raise ValueError(
            f"Mesh file is too large ({size / 2**20:.0f} MiB; limit "
            f"{_MAX_MESH_FILE_BYTES // 2**20} MiB)"
        )
    o3d = _load_open3d()
    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertices():
        raise ValueError(f"Failed to load mesh from {file_path}")
    n_verts, n_tris = len(mesh.vertices), len(mesh.triangles)
    if n_verts > _MAX_MESH_VERTICES or n_tris > _MAX_MESH_TRIANGLES:
        raise ValueError(
            f"Mesh is too large ({n_verts:,} vertices / {n_tris:,} triangles; limits "
            f"{_MAX_MESH_VERTICES:,} / {_MAX_MESH_TRIANGLES:,})"
        )
    return mesh

def save_mesh(mesh: TriangleMesh, file_path: str) -> None:
    """Save a TriangleMesh to STL/OBJ/3MF.
    The output format is inferred from the extension.
    """
    o3d = _load_open3d()
    # Compute vertex normals unconditionally before saving (Open3D requires them)
    mesh.compute_vertex_normals()
    ext = Path(file_path).suffix.lower()
    if ext == ".stl":
        o3d.io.write_triangle_mesh(file_path, mesh)
    elif ext == ".obj":
        # Open3D can write OBJ directly; ensure normals are present as well
        mesh.compute_vertex_normals()  # redundant but safe
        o3d.io.write_triangle_mesh(file_path, mesh)
    elif ext == ".3mf":
        # Use trimesh fallback for 3MF export
        try:
            trimesh = _load_trimesh()
            tri = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles)
            )
            tri.export(file_path)
        except Exception as e:
            raise RuntimeError(f"Export to 3MF failed: {e}") from e
    else:
        raise ValueError(f"Unsupported output format: {ext}")

def boolean_difference(mesh_a: MeshLike, mesh_b: MeshLike) -> TriangleMesh:
    """Compute Boolean difference (mesh_a minus mesh_b) using trimesh.
    Returns a new Open3D TriangleMesh.
    """
    trimesh = _load_trimesh()
    tri_a = _as_volume(trimesh, np.asarray(mesh_a.vertices), np.asarray(mesh_a.triangles))
    tri_b = _as_volume(trimesh, np.asarray(mesh_b.vertices), np.asarray(mesh_b.triangles))
    result = tri_a.difference(tri_b)
    o3d = _load_open3d()
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(result.vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(result.faces)
    return new_mesh

def boolean_union(mesh_a: MeshLike, mesh_b: MeshLike) -> TriangleMesh:
    """Compute Boolean union of two meshes using trimesh.
    Returns a new Open3D TriangleMesh.
    """
    trimesh = _load_trimesh()
    tri_a = _as_volume(trimesh, np.asarray(mesh_a.vertices), np.asarray(mesh_a.triangles))
    tri_b = _as_volume(trimesh, np.asarray(mesh_b.vertices), np.asarray(mesh_b.triangles))
    result = tri_a.union(tri_b)
    o3d = _load_open3d()
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(result.vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(result.faces)
    return new_mesh

def create_primitive(shape: str, **kwargs: float) -> TriangleMesh:
    """Generate a primitive shape mesh using Open3D.

    Args:
        shape: Primitive type - 'cube', 'sphere', 'cylinder', or 'cone'.
        **kwargs: Shape parameters (size, radius, height).

    Returns:
        TriangleMesh with the requested primitive geometry.

    Raises:
        ValueError: If an unsupported shape is specified.

    Example:
        >>> cube = create_primitive("cube", size=2.0)
        >>> sphere = create_primitive("sphere", radius=1.0)
    """
    o3d = _load_open3d()
    if shape == "cube":
        size = kwargs.get("size", 1.0)
        mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
    elif shape == "sphere":
        radius = kwargs.get("radius", 1.0)
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    elif shape == "cylinder":
        radius = kwargs.get("radius", 1.0)
        height = kwargs.get("height", 2.0)
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
    elif shape == "cone":
        radius = kwargs.get("radius", 1.0)
        height = kwargs.get("height", 2.0)
        mesh = o3d.geometry.TriangleMesh.create_cone(radius=radius, height=height)
    else:
        raise ValueError(f"Unsupported primitive shape: {shape}")
    return mesh


def primitive_kwargs(
    shape: str, size: float = 1.0, radius: float = 1.0, height: float = 2.0
) -> dict[str, float]:
    """Map a shape + user parameters to :func:`create_primitive` keyword arguments.

    Only the dimensions relevant to ``shape`` are returned (cube->size; sphere->radius;
    cylinder/cone->radius+height). Raises ValueError for an unsupported shape or a
    non-positive dimension.
    """
    if shape == "cube":
        params = {"size": size}
    elif shape == "sphere":
        params = {"radius": radius}
    elif shape in ("cylinder", "cone"):
        params = {"radius": radius, "height": height}
    else:
        raise ValueError(f"Unsupported primitive shape: {shape}")
    if any(value <= 0 for value in params.values()):
        raise ValueError("Primitive dimensions must be positive")
    return params

def cut_mesh_at_z(
    mesh: TriangleMesh, z_height: float
) -> tuple[TriangleMesh | None, TriangleMesh | None]:
    """Cut a mesh horizontally at the given Z height.

    Watertight volumes are cut by boolean-intersecting them with oversized
    boxes (manifold3d engine, the same path as the Boolean menu). Those caps
    are exact and, unlike trimesh's slicer, free of zero-area sliver faces
    when the cross-section has collinear cut vertices (washers, tubes and
    prisms whose walls are diagonally triangulated, toroids, ...). Meshes
    that are not watertight volumes fall back to trimesh
    ``slice_mesh_plane``, which keeps working on open surfaces.
    """
    import open3d as o3d
    import trimesh
    from trimesh import intersections

    tri_mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices).copy(),
        faces=np.asarray(mesh.triangles).copy()
    )

    z_coords = np.asarray(tri_mesh.vertices)[:, 2]

    all_below = np.all(z_coords <= z_height)
    all_above = np.all(z_coords >= z_height)

    if all_below:
        return (mesh, None)
    if all_above:
        return (None, mesh)

    def to_open3d(verts: np.ndarray, faces: np.ndarray) -> TriangleMesh | None:
        if verts is None or len(verts) == 0:
            return None
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        ensure_normals(mesh)
        return mesh

    def from_trimesh(tri_geom: Any | None) -> TriangleMesh | None:
        """Convert a boolean/slice half to Open3D with consistent outward winding."""
        if tri_geom is None or len(np.asarray(tri_geom.faces)) == 0:
            return None
        with contextlib.suppress(Exception):  # best-effort winding repair
            trimesh.repair.fix_normals(tri_geom)
        return to_open3d(np.asarray(tri_geom.vertices).copy(), np.asarray(tri_geom.faces).copy())

    # Primary path: robust boolean against half-space boxes.
    try:
        vol = _as_volume(trimesh, np.asarray(tri_mesh.vertices), np.asarray(tri_mesh.faces))
        lo = vol.vertices.min(axis=0)
        hi = vol.vertices.max(axis=0)
        big = 10.0 * max(float((hi - lo).max()), 1.0)
        cx, cy = float((lo[0] + hi[0]) / 2.0), float((lo[1] + hi[1]) / 2.0)
        below = trimesh.creation.box(extents=[big, big, z_height - (float(lo[2]) - big)])
        below.apply_translation([cx, cy, (float(lo[2]) - big + z_height) / 2.0])
        above = trimesh.creation.box(extents=[big, big, (float(hi[2]) + big) - z_height])
        above.apply_translation([cx, cy, (z_height + float(hi[2]) + big) / 2.0])
        bottom = from_trimesh(vol.intersection(below))
        top = from_trimesh(vol.intersection(above))
        if bottom is not None or top is not None:
            return (bottom, top)
    except Exception:  # boolean engine unusable (e.g. not a watertight volume)
        pass

    # Fallback: trimesh slicer with capped halves; works on open surfaces but
    # its capping can leave zero-area sliver faces on collinear cross-sections.
    top_tri = intersections.slice_mesh_plane(tri_mesh, [0, 0, 1], [0, 0, z_height], None, True)
    bottom_tri = intersections.slice_mesh_plane(tri_mesh, [0, 0, -1], [0, 0, z_height], None, True)

    return (from_trimesh(bottom_tri), from_trimesh(top_tri))


def is_watertight(mesh: Any) -> bool:
    """Fast watertightness test for any mesh-like object.

    Uses trimesh's sort-based edge pairing; linear-ish and safe for scans with
    hundreds of thousands of facets (Open3D's equivalent stalls on those).
    """
    verts = np.asarray(mesh.vertices, dtype=float)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    if len(verts) == 0 or len(tris) == 0:
        return False
    tm = _load_trimesh().Trimesh(vertices=verts, faces=tris, process=False)
    tm.merge_vertices()
    return bool(tm.is_watertight)


@dataclass(frozen=True)
class MeshHealth:
    """Validity report for a triangle mesh; see :func:`mesh_health`."""

    watertight: bool
    edge_manifold: bool
    vertex_manifold: bool
    self_intersecting: bool
    self_intersection_pairs: int
    self_intersection_checked: bool
    winding_consistent: bool
    components: int
    degenerate_faces: int
    duplicate_faces: int
    volume: float

    @property
    def ok(self) -> bool:
        return (
            self.watertight
            and self.edge_manifold
            and self.vertex_manifold
            and not self.self_intersecting
            and self.winding_consistent
            and self.degenerate_faces == 0
            and self.duplicate_faces == 0
            and self.volume > 0.0
        )

    @property
    def problems(self) -> list[str]:
        out: list[str] = []
        if not self.watertight:
            out.append("Not watertight: the surface has holes or open edges.")
        if not self.edge_manifold:
            out.append("Non-manifold edges: edges shared by more than two faces.")
        if not self.vertex_manifold:
            out.append("Non-manifold vertices.")
        if self.self_intersecting:
            out.append(
                f"Self-intersections: {self.self_intersection_pairs} "
                "triangle pair(s) cut through each other."
            )
        if not self.winding_consistent:
            out.append(
                "Inconsistent face winding: neighbours disagree about which side is outside."
            )
        if self.degenerate_faces:
            out.append(f"{self.degenerate_faces} degenerate (zero-area) face(s).")
        if self.duplicate_faces:
            out.append(f"{self.duplicate_faces} duplicated face(s).")
        if self.volume <= 0.0:
            out.append("Non-positive volume: inverted or flat mesh.")
        if self.components > 1:
            out.append(f"{self.components} disconnected pieces.")
        return out


def mesh_health(mesh: Any, check_self_intersections: bool | None = None) -> MeshHealth:
    """Validate any mesh-like object (``vertices``/``triangles`` attributes).

    Combines Open3D (watertight / manifold / self-intersection), trimesh (winding
    consistency, connected components) and numpy (degenerate faces, geometrically
    duplicate faces, signed volume) into a single report. Open3D's own
    ``get_volume()`` refuses non-watertight input, so signed volume is computed
    here via the divergence theorem and works for any mesh.

    The self-intersection test is quadratic in triangle count (minutes on a
    500k-facet scan), so it is skipped by default above 100k triangles; pass
    ``check_self_intersections=True`` to force it or ``False`` to skip.
    Skipped runs report ``self_intersection_checked=False`` and never fail on it.
    """
    verts = np.asarray(mesh.vertices, dtype=float)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    if len(verts) == 0 or len(tris) == 0:
        return MeshHealth(False, False, False, False, 0, False, False, 0, 0, 0, 0.0)

    trimesh_mod = _load_trimesh()
    tm = trimesh_mod.Trimesh(vertices=verts, faces=tris, process=False)
    tm.merge_vertices()
    # trimesh for watertightness: sort-based and linear. Open3D's own
    # is_watertight() takes over 30s on a 600k-triangle scan.
    watertight = bool(tm.is_watertight)
    winding_consistent = bool(tm.is_winding_consistent)
    components = int(tm.body_count)

    o3d = _load_open3d()
    tri_mesh = o3d.geometry.TriangleMesh()
    tri_mesh.vertices = o3d.utility.Vector3dVector(verts)
    tri_mesh.triangles = o3d.utility.Vector3iVector(tris)
    # STL/OBJ files often store per-face vertices; weld first, otherwise the
    # Open3D topology checks see unshared edges everywhere. The numpy checks
    # below still run on the raw arrays so duplicates remain visible.
    tri_mesh.remove_duplicated_vertices()
    tri_mesh.remove_degenerate_triangles()
    # get_non_manifold_edges(): edges shared by >2 faces (unlike
    # is_edge_manifold, it does not conflate these with open boundary edges,
    # which the watertight check already covers).
    non_manifold_edges = int(len(tri_mesh.get_non_manifold_edges()))
    vertex_manifold = bool(tri_mesh.is_vertex_manifold())
    if check_self_intersections is None:
        check_self_intersections = len(tris) <= 100_000
    self_intersection_pairs = (
        int(len(tri_mesh.get_self_intersecting_triangles())) if check_self_intersections else 0
    )

    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    degenerate_faces = int(np.sum(np.linalg.norm(cross, axis=1) < 1e-12))
    # Geometric duplicates: unwelded STL vertices make index-based comparison useless,
    # so compare rounded corner coordinates, made independent of vertex order by
    # sorting the three corners of each triangle (not the flattened xyz values).
    corners = np.round(np.sort(np.stack([v0, v1, v2], axis=1), axis=1), 9).reshape(len(tris), 9)
    _, counts = np.unique(corners, axis=0, return_counts=True)
    repeated = counts[counts > 1]
    duplicate_faces = int(repeated.sum() - len(repeated))
    volume = float(np.einsum("ij,ij->i", v0, cross).sum() / 6.0)

    return MeshHealth(
        watertight=watertight,
        edge_manifold=non_manifold_edges == 0,
        vertex_manifold=vertex_manifold,
        self_intersecting=self_intersection_pairs > 0,
        self_intersection_pairs=self_intersection_pairs,
        self_intersection_checked=check_self_intersections,
        winding_consistent=winding_consistent,
        components=components,
        degenerate_faces=degenerate_faces,
        duplicate_faces=duplicate_faces,
        volume=volume,
    )


# ---------------------------------------------------------------------- #
# Repair
# ---------------------------------------------------------------------- #

def _boundary_loops(verts: np.ndarray, faces: np.ndarray) -> list[list[int]]:
    """Closed vertex loops around the open edges of a welded face array.

    Boundary edges - those used by exactly one face - form an undirected graph in
    which every vertex has degree 2, except where several holes touch each other
    at a single corner (a common scan/boolean artifact; beanpod STLs have two
    holes sharing one vertex). With no branch vertices each component is already
    one closed loop. Otherwise the graph is split into paths between branch
    vertices and greedily rejoined by directional continuity, so touching holes
    are capped separately instead of as one tangled ring. Loops that cannot be
    closed (e.g. dangling edges) are dropped rather than producing bad caps.

    ``faces`` must already have welded (shared) vertex indices.
    """
    if len(faces) == 0:
        return []
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = uniq[counts == 1]
    if len(boundary) == 0:
        return []

    adjacency: dict[int, list[int]] = {}
    for a, b in boundary:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))

    def edge_key(x: int, y: int) -> tuple[int, int]:
        return (x, y) if x < y else (y, x)

    remaining = {edge_key(int(a), int(b)) for a, b in boundary}
    branch = {v for v, nbrs in adjacency.items() if len(nbrs) != 2}

    def walk(start: int, first: int) -> tuple[list[int], bool]:
        """Follow unused edges from ``start``, stopping at any branch vertex.

        Returns the visited path and whether it closed back onto ``start``; a
        closed walk is already a complete loop, an open one runs between two
        branch vertices (possibly the same pair) and needs rejoining.
        """
        path = [start]
        prev, cur = start, first
        while True:
            remaining.discard(edge_key(prev, cur))
            path.append(cur)
            if cur == start or cur in branch:
                return path, cur == start
            nxts = [n for n in adjacency[cur] if edge_key(cur, n) in remaining and n != prev]
            if not nxts:  # dangling chain end; cannot be capped cleanly
                return path, False
            prev, cur = cur, nxts[0]

    loops: list[list[int]] = []
    if not branch:
        # Every component is a simple cycle: walk it closed.
        for start in adjacency:
            if not any(edge_key(start, n) in remaining for n in adjacency[start]):
                continue
            path, closed = walk(start, adjacency[start][0])
            if closed and len(path) > 2:
                loops.append(path[:-1])  # drop the repeated start vertex
        return loops

    # Split into maximal paths between branch vertices. A walk that closes back
    # onto its own branch vertex (two holes touching at one corner, as in the
    # beanpod samples) is already a complete loop; open runs are joined below.
    paths: list[list[int]] = []
    for b in sorted(branch):
        for nb in adjacency[b]:
            if edge_key(b, nb) not in remaining:
                continue
            path, closed = walk(b, nb)
            if closed and len(path) > 2:
                loops.append(path[:-1])
            elif not closed and len(path) >= 2:
                paths.append(path)

    # ...then rejoin them into closed loops. At each branch vertex the next path
    # is chosen to best preserve travel direction (max dot product of edge vectors).
    used = [False] * len(paths)
    for i, path in enumerate(paths):
        if used[i]:
            continue
        used[i] = True
        loop = list(path)
        while not (len(loop) > 2 and loop[0] == loop[-1]):
            end = loop[-1]
            best: tuple[float, int, list[int]] | None = None
            for j, other in enumerate(paths):
                if used[j]:
                    continue
                if other[0] == end:
                    continuation = other[1:]
                elif other[-1] == end:
                    continuation = other[-2::-1]
                else:
                    continue
                if not continuation or len(loop) < 2:
                    continue
                d_in = verts[end] - verts[loop[-2]]
                d_out = verts[continuation[0]] - verts[end]
                n_in, n_out = float(np.linalg.norm(d_in)), float(np.linalg.norm(d_out))
                score = 0.0
                if n_in > 0 and n_out > 0:
                    score = float(np.dot(d_in, d_out) / (n_in * n_out))
                if best is None or score > best[0]:
                    best = (score, j, continuation)
            if best is None:
                break
            _, j, continuation = best
            used[j] = True
            loop.extend(continuation)
        if len(loop) > 2 and loop[0] == loop[-1]:
            loops.append(loop[:-1])
    return loops


def _fill_loop(verts: np.ndarray, loop: list[int]) -> np.ndarray:
    """Triangulate one boundary loop as a flat cap; returns new faces (may be empty).

    The loop is projected onto its best-fit plane (SVD) and ear-clipped with
    mapbox-earcut. Loops that project to a degenerate polygon yield no faces.
    """
    if len(loop) < 3:
        return np.empty((0, 3), dtype=np.int64)
    points = verts[loop]
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid)
    basis = vt[:2].T  # 3x2: projects the 3-D loop points onto the plane's axes
    uv = (points - centroid) @ basis
    earcut = _load_mapbox_earcut()
    ring_ends = np.array([len(loop)], dtype=np.uint32)  # single outer ring, no holes
    triangles = np.asarray(earcut.triangulate_float64(np.ascontiguousarray(uv), ring_ends))
    if len(triangles) == 0:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(loop, dtype=np.int64)[triangles.reshape(-1, 3)]


def repair_mesh(mesh: MeshLike) -> TriangleMesh:
    """Repair a mesh so it becomes a watertight printable volume.

    Welds duplicate vertices, drops degenerate (zero-area) and duplicated faces,
    caps every boundary hole with an ear-clipped flat patch, and normalises face
    winding so the enclosed volume is positive. Existing triangles are never
    moved - repair only removes unusable faces and adds cap faces - so a valid
    mesh passes through geometrically unchanged.

    Returns a new Open3D TriangleMesh; the input is not modified. Self-
    intersections between pre-existing triangles cannot be fixed without moving
    geometry, so they are left for manual editing (see :func:`mesh_health`).
    """
    o3d = _load_open3d()
    trimesh_mod = _load_trimesh()

    verts = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(mesh.triangles, dtype=np.int64).reshape(-1, 3)

    if len(verts) and len(faces):
        # Weld duplicate vertices (STL files store per-face copies) so topology
        # analysis sees shared edges. Same path as mesh_health/is_watertight.
        welded = trimesh_mod.Trimesh(vertices=verts, faces=faces, process=False)
        welded.merge_vertices()
        verts = np.asarray(welded.vertices, dtype=np.float64).copy()
        faces = np.asarray(welded.faces, dtype=np.int64).copy()

        # Drop zero-area faces (they poison winding repair and boolean engines).
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        areas_ok = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) >= 1e-12
        faces = faces[areas_ok]
        # Drop duplicated faces: after welding they share vertex indices, so keep
        # one of each. The surface stays closed because the surviving copy remains.
        if len(faces):
            _, first_of_group = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
            faces = faces[np.sort(first_of_group)]

        caps = [_fill_loop(verts, lp) for lp in _boundary_loops(verts, faces)]
        cap_faces = [c for c in caps if len(c)]
        if cap_faces:
            faces = np.vstack([faces, *cap_faces])

        # Consistent outward winding; then a divergence-theorem volume sign check
        # so an inverted shell is flipped even when the repair helpers abstain.
        if len(faces):
            result = trimesh_mod.Trimesh(vertices=verts, faces=faces, process=False)
            trimesh_mod.repair.fix_normals(result)
            faces = np.asarray(result.faces, dtype=np.int64).copy()
            v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
            signed_volume = float(np.einsum("ij,ij->i", v0, np.cross(v1 - v0, v2 - v0)).sum() / 6.0)
            if signed_volume < 0.0:
                faces = faces[:, ::-1]

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(verts)
    out.triangles = o3d.utility.Vector3iVector(faces.astype(np.uint32))
    return out
