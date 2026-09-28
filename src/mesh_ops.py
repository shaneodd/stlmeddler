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
    import trimesh

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

def load_mesh(file_path: str) -> TriangleMesh:
    """Load a mesh from STL or OBJ using Open3D.
    Returns an Open3D TriangleMesh object.
    """
    o3d = _load_open3d()
    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertices():
        raise ValueError(f"Failed to load mesh from {file_path}")
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

    Uses trimesh slice_mesh_plane which properly creates capped surfaces
    at the cut plane. The caps are preserved by NOT filtering vertices
    based on Z position after slicing.
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

    # Get bottom half (keep vertices with z < z_height)
    top_tri = intersections.slice_mesh_plane(tri_mesh, [0, 0, 1], [0, 0, z_height], None, True)

    # Get top half (keep vertices with z > z_height)
    bottom_tri = intersections.slice_mesh_plane(tri_mesh, [0, 0, -1], [0, 0, z_height], None, True)

    def process_half(tri_geom: trimesh.Trimesh | None) -> tuple[np.ndarray, np.ndarray]:
        """Convert a sliced half to Open3D arrays with consistent outward winding.

        ``slice_mesh_plane`` returns capped halves; normalise their winding so each piece
        is a watertight volume (required by downstream booleans and slicers).
        """
        if tri_geom is None or len(np.asarray(tri_geom.faces)) == 0:
            return np.array([]), np.array([])

        with contextlib.suppress(Exception):  # best-effort winding repair
            trimesh.repair.fix_normals(tri_geom)

        verts = np.asarray(tri_geom.vertices).copy()
        faces = np.asarray(tri_geom.faces).copy()
        return verts, faces

    b_verts, b_faces = process_half(bottom_tri)
    t_verts, t_faces = process_half(top_tri)

    def to_open3d(verts: np.ndarray, faces: np.ndarray) -> TriangleMesh | None:
        if verts is None or len(verts) == 0:
            return None
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        ensure_normals(mesh)
        return mesh

    return (to_open3d(b_verts, b_faces), to_open3d(t_verts, t_faces))


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
