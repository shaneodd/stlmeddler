"""
Pure-Python scene model backing :class:`src.gui.viewer.MeshViewer`.

Holds an ordered list of independent objects (geometry + per-object transform +
colour) with no dependency on Qt or OpenGL, so the object-management and
transform logic can be unit-tested headlessly.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypedDict

import numpy as np

from .picking import ray_aabb, ray_triangles_nearest

if TYPE_CHECKING:  # imported for static analysis only, never at runtime here
    import open3d as o3d

    type TriangleMesh = "o3d.geometry.TriangleMesh"


class _SceneObjectRequired(TypedDict):
    verts: np.ndarray
    tris: np.ndarray
    name: str
    rot_x: float
    rot_y: float
    rot_z: float
    trans_x: float
    trans_y: float
    trans_z: float
    color: tuple[float, ...]


class SceneObject(_SceneObjectRequired, total=False):
    # Optional raw 4x4 matrix that overrides the Euler+translation model when set.
    matrix_override: np.ndarray
    # Per-axis scale factors applied before rotation (M = T * R * S). Default 1.0.
    scale_x: float
    scale_y: float
    scale_z: float
    # Optional expanded (positions, normals) float32 arrays cached by the viewer
    # for fast vertex-array draws.
    render_arrays: tuple[np.ndarray, np.ndarray]


def rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Return a 3x3 rotation matrix combining Rx, Ry, Rz (applied X then Y then Z)."""
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    R_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    R_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return R_z @ R_y @ R_x


def model_matrix(obj: SceneObject) -> np.ndarray:
    """Build a 4x4 model matrix (T * Rz*Ry*Rx * S) for an object dict."""
    M = np.eye(4)
    S = np.diag([
        obj.get("scale_x", 1.0), obj.get("scale_y", 1.0), obj.get("scale_z", 1.0),
    ])
    M[:3, :3] = rotation_matrix(obj["rot_x"], obj["rot_y"], obj["rot_z"]) @ S
    M[0, 3] = obj["trans_x"]
    M[1, 3] = obj["trans_y"]
    M[2, 3] = obj["trans_z"]
    return M


def rotation_from_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest-arc 3x3 rotation mapping unit vector ``a`` onto unit vector ``b``."""
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-12:
        if c > 0.0:
            return np.eye(3)
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if float(np.linalg.norm(axis)) < 1e-6:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * (K @ K)  # exact Rodrigues at 180 degrees
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + (K @ K) * ((1.0 - c) / (s * s))


def matrix_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Extract (rx, ry, rz) degrees from a 3x3 rotation matrix (Z*Y*X order)."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0
    return float(np.degrees(rx)), float(np.degrees(ry)), float(np.degrees(rz))


class SceneModel:
    """Ordered collection of independently transformable mesh objects."""

    def __init__(self) -> None:
        self.objects: list[SceneObject] = []
        self.active_index: int = -1

    # ------------------------------------------------------------------ #
    # Object management
    # ------------------------------------------------------------------ #
    def add_object(
        self, mesh: TriangleMesh, name: str | None = None, auto_align: bool = True
    ) -> int:
        """Add a mesh as an independent object and make it active. Returns index."""
        if not hasattr(mesh, "vertices") or not hasattr(mesh, "triangles"):
            raise TypeError("Expected a mesh object with vertices and triangles attributes")
        verts = np.asarray(mesh.vertices).copy()
        tris = np.asarray(mesh.triangles).copy()

        trans_x = trans_y = trans_z = 0.0
        if auto_align and verts.size:
            centroid = verts.mean(axis=0)
            min_z = float(verts[:, 2].min())
            trans_x, trans_y = -float(centroid[0]), -float(centroid[1])
            trans_z = -min_z

        obj: SceneObject = {
            "verts": verts,
            "tris": tris,
            "name": name or f"object_{len(self.objects)}",
            "rot_x": 0.0, "rot_y": 0.0, "rot_z": 0.0,
            "trans_x": trans_x, "trans_y": trans_y, "trans_z": trans_z,
            "scale_x": 1.0, "scale_y": 1.0, "scale_z": 1.0,
            "color": tuple(float(c) for c in np.random.uniform(0.45, 1.0, 3)),
        }
        self.objects.append(obj)
        self.set_active(len(self.objects) - 1)
        return len(self.objects) - 1

    def remove_object(self, index: int) -> None:
        if not (0 <= index < len(self.objects)):
            return
        self.objects.pop(index)
        if not self.objects:
            self.active_index = -1
        elif index <= self.active_index:
            self.active_index = max(0, min(self.active_index - 1, len(self.objects) - 1))

    def remove_by_name(self, name: str) -> bool:
        for i, obj in enumerate(self.objects):
            if obj["name"] == name:
                self.remove_object(i)
                return True
        return False

    def get_objects(self) -> list[SceneObject]:
        return self.objects

    def get_index(self, name: str) -> int:
        for i, obj in enumerate(self.objects):
            if obj["name"] == name:
                return i
        return -1

    def set_active(self, index: int) -> None:
        if not (-1 <= index < len(self.objects)):
            return
        self.active_index = index

    # ------------------------------------------------------------------ #
    # Transforms (default to active object)
    # ------------------------------------------------------------------ #
    def _resolve(self, index: int | None) -> SceneObject | None:
        if index is None:
            index = self.active_index
        if not (0 <= index < len(self.objects)):
            return None
        return self.objects[index]

    def apply_transform(self, matrix: np.ndarray, index: int | None = None) -> None:
        obj = self._resolve(index)
        if obj is None:
            return
        new_M = np.asarray(matrix, dtype=float) @ model_matrix(obj)
        obj["matrix_override"] = new_M

    def translate(self, vec: Sequence[float], index: int | None = None) -> None:
        obj = self._resolve(index)
        if obj is None:
            return
        obj["trans_x"] += float(vec[0])
        obj["trans_y"] += float(vec[1])
        obj["trans_z"] += float(vec[2])

    def rotate(self, axis: Sequence[float], angle_rad: float, index: int | None = None) -> None:
        obj = self._resolve(index)
        if obj is None:
            return
        axis_arr = np.asarray(axis, dtype=float)
        n = np.linalg.norm(axis_arr)
        if n == 0:
            return
        axis_arr = axis_arr / n
        a = float(angle_rad)
        K = np.array([[0, -axis_arr[2], axis_arr[1]],
                      [axis_arr[2], 0, -axis_arr[0]],
                      [-axis_arr[1], axis_arr[0], 0]])
        R_inc = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)
        new_R = R_inc @ rotation_matrix(obj["rot_x"], obj["rot_y"], obj["rot_z"])
        obj["rot_x"], obj["rot_y"], obj["rot_z"] = matrix_to_euler(new_R)

    def set_rotation(
        self,
        rx: float | None = None,
        ry: float | None = None,
        rz: float | None = None,
        index: int | None = None,
    ) -> None:
        obj = self._resolve(index)
        if obj is None:
            return
        if rx is not None:
            obj["rot_x"] = float(rx)
        if ry is not None:
            obj["rot_y"] = float(ry)
        if rz is not None:
            obj["rot_z"] = float(rz)

    def set_translation(
        self,
        tx: float | None = None,
        ty: float | None = None,
        tz: float | None = None,
        index: int | None = None,
    ) -> None:
        obj = self._resolve(index)
        if obj is None:
            return
        if tx is not None:
            obj["trans_x"] = float(tx)
        if ty is not None:
            obj["trans_y"] = float(ty)
        if tz is not None:
            obj["trans_z"] = float(tz)

    def set_scale(
        self,
        sx: float | None = None,
        sy: float | None = None,
        sz: float | None = None,
        index: int | None = None,
    ) -> None:
        """Set absolute per-axis scale factors (defaults 1.0; non-positive ignored)."""
        obj = self._resolve(index)
        if obj is None:
            return
        for key, val in (("scale_x", sx), ("scale_y", sy), ("scale_z", sz)):
            if val is not None and float(val) > 0.0:
                obj[key] = float(val)

    @staticmethod
    def _local_extent(obj: SceneObject) -> np.ndarray:
        """Unscaled bounding-box extent of the object's base geometry."""
        if obj["verts"].size == 0:
            return np.zeros(3)
        return obj["verts"].max(axis=0) - obj["verts"].min(axis=0)

    def get_dimensions(self, index: int | None = None) -> tuple[float, float, float]:
        """Intrinsic size (width, depth, height) in mm: local AABB extent x scale."""
        obj = self._resolve(index)
        if obj is None:
            return (0.0, 0.0, 0.0)
        extent = self._local_extent(obj)
        s = np.array([
            obj.get("scale_x", 1.0), obj.get("scale_y", 1.0), obj.get("scale_z", 1.0),
        ])
        return tuple(float(d) for d in extent * s)

    def set_dimensions(
        self,
        width: float | None = None,
        depth: float | None = None,
        height: float | None = None,
        index: int | None = None,
    ) -> bool:
        """Scale the object so its intrinsic dimensions match the given mm sizes.

        Each axis sets ``scale = desired / local extent``; axes with a degenerate
        (flat) extent or a non-positive target are skipped. Returns True if any
        axis was applied.
        """
        obj = self._resolve(index)
        if obj is None:
            return False
        extent = self._local_extent(obj)
        applied = False
        for key, want, ext in (
            ("scale_x", width, extent[0]),
            ("scale_y", depth, extent[1]),
            ("scale_z", height, extent[2]),
        ):
            if want is None or float(want) <= 0.0 or ext <= 1e-12:
                continue
            obj[key] = float(want) / float(ext)
            applied = True
        return applied

    def get_transform(self, index: int | None = None) -> dict[str, float]:
        obj = self._resolve(index)
        if obj is None:
            return {"rot_x": 0.0, "rot_y": 0.0, "rot_z": 0.0,
                    "trans_x": 0.0, "trans_y": 0.0, "trans_z": 0.0,
                    "scale_x": 1.0, "scale_y": 1.0, "scale_z": 1.0}
        return {
            "rot_x": obj["rot_x"], "rot_y": obj["rot_y"], "rot_z": obj["rot_z"],
            "trans_x": obj["trans_x"], "trans_y": obj["trans_y"], "trans_z": obj["trans_z"],
            "scale_x": obj.get("scale_x", 1.0), "scale_y": obj.get("scale_y", 1.0),
            "scale_z": obj.get("scale_z", 1.0),
        }

    def align_active_to_plate(self, index: int | None = None) -> None:
        """Drop an object onto Z=0 keeping its current rotation."""
        obj = self._resolve(index)
        if obj is None or obj["verts"].size == 0:
            return
        R = rotation_matrix(obj["rot_x"], obj["rot_y"], obj["rot_z"])
        t = np.array([obj["trans_x"], obj["trans_y"], obj["trans_z"]])
        world_z = (R @ obj["verts"].T).T[:, 2] + t[2]
        obj["trans_z"] -= float(world_z.min())

    def center_active_on_plate(self, index: int | None = None) -> bool:
        """Centre the object's bounding box over the plate origin, resting on Z=0.

        Only the translation changes: the rotation is kept, the XY bounding-box
        centre is moved to (0, 0), and the lowest point is set down on the plate.
        Any ``matrix_override`` is decomposed back to euler + translation so the
        transform spin boxes stay in sync. Returns False if there is nothing to do.
        """
        idx = self.active_index if index is None else index
        if not (0 <= idx < len(self.objects)):
            return False
        obj = self.objects[idx]
        if obj["verts"].size == 0:
            return False
        world = self.transformed_vertices(idx)
        bmin = world.min(axis=0)
        bmax = world.max(axis=0)
        center = (bmin + bmax) / 2.0
        dx, dy, dz = -float(center[0]), -float(center[1]), -float(bmin[2])
        if "matrix_override" in obj:
            M = np.asarray(obj["matrix_override"], dtype=float)
            M = M.copy()
            M[0, 3] += dx
            M[1, 3] += dy
            M[2, 3] += dz
            obj["rot_x"], obj["rot_y"], obj["rot_z"] = matrix_to_euler(M[:3, :3])
            obj["trans_x"], obj["trans_y"], obj["trans_z"] = (
                float(M[0, 3]), float(M[1, 3]), float(M[2, 3]),
            )
            obj.pop("matrix_override")
        else:
            obj["trans_x"] += dx
            obj["trans_y"] += dy
            obj["trans_z"] += dz
        return True

    def align_face_to_plate(
        self,
        tri_index: int,
        index: int | None = None,
        view_dir: Sequence[float] | None = None,
    ) -> bool:
        """Rotate the object so the given face rests flat on the build plate (Z=0).

        The face's outward normal is turned to point straight down (-Z), its longest
        edge is yaw-snapped to the nearest horizontal axis, and the object is dropped
        so its lowest point sits on Z=0. ``view_dir`` (the picking ray direction)
        disambiguates inside-out winding: the camera-facing side is treated as
        outside. Returns False if there is nothing to align.
        """
        idx = self.active_index if index is None else index
        if not (0 <= idx < len(self.objects)):
            return False
        obj = self.objects[idx]
        if obj["verts"].size == 0 or not (0 <= tri_index < len(obj["tris"])):
            return False
        tri = obj["tris"][tri_index]
        v = obj["verts"]
        n = np.cross(v[tri[1]] - v[tri[0]], v[tri[2]] - v[tri[0]])
        n_len = float(np.linalg.norm(n))
        if n_len < 1e-12:
            return False
        n = n / n_len

        M = np.asarray(obj.get("matrix_override", model_matrix(obj)), dtype=float)
        n_world = M[:3, :3] @ n
        n_world /= max(float(np.linalg.norm(n_world)), 1e-12)
        if view_dir is not None and float(np.dot(n_world, np.asarray(view_dir, dtype=float))) > 0:
            n_world = -n_world  # normal points away from the viewer -> inside-out

        R_a = rotation_from_to(n_world, np.array([0.0, 0.0, -1.0]))
        world = self.transformed_vertices(idx)
        rotated = (R_a @ world.T).T

        # Dominant-edge snap: yaw so the clicked planar region's longest boundary
        # edge hits the nearest horizontal axis. The region merges triangles that
        # share the clicked plane, so quad-face triangulation diagonals are ignored.
        tris = obj["tris"]
        w_tri = world[tris]
        nrm = np.cross(w_tri[:, 1] - w_tri[:, 0], w_tri[:, 2] - w_tri[:, 0])
        ln = np.linalg.norm(nrm, axis=1)
        nrm[ln > 1e-12] /= ln[ln > 1e-12][:, None]
        same_plane = (
            (np.abs(nrm @ n_world) > 1.0 - 1e-6)
            & (np.abs((w_tri[:, 0] - world[tri[0]]) @ n_world) < 1e-6)
        )
        edge_count: dict[tuple[int, int], int] = {}
        for face in tris[same_plane]:
            for a, b in ((0, 1), (1, 2), (2, 0)):
                key = (int(face[a]), int(face[b]))
                key = (min(key), max(key))
                edge_count[key] = edge_count.get(key, 0) + 1

        theta = 0.0
        best_len = 1e-9
        for (i, j), count in edge_count.items():
            if count != 1:  # interior edge (e.g. triangulation diagonal)
                continue
            e = (R_a @ (world[j] - world[i]))[:2]
            length = float(np.linalg.norm(e))
            if length > best_len:
                best_len = length
                phi = float(np.arctan2(e[1], e[0]))
                theta = round(phi / (math.pi / 2.0)) * (math.pi / 2.0) - phi
        c, s = math.cos(theta), math.sin(theta)
        R_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        R_new = R_z @ R_a @ M[:3, :3]
        t_new = R_z @ R_a @ M[:3, 3]
        t_new[2] -= float(rotated[:, 2].min())

        obj["rot_x"], obj["rot_y"], obj["rot_z"] = matrix_to_euler(R_new)
        obj["trans_x"], obj["trans_y"], obj["trans_z"] = (float(t_new[0]),
                                                          float(t_new[1]),
                                                          float(t_new[2]))
        obj.pop("matrix_override", None)
        return True

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    def transformed_vertices(self, index: int) -> np.ndarray:
        obj = self.objects[index]
        M = obj.get("matrix_override", model_matrix(obj))
        h = np.hstack([obj["verts"], np.ones((len(obj["verts"]), 1))])
        return (M @ h.T).T[:, :3]

    def merged_mesh(self) -> TriangleMesh:
        """Combine all objects into a single Open3D mesh with transforms baked in."""
        import open3d as o3d
        all_verts, all_tris = [], []
        offset = 0
        for i in range(len(self.objects)):
            obj = self.objects[i]
            if obj["tris"].size == 0:
                continue
            verts = self.transformed_vertices(i)
            all_verts.append(verts)
            all_tris.append(obj["tris"] + offset)
            offset += len(verts)
        mesh = o3d.geometry.TriangleMesh()
        if all_verts:
            mesh.vertices = o3d.utility.Vector3dVector(np.vstack(all_verts))
            mesh.triangles = o3d.utility.Vector3iVector(np.vstack(all_tris))
            mesh.compute_vertex_normals()
        return mesh

    # ------------------------------------------------------------------ #
    # Selection / picking
    # ------------------------------------------------------------------ #
    def hit_test(
        self, ray_origin: np.ndarray, ray_dir: np.ndarray
    ) -> tuple[int, int] | None:
        """Return ``(object_index, triangle_row)`` of the nearest object a ray hits.

        AABB broad phase then two-sided Moeller-Trumbore per triangle; the closest hit
        across all objects wins. Returns None when nothing is hit.
        """
        hit = self.pick_surface(ray_origin, ray_dir)
        return None if hit is None else (hit[0], hit[1])

    def pick_surface(
        self, ray_origin: np.ndarray, ray_dir: np.ndarray
    ) -> tuple[int, int, np.ndarray] | None:
        """Return ``(object_index, triangle_row, world_point)`` of the nearest hit.

        Same search as :meth:`hit_test` but also reports the exact intersection
        point, used by interactive tools like distance measurement.
        """
        origin = np.asarray(ray_origin, dtype=float)
        direction = np.asarray(ray_dir, dtype=float)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm < 1e-12:
            return None
        direction = direction / norm

        best: tuple[float, int, int] | None = None
        for i in range(len(self.objects)):
            obj = self.objects[i]
            if obj["tris"].size == 0 or obj["verts"].size == 0:
                continue
            verts = self.transformed_vertices(i)
            bmin = verts.min(axis=0)
            bmax = verts.max(axis=0)
            if not ray_aabb(origin, direction, bmin, bmax):
                continue
            hit = ray_triangles_nearest(origin, direction, verts, obj["tris"])
            if hit is not None and (best is None or hit[0] < best[0]):
                best = (hit[0], i, hit[1])
        if best is None:
            return None
        t, obj_index, tri_index = best
        point = origin + t * direction
        return obj_index, tri_index, point

    # ------------------------------------------------------------------ #
    # Operations that mutate the object list
    # ------------------------------------------------------------------ #
    def _unique_name(self, base: str) -> str:
        existing = {o["name"] for o in self.objects}
        if base not in existing:
            return base
        i = 1
        while f"{base}_{i}" in existing:
            i += 1
        return f"{base}_{i}"

    def cut_active_at_z(self, z_height: float) -> list[str]:
        """Cut the active object's world-space geometry at ``z_height``.

        Replaces the original object with up to two new objects named
        ``<base>_bottom`` and ``<base>_top`` (already in world coordinates).
        Returns the list of added object names.
        """
        import open3d as o3d

        from src.mesh_ops import cut_mesh_at_z

        if not (0 <= self.active_index < len(self.objects)):
            return []
        active = self.active_index
        base_name = self.objects[active]["name"]
        world_verts = self.transformed_vertices(active)
        tris = self.objects[active]["tris"]
        work = o3d.geometry.TriangleMesh()
        work.vertices = o3d.utility.Vector3dVector(world_verts)
        work.triangles = o3d.utility.Vector3iVector(tris)

        bottom, top = cut_mesh_at_z(work, z_height)
        self.remove_object(active)

        added: list[str] = []
        if bottom is not None and len(bottom.vertices):
            idx = self.add_object(
                bottom, name=self._unique_name(f"{base_name}_bottom"), auto_align=False
            )
            added.append(self.objects[idx]["name"])
        if top is not None and len(top.vertices):
            idx = self.add_object(top, name=self._unique_name(f"{base_name}_top"), auto_align=False)
            added.append(self.objects[idx]["name"])
        return added

    def boolean_active_with(self, other_index: int, kind: str) -> str | None:
        """Boolean the active object (A) with another loaded object (B).

        Operates in world space so results match what is displayed. Both operand
        objects are consumed and replaced by a single result named after A.
        ``kind`` is "difference" (A - B) or "union". Returns the new object's name,
        or None if the operands are invalid.
        """
        from types import SimpleNamespace

        from src.mesh_ops import boolean_difference, boolean_union

        active = self.active_index
        if not (0 <= active < len(self.objects)):
            return None
        if other_index == active or not (0 <= other_index < len(self.objects)):
            return None

        name_a = self.objects[active]["name"]
        name_b = self.objects[other_index]["name"]

        mesh_a = SimpleNamespace(
            vertices=self.transformed_vertices(active),
            triangles=self.objects[active]["tris"],
        )
        mesh_b = SimpleNamespace(
            vertices=self.transformed_vertices(other_index),
            triangles=self.objects[other_index]["tris"],
        )
        func = boolean_difference if kind == "difference" else boolean_union
        result = func(mesh_a, mesh_b)

        # Remove operands by name (order-independent), then add the combined part.
        self.remove_by_name(name_a)
        self.remove_by_name(name_b)
        idx = self.add_object(result, name=self._unique_name(name_a), auto_align=False)
        return self.objects[idx]["name"]


    # ------------------------------------------------------------------ #
    # Project persistence (.stlproj zip bundle)
    # ------------------------------------------------------------------ #
    def save_project(self, path: str) -> None:
        """Serialise the whole scene to a self-contained ``.stlproj`` zip bundle."""
        objects_meta = []
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, obj in enumerate(self.objects):
                if "matrix_override" in obj:
                    # Bake the override into base geometry; emit an identity transform.
                    verts = self.transformed_vertices(i)
                    rot_x = rot_y = rot_z = 0.0
                    trans_x = trans_y = trans_z = 0.0
                    scale_x = scale_y = scale_z = 1.0
                else:
                    verts = obj["verts"]
                    rot_x, rot_y, rot_z = obj["rot_x"], obj["rot_y"], obj["rot_z"]
                    trans_x, trans_y, trans_z = obj["trans_x"], obj["trans_y"], obj["trans_z"]
                    scale_x = obj.get("scale_x", 1.0)
                    scale_y = obj.get("scale_y", 1.0)
                    scale_z = obj.get("scale_z", 1.0)
                tris = obj["tris"]

                buf = io.BytesIO()
                np.savez(buf, verts=verts, tris=tris)
                arcname = f"parts/{i}.npz"
                zf.writestr(arcname, buf.getvalue())

                entry = {
                    "name": obj["name"],
                    "color": list(obj["color"]),
                    "rot_x": rot_x, "rot_y": rot_y, "rot_z": rot_z,
                    "trans_x": trans_x, "trans_y": trans_y, "trans_z": trans_z,
                    "scale_x": scale_x, "scale_y": scale_y, "scale_z": scale_z,
                    "mesh": arcname,
                }
                objects_meta.append(entry)

            manifest = {
                "version": 1,
                "active_index": self.active_index,
                "objects": objects_meta,
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    @classmethod
    def load_project(cls, path: str) -> SceneModel:
        """Rebuild a :class:`SceneModel` from a ``.stlproj`` zip bundle."""
        model = cls()
        with zipfile.ZipFile(path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
            if manifest.get("version") != 1:
                raise ValueError(f"Unsupported project version: {manifest.get('version')!r}")
            for entry in manifest["objects"]:
                data = np.load(io.BytesIO(zf.read(entry["mesh"])), allow_pickle=False)
                obj: SceneObject = {
                    "verts": data["verts"],
                    "tris": data["tris"],
                    "name": entry["name"],
                    "rot_x": float(entry.get("rot_x", 0.0)),
                    "rot_y": float(entry.get("rot_y", 0.0)),
                    "rot_z": float(entry.get("rot_z", 0.0)),
                    "trans_x": float(entry.get("trans_x", 0.0)),
                    "trans_y": float(entry.get("trans_y", 0.0)),
                    "trans_z": float(entry.get("trans_z", 0.0)),
                    "scale_x": float(entry.get("scale_x", 1.0)),
                    "scale_y": float(entry.get("scale_y", 1.0)),
                    "scale_z": float(entry.get("scale_z", 1.0)),
                    "color": tuple(float(c) for c in entry["color"]),
                }
                model.objects.append(obj)

        n = len(model.objects)
        active = int(manifest.get("active_index", -1))
        if not (0 <= active < n):
            active = -1 if n == 0 else min(max(active, 0), n - 1)
        model.active_index = active
        return model
