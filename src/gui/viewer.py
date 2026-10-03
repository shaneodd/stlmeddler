"""
MeshViewer widget for displaying Open3D meshes in a Qt OpenGL canvas.

The viewer is a thin Qt/OpenGL facade over :class:`src.gui.scene.SceneModel`,
which owns the list of independent objects, their per-object transforms and
colours. All object-management and transform logic lives in ``SceneModel`` so it
can be tested without a display
this class only renders and handles input.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QLabel, QOpenGLWidget

from .picking import screen_to_world_ray
from .scene import SceneModel, model_matrix

if TYPE_CHECKING:  # static-analysis-only imports
    import open3d as o3d

    from .scene import SceneObject

    type TriangleMesh = "o3d.geometry.TriangleMesh"


class MeshViewer(QOpenGLWidget):
    """Multi-object OpenGL mesh viewer (Qt facade over :class:`SceneModel`)."""

    # Emitted with the new active index whenever selection changes (-1 = none).
    active_changed = pyqtSignal(int)
    # Emitted when face-pick mode ends (after aligning or cancellation).
    face_pick_done = pyqtSignal()
    # Emitted when measure mode ends.
    measure_done = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.scene = SceneModel()
        # Camera parameters (simple orbit view)
        self.zoom = 1.0
        self.camera_rot_x = 0.0
        self.camera_rot_y = 0.0
        self.camera_rot_z = 0.0
        # World-space point the camera orbits around and looks at.
        self.camera_center = np.zeros(3, dtype=float)
        # Build-plate grid metrics, rescaled to the scene on every fit.
        self._plate_half = 10.0
        self._plate_step = 1.0
        # Cut plane preview
        self.cut_plane_z = 0.0
        self.show_cut_plane = False
        self.last_pos = None
        # Click-vs-drag tracking for pick-on-click selection.
        self._press_pos = None
        # When True, the next click picks a face and lays it on the build plate.
        self._face_pick_mode = False
        # Face under the cursor while face-picking: (object_index, triangle_row).
        self._hover_face: tuple[int, int] | None = None
        self._hover_pos_px: tuple[int, int] | None = None
        self._last_hover_at = 0.0
        # Measure mode: two picked world points, plus a live hover endpoint.
        self._measure_mode = False
        self._measure_a: np.ndarray | None = None
        self._measure_b: np.ndarray | None = None
        self._measure_hover: np.ndarray | None = None
        self._measure_hover_px: tuple[int, int] | None = None
        self._measure_last_at = 0.0
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)  # hover feedback while face-picking

        self._help_label = QLabel(
            "Click a face on the model to place it on the plate  \u00b7  Esc to cancel", self
        )
        self._help_label.setAlignment(Qt.AlignCenter)
        self._help_label.setStyleSheet(
            "background-color: rgba(20, 20, 20, 190); color: white;"
            "padding: 6px 14px; border-radius: 8px;"
        )
        self._help_label.hide()

    # ------------------------------------------------------------------ #
    # Object management (delegates to SceneModel, emits signals + repaints)
    # ------------------------------------------------------------------ #
    def add_object(
        self, mesh: TriangleMesh, name: str | None = None, auto_align: bool = True
    ) -> int:
        idx = self.scene.add_object(mesh, name=name, auto_align=auto_align)
        self._fit_view()
        self.active_changed.emit(self.scene.active_index)
        self.update()
        return idx

    def remove_object(self, index: int) -> None:
        self.scene.remove_object(index)
        self._fit_view()
        self.active_changed.emit(self.scene.active_index)
        self.update()

    def remove_by_name(self, name: str) -> bool:
        ok = self.scene.remove_by_name(name)
        if ok:
            self._fit_view()
            self.active_changed.emit(self.scene.active_index)
            self.update()
        return ok

    def get_objects(self) -> list[SceneObject]:
        return self.scene.get_objects()

    def get_index(self, name: str) -> int:
        return self.scene.get_index(name)

    @property
    def active_index(self) -> int:
        return self.scene.active_index

    def set_active(self, index: int) -> None:
        before = self.scene.active_index
        self.scene.set_active(index)
        if self.scene.active_index != before:
            self._fit_view()
            self.active_changed.emit(self.scene.active_index)
            self.update()

    # ------------------------------------------------------------------ #
    # Per-object transforms (delegate + repaint)
    # ------------------------------------------------------------------ #
    def apply_transform(self, matrix: np.ndarray, index: int | None = None) -> None:
        self.scene.apply_transform(matrix, index)
        self.update()

    def translate(self, vec: Sequence[float], index: int | None = None) -> None:
        self.scene.translate(vec, index)
        self.update()

    def rotate(self, axis: Sequence[float], angle_rad: float, index: int | None = None) -> None:
        self.scene.rotate(axis, angle_rad, index)
        self.update()

    def set_rotation(
        self,
        rx: float | None = None,
        ry: float | None = None,
        rz: float | None = None,
        index: int | None = None,
    ) -> None:
        self.scene.set_rotation(rx, ry, rz, index)
        self.update()

    def set_translation(
        self,
        tx: float | None = None,
        ty: float | None = None,
        tz: float | None = None,
        index: int | None = None,
    ) -> None:
        self.scene.set_translation(tx, ty, tz, index)
        self.update()

    def set_scale(
        self,
        sx: float | None = None,
        sy: float | None = None,
        sz: float | None = None,
        index: int | None = None,
    ) -> None:
        self.scene.set_scale(sx, sy, sz, index)
        self.update()

    def get_dimensions(self, index: int | None = None) -> tuple[float, float, float]:
        return self.scene.get_dimensions(index)

    def set_dimensions(
        self,
        width: float | None = None,
        depth: float | None = None,
        height: float | None = None,
        index: int | None = None,
    ) -> bool:
        ok = self.scene.set_dimensions(width, depth, height, index)
        self.update()
        return ok

    def get_transform(self, index: int | None = None) -> dict[str, float]:
        return self.scene.get_transform(index)

    def center_active_on_plate(self) -> bool:
        """Centre the active object over the plate origin, resting on Z=0."""
        if not self.scene.center_active_on_plate():
            return False
        self._fit_view()
        self.active_changed.emit(self.scene.active_index)  # refresh transform UI
        self.update()
        return True

    # ------------------------------------------------------------------ #
    # Face-pick mode: the next clicked face is laid flat on the build plate
    # ------------------------------------------------------------------ #
    def start_face_pick_mode(self) -> None:
        """Arm face-picking: the next click on a face places it on the plate."""
        if self._measure_mode:
            self._exit_measure_mode()
        self._face_pick_mode = True
        self.setCursor(Qt.CrossCursor)
        self._reposition_help_label()
        self._help_label.show()
        self._help_label.raise_()

    def cancel_face_pick_mode(self) -> None:
        self._exit_face_pick_mode()

    @property
    def face_pick_mode(self) -> bool:
        return self._face_pick_mode

    def _exit_face_pick_mode(self) -> None:
        if self._face_pick_mode:
            self._face_pick_mode = False
            self._hover_face = None
            self._hover_pos_px = None
            self._help_label.hide()
            self.unsetCursor()
            self.face_pick_done.emit()
            self.update()

    def _align_picked_face(self, x_px: int, y_px: int) -> bool:
        """Lay the face under the given pixel on the plate; consumes the click."""
        ray = self._ray_at(x_px, y_px)
        if ray is None:
            return False
        hit = self.scene.hit_test(ray[0], ray[1])
        if hit is None:
            return False  # stay armed until a real face is clicked
        idx, tri = hit
        if idx != self.scene.active_index:
            self.set_active(idx)
        if not self.scene.align_face_to_plate(tri, index=idx, view_dir=tuple(ray[1])):
            self.scene.align_active_to_plate(idx)
        self._exit_face_pick_mode()
        self._fit_view()
        self.active_changed.emit(self.scene.active_index)  # refresh transform UI
        self.update()
        return True

    # ------------------------------------------------------------------ #
    # Measure mode: click two surface points to read their distance
    # ------------------------------------------------------------------ #
    def start_measure_mode(self) -> None:
        """Arm measuring: two clicks pick surface points; the distance is shown."""
        if self._face_pick_mode:
            self._exit_face_pick_mode()
        self._measure_mode = True
        self._measure_a = None
        self._measure_b = None
        self._measure_hover = None
        self._set_help_text("Click the first point on the model  \u00b7  Esc to cancel")
        self.setCursor(Qt.CrossCursor)
        self._reposition_help_label()
        self._help_label.show()
        self._help_label.raise_()

    def cancel_measure_mode(self) -> None:
        self._exit_measure_mode()

    @property
    def measure_mode(self) -> bool:
        return self._measure_mode

    def _exit_measure_mode(self) -> None:
        if self._measure_mode:
            self._measure_mode = False
            self._measure_a = None
            self._measure_b = None
            self._measure_hover = None
            self._measure_hover_px = None
            self._help_label.hide()
            self.unsetCursor()
            self.measure_done.emit()
            self.update()

    @staticmethod
    def _world_per_pixel(cam_dist: float, viewport_height: int) -> float:
        """World units per screen pixel at the orbit pivot depth (45-degree fov)."""
        if viewport_height <= 0:
            return 0.0
        return 2.0 * math.tan(math.radians(45.0) / 2.0) * cam_dist / viewport_height

    @staticmethod
    def _snap_to_vertex(
        hit: np.ndarray, tri_verts: np.ndarray, threshold: float
    ) -> np.ndarray:
        """Snap a hit point to the nearest triangle corner within ``threshold``."""
        distances = np.linalg.norm(tri_verts - hit, axis=1)
        nearest = int(np.argmin(distances))
        if threshold > 0.0 and distances[nearest] <= threshold:
            return np.asarray(tri_verts[nearest], dtype=float)
        return np.asarray(hit, dtype=float)

    def _pick_point(self, x_px: int, y_px: int) -> np.ndarray | None:
        """World point under the cursor, snapped to a nearby mesh vertex."""
        ray = self._ray_at(x_px, y_px)
        if ray is None:
            return None
        hit = self.scene.pick_surface(ray[0], ray[1])
        if hit is None:
            return None
        obj_index, tri_row, point = hit
        tri_idx = self.scene.objects[obj_index]["tris"][tri_row]
        world = self.scene.transformed_vertices(obj_index)
        tri_verts = world[tri_idx]
        dist = 3.0 * self.zoom
        threshold = 10.0 * self._world_per_pixel(dist, self.height())
        return self._snap_to_vertex(point, tri_verts, threshold)

    def _measure_click(self, x_px: int, y_px: int) -> None:
        """Handle a click while measuring: set A, then B, then restart."""
        point = self._pick_point(x_px, y_px)
        if point is None:
            return
        if self._measure_a is None or self._measure_b is not None:
            self._measure_a = point
            self._measure_b = None
            self._measure_hover = None
            self._set_help_text("Click the second point  \u00b7  Esc to cancel")
        else:
            self._measure_b = point
            self._measure_hover = None
            self._set_help_text(
                f"Distance: {self._measure_distance_text()} mm"
                "  \u00b7  click to measure again  \u00b7  Esc to finish"
            )
        self.update()

    def _measure_distance_text(self) -> str:
        if self._measure_a is None or self._measure_b is None:
            return ""
        return f"{float(np.linalg.norm(self._measure_b - self._measure_a)):.2f}"

    def _update_measure_hover(self, x_px: int, y_px: int) -> None:
        """Live second endpoint while dragging the measure after the first click."""
        if self._measure_a is None or self._measure_b is not None:
            return
        if self._measure_hover_px is not None:
            moved = abs(x_px - self._measure_hover_px[0]) + abs(y_px - self._measure_hover_px[1])
            if moved < 2:
                return
        now = time.monotonic()
        if now - self._measure_last_at < 0.03:
            return
        self._measure_hover_px = (x_px, y_px)
        self._measure_last_at = now
        point = self._pick_point(x_px, y_px)
        self._measure_hover = point
        if point is not None:
            d = float(np.linalg.norm(point - self._measure_a))
            self._set_help_text(f"Click the second point  \u2014  {d:.2f} mm")
        self.update()

    def _set_help_text(self, text: str) -> None:
        self._help_label.setText(text)
        self._reposition_help_label()

    def cut_active_at_z(self, z_height: float) -> list[str]:
        added = self.scene.cut_active_at_z(z_height)
        self._fit_view()
        self.active_changed.emit(self.scene.active_index)
        self.update()
        return added

    def boolean_active_with(self, other_index: int, kind: str) -> str | None:
        name = self.scene.boolean_active_with(other_index, kind)
        if name is not None:
            self._fit_view()
            self.active_changed.emit(self.scene.active_index)
            self.update()
        return name

    # ------------------------------------------------------------------ #
    # Geometry helpers for export / framing
    # ------------------------------------------------------------------ #
    def transformed_vertices(self, index: int) -> np.ndarray:
        return self.scene.transformed_vertices(index)

    def merged_mesh(self) -> TriangleMesh:
        return self.scene.merged_mesh()

    # ------------------------------------------------------------------ #
    # Project persistence (delegates to SceneModel; load refreshes the view)
    # ------------------------------------------------------------------ #
    def save_project(self, path: str) -> None:
        """Save the whole multi-object scene to a ``.stlproj`` bundle."""
        self.scene.save_project(path)

    def load_project(self, path: str) -> None:
        """Replace the scene with objects loaded from a ``.stlproj`` bundle."""
        self.scene = SceneModel.load_project(path)
        self._fit_view()
        self.active_changed.emit(self.scene.active_index)
        self.update()

    def _fit_view(self) -> None:
        """Frame the active object (or the whole scene) centred in the viewport."""
        all_pts = [self.scene.transformed_vertices(i) for i in range(len(self.scene.objects))]
        idx = self.scene.active_index
        active = all_pts[idx] if 0 <= idx < len(all_pts) and len(all_pts[idx]) else None
        if active is not None:
            verts = active
        else:
            filled = [p for p in all_pts if len(p)]
            verts = np.vstack(filled) if filled else np.empty((0, 3))
        # The plate should cover every object so relative scale reads correctly.
        world_extent = max((float(np.abs(p).max()) for p in all_pts if len(p)), default=0.0)
        self._plate_half, self._plate_step = self._plate_metrics(world_extent)
        if verts.size == 0:
            self.camera_center = np.zeros(3, dtype=float)
            return
        bmin = verts.min(axis=0)
        bmax = verts.max(axis=0)
        center = (bmin + bmax) / 2.0
        # Bounding-sphere radius from the box centre so corners stay on screen.
        radius = max(float(np.linalg.norm(bmax - bmin)) / 2.0, 0.1)
        self.camera_center = center.astype(float)
        self.zoom = (self._fit_distance(radius) / 3.0)

    @staticmethod
    def _plate_metrics(world_extent: float) -> tuple[float, float]:
        """Grid (half-size, step) sized to cover the scene with round units."""
        if world_extent <= 0.0 or not math.isfinite(world_extent):
            return 10.0, 1.0
        raw = world_extent * 1.35
        step_raw = max(raw / 10.0, 1e-9)
        mag = 10.0 ** math.floor(math.log10(step_raw))
        frac = step_raw / mag
        nice = 1.0 if frac < 1.5 else 2.0 if frac < 3.0 else 5.0 if frac < 7.0 else 10.0
        step = nice * mag
        half = math.ceil(raw / step) * step
        return half, step

    def _fit_distance(self, radius: float) -> float:
        """Camera distance at which a sphere of ``radius`` fits the viewport."""
        w, h = self.width(), self.height()
        aspect = w / max(h, 1) if w > 0 and h > 0 else 1.0
        # Half-FOV of the narrower axis; distance = r / sin(half_fov) is tangent fit.
        tan_half = math.tan(math.radians(45.0) / 2.0) * min(aspect, 1.0)
        sin_min = tan_half / math.sqrt(1.0 + tan_half * tan_half)
        return 1.15 * radius / sin_min

    # ------------------------------------------------------------------ #
    # Cut plane preview
    # ------------------------------------------------------------------ #
    def set_cut_plane(self, z: float) -> None:
        self.cut_plane_z = float(z)
        self.update()

    def set_show_cut_plane(self, visible: bool) -> None:
        self.show_cut_plane = bool(visible)
        self.update()

    # ------------------------------------------------------------------ #
    # OpenGL rendering
    # ------------------------------------------------------------------ #
    def initializeGL(self) -> None:
        try:
            from OpenGL.GL import (
                GL_AMBIENT,
                GL_AMBIENT_AND_DIFFUSE,
                GL_COLOR_BUFFER_BIT,
                GL_COLOR_MATERIAL,
                GL_CULL_FACE,
                GL_DEPTH_BUFFER_BIT,
                GL_DEPTH_TEST,
                GL_DIFFUSE,
                GL_EMISSION,
                GL_FLOAT,
                GL_FRONT_AND_BACK,
                GL_LIGHT0,
                GL_LIGHT1,
                GL_LIGHT_MODEL_TWO_SIDE,
                GL_LIGHTING,
                GL_LINES,
                GL_MODELVIEW,
                GL_NORMAL_ARRAY,
                GL_NORMALIZE,
                GL_POSITION,
                GL_PROJECTION,
                GL_SHININESS,
                GL_SPECULAR,
                GL_TRIANGLES,
                GL_VERTEX_ARRAY,
                glBegin,
                glClear,
                glClearColor,
                glColor3f,
                glColorMaterial,
                glDisable,
                glDisableClientState,
                glDrawArrays,
                glEnable,
                glEnableClientState,
                glEnd,
                glFrustum,
                glLightfv,
                glLightModelf,
                glLoadIdentity,
                glMaterialfv,
                glMatrixMode,
                glMultMatrixf,
                glNormalPointer,
                glPopMatrix,
                glPushMatrix,
                glRotatef,
                glTranslatef,
                glVertex3f,
                glVertexPointer,
            )
        except ImportError as e:
            raise RuntimeError("PyOpenGL is required for rendering") from e
        import math
        self._gl = {
            "glClear": glClear, "glMatrixMode": glMatrixMode,
            "GL_COLOR_BUFFER_BIT": GL_COLOR_BUFFER_BIT, "GL_DEPTH_BUFFER_BIT": GL_DEPTH_BUFFER_BIT,
            "glLoadIdentity": glLoadIdentity, "glTranslatef": glTranslatef,
            "glRotatef": glRotatef, "glBegin": glBegin, "GL_TRIANGLES": GL_TRIANGLES,
            "GL_LINES": GL_LINES, "glVertex3f": glVertex3f, "glEnd": glEnd,
            "glPushMatrix": glPushMatrix, "glPopMatrix": glPopMatrix,
            "glEnable": glEnable, "GL_DEPTH_TEST": GL_DEPTH_TEST,
            "GL_CULL_FACE": GL_CULL_FACE, "glDisable": glDisable,
            "glClearColor": glClearColor, "glColor3f": glColor3f,
            "GL_PROJECTION": GL_PROJECTION, "GL_MODELVIEW": GL_MODELVIEW,
            "glFrustum": glFrustum, "glMultMatrixf": glMultMatrixf, "math": math,
            "GL_LIGHTING": GL_LIGHTING, "GL_LIGHT0": GL_LIGHT0, "GL_LIGHT1": GL_LIGHT1,
            "GL_NORMALIZE": GL_NORMALIZE, "GL_POSITION": GL_POSITION,
            "glLightfv": glLightfv,
            "GL_FRONT_AND_BACK": GL_FRONT_AND_BACK, "GL_EMISSION": GL_EMISSION,
            "glMaterialfv": glMaterialfv,
            "GL_VERTEX_ARRAY": GL_VERTEX_ARRAY, "GL_NORMAL_ARRAY": GL_NORMAL_ARRAY,
            "GL_FLOAT": GL_FLOAT, "glDrawArrays": glDrawArrays,
            "glVertexPointer": glVertexPointer, "glNormalPointer": glNormalPointer,
            "glEnableClientState": glEnableClientState,
            "glDisableClientState": glDisableClientState,
        }
        g = self._gl
        g["glEnable"](g["GL_DEPTH_TEST"])
        g["glDisable"](g["GL_CULL_FACE"])  # see all faces incl. inside-out caps
        g["glClearColor"](0.2, 0.2, 0.2, 1.0)

        # Fixed-function lighting: a head key light plus an angled fill light so
        # faces are distinguishable from any camera angle.
        glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, 1.0)  # light back faces too (open caps)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_NORMALIZE)  # keep normals unit-length under scaling
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.35, 0.35, 0.35, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.7, 0.7, 0.7, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (0.35, 0.35, 0.35, 1.0))
        glLightfv(GL_LIGHT1, GL_DIFFUSE, (0.25, 0.25, 0.25, 1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (0.3, 0.3, 0.3, 1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, 24.0)
        self._setup_projection(self.width(), self.height())

    def _frustum(self, w: int, h: int) -> tuple[float, float, float, float, float, float]:
        """Camera frustum bounds (left,right,bottom,top,near,far).

        Single source of truth shared by :meth:`_setup_projection` (rendering) and
        click-to-select ray casting, so picking always matches what is drawn.
        Near/far follow the current camera distance so framing never clips.
        """
        aspect = w / max(h, 1)
        fov = 45.0
        dist = 3.0 * self.zoom
        near = max(0.001, dist * 0.01)
        far = dist * 50.0 + 100.0
        # Half-height of the frustum at the near plane: near * tan(fov / 2).
        fovy = math.tan(math.radians(fov) / 2.0) * near
        return (-fovy * aspect, fovy * aspect, -fovy, fovy, near, far)

    def _setup_projection(self, w: int, h: int) -> None:
        g = self._gl
        g["glMatrixMode"](g["GL_PROJECTION"])
        g["glLoadIdentity"]()
        left, right, bottom, top, near, far = self._frustum(w, h)
        g["glFrustum"](left, right, bottom, top, near, far)

    def resizeGL(self, w: int, h: int) -> None:
        try:
            from OpenGL.GL import glViewport
        except ImportError as e:
            raise RuntimeError("PyOpenGL is required for rendering") from e
        glViewport(0, 0, w, h)
        self._setup_projection(w, h)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._help_label.isVisible():
            self._reposition_help_label()

    @staticmethod
    def _render_cache(obj: SceneObject) -> tuple[np.ndarray, np.ndarray]:
        """Per-object ``(positions, normals)`` float32 arrays for vertex-array draws.

        Triangles are expanded once (face normals duplicated to their three
        vertices for flat shading) and cached on the object, so a mesh with
        hundreds of thousands of facets costs a single ``glDrawArrays`` per
        frame instead of millions of immediate-mode calls.
        """
        cached = obj.get("render_arrays")
        n_tris = len(obj["tris"])
        if cached is not None and len(cached[0]) == 3 * n_tris:
            return cached
        tris = obj["tris"]
        v0 = obj["verts"][tris[:, 0]]
        v1 = obj["verts"][tris[:, 1]]
        v2 = obj["verts"][tris[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths[lengths == 0.0] = 1.0
        normals = normals / lengths
        positions = np.ascontiguousarray(
            np.concatenate([v0[:, None], v1[:, None], v2[:, None]], axis=1).reshape(-1, 3),
            dtype=np.float32,
        )
        normals = np.ascontiguousarray(np.repeat(normals, 3, axis=0), dtype=np.float32)
        obj["render_arrays"] = (positions, normals)
        return obj["render_arrays"]

    def paintGL(self) -> None:
        g = getattr(self, "_gl", None)
        if g is None:
            return
        g["glClear"](g["GL_COLOR_BUFFER_BIT"])
        g["glClear"](g["GL_DEPTH_BUFFER_BIT"])
        # Frustum tracks the orbit distance, so refresh it every frame.
        self._setup_projection(self.width(), self.height())
        g["glMatrixMode"](g["GL_MODELVIEW"])
        g["glLoadIdentity"]()
        # Directional lights specified in eye space (before camera rotation):
        # a head key light plus a fill from the upper-left for shape cues.
        g["glLightfv"](g["GL_LIGHT0"], g["GL_POSITION"], (0.0, 0.0, 1.0, 0.0))
        g["glLightfv"](g["GL_LIGHT1"], g["GL_POSITION"], (-0.5, 0.6, 0.4, 0.0))
        g["glTranslatef"](0.0, 0.0, -3.0 * self.zoom)
        g["glRotatef"](self.camera_rot_x, 1.0, 0.0, 0.0)
        g["glRotatef"](self.camera_rot_y, 0.0, 1.0, 0.0)
        g["glRotatef"](self.camera_rot_z, 0.0, 0.0, 1.0)
        # Move the orbit pivot to the eye-space origin so it stays screen-centred.
        cx, cy, cz = self.camera_center
        g["glTranslatef"](-cx, -cy, -cz)

        g["glEnableClientState"](g["GL_VERTEX_ARRAY"])
        g["glEnableClientState"](g["GL_NORMAL_ARRAY"])
        no_emission = (0.0, 0.0, 0.0, 1.0)
        glow = (0.12, 0.12, 0.12, 1.0)
        for i, obj in enumerate(self.scene.objects):
            if obj["tris"].size == 0 or obj["verts"].size == 0:
                continue
            M = obj.get("matrix_override", model_matrix(obj))
            g["glPushMatrix"]()
            # glMultMatrixf expects column-major order.
            g["glMultMatrixf"](np.asarray(M, dtype=np.float32).flatten(order="F").tolist())
            r, gr, b = obj["color"]
            # Highlight the active object with a subtle emission glow (hue preserved).
            g["glMaterialfv"](g["GL_FRONT_AND_BACK"], g["GL_EMISSION"],
                              glow if i == self.scene.active_index else no_emission)
            g["glColor3f"](r, gr, b)
            positions, normals = self._render_cache(obj)
            g["glVertexPointer"](3, g["GL_FLOAT"], 0, positions)
            g["glNormalPointer"](g["GL_FLOAT"], 0, normals)
            g["glDrawArrays"](g["GL_TRIANGLES"], 0, len(positions))
            g["glMaterialfv"](g["GL_FRONT_AND_BACK"], g["GL_EMISSION"], no_emission)
            g["glPopMatrix"]()
        g["glDisableClientState"](g["GL_VERTEX_ARRAY"])
        g["glDisableClientState"](g["GL_NORMAL_ARRAY"])

        # Scene overlays drawn unlit so their line colours stay constant.
        g["glDisable"](g["GL_LIGHTING"])
        self._draw_build_plate()
        self._draw_axes()
        if self.show_cut_plane:
            self._draw_cut_plane(self.cut_plane_z)
        if self._face_pick_mode:
            self._draw_hover_face()
        if self._measure_mode:
            self._draw_measure()
        g["glEnable"](g["GL_LIGHTING"])

    def _draw_measure(self) -> None:
        """Endpoints + connecting line for the active measurement, drawn through geometry."""
        a = self._measure_a
        if a is None:
            return
        b = self._measure_b if self._measure_b is not None else self._measure_hover
        try:
            from OpenGL.GL import (
                GL_DEPTH_TEST,
                GL_LINES,
                glBegin,
                glColor3f,
                glDisable,
                glEnable,
                glEnd,
                glVertex3f,
            )
            marker = 8.0 * self._world_per_pixel(3.0 * self.zoom, self.height())
            glDisable(GL_DEPTH_TEST)  # measurement stays visible through the model
            glEnable(GL_LINES)
            if b is not None:
                glColor3f(1.0, 0.85, 0.2)
                glBegin(GL_LINES)
                glVertex3f(*a)
                glVertex3f(*b)
                glEnd()
            glColor3f(1.0, 0.35, 0.1)
            glBegin(GL_LINES)
            for point in (a, b):
                if point is None:
                    continue
                for axis in range(3):
                    lo = list(point)
                    hi = list(point)
                    lo[axis] -= marker
                    hi[axis] += marker
                    glVertex3f(*lo)
                    glVertex3f(*hi)
            glEnd()
            glDisable(GL_LINES)
            glEnable(GL_DEPTH_TEST)
        except Exception:
            pass

    def _draw_hover_face(self) -> None:
        """Outline + translucent fill on the triangle under the cursor while picking."""
        face = self._hover_face
        if face is None:
            return
        obj_index, tri_row = face
        if not (0 <= obj_index < len(self.scene.objects)):
            return
        obj = self.scene.objects[obj_index]
        if not (0 <= tri_row < len(obj["tris"])):
            return
        g = self._gl
        world = self.scene.transformed_vertices(obj_index)
        tri = world[obj["tris"][tri_row]]
        try:
            from OpenGL.GL import (
                GL_BLEND,
                GL_ONE_MINUS_SRC_ALPHA,
                GL_POLYGON_OFFSET_FILL,
                GL_SRC_ALPHA,
                GL_TRIANGLES,
                glBegin,
                glBlendFunc,
                glColor4f,
                glDisable,
                glEnable,
                glEnd,
                glPolygonOffset,
                glVertex3f,
            )
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(-1.0, -1.0)  # nudge toward camera to avoid z-fighting
            glColor4f(1.0, 0.8, 0.2, 0.4)
            glBegin(GL_TRIANGLES)
            for k in range(3):
                glVertex3f(*tri[k])
            glEnd()
            glDisable(GL_POLYGON_OFFSET_FILL)
            glColor4f(1.0, 0.6, 0.0, 0.95)
            glBegin(g["GL_LINES"])
            for a, b in ((0, 1), (1, 2), (2, 0)):
                glVertex3f(*tri[a])
                glVertex3f(*tri[b])
            glEnd()
            glDisable(GL_BLEND)
        except Exception:
            pass

    def _draw_cut_plane(self, z: float) -> None:
        try:
            from OpenGL.GL import (
                GL_BLEND,
                GL_LINES,
                GL_ONE_MINUS_SRC_ALPHA,
                GL_QUADS,
                GL_SRC_ALPHA,
                glBegin,
                glBlendFunc,
                glColor4f,
                glDisable,
                glEnable,
                glEnd,
                glVertex3f,
            )
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(0.2, 0.8, 0.2, 0.3)
            size = max(self._plate_half, 10.0)
            glBegin(GL_QUADS)
            glVertex3f(-size, -size, z)
            glVertex3f(size, -size, z)
            glVertex3f(size, size, z)
            glVertex3f(-size, size, z)
            glEnd()
            glColor4f(0.2, 1.0, 0.2, 0.8)
            glBegin(GL_LINES)
            glVertex3f(-size, -size, z)
            glVertex3f(size, -size, z)
            glVertex3f(size, -size, z)
            glVertex3f(size, size, z)
            glVertex3f(size, size, z)
            glVertex3f(-size, size, z)
            glVertex3f(-size, size, z)
            glVertex3f(-size, -size, z)
            glEnd()
            glDisable(GL_BLEND)
        except Exception:
            pass

    def _draw_build_plate(self) -> None:
        g = self._gl
        half = self._plate_half
        step = self._plate_step if self._plate_step > 0 else 1.0
        g["glColor3f"](0.5, 0.5, 0.5)
        g["glBegin"](g["GL_LINES"])
        n = int(round(half / step))
        for i in range(-n, n + 1):
            x = i * step
            g["glVertex3f"](x, -half, 0.0)
            g["glVertex3f"](x, half, 0.0)
            g["glVertex3f"](-half, x, 0.0)
            g["glVertex3f"](half, x, 0.0)
        g["glEnd"]()

    def _draw_axes(self) -> None:
        g = self._gl
        length = self._plate_half * 0.6
        g["glBegin"](g["GL_LINES"])
        g["glColor3f"](1.0, 0.0, 0.0)
        g["glVertex3f"](0, 0, 0)
        g["glVertex3f"](length, 0, 0)
        g["glColor3f"](0.0, 1.0, 0.0)
        g["glVertex3f"](0, 0, 0)
        g["glVertex3f"](0, length, 0)
        g["glColor3f"](0.0, 0.0, 1.0)
        g["glVertex3f"](0, 0, 0)
        g["glVertex3f"](0, 0, length)
        g["glEnd"]()

    # ------------------------------------------------------------------ #
    # Mouse interaction (camera orbit + zoom)
    # ------------------------------------------------------------------ #
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.last_pos = event.pos()
            self._press_pos = event.pos()
            if self._hover_face is not None:  # dragging hides the highlight
                self._hover_face = None
                self.update()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        # A left click with negligible movement either picks a face to place on
        # the plate (when armed) or selects the object under the cursor; a drag
        # (orbit) is ignored so orbiting never changes selection.
        if event.button() == Qt.LeftButton and self._press_pos is not None:
            moved = abs(event.x() - self._press_pos.x()) + abs(event.y() - self._press_pos.y())
            if moved <= 4:
                if self._face_pick_mode:
                    self._align_picked_face(event.x(), event.y())
                elif self._measure_mode:
                    self._measure_click(event.x(), event.y())
                else:
                    self._pick_at(event.x(), event.y())
        self.last_pos = None
        self._press_pos = None

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            if self._face_pick_mode:
                self.cancel_face_pick_mode()
            elif self._measure_mode:
                self.cancel_measure_mode()
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def _ray_at(self, x_px: int, y_px: int):
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return None
        left, right, bottom, top, near, far = self._frustum(w, h)
        return screen_to_world_ray(
            width=w, height=h, x_px=x_px, y_px=y_px, zoom=self.zoom,
            rot_x_deg=self.camera_rot_x, rot_y_deg=self.camera_rot_y,
            rot_z_deg=self.camera_rot_z, center=self.camera_center,
            left=left, right=right, bottom=bottom,
            top=top, near=near, far=far,
        )

    def _pick_at(self, x_px: int, y_px: int) -> None:
        ray = self._ray_at(x_px, y_px)
        if ray is None:
            return
        idx_tri = self.scene.hit_test(ray[0], ray[1])
        if idx_tri is None:
            return
        idx, _tri = idx_tri
        if idx != self.active_index:
            self.set_active(idx)

    def mouseMoveEvent(self, event) -> None:
        if self.last_pos is not None:
            dx = event.x() - self.last_pos.x()
            dy = event.y() - self.last_pos.y()
            self.camera_rot_x += dy * 0.5
            self.camera_rot_y += dx * 0.5
            self.last_pos = event.pos()
            self.update()
            return
        if self._face_pick_mode:
            self._update_hover(event.x(), event.y())
        elif self._measure_mode:
            self._update_measure_hover(event.x(), event.y())

    def _update_hover(self, x_px: int, y_px: int) -> None:
        """Re-pick the face under the cursor, throttled to stay cheap on big scans."""
        if self._hover_pos_px is not None:
            moved = abs(x_px - self._hover_pos_px[0]) + abs(y_px - self._hover_pos_px[1])
            if moved < 2:
                return
        now = time.monotonic()
        if now - self._last_hover_at < 0.03:
            return
        self._hover_pos_px = (x_px, y_px)
        self._last_hover_at = now
        ray = self._ray_at(x_px, y_px)
        hit = self.scene.hit_test(ray[0], ray[1]) if ray is not None else None
        if hit != self._hover_face:
            self._hover_face = hit
            self.update()

    def _reposition_help_label(self) -> None:
        self._help_label.adjustSize()
        x = (self.width() - self._help_label.width()) // 2
        self._help_label.move(max(x, 4), 10)

    def leaveEvent(self, event) -> None:
        if self._hover_face is not None:
            self._hover_face = None
            self.update()
        if self._measure_hover is not None:
            self._measure_hover = None
            self.update()
        super().leaveEvent(event)

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y() / 120
        if delta > 0:
            self.zoom *= 1.2 ** delta
        elif delta < 0:
            self.zoom /= 1.2 ** (-delta)
        self.zoom = max(0.001, min(1000.0, self.zoom))
        self.update()
