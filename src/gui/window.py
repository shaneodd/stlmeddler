"""
MainWindow definition for the STL/OBJ Processing Tool desktop GUI.

Uses PyQt5 and embeds a :class:`MeshViewer` widget for 3-D visualisation. The
viewer owns the authoritative list of loaded objects; this window is a thin
controller that mirrors the object list, edits the active object's transform via
spin boxes, and drives operations (open, primitives, boolean, cut, export).
"""

from pathlib import Path

from PyQt5.QtWidgets import (
    QAction,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .viewer import MeshViewer


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("STL/OBJ Processing Tool")
        self.resize(1200, 768)

        central = QWidget(self)
        main_layout = QVBoxLayout(central)
        content_layout = QHBoxLayout()

        # Left: viewer
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.viewer = MeshViewer(parent=left_widget)
        left_layout.addWidget(self.viewer)

        # Right: object list + controls
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 10, 10, 10)

        object_label = QLabel("Loaded Objects:")
        object_label.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(object_label)
        self.object_list = QListWidget()
        self.object_list.setMaximumWidth(150)
        self.object_list.setMinimumWidth(100)
        self.object_list.itemSelectionChanged.connect(self._on_list_selection_changed)
        right_layout.addWidget(self.object_list)

        self.delete_button = QPushButton("Delete Selected Object")
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self._delete_selected_object)
        right_layout.addWidget(self.delete_button)

        # Cut plane panel
        cut_label = QLabel("Cut Plane")
        cut_label.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(cut_label)
        cut_panel = QWidget()
        cut_layout = QVBoxLayout(cut_panel)
        self.cut_z_spin = QDoubleSpinBox()
        self.cut_z_spin.setRange(-100000.0, 100000.0)
        self.cut_z_spin.setSingleStep(0.1)
        self.cut_z_spin.setValue(0.0)
        cut_layout.addWidget(QLabel("Z Height:"))
        cut_layout.addWidget(self.cut_z_spin)
        self.cut_z_spin.valueChanged.connect(self._update_cut_plane_position)

        show_cut_btn = QPushButton("Show Cut Plane")
        show_cut_btn.setCheckable(True)
        show_cut_btn.toggled.connect(self._toggle_cut_plane_preview)
        cut_layout.addWidget(show_cut_btn)

        cut_btn = QPushButton("Cut Object Here")
        cut_btn.clicked.connect(self.perform_cut)
        cut_layout.addWidget(cut_btn)
        right_layout.addWidget(cut_panel)

        content_layout.addWidget(left_widget, stretch=3)
        content_layout.addWidget(right_widget, stretch=1)
        main_layout.addLayout(content_layout)

        # Transform control panel (edits the active object)
        rotation_panel = QWidget()
        panel_layout = QGridLayout(rotation_panel)
        self.spin_x = self._add_spin(panel_layout, 0, "Rotate X (\u00b0):", -180.0, 180.0, 1.0)
        self.spin_y = self._add_spin(panel_layout, 1, "Rotate Y (\u00b0):", -180.0, 180.0, 1.0)
        self.spin_z = self._add_spin(panel_layout, 2, "Rotate Z (\u00b0):", -180.0, 180.0, 1.0)
        self.spin_tx = self._add_spin(panel_layout, 3, "Move X:", -100000.0, 100000.0, 0.1)
        self.spin_ty = self._add_spin(panel_layout, 4, "Move Y:", -100000.0, 100000.0, 0.1)
        self.spin_tz = self._add_spin(panel_layout, 5, "Move Z:", -100000.0, 100000.0, 0.1)
        right_layout.addWidget(rotation_panel)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

        # Guard to avoid feedback loops when populating UI from object state.
        self._syncing = False

        for spin in (self.spin_x, self.spin_y, self.spin_z,
                     self.spin_tx, self.spin_ty, self.spin_tz):
            spin.valueChanged.connect(self._on_transform_edited)

        # Viewer selection changes -> reflect in list + UI.
        self.viewer.active_changed.connect(self._on_viewer_active_changed)
        self.viewer.face_pick_done.connect(lambda: self.statusBar().clearMessage())
        self.viewer.measure_done.connect(lambda: self.statusBar().clearMessage())

        self._create_actions()
        self._setup_menu()

    @staticmethod
    def _add_spin(layout, row, label, lo, hi, step):
        layout.addWidget(QLabel(label), row, 0)
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setValue(0.0)
        layout.addWidget(spin, row, 1)
        return spin

    # ------------------------------------------------------------------ #
    # Menu / actions
    # ------------------------------------------------------------------ #
    def _create_actions(self) -> None:
        self.open_action = QAction("&Open", self)
        self.open_action.triggered.connect(self.open_file)

        self.project_open_action = QAction("Open Project...", self)
        self.project_open_action.triggered.connect(self.open_project)
        self.project_save_action = QAction("Save Project As...", self)
        self.project_save_action.triggered.connect(self.save_project)

        self.save_action = QAction("Export Mesh...", self)
        self.save_action.triggered.connect(self.save_file)

        self.diff_action = QAction("Boolean Difference", self)
        self.diff_action.triggered.connect(self.boolean_difference)
        self.union_action = QAction("Boolean Union", self)
        self.union_action.triggered.connect(self.boolean_union)
        self.primitive_action = QAction("Create Primitive", self)
        self.primitive_action.triggered.connect(self.create_primitive)

        self.cut_action = QAction("Cut at Z Height...", self)
        self.cut_action.triggered.connect(self.perform_cut)

        self.align_face_action = QAction("Align Face to Build Plate", self)
        self.align_face_action.setStatusTip(
            "Click a face in the viewer, then that face is laid flat on the plate"
        )
        self.align_face_action.triggered.connect(self.align_face_to_plate)

        self.center_action = QAction("Center on Build Plate", self)
        self.center_action.setStatusTip(
            "Move the active object so it is centred over the plate origin"
        )
        self.center_action.triggered.connect(self.center_on_plate)

        self.health_action = QAction("Check Mesh Health...", self)
        self.health_action.setStatusTip(
            "Validate the active object: watertight, manifold, self-intersections, winding"
        )
        self.health_action.triggered.connect(self.check_mesh_health)

        self.measure_action = QAction("Measure Distance", self)
        self.measure_action.setStatusTip(
            "Click two points on the model to read the distance between them"
        )
        self.measure_action.triggered.connect(self.measure_distance)

    def _setup_menu(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.project_open_action)
        file_menu.addAction(self.project_save_action)
        file_menu.addAction(self.save_action)
        ops_menu = menubar.addMenu("&Operations")
        ops_menu.addAction(self.diff_action)
        ops_menu.addAction(self.union_action)
        ops_menu.addAction(self.primitive_action)
        ops_menu.addAction(self.cut_action)
        ops_menu.addSeparator()
        ops_menu.addAction(self.align_face_action)
        ops_menu.addAction(self.center_action)
        ops_menu.addSeparator()
        ops_menu.addAction(self.measure_action)
        ops_menu.addAction(self.health_action)

    # ------------------------------------------------------------------ #
    # Object list <-> viewer sync
    # ------------------------------------------------------------------ #
    def _refresh_object_list(self, select_name=None):
        """Rebuild the QListWidget from the viewer's objects."""
        self._syncing = True
        self.object_list.clear()
        for obj in self.viewer.get_objects():
            self.object_list.addItem(obj["name"])
        if select_name is not None:
            idx = self.viewer.get_index(select_name)
            if idx >= 0:
                self.object_list.setCurrentRow(idx)
        self._syncing = False
        self._update_delete_button_state()

    def _on_viewer_active_changed(self, index):
        if self._syncing:
            return
        self._syncing = True
        row = index if 0 <= index < self.object_list.count() else -1
        self.object_list.setCurrentRow(row)
        self._syncing = False
        self._update_delete_button_state()
        self._load_active_transform_into_ui()

    def _on_list_selection_changed(self) -> None:
        if self._syncing:
            return
        row = self.object_list.currentRow()
        self.viewer.set_active(row)
        self._update_delete_button_state()

    def _update_delete_button_state(self) -> None:
        self.delete_button.setEnabled(len(self.object_list.selectedItems()) > 0)

    # ------------------------------------------------------------------ #
    # Transform UI <-> active object
    # ------------------------------------------------------------------ #
    def _load_active_transform_into_ui(self) -> None:
        t = self.viewer.get_transform()
        self._syncing = True
        for spin, key in ((self.spin_x, "rot_x"), (self.spin_y, "rot_y"),
                          (self.spin_z, "rot_z"), (self.spin_tx, "trans_x"),
                          (self.spin_ty, "trans_y"), (self.spin_tz, "trans_z")):
            spin.setValue(t[key])
        self._syncing = False

    def _on_transform_edited(self) -> None:
        if self._syncing or self.viewer.active_index < 0:
            return
        self.viewer.set_rotation(
            rx=self.spin_x.value(), ry=self.spin_y.value(), rz=self.spin_z.value()
        )
        self.viewer.set_translation(
            tx=self.spin_tx.value(), ty=self.spin_ty.value(), tz=self.spin_tz.value()
        )

    def align_face_to_plate(self) -> None:
        if not self.viewer.get_objects():
            QMessageBox.warning(self, "Warning", "Load an object first.")
            return
        self.statusBar().showMessage(
            "Click a face to lay on the build plate (Esc to cancel)", 8000
        )
        self.viewer.start_face_pick_mode()

    def measure_distance(self) -> None:
        if not self.viewer.get_objects():
            QMessageBox.warning(self, "Warning", "Load an object first.")
            return
        self.statusBar().showMessage(
            "Measure mode: click two points on the model (Esc to cancel)", 8000
        )
        self.viewer.start_measure_mode()

    def center_on_plate(self) -> None:
        if self.viewer.active_index < 0:
            QMessageBox.warning(self, "Warning", "Select an object first.")
            return
        self.viewer.center_active_on_plate()

    def check_mesh_health(self) -> None:
        if self.viewer.active_index < 0:
            QMessageBox.warning(self, "Warning", "Select an object first.")
            return
        from types import SimpleNamespace

        from src.mesh_ops import mesh_health

        obj = self.viewer.get_objects()[self.viewer.active_index]
        try:
            report = mesh_health(
                SimpleNamespace(vertices=obj["verts"], triangles=obj["tris"])
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Health check failed: {e}")
            return
        checks = [
            ("Watertight", report.watertight),
            ("Edge-manifold", report.edge_manifold),
            ("Vertex-manifold", report.vertex_manifold),
            ("No self-intersections", not report.self_intersecting),
            ("Consistent winding", report.winding_consistent),
            ("No degenerate faces", report.degenerate_faces == 0),
            ("No duplicate faces", report.duplicate_faces == 0),
            ("Positive volume", report.volume > 0.0),
        ]
        lines = []
        for label, ok in checks:
            if label == "No self-intersections" and not report.self_intersection_checked:
                lines.append(f"SKIP   {label} (mesh too large)")
            else:
                lines.append(f"{'OK    ' if ok else 'PROBLEM'} {label}")
        body = "\n".join(lines)
        body += f"\n\nSigned volume: {report.volume:.3f}   Pieces: {report.components}"
        if report.ok:
            QMessageBox.information(
                self, f"Mesh Health - {obj['name']}",
                "This mesh is valid.\n\n" + body,
            )
        else:
            detail = "\n".join(f"- {p}" for p in report.problems)
            QMessageBox.warning(
                self, f"Mesh Health - {obj['name']}",
                "This mesh has problems:\n" + detail + "\n\n" + body,
            )

    # ------------------------------------------------------------------ #
    # File operations
    # ------------------------------------------------------------------ #
    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Mesh", "", "Mesh Files (*.stl *.obj)")
        if not path:
            return
        try:
            from src.mesh_ops import is_watertight, load_mesh
            mesh = load_mesh(path)
            name = self._unique_name(Path(path).stem)
            idx = self.viewer.add_object(mesh, name=name)
            self._refresh_object_list(select_name=self.viewer.get_objects()[idx]["name"])
            try:
                if not is_watertight(mesh):
                    self.statusBar().showMessage(
                        f"'{name}' is not watertight - booleans may fail; run "
                        "Operations > Check Mesh Health for details", 8000,
                    )
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open file: {e}")

    def save_file(self) -> None:
        if not self.viewer.get_objects():
            QMessageBox.warning(self, "Warning", "No mesh loaded to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Mesh As", "", "Mesh Files (*.stl *.obj *.3mf)"
        )
        if not path:
            return
        try:
            from src.mesh_ops import save_mesh
            merged = self.viewer.merged_mesh()
            save_mesh(merged, path)
            QMessageBox.information(self, "Saved", f"Mesh saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file: {e}")

    # ------------------------------------------------------------------ #
    # Project persistence (.stlproj)
    # ------------------------------------------------------------------ #
    def open_project(self) -> None:
        if self.viewer.get_objects():
            resp = QMessageBox.question(
                self,
                "Open Project",
                "Replace current scene?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if resp != QMessageBox.Yes:
                return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "STL Meddler Project (*.stlproj)"
        )
        if not path:
            return
        try:
            self.viewer.load_project(path)
            objs = self.viewer.get_objects()
            sel = None
            if 0 <= self.viewer.active_index < len(objs):
                sel = objs[self.viewer.active_index]["name"]
            self._refresh_object_list(select_name=sel)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open project: {e}")

    def save_project(self) -> None:
        if not self.viewer.get_objects():
            QMessageBox.warning(self, "Warning", "No objects to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", "", "STL Meddler Project (*.stlproj)"
        )
        if not path:
            return
        if not path.lower().endswith(".stlproj"):
            path += ".stlproj"
        try:
            self.viewer.save_project(path)
            QMessageBox.information(self, "Saved", f"Project saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save project: {e}")

    def _unique_name(self, base):
        existing = {o["name"] for o in self.viewer.get_objects()}
        if base not in existing:
            return base
        i = 1
        while f"{base}_{i}" in existing:
            i += 1
        return f"{base}_{i}"

    # ------------------------------------------------------------------ #
    # Operations
    # ------------------------------------------------------------------ #
    def create_primitive(self) -> None:
        from PyQt5.QtWidgets import QDialog

        from src.gui.primitive_dialog import PrimitiveDialog

        dialog = PrimitiveDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return
        shape, kwargs = dialog.values()
        try:
            from src.mesh_ops import create_primitive as make_primitive
            mesh = make_primitive(shape, **kwargs)
            name = self._unique_name(shape)
            idx = self.viewer.add_object(mesh, name=name)
            self._refresh_object_list(select_name=self.viewer.get_objects()[idx]["name"])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create primitive: {e}")

    def _boolean_op(self, kind):
        if self.viewer.active_index < 0:
            QMessageBox.warning(self, "Warning", "Select an object first.")
            return
        objs = self.viewer.get_objects()
        active = self.viewer.active_index
        others = [o["name"] for i, o in enumerate(objs) if i != active]
        if not others:
            QMessageBox.warning(
                self, "Warning", "Need at least two objects to perform a boolean."
            )
            return

        from PyQt5.QtWidgets import QInputDialog

        active_name = objs[active]["name"]
        if kind == "difference":
            title = "Boolean Difference"
            prompt = f"Subtract which object from '{active_name}'?"
        else:
            title = "Boolean Union"
            prompt = f"Union '{active_name}' with which object?"

        other_name, ok = QInputDialog.getItem(
            self, title, prompt, others, 0, False
        )
        if not ok or not other_name:
            return
        try:
            result_name = self.viewer.boolean_active_with(
                self.viewer.get_index(other_name), kind
            )
            if result_name is not None:
                self._refresh_object_list(select_name=result_name)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"{title} operation failed: {e}")

    def boolean_difference(self) -> None:
        self._boolean_op("difference")

    def boolean_union(self) -> None:
        self._boolean_op("union")

    # ------------------------------------------------------------------ #
    # Cut (single unified implementation)
    # ------------------------------------------------------------------ #
    def perform_cut(self) -> None:
        if self.viewer.active_index < 0:
            QMessageBox.warning(self, "Warning", "No mesh loaded to cut.")
            return
        z_height = self.cut_z_spin.value()
        try:
            added = self.viewer.cut_active_at_z(z_height)
            select = added[0] if added else None
            self._refresh_object_list(select_name=select)
            if added:
                msg = f"Mesh cut at Z={z_height}. Parts: {', '.join(added)}."
                QMessageBox.information(self, "Cut Complete", msg)
            else:
                QMessageBox.warning(
                    self, "Cut Result", f"Mesh cut at Z={z_height} but no parts remain."
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to cut mesh: {e}")

    # ------------------------------------------------------------------ #
    # Delete
    # ------------------------------------------------------------------ #
    def _delete_selected_object(self) -> None:
        selected = self.object_list.selectedItems()
        if not selected:
            return
        name = selected[0].text()
        idx = self.viewer.get_index(name)
        if idx >= 0:
            self.viewer.remove_object(idx)
        active_name = None
        if 0 <= self.viewer.active_index < len(self.viewer.get_objects()):
            active_name = self.viewer.get_objects()[self.viewer.active_index]["name"]
        self._refresh_object_list(select_name=active_name)

    # ------------------------------------------------------------------ #
    # Cut plane preview
    # ------------------------------------------------------------------ #
    def _toggle_cut_plane_preview(self, checked):
        self.viewer.set_show_cut_plane(checked)

    def _update_cut_plane_position(self, z_value):
        self.viewer.set_cut_plane(z_value)
