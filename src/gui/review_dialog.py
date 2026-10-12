"""Modal dialog to review detected features before a CAD rebuild.

Lists every detected hole/outer wall with its raw measured diameter and the
nominal it snapped to, and lets the user override the diameter (pick from the
snap list or type a custom value) or drop a detection entirely. The approved
:class:`src.reconstruct.FeatureSet` is returned by :meth:`feature_set`; this
module is thin Qt glue over the pure, tested detection/build modules.

The dialog also carries the two knobs that most often need adjusting on scans:

* detection tolerances (plane / cylinder inlier distances) with a Re-detect
  button that hands the new :class:`src.reconstruct.DetectOptions` back to the
  caller via :attr:`redetect_requested`, and
* the nominal snap list itself - editable inline, one diameter per line, with
  JSON import/export in the :class:`src.reconstruct.SnapList` schema.
"""

from __future__ import annotations

from dataclasses import replace

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from src.reconstruct import DetectOptions, FeatureSet, SnapList


def _parse_diameters(text: str) -> list[float]:
    values = []
    for token in text.replace(",", "\n").splitlines():
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value <= 0.0:
            raise ValueError(f"diameters must be positive, got {value:g}")
        values.append(value)
    if not values:
        raise ValueError("the snap list needs at least one diameter")
    return sorted(values)


class FeatureReviewDialog(QDialog):
    """Confirm/override detected holes and outer walls before building STEP."""

    def __init__(
        self,
        features: FeatureSet,
        snap: SnapList,
        opts: DetectOptions | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Review Detected Features")
        self.setMinimumWidth(620)
        self._features = features
        self._opts = opts or DetectOptions()
        self.redetect_requested = False

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(features.summary()))
        layout.addWidget(
            QLabel("Override a diameter or untick a detection to exclude it:")
        )

        self.table = QTableWidget(len(features.holes), 5)
        self.table.setHorizontalHeaderLabels(
            ["Feature", "Measured Ø", "Diameter (mm)", "Depth", "Build"]
        )
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._combos: list[QComboBox] = []
        self._uses: list[QCheckBox] = []

        for row, hole in enumerate(features.holes):
            if hole.concave:
                name = "Hole (through)" if hole.is_through else "Hole (blind)"
            else:
                name = f"Outer wall ({hole.arc_deg:.0f} deg)"

            text = QTableWidgetItem(name)
            text.setFlags(text.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, text)

            measured = QTableWidgetItem(f"{hole.raw_diameter:.3f}")
            measured.setFlags(measured.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 1, measured)

            combo = QComboBox()
            combo.setEditable(True)
            combo.setInsertPolicy(QComboBox.NoInsert)
            combo.addItems([f"{d:g}" for d in snap.hole_diameters])
            combo.setCurrentText(
                f"{hole.diameter:g}"
                if hole.diameter is not None
                else f"{hole.raw_diameter:.3f}"
            )
            combo.setValidator(QDoubleValidator(0.01, 10000.0, 4))
            self.table.setCellWidget(row, 2, combo)
            self._combos.append(combo)

            depth = "through" if hole.is_through else f"{hole.depth:.2f}"
            depth_item = QTableWidgetItem(depth)
            depth_item.setFlags(depth_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 3, depth_item)

            use = QCheckBox()
            # Partial outer-wall arcs are not rebuilt as cylinders by default.
            use.setChecked(hole.concave or hole.arc_deg >= 270.0)
            self.table.setCellWidget(row, 4, use)
            self._uses.append(use)

        layout.addWidget(self.table)

        layout.addWidget(self._build_tuning_box())
        layout.addWidget(self._build_snap_box())

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load_snap_into_editor(snap)

    # ------------------------------------------------------------------ #
    # Tuning / snap-list editors
    # ------------------------------------------------------------------ #
    def _build_tuning_box(self) -> QGroupBox:
        box = QGroupBox("Detection tolerances (mm)")
        grid = QGridLayout(box)
        grid.addWidget(QLabel("Plane distance:"), 0, 0)
        self.plane_spin = QDoubleSpinBox()
        self.plane_spin.setRange(0.01, 5.0)
        self.plane_spin.setSingleStep(0.05)
        self.plane_spin.setDecimals(3)
        self.plane_spin.setValue(self._opts.plane_distance)
        grid.addWidget(self.plane_spin, 0, 1)
        grid.addWidget(QLabel("Cylinder distance:"), 0, 2)
        self.cyl_spin = QDoubleSpinBox()
        self.cyl_spin.setRange(0.01, 5.0)
        self.cyl_spin.setSingleStep(0.05)
        self.cyl_spin.setDecimals(3)
        self.cyl_spin.setValue(self._opts.cylinder_distance)
        grid.addWidget(self.cyl_spin, 0, 3)
        self.redetect_button = QPushButton("Re-detect...")
        self.redetect_button.setToolTip(
            "Re-run feature detection on the mesh with these tolerances"
        )
        self.redetect_button.clicked.connect(self._request_redetect)
        grid.addWidget(self.redetect_button, 0, 4)
        return box

    def _build_snap_box(self) -> QGroupBox:
        box = QGroupBox("Nominal snap list (one diameter per mm)")
        box_layout = QHBoxLayout(box)
        self.snap_edit = QPlainTextEdit()
        self.snap_edit.setFixedHeight(64)
        self.snap_edit.setPlaceholderText("e.g. 3.2, 4.2, 5.0, 6.0, 8.0")
        self.snap_edit.textChanged.connect(self._on_snap_edited)
        box_layout.addWidget(self.snap_edit, stretch=1)

        side = QVBoxLayout()
        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("Snap tol:"))
        self.snap_tol_spin = QDoubleSpinBox()
        self.snap_tol_spin.setRange(0.001, 5.0)
        self.snap_tol_spin.setSingleStep(0.05)
        self.snap_tol_spin.setDecimals(3)
        self.snap_tol_spin.valueChanged.connect(self._on_snap_edited)
        tol_row.addWidget(self.snap_tol_spin)
        side.addLayout(tol_row)
        for label, slot in (
            ("Import JSON...", self._import_snap),
            ("Export JSON...", self._export_snap),
            ("Reset defaults", self._reset_snap),
        ):
            button = QPushButton(label)
            button.clicked.connect(slot)
            side.addWidget(button)
        side.addStretch(1)
        box_layout.addLayout(side)
        return box

    def _request_redetect(self) -> None:
        self.redetect_requested = True
        self.accept()

    def _on_snap_edited(self) -> None:
        try:
            snap = self.snap_list()
        except ValueError:
            return  # mid-edit; keep the current table until the text is valid
        self._refresh_combos(snap, resnap=False)

    def _refresh_combos(self, snap: SnapList, resnap: bool) -> None:
        """Reload the dropdown nominals; optionally re-snap every row.

        Live editor edits only refresh the dropdown items so hand-typed
        overrides survive; explicit actions (initial open, import, reset)
        re-snap the whole table.
        """
        for row, hole in enumerate(self._features.holes):
            combo = self._combos[row]
            text = combo.currentText()
            if resnap:
                snapped, _ = snap.snap_diameter(hole.raw_diameter)
                text = (
                    f"{snapped:g}"
                    if snapped is not None
                    else f"{hole.raw_diameter:.3f}"
                )
            combo.blockSignals(True)
            combo.clear()
            combo.addItems([f"{d:g}" for d in snap.hole_diameters])
            combo.setCurrentText(text)
            combo.blockSignals(False)

    def _import_snap(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Snap List", "", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            snap = SnapList.load(path)
        except Exception as e:
            QMessageBox.warning(self, "Import failed", str(e))
            return
        self._load_snap_into_editor(snap)

    def _export_snap(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Snap List", "snap_list.json", "JSON Files (*.json)"
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            self.snap_list().save(path)
        except Exception as e:
            QMessageBox.warning(self, "Export failed", str(e))

    def _reset_snap(self) -> None:
        self._load_snap_into_editor(SnapList.default())

    def _load_snap_into_editor(self, snap: SnapList) -> None:
        self.snap_edit.blockSignals(True)
        self.snap_edit.setPlainText(
            "\n".join(f"{d:g}" for d in snap.hole_diameters)
        )
        self.snap_edit.blockSignals(False)
        self.snap_tol_spin.blockSignals(True)
        self.snap_tol_spin.setValue(snap.tolerance)
        self.snap_tol_spin.blockSignals(False)
        self._refresh_combos(snap, resnap=True)

    # ------------------------------------------------------------------ #
    # Results
    # ------------------------------------------------------------------ #
    def detect_options(self) -> DetectOptions:
        """Current tuning values merged into the options the dialog opened with."""
        return replace(
            self._opts,
            plane_distance=float(self.plane_spin.value()),
            cylinder_distance=float(self.cyl_spin.value()),
        )

    def snap_list(self) -> SnapList:
        """Snap list as currently typed in the editor.

        Raises:
            ValueError: On unparsable, non-positive or empty diameters.
        """
        return SnapList(
            hole_diameters=tuple(_parse_diameters(self.snap_edit.toPlainText())),
            tolerance=float(self.snap_tol_spin.value()),
        )

    def feature_set(self) -> FeatureSet:
        """Approved features: overridden diameters applied, unticked dropped."""
        holes = []
        for row, hole in enumerate(self._features.holes):
            if not self._uses[row].isChecked():
                continue
            try:
                diameter = float(self._combos[row].currentText())
            except ValueError:
                diameter = hole.diameter
            if diameter <= 0.0:
                diameter = hole.diameter
            holes.append(replace(hole, diameter=diameter))
        return replace(self._features, holes=holes)
