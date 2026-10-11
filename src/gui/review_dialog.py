"""Modal dialog to review detected features before a CAD rebuild.

Lists every detected hole/outer wall with its raw measured diameter and the
nominal it snapped to, and lets the user override the diameter (pick from the
snap list or type a custom value) or drop a detection entirely. The approved
:class:`src.reconstruct.FeatureSet` is returned by :meth:`feature_set`; this
module is thin Qt glue over the pure, tested detection/build modules.
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
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from src.reconstruct import FeatureSet, SnapList


class FeatureReviewDialog(QDialog):
    """Confirm/override detected holes and outer walls before building STEP."""

    def __init__(self, features: FeatureSet, snap: SnapList, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Review Detected Features")
        self.setMinimumWidth(560)
        self._features = features

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

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

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
