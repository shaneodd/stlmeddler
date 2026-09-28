"""Modal dialog for creating a primitive shape with appropriate parameters.

Shows only the dimensions relevant to the chosen shape (cube -> size; sphere -> radius;
cylinder/cone -> radius + height). The shape->params mapping is delegated to the pure,
tested :func:`src.mesh_ops.primitive_kwargs`; this module is thin Qt glue.
"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)

from src.mesh_ops import primitive_kwargs


def _spin(value: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(0.01, 1_000_000.0)
    spin.setDecimals(3)
    spin.setSingleStep(0.5)
    spin.setValue(value)
    return spin


class PrimitiveDialog(QDialog):
    """Ask for a primitive shape and its dimensions."""

    SHAPES = ["cube", "sphere", "cylinder", "cone"]
    _FIELDS: dict[str, tuple[str, ...]] = {
        "cube": ("size",),
        "sphere": ("radius",),
        "cylinder": ("radius", "height"),
        "cone": ("radius", "height"),
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Create Primitive")

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.shape_combo = QComboBox()
        self.shape_combo.addItems(self.SHAPES)
        form.addRow("Shape", self.shape_combo)

        self.spin_size = _spin(1.0)
        self.spin_radius = _spin(1.0)
        self.spin_height = _spin(2.0)
        self._rows: dict[str, tuple[QLabel, QDoubleSpinBox]] = {
            "size": (QLabel("Size"), self.spin_size),
            "radius": (QLabel("Radius"), self.spin_radius),
            "height": (QLabel("Height"), self.spin_height),
        }
        for label, field in self._rows.values():
            form.addRow(label, field)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.shape_combo.currentTextChanged.connect(lambda *_: self._update_fields())
        self._update_fields()

    def _update_fields(self) -> None:
        used = set(self._FIELDS.get(self.shape_combo.currentText(), ()))
        for name, (label, field) in self._rows.items():
            visible = name in used
            label.setVisible(visible)
            field.setVisible(visible)

    def values(self) -> tuple[str, dict[str, float]]:
        """Return ``(shape, kwargs)`` for the current selection."""
        shape = self.shape_combo.currentText()
        return shape, primitive_kwargs(
            shape,
            size=self.spin_size.value(),
            radius=self.spin_radius.value(),
            height=self.spin_height.value(),
        )
