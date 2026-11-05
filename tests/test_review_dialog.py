"""Headless tests for the feature-review dialog (Qt glue, offscreen platform).

Covers table population, diameter override/drop, the inline snap-list editor
(parse/import/export/reset) and the re-detect loop wired into MainWindow.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _hole(raw, snapped, *, through, depth, axis_z=True):
    from src.reconstruct import DetectedHole

    axis = np.array([0.0, 0.0, 1.0]) if axis_z else np.array([1.0, 0.0, 0.0])
    return DetectedHole(
        center=(0.0, 0.0, 0.0),
        axis=tuple(float(v) for v in axis),
        raw_diameter=raw,
        diameter=snapped,
        depth=depth,
        is_through=through,
        concave=True,
        inlier_fraction=0.9,
        arc_deg=355.0,
    )


def _features():
    from src.reconstruct import FeatureSet

    return FeatureSet(
        planes=[],
        holes=[_hole(5.994, 6.0, through=True, depth=10.0),
               _hole(7.991, 8.0, through=False, depth=6.0)],
        bbox_min=(-20.0, -15.0, -5.0),
        bbox_max=(20.0, 15.0, 5.0),
        triangles=200,
    )


def _dialog(qapp, **kwargs):
    from src.gui.review_dialog import FeatureReviewDialog
    from src.reconstruct import DetectOptions, SnapList

    return FeatureReviewDialog(
        _features(),
        kwargs.get("snap", SnapList.default()),
        kwargs.get("opts", DetectOptions(plane_distance=0.2, cylinder_distance=0.2)),
    )


# --------------------------------------------------------------------- #
# Table population / approval
# --------------------------------------------------------------------- #
def test_table_rows_and_combos(qapp):
    dialog = _dialog(qapp)
    assert dialog.table.rowCount() == 2
    assert dialog.table.item(0, 0).text() == "Hole (through)"
    assert dialog.table.item(1, 0).text() == "Hole (blind)"
    assert dialog.table.item(0, 1).text() == "5.994"
    assert dialog._combos[0].currentText() == "6"
    assert dialog._combos[1].currentText() == "8"
    assert "3.2" in [dialog._combos[0].itemText(i) for i in
                     range(dialog._combos[0].count())]
    assert dialog._uses[0].isChecked()


def test_override_and_drop(qapp):
    dialog = _dialog(qapp)
    dialog._combos[0].setCurrentText("5.5")
    dialog._uses[1].setChecked(False)
    result = dialog.feature_set()
    assert len(result.holes) == 1
    assert result.holes[0].diameter == pytest.approx(5.5)


# --------------------------------------------------------------------- #
# Snap-list editor
# --------------------------------------------------------------------- #
def test_snap_edit_updates_list_without_clobbering_text(qapp):
    dialog = _dialog(qapp)
    dialog._combos[0].setCurrentText("5.5")  # hand-typed override
    dialog.snap_edit.setPlainText("6.5\n8.5")
    assert dialog.snap_list().hole_diameters == (6.5, 8.5)
    # dropdown items follow the editor...
    assert "6.5" in [dialog._combos[0].itemText(i) for i in
                     range(dialog._combos[0].count())]
    # ...but typed overrides survive live edits
    assert dialog._combos[0].currentText() == "5.5"


def test_snap_list_validation(qapp):
    dialog = _dialog(qapp)
    dialog.snap_edit.setPlainText("abc")
    with pytest.raises(ValueError):
        dialog.snap_list()
    dialog.snap_edit.setPlainText("")
    with pytest.raises(ValueError, match="at least one"):
        dialog.snap_list()
    dialog.snap_edit.setPlainText("-2")
    with pytest.raises(ValueError, match="positive"):
        dialog.snap_list()


def test_reset_defaults(qapp):
    from src.reconstruct import SnapList

    dialog = _dialog(qapp)
    dialog.snap_edit.setPlainText("1.1")
    dialog._reset_snap()
    assert dialog.snap_list() == SnapList.default()


def test_import_export_json(qapp, tmp_path, monkeypatch):
    from src.reconstruct import SnapList

    path = tmp_path / "my_snap.json"
    SnapList(hole_diameters=(5.9, 8.0), tolerance=0.2).save(path)

    dialog = _dialog(qapp)
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))
    dialog._import_snap()
    assert dialog.snap_list().hole_diameters == (5.9, 8.0)
    assert dialog.snap_tol_spin.value() == pytest.approx(0.2)
    # import re-snaps the whole table through the imported list
    assert dialog._combos[0].currentText() == "5.9"
    assert dialog._combos[1].currentText() == "8"

    out = tmp_path / "out.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "")))
    dialog.snap_edit.setPlainText("3.3\n4.4")
    dialog._export_snap()
    assert SnapList.load(out).hole_diameters == (3.3, 4.4)


# --------------------------------------------------------------------- #
# Tuning + re-detect plumbing
# --------------------------------------------------------------------- #
def test_detect_options_merge(qapp):
    from src.reconstruct import DetectOptions

    dialog = _dialog(qapp)
    dialog.plane_spin.setValue(0.35)
    opts = dialog.detect_options()
    base = DetectOptions()
    assert opts.plane_distance == pytest.approx(0.35)
    assert opts.cylinder_distance == pytest.approx(0.2)
    assert opts.plane_min_seed_faces == base.plane_min_seed_faces


def test_redetect_button_accepts_with_flag(qapp):
    dialog = _dialog(qapp)
    dialog.redetect_button.click()
    assert dialog.redetect_requested is True
    assert dialog.result() == QDialog.Accepted


# --------------------------------------------------------------------- #
# MainWindow loop
# --------------------------------------------------------------------- #
def test_window_redetect_loop_uses_edited_settings(qapp, monkeypatch, tmp_path):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("open3d")
    from src.step_ops import _load_ocp

    try:
        _load_ocp()
    except RuntimeError as e:  # pragma: no cover - env without cadquery-ocp
        pytest.skip(str(e))

    import src.reconstruct as reconstruct

    part = trimesh.creation.box(extents=[40.0, 30.0, 10.0])
    cutter = trimesh.creation.cylinder(radius=3.0, height=40.0, sections=48)
    cutter.apply_translation([10.0, 5.0, 0.0])
    part = part.difference(cutter)
    verts, faces = np.asarray(part.vertices), np.asarray(part.faces)

    calls = []
    real_detect = reconstruct.detect_features

    def spy(verts_, faces_, snap=None, opts=None):
        calls.append((snap, opts))
        return real_detect(verts_, faces_, snap=snap, opts=opts)

    monkeypatch.setattr(reconstruct, "detect_features", spy)

    from PyQt5.QtWidgets import QWidget

    from src.gui.window import MainWindow

    # A bare QWidget acts as the dialog parent; the handler only touches
    # self.viewer and self, so MainWindow.__init__ (GL viewer) is not needed.
    window = QWidget()
    window.viewer = SimpleNamespace(
        active_index=0,
        get_objects=lambda: [{"tris": faces, "name": "part"}],
        transformed_vertices=lambda i: verts,
    )

    seen = []

    def fake_exec(dialog_self):
        seen.append(dialog_self)
        if len(seen) == 1:
            dialog_self.plane_spin.setValue(0.3)
            dialog_self.snap_edit.setPlainText("2.0\n6.25\n8.0")
            dialog_self.redetect_requested = True
        return QDialog.Accepted

    monkeypatch.setattr(
        "src.gui.review_dialog.FeatureReviewDialog.exec_", fake_exec
    )
    out = tmp_path / "recon.step"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), ""))
    )
    infos = []
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda parent, title, text: infos.append(text)),
    )
    monkeypatch.setattr(
        QMessageBox, "critical",
        staticmethod(lambda parent, title, text: (_ for _ in ()).throw(
            AssertionError(text))),
    )

    MainWindow.reconstruct_active_cad(window)

    assert len(calls) == 2
    second_snap, second_opts = calls[1]
    assert second_opts.plane_distance == pytest.approx(0.3)
    assert 6.25 in second_snap.hole_diameters
    assert out.exists()
    assert infos and "STEP written" in infos[0]
