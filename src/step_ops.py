"""
STEP (B-rep) export module.

Wraps tessellated meshes (STL/OBJ scans, merged scenes) as faceted B-rep shapes
and writes ISO-10303 STEP files in millimetres for SolidWorks, Fusion 360,
FreeCAD and other STEP-capable CAD applications.

Each triangle becomes one planar B-rep face; adjacent faces are sewn into
shells and closed shells into solids. Open (non-watertight) meshes are
exported as surface shells rather than failing. Decimation is available as a
pre-step because raw scans with hundreds of thousands of triangles produce
STEP files CAD applications cannot open comfortably.

The heavy Open Cascade bindings (cadquery-ocp) are loaded lazily so importing
this module stays cheap and the dependency stays optional.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


@functools.lru_cache(maxsize=1)
def _load_ocp() -> SimpleNamespace:
    """Import the Open Cascade (OCP) modules needed for STEP I/O.

    Cached; raises a clear RuntimeError when cadquery-ocp is not installed.
    """
    try:
        import OCP.Bnd as _bndbox
        import OCP.BRep as _brep
        import OCP.BRepAlgoAPI as _algo
        import OCP.BRepBndLib as _bndlib
        import OCP.BRepBuilderAPI as _builder
        import OCP.BRepCheck as _check
        import OCP.BRepGProp as _gprop
        import OCP.BRepPrimAPI as _prim
        import OCP.gp as _gp
        import OCP.GProp as _gprops
        import OCP.IFSelect as _ifselect
        import OCP.Interface as _interface
        import OCP.STEPControl as _step
        import OCP.TopAbs as _topabs
        import OCP.TopExp as _topexp
        import OCP.TopoDS as _topods
    except ImportError as e:
        raise RuntimeError(
            "cadquery-ocp is required for STEP export. Install it with: "
            "pip install cadquery-ocp"
        ) from e
    return SimpleNamespace(
        brep=_brep,
        algo=_algo,
        bndlib=_bndlib,
        bndbox=_bndbox,
        builder=_builder,
        check=_check,
        gprop=_gprop,
        gprops=_gprops,
        gp=_gp,
        ifselect=_ifselect,
        interface=_interface,
        prim=_prim,
        step=_step,
        topabs=_topabs,
        topexp=_topexp,
        topods=_topods,
    )


@contextlib.contextmanager
def _quiet_native_stdout() -> Iterator[None]:
    """Silence C++-level stdout/stderr chatter emitted by the OCCT STEP writer.

    The statistics the writer prints go to the process file descriptors, so
    Python-level redirect_stdout is not enough; dup2 /dev/null over them for
    the duration of the block and restore afterwards. glibc is flushed before
    the redirect and again before the restore so output buffered in the C++
    runtime lands on /dev/null instead of leaking out after restoration.
    """
    libc = ctypes.CDLL(None)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        out_fd, err_fd = sys.stdout.fileno(), sys.stderr.fileno()
    except (AttributeError, ValueError, OSError):
        os.close(devnull)
        yield
        return
    old_out, old_err = os.dup(out_fd), os.dup(err_fd)
    try:
        libc.fflush(None)
        os.dup2(devnull, out_fd)
        os.dup2(devnull, err_fd)
        yield
    finally:
        libc.fflush(None)
        os.dup2(old_out, out_fd)
        os.dup2(old_err, err_fd)
        os.close(old_out)
        os.close(old_err)
        os.close(devnull)


@dataclass(frozen=True)
class StepExportResult:
    """Summary of a mesh -> STEP export; see :func:`mesh_to_step`."""

    path: str
    triangles_in: int
    faces_written: int
    solids: int
    shells: int
    watertight: bool
    valid: bool
    volume: float
    decimated_from: int | None

    @property
    def summary(self) -> str:
        parts = [f"{self.faces_written} faces"]
        parts.append(f"solid ({self.volume:.2f} mm3)" if self.watertight else "open shell")
        if self.decimated_from is not None:
            parts.append(f"decimated from {self.decimated_from} triangles")
        if not self.valid:
            parts.append("warning: kernel flagged the shape as invalid")
        return ", ".join(parts)


@dataclass(frozen=True)
class StepInfo:
    """Contents of a STEP file as reported by :func:`step_info`."""

    path: str
    solids: int
    shells: int
    faces: int
    volume: float
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]


def _validate_arrays(verts: np.ndarray, faces: np.ndarray) -> None:
    verts = np.asarray(verts)
    faces = np.asarray(faces)
    if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) < 3:
        raise ValueError("verts must be an (N,3) array with N >= 3")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("faces must be a non-empty (M,3) index array")
    if faces.min() < 0 or faces.max() >= len(verts):
        raise ValueError("faces reference vertex indices outside verts")


def _clean_mesh(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weld duplicate vertices, drop degenerate faces and fix winding.

    STL files typically store three unshared vertices per triangle; welding
    first keeps the OCCT sewing step cheap and consistent. Winding is repaired
    best-effort so closed meshes end up outward-facing.
    """
    try:
        import trimesh
    except ImportError as e:  # pragma: no cover - trimesh is a core dependency
        raise RuntimeError("trimesh is required for STEP export") from e

    tri = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces), process=False)
    tri.merge_vertices()
    with contextlib.suppress(Exception):
        trimesh.repair.fix_normals(tri)
    return np.asarray(tri.vertices), np.asarray(tri.faces)


def _decimate(
    verts: np.ndarray, faces: np.ndarray, target_faces: int
) -> tuple[np.ndarray, np.ndarray]:
    """Quadric-decimate a triangle mesh down to ``target_faces`` using Open3D."""
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(verts, dtype=float))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
    mesh.compute_vertex_normals()
    reduced = mesh.simplify_quadric_decimation(
        target_number_of_triangles=max(int(target_faces), 4)
    )
    return np.asarray(reduced.vertices), np.asarray(reduced.triangles)


def _count_subshapes(shape: Any, ocp: SimpleNamespace, shape_type: Any) -> int:
    n = 0
    exp = ocp.topexp.TopExp_Explorer(shape, shape_type)
    while exp.More():
        n += 1
        exp.Next()
    return n


def _shape_volume(shape: Any, ocp: SimpleNamespace) -> float:
    try:
        props = ocp.gprops.GProp_GProps()
        ocp.gprop.BRepGProp.VolumeProperties_s(shape, props)
        return float(props.Mass())
    except Exception:
        return 0.0


def _is_valid(shape: Any, ocp: SimpleNamespace) -> bool:
    """Kernel validity check against top-level solids, else shells, else faces.

    BRepCheck_Analyzer on a bare compound can flag the container itself even
    when every contained shell/face is fine, so the check descends to the
    top-level subshapes and ANDs their results.
    """
    analyzer = ocp.check.BRepCheck_Analyzer
    for shape_type in (
        ocp.topabs.TopAbs_ShapeEnum.TopAbs_SOLID,
        ocp.topabs.TopAbs_ShapeEnum.TopAbs_SHELL,
        ocp.topabs.TopAbs_ShapeEnum.TopAbs_FACE,
    ):
        found = False
        exp = ocp.topexp.TopExp_Explorer(shape, shape_type)
        while exp.More():
            found = True
            if not analyzer(exp.Current()).IsValid():
                return False
            exp.Next()
        if found:
            return True
    return False


def _compound(shapes: list[Any], ocp: SimpleNamespace) -> Any:
    brep_builder = ocp.brep.BRep_Builder()
    compound = ocp.topods.TopoDS_Compound()
    brep_builder.MakeCompound(compound)
    for item in shapes:
        brep_builder.Add(compound, item)
    return compound


def _shell_is_closed(shell: Any, ocp: SimpleNamespace) -> bool:
    """True when every edge of ``shell`` is shared by at least two faces.

    ``BRepBuilderAPI_MakeSolid`` happily "builds" an open shell, and
    ``ShapeAnalysis_Shell.HasFreeEdges`` misses free edges on sewn results, so
    closedness is decided from the edge/face incidence map instead. Faces are
    walked per shell and edges keyed by their rounded endpoint coordinates --
    sewn shared edges are literally the same underlying geometry, so the keys
    match exactly and an edge touched by a single face is a boundary edge.
    """
    topabs = ocp.topabs.TopAbs_ShapeEnum
    counts: dict[tuple[Any, ...], int] = {}
    face_exp = ocp.topexp.TopExp_Explorer(shell, topabs.TopAbs_FACE)
    while face_exp.More():
        face = ocp.topods.TopoDS.Face(face_exp.Current())
        edge_keys: set[tuple[Any, ...]] = set()
        edge_exp = ocp.topexp.TopExp_Explorer(face, topabs.TopAbs_EDGE)
        while edge_exp.More():
            points: list[tuple[float, float, float]] = []
            vert_exp = ocp.topexp.TopExp_Explorer(
                edge_exp.Current(), topabs.TopAbs_VERTEX
            )
            while vert_exp.More():
                p = ocp.brep.BRep_Tool.Pnt_s(
                    ocp.topods.TopoDS.Vertex(vert_exp.Current())
                )
                points.append(
                    (round(p.X(), 6), round(p.Y(), 6), round(p.Z(), 6))
                )
                vert_exp.Next()
            if len(points) == 2:
                edge_keys.add(tuple(sorted([points[0], points[1]])))
            edge_exp.Next()
        for key in edge_keys:
            counts[key] = counts.get(key, 0) + 1
        face_exp.Next()
    return bool(counts) and all(n >= 2 for n in counts.values())


def _build_shape(
    verts: np.ndarray, faces: np.ndarray, tolerance: float, ocp: SimpleNamespace
) -> tuple[Any, int, int, bool]:
    """Convert a cleaned triangle mesh into a TopoDS shape.

    Returns ``(shape, shells, solids, all_closed)``. Closed shells are built
    into solids (compounded when several exist); if no shell closes, the sewn
    shape is returned as-is for surface-body import.
    """
    b = ocp.builder

    vertex_shapes = [
        b.BRepBuilderAPI_MakeVertex(ocp.gp.gp_Pnt(float(x), float(y), float(z))).Vertex()
        for x, y, z in verts
    ]

    sewing = b.BRepBuilderAPI_Sewing(tolerance)
    faces_added = 0
    for a, mid, c in faces:
        try:
            wire = b.BRepBuilderAPI_MakeWire(
                b.BRepBuilderAPI_MakeEdge(vertex_shapes[a], vertex_shapes[mid]).Edge(),
                b.BRepBuilderAPI_MakeEdge(vertex_shapes[mid], vertex_shapes[c]).Edge(),
                b.BRepBuilderAPI_MakeEdge(vertex_shapes[c], vertex_shapes[a]).Edge(),
            ).Wire()
            sewing.Add(b.BRepBuilderAPI_MakeFace(wire, True).Face())
            faces_added += 1
        except Exception:  # skip triangles the kernel refuses (zero-area, coincident pts)
            continue
    if faces_added == 0:
        raise ValueError("No usable faces: every triangle failed B-rep construction")
    sewing.Perform()

    shells: list[Any] = []
    exp = ocp.topexp.TopExp_Explorer(
        sewing.SewedShape(), ocp.topabs.TopAbs_ShapeEnum.TopAbs_SHELL
    )
    while exp.More():
        shells.append(ocp.topods.TopoDS.Shell(exp.Current()))
        exp.Next()

    # Attempt solid construction only on shells that actually close: calling
    # BRepBuilderAPI_MakeSolid on an open shell mutates the shell's type flag
    # into an invalid "solid", which would break downstream CAD import.
    solids: list[Any] = []
    open_shells: list[Any] = []
    for shell in shells:
        if not _shell_is_closed(shell, ocp):
            open_shells.append(shell)
            continue
        maker = b.BRepBuilderAPI_MakeSolid(shell)
        if maker.IsDone():
            solids.append(maker.Solid())
        else:
            open_shells.append(shell)

    all_closed = bool(solids) and not open_shells
    if solids:
        parts: list[Any] = list(solids)
        parts.extend(open_shells)
        if len(parts) == 1:
            return parts[0], len(shells), len(solids), all_closed
        return _compound(parts, ocp), len(shells), len(solids), all_closed

    # Fully open mesh: export the sewn shells (or loose faces) as a surface model.
    if shells:
        return _compound(shells, ocp), len(shells), 0, False
    faces_out = []
    exp = ocp.topexp.TopExp_Explorer(
        sewing.SewedShape(), ocp.topabs.TopAbs_ShapeEnum.TopAbs_FACE
    )
    while exp.More():
        faces_out.append(ocp.topods.TopoDS.Face(exp.Current()))
        exp.Next()
    if not faces_out:
        raise ValueError("Sewing produced no usable shapes")
    return _compound(faces_out, ocp), 0, 0, False


def _write_step(shape: Any, path: str, ocp: SimpleNamespace) -> None:
    ocp.interface.Interface_Static.SetCVal_s("write.step.unit", "MM")
    writer = ocp.step.STEPControl_Writer()
    with _quiet_native_stdout():
        # Transfer prints the statistics chatter; Write prints the filename line.
        writer.Transfer(shape, ocp.step.STEPControl_StepModelType.STEPControl_AsIs)
        status = writer.Write(path)
    if status != ocp.ifselect.IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError(f"STEP write failed (status {status})")


def _read_step(path: str, ocp: SimpleNamespace) -> Any:
    reader = ocp.step.STEPControl_Reader()
    with _quiet_native_stdout():
        status = reader.ReadFile(path)
    if status != ocp.ifselect.IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {path} (status {status})")
    reader.TransferRoots()
    return reader.OneShape()


def mesh_to_step(
    verts: np.ndarray,
    faces: np.ndarray,
    path: str,
    decimate_to: int | None = None,
    tolerance: float = 1e-4,
    verify: bool = True,
) -> StepExportResult:
    """Convert a triangle mesh into a millimetre-unit STEP file.

    Args:
        verts: (N,3) vertex coordinates in millimetres.
        faces: (M,3) triangle indices into ``verts``.
        path: Output ``.step`` / ``.stp`` path.
        decimate_to: When set and the mesh is larger, quadric-decimate to this
            many triangles before conversion (raw scans otherwise produce STEP
            files CAD applications struggle to open).
        tolerance: Edge sewing tolerance in mm; raise it for coarse or noisy
            scans whose welded triangles leave small gaps.
        verify: Re-read the written file and report the shape found there.

    Returns:
        :class:`StepExportResult`; ``watertight=False`` means the mesh did not
        close and was exported as a surface shell (CAD imports it as a surface
        body rather than a solid).

    Raises:
        ValueError: On invalid arrays or a non-STEP output extension.
        RuntimeError: When cadquery-ocp is missing or the write fails.
    """
    _validate_arrays(verts, faces)
    ocp = _load_ocp()
    out = Path(path)
    if out.suffix.lower() not in (".step", ".stp"):
        raise ValueError(f"Output path must end in .step or .stp, got {out.suffix!r}")

    triangles_in = len(faces)
    wverts, wfaces = _clean_mesh(verts, faces)

    decimated_from: int | None = None
    if decimate_to is not None and len(wfaces) > decimate_to:
        decimated_from = len(wfaces)
        wverts, wfaces = _decimate(wverts, wfaces, decimate_to)
        wverts, wfaces = _clean_mesh(wverts, wfaces)

    shape, shells, solids, all_closed = _build_shape(wverts, wfaces, tolerance, ocp)
    valid = _is_valid(shape, ocp)
    volume = _shape_volume(shape, ocp)

    _write_step(shape, str(out), ocp)

    if verify:
        reimported = _read_step(str(out), ocp)
        solids = _count_subshapes(
            reimported, ocp, ocp.topabs.TopAbs_ShapeEnum.TopAbs_SOLID
        )
        shells = _count_subshapes(
            reimported, ocp, ocp.topabs.TopAbs_ShapeEnum.TopAbs_SHELL
        )
        faces_written = _count_subshapes(
            reimported, ocp, ocp.topabs.TopAbs_ShapeEnum.TopAbs_FACE
        )
    else:
        faces_written = len(wfaces)

    watertight = all_closed and solids >= 1 and volume > 0.0
    return StepExportResult(
        path=str(out),
        triangles_in=triangles_in,
        faces_written=faces_written,
        solids=solids,
        shells=shells,
        watertight=watertight,
        valid=valid,
        volume=volume,
        decimated_from=decimated_from,
    )


def step_info(path: str) -> StepInfo:
    """Inspect a STEP file: solid/shell/face counts, volume (mm3) and bounding box."""
    ocp = _load_ocp()
    shape = _read_step(str(path), ocp)

    box = ocp.bndbox.Bnd_Box()
    ocp.bndlib.BRepBndLib.Add_s(shape, box)

    return StepInfo(
        path=str(path),
        solids=_count_subshapes(
            shape, ocp, ocp.topabs.TopAbs_ShapeEnum.TopAbs_SOLID
        ),
        shells=_count_subshapes(
            shape, ocp, ocp.topabs.TopAbs_ShapeEnum.TopAbs_SHELL
        ),
        faces=_count_subshapes(
            shape, ocp, ocp.topabs.TopAbs_ShapeEnum.TopAbs_FACE
        ),
        volume=_shape_volume(shape, ocp),
        bbox_min=(float(box.GetXMin()), float(box.GetYMin()), float(box.GetZMin())),
        bbox_max=(float(box.GetXMax()), float(box.GetYMax()), float(box.GetZMax())),
    )
