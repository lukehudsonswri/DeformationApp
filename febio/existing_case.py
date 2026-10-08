"""Adopt a ``.feb`` that was built outside ``main_1.py`` (e.g. a hand-made
strap-fit case) so ``main_2.py`` can post-process it.

``build_case`` normally writes the logfile block and the two sidecars
(``<case>_node_map.npz``, ``<case>_plate_render.npz``) itself. For a ``.feb``
that already exists, ``prepare_existing_case`` does the same job after the
fact:

1. Checks the plate is already at an acceptable standoff from the torso skin
   (``fitting.standoff.check_standoff``). If it is, no CPD/ICP re-fit is done
   -- the plate is left exactly where the file puts it. If it is not, nothing
   is written and ``StandoffNotAcceptableError`` is raised, because moving the
   plate inside a hand-built case would also have to move anything attached
   to it (straps, springs, ...).
2. Confirms the file's first ``n_torso`` nodes really are this HBM model's
   torso (same coordinates, same order) -- the ids ``main_2.py`` maps back to
   the HBM rely on that.
3. Adds, in place (text insertion, so the rest of the file is untouched; the
   original is kept once as ``<name>.feb.bak``):
   - a ``TorsoNodes`` ``<NodeSet>`` (ids ``1..n_torso``),
   - ``<node_data data="x;y;z;ux;uy;uz" file="<case>_node_displacement.txt"
     node_set="TorsoNodes">`` in ``<Output><logfile>``,
   - a ``<rigid_body_data>`` plate log, only if the file has none yet,
   - only if the file has strap meshes (see ``read_strap_geometry``): a
     ``StrapNodes`` ``<NodeSet>`` and a ``<node_data>`` writing
     ``<case>_strap_displacement.txt``.
   Every ``<logfile>`` entry's ``file`` is also flattened to a bare filename
   (``runs/x.txt`` -> ``x.txt``), so all outputs land next to the ``.feb``
   in ``cases_generated/`` and no subfolder has to exist.
4. Writes ``<case>_node_map.npz`` and ``<case>_plate_render.npz`` (the
   latter also carries the strap rest mesh when straps exist).

The plate in such a case is moved by straps/springs, not a prescribed
``rigid_displacement``, so its motion is unknown before the solve. The
sidecar therefore records the plate's pull axis and the name of its
rigid-body log; ``postprocess.gui_case`` reads the plate's real path (and
the straps' displacement) from the logs after the solve, so the GUI shows
both. A file without straps gets no strap entries anywhere.
"""
from __future__ import annotations

import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import List, Optional, Tuple

import numpy as np

from config.hbm_models import discover_hbm_models
from config.sites import resolve_site
from core.mesh.base import ElementType, SurfaceMesh
from febio.assemble_case import _wrapped_id_block
from fitting.body_volume import TorsoVolume, extract_torso_volume, hex8_boundary_faces
from fitting.standoff import check_standoff
from postprocess.plate_from_feb import find_rigid_plate_elements, read_node_coords

ROOT = Path(__file__).resolve().parent.parent

# The .feb prints coordinates to ~7 significant figures, so exact equality
# against the HBM's own coordinates is not expected.
_TORSO_MATCH_TOL_MM = 0.01

# `file="runs/x.txt"` -> `file="x.txt"`: group 1 is `file="`, group 2 the bare name.
_LOGFILE_DIR_PREFIX = re.compile(r'(file=")(?:[^"/\\]*[/\\])+([^"]*)"')


class StandoffNotAcceptableError(ValueError):
    """The plate in the existing ``.feb`` is outside the acceptable standoff band."""


@dataclass
class ExistingCase:
    """What ``read_existing_case`` extracts from a ``.feb`` (ids are 1-based FEBio ids)."""

    plate_node_ids: np.ndarray  # (N,) ascending
    plate_nodes_rest: np.ndarray  # (N, 3)
    plate_elements_local: np.ndarray  # (M, 4 or 8) 0-based into plate_nodes_rest
    concave_node_ids: np.ndarray  # plate nodes on the contact (body-facing) surface
    free_axes: List[int]  # translational axes (0=x,1=y,2=z) the plate may move along
    rigid_id: str  # id of the plate's rigid material
    plate_log_file: Optional[str]  # existing rigid_body_data file for the plate, if any
    final_time: float
    plate_domain_name: str  # name of the plate's <Elements> block


@dataclass
class StrapGeometry:
    """Deformable strap meshes found in a ``.feb`` (``node_ids`` are 1-based FEBio ids)."""

    node_ids: np.ndarray  # (N,) ascending
    nodes_rest: np.ndarray  # (N, 3)
    faces_local: np.ndarray  # (M, 4) 0-based into nodes_rest
    log_ids: np.ndarray  # (N,) id each node carries in FEBio's logfiles, aligned with node_ids


def _surface_node_ids(surface_el: ET.Element) -> np.ndarray:
    ids = set()
    for face_el in surface_el:
        ids.update(int(n) for n in (face_el.text or "").split(",") if n.strip())
    return np.array(sorted(ids), dtype=np.int64)


def _plate_contact_surface(mesh_el: ET.Element, plate_node_ids: np.ndarray, feb_path: Path) -> np.ndarray:
    """Node ids of the contact surface that lies entirely on the plate."""
    surfaces = {s.get("name"): _surface_node_ids(s) for s in mesh_el.findall("Surface")}
    plate_set = set(int(i) for i in plate_node_ids)
    for pair_el in mesh_el.findall("SurfacePair"):
        for role in ("secondary", "primary"):
            role_el = pair_el.find(role)
            ids = surfaces.get((role_el.text or "").strip()) if role_el is not None else None
            if ids is not None and len(ids) and set(int(i) for i in ids) <= plate_set:
                return ids
    raise ValueError(f"{feb_path}: no <SurfacePair> surface lies entirely on the rigid plate's nodes")


def _free_translation_axes(root: ET.Element, rigid_name: str, rigid_id: str) -> List[int]:
    free = [True, True, True]
    rigid_el = root.find("Rigid")
    if rigid_el is not None:
        for bc_el in rigid_el.findall("rigid_bc"):
            rb_el = bc_el.find("rb")
            if rb_el is None or (rb_el.text or "").strip() not in (rigid_name, rigid_id):
                continue
            bc_type = bc_el.get("type")
            for axis, name in enumerate(("x", "y", "z")):
                if bc_type == "rigid_fixed":
                    dof_el = bc_el.find(f"R{name}_dof")
                    if dof_el is not None and (dof_el.text or "").strip() == "1":
                        free[axis] = False
                elif bc_type == "rigid_displacement":
                    dof_el = bc_el.find("dof")
                    if dof_el is not None and (dof_el.text or "").strip().lower() == name:
                        free[axis] = False
    return [a for a in range(3) if free[a]]


def _plate_log_file(root: ET.Element, rigid_id: str) -> Optional[str]:
    logfile_el = root.find("Output/logfile")
    if logfile_el is None:
        return None
    for el in logfile_el.findall("rigid_body_data"):
        ids = [s.strip() for s in (el.text or "").split(",")]
        if rigid_id in ids and el.get("file"):
            return el.get("file")
    return None


def read_existing_case(root: ET.Element, feb_path: Path) -> ExistingCase:
    """Read the rigid plate, its contact surface, its free axes and its log from a parsed ``.feb``.

    Args:
        root: parsed ``.feb`` root element.
        feb_path: path used only in error messages.

    Returns:
        ``ExistingCase`` describing the plate.

    Raises:
        ValueError: if the plate, its coordinates or a plate contact surface cannot be found.
    """
    mesh_el = root.find("Mesh")
    if mesh_el is None:
        raise ValueError(f"{feb_path}: no <Mesh> element found")

    plate_domain_name, elements_global = find_rigid_plate_elements(root, mesh_el, feb_path)
    plate_node_ids, plate_nodes = read_node_coords(mesh_el, np.unique(elements_global), feb_path)
    id_to_local = np.full(int(plate_node_ids.max()) + 1, -1, dtype=np.int64)
    id_to_local[plate_node_ids] = np.arange(len(plate_node_ids), dtype=np.int64)

    rigid_mat_el = next(m for m in root.find("Material").findall("material") if m.get("type") == "rigid body")
    rigid_name, rigid_id = rigid_mat_el.get("name"), rigid_mat_el.get("id")

    control_el = root.find("Control")
    steps_el = control_el.find("time_steps") if control_el is not None else None
    size_el = control_el.find("step_size") if control_el is not None else None
    final_time = float(steps_el.text) * float(size_el.text) if steps_el is not None and size_el is not None else 1.0

    return ExistingCase(
        plate_node_ids=plate_node_ids,
        plate_nodes_rest=plate_nodes,
        plate_elements_local=id_to_local[elements_global],
        concave_node_ids=_plate_contact_surface(mesh_el, plate_node_ids, feb_path),
        free_axes=_free_translation_axes(root, rigid_name, rigid_id),
        rigid_id=rigid_id,
        plate_log_file=_plate_log_file(root, rigid_id),
        final_time=final_time,
        plate_domain_name=plate_domain_name,
    )


def read_strap_geometry(root: ET.Element, feb_path: Path, plate_domain_name: str, n_torso: int) -> Optional[StrapGeometry]:
    """Find the deformable strap meshes, if the file has any.

    A strap is a ``quad4`` ``<Elements>`` block that is neither the plate nor
    part of the torso (all its nodes have ids above ``n_torso``).

    Args:
        root: parsed ``.feb`` root element.
        feb_path: path used only in error messages.
        plate_domain_name: name of the plate's ``<Elements>`` block (excluded).
        n_torso: number of torso nodes (FEBio ids ``1..n_torso``).

    Returns:
        ``StrapGeometry`` for all strap blocks together, or None if there are none.
    """
    mesh_el = root.find("Mesh")
    blocks = []
    for el in mesh_el.findall("Elements"):
        if el.get("name") == plate_domain_name or el.get("type") != "quad4":
            continue
        conn = np.array(
            [[int(n) for n in (e.text or "").split(",")] for e in el.findall("elem")], dtype=np.int64
        )
        if conn.size and int(conn.min()) > n_torso:
            blocks.append(conn)
    if not blocks:
        return None

    conn_global = np.concatenate(blocks, axis=0)
    node_ids, nodes_rest = read_node_coords(mesh_el, np.unique(conn_global), feb_path)

    # FEBio renumbers nodes 1..N in file order when logging, so a file with
    # sparse ids (e.g. straps at 666643+) logs them under their file position.
    position_of = {}
    for nodes_el in mesh_el.findall("Nodes"):
        for node_el in nodes_el.findall("node"):
            position_of[int(node_el.get("id"))] = len(position_of) + 1
    log_ids = np.array([position_of[int(i)] for i in node_ids], dtype=np.int64)

    return StrapGeometry(
        node_ids=node_ids,
        nodes_rest=nodes_rest,
        faces_local=np.searchsorted(node_ids, conn_global),
        log_ids=log_ids,
    )


def _torso_skin_surface(torso: TorsoVolume) -> SurfaceMesh:
    """Skin ``SurfaceMesh`` from the torso volume, triangulated like ``extract_skin_surface``."""
    q = torso.skin_quads
    faces = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]], axis=0)
    surface = SurfaceMesh(nodes=torso.nodes, faces=faces, face_type=ElementType.TRI3, name="torso_skin")
    surface.compute_normals()
    return surface


def _verify_torso_matches(root: ET.Element, feb_path: Path, torso: TorsoVolume) -> None:
    n_torso = len(torso.nodes)
    ids, coords = read_node_coords(root.find("Mesh"), np.arange(1, n_torso + 1), feb_path)
    worst = float(np.abs(coords - torso.nodes).max())
    if worst > _TORSO_MATCH_TOL_MM:
        raise ValueError(
            f"{feb_path}: its nodes 1..{n_torso} differ from the HBM torso by up to {worst:.3f}mm -- "
            "check HBM_MODEL_KEY / SITE, or the file was not built with the torso as nodes 1..n_torso"
        )


def patch_feb_outputs(
    feb_path: Path, n_torso: int, case: ExistingCase, strap_node_ids: Optional[np.ndarray] = None
) -> Tuple[bool, str]:
    """Insert the torso node set and the displacement/plate logs into ``feb_path`` if missing.

    Args:
        feb_path: the ``.feb`` to edit in place (original kept once as ``<name>.feb.bak``).
        n_torso: number of torso nodes (FEBio ids ``1..n_torso``).
        case: result of ``read_existing_case``.
        strap_node_ids: strap node ids to log (``StrapGeometry.node_ids``); None adds nothing strap-related.

    Returns:
        changed: whether the file was rewritten.
        plate_log_file: the ``file`` of the plate's rigid-body log (existing or newly added).
    """
    feb_path = Path(feb_path)
    stem = feb_path.stem
    node_log_name = f"{stem}_node_displacement.txt"
    strap_log_name = f"{stem}_strap_displacement.txt"
    # Logs must land next to the .feb (FEBio does not create subfolders), so any
    # folder prefix on a logfile entry (e.g. "runs/") is dropped below.
    plate_log_file = PurePosixPath(case.plate_log_file.replace("\\", "/")).name if case.plate_log_file else (
        f"{stem}_plate_rigid_body.txt"
    )

    text = feb_path.read_bytes().decode("ISO-8859-1")
    nl = "\r\n" if "\r\n" in text else "\n"
    original = text

    logfile_start, logfile_end = text.find("<logfile>"), text.find("</logfile>")
    if 0 <= logfile_start < logfile_end:
        flattened = _LOGFILE_DIR_PREFIX.sub(r'\1\2"', text[logfile_start:logfile_end])
        text = text[:logfile_start] + flattened + text[logfile_end:]

    def insert_before_line_of(marker: str, block: str) -> None:
        nonlocal text
        idx = text.rindex(marker)
        line_start = text.rfind("\n", 0, idx) + 1
        text = text[:line_start] + block + text[line_start:]

    def add_node_set(name: str, ids) -> None:
        if f'NodeSet name="{name}"' in text:
            return
        wrapped = _wrapped_id_block(ids).replace("\n", nl)
        insert_before_line_of(
            "</Mesh>", nl.join([f'\t\t<NodeSet name="{name}">', f"\t\t\t{wrapped}", "\t\t</NodeSet>"]) + nl
        )

    add_node_set("TorsoNodes", range(1, n_torso + 1))
    if strap_node_ids is not None:
        add_node_set("StrapNodes", [int(i) for i in strap_node_ids])

    new_records = []
    if f'file="{node_log_name}"' not in text:
        new_records.append(
            f'\t\t\t<node_data data="x;y;z;ux;uy;uz" file="{node_log_name}" node_set="TorsoNodes"></node_data>'
        )
    if strap_node_ids is not None and f'file="{strap_log_name}"' not in text:
        new_records.append(
            f'\t\t\t<node_data data="x;y;z;ux;uy;uz" file="{strap_log_name}" node_set="StrapNodes"></node_data>'
        )
    if case.plate_log_file is None:
        new_records.append(
            f'\t\t\t<rigid_body_data data="x;y;z" file="{plate_log_file}">{case.rigid_id}</rigid_body_data>'
        )
    if new_records:
        if "</logfile>" in text:
            insert_before_line_of("</logfile>", nl.join(new_records) + nl)
        elif "</Output>" in text:
            insert_before_line_of("</Output>", nl.join(["\t\t<logfile>", *new_records, "\t\t</logfile>"]) + nl)
        else:
            raise ValueError(f"{feb_path}: no <Output> element to add the logfile to")

    if text == original:
        return False, plate_log_file

    backup = feb_path.with_name(feb_path.name + ".bak")
    if not backup.exists():
        shutil.copy2(feb_path, backup)
    feb_path.write_bytes(text.encode("ISO-8859-1"))
    return True, plate_log_file


def prepare_existing_case(
    feb_path: Path,
    hbm_model_key: str,
    site_name: str,
    ppe_key: str,
    acceptable_standoff_range_mm: Tuple[float, float],
    hbm_root: Optional[Path] = None,
) -> Path:
    """Standoff-check an existing ``.feb``, add the outputs ``main_2.py`` needs, write the sidecars.

    Args:
        feb_path: the existing ``.feb``.
        hbm_model_key: HBM model folder name the case was built on (e.g. ``"F05_Standing"``).
        site_name: body site (e.g. ``"torso"``).
        ppe_key: PPE registry key, recorded in the sidecars for the GUI.
        acceptable_standoff_range_mm: ``(lo, hi)`` band for the plate's minimum gap to the skin.
        hbm_root: HBM folder (default ``<repo>/HBM``).

    Returns:
        ``feb_path``.

    Raises:
        StandoffNotAcceptableError: if the plate is outside the band (nothing is written).
        ValueError: if the file's torso nodes do not match the HBM model.
    """
    feb_path = Path(feb_path)
    if not feb_path.is_file():
        raise FileNotFoundError(f"existing .feb not found: {feb_path}")

    models = discover_hbm_models(hbm_root or (ROOT / "HBM"))
    if hbm_model_key not in models:
        raise KeyError(f"HBM model '{hbm_model_key}' not found -- available: {sorted(models)}")
    model = models[hbm_model_key]
    site = resolve_site(model, site_name)

    print(f"1) reading plate from {feb_path.name}...")
    root = ET.parse(feb_path).getroot()
    case = read_existing_case(root, feb_path)
    print(
        f"   plate: {len(case.plate_node_ids)} nodes, {len(case.plate_elements_local)} elements; "
        f"free translation axes: {[('x', 'y', 'z')[a] for a in case.free_axes] or 'none'}"
    )

    print(f"2) extracting '{hbm_model_key}' {site_name} and confirming it matches the file's torso...")
    torso = extract_torso_volume(model, site)
    _verify_torso_matches(root, feb_path, torso)
    body_surface = _torso_skin_surface(torso)

    print("3) checking plate standoff...")
    concave_rows = np.searchsorted(case.plate_node_ids, case.concave_node_ids)
    check = check_standoff(case.plate_nodes_rest[concave_rows], body_surface, acceptable_standoff_range_mm)
    gap = check.gap
    print(
        f"   min gap {gap.min_mm:.4f}mm (mean {gap.mean_mm:.4f}, max {gap.max_mm:.4f}), "
        f"{gap.n_penetrating} penetrating nodes; allowed min gap {check.min_ok_mm}..{check.max_ok_mm}mm"
    )
    if not check.is_acceptable:
        raise StandoffNotAcceptableError(
            f"plate min gap {gap.min_mm:.4f}mm is outside {check.min_ok_mm}..{check.max_ok_mm}mm. "
            "Reposition the plate in the .feb (this tool does not move parts of a hand-built case), "
            "or widen ACCEPTABLE_STANDOFF_MM."
        )
    print("   acceptable -- skipping CPD/ICP re-fit")

    straps = read_strap_geometry(root, feb_path, case.plate_domain_name, len(torso.nodes))
    print(
        f"   straps: {len(straps.node_ids)} nodes, {len(straps.faces_local)} quads -- will be logged and shown in the GUI"
        if straps is not None
        else "   no strap meshes found -- nothing strap-related is added"
    )

    print("4) adding outputs to the .feb...")
    changed, plate_log_file = patch_feb_outputs(
        feb_path, len(torso.nodes), case, strap_node_ids=None if straps is None else straps.node_ids
    )
    print("   updated (backup: .feb.bak)" if changed else "   already present, left unchanged")

    # Into-body direction = opposite of the skin's outward normal under the plate.
    into_body = -body_surface.face_normals[gap.triangle_ids].mean(axis=0)
    into_body /= np.linalg.norm(into_body)
    if len(case.free_axes) == 1:
        axis = np.zeros(3)
        axis[case.free_axes[0]] = 1.0
        push_direction = axis if np.dot(axis, into_body) >= 0 else -axis
    else:
        push_direction = into_body

    plate_elems = case.plate_elements_local
    boundary_faces = plate_elems if plate_elems.shape[1] == 4 else hex8_boundary_faces(case.plate_nodes_rest, plate_elems)

    node_map_path = feb_path.with_name(feb_path.stem + "_node_map.npz")
    np.savez_compressed(
        node_map_path,
        torso_node_ids=torso.node_ids,
        n_torso=len(torso.nodes),
        hbm_model_key=hbm_model_key,
        site_name=site_name,
        ppe_key=ppe_key,
    )
    strap_arrays = (
        {}
        if straps is None
        else {
            "strap_node_ids": straps.node_ids,
            "strap_log_ids": straps.log_ids,
            "strap_nodes_rest": straps.nodes_rest,
            "strap_faces": straps.faces_local,
            "strap_log_file": f"{feb_path.stem}_strap_displacement.txt",
        }
    )
    plate_render_path = feb_path.with_name(feb_path.stem + "_plate_render.npz")
    np.savez_compressed(
        plate_render_path,
        nodes_rest=case.plate_nodes_rest,
        boundary_faces=boundary_faces,
        push_direction=push_direction,
        total_travel_mm=0.0,  # unknown until solved; main_2 reads it from plate_log_file
        final_time=case.final_time,
        plate_log_file=plate_log_file,
        hbm_model_key=hbm_model_key,
        site_name=site_name,
        ppe_key=ppe_key,
        **strap_arrays,
    )
    print("wrote:", node_map_path)
    print("wrote:", plate_render_path)
    return feb_path
