"""Read the rigid PPE plate's ACTUAL geometry and prescribed motion
directly from a solved ``.feb`` file, instead of the pre-solve analytic
sidecar (``<case>_plate_render.npz``, written by
``febio.build_preliminary_case.build_case`` at *build* time, before the
case is ever solved).

**Why this exists.** ``build_case()`` writes ``plate_render.npz`` from
``fitting.seat_plate``'s own computed seating -- but nothing stops a user
from opening the generated ``.feb`` in FEBio Studio afterward and manually
fine-tuning the plate's position (or its prescribed rigid displacement)
before hitting solve. When that happens, the pre-solve sidecar is stale:
it still reflects the *original* automatic seating, not whatever was
actually solved. Reported directly: a manually-repositioned F05_Seated
case rendered with the plate back in its original (wrong) position in the
GUI, even though the solve itself used the corrected one.

The fix is to always reconstruct the plate's rest geometry and prescribed
motion from the solved ``.feb`` itself -- the exact file FEBio actually
consumed, so it is guaranteed to reflect any manual edit. This is
preferred over parsing the binary ``.xplt`` plot file directly: this
project's installed ``febio-python`` xplt reader cannot reliably read
FEBio 4.12's plot file version (53) -- confirmed directly (see
``febio/assemble_case.py``'s module docstring and this project's own
history): even after patching its version-number gate to accept 53, its
state-section reader silently returns zero states (a real, separate bug --
it reads the ``<HDR_COMPRESSION>`` flag but never actually passes it to its
own decompression step) and, when that is *also* patched around, the
decompression itself fails outright (a different, still-unresolved
format/library issue). None of this matters for the plate specifically:
``febio/assemble_case.py`` prescribes the plate as a **rigid body with
translation-only DOFs** (rotation permanently locked via
``PlateFixRotation``) -- its entire motion at every solved state is fully
determined by the ``.feb``'s own rest node coordinates plus its prescribed
``rigid_displacement`` BC values and referenced load curve, all plain,
well-documented, easy-to-parse XML that this project's own writer
(``febio/assemble_case.py``) fully controls and already documents. Reading
those two pieces directly is exactly as faithful to "what was actually
solved" as reading the binary plot file would be, with none of that
library's risk.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import numpy as np

from app.plate_render import PlateRenderData
from fitting.body_volume import hex8_boundary_faces


def _parse_points(points_el: ET.Element) -> List[Tuple[float, float]]:
    points = []
    for pt in points_el.findall("pt"):
        t_str, v_str = (pt.text or "").split(",")
        points.append((float(t_str), float(v_str)))
    return points


class ExternallyDrivenPlateError(ValueError):
    """The ``.feb``'s rigid plate has no prescribed ``rigid_displacement``
    drive (e.g. it is pulled by straps/springs instead), so its motion
    cannot be reconstructed analytically from the file alone.
    """


def find_rigid_plate_elements(root: ET.Element, mesh_el: ET.Element, feb_path: Path) -> Tuple[str, np.ndarray]:
    """Locate the rigid plate's ``<Elements>`` block and read its connectivity.

    Args:
        root: parsed ``.feb`` root element.
        mesh_el: the ``<Mesh>`` element of ``root``.
        feb_path: path used only in error messages.

    Returns:
        domain_name: name of the plate's domain / ``<Elements>`` block.
        elements: (n_elem, 4 or 8) int array of 1-based FEBio node ids.

    Raises:
        ValueError: if the rigid material, its domain, or its elements are missing.
    """
    # Find the plate's Elements block by its RIGID MATERIAL, not by a
    # hardcoded "Plate" name -- a real wrinkle found testing this directly:
    # opening/re-saving a generated .feb in FEBio Studio renames the
    # domain/Elements block (e.g. "Plate" -> "Part4"), but the *material*
    # (a "rigid body" type material, "PlateRigid" by this project's own
    # default -- see febio.assemble_case._DEFAULT_RIGID_MATERIAL) and its
    # <SolidDomain mat="..."/> reference survive a Studio round-trip
    # unchanged, so this lookup is robust to that renaming.
    material_el = root.find("Material")
    rigid_material_name = None
    if material_el is not None:
        rigid_material_name = next(
            (m.get("name") for m in material_el.findall("material") if m.get("type") == "rigid body"),
            None,
        )
    if rigid_material_name is None:
        raise ValueError(f'{feb_path}: no rigid-body <material type="rigid body"> found')

    mesh_domains_el = root.find("MeshDomains")
    plate_domain_name = None
    if mesh_domains_el is not None:
        for domain_el in list(mesh_domains_el):
            if domain_el.get("mat") == rigid_material_name:
                plate_domain_name = domain_el.get("name")
                break
    if plate_domain_name is None:
        raise ValueError(
            f'{feb_path}: no <MeshDomains> entry references material "{rigid_material_name}"'
        )

    plate_elements_el = next(
        (el for el in mesh_el.findall("Elements") if el.get("name") == plate_domain_name), None
    )
    if plate_elements_el is None:
        raise ValueError(f'{feb_path}: no <Elements name="{plate_domain_name}"> block found')

    elements = np.array(
        [[int(n) for n in (elem_el.text or "").split(",")] for elem_el in plate_elements_el.findall("elem")],
        dtype=np.int64,
    )  # 1-based FEBio node ids
    if elements.size == 0:
        raise ValueError(f'{feb_path}: <Elements name="{plate_domain_name}"> has no <elem> entries')
    return plate_domain_name, elements


def read_node_coords(mesh_el: ET.Element, wanted_ids: np.ndarray, feb_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read the coordinates of ``wanted_ids`` from every ``<Nodes>`` block.

    Args:
        mesh_el: the ``<Mesh>`` element.
        wanted_ids: 1-based FEBio node ids to look up.
        feb_path: path used only in error messages.

    Returns:
        sorted_ids: (N,) ascending node ids.
        coords: (N, 3) coordinates in mm, same order as ``sorted_ids``.

    Raises:
        ValueError: if there is no ``<Nodes>`` block or any id has no coordinates.
    """
    # A generated .feb writes exactly one <Nodes name="AllNodes"> block, but
    # re-saving it in FEBio Studio can split node coordinates across
    # MULTIPLE <Nodes> blocks -- confirmed directly: Studio moved the
    # plate's own nodes into a separate <Nodes name="Detached1"> block after
    # a manual edit (its own auto-organization for a disconnected mesh
    # island). Search every <Nodes> block, not just the first.
    nodes_els = mesh_el.findall("Nodes")
    if not nodes_els:
        raise ValueError(f"{feb_path}: no <Nodes> block found")

    wanted = set(int(x) for x in wanted_ids)
    coords_by_id = {}
    for nodes_el in nodes_els:
        for node_el in nodes_el.findall("node"):
            nid = int(node_el.get("id"))
            if nid in wanted:
                x_str, y_str, z_str = (node_el.text or "").split(",")
                coords_by_id[nid] = (float(x_str), float(y_str), float(z_str))

    missing = wanted - coords_by_id.keys()
    if missing:
        raise ValueError(
            f"{feb_path}: {len(missing)} node id(s) have no matching <node> entry "
            f"across any <Nodes> block, e.g. {sorted(missing)[:5]}"
        )

    sorted_ids = np.array(sorted(coords_by_id), dtype=np.int64)
    return sorted_ids, np.array([coords_by_id[int(nid)] for nid in sorted_ids], dtype=np.float64)


def read_plate_render_from_feb(feb_path: Path) -> PlateRenderData:
    """Reconstruct a ``PlateRenderData`` (rest geometry + prescribed rigid
    motion) directly from a solved ``.feb`` -- see module docstring.

    Raises:
        ValueError: if any expected element (the ``Elements name="Plate"``
            block, its node coordinates, the rigid displacement BCs, or
            their referenced load curve) is missing or malformed -- fails
            loudly rather than silently falling back to a stale/incorrect
            geometry.
    """
    feb_path = Path(feb_path)
    root = ET.parse(feb_path).getroot()

    mesh_el = root.find("Mesh")
    if mesh_el is None:
        raise ValueError(f"{feb_path}: no <Mesh> element found")

    # A plate pulled by straps/springs has no rigid_displacement drive to
    # reconstruct, so say so up front (callers fall back to the sidecar).
    early_rigid_el = root.find("Rigid")
    if early_rigid_el is not None and not any(
        bc.get("type") == "rigid_displacement" for bc in early_rigid_el.findall("rigid_bc")
    ):
        raise ExternallyDrivenPlateError(f"{feb_path}: rigid plate has no prescribed rigid_displacement drive")

    _, plate_hexes_global = find_rigid_plate_elements(root, mesh_el, feb_path)
    sorted_ids, nodes_rest = read_node_coords(mesh_el, np.unique(plate_hexes_global), feb_path)

    # Dense id -> local (0-based) index lookup, e.g. app.hbm_reader's own
    # node_id_to_index_map pattern -- much faster than a per-element Python
    # dict lookup across ~600k+ plate elements.
    id_to_local = np.full(int(sorted_ids.max()) + 1, -1, dtype=np.int64)
    id_to_local[sorted_ids] = np.arange(len(sorted_ids), dtype=np.int64)
    plate_hexes_local = id_to_local[plate_hexes_global]

    boundary_faces = hex8_boundary_faces(nodes_rest, plate_hexes_local)

    # --- rigid body prescribed motion: 3 axis dofs (dv), each pointing at
    # the same load curve (see febio/assemble_case.py's own derivation of
    # push_direction = -dv, total_travel = |load curve's peak value|) ---
    rigid_el = root.find("Rigid")
    if rigid_el is None:
        raise ValueError(f"{feb_path}: no <Rigid> element found")

    dof_index = {"x": 0, "y": 1, "z": 2}
    dv = np.zeros(3)
    found_dofs = set()
    lc_id = None
    for bc_el in rigid_el.findall("rigid_bc"):
        if bc_el.get("type") != "rigid_displacement":
            continue
        dof_el = bc_el.find("dof")
        value_el = bc_el.find("value")
        if dof_el is None or value_el is None or (dof_el.text or "").strip().lower() not in dof_index:
            continue
        dof = (dof_el.text or "").strip().lower()
        dv[dof_index[dof]] = float(value_el.text)
        found_dofs.add(dof)
        if value_el.get("lc") is not None:
            lc_id = value_el.get("lc")

    if found_dofs != set(dof_index):
        raise ValueError(
            f"{feb_path}: expected rigid_displacement bcs for all of x/y/z, found only {sorted(found_dofs)}"
        )
    if lc_id is None:
        raise ValueError(f"{feb_path}: no load curve ('lc' attribute) referenced by the rigid displacement BCs")

    push_direction = -dv
    push_norm = np.linalg.norm(push_direction)
    if push_norm < 1e-12:
        raise ValueError(f"{feb_path}: rigid displacement direction is zero -- plate is not prescribed to move")
    push_direction = push_direction / push_norm

    # --- load curve: the last (highest-time) point's value is the fully-
    # applied displacement at load factor 1.0 -- this pipeline always
    # writes exactly two points, (0,0) and (1,-total_travel_mm) (see
    # febio.loadcurve.LoadCurveSpec.points), but this reads whatever is
    # ACTUALLY in the file, so a manual edit to the curve is honored too ---
    load_data_el = root.find("LoadData")
    points = None
    if load_data_el is not None:
        for lc_el in load_data_el.findall("load_controller"):
            if lc_el.get("id") == lc_id:
                points_el = lc_el.find("points")
                if points_el is not None:
                    points = _parse_points(points_el)
                break
    if not points:
        raise ValueError(f"{feb_path}: load_controller id={lc_id} not found or has no usable <points>")

    total_travel_mm = abs(points[-1][1])

    control_el = root.find("Control")
    time_steps_el = control_el.find("time_steps") if control_el is not None else None
    step_size_el = control_el.find("step_size") if control_el is not None else None
    if time_steps_el is not None and step_size_el is not None:
        final_time = int(time_steps_el.text) * float(step_size_el.text)
    else:
        final_time = float(points[-1][0])

    return PlateRenderData(
        nodes_rest=nodes_rest,
        boundary_faces=boundary_faces,
        push_direction=push_direction,
        total_travel_mm=total_travel_mm,
        final_time=final_time,
    )
