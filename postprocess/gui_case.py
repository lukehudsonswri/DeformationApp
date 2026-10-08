"""Build and discover the single, self-contained "GUI case" artifact each
solved FEBio case is reduced to for viewing (``main_2.py`` /
``app/viewer_app.py``).

One ``<case_name>_gui_case.npz`` combines everything the GUI needs to
render and animate one HBM-model/PPE pairing, so the GUI never has to
re-derive anything or guess at file naming conventions:

- Every solved step's torso displacement (not just the final one -- see
  ``postprocess.project_displacement.project_case_all_steps``), keyed by
  original HBM node id and each step's own real load-curve time.
- The rigid PPE's rest geometry + everything needed to reconstruct its
  position at any time analytically. Read directly from the **solved**
  ``.feb`` (via ``postprocess.plate_from_feb.read_plate_render_from_feb``)
  whenever it's present, falling back to the pre-solve
  ``<case>_plate_render.npz`` sidecar (``febio.build_preliminary_case.
  build_case``'s own output, from before the case was ever solved) only if
  the ``.feb`` is missing.

  **Why prefer the solved .feb.** Nothing stops a user from opening the
  ``.feb`` main_1.py wrote in FEBio Studio and manually fine-tuning the
  plate's position (or its prescribed rigid displacement) before solving --
  reported directly: a manually-repositioned case rendered with the plate
  back in its stale, pre-adjustment position, a **162mm** discrepancy on
  one real case. Since the plate is a rigid body with translation-only
  DOFs (rotation locked), its solved motion is fully recoverable from the
  solved ``.feb``'s own rest node coordinates + prescribed rigid
  displacement BC/load curve -- see ``postprocess.plate_from_feb``'s module
  docstring for why this is preferred over parsing the binary ``.xplt``
  plot file directly.
- ``hbm_model_key`` / ``site_name`` / ``ppe_key`` metadata, so the GUI can
  populate its dropdowns and load the right full-body mesh, by simply
  *scanning* a directory for ``*_gui_case.npz`` files and reading this
  metadata back out -- no filename-parsing convention to keep in sync.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from app.plate_render import PlateRenderData, StrapRenderData
from postprocess.node_log import read_node_log
from postprocess.plate_from_feb import ExternallyDrivenPlateError, read_plate_render_from_feb
from postprocess.project_displacement import project_case_all_steps
from postprocess.rigid_body_log import read_rigid_body_trajectory


@dataclass
class GuiCaseInfo:
    """Lightweight metadata for one discovered GUI case -- cheap to read
    (no displacement/geometry arrays touched) for populating dropdowns.
    """

    path: Path
    hbm_model_key: str
    site_name: str
    ppe_key: str


def _require_log(path: Path, what: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{what} not found at {path} -- solve the case first (FEBio writes it relative to the .feb's folder)."
        )


def _load_plate_render_sidecar(plate_render_path: Path, cases_dir: Path) -> PlateRenderData:
    with np.load(plate_render_path) as data:
        total_travel_mm = float(data["total_travel_mm"])
        push_direction = data["push_direction"]
        trajectory_times = trajectory_offsets = None
        # Strap/spring-driven plates record their rigid-body log instead of a
        # travel (see febio.existing_case); the real path only exists after the solve.
        plate_log_file = str(data["plate_log_file"]) if "plate_log_file" in data.files else ""
        if plate_log_file:
            plate_log_path = cases_dir / plate_log_file
            _require_log(plate_log_path, "plate rigid-body log")
            trajectory_times, positions = read_rigid_body_trajectory(plate_log_path)
            trajectory_offsets = positions - positions[0]
            total_travel_mm = float(np.dot(trajectory_offsets[-1], push_direction))
        return PlateRenderData(
            nodes_rest=data["nodes_rest"],
            boundary_faces=data["boundary_faces"],
            push_direction=push_direction,
            total_travel_mm=total_travel_mm,
            final_time=float(data["final_time"]),
            trajectory_times=trajectory_times,
            trajectory_offsets=trajectory_offsets,
        )


def _load_strap_sidecar(plate_render_path: Path, cases_dir: Path) -> Optional[StrapRenderData]:
    """Strap meshes + their solved displacement, or None if the case has no straps.

    Straps exist only when ``febio.existing_case`` found strap geometry and
    stored it in the sidecar; any other case returns None.
    """
    if not plate_render_path.is_file():
        return None
    with np.load(plate_render_path) as data:
        if "strap_nodes_rest" not in data.files:
            return None
        strap_log_ids = data["strap_log_ids"]
        nodes_rest = data["strap_nodes_rest"]
        faces = data["strap_faces"]
        strap_log_path = cases_dir / str(data["strap_log_file"])

    _require_log(strap_log_path, "strap displacement log")
    steps = read_node_log(strap_log_path)
    displacement = np.empty((len(steps), len(strap_log_ids), 3), dtype=np.float64)
    # Rows are keyed by the id FEBio logs (see febio.existing_case.StrapGeometry.log_ids),
    # not by file order, so match each step's rows to the sidecar's node order.
    ref_order = np.argsort(strap_log_ids)
    ref_sorted = strap_log_ids[ref_order]
    for k, step in enumerate(steps):
        pos = np.searchsorted(ref_sorted, step.node_ids)
        matches = len(step.node_ids) == len(ref_sorted) and np.array_equal(
            ref_sorted[np.clip(pos, 0, len(ref_sorted) - 1)], step.node_ids
        )
        if not matches:
            raise ValueError(
                f"{strap_log_path}: step {step.step} logs different nodes than the strap set in "
                f"'{plate_render_path.name}' -- re-run main_1.py and re-solve."
            )
        cols = [step.fields.index(c) for c in ("ux", "uy", "uz")]
        displacement[k][ref_order[pos]] = step.values[:, cols]
    return StrapRenderData(
        nodes_rest=nodes_rest,
        faces=faces,
        times=np.array([s.time for s in steps], dtype=np.float64),
        displacement=displacement,
    )


def build_gui_case(case_name: str, cases_dir: Path) -> Path:
    """Assemble ``<cases_dir>/<case_name>_gui_case.npz`` from the three
    files a solved case (``main_1.py`` + a FEBio solve) produces:
    ``<case_name>_node_displacement.txt``, ``<case_name>_node_map.npz``, and
    the rigid PPE's geometry/motion -- read from the solved
    ``<case_name>.feb`` when present (preferred, always reflects the actual
    solved case -- see module docstring), falling back to the pre-solve
    ``<case_name>_plate_render.npz`` sidecar only if the ``.feb`` is missing.

    Returns the written path.
    """
    cases_dir = Path(cases_dir)
    node_log_path = cases_dir / f"{case_name}_node_displacement.txt"
    node_map_path = cases_dir / f"{case_name}_node_map.npz"
    feb_path = cases_dir / f"{case_name}.feb"
    plate_render_path = cases_dir / f"{case_name}_plate_render.npz"

    for p, what in (
        (node_log_path, "node-displacement log"),
        (node_map_path, "node-id map"),
    ):
        if not p.is_file():
            raise FileNotFoundError(
                f"{what} not found at {p} -- solve '{case_name}.feb' in FEBio first "
                "(FEBio Studio, or febio4.exe -i <path>.feb)."
            )

    if feb_path.is_file():
        # Preferred: reflects whatever was ACTUALLY solved, including any
        # manual repositioning done in FEBio Studio before solving -- do
        # NOT silently fall back to the (possibly stale) sidecar if this
        # fails; that would defeat the whole point and mask a real bug.
        try:
            plate = read_plate_render_from_feb(feb_path)
        except ExternallyDrivenPlateError:
            # Strap/spring-driven plate: nothing to reconstruct from the .feb,
            # so use the sidecar main_1.py wrote (rest geometry + pull axis).
            if not plate_render_path.is_file():
                raise FileNotFoundError(
                    f"'{feb_path.name}' drives its plate with straps/springs, so '{plate_render_path.name}' "
                    f"is required in {cases_dir} (written by main_1.py)."
                )
            plate = _load_plate_render_sidecar(plate_render_path, cases_dir)
    elif plate_render_path.is_file():
        plate = _load_plate_render_sidecar(plate_render_path, cases_dir)
    else:
        raise FileNotFoundError(
            f"neither the solved '{feb_path.name}' nor the pre-solve "
            f"'{plate_render_path.name}' sidecar was found in {cases_dir} -- "
            "need at least one to know the PPE's geometry."
        )

    multi_step = project_case_all_steps(node_log_path, node_map_path)
    strap = _load_strap_sidecar(plate_render_path, cases_dir)

    with np.load(node_map_path) as data:
        hbm_model_key = str(data["hbm_model_key"])
        site_name = str(data["site_name"])
        ppe_key = str(data["ppe_key"])

    # Optional blocks are written only when present, so older readers and
    # cases without them see exactly the previous file layout.
    optional = {}
    if plate.trajectory_times is not None:
        optional["plate_traj_times"] = plate.trajectory_times
        optional["plate_traj_offsets"] = plate.trajectory_offsets
    if strap is not None:
        optional["strap_nodes_rest"] = strap.nodes_rest
        optional["strap_faces"] = strap.faces
        optional["strap_times"] = strap.times
        optional["strap_displacement"] = strap.displacement

    out_path = cases_dir / f"{case_name}_gui_case.npz"
    np.savez_compressed(
        out_path,
        # torso, all solved steps
        node_ids=multi_step.node_ids,
        times=multi_step.times,
        displacement=multi_step.displacement,
        # rigid PPE, rest geometry + analytic motion
        plate_nodes_rest=plate.nodes_rest,
        plate_boundary_faces=plate.boundary_faces,
        push_direction=plate.push_direction,
        total_travel_mm=plate.total_travel_mm,
        final_time=plate.final_time,
        # metadata for GUI discovery/dropdowns
        hbm_model_key=hbm_model_key,
        site_name=site_name,
        ppe_key=ppe_key,
        case_name=case_name,
        source_node_log=str(node_log_path),
        **optional,
    )
    return out_path


def discover_gui_cases(cases_dir: Path) -> List[GuiCaseInfo]:
    """Scan ``cases_dir`` for ``*_gui_case.npz`` files and read back just
    their metadata (cheap -- doesn't touch the displacement/geometry
    arrays), for populating the GUI's HBM-model/PPE dropdowns.
    """
    cases_dir = Path(cases_dir)
    results: List[GuiCaseInfo] = []
    if not cases_dir.is_dir():
        return results
    for path in sorted(cases_dir.glob("*_gui_case.npz")):
        with np.load(path) as data:
            results.append(
                GuiCaseInfo(
                    path=path,
                    hbm_model_key=str(data["hbm_model_key"]),
                    site_name=str(data["site_name"]),
                    ppe_key=str(data["ppe_key"]),
                )
            )
    return results
