"""Export the GUI's *current* view (deformed torso + rigid PPE at whatever
slider time is on screen) back into full LS-DYNA ``.k`` mesh files -- the
"Export Current State" button in ``app/viewer_app.py``.

Two phases, matching the button's own status messages:

1. **"Meshes Exported"** -- ``export_meshes()`` writes three small,
   self-contained ``.k`` files (deformed torso skin, deformed torso flesh,
   PPE at its current position), each with its own local node numbering.
   This is the "export the meshes" half of the request; useful on its own
   if you just want one part re-opened elsewhere.
2. **"FE Model Built with PPE"** -- ``build_fe_model_with_ppe()`` writes
   ONE combined ``.k`` file: the *entire* HBM model (every original node,
   every original element, every other body part completely untouched)
   with the torso's node coordinates patched to the current deformed
   state and the PPE appended as a new part. This is "throw them back
   into the human body model" -- a single, complete, ready-to-open FE
   model, not just the three isolated parts.

Both phases read via ``core.lsdyna.bridge.iter_wanted_cards`` -- the same
parser ``fitting.body_volume``/``app.hbm_reader`` already use for these
exact files -- rather than a hand-rolled line splitter or the generic
``core.mesh`` load/save machinery. **A real bug was caught switching to
this**: an earlier version assumed every ``HBM/<model>/{Nodes.k,Elements.k}``
is in ``config/hbm_normalize.py``'s own simple, always-comma-only written
format (which some models' folders genuinely are), and naively split each
line on commas for a fast byte-preserving copy. This crashed immediately
on ``M50_Standing``, whose folder instead holds a more raw LS-DYNA-style
deck -- ``$`` comments, other card types like
``*ELEMENT_BEAM_ORIENTATION``, different field spacing -- not something a
plain ``line.split(",")`` can safely skip over. ``iter_wanted_cards``
already handles exactly this diversity correctly (it's what the rest of
this project's pipeline already relies on for both model folders); writing
output in one controlled, always-comma format (rather than trying to
preserve each source file's own original formatting byte-for-byte) is what
makes reading robust to it. The generic ``core.mesh`` writer is still
avoided for its own separate reason: it reclassifies elements into named
per-PID parts and doesn't retain original PID values directly, risking
silently losing or renumbering parts across a ~1.79M-element round trip.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from app.gui_case_loader import GuiCase, displacement_at_time, gather_indices_for_target
from app.hbm_reader import BodyMesh
from app.plate_render import PlateRenderData, plate_points_at_time
from config.hbm_models import HbmModel
from config.ppe import get_ppe
from config.sites import SiteResolution
from core.lsdyna.bridge import iter_wanted_cards
from core.mesh.base import ElementType
from fitting.body_volume import extract_torso_volume

ProgressFn = Optional[Callable[[str, float], None]]


def _report(progress: ProgressFn, msg: str, fraction: float) -> None:
    if progress:
        progress(msg, fraction)


def _fmt_float(v: float) -> str:
    # Match config/hbm_normalize.py's own node-coordinate formatting
    # exactly (Python repr, shortest round-trippable form) so a patched
    # (deformed) node's line is textually indistinguishable in style from
    # an untouched one.
    return repr(float(v))


def _write_part_k(
    out_path: Path,
    node_ids: np.ndarray,
    xyz: np.ndarray,
    conn: np.ndarray,
    pid: int,
    element_keyword: str,
) -> None:
    """Write one small, self-contained ``.k`` file: ``*NODE`` (only the
    nodes this part actually references) + one ``*ELEMENT_SHELL`` or
    ``*ELEMENT_SOLID`` block, using the part's own *original* node ids
    (so it lines up with the full model if re-imported) and a single pid.
    ``conn`` is 0-based indices into ``node_ids``/``xyz``.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="ascii") as f:
        f.write("*KEYWORD\n*NODE\n")
        for nid, (x, y, z) in zip(node_ids.tolist(), xyz.tolist()):
            f.write(f"{nid},{_fmt_float(x)},{_fmt_float(y)},{_fmt_float(z)},0,0\n")
        f.write(f"{element_keyword}\n")
        node_ids_list = node_ids.tolist()
        for eid, row in enumerate(conn.tolist(), start=1):
            n_ids = [node_ids_list[i] for i in row]
            f.write(f"{eid},{pid}," + ",".join(str(v) for v in n_ids) + "\n")
        f.write("*END\n")


def export_meshes(
    model: HbmModel,
    site: SiteResolution,
    case: GuiCase,
    plate: PlateRenderData,
    ppe_key: str,
    t: float,
    out_dir: Path,
    progress: ProgressFn = None,
) -> Dict[str, Path]:
    """Phase 1: write ``torso_skin.k``, ``torso_flesh.k``, ``ppe.k`` under
    ``out_dir`` -- the deformed torso skin/flesh (at load-curve time ``t``)
    and the rigid PPE at its corresponding position, each a standalone
    LS-DYNA ``.k`` file.

    ``progress``, if given, is called as ``progress(message, fraction)``
    with ``fraction`` climbing from ``0.0`` to ``1.0`` across THIS
    function's own work only -- the caller (``app.viewer_app``) blends
    this with ``build_fe_model_with_ppe``'s own 0-1 fraction into one
    overall 0-100% bar using known relative stage weights, so neither
    function needs to know about the other's existence.

    Returns a dict of the three written paths, keyed
    ``"torso_skin"``/``"torso_flesh"``/``"ppe"``.
    """
    _report(progress, "Extracting torso skin+flesh geometry...", 0.0)

    def _torso_progress(msg: str, frac: float) -> None:
        # extract_torso_volume's own slow *NODE pass is this function's
        # single biggest stretch -- fold its 0-1 progress into the first
        # 65% of THIS function's own 0-1 range, rather than reporting
        # nothing until it's entirely done (a real "GUI looks frozen /
        # Not Responding" issue this was built specifically to fix).
        _report(progress, msg, frac * 0.65)

    torso = extract_torso_volume(model, site, progress=_torso_progress if progress else None)

    # torso.node_ids is a fresh re-derivation of the exact same node set/
    # order main_1.py used to build this case (see fitting/body_volume.py's
    # module docstring) -- gather explicitly by id rather than assuming
    # order-for-order equality, same defensive pattern as
    # app.gui_case_loader.gather_indices_for_target (a real bug was once
    # caught by NOT assuming two "the same conceptual node set" arrays are
    # already aligned).
    gather_idx = gather_indices_for_target(case.node_ids, torso.node_ids)
    full_disp = displacement_at_time(case, t)
    safe_idx = np.clip(gather_idx, 0, len(full_disp) - 1)
    disp = full_disp[safe_idx].copy()
    disp[gather_idx < 0] = 0.0
    deformed_xyz = torso.nodes + disp

    out_dir = Path(out_dir)
    paths: Dict[str, Path] = {}

    # extract_torso_volume (just completed) is this function's own slow
    # part -- writing the three small files afterward is fast, so most of
    # the 0-1 range is already "spent" by the time we get here.
    _report(progress, "Writing torso_skin.k...", 0.70)
    skin_path = out_dir / "torso_skin.k"
    _write_part_k(
        skin_path, torso.node_ids, deformed_xyz, torso.skin_quads,
        pid=site.shell_pids[0], element_keyword="*ELEMENT_SHELL",
    )
    paths["torso_skin"] = skin_path

    _report(progress, "Writing torso_flesh.k...", 0.85)
    flesh_path = out_dir / "torso_flesh.k"
    _write_part_k(
        flesh_path, torso.node_ids, deformed_xyz, torso.flesh_hexes,
        pid=site.solid_pids[0], element_keyword="*ELEMENT_SOLID",
    )
    paths["torso_flesh"] = flesh_path

    _report(progress, "Writing ppe.k...", 0.95)
    ppe_mesh = get_ppe(ppe_key).load()
    ppe_hexes = ppe_mesh.element_groups[ElementType.HEX8]
    ppe_xyz_now = plate_points_at_time(plate, t)
    ppe_node_ids = np.arange(1, len(ppe_xyz_now) + 1, dtype=np.int64)
    ppe_path = out_dir / "ppe.k"
    _write_part_k(
        ppe_path, ppe_node_ids, ppe_xyz_now, ppe_hexes,
        pid=1, element_keyword="*ELEMENT_SOLID",
    )
    paths["ppe"] = ppe_path

    _report(progress, "Meshes exported.", 1.0)

    return paths


def build_fe_model_with_ppe(
    model: HbmModel,
    body: BodyMesh,
    case: GuiCase,
    plate: PlateRenderData,
    ppe_key: str,
    t: float,
    out_path: Path,
    progress: ProgressFn = None,
) -> Path:
    """Phase 2: write ``out_path`` -- the *entire* HBM model with the
    torso's node coordinates patched to their deformed state at load-curve
    time ``t``, plus the rigid PPE (at its corresponding position) added
    as a brand-new part. Every other node and every original element's
    connectivity (torso's own included -- only coordinates move, never
    connectivity) is carried through unchanged from
    ``model.nodes_path``/``model.elements_path``.

    Reads via ``core.lsdyna.bridge.iter_wanted_cards`` (the same parser
    ``fitting.body_volume``/``app.hbm_reader`` already use for these exact
    files), not a hand-rolled line splitter -- a real bug was caught
    switching away from naive ``line.split(",")`` passthrough: not every
    HBM model's ``Elements.k``/``Nodes.k`` is in
    ``config/hbm_normalize.py``'s own simple comma-only format (that
    module writes a *clean, re-derived* copy -- some models' folders, e.g.
    this project's ``M50_Standing``, contain a more raw LS-DYNA-style deck
    with ``$`` comments, other card types like
    ``*ELEMENT_BEAM_ORIENTATION``, and different field spacing). Writing
    output in one controlled, always-comma format (rather than trying to
    preserve every source file's own original formatting byte-for-byte)
    is what makes this robust across both.

    ``progress``, if given, is called as ``progress(message, fraction)``
    with ``fraction`` climbing from ``0.0`` to ``1.0`` across THIS
    function's own work only -- see ``export_meshes``'s docstring for how
    the caller blends the two functions' independent 0-1 fractions into
    one overall bar.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Displacement at time t, keyed by original HBM node id -- a plain
    # dict lookup is enough here (unlike export_meshes, there is no
    # separate "rest xyz" array to align against: the rest xyz for every
    # node, torso included, comes straight from the source Nodes.k card
    # itself as we stream past it, so all we need per torso node is its
    # (dx, dy, dz)).
    full_disp = displacement_at_time(case, t)
    disp_by_id: Dict[int, np.ndarray] = {
        int(nid): full_disp[i] for i, nid in enumerate(case.node_ids.tolist())
    }

    ppe_mesh = get_ppe(ppe_key).load()
    ppe_hexes = ppe_mesh.element_groups[ElementType.HEX8]
    ppe_xyz_now = plate_points_at_time(plate, t)

    max_existing_node_id = int(body.node_ids.max())
    ppe_node_id_start = max_existing_node_id + 1
    ppe_node_ids = np.arange(ppe_node_id_start, ppe_node_id_start + len(ppe_xyz_now), dtype=np.int64)

    if progress:
        progress(f"Patching {len(disp_by_id)} torso node(s) and streaming {model.nodes_path.name}...", 0.05)

    with open(out_path, "w", encoding="ascii") as dst:
        dst.write("*KEYWORD\n*NODE\n")
        for _kw, params in iter_wanted_cards(model.nodes_path, ("*NODE",)):
            nid = params["nid"]
            x, y, z = params["x"], params["y"], params["z"]
            d = disp_by_id.get(nid)
            if d is not None:
                x, y, z = x + d[0], y + d[1], z + d[2]
            dst.write(f"{nid},{_fmt_float(x)},{_fmt_float(y)},{_fmt_float(z)},0,0\n")

        for nid, (x, y, z) in zip(ppe_node_ids.tolist(), ppe_xyz_now.tolist()):
            dst.write(f"{nid},{_fmt_float(x)},{_fmt_float(y)},{_fmt_float(z)},0,0\n")

        # A fast raw-line count (no card parsing at all -- just counting
        # newlines) to use as a rough progress denominator for the loop
        # below. That loop is this whole function's single biggest, single
        # unbroken stretch of work (90-150+ seconds on the real production
        # decks) -- without SOME progress call partway through it, the
        # caller's progress callback (which is also what pumps the GUI's
        # Qt event loop, via QApplication.processEvents()) never runs for
        # that entire stretch, and the window can be reported
        # "Not Responding" by the OS even though nothing has actually
        # frozen. A byte-exact element count isn't needed here -- only a
        # reasonable, steadily-advancing progress indicator and a
        # regularly-pumped event loop.
        total_lines = 1
        if progress:
            with open(model.elements_path, "rb") as count_f:
                total_lines = sum(1 for _ in count_f) or 1
            progress(f"Reading {model.elements_path.name} (solid + shell blocks, this is the slow part)...", 0.20)

        # One pass over the whole elements file, splitting into two
        # buffers by keyword -- needed regardless of whichever block order
        # the source file actually uses (unlike the F05-only-format
        # assumption this replaced, M50's raw-er deck isn't guaranteed to
        # keep every solid before every shell) -- while tracking the true
        # global max eid/pid so the new PPE part's numbering can never
        # collide with an existing part, no matter which block held the
        # previous largest id (a real bug caught with F05's own data: the
        # shell block contained a higher pid than any solid, so computing
        # "new pid" from the solid block alone collided with it).
        max_eid = 0
        max_pid = 0
        solid_lines: list = []
        shell_lines: list = []
        lines_seen = 0
        report_every = max(total_lines // 200, 20000)  # ~200 updates across the whole read
        for kw, params in iter_wanted_cards(
            model.elements_path, ("*ELEMENT_SOLID", "*ELEMENT_SHELL")
        ):
            lines_seen += 1
            if progress and lines_seen % report_every == 0:
                # 0.20-0.85 of this function's own 0-1 range is "reading
                # the elements file"; total_lines (raw line count) over-
                # estimates the true element-card count (it also counts
                # comments/keyword headers), so this fraction is always a
                # slight underestimate of true progress -- fine for a
                # progress bar, never overshoots past what's actually done.
                frac = 0.20 + 0.65 * min(1.0, lines_seen / total_lines)
                progress(f"Reading {model.elements_path.name}... ({lines_seen:,} rows)", frac)
            if "eid" not in params:
                continue  # e.g. *ELEMENT_SHELL_THICKNESS's second card -- no connectivity
            eid = params["eid"]
            pid = params.get("pid", 0)
            if eid > max_eid:
                max_eid = eid
            if pid > max_pid:
                max_pid = pid
            if kw.startswith("*ELEMENT_SOLID"):
                n = [params.get(f"n{i}") for i in range(1, 9)]
                if any(v is None for v in n):
                    continue
                solid_lines.append(f"{eid},{pid}," + ",".join(str(v) for v in n) + "\n")
            else:
                n = [params.get(f"n{i}") for i in range(1, 5)]
                if any(v is None for v in n):
                    continue
                shell_lines.append(f"{eid},{pid}," + ",".join(str(v) for v in n) + "\n")

        if progress:
            progress(f"Writing {len(solid_lines)} solid + {len(shell_lines)} shell element rows...", 0.90)

        dst.write("*ELEMENT_SOLID\n")
        dst.writelines(solid_lines)

        # New PPE part as its own trailing block within the same
        # *ELEMENT_SOLID keyword -- a new, guaranteed-unique pid
        # (max_pid + 1) and eid range (max_eid + 1 ..).
        new_pid = max_pid + 1
        eid = max_eid
        for row in ppe_hexes.tolist():
            eid += 1
            n_ids = [int(ppe_node_ids[i]) for i in row]
            dst.write(f"{eid},{new_pid}," + ",".join(str(v) for v in n_ids) + "\n")

        dst.write("*ELEMENT_SHELL\n")
        dst.writelines(shell_lines)

        dst.write("*END\n")

    if progress:
        progress("FE model built.", 1.0)

    return out_path
