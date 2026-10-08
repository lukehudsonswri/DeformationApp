"""Build the preliminary F05_Standing + armored-plate FEBio case
(AGENTS.md section 2.5 / VERIFICATION.md).

Usage: run ``main_1.py`` from the repo root (edit its USER SETTINGS block
first), or call ``build_case(...)`` directly / via ``python
febio/build_preliminary_case.py`` for the defaults. Writes
``cases_generated/<model>_<site>_<ppe>_preliminary.feb``.

**Resolved.** Earlier revisions of this pipeline generated a case that
parsed fine in FEBio 4.12 but never converged (residuals stuck at ~1e-20,
"No force acting on the system", "maximum gap: 0" repeating through every
reformation/retry). Extensive isolated testing at the time ruled out
contact type/parameters, mesh scale, single- vs multi-axis rigid
displacement, and shell rotation dofs, and also reproduced a suspicious
"any elastic domain silently stops a rigid body's prescribed displacement"
symptom that was never fully explained.

The actual root cause turned out to be much simpler and is now fixed: the
load curve's step size was derived from ``target_standoff_mm`` (this
project's *seating* precision -- deliberately tiny, ~0.1mm, for accurate
geometric fit), which made the very first solved step move the plate only
~0.1mm -- far too small for the current whole-torso mesh's ~3.5-7mm element
edges to register as engaged contact. The fix (``febio/loadcurve.py``)
decouples this into a separate ``initial_contact_mm`` parameter (default
0.5mm, matching the gold-standard reference case), which requires a real,
mesh-resolvable first step regardless of how tightly the plate is seated.
With this fix, ``build_case(...)``'s own default output solves to NORMAL
TERMINATION with no manual editing -- confirmed by running the generated
``.feb`` through ``febio4.exe`` directly (9 nominal / 13 actual time steps,
final time 1.034, real physical gap values during augmentation). The
"any elastic domain breaks a rigid body" symptom observed during the
original investigation was most likely a test-construction artifact of
those specific minimal repro cases (which also used the too-small step
size) rather than a separate real bug -- it was not encountered again once
the step size was fixed, but was also not deliberately re-isolated, so it
is noted here rather than declared fully explained.

Separately (not a convergence blocker, but affects result quality): the
solved contact pressure/strain is noticeably lower than the gold-standard
reference's own values. This is very likely explained by mesh resolution,
not insufficient travel/indentation: our whole-torso HBM flesh mesh has
~3.5mm through-thickness / ~7.3mm in-plane elements under the contact
patch, vs. the reference's locally-refined mesh at ~0.85mm through-
thickness (down to 0.43mm near the surface) / ~3.7mm in-plane -- roughly
4x finer directly under the indenter. A coarser mesh averages stress/strain
over a much larger volume and will report smaller peak values for the same
underlying physical deformation. The real fix is local graded mesh
refinement under the contact patch (AGENTS.md's still-unbuilt Goal 3), not
more indentation travel -- pushing more displacement through an
under-resolved mesh would not correct the resolution gap itself.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from config.hbm_models import discover_hbm_models
from config.ppe import get_ppe
from config.sites import resolve_site
from core.mesh.base import ElementType
from core.registration.point_to_plane_icp import ICPConfig
from febio.assemble_case import assemble_torso_plate_case
from febio.loadcurve import derive_load_curve
from fitting.body_axes import derive_torso_axes
from fitting.body_volume import extract_torso_volume, hex8_boundary_faces
from fitting.seat_plate import seat_plate

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"


def build_case(
    hbm_model_key: str = "F05_Standing",
    site_name: str = "torso",
    ppe_key: str = "armored_plate",
    target_standoff_mm: float = 0.1,
    indentation_mm: float = 4.25,
    initial_contact_mm: float = 0.5,
    vertical_offset_mm: float = 0.0,
    accept_standoff_range_mm: tuple[float, float] | None = None,
    output_path: Path | None = None,
    hbm_root: Path | None = None,
) -> Path:
    """Seat ``ppe_key`` against ``site_name`` in ``hbm_model_key`` and write
    a runnable (see module docstring for the current known solve issue)
    FEBio case. Returns the written ``.feb`` path.

    This is the function ``main_1.py``'s user-settings block calls -- edit
    the settings there, not this function, for routine use.

    ``initial_contact_mm`` (default 0.5mm) is the first solved step's
    target travel -- deliberately independent of ``target_standoff_mm``
    (the seating gap, typically much smaller). See ``febio.loadcurve``'s
    module docstring for the real bug this decoupling fixes: conflating the
    two shrank the first step to the tiny seating gap, which never
    registered as engaged contact on the current (unrefined, whole-torso)
    mesh.

    ``accept_standoff_range_mm`` is an optional ``(lo, hi)`` band on the
    plate's minimum gap to the skin; if the PPE file already sits inside it
    (and ``vertical_offset_mm`` is 0), the ICP/standoff/symmetry fit is
    skipped and the plate is used as-is.
    """
    models = discover_hbm_models(hbm_root or (ROOT / "HBM"))
    if hbm_model_key not in models:
        raise KeyError(
            f"HBM model '{hbm_model_key}' not found under {hbm_root or (ROOT / 'HBM')} "
            f"-- available: {sorted(models)}"
        )
    model = models[hbm_model_key]
    site = resolve_site(model, site_name)
    ppe_spec = get_ppe(ppe_key)

    print(f"1) deriving '{hbm_model_key}'s own anterior/up/lateral axes from its skeleton landmarks...")
    t0 = time.time()
    axes = derive_torso_axes(model, cache_dir=CACHE_DIR)
    print(
        f"   done in {time.time()-t0:.1f}s: anterior={axes.anterior}, "
        f"up={axes.up}, lateral={axes.lateral}"
    )

    print(f"2) seating '{ppe_key}' against '{site_name}' on '{hbm_model_key}'...")
    t0 = time.time()
    seating = seat_plate(
        model,
        site,
        ppe_spec,
        target_standoff_mm=target_standoff_mm,
        icp_config=ICPConfig(max_iterations=60),
        vertical_offset_mm=vertical_offset_mm,
        accept_standoff_range_mm=accept_standoff_range_mm,
        body_up_axis=axes.up,
        anterior_axis=axes.anterior,
        lateral_axis=axes.lateral,
    )
    if seating.fit_skipped:
        print(
            f"   plate already at an acceptable standoff (min gap {seating.final_gap.min_mm:.4f}mm, "
            f"allowed {accept_standoff_range_mm[0]}..{accept_standoff_range_mm[1]}mm) -- "
            "skipped ICP/standoff/symmetry fit"
        )
    print(f"   done in {time.time()-t0:.1f}s, final gap min={seating.final_gap.min_mm:.4f}mm")

    print("3) extracting torso volume (skin+flesh)...")
    t0 = time.time()
    torso = extract_torso_volume(model, site)
    print(
        f"   done in {time.time()-t0:.1f}s: {len(torso.nodes)} nodes, "
        f"{len(torso.skin_quads)} skin quads, {len(torso.flesh_hexes)} flesh hexes"
    )

    print("4) deriving load curve...")
    lc = derive_load_curve(
        target_standoff_mm=seating.resolved_target_standoff_mm,
        indentation_mm=indentation_mm,
        initial_contact_mm=initial_contact_mm,
    )
    print(f"   total_travel={lc.total_travel_mm:.4f}mm step_size={lc.step_size:.6f} time_steps={lc.time_steps}")

    concave_normal_final = seating.plate_orientation.concave_normal @ seating.seed_rotation @ seating.icp_result.rotation
    concave_normal_final /= np.linalg.norm(concave_normal_final)

    plate_mesh = ppe_spec.load()
    plate_hexes = plate_mesh.element_groups[ElementType.HEX8]

    print("5) assembling .feb...")
    out_path = Path(output_path) if output_path is not None else (
        ROOT / "cases_generated" / f"{hbm_model_key}_{site_name}_{ppe_key}_preliminary.feb"
    )
    assemble_torso_plate_case(
        torso=torso,
        plate_nodes=seating.transformed_plate_nodes,
        plate_hexes=plate_hexes,
        plate_concave_face_node_ids=seating.plate_orientation.concave_face_node_ids,
        load_curve=lc,
        into_body_direction=concave_normal_final,
        out_path=out_path,
    )

    # Sidecar node-id map: TorsoVolume.node_ids gives the *original* HBM node
    # id for each local (0-based) torso node index; assemble_torso_plate_case
    # writes torso nodes as FEBio ids 1..n_torso in that same order (plate
    # nodes come after, at n_torso+1..). main_2.py needs this to map the
    # solved case's own node numbering (and the node_data logfile it wrote,
    # named "<out_path.stem>_node_displacement.txt") back onto the original
    # whole-body HBM node id space for projection.
    node_map_path = out_path.with_name(out_path.stem + "_node_map.npz")
    np.savez_compressed(
        node_map_path,
        torso_node_ids=torso.node_ids,
        n_torso=len(torso.nodes),
        hbm_model_key=hbm_model_key,
        site_name=site_name,
        ppe_key=ppe_key,
    )

    # Sidecar plate-render geometry: the seated (pre-press, t=0) plate node
    # positions + its own boundary quad faces (for a renderable surface,
    # via the same orientation-corrected hex8 boundary extraction used for
    # the FEBio contact surface), plus everything needed to reconstruct the
    # plate's rigid-body position at any load-curve time analytically --
    # no need to log the plate's own nodes in the FEBio solve at all (a
    # rigid body's node-by-node "displacement" isn't meaningful to animate
    # individually; it's fully described by one direction + one distance).
    # main_2.py (postprocess/gui_case.py) reads this to build the GUI's
    # animated plate mesh.
    plate_boundary_faces = hex8_boundary_faces(seating.transformed_plate_nodes, plate_hexes)
    plate_render_path = out_path.with_name(out_path.stem + "_plate_render.npz")
    np.savez_compressed(
        plate_render_path,
        nodes_rest=seating.transformed_plate_nodes,
        boundary_faces=plate_boundary_faces,
        push_direction=concave_normal_final,
        total_travel_mm=lc.total_travel_mm,
        final_time=lc.time_steps * lc.step_size,
        hbm_model_key=hbm_model_key,
        site_name=site_name,
        ppe_key=ppe_key,
    )

    print("wrote:", out_path, " size(MB):", out_path.stat().st_size / 1e6)
    print("wrote:", node_map_path)
    print("wrote:", plate_render_path)
    return out_path


def main() -> None:
    build_case()


if __name__ == "__main__":
    main()
