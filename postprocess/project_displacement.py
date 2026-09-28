"""Project a solved case's torso displacement onto the original whole-body
HBM node id space, for the GUI (``postprocess.gui_case``/``app.viewer_app``).

This is the "projection" step: our own pipeline solves directly on the raw
whole-torso HBM mesh (no coarse-to-fine remap needed, unlike the gold
standard's decimate-then-project pipeline in ``Plate_Skin_Deformation``) --
so "projecting" here means mapping the FEBio case's own local, 1-based node
numbering (``fitting.body_volume.TorsoVolume``'s node order: torso nodes
first, ids ``1..n_torso``, plate nodes after) back onto the *original*
LS-DYNA/HBM node ids the GUI's full-body mesh is keyed by. See
``febio/build_preliminary_case.py``'s ``build_case`` for where the sidecar
``<case>_node_map.npz`` (``torso_node_ids``, ``n_torso``) is written
alongside the ``.feb``.

Entry point: ``project_case_all_steps`` -- every solved step, so the GUI's
slider can interpolate between real solved states rather than a synthetic
loading curve.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from postprocess.node_log import NodeLogStep, read_node_log


def _map_step_to_original_ids(step: NodeLogStep, torso_node_ids: np.ndarray, n_torso: int) -> np.ndarray:
    """Displacement (N, 3), ordered to match ``torso_node_ids`` (i.e.
    original HBM node id order), from one parsed ``NodeLogStep``.

    Raises if the step's torso-range ids aren't exactly ``1..n_torso`` with
    no gaps -- the assumption ``assemble_torso_plate_case`` relies on
    (torso nodes enumerated first, contiguously) to make this mapping safe
    without a slower id-to-id lookup.
    """
    if not {"ux", "uy", "uz"}.issubset(step.fields):
        raise ValueError(
            f"node log step (time={step.time}) does not have ux;uy;uz columns "
            f"(has {step.fields}) -- was this case assembled with "
            "febio.assemble_case's default Output/logfile block?"
        )
    u_cols = [step.fields.index(c) for c in ("ux", "uy", "uz")]

    # torso nodes are FEBio ids 1..n_torso (see assemble_case.py); anything
    # beyond that (only present in logs from before the TorsoNodes NodeSet
    # scoping fix) is the rigid plate's own nodes, not part of the HBM body.
    torso_mask = step.node_ids <= n_torso
    torso_local_ids = step.node_ids[torso_mask]
    torso_disp = step.values[torso_mask][:, u_cols]

    order = np.argsort(torso_local_ids)
    torso_local_ids = torso_local_ids[order]
    torso_disp = torso_disp[order]
    if not np.array_equal(torso_local_ids, np.arange(1, n_torso + 1)):
        raise ValueError(
            "node log's torso node ids are not exactly 1..n_torso with no gaps -- "
            "cannot safely map local FEBio ids back to original HBM node ids."
        )
    return torso_disp


@dataclass
class MultiStepProjectedDisplacement:
    node_ids: np.ndarray  # (N,) int64, ORIGINAL HBM node ids, torso only
    times: np.ndarray  # (K,) float64, each solved step's own load-curve time
    displacement: np.ndarray  # (K, N, 3) float64
    source_log: str


def project_case_all_steps(
    node_log_path: Path,
    node_map_path: Path,
) -> MultiStepProjectedDisplacement:
    """Read a solved case's node-displacement logfile and the sidecar
    node-id map written alongside its ``.feb``, and return *every* solved
    step's torso-only displacement field, indexed by *original* HBM node
    ids -- for animating through the real solved states rather than
    approximating intermediate ones with a synthetic curve.
    """
    steps = read_node_log(node_log_path)
    with np.load(node_map_path) as data:
        torso_node_ids = data["torso_node_ids"].astype(np.int64)
        n_torso = int(data["n_torso"])

    times = np.array([s.time for s in steps], dtype=np.float64)
    displacement = np.stack(
        [_map_step_to_original_ids(s, torso_node_ids, n_torso) for s in steps], axis=0
    )

    return MultiStepProjectedDisplacement(
        node_ids=torso_node_ids,
        times=times,
        displacement=displacement,
        source_log=str(node_log_path),
    )
