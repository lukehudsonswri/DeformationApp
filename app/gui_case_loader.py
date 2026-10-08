"""Loads a ``<case>_gui_case.npz`` (``postprocess.gui_case.build_gui_case``)
and interpolates the torso's real solved displacement steps at an
arbitrary slider time -- used by ``app.viewer_app``.

Interpolating between the real solved steps (rather than approximating
intermediate states with a synthetic loading curve, as an earlier version
of this app did when only one fully-converged state was available) is
possible now because ``main_2.py`` keeps every solved step, not just the
final one -- quasi-static FEBio steps are close enough together that
linear interpolation between adjacent solved states is a reasonable,
literal reading of "what the solver actually computed in between."
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from app.plate_render import StrapRenderData, strap_from_npz


@dataclass
class GuiCase:
    hbm_model_key: str
    site_name: str
    ppe_key: str

    node_ids: np.ndarray  # (N,) int64, original HBM ids
    times: np.ndarray  # (K,) float64, ascending
    displacement: np.ndarray  # (K, N, 3) float64

    plate_nodes_rest: np.ndarray
    plate_boundary_faces: np.ndarray
    push_direction: np.ndarray
    total_travel_mm: float
    final_time: float
    strap: Optional[StrapRenderData] = None  # only for cases whose .feb has strap meshes


def load_gui_case(path: Path) -> GuiCase:
    with np.load(path) as data:
        return GuiCase(
            hbm_model_key=str(data["hbm_model_key"]),
            site_name=str(data["site_name"]),
            ppe_key=str(data["ppe_key"]),
            node_ids=data["node_ids"],
            times=data["times"],
            displacement=data["displacement"],
            plate_nodes_rest=data["plate_nodes_rest"],
            plate_boundary_faces=data["plate_boundary_faces"],
            push_direction=data["push_direction"],
            total_travel_mm=float(data["total_travel_mm"]),
            final_time=float(data["final_time"]),
            strap=strap_from_npz(data),
        )


def displacement_at_time(case: GuiCase, t: float) -> np.ndarray:
    """Torso displacement (N, 3), linearly interpolated between the two
    real solved steps bracketing ``t`` (clamped to the case's own solved
    time range -- never extrapolated).
    """
    t = float(np.clip(t, case.times[0], case.times[-1]))
    # searchsorted gives the insertion index; the bracketing pair is
    # (idx-1, idx) unless t lands exactly on/before the first sample.
    idx = int(np.searchsorted(case.times, t))
    if idx <= 0:
        return case.displacement[0]
    if idx >= len(case.times):
        return case.displacement[-1]

    t0, t1 = case.times[idx - 1], case.times[idx]
    if t1 <= t0:
        return case.displacement[idx]
    frac = (t - t0) / (t1 - t0)
    return case.displacement[idx - 1] * (1.0 - frac) + case.displacement[idx] * frac


def gather_indices_for_target(case_node_ids: np.ndarray, target_node_ids: np.ndarray) -> np.ndarray:
    """Indices into ``case_node_ids`` (and therefore into
    ``GuiCase.displacement``'s node axis) for each id in ``target_node_ids``,
    in ``target_node_ids``'s own order.

    A solved case's own node set (``case_node_ids``) is the *whole* torso
    region (skin + flesh -- ``fitting.body_volume.extract_torso_volume``'s
    combined node array), which is not the same set, or the same order, as
    a rendered target region's own nodes (e.g. skin shell only, in
    ``app.hbm_reader.TargetRegion.node_ids`` order) -- these must be
    explicitly gathered/reordered, not assumed to already align (a real
    bug caught directly: without this, ``case.displacement``'s (47544, 3)
    array was added straight to a (11412, 3) skin-only mesh and raised a
    broadcast error immediately).

    Any target id not present in ``case_node_ids`` maps to ``-1`` (should
    not happen for a skin region, which is always a subset of the torso
    node set by construction, but checked rather than assumed).
    """
    max_id = int(max(case_node_ids.max(), target_node_ids.max()))
    index_of = np.full(max_id + 1, -1, dtype=np.int64)
    index_of[case_node_ids] = np.arange(case_node_ids.shape[0], dtype=np.int64)
    return index_of[target_node_ids]
