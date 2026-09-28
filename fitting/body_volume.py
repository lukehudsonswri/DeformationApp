"""Extract a body region's full torso volume (flesh solid + skin shell,
sharing nodes) plus its "free boundary" -- the artificial cut surface where
the region was isolated from the rest of the body (neck, armholes, waist)
-- for FEBio case assembly (AGENTS.md section 2.5).

Confirmed on real data: every one of F05_Standing's 11,412 torso skin nodes
is also a flesh node (100% shared) -- the skin is a true "wrapped" surface
over the flesh volume, not a separate unconnected shell. This is what makes
a physically consistent skin+flesh case possible: both parts share the same
node array, so the skin deforms together with the flesh underneath it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from config.hbm_models import HbmModel
from config.sites import SiteResolution
from core.lsdyna.bridge import iter_wanted_cards

ProgressFn = Optional[Callable[[str, float], None]]


@dataclass
class TorsoVolume:
    """Combined flesh (solid) + skin (shell) torso, sharing one node array.

    ``skin_quads`` / ``flesh_hexes`` are 0-based indices into ``nodes``
    (not the original LS-DYNA node ids) -- ready to feed directly into a
    FEBio (or any other) mesh writer. ``node_ids`` preserves the *original*
    LS-DYNA node ids in the same order as ``nodes``, so a solved case's
    displacement field (indexed by the FEBio file's own 1-based node
    numbering, ``node_ids[i]`` for local index ``i``, i.e. FEBio id ``i+1``
    -- see ``febio/assemble_case.py``) can be mapped back onto the original
    whole-body HBM node id space for ``main_2.py``'s projection step.
    """

    nodes: np.ndarray  # (N, 3)
    node_ids: np.ndarray  # (N,) int64, original LS-DYNA node ids, same order as nodes
    skin_quads: np.ndarray  # (n_skin, 4) indices into nodes
    flesh_hexes: np.ndarray  # (n_flesh, 8) indices into nodes
    free_boundary_node_ids: np.ndarray  # 0-based indices into nodes: the artificial cut surface


def hex8_boundary_faces(nodes: np.ndarray, hexes: np.ndarray) -> np.ndarray:
    """Boundary quad faces (0-based node indices) of a hex8 volume --
    faces that belong to exactly one element (shared internal faces appear
    twice and are dropped) -- with orientation corrected so each face's
    normal (right-hand rule from its node order) points *outward*, away
    from its owning element's centroid.

    Orientation correction matters for anything feeding FEBio contact
    surfaces (unlike the free-boundary-node lookup in
    ``extract_torso_volume``, which only needs node membership): FEBio's
    sliding-elastic contact uses the surface element's node order to
    determine which side faces the opposing surface, and a hex8 mesh's own
    per-element node ordering isn't guaranteed consistent enough to trust
    the raw face-index pattern alone (unlike ``core.mesh.base``'s
    boundary extraction, which already does exactly this correction for
    the same reason -- this is the hex8-only, quad-preserving analog,
    needed because that extractor always triangulates).
    """
    face_specs = np.array(
        [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]],
        dtype=np.int64,
    )
    n_hex = len(hexes)
    all_faces = hexes[:, face_specs].reshape(-1, 4)  # (6*n_hex, 4)
    owning_elem = np.repeat(np.arange(n_hex), 6)
    elem_centroids = nodes[hexes].mean(axis=1)  # (n_hex, 3)

    sorted_faces = np.sort(all_faces, axis=1)
    _, inverse, counts = np.unique(sorted_faces, axis=0, return_inverse=True, return_counts=True)
    boundary_mask = counts[inverse] == 1
    boundary_faces = all_faces[boundary_mask]
    boundary_owner = owning_elem[boundary_mask]

    p0 = nodes[boundary_faces[:, 0]]
    p1 = nodes[boundary_faces[:, 1]]
    p2 = nodes[boundary_faces[:, 2]]
    raw_normal = np.cross(p1 - p0, p2 - p0)
    face_center = nodes[boundary_faces].mean(axis=1)
    to_elem = elem_centroids[boundary_owner] - face_center
    inward = np.sum(raw_normal * to_elem, axis=1) > 0  # normal points toward own element -> flip

    corrected = boundary_faces.copy()
    corrected[inward] = boundary_faces[inward][:, ::-1]
    return corrected


def extract_torso_volume(model: HbmModel, site: SiteResolution, progress: ProgressFn = None) -> TorsoVolume:
    """Build a ``TorsoVolume`` for ``site`` in ``model``.

    ``progress``, if given, is called as ``progress(message, fraction)``
    with ``fraction`` climbing ``0.0``->``1.0`` across this whole call --
    optional and purely additive (every existing caller that doesn't pass
    it behaves exactly as before). Added specifically because this
    function's ``*NODE`` pass below (parsing every node in the model to
    keep only the torso's ~11k-47k) is this project's single longest
    unbroken blocking stretch when called with no progress reporting at
    all: with no callback firing for 60-90+ seconds, a GUI caller has no
    chance to pump its own event loop in between (see
    ``postprocess.export_state.export_meshes``, the caller that needed
    this), and the OS can report the window as "Not Responding" even
    though nothing has actually frozen.
    """
    shell_pids = set(site.shell_pids)
    solid_pids = set(site.solid_pids)

    quads_raw: list = []
    hexes_raw: list = []
    referenced_ids: set = set()

    if progress:
        progress(f"Reading {model.elements_path.name} (skin shells)...", 0.0)
    shell_count = 0
    for _kw, params in iter_wanted_cards(model.elements_path, ("*ELEMENT_SHELL",)):
        shell_count += 1
        if progress and shell_count % 100000 == 0:
            # Not enough info here for an accurate fraction within this
            # narrow phase (this loop parses every shell in the whole-body
            # deck to filter down to the torso's own pids, with no cheap
            # upfront total) -- just keep pinging at this phase's own
            # start fraction so the callback still fires regularly and
            # keeps the GUI's event loop pumped (see this function's
            # docstring: avoiding long silent stretches is the actual
            # point, a perfectly smooth bar is secondary).
            progress(f"Reading {model.elements_path.name} (skin shells)... ({shell_count:,} seen)", 0.0)
        if "eid" not in params or params.get("pid") not in shell_pids:
            continue
        n = [params.get(f"n{i}") for i in range(1, 5)]
        if any(v is None for v in n):
            continue
        quads_raw.append(n)
        referenced_ids.update(n)

    if progress:
        progress(f"Reading {model.elements_path.name} (flesh solids)...", 0.10)
    solid_count = 0
    for _kw, params in iter_wanted_cards(model.elements_path, ("*ELEMENT_SOLID",)):
        solid_count += 1
        if progress and solid_count % 100000 == 0:
            progress(f"Reading {model.elements_path.name} (flesh solids)... ({solid_count:,} seen)", 0.10)
        if "eid" not in params or params.get("pid") not in solid_pids:
            continue
        n = [params.get(f"n{i}") for i in range(1, 9)]
        if any(v is None for v in n):
            continue
        hexes_raw.append(n)
        referenced_ids.update(n)

    if not quads_raw:
        raise ValueError(f"no *ELEMENT_SHELL cards found for site '{site.site}' shell pids {site.shell_pids}")
    if not hexes_raw:
        raise ValueError(f"no *ELEMENT_SOLID cards found for site '{site.site}' solid pids {site.solid_pids}")

    # A fast raw-line count (no card parsing) purely to get a rough
    # progress denominator for the *NODE pass below -- this project's
    # single slowest stretch here, since it must parse every node in the
    # whole-body deck (1M+ typically) to keep only the ~11k-47k that are
    # actually torso nodes; see the docstring above.
    total_lines = 1
    if progress:
        with open(model.nodes_path, "rb") as count_f:
            total_lines = sum(1 for _ in count_f) or 1
        progress(f"Reading {model.nodes_path.name} (this is the slow part)...", 0.20)

    coords: dict = {}
    lines_seen = 0
    report_every = max(total_lines // 200, 20000)
    for _kw, params in iter_wanted_cards(model.nodes_path, ("*NODE",)):
        lines_seen += 1
        if progress and lines_seen % report_every == 0:
            frac = 0.20 + 0.70 * min(1.0, lines_seen / total_lines)
            progress(f"Reading {model.nodes_path.name}... ({lines_seen:,} rows)", frac)
        nid = params.get("nid")
        if nid in referenced_ids:
            coords[nid] = (params["x"], params["y"], params["z"])

    missing = referenced_ids - coords.keys()
    if missing:
        raise ValueError(f"{len(missing)} referenced node id(s) missing from {model.nodes_path}")

    if progress:
        progress("Building torso volume arrays...", 0.95)

    id_to_index = {nid: i for i, nid in enumerate(coords)}
    nodes = np.array([coords[nid] for nid in coords], dtype=np.float64)
    node_ids = np.array(list(coords.keys()), dtype=np.int64)

    skin_quads = np.array([[id_to_index[nid] for nid in q] for q in quads_raw], dtype=np.int64)
    flesh_hexes = np.array([[id_to_index[nid] for nid in h] for h in hexes_raw], dtype=np.int64)

    skin_node_idx = set(np.unique(skin_quads).tolist())
    flesh_boundary_faces = hex8_boundary_faces(nodes, flesh_hexes)
    flesh_surface_idx = set(np.unique(flesh_boundary_faces).tolist())
    free_boundary_idx = np.array(sorted(flesh_surface_idx - skin_node_idx), dtype=np.int64)

    if progress:
        progress("Torso volume extracted.", 1.0)

    return TorsoVolume(
        nodes=nodes,
        node_ids=node_ids,
        skin_quads=skin_quads,
        flesh_hexes=flesh_hexes,
        free_boundary_node_ids=free_boundary_idx,
    )
