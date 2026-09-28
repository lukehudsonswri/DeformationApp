"""
Reader that builds a renderable full-body mesh from this project's own
clean per-model HBM decks (``HBM/<model_key>/{Nodes,Elements}.k`` -- see
``config.hbm_models.discover_hbm_models``), for the GUI (``app.viewer_app``).

**Rewritten from an earlier version that read the old, messy client source
tree directly** (``I-PREDICT_v0.12_90-90-90_*_split.k``, via
``app.client_parser_bridge``) with a hardcoded single skin part id
(``3000451``). That version's node numbering did not match this project's
own pipeline at all: confirmed directly, only ~1,387 of 49,006 target-
region nodes in the old cached mesh matched an id in a freshly projected
displacement field -- coincidental overlap, not real alignment (see
AGENTS.md section 4.5). This version reads from the exact same clean
``HBM/<model_key>/`` folder ``fitting.body_volume`` and the rest of the
pipeline already use, so node ids are guaranteed to align with a
projected displacement field by construction -- no separate lookup or
remapping needed.

Produces:
  * A full-body reference shell mesh (every ``*ELEMENT_SHELL`` in the
    model) purely for visual context.
  * A named target region (any site's skin part ids, e.g.
    ``config.sites.resolve_site(model, "torso").shell_pids``) as its own
    local mesh, so it can be displaced independently and cheaply every
    frame.

Parsing the full model (~220MB Elements.k) takes roughly a minute, so
results are cached to a local ``.npz`` per model key after the first run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from config.hbm_models import HbmModel
from core.lsdyna.bridge import iter_wanted_cards

CACHE_VERSION = 3


@dataclass
class BodyMesh:
    """Full-body reference geometry (context only, never deformed)."""

    node_ids: np.ndarray  # (N,) int64, sorted, ORIGINAL HBM node ids
    xyz: np.ndarray  # (N, 3) float64, rest/undeformed positions
    shell_pid: np.ndarray  # (M,) int64
    shell_conn: np.ndarray  # (M, 4) int64 indices into node_ids/xyz


@dataclass
class TargetRegion:
    """A named site's skin patch, e.g. the torso -- the part that gets
    displaced by a solved case.
    """

    node_ids: np.ndarray  # (K,) int64, sorted, ORIGINAL HBM ids, subset of BodyMesh.node_ids
    body_indices: np.ndarray  # (K,) int64 indices into BodyMesh.xyz for these nodes
    conn_local: np.ndarray  # (P, 4) int64 indices into node_ids/body_indices (local)


def _node_id_to_index_map(node_ids: np.ndarray) -> np.ndarray:
    """Build an array such that ``index_of[node_id] == position in node_ids``.

    Uses a dense lookup array sized to the max node id, which is fine
    here since HBM node ids top out in the tens of millions.
    """
    max_id = int(node_ids.max())
    index_of = np.full(max_id + 1, -1, dtype=np.int64)
    index_of[node_ids] = np.arange(node_ids.shape[0], dtype=np.int64)
    return index_of


def _parse_full_body(model: HbmModel) -> BodyMesh:
    node_ids = []
    xyz = []
    for _keyword, params in iter_wanted_cards(model.nodes_path, ("*NODE",)):
        node_ids.append(params["nid"])
        xyz.append((params["x"], params["y"], params["z"]))

    shell_pid = []
    shell_conn = []
    for _keyword, params in iter_wanted_cards(model.elements_path, ("*ELEMENT_SHELL",)):
        if "eid" not in params:
            # *ELEMENT_SHELL_THICKNESS's second card (t1-t4 thickness values,
            # no connectivity) -- irrelevant for geometry, skip it.
            continue
        shell_pid.append(params["pid"])
        shell_conn.append((params["n1"], params["n2"], params["n3"], params["n4"]))

    node_ids_arr = np.asarray(node_ids, dtype=np.int64)
    xyz_arr = np.asarray(xyz, dtype=np.float64)

    order = np.argsort(node_ids_arr)
    node_ids_arr = node_ids_arr[order]
    xyz_arr = xyz_arr[order]

    index_of = _node_id_to_index_map(node_ids_arr)
    shell_conn_arr = index_of[np.asarray(shell_conn, dtype=np.int64)]

    return BodyMesh(
        node_ids=node_ids_arr,
        xyz=xyz_arr,
        shell_pid=np.asarray(shell_pid, dtype=np.int64),
        shell_conn=shell_conn_arr,
    )


def _cache_path(cache_dir: Path, model_key: str) -> Path:
    return cache_dir / f"hbm_body_cache_{model_key}.npz"


def _save_cache(body: BodyMesh, cache_path: Path) -> None:
    np.savez_compressed(
        cache_path,
        version=CACHE_VERSION,
        node_ids=body.node_ids,
        xyz=body.xyz,
        shell_pid=body.shell_pid,
        shell_conn=body.shell_conn,
    )


def _load_cache(cache_path: Path) -> Optional[BodyMesh]:
    if not cache_path.exists():
        return None
    with np.load(cache_path) as data:
        if int(data["version"]) != CACHE_VERSION:
            return None
        return BodyMesh(
            node_ids=data["node_ids"],
            xyz=data["xyz"],
            shell_pid=data["shell_pid"],
            shell_conn=data["shell_conn"],
        )


def load_body_mesh(
    model: HbmModel, cache_dir: Path, force_reparse: bool = False, progress=None
) -> BodyMesh:
    """Load ``model``'s full-body reference mesh, using a per-model-key
    cache under ``cache_dir`` when possible.
    """
    cache_path = _cache_path(Path(cache_dir), model.key)
    if not force_reparse:
        cached = _load_cache(cache_path)
        if cached is not None:
            return cached

    if progress:
        progress(f"Parsing {model.key}'s HBM deck (first run only)...")
    body = _parse_full_body(model)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _save_cache(body, cache_path)
    return body


def extract_target_region(body: BodyMesh, target_pids: Iterable[int]) -> TargetRegion:
    """Pull out a named site's skin patch (e.g. torso) as its own local
    mesh. ``target_pids`` is a site's ``shell_pids`` tuple (usually one
    part id, but not assumed to be exactly one).
    """
    target_pids = set(target_pids)
    mask = np.isin(body.shell_pid, list(target_pids))
    conn_global_nodes = body.shell_conn[mask]  # indices into body.xyz already

    unique_body_indices, conn_local = np.unique(conn_global_nodes, return_inverse=True)
    conn_local = conn_local.reshape(conn_global_nodes.shape)

    return TargetRegion(
        node_ids=body.node_ids[unique_body_indices],
        body_indices=unique_body_indices,
        conn_local=conn_local,
    )
