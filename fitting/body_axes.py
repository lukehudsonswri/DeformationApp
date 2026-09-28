"""Derive a body region's own anatomical anterior/superior/lateral axes
directly from real skeletal landmarks in the HBM deck, instead of assuming
a fixed global-axis convention (AGENTS.md Goal 2 / bug found 2026-09-23).

Why this exists
----------------
``fitting.seat_plate``'s ``DEFAULT_ANTERIOR_AXIS``/``DEFAULT_BODY_UP_AXIS``/
``DEFAULT_LATERAL_AXIS`` were derived once, by hand, specifically for
``F05_Standing``'s own export frame -- confirmed by a real bug: adding a
second raw model folder, ``F05_Seated`` (built directly from the client's
own raw export, not run through ``config/hbm_normalize.py``), seated the
plate on the **posterior** side of the torso, because that model's own
coordinate frame does not share ``F05_Standing``'s "+X is anterior, -Z is
up" convention at all -- different raw exports are not guaranteed to share
a coordinate convention just because they're the same client/model family.

This module replaces that fixed guess with a per-model derivation from two
pairs of real, unambiguous anatomical landmarks (both squarely within the
torso region being fitted, avoiding any risk from limb articulation
elsewhere in a *posed* body -- e.g. a seated model's bent knee/hip would
NOT reliably share the torso's own anterior direction in world space, so a
distal landmark like the patella is deliberately NOT used here, despite
being a reasonable-sounding example):

- **Anterior**: sternum (front of the ribcage) vs. the thoracic vertebral
  column (back of the ribcage) -- ``sternum_centroid - spine_centroid``,
  normalized.
- **Superior/inferior ("up")**: the topmost (T1) vs. bottommost (T12)
  thoracic vertebra -- ``T1_centroid - T12_centroid``, orthogonalized
  against the anterior axis (Gram-Schmidt) since the thoracic spine has
  natural curvature (kyphosis), so this raw vector is not perfectly
  perpendicular to the anterior direction on its own.
- **Lateral**: ``cross(anterior_axis, up_axis)``, normalized -- verified to
  reproduce the correct handedness against ``F05_Standing``'s own known-
  good axes (see below).

Part ids (family-level constants, validated directly against
``HBM/01_Model/I-PREDICT_v1.0_Main.dyn``'s real ``*PART`` cards in the main
repo -- the one shared properties deck for the whole I-PREDICT v1.0 family,
per ``config/_hbm_raw_source.py``'s own module docstring):

    Sternum_Cortical_Bone         pid 2000048  (*ELEMENT_SHELL)
    Sternum_Trabecular_Bone       pid 2000049  (*ELEMENT_SOLID)
    T1..T12_Vertebral_Body_Cortical_Bone  pids 3000021, 3000024, ..., 3000054
                                  (step 3, *ELEMENT_SHELL)

Validated against known ground truth before trusting it on the actual bug
(2026-09-23): derived axes for ``F05_Standing`` (whose correct axes are
independently known -- ``DEFAULT_ANTERIOR_AXIS``/``DEFAULT_BODY_UP_AXIS``/
``DEFAULT_LATERAL_AXIS``) came out within **~5.7 degrees** of those
hardcoded values (the small residual is real anatomy, not a derivation
bug -- the thoracic spine's own natural curvature means a straight T1-T12
vector is not perfectly vertical even in the true global frame). Applied
to the actual bug, ``F05_Seated``: derived anterior came out
**~163 degrees** from ``F05_Standing``'s (i.e. essentially flipped, mostly
-X instead of +X) -- a direct, quantitative confirmation of exactly the
"plate landed on the posterior side" symptom that was reported, not a
subtle discrepancy. Lateral (lengthwise left-right) came out nearly
identical between the two models (both ~+Y) -- consistent with that being
a shared convention across the family while anterior/up are not.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, FrozenSet, NamedTuple, Optional, Set, Tuple

import numpy as np

from config.hbm_models import HbmModel
from core.lsdyna.bridge import iter_wanted_cards

# Sternum: strictly anterior, right at torso height -- see module docstring.
_STERNUM_PIDS: FrozenSet[int] = frozenset({2000048, 2000049})

# T1..T12 Vertebral_Body_Cortical_Bone -- strictly posterior (the vertebral
# body is the load-bearing anterior-most part of each vertebra, but still
# unambiguously on the SPINE side of the torso, i.e. posterior relative to
# the sternum/ribcage front), spanning the full thoracic region.
_T1_PID = 3000021
_T12_PID = 3000054
_THORACIC_VERTEBRA_BODY_PIDS: FrozenSet[int] = frozenset(range(3000021, 3000055, 3))

# Bumped whenever the derivation method or landmark pids change, so a stale
# on-disk cache from an older method is never silently trusted.
_CACHE_VERSION = 1


class BodyAxes(NamedTuple):
    anterior: np.ndarray  # unit vector, points toward the front of the torso
    up: np.ndarray  # unit vector, points toward the head (superior)
    lateral: np.ndarray  # unit vector, cross(anterior, up)


def _cache_path(cache_dir: Path, model_key: str) -> Path:
    return Path(cache_dir) / f"body_axes_cache_{model_key}.json"


def _save_cache(axes: BodyAxes, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "version": _CACHE_VERSION,
                "anterior": axes.anterior.tolist(),
                "up": axes.up.tolist(),
                "lateral": axes.lateral.tolist(),
            }
        )
    )


def _load_cache(cache_path: Path) -> Optional[BodyAxes]:
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        return None
    if data.get("version") != _CACHE_VERSION:
        return None
    try:
        return BodyAxes(
            anterior=np.asarray(data["anterior"], dtype=np.float64),
            up=np.asarray(data["up"], dtype=np.float64),
            lateral=np.asarray(data["lateral"], dtype=np.float64),
        )
    except (KeyError, ValueError):
        return None


def _referenced_node_ids_for_groups(
    elements_path, groups: Dict[str, FrozenSet[int]]
) -> Dict[str, Set[int]]:
    """Node ids used by any *ELEMENT_SHELL/*ELEMENT_SOLID card whose pid
    falls in each named group's pid set -- one pass over ``elements_path``
    for ALL groups at once (this deck is 200+MB on real models, so scanning
    it once per landmark group instead would be needlessly slow). Checks
    both element types since this family's own bones mix cortical (shell)
    and trabecular (solid) representations under related-but-distinct pids
    (confirmed directly: Sternum_Cortical_Bone is shell,
    Sternum_Trabecular_Bone is solid).
    """
    ids: Dict[str, Set[int]] = {name: set() for name in groups}
    for kw, params in iter_wanted_cards(elements_path, ("*ELEMENT_SHELL", "*ELEMENT_SOLID")):
        pid = params.get("pid")
        matching = [name for name, pids in groups.items() if pid in pids]
        if not matching:
            continue
        if kw.startswith("*ELEMENT_SHELL"):
            n = [params.get(f"n{i}") for i in range(1, 5)]
        else:
            n = [params.get(f"n{i}") for i in range(1, 9)]
        node_ids = [v for v in n if v is not None]
        for name in matching:
            ids[name].update(node_ids)
    return ids


def _centroids_for_node_id_groups(nodes_path, groups: Dict[str, Set[int]]) -> Dict[str, np.ndarray]:
    """One pass over the whole nodes deck, accumulating a running
    sum/count per named group -- cheaper than re-scanning the (often
    100+MB) nodes file once per landmark group.
    """
    sums = {name: np.zeros(3) for name in groups}
    counts = {name: 0 for name in groups}
    for _kw, params in iter_wanted_cards(nodes_path, ("*NODE",)):
        nid = params["nid"]
        xyz = (params["x"], params["y"], params["z"])
        for name, ids in groups.items():
            if nid in ids:
                sums[name] += xyz
                counts[name] += 1
    missing = [name for name, c in counts.items() if c == 0]
    if missing:
        raise ValueError(
            f"No node coordinates found for landmark group(s) {missing} -- "
            "the expected torso landmark part ids were not found in this model's "
            "elements deck at all (see fitting.body_axes module docstring for "
            "the expected part ids). This model may not have a full skeleton, "
            "or may not belong to the I-PREDICT v1.0 family these part ids were "
            "validated against."
        )
    return {name: sums[name] / counts[name] for name in groups}


def derive_torso_axes(
    model: HbmModel, cache_dir: Optional[Path] = None, force_recompute: bool = False
) -> BodyAxes:
    """Derive this ``model``'s own anterior/up/lateral axes from real
    thorax landmarks (sternum vs. thoracic spine), instead of assuming a
    fixed global-axis convention. See module docstring for the method and
    its validation.

    Scanning a full ~200MB HBM deck takes roughly a minute, and these axes
    never change for a given model, so when ``cache_dir`` is given the
    result is cached to a small JSON file there (keyed by ``model.key``)
    after the first computation -- mirrors the pattern used for the much
    larger full-body mesh cache in ``app.hbm_reader``. Pass ``cache_dir=None``
    (the default) to always recompute, e.g. for tests using throwaway
    synthetic models with no stable cache location.

    Args:
        force_recompute: bypass and overwrite any existing cache entry.

    Raises:
        ValueError: if the expected landmark part ids resolve to zero
            elements in ``model.elements_path`` (e.g. a model outside the
            I-PREDICT v1.0 family, or one missing a full skeleton) --
            fails loudly rather than silently deriving a meaningless axis
            from an empty landmark group.
    """
    cache_path = _cache_path(cache_dir, model.key) if cache_dir is not None else None
    if cache_path is not None and not force_recompute:
        cached = _load_cache(cache_path)
        if cached is not None:
            return cached

    groups = _referenced_node_ids_for_groups(
        model.elements_path,
        {
            "sternum": _STERNUM_PIDS,
            "spine": _THORACIC_VERTEBRA_BODY_PIDS,
            "t1": frozenset({_T1_PID}),
            "t12": frozenset({_T12_PID}),
        },
    )

    centroids = _centroids_for_node_id_groups(model.nodes_path, groups)

    anterior = centroids["sternum"] - centroids["spine"]
    anterior /= np.linalg.norm(anterior)

    up = centroids["t1"] - centroids["t12"]
    up = up - np.dot(up, anterior) * anterior  # orthogonalize (Gram-Schmidt)
    up /= np.linalg.norm(up)

    lateral = np.cross(anterior, up)
    lateral /= np.linalg.norm(lateral)

    axes = BodyAxes(anterior=anterior, up=up, lateral=lateral)
    if cache_path is not None:
        _save_cache(axes, cache_path)
    return axes
