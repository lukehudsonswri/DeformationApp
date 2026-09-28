"""INTERNAL: locate the messy, original HBM source data (the client's raw
export tree) so it can be extracted into the clean, per-model layout this
project actually uses.

**Not the public model registry.** That is ``config/hbm_models.py`` --  a
much simpler module that just scans ``HBM/<model_name>/`` folders for a
``Nodes.k`` + ``Elements.k`` pair, per explicit instruction: "I want a 'HBM'
directory and in there I will have folders like 'F05_Standing'. I will
define which one in 'main.py'." This module exists only to answer "where,
in the original messy source tree, do I find the nodes/elements for model
X" -- consumed exclusively by ``config/hbm_normalize.py``'s one-time
extraction step, which writes the clean per-model folders that
``config/hbm_models.py`` actually reads from at runtime.

Why the raw source is messy enough to need this at all
----------------------------------------------------------------------------
The client's original HBM/ export is not organized one-folder-per-model:
several models share a single elements deck, some models are split across
a separate "Bone" file and "Skin" file, and every model's parts are named by
one properties deck shared across the whole family. None of that ever
needs to be re-derived once the clean layout has been built once -- this
module is a one-time-use bridge, not a subsystem your everyday code should
import.

What was actually verified against the real HBM/ tree (2026-09-18), not
assumed
----------------------------------------------------------------------------
* ``01_Model/I-PREDICT_v1.0_Main.dyn`` is the ONE real properties deck
  (``*PART``/``*SECTION``/``*MAT``/``*HOURGLASS`` cards) for the entire
  I-PREDICT v1.0 model family.
* ``F05_Seated``, ``M50_Seated_Skin``, ``M50_Standing_Skin``, ``M95_Seated``
  each pair a "Bone" deck with a "Skin" deck. **Correction to an earlier,
  wrong manual spot-check**: these "Bone" decks are NOT shell-only -- e.g.
  ``M50_Standing_Bone_Element.k`` contains a real ``*ELEMENT_SOLID`` section
  (skeletal bone volumes: pids like 1000007 Skull_Trabecular_Bone and the
  2000001-2000049 vertebra/rib range) starting partway through the file,
  after an initial ``*ELEMENT_SHELL`` section (bone surfaces). An early
  check here that only inspected each file's first ``*ELEMENT_*`` keyword
  missed this; the actual per-file streaming scan (``_deck_has_solid_elements``)
  does not. **None of the four trees contain Thorax_Flesh (pid 2000500)
  specifically**, confirmed by direct PID extraction -- so they have SOME
  solid tissue (bone), just not the flesh a torso PPE case needs. See
  ``RawSourceModel.has_solid_elements``'s own docstring for why that flag answers
  "any solid at all", not "has this site's specific flesh" -- use
  ``RawSourceModel.has_pid(pid)`` for the latter.
* Despite the different directory names, these four all share IDENTICAL
  topology: the same 71,671-node count and the exact same PID set
  (1000015 Head_Skin, 2000501 Thorax_Skin, 2000504, 4000012/19/20,
  5000012/19/20, 6000000/6100000/2/4/6, 7000000/7100000/2/4/6 -- confirmed
  by direct extraction from both ``F05_Seated_Skin_Elements.k`` and
  ``M50_Standing_Skin_Elements.k``). Different node COORDINATES (different
  anthropometry/posture), same mesh CONNECTIVITY. This means
  ``01_Model/I-PREDICT_v1.0_Main.dyn``'s ``*PART`` title -> pid mapping is
  valid for ALL of these trees, not just for 01_Model's own node/element
  pairs -- so it is used here as the shared ``properties_deck`` for every
  discovered model, not only the 01_Model-sourced ones.
* ``01_Model``'s own per-family node decks (``I-PREDICT_v1.0_F05_Nodes.k``,
  etc.) are a DIFFERENT, much larger topology (~1.4M lines / whole-body
  including bones and internal organs, not just skin) that pairs with a
  SHARED elements deck per family group, not a same-named one -- confirmed
  directly against the client's own decks and restated as the
  ``_ONE_MODEL_ENTRIES`` table below:
  F05 and F50 both pair with ``I-PREDICT_v1.0_Morphing_F50_Elements.k``;
  M50 and M95 both pair with ``I-PREDICT_v1.0_Morphing_M50_Elements.k``.
  This project's own naming is simply inconsistent between "F50" (elements)
  and "F05" (one of the two node sets that uses those elements) -- this is
  not a typo introduced here, it is preserved from the source data.
  ``I-PREDICT_v1.0_Morphing_M50_Elements.k``'s ``*ELEMENT_SOLID`` section
  DOES include pid 2000500 (Thorax_Flesh), directly confirmed.
* A genuine typo WAS found in the source data: ``M50_Standing_Skin/Main.dyn``
  ``*INCLUDE``s ``M50_Standing_Bone_Elements.k`` (plural), but the file on
  disk is named ``M50_Standing_Bone_Element.k`` (singular) -- that
  ``Main.dyn`` is broken and would fail to resolve. The sibling
  ``M50_Standing.dyn`` in the same directory has the correct (singular)
  filename. This registry does not read either ``.dyn`` file at all (they
  are pure ``*INCLUDE`` lists with no properties data of their own -- see
  above) so the typo doesn't affect discovery, but it is worth knowing if
  anyone tries to hand-run either ``.dyn`` through LS-DYNA/LS-PrePost
  directly.

Design consequence worth flagging (see AGENTS.md section 5)
----------------------------------------------------------------------------
Thorax_Flesh (pid 2000500) is a real, fully material-assigned part (Ogden
rubber) with real solid geometry -- 34,980 hex8 elements, confirmed via the
authoritative client parser (``core.lsdyna.iter_wanted_cards``, not a naive
``line.split()``) in the SEATED ``01_Model`` decks, matching the element
count found directly in the F05 case's own combined mesh. It is not
"missing" from this project's data in any general sense.

What's actually missing is a STANDING-posture export of that flesh mesh.
``M50_Standing_Skin`` is the only standing-posture M50 source in this
``HBM/`` tree, and its two files (Bone, Skin) are a stripped export that
never included the flesh volume -- confirmed absent from both, via the same
authoritative parser, not the naive PID-extraction check used elsewhere in
this module. So a torso PPE-fitting case built from ``M50_Standing`` has no
flesh tissue to solve against on its own, and getting it one is NOT a
same-posture anthropometry morph (the CPD+RBF scope in AGENTS.md sections
2.2/2.3) -- it requires re-posing a seated flesh mesh into a standing
configuration (hips/knees/spine actually re-articulate), a substantially
larger problem. ``RawSourceModel.has_pid(2000500)`` surfaces the gap precisely;
the coarser ``has_solid_elements`` alone would (incorrectly) suggest this
model is usable as-is, since it does carry solid bone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Directories under HBM/ that are not model sources at all.
_NON_MODEL_DIRS = {"02_BCs", "macros"}

# Family / posture tokens recognized in filenames and directory names.
# Order matters for posture: "90-90-90" must be checked before "Standing"/
# "Seated" since it contains neither substring.
_FAMILY_TOKENS = ("F05", "F50", "M50", "M95")
_POSTURE_TOKENS: Tuple[Tuple[str, str], ...] = (
    ("90-90-90", "90-90-90"),
    ("Standing", "Standing"),
    ("Seated", "Seated"),
)


@dataclass(frozen=True)
class RawSourceModel:
    """One discovered human body model: enough deck paths to parse its full
    geometry, plus (when available) the properties deck that names its
    parts. A model may be made up of MORE THAN ONE nodes/elements deck pair
    (e.g. a separate "Bone" deck and "Skin" deck) -- callers should read
    them together, in order, via ``core.lsdyna.iter_wanted_cards`` per file.
    """

    key: str  # e.g. "F05_Seated", "M50_Standing", "M50_90-90-90"
    family: str  # "F05" | "F50" | "M50" | "M95"
    posture: str  # "Seated" | "Standing" | "90-90-90"
    source_dir: Path
    nodes_decks: Tuple[Path, ...]
    elements_decks: Tuple[Path, ...]
    properties_deck: Optional[Path]

    @property
    def has_solid_elements(self) -> bool:
        """Whether ANY of this model's elements decks contain at least one
        ``*ELEMENT_SOLID`` card, of ANY tissue type.

        **This does NOT mean "has torso flesh" or any other specific
        site's solid tissue.** Verified concretely: ``M50_Standing_Skin``'s
        "Bone" deck (``M50_Standing_Bone_Element.k``) DOES contain
        ``*ELEMENT_SOLID`` cards -- an earlier manual spot-check (looking
        only at the file's *ELEMENT_SHELL header and assuming that was the
        only section) missed this; the automated scan here caught it. But
        those solid elements are skeletal bone volumes (pids like 1000007
        Skull_Trabecular_Bone, 2000001-2000049 vertebra/rib range) -- NOT
        Thorax_Flesh (pid 2000500), which is absent from that deck.
        ``01_Model``'s shared elements decks DO contain pid 2000500.

        So this flag answers "is this model shell-only (bone surfaces +
        skin, e.g. a pure visualization/posture reference) or does it carry
        at least some solid tissue" -- a coarse, cheap, useful triage. It
        is NOT a substitute for the site-specific PID resolution
        (``config/sites.py``, AGENTS.md section 1.2) that determines
        whether a particular site's flesh actually exists in this model.
        Use ``has_pid(pid)`` for a precise per-part check once you know
        which pid a site resolves to.

        Result is cached per-instance-key (dataclass is frozen, so the
        cache lives in a module-level dict rather than on the instance).
        """
        cache = _SOLID_CHECK_CACHE.setdefault(self.key, {})
        if "value" not in cache:
            cache["value"] = any(_deck_has_solid_elements(p) for p in self.elements_decks)
        return cache["value"]

    def has_pid(self, pid: int) -> bool:
        """Whether any element in this model's elements decks (solid OR
        shell) references part id ``pid``. Cheap streaming scan, cached per
        ``(model key, pid)``. This is the precise question site resolution
        actually needs -- e.g. ``model.has_pid(2000500)`` for Thorax_Flesh
        -- as opposed to the coarse ``has_solid_elements`` above.
        """
        cache_key = (self.key, pid)
        if cache_key in _PID_CHECK_CACHE:
            return _PID_CHECK_CACHE[cache_key]
        found = any(_deck_has_pid(p, pid) for p in self.elements_decks)
        _PID_CHECK_CACHE[cache_key] = found
        return found


# Keyed by RawSourceModel.key -- avoids re-scanning multi-hundred-MB elements
# decks on every access. Not an lru_cache on a method because RawSourceModel is a
# frozen dataclass (unhashable mutable cache field would break that).
_SOLID_CHECK_CACHE: Dict[str, Dict[str, bool]] = {}
# Keyed by resolved path string -- avoids re-scanning the same deck file
# across different RawSourceModel instances that happen to share it (e.g. two
# family/posture combos sharing one properties deck).
_DECK_SOLID_CACHE: Dict[str, bool] = {}
# Keyed by (model key, pid) -- see RawSourceModel.has_pid.
_PID_CHECK_CACHE: Dict[Tuple[str, int], bool] = {}


def _deck_has_solid_elements(path: Path) -> bool:
    """Cheap streaming check for ``*ELEMENT_SOLID`` anywhere in ``path``,
    without a full parse. Verified fast in practice: ~2.5s on a 239 MB deck
    where the keyword appears early; worst case (keyword absent, or very
    late) is a single sequential read of the file, still far cheaper than
    parsing every card.
    """
    key = str(path.resolve())
    if key in _DECK_SOLID_CACHE:
        return _DECK_SOLID_CACHE[key]

    found = False
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("*ELEMENT_SOLID"):
                    found = True
                    break
    except OSError:
        found = False

    _DECK_SOLID_CACHE[key] = found
    return found


def _deck_has_pid(path: Path, pid: int) -> bool:
    """Cheap streaming check for whether any ``*ELEMENT_SOLID`` or
    ``*ELEMENT_SHELL`` card in ``path`` references part id ``pid`` (the
    second field on each fixed-width/free-field element data line, per
    LS-DYNA's ``eid, pid, n1, n2, ...`` card layout).

    **Best-effort, not authoritative.** This uses a naive ``line.split()``,
    which is exactly the parsing shortcut ``core/lsdyna/bridge.py`` exists
    to avoid for large HBM decks (adjacent fixed-width fields can abut with
    no separator once IDs exceed 7 digits). In practice this was verified
    to work correctly on the real decks checked here (both a small deck,
    ``M50_Standing_Bone_Element.k``, and 01_Model's 239 MB
    ``I-PREDICT_v1.0_Morphing_M50_Elements.k``, correctly found pid
    2000500 in the latter) -- but a fast, approximate, existence-only
    triage is what this is for. Use ``core.lsdyna.iter_wanted_cards`` (the
    client's real fixed-width-aware parser) for anything where a false
    negative would matter, e.g. actually extracting a part's geometry.
    Not cached across calls with different pids for the same file (unlike
    ``_deck_has_solid_elements``'s single yes/no) -- a per-pid cache would
    need to hold every pid ever queried, which isn't worth it for the
    handful of site-resolution lookups this supports; a fresh scan per
    (file, pid) pair is still a single sequential read, same cost class as
    ``_deck_has_solid_elements``.
    """
    pid_str = str(pid)
    in_element_section = False
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("*"):
                    in_element_section = line.startswith(("*ELEMENT_SOLID", "*ELEMENT_SHELL"))
                    continue
                if in_element_section and line.strip():
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] == pid_str:
                        return True
    except OSError:
        return False
    return False


def _match_family(name: str) -> Optional[str]:
    for token in _FAMILY_TOKENS:
        if token in name:
            return token
    return None


def _match_posture(name: str) -> Optional[str]:
    for token, label in _POSTURE_TOKENS:
        if token in name:
            return label
    return None


# ---------------------------------------------------------------------------
# 01_Model: one shared properties deck + shared-per-family-group elements
# decks + one nodes deck per family/posture. The nodes<->elements pairing is
# NOT a filename-pattern match (see module docstring: "F05" nodes pair with
# "F50"-named elements) -- this table was validated directly against the
# client's own *PART/*SECTION cards (see the module docstring above) and is
# restated here as plain data.
# ---------------------------------------------------------------------------
_MODEL_DECK_FILENAME = "I-PREDICT_v1.0_Main.dyn"

# (nodes filename, family, posture, elements filename)
_ONE_MODEL_ENTRIES: Tuple[Tuple[str, str, str, str], ...] = (
    ("I-PREDICT_v1.0_F05_Nodes.k", "F05", "Seated", "I-PREDICT_v1.0_Morphing_F50_Elements.k"),
    ("I-PREDICT_v1.0_F50_Nodes.k", "F50", "Seated", "I-PREDICT_v1.0_Morphing_F50_Elements.k"),
    ("I-PREDICT_v1.0_M50_Nodes.k", "M50", "Seated", "I-PREDICT_v1.0_Morphing_M50_Elements.k"),
    ("I-PREDICT_v1.0_M95_Nodes.k", "M95", "Seated", "I-PREDICT_v1.0_Morphing_M50_Elements.k"),
    ("I-PREDICT_v1.0_90-90-90_Nodes.k", "M50", "90-90-90", "I-PREDICT_v1.0_90-90-90_Elements.k"),
    (
        "Nodes_Resolved_LowerLimb_repaired_F05_Standing.k",
        "F05",
        "Standing",
        "I-PREDICT_v1.0_ElementsMod_repaired_F05_Standing.k",
    ),
)


def _discover_one_model_dir(source_dir: Path) -> Dict[str, RawSourceModel]:
    properties_deck = source_dir / _MODEL_DECK_FILENAME
    properties_deck = properties_deck if properties_deck.is_file() else None

    models: Dict[str, RawSourceModel] = {}
    for nodes_name, family, posture, elements_name in _ONE_MODEL_ENTRIES:
        nodes_path = source_dir / nodes_name
        elements_path = source_dir / elements_name
        if not (nodes_path.is_file() and elements_path.is_file()):
            continue  # this source tree doesn't have this particular pairing
        key = f"{family}_{posture}"
        models[key] = RawSourceModel(
            key=key,
            family=family,
            posture=posture,
            source_dir=source_dir,
            nodes_decks=(nodes_path,),
            elements_decks=(elements_path,),
            properties_deck=properties_deck,
        )
    return models


# ---------------------------------------------------------------------------
# Standalone trees: F05_Seated, M50_Seated_Skin, M50_Standing_Skin,
# M95_Seated. Each pairs a "Bone" deck (bone surfaces + bone VOLUME -- see
# module docstring's correction) with a "Skin" deck (shell only);
# family/posture are read from the DIRECTORY name (the per-file names are
# consistent with it). No local properties deck exists (their Main.dyn/
# *.dyn files are pure *INCLUDE lists, not *PART data) -- properties_deck is
# instead the shared 01_Model deck, valid here because the topology (node
# count + PID set) was confirmed identical to 01_Model's family (see module
# docstring).
# ---------------------------------------------------------------------------


def _find_deck_pair(source_dir: Path, tag: str) -> Optional[Tuple[Path, Path]]:
    """Find a ``*_{tag}_Nodes.k`` / ``*_{tag}_Element(s).k`` pair in
    ``source_dir``, tolerating the singular/plural "Element(s)" filename
    inconsistency confirmed in the real data (see module docstring's note
    on ``M50_Standing_Bone_Element.k``).
    """
    nodes_candidates = sorted(source_dir.glob(f"*_{tag}_Nodes.k"))
    if not nodes_candidates:
        return None
    elements_candidates = sorted(source_dir.glob(f"*_{tag}_Element*.k"))
    if not elements_candidates:
        return None
    return nodes_candidates[0], elements_candidates[0]


def _discover_standalone_shell_dir(
    source_dir: Path, shared_properties_deck: Optional[Path]
) -> Dict[str, RawSourceModel]:
    family = _match_family(source_dir.name)
    posture = _match_posture(source_dir.name)
    if family is None or posture is None:
        return {}

    nodes_decks: List[Path] = []
    elements_decks: List[Path] = []
    for tag in ("Bone", "Skin"):
        pair = _find_deck_pair(source_dir, tag)
        if pair is None:
            continue
        nodes_decks.append(pair[0])
        elements_decks.append(pair[1])

    if not nodes_decks:
        return {}

    key = f"{family}_{posture}"
    model = RawSourceModel(
        key=key,
        family=family,
        posture=posture,
        source_dir=source_dir,
        nodes_decks=tuple(nodes_decks),
        elements_decks=tuple(elements_decks),
        properties_deck=shared_properties_deck,
    )
    return {key: model}


def discover_raw_source_models(hbm_root: Path) -> Dict[str, RawSourceModel]:
    """Discover every HBM model under ``hbm_root`` (e.g.
    ``DeformationApp/HBM``).

    Returns a dict keyed by ``"{family}_{posture}"`` (e.g. ``"F05_Seated"``,
    ``"M50_Standing"``, ``"M50_90-90-90"``). Today's real data has exactly
    one key collision: ``01_Model``'s F05_Seated entry (full body, includes
    Thorax_Flesh pid 2000500) vs. the standalone ``F05_Seated`` tree
    (bone + skin only, no Thorax_Flesh -- see module docstring). **The
    ``01_Model``-sourced variant always wins** when both exist for the same
    key: it is the richer, properties-bearing source tree.

    This is a DETERMINISTIC preference by source tree, not a
    ``has_solid_elements``-based heuristic -- that flag is not reliable for
    this tie-break, since the standalone trees also carry solid BONE
    elements (see module docstring's correction) and would otherwise look
    just as "solid-bearing" as ``01_Model``'s actual flesh. Use
    ``discover_raw_source_models_multi`` if you need every variant, not just the
    preferred one per key.
    """
    all_models = discover_raw_source_models_multi(hbm_root)
    resolved: Dict[str, RawSourceModel] = {}
    for key, variants in all_models.items():
        one_model_variants = [m for m in variants if m.source_dir.name == "01_Model"]
        resolved[key] = one_model_variants[0] if one_model_variants else variants[0]
    return resolved


def discover_raw_source_models_multi(hbm_root: Path) -> Dict[str, List[RawSourceModel]]:
    """Like ``discover_raw_source_models``, but keeps every discovered variant per
    key instead of picking one. Most callers want ``discover_raw_source_models``.
    """
    hbm_root = Path(hbm_root)
    variants: Dict[str, List[RawSourceModel]] = {}

    def _add(models: Dict[str, RawSourceModel]) -> None:
        for key, model in models.items():
            variants.setdefault(key, []).append(model)

    one_model_dir = hbm_root / "01_Model"
    shared_properties_deck: Optional[Path] = None
    if one_model_dir.is_dir():
        _add(_discover_one_model_dir(one_model_dir))
        candidate = one_model_dir / _MODEL_DECK_FILENAME
        shared_properties_deck = candidate if candidate.is_file() else None

    for entry in sorted(hbm_root.iterdir()):
        if not entry.is_dir() or entry.name in _NON_MODEL_DIRS or entry == one_model_dir:
            continue
        _add(_discover_standalone_shell_dir(entry, shared_properties_deck))

    return variants
