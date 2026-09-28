"""The actual, simple HBM model registry this project uses at runtime.

Per explicit instruction: "I only want a HBM directory and in there I will
have folders like 'F05_Standing'. I will define which one in 'main.py'. You
will then go in, and just parse everything out to get just the elements and
nodes for whatever I need."

So: ``HBM/`` contains one subdirectory per model (e.g. ``F05_Standing/``,
``M50_Seated/``), and each folder is expected to hold a nodes file and an
elements file. Two ways a folder can satisfy that:

1. The clean, normalized layout ``config/hbm_normalize.py`` writes:
   exactly ``Nodes.k`` and ``Elements.k`` (geometry only, no
   materials/sections -- see that module's docstring). This is the fast,
   exact-filename path, tried first.
2. **A raw folder dropped in directly**, matching the client's own export
   naming (e.g. ``I-PREDICT_v1.0_F05_Nodes.k`` /
   ``I-PREDICT_v1.0_Morphing_F05_Elements.k``) -- no separate
   normalization step required. Per explicit instruction: search the
   folder for any file whose name contains "Node"/"Nodes" (case
   insensitive) for the nodes file, and "Element"/"Elements" for the
   elements file. This works directly (no ``hbm_normalize`` step needed)
   because ``core.lsdyna.bridge.iter_wanted_cards`` -- what every
   downstream reader in this project already uses -- parses the client's
   real fixed-width card format natively, not just the clean comma-only
   format ``hbm_normalize`` happens to also produce. Confirmed directly
   against a real raw folder (``HBM/F05_Seated/``, dropped in with its
   original client filenames): nodes and both element types parse
   correctly, and the torso site's own part ids (Thorax_Flesh 2000500 /
   Thorax_Skin 2000501) resolve to the same element counts as the
   already-normalized ``F05_Standing`` -- confirming this family's part
   ids are consistent across raw and normalized sources alike.

If a folder has multiple files matching the same substring (e.g. several
``*Node*`` files), the largest one is used -- in every case seen so far
the real, full deck is reliably the biggest file; a same-substring match
that's much smaller is more likely an unrelated aux/partial export, not a
second copy of the real data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

# Family / posture tokens recognized in folder names, purely for display /
# grouping in the GUI -- discovery itself does not depend on a folder
# matching any of these; an unrecognized name is still a usable model, just
# with family/posture left as None.
_FAMILY_TOKENS = ("F05", "F50", "M50", "M95")
_POSTURE_TOKENS: Tuple[Tuple[str, str], ...] = (
    ("90-90-90", "90-90-90"),
    ("Standing", "Standing"),
    ("Seated", "Seated"),
)

_NODES_FILENAME = "Nodes.k"
_ELEMENTS_FILENAME = "Elements.k"
_DECK_EXTENSIONS = (".k", ".dyn")


@dataclass(frozen=True)
class HbmModel:
    """One HBM model: a folder name, its Nodes.k / Elements.k paths, and
    (best-effort, for display/grouping only) a parsed family/posture.
    """

    key: str  # the folder name itself, e.g. "F05_Standing"
    family: Optional[str]  # "F05" | "F50" | "M50" | "M95" | None
    posture: Optional[str]  # "Seated" | "Standing" | "90-90-90" | None
    folder: Path
    nodes_path: Path
    elements_path: Path


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


def _find_by_substring(folder: Path, substring: str) -> Optional[Path]:
    """Case-insensitive filename substring search within ``folder`` (not
    recursive -- every HBM model folder in this project is flat), among
    ``.k``/``.dyn`` files only. Returns the largest matching file if more
    than one is found, or ``None`` if there's no match.
    """
    candidates = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in _DECK_EXTENSIONS and substring in p.name.lower()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def discover_hbm_models(hbm_root: Path) -> Dict[str, HbmModel]:
    """Scan ``hbm_root`` for model folders.

    A subdirectory is a model if it resolves both a nodes file and an
    elements file -- either the exact clean ``Nodes.k``/``Elements.k``
    ``config/hbm_normalize.py`` writes, or (falling back per-file,
    independently) the largest ``.k``/``.dyn`` file whose name contains
    "node" or "element" respectively (case insensitive) -- see this
    module's docstring for why the fallback works without any separate
    normalization step. Anything else under ``hbm_root`` -- stray files,
    folders that resolve neither file, non-model folders like an old
    ``macros/`` -- is silently skipped, not an error, since ``hbm_root``
    may reasonably contain other things over time.

    Args:
        hbm_root: e.g. ``DeformationApp/HBM``.

    Returns:
        Dict keyed by folder name (e.g. ``{"F05_Standing": HbmModel(...)}``).
    """
    hbm_root = Path(hbm_root)
    models: Dict[str, HbmModel] = {}
    if not hbm_root.is_dir():
        return models

    for entry in sorted(hbm_root.iterdir()):
        if not entry.is_dir():
            continue

        nodes_path = entry / _NODES_FILENAME
        if not nodes_path.is_file():
            nodes_path = _find_by_substring(entry, "node")

        elements_path = entry / _ELEMENTS_FILENAME
        if not elements_path.is_file():
            elements_path = _find_by_substring(entry, "element")

        if nodes_path is None or elements_path is None:
            continue
        if not (nodes_path.is_file() and elements_path.is_file()):
            continue

        models[entry.name] = HbmModel(
            key=entry.name,
            family=_match_family(entry.name),
            posture=_match_posture(entry.name),
            folder=entry,
            nodes_path=nodes_path,
            elements_path=elements_path,
        )
    return models
