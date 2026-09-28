"""Build the clean, per-model ``HBM/<model_name>/`` layout this project
actually uses, by extracting geometry (nodes + element connectivity, no
materials) out of the client's original, messy source export.

Per explicit instruction: "I want a 'HBM' directory and in there I will have
folders like 'F05_Standing'. I will define which one in 'main.py'. You will
then go in, and just parse everything out to get just the elements and
nodes for whatever I need." This module is the "parse everything out" step,
run once per model (or re-run if the source data changes); the result is
what ``config/hbm_models.py`` reads at runtime -- that module knows nothing
about the messy source layout at all.

Materials/sections/parts are deliberately NOT carried over. Kept:
    * ``*NODE``: node id + xyz coordinates.
    * ``*ELEMENT_SOLID`` / ``*ELEMENT_SHELL``: element id, PART ID, and node
      connectivity.
Dropped: `*MAT_*`, `*SECTION_*`, `*HOURGLASS_*`, `*PART` title cards,
`*INCLUDE`, comments -- all of it. The part id (``pid``) on each element IS
kept: it is not a material, it is the topological tag that lets downstream
code tell "this hex is Thorax_Flesh" from "this hex is
Skull_Trabecular_Bone" (site resolution, ``config/sites.py``, needs exactly
this and nothing else).

Output format: LS-DYNA keyword, comma-separated free-field for every card
this module writes -- unambiguous at any node/element id magnitude, so
files this module produces never need the client's fixed-width-aware parser
to be read back correctly (a plain ``line.split(",")`` is exact). Two real
card-shape requirements were discovered empirically (not assumed) by
writing files and feeding them back through the client's own parser until
it accepted them without error -- see ``_write_nodes``/``_write_elements``:
``*NODE`` needs exactly 6 comma fields (``nid,x,y,z,tc,rc`` -- the parser's
delimited reader does plain ``split(",")[idx]`` with no tolerance for
missing trailing fields, so the otherwise-unused ``tc``/``rc`` constraint
flags must still be written, as ``0``); ``*ELEMENT_SHELL`` is exactly 6
fields (``eid,pid,n1,n2,n3,n4``) and the single-line 8-node
``*ELEMENT_SOLID`` form is exactly 10 fields (``eid,pid,n1..n8``).

Verified end to end (2026-09-18) for ``F05_Standing`` (the first model
built this way): re-parsing the written files with the SAME authoritative
client parser and comparing against the original source deck gave an exact
node-coordinate match and an exact element-connectivity match for sampled
nodes/elements, and the Thorax_Flesh (pid 2000500) solid-element count
(34,980) and Thorax_Skin (pid 2000501) shell-element count (11,158) both
matched the counts independently found in this project's other case data.
See ``tests/test_hbm_normalize.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config._hbm_raw_source import RawSourceModel
from core.lsdyna import iter_wanted_cards


@dataclass(frozen=True)
class BuiltHbmModel:
    """One model's freshly-written clean nodes+elements file pair."""

    key: str
    family: str
    posture: str
    nodes_path: Path
    elements_path: Path
    n_nodes: int
    n_solid_elements: int
    n_shell_elements: int


def build_hbm_model_folder(model: RawSourceModel, hbm_root: Path, progress=None) -> BuiltHbmModel:
    """Extract ``model``'s geometry from the raw source and write it to
    ``{hbm_root}/{model.key}/Nodes.k`` and ``{hbm_root}/{model.key}/Elements.k``.

    Args:
        model: a ``RawSourceModel`` from
            ``config._hbm_raw_source.discover_raw_source_models`` -- knows
            where to find this model's data in the messy original export.
        hbm_root: the clean ``HBM/`` root this project actually uses (e.g.
            ``DeformationApp-agent/HBM``). ``{hbm_root}/{model.key}/`` is
            created if needed.
        progress: optional ``callable(str)`` for progress messages -- these
            decks are large (the biggest source elements deck is 239 MB) so
            a silent multi-second wait is worth narrating.

    Returns:
        BuiltHbmModel with the written paths and element/node counts, for
        the caller to sanity-check or log.
    """
    return _build_from_deck_lists(
        key=model.key,
        family=model.family,
        posture=model.posture,
        nodes_decks=model.nodes_decks,
        elements_decks=model.elements_decks,
        hbm_root=hbm_root,
        progress=progress,
    )


def build_hbm_model_from_folder(
    key: str,
    source_dir: Path,
    hbm_root: Path,
    family: str = None,
    posture: str = None,
    progress=print,
) -> BuiltHbmModel:
    """**The generic, no-naming-convention-required entry point.** Given
    ANY folder containing LS-DYNA keyword files -- arbitrarily named, any
    number of them, whatever the client happened to export -- scan every
    ``.k`` / ``.dyn`` file found (recursively) for ``*NODE`` /
    ``*ELEMENT_SOLID`` / ``*ELEMENT_SHELL`` cards and write the extracted
    geometry to ``{hbm_root}/{key}/Nodes.k`` + ``Elements.k``.

    This is the answer to "if I drop in a folder with all the .k files and
    cards for a model, will it get stripped down automatically" -- yes,
    via THIS function, not via ``config._hbm_raw_source`` (which only
    matches the two specific naming patterns already present in the
    client's existing ``HBM/`` export, and will find nothing for a
    genuinely new folder shaped differently).

    No assumption is made about which file is "the" nodes file or "the"
    elements file, or about any `*INCLUDE` structure -- every `.k`/`.dyn`
    file under ``source_dir`` is scanned directly for the cards this
    project cares about. This is deliberately simple and matches the
    client's own included files already sitting flat in the folder (an
    `*INCLUDE`'d sub-file is just another file that gets scanned directly,
    rather than requiring `*INCLUDE` resolution to find it).

    Node/element ids are validated for a specific, real failure mode: the
    SAME id appearing in more than one scanned file. A duplicate node id
    with IDENTICAL coordinates across files is silently deduplicated (kept
    once) -- this is common when a shared node sits on the boundary between
    two decks. A duplicate node id with DIFFERING coordinates, or a
    duplicate element id at all, raises ``ValueError`` rather than silently
    picking one arbitrarily or writing a file FEBio/downstream code would
    choke on -- ambiguous source data should fail loudly, not guess.

    Args:
        key: the model's folder name under ``hbm_root`` (e.g.
            ``"M50_Standing"``) -- and how it will be looked up later via
            ``config.hbm_models.discover_hbm_models``.
        source_dir: the folder containing the dropped-in LS-DYNA files.
        hbm_root: the clean ``HBM/`` root (e.g. ``DeformationApp-agent/HBM``).
        family / posture: optional override for display metadata -- if
            omitted, parsed from ``key`` the same way
            ``config.hbm_models._match_family``/``_match_posture`` do (so
            a manually-supplied model still shows up sensibly in the GUI).
        progress: optional ``callable(str)``.

    Returns:
        BuiltHbmModel with the written paths and element/node counts.

    Raises:
        FileNotFoundError: if ``source_dir`` has no ``.k``/``.dyn`` files.
        ValueError: on a genuine id conflict (see above).
    """
    deck_files = sorted(
        p for p in Path(source_dir).rglob("*") if p.is_file() and p.suffix.lower() in (".k", ".dyn")
    )
    if not deck_files:
        raise FileNotFoundError(f"No .k/.dyn files found under {source_dir}")

    if progress:
        progress(f"[{key}] found {len(deck_files)} deck file(s) in {source_dir}: "
                  f"{[p.name for p in deck_files]}")

    if family is None:
        from config.hbm_models import _match_family

        family = _match_family(key)
    if posture is None:
        from config.hbm_models import _match_posture

        posture = _match_posture(key)

    return _build_from_deck_lists(
        key=key,
        family=family,
        posture=posture,
        nodes_decks=tuple(deck_files),
        elements_decks=tuple(deck_files),
        hbm_root=hbm_root,
        progress=progress,
        strict_id_conflicts=True,
    )


def _build_from_deck_lists(
    key: str,
    family,
    posture,
    nodes_decks,
    elements_decks,
    hbm_root: Path,
    progress,
    strict_id_conflicts: bool = False,
) -> BuiltHbmModel:
    model_dir = Path(hbm_root) / key
    model_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = model_dir / "Nodes.k"
    elements_path = model_dir / "Elements.k"

    n_nodes = _write_nodes(key, nodes_decks, nodes_path, progress, strict_id_conflicts)
    n_solid, n_shell = _write_elements(key, elements_decks, elements_path, progress, strict_id_conflicts)

    return BuiltHbmModel(
        key=key,
        family=family,
        posture=posture,
        nodes_path=nodes_path,
        elements_path=elements_path,
        n_nodes=n_nodes,
        n_solid_elements=n_solid,
        n_shell_elements=n_shell,
    )


def _write_nodes(key: str, decks, nodes_path: Path, progress, strict_id_conflicts: bool) -> int:
    if progress:
        progress(f"[{key}] reading {len(decks)} node deck(s)...")

    seen: dict = {}  # nid -> (x, y, z), only populated/checked when strict_id_conflicts
    n_nodes = 0
    with open(nodes_path, "w", encoding="utf-8", newline="\n") as out:
        out.write("*KEYWORD\n*NODE\n")
        for deck in decks:
            for _keyword, params in iter_wanted_cards(deck, ("*NODE",)):
                nid = params["nid"]
                xyz = (params["x"], params["y"], params["z"])
                if strict_id_conflicts:
                    prior = seen.get(nid)
                    if prior is not None:
                        if prior != xyz:
                            raise ValueError(
                                f"[{key}] node id {nid} appears twice with DIFFERENT coordinates "
                                f"({prior} vs {xyz}) across scanned files -- ambiguous source data, "
                                f"refusing to guess which is correct."
                            )
                        continue  # identical duplicate -- silently deduplicated
                    seen[nid] = xyz
                # 6 fields required -- nid,x,y,z,tc,rc -- see module
                # docstring: the client's delimited parser has no
                # tolerance for missing trailing fields. tc/rc (constraint
                # flags) are written as 0 (unconstrained); this project
                # has no use for them but the read-back parser requires
                # the columns to exist.
                out.write(f"{nid},{xyz[0]!r},{xyz[1]!r},{xyz[2]!r},0,0\n")
                n_nodes += 1
        out.write("*END\n")

    if progress:
        progress(f"[{key}] wrote {n_nodes} nodes -> {nodes_path.name}")
    return n_nodes


def _write_elements(key: str, decks, elements_path: Path, progress, strict_id_conflicts: bool):
    if progress:
        progress(f"[{key}] reading {len(decks)} element deck(s)...")

    n_solid = 0
    n_shell = 0
    seen_solid_eids: set = set()
    seen_shell_eids: set = set()
    with open(elements_path, "w", encoding="utf-8", newline="\n") as out:
        out.write("*KEYWORD\n")

        # One *ELEMENT_SOLID block covering every solid across every source
        # deck, then one *ELEMENT_SHELL block -- not a repeated keyword
        # header per element. Costs one extra sequential pass per deck
        # (once for solids, once for shells) instead of a single combined
        # pass -- worth it for a correctly-shaped output file.
        out.write("*ELEMENT_SOLID\n")
        for deck in decks:
            for _keyword, params in iter_wanted_cards(deck, ("*ELEMENT_SOLID",)):
                if "eid" not in params:
                    continue
                n = [params.get(f"n{i}") for i in range(1, 9)]
                if any(v is None for v in n):
                    continue
                eid = params["eid"]
                if strict_id_conflicts:
                    if eid in seen_solid_eids:
                        raise ValueError(
                            f"[{key}] solid element id {eid} appears in more than one scanned "
                            f"file -- ambiguous source data, refusing to guess which is correct."
                        )
                    seen_solid_eids.add(eid)
                out.write(f"{eid},{params['pid']}," + ",".join(str(v) for v in n) + "\n")
                n_solid += 1

        out.write("*ELEMENT_SHELL\n")
        for deck in decks:
            for _keyword, params in iter_wanted_cards(deck, ("*ELEMENT_SHELL",)):
                if "eid" not in params:
                    # *ELEMENT_SHELL_THICKNESS's second card (t1-t4
                    # thickness values, no connectivity) -- irrelevant to
                    # pure geometry, same skip already validated in
                    # app/hbm_reader.py.
                    continue
                n = [params.get(f"n{i}") for i in range(1, 5)]
                if any(v is None for v in n):
                    continue
                eid = params["eid"]
                if strict_id_conflicts:
                    if eid in seen_shell_eids:
                        raise ValueError(
                            f"[{key}] shell element id {eid} appears in more than one scanned "
                            f"file -- ambiguous source data, refusing to guess which is correct."
                        )
                    seen_shell_eids.add(eid)
                out.write(f"{eid},{params['pid']}," + ",".join(str(v) for v in n) + "\n")
                n_shell += 1

        out.write("*END\n")

    if progress:
        progress(f"[{key}] wrote {n_solid} solid + {n_shell} shell elements -> {elements_path.name}")
    return n_solid, n_shell


def build_model_by_key(key: str, source_hbm_root: Path, out_hbm_root: Path, progress=print) -> BuiltHbmModel:
    """Build ONE named model (e.g. ``"F05_Standing"``) from the client's
    EXISTING ``HBM/`` export tree -- i.e. one of the models
    ``config._hbm_raw_source`` already knows the two hardcoded layouts for.

    For a NEW model dropped in with arbitrary file naming, use
    ``build_hbm_model_from_folder`` instead -- this function will raise
    ``KeyError`` for anything that doesn't match those two known patterns.

    This -- not a batch "build everything" pass -- is the intended usage:
    per explicit instruction, models are added one at a time, as needed,
    driven by whatever model ``main_1.py`` (and the PPE being fitted) actually
    calls for. Discovering/building every model the source tree happens to
    contain is unnecessary work and unwanted output.

    Raises:
        KeyError: if ``key`` isn't found in the source tree (see
            ``config._hbm_raw_source.discover_raw_source_models`` for what
            keys exist there).
    """
    from config._hbm_raw_source import discover_raw_source_models

    models = discover_raw_source_models(source_hbm_root)
    if key not in models:
        raise KeyError(
            f"'{key}' not found in the raw HBM source tree at {source_hbm_root} "
            f"(available: {sorted(models.keys())})"
        )
    return build_hbm_model_folder(models[key], out_hbm_root, progress=progress)
