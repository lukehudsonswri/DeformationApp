"""
LS-DYNA (.k) mesh file loader and saver.

Parses LS-DYNA keyword files to extract mesh data.

--------------------------------------------------------------------------
Vendored from: C:\\Users\\lhudson\\Documents\\Morphing\\fe_personalization\\mesh_io\\lsdyna.py
Vendored on:   2026-09-18, unmodified.
Local changes: none yet.
See AGENTS.md "Vendoring policy". Note this is a *general-purpose* LS-DYNA
mesh reader (fixed-width + free-field), independent of and simpler than the
client's own KeywordProcessor bridged in core/lsdyna/ -- use this one for
PPE .k files, and the client engine (core/lsdyna/bridge.py) for the huge
HBM decks where field widths get ambiguous past 7-digit ids.
--------------------------------------------------------------------------
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Sequence, Union
import logging
import re

from .base import VolumeMesh, NodeSet, ElementSet, Material, ElementType, SurfaceMesh

logger = logging.getLogger(__name__)


def _classify_solid_element(node_ids: List[int]) -> Tuple[Optional[ElementType], List[int]]:
    """Identify a *ELEMENT_SOLID element from its node list.

    LS-DYNA stores tets/wedges as *degenerate hexes*: an 8-node card whose
    trailing nodes repeat (e.g. a tet is ``N1 N2 N3 N4 N4 N4 N4 N4``). The
    number of *distinct* nodes therefore tells us the true element type. A
    genuine 4-node card is already a tet.

    Returns ``(element_type, connectivity)`` where ``connectivity`` is the
    collapsed node list for the detected type, or ``(None, node_ids)`` when the
    pattern isn't recognised (caller should warn and keep it as a hex).
    """
    if len(node_ids) == 4:
        return ElementType.TET4, node_ids

    # 8-node (possibly degenerate) form: collapse to distinct nodes, order-preserving.
    unique = list(dict.fromkeys(node_ids))
    n = len(unique)
    if n == 8:
        return ElementType.HEX8, node_ids
    if n == 4:
        return ElementType.TET4, unique
    if n == 6:
        # Standard LS-DYNA pentahedron (wedge) collapse. Ordering is best-effort.
        return ElementType.WEDGE6, unique
    return None, node_ids


def _classify_shell_element(node_ids: List[int]) -> Tuple[Optional[ElementType], List[int]]:
    """Identify a ``*ELEMENT_SHELL`` element from its node list.

    A 4-node shell card is ``EID, PID, N1, N2, N3, N4``. Triangles are stored
    either with three node fields or as a *degenerate quad* whose last node
    repeats (``N1 N2 N3 N3``); some writers also pad the unused slot with ``0``.
    The number of *distinct* leading nodes therefore gives the true type.

    Returns ``(element_type, connectivity)`` collapsed to the detected type, or
    ``(None, node_ids)`` when the pattern isn't a recognisable tri/quad.
    """
    ids = [n for n in node_ids[:4]]
    # Trailing zero-padding (an unused 4th slot) -> treat as a triangle.
    while len(ids) > 3 and ids[-1] == 0:
        ids = ids[:-1]
    unique = list(dict.fromkeys(ids))
    if len(unique) == 3:
        return ElementType.TRI3, unique
    if len(ids) == 4 and len(unique) == 4:
        return ElementType.QUAD4, ids
    return None, node_ids


def _classify_segment_face(node_ids: Sequence[int]) -> Tuple[Optional[ElementType], List[int]]:
    """Identify a ``*SET_SEGMENT`` face from its node fields.

    LS-DYNA segment sets store one surface face per card as ``N1 N2 N3 N4``.
    Triangles are represented either by three node fields, by a zero-padded
    fourth field, or as a degenerate quad where ``N4 == N3``.
    """
    ids = [int(n) for n in node_ids[:4]]
    if len(ids) < 3:
        return None, ids

    if len(ids) == 3:
        face = ids
    elif ids[3] == 0 or ids[3] == ids[2]:
        face = ids[:3]
    else:
        face = ids[:4]

    if 0 in face or len(set(face)) != len(face):
        return None, ids
    if len(face) == 3:
        return ElementType.TRI3, face
    if len(face) == 4:
        return ElementType.QUAD4, face
    return None, ids


def _parse_lsdyna(filepath: Path) -> Dict[str, Any]:
    """Single-pass parse of an LS-DYNA keyword file.

    Shared by :func:`load_lsdyna` and :func:`extract_lsdyna_element_sets` so the
    two never drift on node compaction, element-type collapse, or part naming.

    Returns a dict with:
      - ``node_order``: original node ids in declaration order
      - ``node_coords``: ``[x, y, z]`` rows parallel to ``node_order``
      - ``raw_elements``: ``(ElementType, [orig_node_ids], pid)`` per solid or
        shell element, in declaration order (connectivity collapsed to the
        detected type but still in *original* node ids)
      - ``part_titles``: ``{pid: title}`` parsed from ``*PART`` headers
      - ``raw_segment_sets``: ``{"sid", "title", "faces"}`` entries from
        ``*SET_SEGMENT`` / ``*SET_SEGMENT_TITLE`` cards. Faces are still in
        original node IDs and may be triangles or quads.
      - ``unrecognized_solids``: count of cards with an unknown collapse pattern
      - ``unrecognized_shells``: count of ``*ELEMENT_SHELL`` cards that weren't a
        recognisable tri/quad
    """
    node_order: List[int] = []
    node_coords: List[List[float]] = []
    seen_ids = set()
    raw_elements: List[Tuple[ElementType, List[int], int]] = []
    raw_segment_sets: List[Dict[str, Any]] = []
    part_titles: Dict[int, str] = {}
    unrecognized_solids = 0
    unrecognized_shells = 0
    current_section = None
    # *PART blocks are a heading (title) line followed by a card whose first
    # field is the pid. Track which of the two we still need to read.
    part_need: Optional[str] = None
    part_pending_title: Optional[str] = None
    segment_need: Optional[str] = None
    segment_pending_title: Optional[str] = None
    segment_pending_sid: Optional[int] = None
    segment_pending_faces: List[List[int]] = []

    def finish_pending_segment_set() -> None:
        nonlocal segment_need, segment_pending_title, segment_pending_sid, segment_pending_faces
        if segment_pending_sid is None:
            segment_need = None
            segment_pending_title = None
            segment_pending_faces = []
            return
        raw_segment_sets.append(
            {
                "sid": segment_pending_sid,
                "title": segment_pending_title,
                "faces": segment_pending_faces,
            }
        )
        segment_need = None
        segment_pending_title = None
        segment_pending_sid = None
        segment_pending_faces = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines and comments
        if not line or line.startswith('$'):
            i += 1
            continue

        # Detect keyword sections
        if line.startswith('*'):
            if current_section == 'SET_SEGMENT':
                finish_pending_segment_set()
            keyword = line.upper()
            if 'NODE' in keyword and 'SET' not in keyword:
                current_section = 'NODE'
            elif 'ELEMENT_SOLID' in keyword:
                current_section = 'ELEMENT_SOLID'
            elif 'ELEMENT_SHELL' in keyword:
                current_section = 'ELEMENT_SHELL'
            elif keyword.startswith('*PART'):
                current_section = 'PART'
                part_need = 'title'
                part_pending_title = None
            elif keyword.startswith('*SET_SEGMENT'):
                current_section = 'SET_SEGMENT'
                segment_need = 'title' if '_TITLE' in keyword else 'sid'
                segment_pending_title = None
                segment_pending_sid = None
                segment_pending_faces = []
            elif 'SET_NODE' in keyword:
                current_section = 'SET_NODE'
            elif 'SET_PART' in keyword or 'SET_SOLID' in keyword:
                current_section = 'SET_ELEMENT'
            elif keyword.startswith('*END'):
                break
            else:
                current_section = None
            i += 1
            continue

        # Parse data based on current section
        if current_section == 'NODE':
            # Handle both comma-separated and space-separated formats
            parts = [p.strip() for p in line.replace(',', ' ').split()]
            if len(parts) >= 4:
                try:
                    node_id = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                except (ValueError, IndexError):
                    pass
                else:
                    if node_id in seen_ids:
                        # Re-declared id: overwrite the previous coords (last wins).
                        node_coords[node_order.index(node_id)] = [x, y, z]
                    else:
                        seen_ids.add(node_id)
                        node_order.append(node_id)
                        node_coords.append([x, y, z])

        elif current_section == 'ELEMENT_SOLID':
            # *ELEMENT_SOLID card: EID PID N1..N8 (or N1..N4 for native tet)
            # Fixed-width fallback: LS-DYNA standard is 8-char fields.
            raw_line = lines[i].rstrip('\n')
            parts = [p.strip() for p in raw_line.replace(',', ' ').split()]
            raw: Optional[List[int]] = None
            pid = 0

            def _try_parse_solid(tokens):
                nonlocal pid
                try:
                    pid = int(tokens[1])
                    if len(tokens) >= 10:
                        return [int(tokens[j]) for j in range(2, 10)]
                    elif len(tokens) >= 6:
                        return [int(tokens[j]) for j in range(2, 6)]
                    return None
                except (ValueError, IndexError):
                    return None

            raw = _try_parse_solid(parts)

            # Fixed-width fallback: 8-char columns
            if raw is None and len(raw_line) >= 16:
                fw = 8
                try:
                    n_cols = max(len(raw_line) // fw, 10)
                    fw_parts = [raw_line[j * fw:(j + 1) * fw].strip() for j in range(n_cols)]
                    fw_parts = [p for p in fw_parts if p]
                    raw = _try_parse_solid(fw_parts)
                except (ValueError, IndexError):
                    raw = None

            if raw is not None:
                etype, conn = _classify_solid_element(raw)
                if etype is None:
                    unrecognized_solids += 1
                    etype, conn = ElementType.HEX8, raw
                raw_elements.append((etype, list(conn), pid))

        elif current_section == 'ELEMENT_SHELL':
            # *ELEMENT_SHELL card: EID PID N1 N2 N3 [N4]
            # Two formats exist:
            #   (a) comma / whitespace separated — split() works directly
            #   (b) fixed-width (8 or 10 char columns, no delimiter) — must
            #       slice the raw line. We try (a) first; if it yields fewer
            #       than 5 tokens or the second token isn't an integer we fall
            #       back to fixed-width column slicing.
            raw_line = lines[i].rstrip('\n')
            parts = [p.strip() for p in raw_line.replace(',', ' ').split()]
            raw: Optional[List[int]] = None
            pid = 0

            def _try_parse_shell(tokens):
                nonlocal pid
                try:
                    pid = int(tokens[1])
                    return [int(tokens[j]) for j in range(2, min(len(tokens), 6))]
                except (ValueError, IndexError):
                    return None

            if len(parts) >= 5:
                raw = _try_parse_shell(parts)

            # Fixed-width fallback: LS-DYNA standard is 8-char fields.
            # Card layout: EID(8) PID(8) N1(8) N2(8) N3(8) N4(8)
            if raw is None and len(raw_line) >= 24:
                fw = 8
                try:
                    fw_parts = [raw_line[j * fw:(j + 1) * fw].strip()
                                for j in range(max(len(raw_line) // fw, 6))]
                    fw_parts = [p for p in fw_parts if p]
                    raw = _try_parse_shell(fw_parts)
                except (ValueError, IndexError):
                    raw = None

            if raw is not None and len(raw) >= 3:
                etype, conn = _classify_shell_element(raw)
                if etype is None:
                    unrecognized_shells += 1
                else:
                    raw_elements.append((etype, list(conn), pid))

        elif current_section == 'PART':
            # First data line is the part title; the next is the card whose
            # first field is the pid. Best-effort: blank titles (which strip to
            # empty and are skipped above) fall back to a pid-based name later.
            if part_need == 'title':
                part_pending_title = line
                part_need = 'pid'
            elif part_need == 'pid':
                fields = line.replace(',', ' ').split()
                try:
                    pid = int(fields[0])
                except (ValueError, IndexError):
                    pass
                else:
                    if part_pending_title:
                        part_titles[pid] = part_pending_title
                part_need = None
                current_section = None  # done with this *PART header

        elif current_section == 'SET_SEGMENT':
            parts = [p.strip() for p in line.replace(',', ' ').split()]
            if segment_need == 'title':
                segment_pending_title = line
                segment_need = 'sid'
            elif segment_need == 'sid':
                try:
                    segment_pending_sid = int(parts[0])
                except (ValueError, IndexError):
                    pass
                else:
                    segment_need = 'faces'
            else:
                try:
                    raw = [int(parts[j]) for j in range(0, min(len(parts), 4))]
                except (ValueError, IndexError):
                    raw = []
                if len(raw) >= 3:
                    etype, face = _classify_segment_face(raw)
                    if etype is not None:
                        segment_pending_faces.append(face)

        i += 1

    if current_section == 'SET_SEGMENT':
        finish_pending_segment_set()

    return {
        "node_order": node_order,
        "node_coords": node_coords,
        "raw_elements": raw_elements,
        "raw_segment_sets": raw_segment_sets,
        "part_titles": part_titles,
        "unrecognized_solids": unrecognized_solids,
        "unrecognized_shells": unrecognized_shells,
    }


def _part_name(pid: int, part_titles: Dict[int, str]) -> str:
    """Human-readable name for a PID: the ``*PART`` title, else ``Part_<pid>``."""
    title = part_titles.get(pid)
    return title if title else f"Part_{pid}"


def load_lsdyna(filepath: str) -> VolumeMesh:
    """
    Load a mesh from an LS-DYNA .k file.

    Args:
        filepath: Path to the .k file

    Returns:
        VolumeMesh object
    """
    filepath = Path(filepath)

    # Parse phase: collect nodes in declaration order and element cards keyed by
    # the *original* (1-based) node IDs from the file. We remap to dense 0-based
    # indices at the end. This avoids the trap where a file with sparse IDs
    # (e.g. R_Shoulder_Flesh.k: 42k nodes but max id ~4.3M) inflates nodes_array
    # to (max_id, 3), which poisons every downstream consumer that sizes
    # structures by num_nodes (neighbor lists, JSON payloads, the viewer).
    parsed = _parse_lsdyna(filepath)
    node_order: List[int] = parsed["node_order"]
    node_coords: List[List[float]] = parsed["node_coords"]
    raw_elements: List[Tuple[ElementType, List[int], int]] = parsed["raw_elements"]
    part_titles: Dict[int, str] = parsed["part_titles"]
    unrecognized_solids: int = parsed["unrecognized_solids"]
    unrecognized_shells: int = parsed["unrecognized_shells"]
    node_sets: Dict[str, NodeSet] = {}

    if not node_order:
        raise ValueError("No nodes found in .k file")

    if not raw_elements:
        raise ValueError("No elements found in .k file")

    if unrecognized_solids:
        logger.warning(
            "%d *ELEMENT_SOLID cards had an unrecognized node-collapse pattern "
            "and were kept as HEX8; their Jacobians may flag as invalid.",
            unrecognized_solids,
        )

    if unrecognized_shells:
        logger.warning(
            "%d *ELEMENT_SHELL cards were not a recognisable tri/quad and were "
            "skipped.",
            unrecognized_shells,
        )

    # Compact: dense (N, 3) coords + orig_id -> compact_idx mapping (declaration
    # order). Stored in metadata so the template saver can find each *NODE line's
    # row regardless of how sparse or out-of-order the original IDs were.
    nodes_array = np.asarray(node_coords, dtype=np.float64)
    orig_to_compact = {oid: i for i, oid in enumerate(node_order)}

    # Remap element connectivity through the mapping and bucket by type. The
    # global (declaration-order) element index doubles as the id stored in the
    # per-PID ElementSets below.
    grouped: Dict[ElementType, List[List[int]]] = {}
    part_global_eids: Dict[str, List[int]] = {}
    bad_elements = 0
    first_bad_conn = None
    for gidx, (etype, conn, pid) in enumerate(raw_elements):
        try:
            grouped.setdefault(etype, []).append([orig_to_compact[n] for n in conn])
        except KeyError as exc:
            bad_elements += 1
            if first_bad_conn is None:
                first_bad_conn = (etype, conn, pid, exc.args[0])
            continue
        part_global_eids.setdefault(_part_name(pid, part_titles), []).append(gidx)

    if bad_elements:
        logger.warning(
            "%d element(s) in %s reference node IDs not found in the *NODE section "
            "and were skipped. Segment sets are unaffected.\n"
            "  First bad element: type=%s pid=%s conn=%s missing_id=%s\n"
            "  If this count is large, restart your Python kernel to pick up "
            "the latest lsdyna.py parser fixes.",
            bad_elements, filepath.name,
            first_bad_conn[0] if first_bad_conn else "?",
            first_bad_conn[2] if first_bad_conn else "?",
            first_bad_conn[1] if first_bad_conn else "?",
            first_bad_conn[3] if first_bad_conn else "?",
        )

    groups_array = {
        etype: np.array(rows, dtype=np.int64) for etype, rows in grouped.items()
    }

    # Expose parts (PID groups, named from *PART) as ElementSets so the .k
    # format reaches parity with .feb for the pairing / merge UI. element_ids
    # are global declaration-order indices (== per-type indices for the common
    # single-element-type part); connectivity for morph/export is sourced from
    # extract_lsdyna_element_sets, which the worker uses directly.
    element_sets = {
        name: ElementSet(name=name, element_ids=np.array(eids, dtype=np.int64))
        for name, eids in part_global_eids.items()
    }

    counts = ", ".join(f"{t.value}: {len(a)}" for t, a in groups_array.items())
    logger.info(
        "Loaded %s from %s (nodes: %d declared, %d after compaction, %d part(s))",
        counts, filepath.name, len(node_order), len(nodes_array), len(element_sets),
    )

    mesh = VolumeMesh(
        nodes=nodes_array,
        element_groups=groups_array,
        name=filepath.stem,
        node_sets=node_sets,
        element_sets=element_sets,
        metadata={
            'source_file': str(filepath),
            # List form (parallel to nodes rows). The template saver looks up
            # ``orig_id -> compact_idx`` from this; downstream consumers that
            # need to round-trip original IDs can rebuild the mapping.
            'original_node_ids': node_order,
            # pid -> *PART title (empty if the file has no *PART cards).
            'part_titles': part_titles,
        },
    )

    return mesh


def extract_lsdyna_element_sets(filepath: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Group an LS-DYNA mesh into parts by PID — the ``.k`` analog of
    :func:`fe_personalization.mesh_io.febio.extract_element_sets`.

    Returns ``{part_name: {"type": ElementType, "elements": np.ndarray}}`` where
    ``elements`` is connectivity expressed in **compact node indices** — the
    same index space as ``load_lsdyna(filepath).nodes`` — so the morph worker's
    ``build_compact_mesh`` and the retain-template export can index the loaded
    node array directly (mirroring how the .feb extractor's ``id - 1`` indices
    line up with a dense 1-based .feb load).

    Part names come from ``*PART`` titles, falling back to ``Part_<pid>``. A PID
    that mixes solid element types (rare — e.g. some hexes collapsed to tets) is
    split into ``<name>__<type>`` sub-parts so each entry stays single-type, the
    shape the worker expects.
    """
    parsed = _parse_lsdyna(Path(filepath))
    node_order: List[int] = parsed["node_order"]
    raw_elements: List[Tuple[ElementType, List[int], int]] = parsed["raw_elements"]
    part_titles: Dict[int, str] = parsed["part_titles"]
    orig_to_compact = {oid: i for i, oid in enumerate(node_order)}

    by_pid: Dict[int, Dict[ElementType, List[List[int]]]] = {}
    bad_elements = 0
    for etype, conn, pid in raw_elements:
        try:
            remapped = [orig_to_compact[n] for n in conn]
        except KeyError:
            bad_elements += 1
            continue
        by_pid.setdefault(pid, {}).setdefault(etype, []).append(remapped)

    if bad_elements:
        logger.warning(
            "%d element(s) in %s reference unknown node IDs and were skipped.",
            bad_elements, Path(filepath).name,
        )

    sets: Dict[str, Dict[str, np.ndarray]] = {}
    for pid, type_map in by_pid.items():
        base = _part_name(pid, part_titles)
        if len(type_map) == 1:
            etype, rows = next(iter(type_map.items()))
            sets[base] = {"type": etype, "elements": np.array(rows, dtype=np.int64)}
        else:
            logger.warning(
                "PID %s (%s) mixes element types %s; splitting into per-type "
                "sub-parts.", pid, base, [t.value for t in type_map],
            )
            for etype, rows in type_map.items():
                sets[f"{base}__{etype.value}"] = {
                    "type": etype,
                    "elements": np.array(rows, dtype=np.int64),
                }
    return sets


def extract_lsdyna_segment_sets(filepath: str) -> Dict[str, List[List[int]]]:
    """Return named ``*SET_SEGMENT`` faces in compact node-index space.

    The returned mapping is ``{title_or_sid: faces}``. ``faces`` is a list of
    compact node-index lists; each face has length 3 (triangle) or 4 (quad) and
    indexes the same node array returned by :func:`load_lsdyna`.
    """
    parsed = _parse_lsdyna(Path(filepath))
    node_order: List[int] = parsed["node_order"]
    raw_segment_sets: List[Dict[str, Any]] = parsed["raw_segment_sets"]
    orig_to_compact = {oid: i for i, oid in enumerate(node_order)}

    sets: Dict[str, List[List[int]]] = {}
    for entry in raw_segment_sets:
        sid = int(entry["sid"])
        title = entry.get("title")
        base_name = str(title).strip() if title else str(sid)
        name = base_name
        if name in sets:
            name = f"{base_name}__SID_{sid}"
            suffix = 2
            while name in sets:
                name = f"{base_name}__SID_{sid}_{suffix}"
                suffix += 1

        compact_faces: List[List[int]] = []
        for face in entry["faces"]:
            try:
                compact_faces.append([orig_to_compact[n] for n in face])
            except KeyError as exc:
                raise ValueError(
                    f"*SET_SEGMENT {base_name!r} references unknown node id "
                    f"{exc.args[0]} in {Path(filepath).name}"
                )
        sets[name] = compact_faces
    return sets


def segment_set_to_surface(
    mesh_or_nodes: Union[VolumeMesh, np.ndarray],
    faces: Sequence[Sequence[int]],
    *,
    name: str = "segment_set",
) -> SurfaceMesh:
    """Build a compact triangulated :class:`SurfaceMesh` from segment-set faces.

    ``faces`` must be in the compact node-index space returned by
    :func:`extract_lsdyna_segment_sets`. Quads are split into two triangles. The
    output mesh stores ``metadata['source_vertex_ids']`` so callers can map its
    nodes back to the full loaded LS-DYNA mesh.
    """
    if isinstance(mesh_or_nodes, VolumeMesh):
        full_nodes = mesh_or_nodes.nodes
        source_metadata = mesh_or_nodes.metadata
    else:
        full_nodes = np.asarray(mesh_or_nodes, dtype=np.float64)
        source_metadata = {}

    triangles: List[List[int]] = []
    for face in faces:
        row = [int(v) for v in face]
        if len(row) == 3:
            triangles.append(row)
        elif len(row) == 4:
            triangles.append([row[0], row[1], row[2]])
            triangles.append([row[0], row[2], row[3]])
        else:
            raise ValueError(
                f"segment-set faces must have 3 or 4 nodes; got {len(row)}"
            )

    if not triangles:
        surface = SurfaceMesh(
            nodes=np.zeros((0, 3), dtype=np.float64),
            faces=np.zeros((0, 3), dtype=np.int64),
            face_type=ElementType.TRI3,
            name=name,
            metadata={"source_vertex_ids": np.array([], dtype=np.int64)},
        )
        return surface

    triangles_array = np.asarray(triangles, dtype=np.int64)
    used = np.unique(triangles_array)
    remap = np.full(len(full_nodes), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)

    metadata: Dict[str, Any] = {
        "source_vertex_ids": used,
    }
    if "source_file" in source_metadata:
        metadata["source_file"] = source_metadata["source_file"]
    original_node_ids = source_metadata.get("original_node_ids")
    if original_node_ids is not None:
        metadata["source_node_ids"] = np.asarray(original_node_ids, dtype=np.int64)[used]

    surface = SurfaceMesh(
        nodes=full_nodes[used].copy(),
        faces=remap[triangles_array],
        face_type=ElementType.TRI3,
        name=name,
        metadata=metadata,
    )
    surface.compute_normals()
    return surface


def segment_set_to_shell_mesh(
    mesh_or_nodes: Union[VolumeMesh, np.ndarray],
    faces: Sequence[Sequence[int]],
    *,
    name: str = "segment_set",
) -> VolumeMesh:
    """Build a compact shell :class:`VolumeMesh` from segment-set faces.

    Unlike :func:`segment_set_to_surface`, this preserves the original element
    type: quad faces stay as ``QUAD4`` elements, triangle faces stay as
    ``TRI3``. The result can be saved directly as a ``.k`` file with
    ``save_lsdyna`` to round-trip the original element topology.

    ``faces`` must be in the compact node-index space returned by
    :func:`extract_lsdyna_segment_sets`.
    """
    if isinstance(mesh_or_nodes, VolumeMesh):
        full_nodes = mesh_or_nodes.nodes
        source_metadata = mesh_or_nodes.metadata
    else:
        full_nodes = np.asarray(mesh_or_nodes, dtype=np.float64)
        source_metadata = {}

    tris: List[List[int]] = []
    quads: List[List[int]] = []
    for face in faces:
        row = [int(v) for v in face]
        if len(row) == 3:
            tris.append(row)
        elif len(row) == 4:
            quads.append(row)
        else:
            raise ValueError(
                f"segment-set faces must have 3 or 4 nodes; got {len(row)}"
            )

    if not tris and not quads:
        return VolumeMesh(
            nodes=np.zeros((0, 3), dtype=np.float64),
            element_groups={},
            name=name,
            metadata={"source_vertex_ids": np.array([], dtype=np.int64)},
        )

    # Collect all referenced node indices — keep tris and quads separate
    # because they have different widths and can't be stacked into one array.
    parts_for_unique = []
    if tris:
        parts_for_unique.append(np.asarray(tris, dtype=np.int64).ravel())
    if quads:
        parts_for_unique.append(np.asarray(quads, dtype=np.int64).ravel())
    used = np.unique(np.concatenate(parts_for_unique))

    remap = np.full(len(full_nodes), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)

    element_groups: Dict[ElementType, np.ndarray] = {}
    if tris:
        element_groups[ElementType.TRI3] = remap[np.asarray(tris, dtype=np.int64)]
    if quads:
        element_groups[ElementType.QUAD4] = remap[np.asarray(quads, dtype=np.int64)]

    metadata: Dict[str, Any] = {"source_vertex_ids": used}
    if "source_file" in source_metadata:
        metadata["source_file"] = source_metadata["source_file"]
    original_node_ids = source_metadata.get("original_node_ids")
    if original_node_ids is not None:
        metadata["source_node_ids"] = np.asarray(original_node_ids, dtype=np.int64)[used]

    return VolumeMesh(
        nodes=full_nodes[used].copy(),
        element_groups=element_groups,
        name=name,
        metadata=metadata,
    )


def _format_lsdyna_float(value: float) -> str:
    """Format coordinates compactly for LS-DYNA keyword files."""
    return f"{float(value):.8e}"


def _split_lsdyna_fields(line: str) -> List[str]:
    """Split comma or whitespace LS-DYNA data fields."""
    if "," in line:
        return [part.strip() for part in line.strip().split(",")]
    return line.strip().split()


def _save_lsdyna_from_template(
    mesh: VolumeMesh,
    filepath: Path,
    source_filepath: str,
) -> bool:
    """
    Save by preserving a source keyword file and replacing only node positions.

    This keeps LS-DYNA readers happy for files that depend on the original
    keyword envelope, part definitions, cards, or extra node fields.

    Prefers the byte-for-byte minimal-diff patcher (rewrites only moved nodes'
    coordinate fields, preserving encoding, line endings, and the original
    fixed-width / comma field format). Falls back to the line-rebuild path below
    when no ``*NODE`` line could be matched.
    """
    source_path = Path(source_filepath)
    if not source_path.exists():
        return False

    from .template_patch import patch_lsdyna_nodes

    try:
        if patch_lsdyna_nodes(mesh, str(source_path), str(filepath)):
            return True
    except Exception:
        logger.exception(
            "Byte-for-byte .k patch failed for %s; falling back to "
            "re-serializing saver.", source_filepath,
        )

    lines = source_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    output_lines: List[str] = []
    in_node_section = False
    updated = 0

    # The loader now stores nodes densely (one row per declared node) along with
    # the original ids in metadata. Look up each source *NODE line's id through
    # that mapping. Fall back to ``node_id - 1`` for legacy meshes that were
    # built without it (preserves the old behavior for dense 1-based files).
    orig_ids = mesh.metadata.get("original_node_ids") if mesh.metadata else None
    orig_to_compact: Optional[Dict[int, int]] = (
        {oid: i for i, oid in enumerate(orig_ids)} if orig_ids else None
    )

    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()

        if stripped.startswith("*"):
            in_node_section = upper.startswith("*NODE") and "SET" not in upper
            output_lines.append(line)
            continue

        if in_node_section and stripped and not stripped.startswith("$"):
            parts = _split_lsdyna_fields(line)
            if len(parts) >= 4:
                try:
                    node_id = int(parts[0])
                except ValueError:
                    output_lines.append(line)
                    continue

                if orig_to_compact is not None:
                    node_idx = orig_to_compact.get(node_id, -1)
                else:
                    node_idx = node_id - 1
                if 0 <= node_idx < mesh.num_nodes:
                    node = mesh.nodes[node_idx]
                    extra = parts[4:]
                    fields = [
                        str(node_id),
                        _format_lsdyna_float(node[0]),
                        _format_lsdyna_float(node[1]),
                        _format_lsdyna_float(node[2]),
                        *extra,
                    ]
                    output_lines.append(",".join(fields))
                    updated += 1
                    continue

        output_lines.append(line)

    if updated == 0:
        return False

    filepath.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return True


def save_lsdyna(
    mesh: VolumeMesh,
    filepath: str,
    source_filepath: Optional[str] = None,
) -> None:
    """
    Save a mesh to LS-DYNA .k format.

    Args:
        mesh: VolumeMesh to save
        filepath: Output file path
    """
    filepath = Path(filepath)

    if source_filepath and _save_lsdyna_from_template(mesh, filepath, source_filepath):
        print(f"Saved mesh to {filepath}:")
        print(f"  Updated {mesh.num_nodes} node coordinates")
        return

    with open(filepath, 'w') as f:
        f.write('*KEYWORD\n')
        f.write('$ LS-DYNA keyword file\n')
        f.write(f'$ Generated by FE Personalization\n')
        f.write(f'$ Mesh: {mesh.name}\n')
        f.write('$\n')

        # Write nodes (comma-separated format)
        f.write('*NODE\n')
        f.write('$ nid, x, y, z\n')
        for i, node in enumerate(mesh.nodes):
            # Format: nid, x, y, z (comma-separated)
            f.write(f'{i+1}, {node[0]:.8e}, {node[1]:.8e}, {node[2]:.8e}\n')

        # Write elements (comma-separated format)
        if mesh.element_type == ElementType.HEX8:
            f.write('*ELEMENT_SOLID\n')
            f.write('$ eid, pid, n1, n2, n3, n4, n5, n6, n7, n8\n')
            for i, elem in enumerate(mesh.elements):
                # Format: eid, pid, n1-n8 (1-based, comma-separated)
                nodes_str = ', '.join(str(n+1) for n in elem)
                f.write(f'{i+1}, 1, {nodes_str}\n')

        elif mesh.element_type == ElementType.TET4:
            f.write('*ELEMENT_SOLID\n')
            f.write('$ eid, pid, n1, n2, n3, n4\n')
            for i, elem in enumerate(mesh.elements):
                nodes_str = ', '.join(str(n+1) for n in elem)
                f.write(f'{i+1}, 1, {nodes_str}\n')

        # Shell elements (quad4/tri3) -> *ELEMENT_SHELL. A triangle is written as
        # a degenerate quad (N4 == N3), the form the loader round-trips. Pulled
        # from element_groups so a mixed quad+tri shell mesh writes both.
        shell_groups = []
        if mesh.element_groups:
            for et in (ElementType.QUAD4, ElementType.TRI3):
                g = mesh.element_groups.get(et)
                if g is not None and len(g) > 0:
                    shell_groups.append(g)
        elif mesh.element_type in (ElementType.QUAD4, ElementType.TRI3):
            shell_groups.append(mesh.elements)

        if shell_groups:
            f.write('*ELEMENT_SHELL\n')
            f.write('$ eid, pid, n1, n2, n3, n4\n')
            eid = 1
            for elems in shell_groups:
                for elem in elems:
                    ns = [int(n) + 1 for n in elem]
                    if len(ns) == 3:
                        ns = ns + [ns[-1]]          # degenerate quad for a tri
                    f.write(f'{eid}, 1, ' + ', '.join(str(n) for n in ns) + '\n')
                    eid += 1

        # Write node sets
        for ns_name, node_set in mesh.node_sets.items():
            f.write(f'*SET_NODE_LIST_TITLE\n')
            f.write(f'{ns_name}\n')
            f.write('$     sid       da1       da2       da3       da4    solver\n')
            f.write(f'{1:10d}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}{0.0:10.1f}MECH\n')
            # Write node IDs in groups of 8
            for j in range(0, len(node_set.node_ids), 8):
                chunk = node_set.node_ids[j:j+8]
                nodes_str = ''.join(f'{n+1:10d}' for n in chunk)
                f.write(f'{nodes_str}\n')

        f.write('*END\n')

    print(f"Saved mesh to {filepath}:")
    print(f"  Nodes: {mesh.num_nodes}")
    print(f"  Elements: {mesh.num_elements}")
