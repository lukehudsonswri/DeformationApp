"""Byte-for-byte minimal-diff node patchers for ``.feb`` and ``.k`` files.

The morph + triage flows need to write a result back into the *exact* original
template structure, changing **only the node coordinates that actually moved**
and leaving every other byte untouched (node ids, element blocks, materials,
BCs, formatting, encoding, and line endings).

The classic template savers in ``febio.py`` / ``lsdyna.py`` re-serialize the
file (ElementTree round-trip for ``.feb``; a full line rebuild forced to LF +
UTF-8 for ``.k``). That is correct but **not** byte-for-byte: it rewrites the
XML declaration encoding, reflows whitespace, and normalizes line endings.

These patchers instead scan the source as raw bytes (decoded as Latin-1, which
is a lossless 1:1 byte<->codepoint map so any untouched content re-encodes to
identical bytes), locate the node lines, and rewrite *only* the coordinate
fields that differ from the in-memory mesh. A node whose in-memory coordinate
parse-equals the value already on the line is emitted verbatim — so unmorphed
parts (which carry the exact parsed-original coordinates) stay byte-identical.

Each patcher returns ``True`` on success and ``False`` when the node block is
not the simple one-node-per-line shape it expects (e.g. multiple nodes per
line). On ``False`` the caller should fall back to the re-serializing saver
(correct, just not byte-for-byte) and warn.

--------------------------------------------------------------------------
Vendored from: C:\\Users\\lhudson\\Documents\\Morphing\\fe_personalization\\mesh_io\\template_patch.py
Vendored on:   2026-09-18, unmodified.
Local changes: none yet.
See AGENTS.md "Vendoring policy".
--------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _format_coord(value: float) -> str:
    """Format a coordinate so it round-trips the full float64 value.

    ``repr`` of a Python float is the shortest decimal string that parses back
    to the identical float64 — so it never *loses* precision relative to the
    in-memory value, and tends to read cleanly (``52.62988`` rather than a
    ``.17g`` smear of trailing digits). This is only ever used for the handful
    of coordinates that actually moved; unchanged coordinates are emitted from
    the original bytes.
    """
    return repr(float(value))


def _orig_to_compact(mesh) -> Optional[Dict[int, int]]:
    """Map original node id -> dense row index, or ``None`` for legacy meshes.

    Mirrors the lookup the existing template savers use: prefer the
    ``original_node_ids`` list stashed in metadata (the loader compacts sparse
    ids into a dense array and records the originals there); fall back to
    ``id - 1`` when a caller hand-built the mesh with dense 1-based ids.
    """
    orig_ids = mesh.metadata.get("original_node_ids") if mesh.metadata else None
    if not orig_ids:
        return None
    return {oid: i for i, oid in enumerate(orig_ids)}


def _node_row(mesh, node_id: int, id_map: Optional[Dict[int, int]]) -> int:
    """Resolve a source node id to its row in ``mesh.nodes`` (or -1)."""
    if id_map is not None:
        return id_map.get(node_id, -1)
    return node_id - 1


# A comma-delimited coordinate field, splitting out the surrounding whitespace
# so it can be preserved when only the numeric content is rewritten:
#   " 135.385429"  ->  (" ", "135.385429", "")
_FIELD_WS_RE = re.compile(r"^(\s*)(\S*)(\s*)$")


def _rewrite_comma_fields(
    text: str, replacements: Dict[int, str]
) -> Optional[str]:
    """Rewrite selected comma-separated fields, preserving every separator.

    ``text`` is split on ``','`` and only the fields named in ``replacements``
    have their numeric content swapped (their surrounding whitespace is kept),
    so ``", "`` vs ``","`` spacing and any trailing fields survive untouched.
    Returns ``None`` if the field count can't accommodate the replacements.
    """
    fields = text.split(",")
    if any(idx >= len(fields) for idx in replacements):
        return None
    out: List[str] = []
    for idx, field in enumerate(fields):
        new_value = replacements.get(idx)
        if new_value is None:
            out.append(field)
            continue
        m = _FIELD_WS_RE.match(field)
        if m is None:  # pragma: no cover - the regex matches any string
            out.append(field)
            continue
        out.append(m.group(1) + new_value + m.group(3))
    return ",".join(out)


def _format_coord_fixed(value: float, token: str, width: int) -> str:
    """Format ``value`` to drop into a fixed-width LS-DYNA column.

    Fixed-width ``*NODE`` lines have **no separators** between coordinate fields,
    so a free-form ``repr(float)`` (which can be far longer than the column)
    would run straight into the next field. Instead, reproduce the original
    ``token``'s numeric style -- scientific vs plain decimal, and its decimal
    precision -- shrinking the precision only if needed to fit ``width``. The
    result reads identically to the file's untouched nodes and stays aligned
    once right-justified into the column.
    """
    value = float(value)
    scientific = "e" in token or "E" in token
    exp_upper = "E" in token
    mantissa = re.split(r"[eE]", token, maxsplit=1)[0]
    prec = len(mantissa.split(".", 1)[1]) if "." in mantissa else 0
    prec = min(prec, 16)
    for p in range(prec, -1, -1):
        if scientific:
            s = f"{value:.{p}e}"
            if exp_upper:
                s = s.replace("e", "E")
        else:
            s = f"{value:.{p}f}"
        if len(s) <= width:
            return s
    # Magnitude too large for the column even at precision 0 (the source file
    # couldn't have held it either) -- best effort; rjust below leaves it as-is.
    if scientific:
        return f"{value:.0E}" if exp_upper else f"{value:.0e}"
    return f"{value:.0f}"


def _rewrite_whitespace_fields(
    line: str, replacements: Dict[int, float]
) -> str:
    """Rewrite selected whitespace-separated coordinate fields in place.

    Tokens are located by span; a replaced coordinate is formatted to match the
    original token's style/precision (:func:`_format_coord_fixed`) and
    right-justified into the same (leading-whitespace + token) width, so
    fixed-width LS-DYNA columns stay aligned and never run together. Separators
    and any trailing fields (and a trailing ``\\r``) are emitted verbatim.
    """
    out: List[str] = []
    prev_end = 0
    for idx, m in enumerate(re.finditer(r"\S+", line)):
        gap = line[prev_end:m.start()]
        if idx not in replacements:
            out.append(gap + m.group(0))
        else:
            width = len(gap) + (m.end() - m.start())
            formatted = _format_coord_fixed(replacements[idx], m.group(0), width)
            out.append(formatted.rjust(width))
        prev_end = m.end()
    out.append(line[prev_end:])
    return "".join(out)


# ``\t\t<node id="444">43.27, 135.38, 66.26</node>\r``
#   g1 = prefix incl. the opening tag through '>'
#   g2 = node id
#   g3 = inner coordinate text
#   g4 = '</node>' + any trailing whitespace / '\r'
_FEB_NODE_RE = re.compile(r'^(\s*<node\s+id="(\d+)"\s*>)(.*?)(</node>\s*)$')


def patch_febio_nodes(mesh, source_path: str, dst_path: str) -> bool:
    """Write ``mesh`` into ``dst_path`` as a minimal-diff patch of ``source_path``.

    Only the coordinate fields of moved nodes change; every other byte of the
    source ``.feb`` is preserved. Returns ``False`` (writing nothing) if a
    ``<Nodes>`` block isn't the expected one-node-per-line shape, so the caller
    can fall back to the re-serializing saver.
    """
    raw = Path(source_path).read_bytes()
    text = raw.decode("latin-1")
    lines = text.split("\n")

    id_map = _orig_to_compact(mesh)
    nodes = mesh.nodes
    n_nodes = len(nodes)

    in_nodes = False
    changed = 0
    for i, line in enumerate(lines):
        stripped = line.strip()

        if not in_nodes:
            if stripped.startswith("<Nodes ") or stripped == "<Nodes>" or stripped.startswith("<Nodes>"):
                in_nodes = True
            continue

        # Inside a <Nodes> block.
        if "</Nodes>" in stripped:
            # Closing tag. A line carrying both a node and the close tag is not
            # the simple shape we patch -> fall back.
            if "<node" in stripped:
                logger.warning(
                    "patch_febio_nodes: <Nodes> close shares a line with a node "
                    "in %s; falling back to re-serializing saver.", source_path,
                )
                return False
            in_nodes = False
            continue

        if "<node" not in stripped:
            continue

        m = _FEB_NODE_RE.match(line)
        if m is None:
            # A <node> that isn't a clean single-line element (multiple nodes
            # per line, split coordinates, etc.). Bail to the safe saver.
            logger.warning(
                "patch_febio_nodes: non-line-oriented node entry in %s; "
                "falling back to re-serializing saver.", source_path,
            )
            return False

        node_id = int(m.group(2))
        inner = m.group(3)
        coords = inner.split(",")
        if len(coords) != 3:
            logger.warning(
                "patch_febio_nodes: node %d has %d coordinate fields in %s; "
                "falling back to re-serializing saver.",
                node_id, len(coords), source_path,
            )
            return False

        row = _node_row(mesh, node_id, id_map)
        if not (0 <= row < n_nodes):
            continue  # unknown node -> leave at template coordinates

        try:
            parsed = [float(c) for c in coords]
        except ValueError:
            logger.warning(
                "patch_febio_nodes: unparseable coordinates for node %d in %s; "
                "falling back to re-serializing saver.", node_id, source_path,
            )
            return False

        target = nodes[row]
        replacements = {
            k: _format_coord(target[k])
            for k in range(3)
            if float(target[k]) != parsed[k]
        }
        if not replacements:
            continue  # coordinate unchanged -> keep the original bytes

        new_inner = _rewrite_comma_fields(inner, replacements)
        if new_inner is None:  # pragma: no cover - guarded by len==3 above
            return False
        lines[i] = m.group(1) + new_inner + m.group(4)
        changed += 1

    Path(dst_path).write_bytes("\n".join(lines).encode("latin-1"))
    logger.info(
        "patch_febio_nodes: wrote %s (%d node line(s) changed)", dst_path, changed
    )
    return True


def patch_lsdyna_nodes(mesh, source_path: str, dst_path: str) -> bool:
    """Write ``mesh`` into ``dst_path`` as a minimal-diff patch of ``source_path``.

    Scans the ``*NODE`` section(s) and rewrites only the coordinate fields of
    moved nodes, preserving the source's field format (free comma/space or
    fixed-width), any trailing per-node fields, the encoding, and line endings.
    Returns ``False`` (writing nothing) if no node line could be matched, so the
    caller can fall back to the re-serializing saver.
    """
    raw = Path(source_path).read_bytes()
    text = raw.decode("latin-1")
    lines = text.split("\n")

    id_map = _orig_to_compact(mesh)
    nodes = mesh.nodes
    n_nodes = len(nodes)

    in_node_section = False
    changed = 0
    saw_node_line = False
    for i, line in enumerate(lines):
        body = line[:-1] if line.endswith("\r") else line
        stripped = body.strip()

        if stripped.startswith("*"):
            upper = stripped.upper()
            in_node_section = upper.startswith("*NODE") and "SET" not in upper
            continue

        if not in_node_section or not stripped or stripped.startswith("$"):
            continue

        is_comma = "," in body
        if is_comma:
            fields = [f.strip() for f in body.split(",")]
        else:
            fields = body.split()
        if len(fields) < 4:
            continue

        try:
            node_id = int(fields[0])
        except ValueError:
            continue

        row = _node_row(mesh, node_id, id_map)
        if not (0 <= row < n_nodes):
            saw_node_line = True
            continue

        try:
            parsed = [float(fields[1]), float(fields[2]), float(fields[3])]
        except ValueError:
            continue

        saw_node_line = True
        target = nodes[row]
        # *NODE fields: 0=id, 1=x, 2=y, 3=z, 4+=extra (constraints, etc.).
        moved = {
            k + 1: float(target[k])
            for k in range(3)
            if float(target[k]) != parsed[k]
        }
        if not moved:
            continue  # coordinates unchanged -> keep original bytes

        if is_comma:
            # Comma-delimited fields are separator-safe, so emit the shortest
            # round-tripping decimal (full float64 precision).
            new_body = _rewrite_comma_fields(
                body, {idx: _format_coord(v) for idx, v in moved.items()}
            )
            if new_body is None:  # pragma: no cover - guarded by len>=4 above
                continue
        else:
            # Fixed-width fields: format each coordinate to its column.
            new_body = _rewrite_whitespace_fields(body, moved)
        lines[i] = new_body + "\r" if line.endswith("\r") else new_body
        changed += 1

    if not saw_node_line:
        logger.warning(
            "patch_lsdyna_nodes: no *NODE lines matched in %s; falling back to "
            "re-serializing saver.", source_path,
        )
        return False

    Path(dst_path).write_bytes("\n".join(lines).encode("latin-1"))
    logger.info(
        "patch_lsdyna_nodes: wrote %s (%d node line(s) changed)", dst_path, changed
    )
    return True
