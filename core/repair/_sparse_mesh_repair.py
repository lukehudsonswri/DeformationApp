# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:10:56 2026

@author: sshaffer

NOTE: This file was edited with a GitHub Copilot AI agent on 2026-06-05,
2026-06-10, 2026-07-10, 2026-07-13, 2026-07-28, 2026-07-29, 2026-07-30,
and 2026-08-05.

*** SPARSE/TRUST-CONSTR EXPERIMENTAL FORK ***
This file is a fork of lsdyna_helper_functions.py (which remains the
validated, unchanged reference implementation -- DO NOT edit that file to
keep this one in sync; they are intentionally independent copies). The
ONLY functional difference is inside repair_inverted_elements: the
optimizer backend was switched from SciPy's SLSQP (dense Jacobian,
O(constraints x variables^2)-or-worse per-iteration cost) to SciPy's
trust-constr (accepts a SPARSE constraint Jacobian directly, plus an
EXACT closed-form objective Hessian, so per-iteration cost scales with
the true number of non-zero Jacobian entries instead of the full dense
size). This was built specifically to handle repair-zone pockets too
large for SLSQP to converge on in practical time (e.g. a single pocket
with 1000+ free nodes / 4000+ constrained elements, observed to make zero
progress -- not even one completed iteration -- after many CPU-minutes
of SLSQP on a real production mesh). See repair_inverted_elements' own
docstring for the full technical rationale. Everything else in this file
(node ID preservation, non-regression floor logic, volume/Jacobian
formulas, continuation-stage schedule, escalation-ladder compatibility)
is UNCHANGED from the validated original.

VERIFICATION: see test_data/test_read_node_element
                  test_data/test_connectivity
                  test_data/test_get_model_info
TODO: Code verification is pending. Confirm tet4/tri3 test data, curve parser
    test data, and update documentation after verification is complete.

--------------------------------------------------------------------------
Vendored from: C:\\Users\\lhudson\\Desktop\\Projects\\ForSarah\\Sparse\\lsdyna_helper_functions_sparse.py
Vendored on:   2026-09-18, unmodified (copied verbatim; see AGENTS.md
               "Vendoring policy" for why this is a copy and not the
               original author's file -- this file is self-contained,
               importing only stdlib + numpy + pandas, so no sibling
               modules from Sparse/ needed to come with it).
Local changes: none yet. If this file is edited locally, record the
               change here rather than in the upstream project (which,
               per its own header above, has its own independent copy
               discipline for lsdyna_helper_functions.py vs. this fork).
Used by (AGENTS.md section 3.3): core/repair/__init__.py exposes a
               curated subset of this file's public functions -- the
               mesh-quality detection, pocket-based repair, and
               solid/shell connectivity functions -- as this project's
               repair escalation ladder (remesh/quality.py).
--------------------------------------------------------------------------
"""
import os
import pandas as pd
import numpy as np
import re
import warnings
import collections


def get_model_information(k_file_path):
    """
    Parse part, section, and material metadata from an LS-DYNA .k model file.

    Returns one row per *PART with linked section/material details:
    - part_name
    - part_id
    - section_id
    - section_type (SOLID/SHELL when detected)
    - material_id
    - material_type (from *MAT_<type>[_TITLE] keyword)
    - material_name

    Parsing notes:
    - Comment lines beginning with '$' or '$#' are ignored.
    - Name/title lines are treated as optional for *PART, *SECTION_*, and *MAT_* cards.
    - Numeric ID cards are parsed from the next valid non-comment line.
    """
    def _is_comment_or_blank(line):
        s = line.strip()
        return (not s) or s.startswith('$')

    def _tokenize(line):
        s = line.strip()
        if ',' in s:
            return [tok.strip() for tok in s.split(',') if tok.strip()]
        return s.split()

    def _to_int(token):
        try:
            return int(float(token))
        except (TypeError, ValueError):
            return None

    def _to_id_int(token):
        """
        Parse integer-like ID tokens only (e.g., 1000 or 1000.0).
        Avoid treating general floats like 2.5 as IDs.
        """
        if token is None:
            return None
        s = str(token).strip()
        if re.fullmatch(r'[+-]?\d+(?:\.0+)?', s):
            return int(float(s))
        return None

    def _looks_like_section_id_line(tokens):
        """
        Heuristic for section header numeric cards.
        Expected shape starts with section ID followed by integer-like elform.
        """
        if not tokens or len(tokens) < 2:
            return False
        return _to_id_int(tokens[0]) is not None and _to_id_int(tokens[1]) is not None

    def _next_data_line(lines, start_idx):
        idx = start_idx
        while idx < len(lines):
            if not _is_comment_or_blank(lines[idx]):
                return idx
            idx += 1
        return None

    def _read_optional_name_then_numeric(lines, start_idx):
        """
        Read either:
          1) numeric data line directly, or
          2) one optional name line followed by numeric data line.

        Returns: (name_or_none, numeric_tokens_or_none, numeric_idx_or_none)
        """
        first_idx = _next_data_line(lines, start_idx)
        if first_idx is None or lines[first_idx].lstrip().startswith('*'):
            return None, None, None

        first_tokens = _tokenize(lines[first_idx])
        if first_tokens and _to_int(first_tokens[0]) is not None:
            return None, first_tokens, first_idx

        name = lines[first_idx].strip()
        second_idx = _next_data_line(lines, first_idx + 1)
        if second_idx is None or lines[second_idx].lstrip().startswith('*'):
            return name, None, None

        second_tokens = _tokenize(lines[second_idx])
        if second_tokens and _to_int(second_tokens[0]) is not None:
            return name, second_tokens, second_idx

        return name, None, None

    with open(k_file_path, 'r') as file:
        lines = file.readlines()

    parts = []
    sections_by_id = {}
    material_type_by_id = {}
    materials_by_id = {}

    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        upper = raw.upper()

        if not raw or raw.startswith('$'):
            i += 1
            continue

        # Parse *PART with optional name line then pid/secid/mid line.
        if upper.startswith('*PART'):
            part_name = None
            data_idx = _next_data_line(lines, i + 1)

            if data_idx is not None and not lines[data_idx].lstrip().startswith('*'):
                first_tokens = _tokenize(lines[data_idx])

                # If first line after *PART is not the numeric card, treat as part name.
                if len(first_tokens) < 3 or _to_int(first_tokens[0]) is None:
                    part_name = lines[data_idx].strip()
                    data_idx = _next_data_line(lines, data_idx + 1)

                if data_idx is not None and not lines[data_idx].lstrip().startswith('*'):
                    data_tokens = _tokenize(lines[data_idx])
                    if len(data_tokens) >= 3:
                        pid = _to_int(data_tokens[0])
                        secid = _to_int(data_tokens[1])
                        mid = _to_int(data_tokens[2])
                        if pid is not None:
                            parts.append(
                                {
                                    'part_name': part_name,
                                    'part_id': pid,
                                    'section_id': secid,
                                    'material_id': mid,
                                }
                            )
                    i = data_idx + 1
                    continue

        # Parse section cards and map secid -> section type.
        if upper.startswith('*SECTION_'):
            section_type = None
            if 'SOLID' in upper:
                section_type = 'SOLID'
            elif 'SHELL' in upper:
                section_type = 'SHELL'

            # SECTION blocks may contain multiple records; parse until next keyword.
            cursor = i + 1
            while cursor < len(lines):
                row_idx = _next_data_line(lines, cursor)
                if row_idx is None:
                    i = len(lines)
                    break

                if lines[row_idx].lstrip().startswith('*'):
                    i = row_idx
                    break

                row_tokens = _tokenize(lines[row_idx])

                # Case 1: numeric section header line directly.
                if _looks_like_section_id_line(row_tokens):
                    secid = _to_id_int(row_tokens[0])
                    if secid is not None and section_type is not None:
                        sections_by_id[secid] = section_type
                    cursor = row_idx + 1
                    continue

                # Case 2: likely a name/title line followed by numeric header line.
                data_idx = _next_data_line(lines, row_idx + 1)
                if data_idx is None:
                    i = len(lines)
                    break

                if lines[data_idx].lstrip().startswith('*'):
                    i = data_idx
                    break

                data_tokens = _tokenize(lines[data_idx])
                if _looks_like_section_id_line(data_tokens):
                    secid = _to_id_int(data_tokens[0])
                    if secid is not None and section_type is not None:
                        sections_by_id[secid] = section_type
                    cursor = data_idx + 1
                    continue

                cursor = row_idx + 1
            else:
                i = len(lines)

            continue

        # Parse material cards and map mid -> material name.
        if upper.startswith('*MAT_'):
            keyword_material_type = upper.replace('*MAT_', '').replace('_TITLE', '')
            is_mat_add = upper.startswith('*MAT_ADD')
            parsed_name, data_tokens, data_idx = _read_optional_name_then_numeric(lines, i + 1)

            if data_tokens is not None:
                mid = _to_int(data_tokens[0])
                if mid is not None:
                    existing_type = material_type_by_id.get(mid)
                    existing_is_mat_add = (
                        isinstance(existing_type, str)
                        and existing_type.startswith('ADD')
                    )

                    # If both base *MAT and *MAT_ADD exist for the same MID,
                    # keep the base *MAT information and ignore *MAT_ADD.
                    should_replace = (
                        existing_type is None
                        or (existing_is_mat_add and not is_mat_add)
                    )

                    if should_replace:
                        material_type_by_id[mid] = keyword_material_type
                        material_name = parsed_name if parsed_name else keyword_material_type
                        materials_by_id[mid] = material_name
                i = data_idx + 1
                continue

        i += 1

    if not parts:
        return pd.DataFrame(
            columns=[
                'part_name',
                'part_id',
                'section_id',
                'section_type',
                'material_id',
                'material_type',
                'material_name'
            ]
        )

    model_info_df = pd.DataFrame(parts)
    model_info_df['section_type'] = model_info_df['section_id'].map(sections_by_id)
    model_info_df['material_type'] = model_info_df['material_id'].map(material_type_by_id)
    model_info_df['material_name'] = model_info_df['material_id'].map(materials_by_id)
    model_info_df = model_info_df[
        [
            'part_name',
            'part_id',
            'section_id',
            'section_type',
            'material_id',
            'material_type',
            'material_name'
        ]
    ]
    return model_info_df


def read_nodes_from_kfile(k_file_path):
    """
    Read the *NODE section from an LS-DYNA .k file.

    This function supports mixed files where *NODE is followed by other keyword
    sections (for example *ELEMENT_SOLID). Node parsing stops at the next keyword
    line (a line beginning with '*'), not only at *END.

    Supported node line formats:
    - Comma-delimited:  NodeID, x, y, z
    - Whitespace-delimited: NodeID x y z
    - Fixed-width LS-DYNA style: [8 char NodeID][16 char x][16 char y][16 char z]
      including cases where coordinates appear concatenated.
    """
    in_node_section = False
    nodes = []

    with open(k_file_path, 'r') as file:
        for line in file:
            line_stripped = line.strip()
            line_upper = line_stripped.upper()
            # EXACT keyword match (first whitespace token), not startswith --
            # LS-DYNA has several distinct keywords that share the '*NODE'
            # PREFIX but are unrelated cards with different data formats
            # (e.g. '*NODE_MERGE_SET', '*NODE_TRANSFORM', '*NODE_RIGID_
            # SURFACE'). A prefix match would misinterpret one of those as
            # the START of a fresh *NODE section and try to parse its
            # (differently-formatted) data lines as NodeID/x/y/z coordinates
            # -- the same false-positive risk found and fixed in
            # read_elements_from_kfile for '*ELEMENT_SHELL_THICKNESS'.
            keyword_token = line_upper.split()[0] if line_upper else ''

            if keyword_token == '*NODE':
                in_node_section = True
                continue

            if in_node_section and line_stripped.startswith('*'):
                break

            if not in_node_section:
                continue

            if not line_stripped or line_stripped.startswith('$'):
                continue

            parsed = None

            if ',' in line_stripped:
                parts = [x.strip() for x in line_stripped.split(',') if x.strip()]
                if len(parts) >= 4:
                    parsed = [int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])]
            else:
                parts = line_stripped.split()
                if len(parts) >= 4:
                    # Standard whitespace-delimited: NodeID x y z [optional extras]
                    parsed = [int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])]

            # Fallback for fixed-width / partially-concatenated coordinate lines.
            # Extracts the NodeID then uses regex to find all three float tokens in
            # the remainder — handles both scientific notation and decimal formats
            # even when coordinates run together without whitespace separators.
            if parsed is None:
                m = re.match(r'^\s*(\d+)\s*(.+)$', line.rstrip('\n'))
                if m:
                    node_id = int(m.group(1))
                    remainder = m.group(2)
                    float_pattern = r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?'
                    nums = re.findall(float_pattern, remainder)
                    if len(nums) >= 3:
                        parsed = [node_id, float(nums[0]), float(nums[1]), float(nums[2])]

            if parsed is not None:
                nodes.append(parsed)

    if not nodes:
        return pd.DataFrame(columns=['NodeID', 'x', 'y', 'z'])

    nodes_df = pd.DataFrame(nodes, columns=['NodeID', 'x', 'y', 'z'])
    nodes_df['NodeID'] = nodes_df['NodeID'].astype(int)
    nodes_df['x'] = nodes_df['x'].astype(float)
    nodes_df['y'] = nodes_df['y'].astype(float)
    nodes_df['z'] = nodes_df['z'].astype(float)
    return nodes_df

def write_nodes_to_kfile(node_df, outname):
    print(f'Writing to {outname}')
    with open(outname, 'w') as f:
        f.write("*KEYWORD\n")
        f.write("*NODE\n")
        for _, row in node_df.iterrows():
            f.write(
                f"{int(row['NodeID']):8d}"
                f"{row['x']:16.8f}"
                f"{row['y']:16.8f}"
                f"{row['z']:16.8f}\n"
            )
        f.write("*END\n")
    print(f"Data written to {outname}")

def write_mesh_to_kfile(nodes_df, elements_df, outname, element_type='SOLID'):
    """
#TODO: VERIFY THIS WORKS -- added by AI agent 2026-07-30, verification pending.

    Write a combined *NODE + *ELEMENT_SOLID/*ELEMENT_SHELL LS-DYNA k-file.

    Combines the node-card format used by write_nodes_to_kfile with an
    element card section for either solid or shell elements, both written
    into one *KEYWORD file. Every element card writes  10
    fields (eid, pid, n1..n8),-- SOLID cards fill all 8 node
    slots with real/repeated node IDs, while SHELL cards only use n1..n4 and
    zero-pad n5..n8 (LS-PrePost writes literal 0, not blank/omitted fields):
        SOLID, 4 unique nodes (tet4)  : n1 n2 n3 n4 n4 n4 n4 n4
        SOLID, 6 unique nodes (penta6): n1 n2 n3 n4 n5 n6 n6 n6
        SOLID, 8 unique nodes (hex8)  : n1 n2 n3 n4 n5 n6 n7 n8
        SHELL, 3 unique nodes (tri)   : n1 n2 n3 n3 0 0 0 0
        SHELL, 4 unique nodes (quad)  : n1 n2 n3 n4 0 0 0 0

    MIXED-TYPE SUPPORT: if elements_df has an 'ETYPE' column (as produced
    by read_elements_from_kfile / read_all_elements_from_kfile), each row
    is written under ITS OWN type's section ('SOLID' rows go under
    *ELEMENT_SOLID, 'SHELL' rows under *ELEMENT_SHELL) regardless of the
    element_type argument -- this is what lets a genuine whole-model file
    (solid muscle/organ parts mixed with shell skin/fascia parts) survive
    a full read -> repair -> write round trip without losing an entire
    element type. Rows with a missing/NaN ETYPE (e.g. an elements_df built
    by hand without that column) fall back to the element_type argument.
    If elements_df has NO 'ETYPE' column at all, behavior is UNCHANGED from
    before: every row is written as element_type (all rows assumed to be
    that one type) -- this preserves existing callers exactly.

    Inputs:
        nodes_df     (pd.DataFrame): Node table; columns 'NodeID', 'x', 'y', 'z'
                                     (e.g. from read_nodes_from_kfile). Every
                                     row is written, including nodes not
                                     referenced by any element (e.g. floating
                                     landmark points) -- this function never
                                     drops a node based on element usage.
        elements_df  (pd.DataFrame): Elements table; columns 'EID', 'PID',
                                     'NID1'..'NID8' (e.g. from
                                     read_elements_from_kfile), optionally
                                     with an 'ETYPE' column (see above).
        outname      (str): Output .k file path.
        element_type (str): 'SOLID' or 'SHELL'. Used for EVERY row when
                            elements_df has no 'ETYPE' column (or as the
                            fallback for rows with a missing/NaN ETYPE).
                            Default 'SOLID'.

    Returns:
        None. Writes outname to disk.

    Raises:
        ValueError: If element_type (or any row's resolved type) is not
            'SOLID' or 'SHELL', or if an element row has a number of
            unique real nodes unsupported for its type (SOLID: 4, 6, 8;
            SHELL: 3, 4).
    """
    element_type = element_type.upper()
    if element_type not in ('SOLID', 'SHELL'):
        raise ValueError("element_type must be 'SOLID' or 'SHELL'.")

    node_cols = _node_columns(elements_df)

    def _write_element_rows(f, rows_df, row_type):
        """Write one *ELEMENT_<row_type> section for the given row subset."""
        f.write(f"*ELEMENT_{row_type}\n")
        for _, row in rows_df.iterrows():
            eid = int(row['EID'])
            pid = int(row['PID'])
            unique_nids = _valid_element_nodes(row, node_cols)
            n = len(unique_nids)

            if row_type == 'SOLID':
                if n == 4:
                    card_nids = unique_nids + [unique_nids[3]] * 4
                elif n == 6:
                    card_nids = unique_nids + [unique_nids[5]] * 2
                elif n == 8:
                    card_nids = unique_nids
                else:
                    raise ValueError(
                        f"Unsupported SOLID element node count for EID {eid}: "
                        f"found {n} unique real nodes; supported counts are 4, 6, 8."
                    )
            else:  # SHELL
                if n == 3:
                    card_nids = unique_nids + [unique_nids[2]] + [0, 0, 0, 0]
                elif n == 4:
                    card_nids = unique_nids + [0, 0, 0, 0]
                else:
                    raise ValueError(
                        f"Unsupported SHELL element node count for EID {eid}: "
                        f"found {n} unique real nodes; supported counts are 3, 4."
                    )

            fields = ''.join(f"{nid:8d}" for nid in card_nids)
            f.write(f"{eid:8d}{pid:8d}{fields}\n")

    # Resolve which rows go under which section. Preserves the exact
    # pre-existing single-type behavior when no ETYPE column is present
    # (the whole elements_df is written under one element_type section).
    if 'ETYPE' in elements_df.columns:
        resolved_type = elements_df['ETYPE'].where(
            elements_df['ETYPE'].notna(), element_type
        ).str.upper()
        sections = []
        for row_type in ('SOLID', 'SHELL'):
            subset = elements_df[resolved_type == row_type]
            if not subset.empty:
                sections.append((row_type, subset))
        unknown_types = set(resolved_type.unique()) - {'SOLID', 'SHELL'}
        if unknown_types:
            raise ValueError(
                f"Unsupported ETYPE value(s) found: {unknown_types}. "
                "Only 'SOLID' and 'SHELL' are supported."
            )
    else:
        sections = [(element_type, elements_df)]

    print(f'Writing to {outname}')
    with open(outname, 'w') as f:
        f.write("*KEYWORD\n")

        # Skip the *NODE section header entirely when there are zero
        # nodes to write (e.g. an elements-only output file under
        # write_multi_part_mesh's own_node_ids-based split -- see that
        # function's docstring) -- an empty *NODE section immediately
        # followed by another keyword is harmless to LS-DYNA, but omitting
        # it entirely keeps a genuinely elements-only file's own content
        # unambiguous (no dangling keyword with zero cards under it).
        if not nodes_df.empty:
            f.write("*NODE\n")
            for _, row in nodes_df.iterrows():
                f.write(
                    f"{int(row['NodeID']):8d}"
                    f"{row['x']:16.8f}"
                    f"{row['y']:16.8f}"
                    f"{row['z']:16.8f}\n"
                )

        for row_type, subset in sections:
            _write_element_rows(f, subset, row_type)

        f.write("*END\n")
    print(f"Data written to {outname}")

def read_elements_from_kfile(k_file_path, element_type=None):
    """
    Read elements from an LS-DYNA .k file with variable node counts.

    Supports *ELEMENT_SOLID, *ELEMENT_SHELL, AND *ELEMENT_SHELL_THICKNESS
    sections in mixed files (see below for why the THICKNESS variant needs
    its own special handling, not just prefix-matching *ELEMENT_SHELL).
    Each data line is expected to be whitespace- or comma-separated:
        Format: EID  PID  N1  N2  N3  [N4]  [N5]  [N6]  [N7]  [N8]

    *** FIXED-WIDTH FALLBACK *** (added after a real production mesh
    crashed with `OverflowError: Python int too large to convert to C
    long` while reading *ELEMENT_SOLID): LS-DYNA's classic fixed
    short-format element card is rigid 8-character-wide columns with NO
    space required between adjacent fields. If a single field (most
    commonly a large NodeID from a renumbering/offset scheme) is exactly
    8 digits, it fills its whole column with zero leading padding and
    silently GLUES to its neighbour under plain whitespace-splitting --
    producing one bogus multi-field "token" (observed: 34-39 fused
    digits). Any token longer than 10 digits is unambiguously such a
    fusion artifact (no real EID/PID/NodeID in this pipeline gets
    anywhere close to that many digits), and triggers re-deriving that
    ENTIRE line's fields via fixed 8-character column slicing of the
    untouched original line instead -- immune to the problem since it
    never depends on inter-field whitespace. Mirrors the equivalent,
    already-shipped fixed-width fallback in read_nodes_from_kfile.

    Solid elements can have 4-node tetrahedra, 6-node pentahedra, or 8-node hexahedra.
    Shell elements can have 3 nodes (triangles) through 4 nodes (quads).

    Element sections are terminated by the next '*' keyword or end of file.
    Comment lines beginning with '$' are skipped.

    *** *ELEMENT_SHELL_THICKNESS SUPPORT *** (added after a real production
    mesh, HBM_F, was found to define 55,168 of its 629,565 shell elements --
    ~8.8% of the total -- EXCLUSIVELY under this keyword, with ZERO overlap
    against elements defined under a plain *ELEMENT_SHELL section; earlier
    code treated these as entirely absent, not duplicates, so this was a
    real, silent 8.8% shell-element loss, not a cosmetic gap). This card's
    data lines ALTERNATE an element-defining line (EID, PID, N1..N4 --
    otherwise IDENTICAL in layout to a plain *ELEMENT_SHELL card) with a
    per-node thickness-VALUE line (4 floats, no element/node IDs) for that
    same element. This function now reads the element-defining line
    exactly like a normal shell element (tagged ETYPE='SHELL' -- these are
    genuine shells, just with per-node thickness data attached, not a
    distinct element type) and explicitly SKIPS the following
    thickness-value line rather than attempting to parse it as more node
    IDs (which would corrupt the row with bogus extra "node" columns) or
    as a brand-new element (which would silently create a phantom/
    duplicate EID -- the ORIGINAL bug this function's exact-keyword-match
    guard was built to prevent, before this fix, by simply skipping the
    ENTIRE section rather than reading it correctly). Assumes exactly one
    thickness-value line follows each element-defining line (matches every
    real *ELEMENT_SHELL_THICKNESS occurrence found in HBM_F) -- LS-DYNA's
    format also permits optional additional cards for other per-element
    THICKNESS-family options; if a file using those is ever encountered,
    this simple line-skip logic would need revisiting.

    Parameters:
        k_file_path (str): Path to the LS-DYNA .k file.
        element_type (str or None): Optional filter: 'SOLID' or 'SHELL'.
                        If None, the function uses the first element section type found
                        and aggregates all sections of that same type across the file.

    Returns:
        pd.DataFrame: columns ['EID', 'PID', 'NID1', 'NID2', 'NID3', 'NID4', 'NID5', 'NID6', 'NID7', 'NID8', 'ETYPE']
                      - All rows have 8 node columns; unused positions are NaN
                      - Zero node placeholders are treated as unused positions
                      - Rows with fewer than 2 nodes are skipped
                      - 'ETYPE' records which keyword section each row came from
                        ('SOLID' or 'SHELL') -- lets write_mesh_to_kfile round-trip
                        a mixed-type table back into separate *ELEMENT_SOLID /
                        *ELEMENT_SHELL sections (see that function's docstring).
                        Rows read from *ELEMENT_SHELL_THICKNESS are also tagged
                        'SHELL' (see above) -- the distinct source keyword is not
                        preserved, so a write-back round trip will emit these
                        under a plain *ELEMENT_SHELL section, NOT reproduce the
                        original THICKNESS card (per-node thickness values
                        themselves are not read/stored at all by this function).
                      Returns an empty DataFrame with all 8 node columns if no elements are found.
    """
    if element_type is not None:
        element_type = element_type.upper()
        if element_type not in ['SOLID', 'SHELL']:
            raise ValueError("element_type must be None, 'SOLID', or 'SHELL'.")

    in_element_section = False
    active_type = None  # First encountered type when element_type is None
    current_section_type = None  # Type of the section currently being read (for ETYPE tagging)
    # True only while inside an *ELEMENT_SHELL_THICKNESS section -- every
    # OTHER data line in that section is a thickness-value line (see
    # docstring above), not a new element, and must be skipped rather
    # than parsed. expect_thickness_line alternates False/True as element/
    # thickness lines are consumed in turn.
    in_thickness_section = False
    expect_thickness_line = False
    elems = []

    with open(k_file_path, 'r') as f:
        for line in f:
            # Preserve the ORIGINAL line (only the newline stripped, no
            # leading/trailing whitespace removed) before the `line =
            # line.strip()` below discards it -- needed as the anchor for
            # the fixed-width column fallback a few blocks down, since that
            # fallback MUST slice starting at true column 0 of the file's
            # actual content, not wherever line.strip() happened to leave
            # the first non-space character.
            raw_line_for_fixed_width = line.rstrip('\r\n')
            line = line.strip()
            line_upper = line.upper()
            # First whitespace-delimited token only -- e.g. for
            # '*ELEMENT_SOLID' this equals the whole line, but this also
            # correctly separates a keyword from same-line trailing content.
            keyword_token = line_upper.split()[0] if line_upper else ''

            # Detect element section header. EXACT match on keyword_token,
            # NOT line_upper.startswith(...) -- LS-DYNA has several
            # distinct keywords that share the '*ELEMENT_SOLID'/
            # '*ELEMENT_SHELL' PREFIX but are different cards entirely.
            # '*ELEMENT_SHELL_THICKNESS' is handled explicitly below (its
            # element-defining lines ARE read, unlike other same-prefix
            # keywords this codebase does not understand at all, e.g.
            # '*ELEMENT_SOLID_ORTHO' -- those still correctly fall through
            # to the generic "any '*' keyword closes the section" check.
            if keyword_token == '*ELEMENT_SOLID':
                if active_type is None:
                    active_type = 'SOLID'
                target_type = element_type if element_type is not None else active_type
                in_element_section = (target_type == 'SOLID')
                current_section_type = 'SOLID'
                in_thickness_section = False
                expect_thickness_line = False
                continue
            elif keyword_token == '*ELEMENT_SHELL':
                if active_type is None:
                    active_type = 'SHELL'
                target_type = element_type if element_type is not None else active_type
                in_element_section = (target_type == 'SHELL')
                current_section_type = 'SHELL'
                in_thickness_section = False
                expect_thickness_line = False
                continue
            elif keyword_token == '*ELEMENT_SHELL_THICKNESS':
                if active_type is None:
                    active_type = 'SHELL'
                target_type = element_type if element_type is not None else active_type
                in_element_section = (target_type == 'SHELL')
                current_section_type = 'SHELL'
                in_thickness_section = True
                expect_thickness_line = False
                continue

            # Any keyword ends the current element section.
            if line.startswith('*'):
                in_element_section = False
                in_thickness_section = False
                expect_thickness_line = False
                continue

            if not in_element_section:
                continue

            # Skip empty lines and LS-DYNA comment lines
            if not line or line.startswith('$'):
                continue

            # Inside an *ELEMENT_SHELL_THICKNESS section, every other data
            # line is that element's per-node thickness VALUES (no EID/PID/
            # node-ID content at all) -- skip it outright rather than
            # parsing it as a new element or extra node columns.
            if in_thickness_section and expect_thickness_line:
                expect_thickness_line = False
                continue

            # Parse fields — support both comma-delimited and whitespace-delimited formats
            if ',' in line:
                parts = [p.strip() for p in line.split(',') if p.strip()]
            else:
                parts = line.split()
                # LS-DYNA's classic fixed short-format element card packs
                # EID/PID/N1..N8 into rigid 8-character-wide columns with
                # NO required space between adjacent fields. Whitespace-
                # splitting works fine as long as every field has at least
                # one byte of its own leading padding -- but the moment a
                # single field's value is large enough to completely FILL
                # its 8-character column (e.g. an 8-digit node ID with zero
                # leading spaces), it silently GLUES onto its neighbour,
                # producing one bogus giant "token" (confirmed on a real
                # production mesh: a 34-39 digit blob made of 4-5 fused
                # 8-digit node IDs -- later crashes pandas' int64 PID/EID
                # conversion with OverflowError: Python int too large to
                # convert to C long). No genuine EID/PID/NodeID in this
                # pipeline is anywhere close to 11 digits (largest real
                # NodeID observed so far is 8 digits), so any token this
                # long is unambiguously a fusion artifact, not a real ID --
                # re-derive the WHOLE line's fields via fixed 8-character
                # column slicing of the untouched original line instead,
                # which is immune to this because it never depends on
                # inter-field whitespace at all (matches the equivalent,
                # already-shipped fixed-width fallback in
                # read_nodes_from_kfile for the same class of problem).
                if any(len(p.lstrip('-')) > 10 for p in parts):
                    n_chunks = len(raw_line_for_fixed_width) // 8
                    parts = [
                        raw_line_for_fixed_width[k:k + 8].strip()
                        for k in range(0, n_chunks * 8, 8)
                    ]
                    parts = [p for p in parts if p]

            try:
                eid = int(parts[0])
                pid = int(parts[1])
                node_ids = [int(p) for p in parts[2:] if int(p) != 0]
            except (ValueError, IndexError):
                continue
            
            # Build row: [EID, PID, N1, N2, ..., N8] with NaN padding
            # Limit to max 8 nodes; at least 2 nodes required for connectivity
            if len(node_ids) < 2:
                continue
            
            node_ids = node_ids[:8]  # Take first 8 nodes
            # Pad with NaN to reach 8 nodes
            node_ids.extend([np.nan] * (8 - len(node_ids)))
            elems.append([eid, pid] + node_ids + [current_section_type])

            if in_thickness_section:
                expect_thickness_line = True

    # Always return 8-node format for consistency
    full_cols = ['EID', 'PID', 'NID1', 'NID2', 'NID3', 'NID4', 'NID5', 'NID6', 'NID7', 'NID8', 'ETYPE']
    
    if not elems:
        return pd.DataFrame(columns=full_cols)
    
    df = pd.DataFrame(elems, columns=full_cols)
    # Convert all node columns to Int64 (nullable int) to preserve NaN
    df['EID'] = df['EID'].astype(int)
    df['PID'] = df['PID'].astype(int)
    for col in ['NID1', 'NID2', 'NID3', 'NID4', 'NID5', 'NID6', 'NID7', 'NID8']:
        df[col] = df[col].astype('Int64')  # Nullable integer type preserves NaN
    
    return df


def read_all_elements_from_kfile(k_file_path):
    """
    Read EVERY supported element type (*ELEMENT_SOLID and *ELEMENT_SHELL)
    from one .k file into a single combined DataFrame, tagged by the new
    'ETYPE' column -- unlike read_elements_from_kfile(element_type=None),
    which only aggregates whichever type it meets FIRST and silently skips
    the other, this always reads BOTH regardless of section order or which
    appears first in the file.

    WHY THIS EXISTS: a genuine whole-model biomechanical file mixes solid
    volume parts (muscle, fat, organs) with shell surface parts (skin wrap,
    fascia, ligament sheets). Repair only ever targets SOLID elements (see
    repair_inverted_elements -- its Jacobian/volume math has no shell
    concept), but the ORIGINAL shell elements still need to survive
    unchanged through the whole read -> repair -> write round trip so the
    output file isn't missing entire parts. Use this function (or
    read_multi_part_mesh(..., element_type='ALL')) to load a file for
    OUTPUT purposes, then filter elements_df to ETYPE == 'SOLID' for the
    actual repair/adjacency computation.

    NOTE: any element card type this codebase does not yet parse (e.g.
    *ELEMENT_BEAM, *ELEMENT_DISCRETE, *ELEMENT_MASS, *ELEMENT_SEATBELT)
    is NOT read by this function either, and will therefore be silently
    absent from the returned table -- and, if this table is later written
    back out via write_mesh_to_kfile/write_multi_part_mesh, absent from
    the output file too. Confirm which element card types are actually
    present in your file(s) before relying on a full round trip.

    Inputs:
        k_file_path (str): Path to the LS-DYNA .k file.

    Returns:
        pd.DataFrame: Concatenation of read_elements_from_kfile(k_file_path,
            element_type='SOLID') and the same call with element_type=
            'SHELL', in that order, re-indexed. Columns as documented in
            read_elements_from_kfile, including 'ETYPE'.
    """
    solid_df = read_elements_from_kfile(k_file_path, element_type='SOLID')
    shell_df = read_elements_from_kfile(k_file_path, element_type='SHELL')
    if solid_df.empty and shell_df.empty:
        return solid_df
    return pd.concat([solid_df, shell_df], ignore_index=True)


def _node_columns(mesh_info):
    """
    Return available node columns in NID1..NID8 order.
    """
    return [f'NID{i}' for i in range(1, 9) if f'NID{i}' in mesh_info.columns]


def _valid_element_nodes(row, node_cols):
    """
    Return unique, real node IDs for one element row.

    Skips NaN/pd.NA, zero placeholders, and repeated node IDs. Repeated node IDs
    can occur for degenerate tet and penta elements stored in an 8-node LS-DYNA
    solid card.
    """
    nodes = []
    seen = set()
    for col in node_cols:
        nid = row[col]
        if pd.isna(nid):
            continue

        node_id = int(nid)
        if node_id == 0 or node_id in seen:
            continue

        nodes.append(node_id)
        seen.add(node_id)

    return nodes


def read_multi_part_mesh(k_file_paths, coincidence_tolerance=1e-6, element_type=None):
    """
    Read and combine multiple single-part (or already multi-part) LS-DYNA
    .k files into ONE unified node/element table, correctly merging nodes
    that are shared/welded between parts (same NodeID, same physical
    location in more than one file).

    This is the entry point for cross-part-aware repair: repair_inverted_
    elements' zone-expansion logic (_expand_node_region) works purely by
    NODE ADJACENCY, with no notion of "part" -- so once multiple parts'
    elements/nodes are combined into one table by this function, fixing a
    bad element in one part automatically treats any neighbouring element
    from a DIFFERENT part (connected only via a shared/welded node) as a
    protected neighbour under the existing non-regression floor (see
    _build_floor_and_bad_mask), with NO changes needed to the core
    optimization machinery itself.

    A single file already containing multiple *ELEMENT_SOLID/PID blocks
    (e.g. a WHOLE-MODEL file with every part in it) works fine too --
    pass it as a single-entry list. In that case every shared/welded node
    already appears only where the file's own *NODE section defines it (a
    valid LS-DYNA deck never repeats a NodeID with different coordinates),
    so this function's cross-file merge logic simply has nothing to do
    and passes through safely; the more common single-part function
    (repair_inverted_elements) could equally be called directly on such a
    file's own tables without going through read_multi_part_mesh at all.

    IMPORTANT for whole-model files with MIXED element types: passing
    element_type=None or 'SOLID'/'SHELL' only reads ONE type per file (see
    read_elements_from_kfile) -- fine when repair is your only goal, since
    repair_inverted_elements only ever targets solids, but it means any
    OTHER element type present (e.g. shell skin/fascia parts) is silently
    absent from elements_df, and therefore absent from the OUTPUT file too
    if this table is later written back with write_multi_part_mesh. Pass
    element_type='ALL' to read BOTH *ELEMENT_SOLID and *ELEMENT_SHELL
    sections from every file (via read_all_elements_from_kfile), tagged by
    the 'ETYPE' column, so a mixed-type whole-model file round-trips
    completely. When using 'ALL', filter elements_df to ETYPE == 'SOLID'
    before passing it to repair_inverted_elements / pocket-splitting --
    those only understand solid volume elements; shell rows would be
    misinterpreted as tet4/quad-shaped solids (a shell quad's 4 unique
    nodes are indistinguishable from a tet4's 4 unique nodes by node count
    alone). See read_all_elements_from_kfile's docstring for the further
    caveat that any element card type beyond SOLID/SHELL (beams, discrete
    springs, lumped mass, seatbelts, etc.) is not read/round-tripped by
    this codebase at all yet.

    FLOATING / UNREFERENCED NODES (e.g. anatomical landmark points that
    are not part of any element): read_nodes_from_kfile reads a file's
    entire *NODE section unconditionally, so these already appear in
    nodes_df/per-file node tables regardless of element type filtering.
    They are automatically inert everywhere in the repair pipeline (never
    referenced by any element means they can never become a "free" node
    via adjacency expansion, so they are never targeted or moved) and are
    now correctly tracked in manifest so write_multi_part_mesh preserves
    them in their OWNING file's output -- see manifest's docstring entry
    below for exactly how 'node_ids' is computed.

    Inputs:
        k_file_paths (list[str]): One or more paths to .k files. Each file
            is expected to contain its own *NODE and *ELEMENT_SOLID (or
            *ELEMENT_SHELL) sections, as read by read_nodes_from_kfile /
            read_elements_from_kfile.
        coincidence_tolerance (float): Maximum allowed per-axis coordinate
            difference for a NodeID that appears in more than one input
            file (a "shared/welded" node) before it is treated as a real
            data problem rather than the same physical point. Real welded
            nodes are expected to match exactly (see the docstring's
            note about the SharedNodes example data matching to 0.0), so
            this is a generous safety margin, not a tuned merge tolerance.
        element_type (str or None): Passed straight through to
            read_elements_from_kfile for every input file. 'SOLID' or
            'SHELL' to force one type; None (default) keeps the original
            single-file behaviour of auto-detecting from the first section
            found -- fine for files that only ever contain one element
            type, but see the whole-model caveat above. 'ALL' reads BOTH
            types (via read_all_elements_from_kfile) tagged by 'ETYPE'.

    Returns:
        tuple:
            nodes_df (pd.DataFrame): Combined, de-duplicated node table
                (columns 'NodeID', 'x', 'y', 'z'), one row per unique
                NodeID across all input files -- includes nodes never
                referenced by any element.
            elements_df (pd.DataFrame): Combined element table (all rows
                from all input files, concatenated; columns 'EID', 'PID',
                'NID1'..'NID8', and 'ETYPE' when element_type='ALL'),
                preserving each file's own PID values.
            manifest (dict): {k_file_path: {'eids': set[int], 'node_ids':
                set[int], 'referenced_node_ids': set[int], 'own_node_ids':
                set[int]}}. 'node_ids' is the UNION of (a) every node this
                file's own elements reference, and (b) every node this
                file's own *NODE section defines -- NOT just (a) alone.
                (b) is what makes an unreferenced landmark node round-trip
                back into the correct output file; (a) is still needed on
                its own for cases where a shared/welded node's coordinate
                is only DEFINED in one file's *NODE section but referenced
                by another file's elements (common with LS-DYNA *INCLUDE
                workflows) -- dropping either half would either lose
                landmarks or produce an output file with a dangling,
                undefined node reference. 'referenced_node_ids' is (a)
                alone (elements only, no *NODE-section nodes) -- kept
                separately since it is exactly what the missing-node
                validation below needs, without re-deriving it.
                'own_node_ids' is (b) alone (this file's own *NODE section
                only, none of its elements' referenced nodes) -- used by
                write_multi_part_mesh so an elements-only input file (no
                *NODE section, e.g. a dedicated I-PREDICT_v1.0_Elements.k
                *INCLUDE'd alongside a separate nodes file) writes back
                out with ZERO nodes of its own, matching a real user-
                requested workflow: nodes belong in ONE file only (the
                one that originally owned them), never re-derived/
                duplicated into an elements-only file.

    Raises:
        ValueError: If the same EID appears in more than one input file
            (element IDs must be globally unique across the combined
            mesh -- a collision indicates the files are not disjoint parts
            of one model, which this function cannot safely merge); if
            a NodeID shared between files has coordinates disagreeing by
            more than coincidence_tolerance in any axis (indicates the
            "shared" nodes are not actually the same welded point); or if
            any element in ANY loaded file references a NodeID that has NO
            coordinate in ANY loaded file's *NODE section (indicates a
            file that OWNS those node coordinates -- e.g. a shared/master
            node file, or a neighbouring part's file -- is missing from
            k_file_paths; this is caught here with a clear, actionable
            message specifically so it does NOT surface later as an
            opaque pandas KeyError deep inside repair_inverted_elements).
    """
    per_file_nodes = {}
    per_file_elements = {}
    manifest = {}

    read_all_types = (isinstance(element_type, str) and element_type.upper() == 'ALL')

    for path in k_file_paths:
        file_nodes = read_nodes_from_kfile(path)
        if read_all_types:
            file_elements = read_all_elements_from_kfile(path)
        else:
            file_elements = read_elements_from_kfile(path, element_type=element_type)
        per_file_nodes[path] = file_nodes
        per_file_elements[path] = file_elements

        # Vectorized (see _flat_referenced_node_ids_fast's docstring) --
        # replaces a per-row file_elements.iterrows() + _valid_element_nodes
        # loop that dominated load time on large whole-model files (~8 min
        # on a real 3M-element mesh, cut to a few seconds by this change).
        referenced_nids = _flat_referenced_node_ids_fast(file_elements)

        # Union of referenced nodes (a) and this file's own *NODE-defined
        # nodes (b) -- see manifest's docstring entry above for why both
        # halves matter (floating landmarks vs. cross-file *INCLUDE welds).
        own_node_ids = set(int(n) for n in file_nodes['NodeID'].tolist())

        manifest[path] = {
            'eids': set(int(e) for e in file_elements['EID'].tolist()),
            'node_ids': referenced_nids | own_node_ids,
            'referenced_node_ids': referenced_nids,
            # 'own_node_ids': ONLY the nodes this file's own *NODE section
            # actually defined (b alone, not the union) -- used by
            # write_multi_part_mesh so an elements-only input file (no
            # *NODE section of its own, e.g. I-PREDICT_v1.0_Elements.k)
            # writes ZERO nodes back out, instead of re-deriving a *NODE
            # section from whichever nodes its elements happen to
            # reference. See write_multi_part_mesh's own docstring for
            # the full "write nodes only to the file that owned them"
            # rationale (a real user-facing workflow requirement: a
            # combined main.k *INCLUDEs a dedicated nodes file THEN the
            # elements file, so the elements file re-defining nodes was
            # redundant, and risked ambiguity about which file's copy is
            # authoritative if the two were ever edited independently).
            'own_node_ids': own_node_ids,
        }

    # EIDs must be globally unique across the combined mesh -- otherwise
    # concatenating elements_df tables would silently create two different
    # elements sharing one ID, corrupting every downstream EID-keyed lookup
    # (bad-element matching, floor/baseline computation, output writing).
    seen_eids = {}
    for path, info in manifest.items():
        for eid in info['eids']:
            if eid in seen_eids:
                raise ValueError(
                    f"EID {eid} appears in both {seen_eids[eid]} and {path}. "
                    "Element IDs must be globally unique across all files "
                    "passed to read_multi_part_mesh -- these files cannot be "
                    "safely combined as-is."
                )
            seen_eids[eid] = path

    elements_df = pd.concat(per_file_elements.values(), ignore_index=True)

    # Merge nodes, validating that any NodeID shared between files really
    # is the same physical (welded/coincident) point, not a coincidental ID
    # reuse.
    all_nodes = pd.concat(per_file_nodes.values(), ignore_index=True)

    # PERFORMANCE: only NodeIDs that appear in the *NODE section of MORE
    # THAN ONE input file are even candidates for a "shared/welded but
    # disagreeing" data problem -- a NodeID appearing exactly once cannot
    # conflict with anything. value_counts() (vectorized, fast even on
    # millions of rows) finds those candidates; the groupby+max/min
    # coordinate-agreement check then runs ONLY on that (typically tiny)
    # subset. This matters enormously in practice: running that check via
    # a groupby over the FULL node table (as an earlier version of this
    # function did unconditionally) means grouping into ~as-many-groups-
    # as-rows when few/no nodes are actually shared -- measured at ~7
    # minutes (176s .max() + 238s .min()) on a real 1.4M-node whole-model
    # file that turned out to have ZERO shared nodes across its files.
    # Restricting to only genuinely-repeated NodeIDs cuts this to a
    # fraction of a second on that same file, and to well under a second
    # on the validated SharedNodes case (126 genuinely shared out of
    # ~2500 total).
    node_id_counts = all_nodes['NodeID'].value_counts()
    duplicated_nids = node_id_counts.index[node_id_counts.gt(1)]
    n_shared = len(duplicated_nids)

    if n_shared:
        dup_rows = all_nodes[all_nodes['NodeID'].isin(duplicated_nids)]
        grouped = dup_rows.groupby('NodeID')[['x', 'y', 'z']]
        spread = grouped.max() - grouped.min()
        bad_nodes = spread[(spread > coincidence_tolerance).any(axis=1)]
        if not bad_nodes.empty:
            worst = bad_nodes.max(axis=1).sort_values(ascending=False)
            raise ValueError(
                f"{len(bad_nodes)} shared NodeID(s) have DIFFERENT coordinates "
                f"across the input files (expected identical/welded positions, "
                f"tolerance={coincidence_tolerance}). Worst offenders (max axis "
                f"spread): {worst.head(10).to_dict()}. These files cannot be "
                "safely combined as-is -- check for a NodeID collision that "
                "isn't actually a shared/welded node."
            )

    nodes_df = all_nodes.drop_duplicates(subset='NodeID', keep='first').reset_index(drop=True)

    # Every node ANY loaded element references must actually have a
    # coordinate SOMEWHERE in the merged node table, or every downstream
    # step (repair_inverted_elements' zone-coordinate lookup, in
    # particular) will fail with an opaque pandas KeyError deep inside the
    # optimizer instead of a clear, early, actionable message. This is a
    # real (not hypothetical) failure mode for industrial LS-DYNA exports:
    # some per-part files only contain elements plus whatever nodes are
    # NEW to that part, relying on a shared/master node file (or another
    # part's file) to supply coordinates for boundary nodes reused from
    # elsewhere -- if that other file isn't included in k_file_paths, its
    # nodes are simply missing from every loaded file's own *NODE section.
    all_referenced_nids = set()
    for info in manifest.values():
        all_referenced_nids |= info['referenced_node_ids']
    missing_nids = all_referenced_nids - set(int(n) for n in nodes_df['NodeID'].tolist())
    if missing_nids:
        # Identify which file(s)/PID(s) reference the missing nodes, for a
        # useful error message -- restricted to the (already computed)
        # per-file elements, only entered on this rare error path.
        offending = []
        for path, file_elements in per_file_elements.items():
            node_cols = _node_columns(file_elements)
            for _, row in file_elements.iterrows():
                row_nodes = set(_valid_element_nodes(row, node_cols))
                hit = row_nodes & missing_nids
                if hit:
                    offending.append((path, int(row['PID']), int(row['EID']), sorted(hit)))
                    if len(offending) >= 10:
                        break
            if len(offending) >= 10:
                break
        raise ValueError(
            f"{len(missing_nids)} NodeID(s) are referenced by loaded elements "
            f"but have NO coordinates in ANY of the {len(k_file_paths)} loaded "
            f"file(s)' *NODE sections. This means at least one file that OWNS "
            f"these nodes' coordinates (e.g. a master/shared node file, or "
            f"another part's file) is missing from k_file_paths. Sample "
            f"missing NodeID(s): {sorted(missing_nids)[:20]}. Sample "
            f"referencing (file, PID, EID, missing_node_ids): {offending}. "
            "Add the missing file(s) to k_file_paths, or confirm every part "
            "that shares a boundary node with a loaded part is also loaded."
        )

    print(
        f"read_multi_part_mesh: combined {len(k_file_paths)} file(s) -> "
        f"{len(nodes_df)} unique nodes ({n_shared} shared/welded across "
        f"more than one file), {len(elements_df)} elements."
    )

    return nodes_df, elements_df, manifest


def write_multi_part_mesh(nodes_df, elements_df, manifest, suffix='_repaired',
                           element_type='SOLID', write_combined=True,
                           combined_path=None):
    """
    Split a combined (possibly repaired) multi-part mesh -- as produced by
    read_multi_part_mesh plus repair_inverted_elements -- back into one
    output .k file per ORIGINAL input file, plus (optionally) one combined
    file with everything together.

    Each per-part output file contains exactly that file's own elements
    (by EID, from manifest), at whatever (possibly moved) coordinates
    they ended up at in nodes_df. NODES ARE WRITTEN ONLY INTO THE FILE(S)
    THAT ORIGINALLY OWNED THEM (manifest['own_node_ids'] -- this file's
    own *NODE section, NOT the elements-referenced union) -- a real,
    explicit user workflow requirement: an elements-only input file (no
    *NODE section of its own, e.g. a dedicated I-PREDICT_v1.0_Elements.k
    *INCLUDE'd alongside a separate nodes file) is written back out with
    ZERO nodes, instead of re-deriving/duplicating a *NODE section from
    whichever nodes its elements happen to reference. This avoids any
    ambiguity about which file's copy of a node's coordinates is
    authoritative if the two output files were ever edited/diffed
    independently, and keeps each output file's role identical to its
    input file's role (a nodes file stays a nodes file; an elements file
    stays elements-only). Since a shared NodeID still has exactly ONE row
    in the combined nodes_df regardless of which output file eventually
    writes it, there is no code path by which two parts' output files
    could disagree about a shared node's position -- this change affects
    ONLY which file a node's definition physically lands in, never its
    value.

    Inputs:
        nodes_df (pd.DataFrame): Combined node table (e.g. the repaired
            updated_nodes_df returned by repair_inverted_elements when
            called on read_multi_part_mesh's combined tables).
        elements_df (pd.DataFrame): Combined element table (unchanged from
            read_multi_part_mesh -- repair_inverted_elements never adds,
            removes, or renumbers elements).
        manifest (dict): The manifest returned by read_multi_part_mesh.
        suffix (str): Inserted before the file extension for each per-part
            output (e.g. "PID_1000034.k" -> "PID_1000034_repaired.k").
        element_type (str): Passed through to write_mesh_to_kfile as its
            FALLBACK type ('SOLID' or 'SHELL') -- only actually used for
            rows with no 'ETYPE' column/value. If elements_df carries an
            'ETYPE' column (e.g. read_multi_part_mesh was called with
            element_type='ALL'), each row is written under its OWN type
            regardless of this argument -- see write_mesh_to_kfile's
            docstring for the mixed-type behavior.
        write_combined (bool): If True, also write one file with the FULL
            combined nodes_df/elements_df (unaffected by the own_node_ids
            change above -- the combined file always contains every node
            AND every element, exactly as before, since it's meant to be
            a single self-contained whole-model dump).
        combined_path (str or None): Output path for the combined file. If
            None and write_combined is True, defaults to
            "combined_repaired.k" in the same directory as the first input
            file in manifest.

    Returns:
        dict: {original_path: output_path} for each per-part file, plus
            (if write_combined) a 'combined' key mapping to the combined
            file's path.
    """
    outputs = {}
    for path, info in manifest.items():
        root, ext = os.path.splitext(path)
        out_path = f"{root}{suffix}{ext}"

        part_elements = elements_df[elements_df['EID'].isin(info['eids'])]
        part_nodes = nodes_df[nodes_df['NodeID'].isin(info['own_node_ids'])]

        write_mesh_to_kfile(part_nodes, part_elements, out_path, element_type=element_type)
        outputs[path] = out_path

    if write_combined:
        if combined_path is None:
            first_dir = os.path.dirname(next(iter(manifest.keys())))
            combined_path = os.path.join(first_dir, "combined_repaired.k")
        write_mesh_to_kfile(nodes_df, elements_df, combined_path, element_type=element_type)
        outputs['combined'] = combined_path

    return outputs


def _unique_node_lists_fast(elements_df):
    """
    Vectorized-friendly replacement for looping elements_df.iterrows() plus
    _valid_element_nodes() per row. Converts node columns to a single NumPy
    array once, then iterates that plain array in pure Python.

    DataFrame.iterrows()/itertuples() materialize a pandas object per row,
    which dominates runtime on element counts in the tens of thousands (this
    is a one-time preprocessing cost inside repair_inverted_elements, called
    on every repair attempt, so keeping it fast matters for escalation
    ladders that retry with wider node layers).

    Skips NaN, zero placeholders, and repeated node IDs -- identical
    semantics to _valid_element_nodes, just applied to every row up front.

    Inputs:
        elements_df (pd.DataFrame): Elements from read_elements_from_kfile;
            columns include 'NID1'..'NID8' (NaN-padded).

    Returns:
        list[list[int]]: One entry per input row (same order), each the
            unique, real node IDs in original column order.
    """
    node_cols = _node_columns(elements_df)
    if not node_cols:
        return [[] for _ in range(len(elements_df))]

    # See _flat_referenced_node_ids_fast's docstring for the full story:
    # a cross-file pd.concat inside read_multi_part_mesh (element_type=
    # 'ALL', or any file with zero elements, e.g. a nodes-only file) can
    # silently degrade these columns from nullable Int64 to plain
    # `object` dtype containing pd.NA -- `.to_numpy(dtype=float, na_value
    # =np.nan)` raises TypeError on that dtype (confirmed via a real
    # crash calling this function on an element_type='ALL'-derived shell
    # subset). pd.to_numeric(..., errors='coerce') normalizes any of
    # (plain float+np.nan / nullable Int64+pd.NA / object+pd.NA) to a
    # clean float64 array with real np.nan before the loop below runs
    # unchanged.
    raw = elements_df[node_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)

    result = []
    for row in raw:
        nodes = []
        seen = set()
        for val in row:
            if val != val:  # NaN check without a function-call/pd.isna overhead
                continue
            node_id = int(val)
            if node_id == 0 or node_id in seen:
                continue
            nodes.append(node_id)
            seen.add(node_id)
        result.append(nodes)

    return result


def _flat_referenced_node_ids_fast(elements_df):
    """
    Fully-vectorized computation of the FLAT set of every real (non-NaN,
    non-zero) node ID referenced ANYWHERE in elements_df -- the per-row
    grouping/order does not matter here, only the union across all rows
    and all node columns.

    Unlike _unique_node_lists_fast (which preserves per-row structure,
    needed for building topological adjacency/zones), this never builds a
    Python list or set per row: the entire node-column block is flattened
    and de-duplicated in ONE vectorized NumPy call (np.unique), which
    avoids O(n_elements) individual Python-level set insertions. This is
    the difference between an unusable and a practical load time on a
    real multi-million-element whole-model file -- measured to cut
    read_multi_part_mesh's load step from ~8 minutes to a few seconds on
    a 3,010,175-element real mesh (previously done via
    elements_df.iterrows() + _valid_element_nodes() per row).

    Inputs:
        elements_df (pd.DataFrame): Elements from read_elements_from_kfile;
            columns include 'NID1'..'NID8' (NaN-padded).

    Returns:
        set[int]: Every unique, real (non-zero) node ID referenced by any
            row's node columns.
    """
    node_cols = _node_columns(elements_df)
    if not node_cols or elements_df.empty:
        return set()

    # NOTE: cannot assume elements_df[node_cols] is already a clean
    # float/np.nan-padded block -- read_multi_part_mesh's cross-FILE
    # pd.concat (used whenever element_type='ALL', or whenever ANY loaded
    # file has zero elements, e.g. a nodes-only file) can silently
    # degrade these columns from pandas' nullable Int64 to plain `object`
    # dtype containing pd.NA (not np.nan) -- a real, confirmed case:
    # concatenating an EMPTY per-file element table (a nodes-only file
    # correctly produces 0 element rows) with a real one triggers
    # pandas' own documented "concatenation with empty or all-NA entries"
    # deprecated-behavior path (the same FutureWarning seen in real
    # cluster runs). `.to_numpy(dtype=float, na_value=np.nan)` raises
    # TypeError on an object-dtype column containing pd.NA (unlike a
    # proper nullable-Int64 column, where na_value=np.nan is honored
    # correctly). pd.to_numeric(..., errors='coerce') normalizes ANY of
    # these representations (plain float+np.nan, nullable Int64+pd.NA, or
    # object+pd.NA) to a clean float64 array with real np.nan, before the
    # existing fast numpy path below runs unchanged.
    raw = elements_df[node_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    flat = raw.ravel()
    flat = flat[~np.isnan(flat)]
    if flat.size == 0:
        return set()
    flat = flat.astype(np.int64)
    flat = flat[flat != 0]
    if flat.size == 0:
        return set()
    return set(np.unique(flat).tolist())


def _sorted_key(nodes):
    """
    Build an order-independent tuple key from node IDs.
    """
    return tuple(sorted(nodes))


def _quad_face_keys(nodes):
    """
    Build a quadrilateral face key and its triangular subface keys.
    """
    return [
        _sorted_key(nodes),
        _sorted_key([nodes[0], nodes[1], nodes[2]]),
        _sorted_key([nodes[0], nodes[1], nodes[3]]),
        _sorted_key([nodes[0], nodes[2], nodes[3]]),
        _sorted_key([nodes[1], nodes[2], nodes[3]]),
    ]


def _element_id_for_warning(row):
    """
    Return an element ID for warning messages when a row is available.
    """
    if row is not None and 'EID' in row:
        return row['EID']
    return 'unknown'


def _warn_unsupported_element(element_kind, element_id, node_count, supported_counts):
    """
    Warn when connectivity cannot be computed for an unsupported node count.
    """
    supported = ', '.join(str(count) for count in supported_counts)
    warnings.warn(
        f"Unsupported {element_kind} element node count for EID {element_id}: "
        f"found {node_count} unique real nodes; supported counts are {supported}.",
        RuntimeWarning,
        stacklevel=3,
    )


def _solid_face_keys(nodes, row=None):
    """
    Build order-independent face keys for supported LS-DYNA solid elements.

    Supported solid shapes:
      - 4 unique nodes: tetrahedron, 4 triangular faces
      - 6 unique nodes: LS-DYNA pentahedron/wedge, 2 triangular faces and 3 quadrilateral faces
      - 8 unique nodes: hexahedron, 6 quadrilateral faces plus triangular subface keys
    """
    if len(nodes) == 4:
        return [
            _sorted_key([nodes[0], nodes[1], nodes[2]]),
            _sorted_key([nodes[0], nodes[1], nodes[3]]),
            _sorted_key([nodes[0], nodes[2], nodes[3]]),
            _sorted_key([nodes[1], nodes[2], nodes[3]]),
        ]

    if len(nodes) == 6:
        face_keys = [
            _sorted_key([nodes[0], nodes[1], nodes[4]]),
            _sorted_key([nodes[2], nodes[3], nodes[5]]),
        ]
        for face_nodes in [
            [nodes[0], nodes[1], nodes[2], nodes[3]],
            [nodes[1], nodes[2], nodes[5], nodes[4]],
            [nodes[3], nodes[0], nodes[4], nodes[5]],
        ]:
            face_keys.extend(_quad_face_keys(face_nodes))

        return face_keys

    if len(nodes) == 8:
        face_keys = []
        for face_nodes in [
            [nodes[0], nodes[1], nodes[2], nodes[3]],
            [nodes[4], nodes[5], nodes[6], nodes[7]],
            [nodes[0], nodes[1], nodes[5], nodes[4]],
            [nodes[1], nodes[2], nodes[6], nodes[5]],
            [nodes[2], nodes[3], nodes[7], nodes[6]],
            [nodes[3], nodes[0], nodes[4], nodes[7]],
        ]:
            face_keys.extend(_quad_face_keys(face_nodes))

        return face_keys

    _warn_unsupported_element('solid', _element_id_for_warning(row), len(nodes), [4, 6, 8])
    return []


def _shell_edge_keys(nodes, row=None):
    """
    Build order-independent edge keys for supported LS-DYNA shell elements.

    Supported shell shapes:
      - 3 unique nodes: triangle, 3 edges
      - 4 unique nodes: quadrilateral, 4 edges numbered around the face
    """
    if len(nodes) == 3:
        return [
            _sorted_key([nodes[0], nodes[1]]),
            _sorted_key([nodes[1], nodes[2]]),
            _sorted_key([nodes[2], nodes[0]]),
        ]

    if len(nodes) == 4:
        return [
            _sorted_key([nodes[0], nodes[1]]),
            _sorted_key([nodes[1], nodes[2]]),
            _sorted_key([nodes[2], nodes[3]]),
            _sorted_key([nodes[3], nodes[0]]),
        ]

    _warn_unsupported_element('shell', _element_id_for_warning(row), len(nodes), [3, 4])
    return []


def _shell_face_keys(nodes, row=None):
    """
    Build the one surface face key contributed by a shell element.
    """
    if len(nodes) in [3, 4]:
        return [_sorted_key(nodes)]

    _warn_unsupported_element('shell', _element_id_for_warning(row), len(nodes), [3, 4])
    return []


def _build_connectivity_key_index(mesh_info, key_function):
    """
    Build lookup dictionaries for elements that share a topology key.

    Returns:
        tuple:
          - key_to_elements: maps face/edge key -> list of EIDs containing that key
          - element_to_keys: maps EID -> list of that element's topology keys
          - eid_order: maps EID -> input row order for stable neighbor ordering
    """
    node_cols = _node_columns(mesh_info)
    key_to_elements = {}
    element_to_keys = {}
    eid_order = {}

    for row_order, (_, row) in enumerate(mesh_info.iterrows()):
        eid = row['EID']
        eid_order[eid] = row_order

        nodes = _valid_element_nodes(row, node_cols)
        keys = key_function(nodes, row)
        element_to_keys[eid] = keys

        for key in keys:
            key_to_elements.setdefault(key, []).append(eid)

    return key_to_elements, element_to_keys, eid_order


def _connectivity_from_key_index(mesh_info, key_function, element_mismatch_message):
    """
    Compute connectivity for elements that share any topology-specific key.

    Always returns fixed columns ['Element', 'Conn1', ..., 'Conn8'].
    """
    key_to_elements, element_to_keys, eid_order = _build_connectivity_key_index(mesh_info, key_function)
    rows = []

    for eid in mesh_info['EID']:
        neighbor_set = set()
        for key in element_to_keys[eid]:
            neighbor_set.update(key_to_elements[key])

        neighbor_set.discard(eid)
        neighbors = sorted(neighbor_set, key=lambda neighbor_eid: eid_order[neighbor_eid])
        rows.append([eid] + neighbors)

    fixed_cols = ['Element'] + [f'Conn{k}' for k in range(1, 9)]

    if not rows:
        return pd.DataFrame(columns=fixed_cols)

    fixed_rows = []
    for row in rows:
        element_id = row[0]
        neighbors = row[1:]

        if len(neighbors) > 8:
            warnings.warn(
                f"Element {element_id} has {len(neighbors)} connected elements; "
                "only Conn1 through Conn8 will be returned.",
                RuntimeWarning,
                stacklevel=2,
            )

        padded_neighbors = neighbors[:8] + [np.nan] * max(0, 8 - len(neighbors))
        fixed_rows.append([element_id] + padded_neighbors)

    df_out = pd.DataFrame(fixed_rows, columns=fixed_cols)

    if len(mesh_info) != len(df_out):
        raise ValueError("Not all elements in connectivity matrix")
    if set(mesh_info['EID']) != set(df_out['Element']):
        raise ValueError(element_mismatch_message)

    return df_out

def solid_connect_fun(mesh_info):
    """
    Compute face connectivity for supported solid elements.
    
    Supports 4-node tetrahedra, 6-node pentahedra, and 8-node hexahedra. For
    each element, finds all neighboring elements that share all nodes of a
    topology-defined face. Penta and hex quadrilateral faces also contribute
    triangular subface keys so split tet interfaces can be connected.

    Parameters:
        mesh_info (pd.DataFrame): DataFrame from read_elements_from_kfile with columns
                                  ['EID', 'PID', 'NID1'...'NID8'] (NaN-padded to 8 nodes)

    Returns:
        pd.DataFrame: columns ['Element', 'Conn1', ..., 'Conn8']
                      NaN-padded where fewer neighbors exist.
                      All input elements appear in output.
    """
    return _connectivity_from_key_index(mesh_info, _solid_face_keys, "Element mismatch")


def shell_connect_fun(mesh_info):
    """
    Compute edge connectivity for supported shell elements sharing one full edge.
    
    Supports 3-node triangles and 4-node quadrilaterals. For each element, finds
    all neighboring elements that share both nodes of a topology-defined shell edge.
    
    A quad element has at most 4 edge-sharing neighbors (one per edge).
    A triangle element has at most 3 edge-sharing neighbors (one per edge).
    Boundary elements have fewer neighbors.

    Parameters:
        mesh_info (pd.DataFrame): DataFrame from read_elements_from_kfile with columns
                                  ['EID', 'PID', 'NID1'...'NID8'] (NaN-padded to 8 nodes)

    Returns:
        pd.DataFrame: columns ['Element', 'Conn1', ..., 'Conn8']
                      NaN-padded where fewer neighbors exist.
                      All input elements appear in output.
    """
    return _connectivity_from_key_index(mesh_info, _shell_edge_keys, "Element sets not equivalent")


def parse_lsdyna_curves(filename):
    """
    Parse LS-PrePost curve export text files into element-wise arrays.

    Expected file layout per curve block:
    - curve label line containing "#pts=<n>"
    - optional "@ <element_id>" within the label line
      - optional min-value metadata line
      - optional max-value metadata line
      - numeric time/value rows
      - endcurve

    Header/preamble lines before the first curve block are ignored.

    Parameters:
        filename (str): Path to a curve export text file.

    Returns:
        list[dict]: Each dict contains:
            - element_id (int or None)
            - type (str)
            - data (np.ndarray with shape [n_points, n_columns])
    """
    def _is_curve_label(line):
        return '#pts=' in line.lower()

    with open(filename, "r") as f:
        lines = [line.strip() for line in f]

    curves = []
    n = len(lines)
    i = 0

    while i < n:
        line = lines[i]

        # Skip file preamble and non-label metadata until a real curve label.
        if not _is_curve_label(line):
            i += 1
            continue

        label_line = line
        i += 1

        # LS-PrePost exports typically include min/max metadata lines here.
        while i < n and lines[i].startswith("*"):
            i += 1

        data = []
        while i < n and lines[i].lower() != "endcurve":
            parts = lines[i].split()
            try:
                data.append([float(x) for x in parts])
            except ValueError:
                pass
            i += 1

        if i < n and lines[i].lower() == "endcurve":
            i += 1

        match = re.search(r"@\s*(\d+)\b", label_line)
        element_id = int(match.group(1)) if match else None
        curve_type = re.split(r'\s*@\s*\d+\b|\s+#pts=', label_line, maxsplit=1)[0].strip()

        curves.append(
            {
                "element_id": element_id,
                "type": curve_type,
                "data": np.array(data, dtype=float),
            }
        )

    return curves

def solid_shell_connect_fun(mesh_info_shell, mesh_info_solid):
    """
    Compute combined solid/shell connectivity using shared face keys.

    Solid rows contribute their topology-defined solid face keys. Shell rows
    contribute their one surface face key, so solid-shell interfaces are matched
    by shared faces.

    Parameters:
        mesh_info_XXXX (pd.DataFrame): DataFrame from read_elements_from_kfile for a solid and shell


    """
    # TODO: Update
    raise ValueError("Function not correct")
    # Combine solid and shell meshes so connectivity can cross the solid-shell interface.
    # Shell rows can have NaN for unused NID columns; zero placeholders are ignored.
    mesh_info_combined = pd.concat([mesh_info_solid, mesh_info_shell], ignore_index=True)
    set_solid_ele = set(mesh_info_solid.EID)
    set_shell_ele = set(mesh_info_shell.EID)  
    
    def _solid_shell_face_keys(nodes, row=None):
        if row['EID'] in set_solid_ele:
            return _solid_face_keys(nodes)
        return _shell_face_keys(nodes)
    
    # --- Solid connectivity on the combined mesh ---
    # df1 contains solid elements AND shell elements, each listing shared-face neighbors
    df1 = _connectivity_from_key_index(mesh_info_combined, _solid_shell_face_keys, "Element mismatch")
    
    # --- Shell connectivity on the shell-only mesh ---
    # df2 contains shell elements, each listing their edge-sharing shell neighbors
    df2 = shell_connect_fun(mesh_info_shell)
    
    # Split df1 into pure-solid rows and shell rows
    df1_solids = df1[~df1['Element'].isin(df2['Element'])]
    df1_shells  = df1[ df1['Element'].isin(df2['Element'])]
    
    if set_solid_ele != set(df1_solids.Element):
        raise ValueError ('lost elements')
    if set_shell_ele != set(df1_shells.Element):
        raise ValueError ('lost elements')            
    
    # Rename the first combined-mesh connection column so it can be appended to
    # the shell-only connectivity columns.
    df1_shells = df1_shells.dropna(axis=1, how='all')
    df1_shells = df1_shells.rename(columns={'Conn1': 'Conn5'})
    df_shell_final = pd.merge(df2, df1_shells, on=['Element'])
    
    # Combine solid rows and enriched shell rows into one DataFrame
    df_final = pd.concat([df1_solids, df_shell_final])
    
    return df_final

def create_segment_set(elements_df: pd.DataFrame):
    """
    Build LS-DYNA segment-set rows from element connectivity.

    Inputs:
        elements_df (pd.DataFrame): DataFrame from read_elements_from_kfile with
            node columns NID1..NID4 in the required ordering.

    Returns:
        pd.DataFrame: Columns ['n1', 'n2', 'n3', 'n4', 'a1', 'a2', 'a3', 'a4']
        where each output row corresponds to one element row. Node values are
        copied from NID1..NID4 and a1..a4 are initialized to 0.0.
    """
    if not isinstance(elements_df, pd.DataFrame):
        raise TypeError("elements_df must be a pandas DataFrame.")

    required_node_cols = ['NID1', 'NID2', 'NID3', 'NID4']
    missing_cols = [col for col in required_node_cols if col not in elements_df.columns]
    if missing_cols:
        raise ValueError(
            "elements_df must contain columns NID1..NID4. "
            f"Missing: {', '.join(missing_cols)}"
        )

    out_columns = ['n1', 'n2', 'n3', 'n4', 'a1', 'a2', 'a3', 'a4']
    if elements_df.empty:
        return pd.DataFrame(columns=out_columns)

    seg_df = pd.DataFrame(
        {
            'n1': pd.to_numeric(elements_df['NID1'], errors='coerce').astype('Int64'),
            'n2': pd.to_numeric(elements_df['NID2'], errors='coerce').astype('Int64'),
            'n3': pd.to_numeric(elements_df['NID3'], errors='coerce').astype('Int64'),
            'n4': pd.to_numeric(elements_df['NID4'], errors='coerce').astype('Int64'),
            'a1': 0.0,
            'a2': 0.0,
            'a3': 0.0,
            'a4': 0.0,
        }
    )

    return seg_df


def build_element_node_map(elements_df):
    """
    Build an EID -> ordered node tuple map from an elements DataFrame.

    NEEDS CHECKED
    
    Inputs:
        elements_df (pd.DataFrame): DataFrame from read_elements_from_kfile with
            columns ['EID', 'PID', 'NID1'...'NID8'] (NaN-padded to 8 nodes).

    Returns:
        dict: Maps each integer EID to a tuple of integer node IDs, with NaN
            entries omitted. Returns an empty dict if elements_df is empty.

    Raises:
        ValueError: If no NID* columns are found, or if EID values are not unique.
    """
    if elements_df.empty:
        return {}

    node_cols = [col for col in elements_df.columns if col.startswith("NID")]
    if not node_cols:
        raise ValueError("No node columns (NID*) found in elements DataFrame.")

    if elements_df["EID"].duplicated().any():
        dup_ids = elements_df.loc[elements_df["EID"].duplicated(), "EID"].tolist()
        raise ValueError(
            f"Duplicate EID values found; cannot compare uniquely. Sample duplicates: {dup_ids[:10]}"
        )

    element_map = {}
    for _, row in elements_df.iterrows():
        eid = int(row["EID"])
        nodes = [int(row[col]) for col in node_cols if pd.notna(row[col])]
        element_map[eid] = tuple(nodes)

    return element_map


def compare_elements(elements_a_df, elements_b_df):
    """
    Compare EID -> node ordering between two element DataFrames.

    
        NEEDS CHECKED

    For each EID present in both tables, checks whether the ordered node tuple
    is identical. Reports EIDs present in only one table and EIDs where the
    node list differs.

    Inputs:
        elements_a_df (pd.DataFrame): Elements from mesh A, from read_elements_from_kfile.
        elements_b_df (pd.DataFrame): Elements from mesh B, from read_elements_from_kfile.

    Returns:
        tuple:
            - missing_in_a (list[int]): EIDs in B but not in A.
            - missing_in_b (list[int]): EIDs in A but not in B.
            - mismatch_df (pd.DataFrame): Rows for EIDs whose node tuples differ;
              columns ['EID', 'nodes_a', 'nodes_b', 'same_node_set_diff_order'].
              Empty DataFrame if all common EIDs match.
    """
    elem_map_a = build_element_node_map(elements_a_df)
    elem_map_b = build_element_node_map(elements_b_df)

    eids_a = set(elem_map_a.keys())
    eids_b = set(elem_map_b.keys())
    missing_in_b = sorted(eids_a - eids_b)
    missing_in_a = sorted(eids_b - eids_a)

    mismatch_rows = []
    for eid in sorted(eids_a & eids_b):
        nodes_a = elem_map_a[eid]
        nodes_b = elem_map_b[eid]
        if nodes_a != nodes_b:
            mismatch_rows.append({
                "EID": eid,
                "nodes_a": " ".join(str(x) for x in nodes_a),
                "nodes_b": " ".join(str(x) for x in nodes_b),
                "same_node_set_diff_order": (
                    len(nodes_a) == len(nodes_b) and set(nodes_a) == set(nodes_b)
                ),
            })

    return missing_in_a, missing_in_b, pd.DataFrame(mismatch_rows)


def compare_nodes(nodes_a_df, nodes_b_df):
    """
    Compute per-node displacement between two node tables matched by NodeID.

        NEEDS CHECKED

    Inputs:
        nodes_a_df (pd.DataFrame): Node table from mesh A; columns ['NodeID', 'x', 'y', 'z'].
        nodes_b_df (pd.DataFrame): Node table from mesh B; columns ['NodeID', 'x', 'y', 'z'].

    Returns:
        tuple:
            - missing_in_a (list[int]): NodeIDs in B but not in A.
            - missing_in_b (list[int]): NodeIDs in A but not in B.
            - displacement_df (pd.DataFrame): One row per shared NodeID; columns
              ['NodeID', 'x_a', 'y_a', 'z_a', 'x_b', 'y_b', 'z_b',
               'dx', 'dy', 'dz', 'displacement_magnitude'].
              Sorted by displacement_magnitude descending.
              Empty DataFrame (with correct columns) when no shared nodes exist.

    Raises:
        ValueError: If NodeID values are not unique within either input table.
    """
    if nodes_a_df["NodeID"].duplicated().any():
        dup_ids = nodes_a_df.loc[nodes_a_df["NodeID"].duplicated(), "NodeID"].tolist()
        raise ValueError(f"Duplicate NodeID values in mesh A. Sample: {dup_ids[:10]}")
    if nodes_b_df["NodeID"].duplicated().any():
        dup_ids = nodes_b_df.loc[nodes_b_df["NodeID"].duplicated(), "NodeID"].tolist()
        raise ValueError(f"Duplicate NodeID values in mesh B. Sample: {dup_ids[:10]}")

    nodes_a = nodes_a_df.set_index("NodeID")[["x", "y", "z"]]
    nodes_b = nodes_b_df.set_index("NodeID")[["x", "y", "z"]]

    ids_a = set(nodes_a.index.tolist())
    ids_b = set(nodes_b.index.tolist())
    missing_in_b = sorted(ids_a - ids_b)
    missing_in_a = sorted(ids_b - ids_a)

    disp_cols = ["NodeID", "x_a", "y_a", "z_a", "x_b", "y_b", "z_b",
                 "dx", "dy", "dz", "displacement_magnitude"]
    common = sorted(ids_a & ids_b)
    if not common:
        return missing_in_a, missing_in_b, pd.DataFrame(columns=disp_cols)

    left = nodes_a.rename(columns={"x": "x_a", "y": "y_a", "z": "z_a"})
    right = nodes_b.rename(columns={"x": "x_b", "y": "y_b", "z": "z_b"})
    disp = left.merge(right, on = 'NodeID')
    disp["dx"] = disp["x_b"] - disp["x_a"]
    disp["dy"] = disp["y_b"] - disp["y_a"]
    disp["dz"] = disp["z_b"] - disp["z_a"]
    disp["displacement_magnitude"] = (
        disp["dx"].pow(2) + disp["dy"].pow(2) + disp["dz"].pow(2)
    ).pow(0.5)

    displacement_df = disp.reset_index()
    displacement_df = displacement_df.sort_values("displacement_magnitude", ascending=False)
    return missing_in_a, missing_in_b, displacement_df


def _tet_signed_volume(p0, p1, p2, p3):
    """
    Compute the signed volume of a 4-node tetrahedron defined by four 3-D points.

    A positive result indicates p3 lies on the positive side of the plane
    formed by p0->p1->p2 (right-hand rule, CCW base viewed from below).

    Inputs:
        p0, p1, p2, p3 (np.ndarray): Shape-(3,) arrays of x, y, z coordinates.

    Returns:
        float: Signed volume of the tetrahedron.

        #TODO: VERIFY THIS WORKS
    """
    return float(np.dot(np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0)


def _penta_signed_volume(p1, p2, p3, p4, p5, p6):
    """
    Compute the signed volume of a 6-node LS-DYNA pentahedron using a
    reduced (1-point) integration Jacobian-at-centroid formula. (Collapsed-Brick Centroid)

    LS-DYNA represents a pentahedron in the 8-node *ELEMENT_SOLID card as
    n1 n2 n3 n4 n5 n6 n6 n6 (node 6 repeated in the last three slots).
    Node convention:
        n1, n2, n3, n4 : quadrilateral base (counter-clockwise viewed
                          from outside the element).
        n5             : top node above the n1-n2 edge.
        n6             : top node above the n3-n4 edge (repeated in
                          positions 6-8 of the hexahedron connectivity).

    Inputs:
        p1..p6 (np.ndarray): Shape-(3,) coordinate arrays for the 6 unique
            pentahedron nodes, in n1..n6 order.

    Returns:
        float: Signed volume 
    """
    a = p2 + p3 - p1 - p4 - p5 + p6
    b = p3 + p4 - p1 - p2 - p5 + p6
    c = p5 + 3.0 * p6 - p1 - p2 - p3 - p4
    return float(np.dot(a, np.cross(b, c)) / 64.0)


def _hex_signed_volume(p1, p2, p3, p4, p5, p6, p7, p8):
    """
    Compute the signed volume of an 8-node LS-DYNA hexahedron using a
    reduced (1-point) integration Jacobian-at-centroid formula, matching
    LS-DYNA's default constant-stress (ELFORM=1) solid element formulation.

    Node convention:
        n1-n2-n3-n4: bottom face (CCW from below).
        n5-n6-n7-n8: top face (n5 above n1, n6 above n2, n7 above n3, n8 above n4).

    Inputs:
        p1..p8 (np.ndarray): Shape-(3,) coordinate arrays for the 8
            hexahedron nodes, in n1..n8 order.

    Returns:
        float: Signed volume (positive for a valid, non-inverted element).
    """
    a = -p1 + p2 + p3 - p4 - p5 + p6 + p7 - p8
    b = -p1 - p2 + p3 + p4 - p5 - p6 + p7 + p8
    c = -p1 - p2 - p3 - p4 + p5 + p6 + p7 + p8
    return float(np.dot(a, np.cross(b, c)) / 64.0)


_HEX_NATURAL_COORDS = [
    (-1.0, -1.0, -1.0),
    ( 1.0, -1.0, -1.0),
    ( 1.0,  1.0, -1.0),
    (-1.0,  1.0, -1.0),
    (-1.0, -1.0,  1.0),
    ( 1.0, -1.0,  1.0),
    ( 1.0,  1.0,  1.0),
    (-1.0,  1.0,  1.0),
]


def _hex_jacobian_at(hex_pts, r, s, t):
    """
    Evaluate the isoparametric Jacobian determinant of an 8-node trilinear
    hexahedron map at a single parametric location (r, s, t).

    Unlike _hex_signed_volume (which only evaluates the Jacobian at the
    element centroid, matching LS-DYNA's reduced 1-point integration), this
    evaluates the local Jacobian at an arbitrary (r, s, t), which is needed
    to detect elements that are locally inverted/tangled in one region even
    though their centroid Jacobian (and therefore reported volume) is
    positive -- e.g. a "bow-tie" quad face where two corners are transposed.

    Inputs:
        hex_pts (list[np.ndarray]): Eight shape-(3,) coordinate arrays,
            p1..p8, in standard LS-DYNA hex8 node order (n1-n4 bottom face
            CCW from below, n5-n8 top face). For collapsed tet4/penta6
            elements, pass the 8-slot representation with repeated nodes
            (matching the *ELEMENT_SOLID card convention).
        r, s, t (float): Parametric coordinates in [-1, 1].

    Returns:
        float: Determinant of the Jacobian matrix [dx/dr; dx/ds; dx/dt] at
            (r, s, t). Positive for a locally right-handed (valid) mapping;
            negative indicates a locally inverted/tangled region.

    #TODO: VERIFY THIS WORKS
    """
    dxdr = np.zeros(3)
    dxds = np.zeros(3)
    dxdt = np.zeros(3)
    for p, (ri, si, ti) in zip(hex_pts, _HEX_NATURAL_COORDS):
        dxdr += 0.125 * ri * (1.0 + s * si) * (1.0 + t * ti) * p
        dxds += 0.125 * si * (1.0 + r * ri) * (1.0 + t * ti) * p
        dxdt += 0.125 * ti * (1.0 + r * ri) * (1.0 + s * si) * p
    return float(np.dot(dxdr, np.cross(dxds, dxdt)))


def _vectorized_element_corner_points(elements_df, nodes_df, _precomputed_id_to_idx=None, _precomputed_coords_arr=None):
    """
    Shared vectorized preprocessing for compute_element_volumes and
    compute_element_distortion: groups elements by unique real node count
    (4/6/8) and gathers every group's corner coordinates into ONE numpy
    array per group, via a single vectorized NodeID->row-position lookup
    (pandas Series.reindex, hash-table based, implemented in C) -- NOT a
    per-node Python-level dict/`.loc[]` lookup repeated per element.

    WHY THIS EXISTS: compute_element_volumes/compute_element_distortion
    used to (a) loop over elements_df one row at a time (via iterrows()),
    and (b) apply their geometric formula (a handful of vector adds/
    cross/dot products) to ONE element's points at a time, in pure Python.
    Even after fixing the node-coordinate LOOKUP itself to use a plain
    dict instead of pandas `.loc[]` (a real, separate speedup), the
    per-element formula evaluation loop itself remained the dominant cost
    at real whole-model scale: measured at roughly 60-70 microseconds per
    element even with fast dict lookups, which is ~150+ seconds for 2.4
    million elements, PER CALL -- and both functions are called TWICE
    each (before/after) in the multi-part repair scripts, on the FULL
    element table, not just bad elements. The volume/Jacobian formulas
    themselves (see _tet_signed_volume, _penta_signed_volume,
    _hex_signed_volume, _hex_jacobian_at) are simple linear combinations
    of 3-D vectors followed by one cross/dot product -- entirely
    vectorizable with numpy across ALL elements of a given node-count
    GROUP simultaneously (one array operation covering millions of
    elements at once, instead of a Python-level loop iterating that many
    times). This helper does the grouping/gathering; the two calling
    functions apply their own (now-vectorized) formula per group.

    Inputs:
        elements_df (pd.DataFrame): Elements; columns include 'EID',
            'PID', 'NID1'..'NID8'.
        nodes_df (pd.DataFrame): Node coordinates; columns 'NodeID', 'x',
            'y', 'z'.
        _precomputed_id_to_idx (pd.Series | None): Internal perf hook --
            if the caller already has a NodeID->row-position map built
            (e.g. build_all_part_surfaces, which calls this function
            once PER PART and would otherwise rebuild this O(total mesh
            nodes) index from scratch on every single call -- a REAL,
            confirmed production bottleneck at ~700-part whole-body
            scale), pass it here to skip rebuilding it. None (default)
            preserves the EXACT original behavior for every existing
            caller (compute_element_volumes, compute_element_distortion,
            build_hard_surface_index's own default path).
        _precomputed_coords_arr (np.ndarray | None): Companion to the
            above -- the matching (N, 3) coordinate array. Must be
            provided together with _precomputed_id_to_idx (both or
            neither); using one without the other raises.

    Returns:
        tuple:
            eid_arr (np.ndarray): EID per row, original elements_df order.
            pid_arr (np.ndarray): PID per row, original elements_df order.
            n_nodes_arr (np.ndarray[int]): Unique real node count per row
                (original order) -- used by callers to detect/raise on any
                unsupported count (not 4, 6, or 8).
            groups (dict[int, tuple]): {node_count: (row_indices, pts,
                valid_mask)} for each of 4/6/8 actually present --
                row_indices (np.ndarray[int]): positions (into the
                    ORIGINAL elements_df/eid_arr/pid_arr/n_nodes_arr) of
                    every element in this group, in original relative
                    order.
                pts (np.ndarray, shape (M, node_count, 3)): Gathered
                    corner coordinates for this group's M elements, in
                    the element's own n1..n[4|6|8] column order.
                valid_mask (np.ndarray[bool], shape (M,)): False for any
                    element in this group with at least one node ID not
                    present in nodes_df (mirrors the old per-element
                    try/except KeyError -> NaN-out-this-row behavior);
                    `pts` still has SOME (arbitrary, safe-to-ignore)
                    placeholder values at those positions so the group's
                    vectorized formula can still run over the whole
                    array without special-casing -- callers must apply
                    valid_mask (e.g. np.where(valid_mask, result, nan))
                    to blank out results for those rows before use.
    """
    n = len(elements_df)
    element_node_lists = _unique_node_lists_fast(elements_df)
    n_nodes_arr = np.fromiter((len(nl) for nl in element_node_lists), dtype=np.int64, count=n)
    eid_arr = elements_df['EID'].to_numpy()
    pid_arr = elements_df['PID'].to_numpy()

    if _precomputed_id_to_idx is not None or _precomputed_coords_arr is not None:
        if _precomputed_id_to_idx is None or _precomputed_coords_arr is None:
            raise ValueError(
                "_precomputed_id_to_idx and _precomputed_coords_arr must both be "
                "provided together, or both left as None."
            )
        id_to_idx = _precomputed_id_to_idx
        coords_arr = _precomputed_coords_arr
    else:
        node_ids_arr = nodes_df['NodeID'].to_numpy()
        coords_arr = nodes_df[['x', 'y', 'z']].to_numpy(dtype=float)
        # NodeID -> integer row-position map, used for one vectorized
        # reindex() per group instead of per-node dict/`.loc[]` lookups.
        id_to_idx = pd.Series(np.arange(len(node_ids_arr)), index=node_ids_arr)

    groups = {}
    for count in (4, 6, 8):
        group_mask = (n_nodes_arr == count)
        row_indices = np.nonzero(group_mask)[0]
        if row_indices.size == 0:
            continue

        node_id_matrix = np.array(
            [element_node_lists[i] for i in row_indices], dtype=np.int64
        )
        flat_positions = id_to_idx.reindex(node_id_matrix.ravel())
        missing = flat_positions.isna().to_numpy().reshape(node_id_matrix.shape)
        valid_mask = ~missing.any(axis=1)
        # Missing NodeIDs get a placeholder position of 0 (arbitrary,
        # always in-bounds) -- safe because valid_mask blanks out any
        # row that used a placeholder before the caller uses the result.
        positions = flat_positions.fillna(0).to_numpy().astype(np.int64).reshape(node_id_matrix.shape)
        pts = coords_arr[positions]  # (M, count, 3)

        groups[count] = (row_indices, pts, valid_mask)

    return eid_arr, pid_arr, n_nodes_arr, groups


def _raise_on_unsupported_node_count(eid_arr, n_nodes_arr):
    """
    Shared check for compute_element_volumes/compute_element_distortion:
    raises on the FIRST (in original elements_df row order) element whose
    unique real node count isn't 4, 6, or 8 -- matching the original
    per-row iterrows()-based implementations' "raise as soon as
    encountered" behavior/error message exactly.
    """
    unsupported_mask = ~np.isin(n_nodes_arr, [4, 6, 8])
    if unsupported_mask.any():
        bad_idx = int(np.nonzero(unsupported_mask)[0][0])
        raise ValueError(
            f"Unsupported element node count for EID {int(eid_arr[bad_idx])}: "
            f"found {int(n_nodes_arr[bad_idx])} unique real nodes; supported counts are 4, 6, 8."
        )


def compute_element_distortion(elements_df, nodes_df):
    """
    Detect LOCALLY TANGLED solid elements (e.g. a "bow-tie" quad face where
    two corners are transposed) by checking whether the isoparametric
    Jacobian determinant CHANGES SIGN across an element's corners.

    This is a LOCAL, corner-to-corner sign-change check only. It answers
    "does this element fold back on itself somewhere between its corners?"
    It does NOT answer "is this element valid/non-inverted overall?" -- for
    that, use the volume sign from compute_element_volumes instead. In
    particular, this function will MISS (not flag) an element that is
    COMPLETELY / UNIFORMLY inverted (all corners share the same negative
    sign, e.g. a hex whose top and bottom faces are simply swapped) because
    there is no sign CHANGE among the corners to detect -- that case is a
    global inversion, not a local tangle, and must be caught separately via
    compute_element_volumes' negative volume.

    A true tet4 has a constant Jacobian everywhere (its map is affine), so
    it cannot be locally tangled the way a penta6/hex8 can; tet4 elements
    are therefore never flagged as distorted here (a negative tet4 volume
    from compute_element_volumes already indicates a fully inverted, not
    partially tangled, element).

    Node ordering / element-type detection matches compute_element_volumes:
        4 unique nodes -> tet4 (n4 repeated in slots 4-8)
        6 unique nodes -> penta6 (n6 repeated in slots 6-8)
        8 unique nodes -> hex8

    Inputs:
        elements_df (pd.DataFrame): Elements from read_elements_from_kfile;
            columns include 'EID', 'PID', 'NID1'..'NID8'.
        nodes_df (pd.DataFrame): Node coordinates from read_nodes_from_kfile;
            columns 'NodeID', 'x', 'y', 'z'.

    Returns:
        pd.DataFrame: One row per element; columns:
            'EID', 'PID', 'n_nodes',
            'jacobian_min'   : smallest sampled corner Jacobian determinant.
            'jacobian_max'   : largest sampled corner Jacobian determinant.
            'jacobian_ratio' : jacobian_min / jacobian_max (NaN if
                               jacobian_max is 0).
            'is_distorted'   : True ONLY if the corner Jacobians CHANGE SIGN
                               (jacobian_min <= 0 < jacobian_max) for a
                               penta6/hex8 element, meaning the element is
                               locally tangled/bow-tied. False for a
                               uniformly-signed element even if that sign
                               is negative (fully inverted, not tangled) --
                               check compute_element_volumes' volume sign
                               separately to catch that case. Always False
                               for tet4.

    Raises:
        ValueError: If an element has a number of unique real nodes other
            than 4, 6, or 8.

    Limitations (sampling strategy is a heuristic, not exhaustive):
        The hex/penta Jacobian is a MULTILINEAR (not linear) function of
        (r, s, t), so sampling only at the nodal corners 
        is not a  guarantee that no sign change exists strictly between corners. 

        Penta6 uses ONLY the 4 quad-base corners (n1-n4); the n5/n6 ridge
        corners are excluded because their Jacobian is algebraically 0
        for ANY penta6 (valid or not) in this collapsed-hex representation

        This function DOES NOT detect complete/uniform inversion (all
        corners negative, no sign change) -- e.g. a hex with its top and
        bottom faces simply swapped reports jacobian_min == jacobian_max ==
        a negative value and is_distorted == False here.

    #TODO: VERIFY THIS WORKS

    PERFORMANCE: fully vectorized with numpy across all elements of each
    node-count group (4/6/8) at once, via _vectorized_element_corner_points
    -- NOT a per-element Python loop. An earlier version looped over
    elements_df one row at a time (first via iterrows() + pandas `.loc[]`
    per node, then, after a first fix, via a plain dict lookup per node
    but STILL evaluating the 8-corner Jacobian formula one element at a
    time in pure Python). Even with fast O(1) dict lookups, that
    per-element formula evaluation alone was measured at roughly
    60-70+ microseconds/element at scale -- on a real 2.4-million-solid-
    element whole-model mesh, that is 150+ seconds PER CALL, and this
    function is called TWICE (before/after) in the multi-part repair
    scripts, on the FULL element table (not just bad elements). The
    corner-Jacobian formula (see _hex_jacobian_at) is a linear combination
    of 3-D corner points -- entirely vectorizable across every element in
    a group simultaneously (one small, fixed number of numpy array
    operations covering millions of elements at once, rather than a
    Python-level loop iterating that many times).
    """
    eid_arr, pid_arr, n_nodes_arr, groups = _vectorized_element_corner_points(elements_df, nodes_df)
    _raise_on_unsupported_node_count(eid_arr, n_nodes_arr)

    n_total = len(elements_df)
    jmin_arr = np.full(n_total, np.nan)
    jmax_arr = np.full(n_total, np.nan)
    jratio_arr = np.full(n_total, np.nan)
    distorted_arr = np.zeros(n_total, dtype=bool)

    for count, (row_indices, pts, valid_mask) in groups.items():
        # Build the 8-slot "hex_pts" representation exactly as the
        # scalar _hex_jacobian_at path did (repeated corner for
        # collapsed tet4/penta6 elements), now as (M, 3) arrays -- one
        # entry per hex corner SLOT, each holding all M elements' points
        # for that slot at once.
        if count == 4:
            hex_pts = [pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3],
                       pts[:, 3], pts[:, 3], pts[:, 3], pts[:, 3]]
            corner_natural_coords = _HEX_NATURAL_COORDS[:4]
        elif count == 6:
            hex_pts = [pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3],
                       pts[:, 4], pts[:, 5], pts[:, 5], pts[:, 5]]
            corner_natural_coords = _HEX_NATURAL_COORDS[:4]
        else:  # count == 8
            hex_pts = [pts[:, k] for k in range(8)]
            corner_natural_coords = _HEX_NATURAL_COORDS[:8]

        m = pts.shape[0]
        jac_per_corner = []
        # Outer loop: which SAMPLE point (r, s, t) to evaluate the
        # Jacobian at -- 4 sample points for tet4/penta6 (quad-base
        # corners only), 8 for hex8. Inner loop: the FULL 8-slot
        # trilinear-shape-function sum (matches _hex_jacobian_at exactly
        # -- always all 8 slots/coefficients, regardless of sample count).
        # Both loops are only 4-8 iterations each (not per-element), so
        # this is a small, fixed number of vectorized numpy operations,
        # each covering all M elements in the group simultaneously.
        for (r, s, t) in corner_natural_coords:
            dxdr = np.zeros((m, 3))
            dxds = np.zeros((m, 3))
            dxdt = np.zeros((m, 3))
            for p, (ri, si, ti) in zip(hex_pts, _HEX_NATURAL_COORDS):
                dxdr += 0.125 * ri * (1.0 + s * si) * (1.0 + t * ti) * p
                dxds += 0.125 * si * (1.0 + r * ri) * (1.0 + t * ti) * p
                dxdt += 0.125 * ti * (1.0 + r * ri) * (1.0 + s * si) * p
            jac = np.einsum('ij,ij->i', dxdr, np.cross(dxds, dxdt))
            jac_per_corner.append(jac)

        jac_stack = np.stack(jac_per_corner, axis=1)  # (M, n_sample_corners)
        jmin = jac_stack.min(axis=1)
        jmax = jac_stack.max(axis=1)
        # np.where evaluates BOTH branches eagerly (unlike the scalar
        # `(jmin/jmax) if jmax != 0 else nan` this replaces), so a
        # division by an exact 0.0 jmax triggers a harmless
        # RuntimeWarning even though the result is correctly overwritten
        # with NaN right after -- suppressed here since it's expected,
        # not a real problem.
        with np.errstate(divide='ignore', invalid='ignore'):
            jratio = np.where(jmax != 0, jmin / jmax, np.nan)
        is_distorted = (count > 4) & (jmin < 0.0) & (0.0 < jmax)

        jmin_arr[row_indices] = np.where(valid_mask, jmin, np.nan)
        jmax_arr[row_indices] = np.where(valid_mask, jmax, np.nan)
        jratio_arr[row_indices] = np.where(valid_mask, jratio, np.nan)
        distorted_arr[row_indices] = np.where(valid_mask, is_distorted, False)

    return pd.DataFrame({
        'EID': eid_arr.astype(int), 'PID': pid_arr.astype(int), 'n_nodes': n_nodes_arr,
        'jacobian_min': jmin_arr, 'jacobian_max': jmax_arr,
        'jacobian_ratio': jratio_arr, 'is_distorted': distorted_arr,
    })


def _min_valid_jacobian(pts, n):
    """
    Return a single scalar "is this element valid?" measure for one solid
    element, used as an optimization constraint by repair_inverted_elements.

    Uses the SAME reduced-integration volume as compute_element_volumes for
    tet4 (its map is affine, so a single volume fully describes validity),
    and the SAME corner-Jacobian sampling as compute_element_distortion for
    penta6/hex8 (restricted to the quad-base corners for penta6, since the
    n5/n6 ridge corners are algebraically degenerate there regardless of
    validity -- see compute_element_distortion). Requiring this single
    scalar to stay positive enforces BOTH a non-inverted element (matches
    compute_element_volumes' sign) AND a non-tangled/bow-tied element
    (matches compute_element_distortion's is_distorted check) in one
    constraint, which a centroid-volume-only constraint cannot guarantee.

    Inputs:
        pts (list[np.ndarray]): Unique element node coordinates in
            n1..n[4|6|8] order.
        n (int): Number of unique nodes (4, 6, or 8).

    Returns:
        float: tet4 -> the (constant) signed volume.
               penta6/hex8 -> the smallest sampled corner Jacobian.
               NaN for any other node count.

    #TODO: VERIFY THIS WORKS
    """
    if n == 4:
        return _tet_signed_volume(pts[0], pts[1], pts[2], pts[3])
    if n == 6:
        hex_pts = [pts[0], pts[1], pts[2], pts[3], pts[4], pts[5], pts[5], pts[5]]
        corners = _HEX_NATURAL_COORDS[:4]
    elif n == 8:
        hex_pts = pts
        corners = _HEX_NATURAL_COORDS[:8]
    else:
        return float('nan')
    return min(_hex_jacobian_at(hex_pts, r, s, t) for (r, s, t) in corners)


_HEX_NATURAL_COORDS_ARR = np.array(_HEX_NATURAL_COORDS)  # shape (8, 3)


def _hex_corner_coeff_matrices():
    """
    Precompute the constant (8 corners x 8 slots) coefficient matrices that
    evaluate the hex/penta isoparametric Jacobian, AND its analytic
    gradient, at every sampled corner simultaneously for an arbitrary batch
    of elements. These depend only on the fixed natural-coordinate corner
    layout (_HEX_NATURAL_COORDS), never on element geometry, so they are
    computed once at import time.

    CR[c, i] is the coefficient of slot-i's point when forming dx/dr at
    corner c (see _hex_jacobian_at for the scalar, single-element,
    single-corner equivalent); CS/CT are the analogous dx/ds, dx/dt
    coefficients.

    Returns:
        tuple[np.ndarray]: (CR, CS, CT), each shape (8, 8).
    """
    r = _HEX_NATURAL_COORDS_ARR[:, 0]
    s = _HEX_NATURAL_COORDS_ARR[:, 1]
    t = _HEX_NATURAL_COORDS_ARR[:, 2]
    r_slot, s_slot, t_slot = r[None, :], s[None, :], t[None, :]
    r_crn, s_crn, t_crn = r[:, None], s[:, None], t[:, None]
    CR = 0.125 * r_slot * (1.0 + s_crn * s_slot) * (1.0 + t_crn * t_slot)
    CS = 0.125 * s_slot * (1.0 + r_crn * r_slot) * (1.0 + t_crn * t_slot)
    CT = 0.125 * t_slot * (1.0 + r_crn * r_slot) * (1.0 + s_crn * s_slot)
    return CR, CS, CT


_HEX_CR, _HEX_CS, _HEX_CT = _hex_corner_coeff_matrices()


def _batched_hexlike_value_and_grad(point_batch, n_corners):
    """
    Vectorized value + analytic gradient of _min_valid_jacobian's hex/penta
    branch (the minimum sampled corner Jacobian) for a whole batch of
    elements at once, used by repair_inverted_elements so SciPy never has
    to finite-difference the repair-zone constraints element-by-element.

    Point ordering / repeated-slot convention matches _min_valid_jacobian:
    hex8 passes 8 unique points (n_corners=8); penta6 passes 8 slots with
    the ridge point repeated in the last 3 (see _min_valid_jacobian), and
    only the first 4 corners are sampled (n_corners=4) since the ridge
    corners are algebraically degenerate for any penta6 (see
    compute_element_distortion).

    The analytic gradient uses the scalar triple product identity for
    f(a, b, c) = a . (b x c) with a, b, c each linear in the 8 slot points:
        grad_slot_i(f) = CR[c*, i] * (b x c) + CS[c*, i] * (c x a)
                         + CT[c*, i] * (a x b)
    evaluated at each element's own arg-min corner c* (a valid subgradient
    of the corner-wise minimum). Verified against central finite
    differences to ~1e-10 during development.

    Inputs:
        point_batch (np.ndarray): shape (M, 8, 3); the 8 slot coordinates
            for M elements (repeated slots duplicated, matching
            _min_valid_jacobian's hex_pts construction).
        n_corners (int): 8 for hex8, 4 for penta6.

    Returns:
        tuple:
            values (np.ndarray): shape (M,); the minimum sampled corner
                Jacobian per element (matches _min_valid_jacobian for n in
                {6, 8}).
            slot_grad (np.ndarray): shape (M, 8, 3); analytic gradient of
                `values[m]` with respect to the physical point occupying
                each of the 8 slots. Repeated slots (e.g. penta6's ridge
                point in slots 5-7) each carry only THEIR OWN partial
                contribution -- callers must SUM contributions for slots
                that share one physical node (e.g. via COO-matrix duplicate
                summation), exactly as the chain rule requires for a
                repeated variable.
    """
    m = point_batch.shape[0]
    CR, CS, CT = _HEX_CR[:n_corners], _HEX_CS[:n_corners], _HEX_CT[:n_corners]

    dxdr = np.einsum('ci,mij->mcj', CR, point_batch)
    dxds = np.einsum('ci,mij->mcj', CS, point_batch)
    dxdt = np.einsum('ci,mij->mcj', CT, point_batch)

    bxc = np.cross(dxds, dxdt, axis=-1)
    cxa = np.cross(dxdt, dxdr, axis=-1)
    axb = np.cross(dxdr, dxds, axis=-1)

    corner_jac = np.sum(dxdr * bxc, axis=-1)            # (M, n_corners)
    argmin_corner = np.argmin(corner_jac, axis=1)       # (M,)
    row_idx = np.arange(m)
    values = corner_jac[row_idx, argmin_corner]

    bxc_at = bxc[row_idx, argmin_corner]                # (M, 3)
    cxa_at = cxa[row_idx, argmin_corner]
    axb_at = axb[row_idx, argmin_corner]

    CR_at = CR[argmin_corner]                           # (M, 8)
    CS_at = CS[argmin_corner]
    CT_at = CT[argmin_corner]

    slot_grad = (
        CR_at[:, :, None] * bxc_at[:, None, :]
        + CS_at[:, :, None] * cxa_at[:, None, :]
        + CT_at[:, :, None] * axb_at[:, None, :]
    )
    return values, slot_grad


def _corner_jacobians_batch(point_batch, n_corners):
    """
    Raw (M, n_corners) sampled corner Jacobians only (no gradient). Used
    once, up front, to CALIBRATE the fixed per-element softmin sharpness
    (see _frozen_softmin_beta) -- not used inside the main optimization
    loop, so it doesn't need a paired gradient.
    """
    CR, CS, CT = _HEX_CR[:n_corners], _HEX_CS[:n_corners], _HEX_CT[:n_corners]
    dxdr = np.einsum('ci,mij->mcj', CR, point_batch)
    dxds = np.einsum('ci,mij->mcj', CS, point_batch)
    dxdt = np.einsum('ci,mij->mcj', CT, point_batch)
    bxc = np.cross(dxds, dxdt, axis=-1)
    return np.sum(dxdr * bxc, axis=-1)


_SOFTMIN_SHARPNESS = 6.0


def _frozen_softmin_beta(coords, group_idx, n_corners, k=_SOFTMIN_SHARPNESS):
    """
    Per-element softmin sharpness (see _batched_hexlike_softmin_value_and_
    grad), calibrated ONCE from a baseline configuration (coords, typically
    the repair zone's ORIGINAL, pre-optimization geometry) and then held
    fixed for the entire optimization. beta_m = k / range-of-corner-
    Jacobians-within-element-m, so elements with widely spread corners get
    a gentler (smaller-beta) blend and elements with tightly clustered
    corners get a sharper (larger-beta, closer to the true min) blend,
    without needing a single global constant tuned to one mesh's units/
    element size. Must NOT be recomputed from the CURRENT (mid-optimization)
    coordinates -- see _batched_hexlike_softmin_value_and_grad's docstring
    for why that would silently corrupt the analytic gradient.
    """
    if len(group_idx) == 0:
        return np.zeros(0)
    corner_jac = _corner_jacobians_batch(coords[group_idx], n_corners)
    scale = np.maximum(np.ptp(corner_jac, axis=1), 1e-9)
    return k / scale


def _batched_hexlike_softmin_value_and_grad(point_batch, n_corners, beta):
    """
    Smooth (soft-min) stand-in for _batched_hexlike_value_and_grad's hard
    corner-wise minimum, used ONLY as the SLSQP-facing constraint during
    optimization (see repair_inverted_elements). The hard min's argmin-based
    gradient is a valid subgradient but creates a genuine non-smooth kink
    whenever two corners are near-tied -- in practice this was observed to
    make SLSQP oscillate/stall on real meshes (the "active" corner
    flip-flopping between iterations, with the worst-constraint value
    swinging by 10s of units iteration to iteration instead of settling).

    Replaces min(v_1..v_C) with
        softmin(v) = v_min - (1/beta)*log(sum_i exp(-beta*(v_i - v_min)))
    (v_min-shifted purely for numerical stability; algebraically identical
    to the unshifted log-sum-exp). This is smooth (C-infinity) everywhere
    and satisfies the standard log-sum-exp sandwich
        min(v) - log(C)/beta <= softmin(v) <= min(v)
    so requiring softmin(v) >= floor is ALWAYS at least as strict as the
    true min(v) >= floor (never a false accept of a genuinely-invalid
    element), at the cost of some conservatism (a false reject only
    possible within log(C)/beta of the boundary).

    IMPORTANT: `beta` must be a FIXED array (one value per batch element),
    precomputed once by _frozen_softmin_beta and held constant for the life
    of the optimization. If beta were instead re-derived from the CURRENT
    point_batch on every call (e.g. from that call's own corner spread), it
    would be a hidden function of x whose derivative this formula does not
    account for, silently corrupting the analytic gradient -- caught during
    development by comparing against finite differences (~1e-11 relative
    error with frozen beta, vs. an ~4e-2 error that did NOT shrink with
    smaller epsilon when beta was recomputed from x each call, proving it
    was a real formula bug and not finite-difference truncation error).

    Inputs:
        point_batch (np.ndarray): shape (M, 8, 3), same slot convention as
            _batched_hexlike_value_and_grad.
        n_corners (int): 8 for hex8, 4 for penta6.
        beta (np.ndarray): shape (M,), fixed per-element sharpness from
            _frozen_softmin_beta.

    Returns:
        tuple: (values, slot_grad) with the same shapes/semantics as
            _batched_hexlike_value_and_grad, but the smooth softmin instead
            of the hard corner-wise minimum.
    """
    CR, CS, CT = _HEX_CR[:n_corners], _HEX_CS[:n_corners], _HEX_CT[:n_corners]

    dxdr = np.einsum('ci,mij->mcj', CR, point_batch)
    dxds = np.einsum('ci,mij->mcj', CS, point_batch)
    dxdt = np.einsum('ci,mij->mcj', CT, point_batch)

    bxc = np.cross(dxds, dxdt, axis=-1)
    cxa = np.cross(dxdt, dxdr, axis=-1)
    axb = np.cross(dxdr, dxds, axis=-1)

    corner_jac = np.sum(dxdr * bxc, axis=-1)            # (M, n_corners)
    v_min = corner_jac.min(axis=1, keepdims=True)
    beta_col = beta[:, None]
    shifted = corner_jac - v_min                          # (M, n_corners), >= 0
    unnorm_w = np.exp(-beta_col * shifted)
    z_sum = unnorm_w.sum(axis=1, keepdims=True)
    values = (v_min - np.log(z_sum) / beta_col)[:, 0]
    weights = unnorm_w / z_sum                            # (M, n_corners), rows sum to 1

    grad_per_corner = (
        CR[None, :, :, None] * bxc[:, :, None, :]
        + CS[None, :, :, None] * cxa[:, :, None, :]
        + CT[None, :, :, None] * axb[:, :, None, :]
    )  # (M, n_corners, 8, 3)
    slot_grad = np.einsum('mc,mcsj->msj', weights, grad_per_corner)
    return values, slot_grad


# Coefficients for the isoparametric Jacobian evaluated ONLY at the element
# centroid (r=s=t=0), i.e. LS-DYNA's reduced 1-point integration location --
# see _hex_signed_volume/_penta_signed_volume. At (0,0,0) the (1+s*si) and
# (1+t*ti) factors in _hex_corner_coeff_matrices collapse to 1, leaving a
# fixed (not per-corner) coefficient per slot.
_HEX_CENTROID_CR = 0.125 * _HEX_NATURAL_COORDS_ARR[:, 0]  # shape (8,)
_HEX_CENTROID_CS = 0.125 * _HEX_NATURAL_COORDS_ARR[:, 1]
_HEX_CENTROID_CT = 0.125 * _HEX_NATURAL_COORDS_ARR[:, 2]


def _batched_hexlike_centroid_value_and_grad(point_batch):
    """
    Vectorized value + analytic gradient of the CENTROID (single-point,
    r=s=t=0) isoparametric Jacobian for a whole batch of hex8/penta6
    elements -- equal to 1/8 of _hex_signed_volume's reported LS-DYNA
    volume (volume = centroid Jacobian * 8, the [-1,1]^3 reference cube's
    quadrature weight under 1-point/midpoint integration). Substituting the
    penta6 repeated-ridge-node convention (p6=p7=p8) into this same formula
    exactly reproduces _penta_signed_volume (verified algebraically), so
    ONE function correctly covers both hex8 and penta6 -- no n_corners
    argument needed; always pass the full 8-slot point_batch.

    This is a NECESSARY companion to _batched_hexlike_value_and_grad (the
    8-corner minimum), not a replacement: the corner check catches local/
    internal tangles invisible at the centroid, but does NOT guarantee the
    centroid value -- LS-DYNA's actual reported element volume, and the
    metric solid_elements_failed_quality_check.k is generated from -- is
    also positive. This was discovered to be a real, not merely
    theoretical, gap via a synthetic counter-example during development: an
    element optimization "succeeded" with all 8 corner Jacobians above the
    epsilon floor while compute_element_volumes' centroid volume was still
    -0.043. repair_inverted_elements requires BOTH constraints together.

    Unlike the corner-minimum, this needs no softmin smoothing: it's a
    single fixed linear combination of the 8 slot points (no argmin/
    branching), hence already exactly smooth, so the same value/gradient
    are used for both the SLSQP-facing constraint and the exact floor/
    final-check reporting.

    Inputs:
        point_batch (np.ndarray): shape (M, 8, 3).

    Returns:
        tuple:
            values (np.ndarray): shape (M,); centroid Jacobian per element.
            slot_grad (np.ndarray): shape (M, 8, 3); analytic gradient of
                `values[m]` with respect to the physical point in each slot
                (repeated slots, e.g. penta6's ridge node, each carry only
                their own partial contribution -- see
                _batched_hexlike_value_and_grad's docstring on why callers
                must sum these for shared physical nodes).
    """
    dxdr = np.einsum('i,mij->mj', _HEX_CENTROID_CR, point_batch)
    dxds = np.einsum('i,mij->mj', _HEX_CENTROID_CS, point_batch)
    dxdt = np.einsum('i,mij->mj', _HEX_CENTROID_CT, point_batch)

    bxc = np.cross(dxds, dxdt, axis=-1)
    cxa = np.cross(dxdt, dxdr, axis=-1)
    axb = np.cross(dxdr, dxds, axis=-1)

    values = np.sum(dxdr * bxc, axis=-1)  # (M,)

    slot_grad = (
        _HEX_CENTROID_CR[None, :, None] * bxc[:, None, :]
        + _HEX_CENTROID_CS[None, :, None] * cxa[:, None, :]
        + _HEX_CENTROID_CT[None, :, None] * axb[:, None, :]
    )  # (M, 8, 3)
    return values, slot_grad


def _batched_tet_value_and_grad(point_batch):
    """
    Vectorized value + analytic gradient of the tet4 signed volume (see
    _tet_signed_volume) for a whole batch of elements at once.

    Uses the same scalar triple product identity as
    _batched_hexlike_value_and_grad, specialized to
    vol = (1/6) u.(v x w) with u = p1-p0, v = p2-p0, w = p3-p0:
        grad_p1 = (v x w)/6,  grad_p2 = (w x u)/6,  grad_p3 = (u x v)/6,
        grad_p0 = -(grad_p1 + grad_p2 + grad_p3)
    Verified against central finite differences to ~1e-11 during
    development.

    Inputs:
        point_batch (np.ndarray): shape (M, 4, 3); n1..n4 tet corner points.

    Returns:
        tuple:
            values (np.ndarray): shape (M,); signed tet volume per element
                (matches _tet_signed_volume).
            slot_grad (np.ndarray): shape (M, 4, 3); analytic gradient of
                `values[m]` with respect to each of its 4 corner points.
    """
    p0 = point_batch[:, 0, :]
    p1 = point_batch[:, 1, :]
    p2 = point_batch[:, 2, :]
    p3 = point_batch[:, 3, :]
    u = p1 - p0
    v = p2 - p0
    w = p3 - p0

    vxw = np.cross(v, w, axis=-1)
    values = np.sum(u * vxw, axis=-1) / 6.0

    g1 = vxw / 6.0
    g2 = np.cross(w, u, axis=-1) / 6.0
    g3 = np.cross(u, v, axis=-1) / 6.0
    g0 = -(g1 + g2 + g3)

    slot_grad = np.stack([g0, g1, g2, g3], axis=1)  # (M, 4, 3)
    return values, slot_grad


def _scatter_group_jacobian(slot_grad, slot_idx, zone_local_to_freevar, row_offset):
    """
    Build dense-Jacobian COO (rows, cols, vals) triplets for one repair-zone
    element group's analytic constraint gradients, skipping any slot whose
    zone-local node is a fixed anchor (zone_local_to_freevar < 0). Repeated
    slots pointing at the same free node (e.g. a penta6 ridge point
    occupying slots 5-7) naturally produce repeated (row, col) triplets,
    which COO-matrix construction sums -- exactly the chain-rule sum
    required for a repeated variable.

    Inputs:
        slot_grad (np.ndarray): shape (M, S, 3); per-slot analytic gradient
            from _batched_hexlike_value_and_grad / _batched_tet_value_and_grad.
        slot_idx (np.ndarray): shape (M, S) int; zone-local node index
            occupying each slot.
        zone_local_to_freevar (np.ndarray): shape (K,) int; maps a
            zone-local node index to its free-variable block index, or -1
            if that node is a fixed anchor.
        row_offset (int): Constraint-row offset for this group within the
            full stacked constraint vector.

    Returns:
        tuple[np.ndarray]: (rows, cols, vals) 1-D arrays ready for
            scipy.sparse.coo_matrix, targeting a dense Jacobian of shape
            (n_constraints, 3 * n_free_vars).
    """
    m_group, n_slots, _ = slot_grad.shape
    freevar_of_slot = zone_local_to_freevar[slot_idx]           # (m_group, n_slots)
    valid = (freevar_of_slot >= 0).reshape(-1)

    m_idx = np.repeat(np.arange(m_group), n_slots)
    row_base = (row_offset + m_idx)[valid]
    col_base = (3 * freevar_of_slot.reshape(-1))[valid]
    vals_block = slot_grad.reshape(m_group * n_slots, 3)[valid]

    rows = np.repeat(row_base, 3)
    cols = (col_base[:, None] + np.arange(3)[None, :]).reshape(-1)
    vals = vals_block.reshape(-1)
    return rows, cols, vals


def compute_element_volumes(elements_df, nodes_df):
    """
    Compute the signed volume of each element in elements_df.

    Element type is detected from the number of unique real node IDs after
    stripping LS-DYNA duplicate-node placeholders (via _valid_element_nodes):
        4 unique nodes -> tetrahedron  (tet4)
        6 unique nodes -> pentahedron  (penta6)
        8 unique nodes -> hexahedron   (hex8)

    Node ordering convention (LS-DYNA, NID1..NID8):
        Tet4  : n1-n2-n3 base triangle (CCW from below), n4 apex.
        Penta6: n1-n2-n3-n4 quadrilateral base (CCW from below); n5 is the
                ridge point over the n1-n2 side, n6 is the ridge point over
                the n3-n4 side (n6 repeated in NID6-NID8 in the source card).
                Volume uses LS-DYNA's reduced (1-point) integration formula,
                not a planar geometric decomposition (see _penta_signed_volume).
        Hex8  : n1-n2-n3-n4 bottom face (CCW from below),
                n5-n6-n7-n8 top face; vertical edges n1-n5, n2-n6, n3-n7, n4-n8.
                Volume uses LS-DYNA's reduced (1-point) integration formula
                (see _hex_signed_volume), which correctly handles distorted/
                skewed hexahedra as well as planar-faced ones.

    Positive volume = valid (non-inverted) element.
    Negative volume = inverted element (volume < 0 in LS-DYNA).

    Inputs:
        elements_df (pd.DataFrame): Elements from read_elements_from_kfile;
            columns include 'EID', 'PID', 'NID1'..'NID8'.
        nodes_df (pd.DataFrame): Node coordinates from read_nodes_from_kfile;
            columns 'NodeID', 'x', 'y', 'z'.

    #TODO: VERIFY THIS WORKS

    PERFORMANCE: fully vectorized with numpy across all elements of each
    node-count group (4/6/8) at once, via _vectorized_element_corner_points
    -- NOT a per-element Python loop. See compute_element_distortion's
    PERFORMANCE note for the full story: the earlier per-element-loop
    version (even after a first fix that replaced pandas `.loc[]` lookups
    with a plain dict) was measured at roughly 60-70+ microseconds per
    element at real whole-model scale (2.4 million solid elements) --
    over 150 seconds PER CALL, and this function is called TWICE
    (before/after) on the FULL element table in the multi-part repair
    scripts. The volume formulas (_tet_signed_volume, _penta_signed_
    volume, _hex_signed_volume) are simple linear combinations of 3-D
    corner points followed by one cross/dot product -- entirely
    vectorizable across every element in a group simultaneously.

    Returns:
        pd.DataFrame: One row per element; columns ['EID', 'PID', 'n_nodes', 'volume'].
            'volume' is NaN for elements with missing node coordinates.

    Raises:
        ValueError: If an element has a number of unique real nodes other than
            4, 6, or 8 (e.g. an unsupported 5-node pyramid), so that unknown
            element types cannot silently pass through as NaN.
    """
    eid_arr, pid_arr, n_nodes_arr, groups = _vectorized_element_corner_points(elements_df, nodes_df)
    _raise_on_unsupported_node_count(eid_arr, n_nodes_arr)

    volumes = np.full(len(elements_df), np.nan, dtype=float)

    for count, (row_indices, pts, valid_mask) in groups.items():
        if count == 4:
            # Tet: base CCW (n1, n2, n3), apex n4 -- see _tet_signed_volume.
            p0, p1, p2, p3 = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
            vol = np.einsum('ij,ij->i', np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0
        elif count == 6:
            # Penta: reduced (1-point) integration Jacobian-at-centroid
            # volume, matching LS-DYNA's default solid formulation --
            # see _penta_signed_volume (vectorized here across all M
            # elements in this group at once instead of one at a time).
            p1, p2, p3, p4, p5, p6 = (pts[:, k] for k in range(6))
            a = p2 + p3 - p1 - p4 - p5 + p6
            b = p3 + p4 - p1 - p2 - p5 + p6
            c = p5 + 3.0 * p6 - p1 - p2 - p3 - p4
            vol = np.einsum('ij,ij->i', a, np.cross(b, c)) / 64.0
        else:  # count == 8
            # Hex: reduced (1-point) integration Jacobian-at-centroid
            # volume, matching LS-DYNA's default solid formulation -- see
            # _hex_signed_volume (vectorized across all M elements at once).
            p1, p2, p3, p4, p5, p6, p7, p8 = (pts[:, k] for k in range(8))
            a = -p1 + p2 + p3 - p4 - p5 + p6 + p7 - p8
            b = -p1 - p2 + p3 + p4 - p5 - p6 + p7 + p8
            c = -p1 - p2 - p3 - p4 + p5 + p6 + p7 + p8
            vol = np.einsum('ij,ij->i', a, np.cross(b, c)) / 64.0

        volumes[row_indices] = np.where(valid_mask, vol, np.nan)

    return pd.DataFrame({
        'EID': eid_arr.astype(int), 'PID': pid_arr.astype(int),
        'n_nodes': n_nodes_arr, 'volume': volumes,
    })


def classify_solid_badness(elements_df, nodes_df):
    """
    One-call convenience wrapper combining compute_element_volumes and
    compute_element_distortion into a single per-EID "how bad is this
    solid element" classification, TIERED by the user's explicit repair
    priority order (a negative/zero volume is worse than a merely
    distorted-but-positive-volume element):
        'bad_tier' == 2 : negative or zero volume (INVERTED) -- highest
                          repair priority.
        'bad_tier' == 1 : positive volume but locally tangled/bow-tied
                          (is_distorted -- see compute_element_distortion).
        'bad_tier' == 0 : neither -- a healthy element.
        'is_bad'        : True for tier 1 or 2 (the union used as a
                          repair TARGET list), False for tier 0.

    WHY THIS EXISTS: previously, the only way a solid element ever became
    a repair TARGET was by appearing in an externally-supplied DYNA
    quality-check report (solid_elements_failed_quality_check.k) -- a
    static snapshot generated once, outside this codebase, before any
    repair ever ran. compute_element_distortion's own is_distorted flag
    was computed and PRINTED for information, but never fed back into
    what actually gets repaired (per user request: "if you are detecting
    the distorted elements, can you just repair them with the list you
    create via calculations?"). This function is the single, reused,
    ONE-PLACE definition of "bad" (by both measures, correctly tiered)
    used everywhere this now matters: building the initial self-detected
    repair target list, and every whole-mesh regression check that runs
    after a repair phase (so "did this phase make anything worse" always
    means the SAME thing, checked the SAME way, regardless of which phase
    or call site is asking).

    Inputs:
        elements_df (pd.DataFrame): SOLID-only elements (columns include
            'EID', 'PID', 'NID1'..'NID8').
        nodes_df (pd.DataFrame): Node table (columns 'NodeID', 'x', 'y', 'z').

    Returns:
        pd.DataFrame: one row per element, columns 'EID', 'PID', 'volume',
            'is_distorted', 'is_bad', 'bad_tier'.
    """
    vol = compute_element_volumes(elements_df, nodes_df)
    dist = compute_element_distortion(elements_df, nodes_df)
    merged = vol[['EID', 'PID', 'volume']].merge(dist[['EID', 'is_distorted']], on='EID')
    merged['is_bad'] = (merged['volume'] <= 0) | merged['is_distorted']
    merged['bad_tier'] = np.where(
        merged['volume'] <= 0, 2, np.where(merged['is_distorted'], 1, 0)
    )
    return merged


def _vectorized_shell_corner_points(elements_df, nodes_df):
    """
    Shell-element analogue of _vectorized_element_corner_points: groups
    elements_df by unique real node count (3 = tri3, 4 = quad4) and
    gathers each group's corner coordinates into ONE numpy array per
    group via a single vectorized NodeID->row-position lookup, exactly
    the same pattern used for solids (see that function's docstring for
    the full performance rationale -- per-row iterrows()/`.loc[]` lookups
    are unusable at real whole-model scale, e.g. ~630,000 shells here).

    Inputs/Returns: identical shape/contract to
    _vectorized_element_corner_points, just grouped over {3, 4} instead
    of {4, 6, 8}.
    """
    n = len(elements_df)
    element_node_lists = _unique_node_lists_fast(elements_df)
    n_nodes_arr = np.fromiter((len(nl) for nl in element_node_lists), dtype=np.int64, count=n)
    eid_arr = elements_df['EID'].to_numpy()
    pid_arr = elements_df['PID'].to_numpy()

    node_ids_arr = nodes_df['NodeID'].to_numpy()
    coords_arr = nodes_df[['x', 'y', 'z']].to_numpy(dtype=float)
    id_to_idx = pd.Series(np.arange(len(node_ids_arr)), index=node_ids_arr)

    groups = {}
    for count in (3, 4):
        group_mask = (n_nodes_arr == count)
        row_indices = np.nonzero(group_mask)[0]
        if row_indices.size == 0:
            continue

        node_id_matrix = np.array(
            [element_node_lists[i] for i in row_indices], dtype=np.int64
        )
        flat_positions = id_to_idx.reindex(node_id_matrix.ravel())
        missing = flat_positions.isna().to_numpy().reshape(node_id_matrix.shape)
        valid_mask = ~missing.any(axis=1)
        positions = flat_positions.fillna(0).to_numpy().astype(np.int64).reshape(node_id_matrix.shape)
        pts = coords_arr[positions]  # (M, count, 3)

        groups[count] = (row_indices, pts, valid_mask)

    return eid_arr, pid_arr, n_nodes_arr, groups


def _raise_on_unsupported_shell_node_count(eid_arr, n_nodes_arr):
    """
    Shell analogue of _raise_on_unsupported_node_count: raises on the
    FIRST (in original elements_df row order) element whose unique real
    node count isn't 3 or 4.
    """
    unsupported_mask = ~np.isin(n_nodes_arr, [3, 4])
    if unsupported_mask.any():
        bad_idx = int(np.nonzero(unsupported_mask)[0][0])
        raise ValueError(
            f"Unsupported SHELL element node count for EID {int(eid_arr[bad_idx])}: "
            f"found {int(n_nodes_arr[bad_idx])} unique real nodes; supported counts are 3, 4."
        )


# Quad4 Jacobian-ratio calibration constants (see compute_shell_jacobian's
# docstring for the full derivation/validation story). LS-DYNA's exact
# internal quad4 Jacobian formula is proprietary and could not be
# reverse-engineered exactly (confirmed with the user's own coworker, who
# independently hit the same wall) -- these two constants are an
# EMPIRICAL LINEAR CALIBRATION fit against ALL 1,309 real quad4 elements
# in HBM_F/shell_report.k (a genuine LS-DYNA quality-check export on the
# actual production mesh, not a synthetic/idealized test case), NOT a
# theoretically derived value. Fit via ordinary least squares:
#   LSDYNA_value ~= QUAD4_CALIBRATION_SCALE * raw_ratio + QUAD4_CALIBRATION_OFFSET
# Achieves (against that same 1,309-element real dataset): mean abs error
# 0.026, median 0.020, max 0.151, 85.9% of elements within 0.05 of
# LS-DYNA's own reported value, Spearman rank correlation 0.72 (vs. only
# 0.22 for the more "obvious" jmin/jmax corner-ratio formula used for
# hex8/penta6 solids -- confirmed NOT to transfer well to quad4 shells).
# LIMITATION (accepted, discussed with user): shell_report.k only exports
# FAILING elements (LS-DYNA value range -0.058 to 0.400 in this dataset),
# so this calibration is only validated within that failing regime --
# exactly the regime this repair tool operates in (pushing bad elements
# INTO a valid range), but not verified against known-good elements
# outside that range (no such reference data was available).
QUAD4_CALIBRATION_SCALE = 0.5037
QUAD4_CALIBRATION_OFFSET = 0.1975


def compute_shell_jacobian(elements_df, nodes_df):
    """
    Compute LS-DYNA-style Jacobian ratio quality metrics for tri3/quad4
    shell elements (the shell analogue of compute_element_volumes /
    compute_element_distortion for solids).

    Element type is detected from the number of unique real node IDs:
        3 unique nodes -> triangle (tri3)
        4 unique nodes -> quadrilateral (quad4)

    *** TRIANGLE (tri3) FORMULA *** -- validated near-EXACT against real
    LS-DYNA output (mean absolute error ~0.0005 across every hand-checked
    real failing triangle from HBM_F/shell_report.k, i.e. within
    floating-point/display-rounding noise of the report's 3-decimal
    values). Adapted directly from lsdyna_helper_functions_NewFunctions.py
    (a colleague's independent implementation, itself apparently
    validated against real LS-DYNA output for this element type):
        j_surf   = |cross(p1-p0, p2-p0)|         (actual triangle "area
                    Jacobian", twice the true area)
        s_eq     = perimeter / 3                 (equivalent equilateral
                    triangle's side length, same perimeter as the actual
                    triangle)
        j_equil  = s_eq^2 * sqrt(3)/2             (that equilateral
                    triangle's own j_surf)
        jratio   = j_surf / j_equil               (1.0 for a perfect
                    equilateral triangle; smaller for a thin/degenerate one)
    A true tri3 is an AFFINE map (constant Jacobian everywhere across the
    element, unlike a bilinear quad4), so there is only ever one value per
    element -- jacobian_min == jacobian_max == j_surf here, matching the
    solid tet4 convention in compute_element_distortion.

    *** QUAD (quad4) FORMULA *** -- NOT an exact match to LS-DYNA's
    proprietary internal formula (confirmed unreverse-engineerable from
    outside; the user's coworker independently hit the same limitation).
    Multiple candidate formulas were tested against ALL 1,309 real quad4
    elements in HBM_F/shell_report.k (not just a handful of hand-picked
    examples) before settling on this one:
      - The "obvious" corner-Jacobian jmin/jmax ratio (same STYLE as the
        working hex8/penta6 solid formula in compute_element_distortion,
        and this same file's earlier fork) was tested FIRST and performed
        badly: mean abs error 0.29 against LS-DYNA's own values (a huge
        error relative to the failing range's own ~0.46-wide span),
        Spearman rank correlation only 0.22 (barely better than random
        ordering).
      - The winning approach instead follows the SAME PRINCIPLE that
        makes the tri3/tet4 formulas work: compare the element's WORST
        corner to what an IDEAL, same-size REFERENCE shape would give,
        rather than a self-referential jmin/jmax ratio. For a quad4, the
        reference shape is a perfect square with the same perimeter.
        Remarkably, that ideal square's own corner Jacobian value (by the
        SAME formula used on the real element) simplifies to an EXACT
        closed form: side**2, where side = perimeter/4 -- verified
        analytically (not just numerically) by evaluating the corner-
        Jacobian formula on a literal unit square, so no per-element
        "build a reference square and re-run the formula" step is needed
        at runtime.
      - This raw ratio (jmin / side**2) alone already reaches Spearman
        rank correlation 0.72 (vs. 0.22 for jmin/jmax) and Pearson
        correlation 0.78 -- a genuinely different, much better formula
        family, not just a tuning tweak of the old one.
      - A final linear calibration (QUAD4_CALIBRATION_SCALE/_OFFSET, see
        those constants' own docstring) was then fit via ordinary least
        squares against all 1,309 real LS-DYNA-reported values to correct
        the remaining systematic scale/offset gap, reaching 85.9% of
        elements within 0.05 of LS-DYNA's own number.
    Steps:
        n_ref     = normalize(cross(p2-p0, p3-p1))  (reference normal from
                    the quad's own two diagonals -- needed since a general
                    quad4 is not planar; this determines a consistent sign
                    convention for the corner-area-vector dot products
                    below, exactly as in the solid hex8 corner-Jacobian
                    formula's own reference-direction convention)
        corner_area_vecs = 4 corner cross-products (bilinear-map corner
                    Jacobians, same construction as the pre-existing
                    hex8-style quad corner formula)
        corner_jacs = dot(each corner_area_vec, n_ref)
        jmin      = min(corner_jacs)
        side      = perimeter / 4
        raw_ratio = jmin / side**2
        jratio    = QUAD4_CALIBRATION_SCALE * raw_ratio + QUAD4_CALIBRATION_OFFSET

    Limitations (documented, accepted -- see QUAD4_CALIBRATION_SCALE's own
    docstring for the full story):
      - The quad4 formula is a well-correlated APPROXIMATION of LS-DYNA's
        real metric, not an exact reproduction -- do not treat jacobian_
        ratio values from this function as literally identical to what
        LS-DYNA itself would report for a quad4, only as a reasonable,
        validated stand-in with the same sign/ordering behavior.
      - The quad4 calibration is only validated within the FAILING range
        (LS-DYNA values roughly -0.06 to 0.40 in the real dataset used);
        behavior well outside that range (e.g. a near-perfect quad
        expected to score close to 1.0) has not been checked against real
        LS-DYNA output, since shell_report.k only exports already-failing
        elements.
      - No warping/out-of-plane check is included (matches
        compute_element_distortion's own documented limitation for
        shells).

    PERFORMANCE: fully vectorized via _vectorized_shell_corner_points --
    NOT a per-row Python loop -- for the same reason as compute_element_
    volumes/compute_element_distortion (a real whole-model mesh has on
    the order of 600,000+ shell elements; a per-row loop at even a modest
    tens-of-microseconds-per-element cost would be tens of seconds PER
    CALL, and this function is expected to be called repeatedly across a
    repair escalation ladder).

    Inputs:
        elements_df (pd.DataFrame): Elements from read_elements_from_kfile
            (or read_all_elements_from_kfile / read_multi_part_mesh with
            element_type='ALL'); columns include 'EID', 'PID',
            'NID1'..'NID8'. Rows with any other element type mixed in
            (e.g. solids) should be filtered out by the caller first (see
            the 'ETYPE' column) -- this function assumes every row is a
            shell.
        nodes_df (pd.DataFrame): Node coordinates from read_nodes_from_kfile;
            columns 'NodeID', 'x', 'y', 'z'.

    Returns:
        pd.DataFrame: One row per element; columns ['EID', 'PID',
            'n_nodes', 'jacobian_min', 'jacobian_max', 'jacobian_ratio'].
            For tri3, jacobian_min == jacobian_max (affine map, single
            value). NaN for elements with missing node coordinates.

    Raises:
        ValueError: If an element has a number of unique real nodes other
            than 3 or 4 (e.g. a degenerate 2-node row), so an unsupported
            shell type cannot silently pass through as NaN.
    """
    eid_arr, pid_arr, n_nodes_arr, groups = _vectorized_shell_corner_points(elements_df, nodes_df)
    _raise_on_unsupported_shell_node_count(eid_arr, n_nodes_arr)

    jmin_arr = np.full(len(elements_df), np.nan)
    jmax_arr = np.full(len(elements_df), np.nan)
    jratio_arr = np.full(len(elements_df), np.nan)

    for count, (row_indices, pts, valid_mask) in groups.items():
        if count == 3:
            p0, p1, p2 = pts[:, 0], pts[:, 1], pts[:, 2]
            area_vec = np.cross(p1 - p0, p2 - p0)
            j_surf = np.linalg.norm(area_vec, axis=1)
            perimeter = (
                np.linalg.norm(p1 - p0, axis=1)
                + np.linalg.norm(p2 - p1, axis=1)
                + np.linalg.norm(p0 - p2, axis=1)
            )
            s_eq = perimeter / 3.0
            j_equil = (s_eq ** 2) * (np.sqrt(3.0) / 2.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                jratio = np.where(j_equil > 0.0, j_surf / j_equil, np.nan)
            jmin = jmax = j_surf
        else:  # count == 4
            p0, p1, p2, p3 = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
            n_ref = np.cross(p2 - p0, p3 - p1)
            n_ref_norm = np.linalg.norm(n_ref, axis=1, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                n_ref_hat = np.where(n_ref_norm > 1e-12, n_ref / n_ref_norm, 0.0)

            corner_area_vecs = [
                np.cross(p1 - p0, p3 - p0),
                np.cross(p1 - p0, p2 - p1),
                np.cross(p2 - p3, p2 - p1),
                np.cross(p2 - p3, p3 - p0),
            ]
            corner_jacs = np.stack(
                [np.einsum('ij,ij->i', cav, n_ref_hat) for cav in corner_area_vecs],
                axis=1,
            )  # (M, 4)
            jmin = corner_jacs.min(axis=1)
            jmax = corner_jacs.max(axis=1)

            perimeter = (
                np.linalg.norm(p1 - p0, axis=1)
                + np.linalg.norm(p2 - p1, axis=1)
                + np.linalg.norm(p3 - p2, axis=1)
                + np.linalg.norm(p0 - p3, axis=1)
            )
            side = perimeter / 4.0
            with np.errstate(divide='ignore', invalid='ignore'):
                raw_ratio = np.where(side > 0.0, jmin / (side ** 2), np.nan)
            jratio = QUAD4_CALIBRATION_SCALE * raw_ratio + QUAD4_CALIBRATION_OFFSET

        jmin_arr[row_indices] = np.where(valid_mask, jmin, np.nan)
        jmax_arr[row_indices] = np.where(valid_mask, jmax, np.nan)
        jratio_arr[row_indices] = np.where(valid_mask, jratio, np.nan)

    return pd.DataFrame({
        'EID': eid_arr.astype(int), 'PID': pid_arr.astype(int), 'n_nodes': n_nodes_arr,
        'jacobian_min': jmin_arr, 'jacobian_max': jmax_arr, 'jacobian_ratio': jratio_arr,
    })


def classify_shell_badness(elements_df, nodes_df, band_floor, band_ceiling):
    """
    One-call convenience wrapper around compute_shell_jacobian classifying
    every shell element's target-band status -- the shell analogue of
    classify_solid_badness. Shells are a single (lowest, per the user's
    explicit priority order: negative-volume solid > distorted solid >
    shell) tier: 'bad_tier' is 1 for any out-of-band shell (below
    band_floor OR above band_ceiling), 0 otherwise. Unlike solids there is
    no further internal ranking between "too thin" and "too flat" here.

    WHY THIS EXISTS: mirrors classify_solid_badness's own rationale --
    used both to SELF-DETECT out-of-band shells that a static, externally
    -supplied shell quality report (parsed once, before this script's own
    solid-repair phase ever moved a node) could never have known about
    (e.g. a shell collaterally pushed out of band by solid repair moving
    a node they share -- see build_solid_node_adjacency's own finding that
    the large majority of shell-referenced nodes also touch a solid), and
    to run whole-mesh "did this phase make anything worse" regression
    checks consistently against the SAME definition of "bad" everywhere.

    Inputs:
        elements_df (pd.DataFrame): SHELL-only elements.
        nodes_df (pd.DataFrame): Node table (columns 'NodeID', 'x', 'y', 'z').
        band_floor, band_ceiling (float): Target jacobian_ratio band.

    Returns:
        pd.DataFrame: one row per element, columns 'EID', 'PID',
            'jacobian_ratio', 'is_bad', 'bad_tier' (1 if is_bad else 0).
    """
    jac = compute_shell_jacobian(elements_df, nodes_df)
    out = jac[['EID', 'PID', 'jacobian_ratio']].copy()
    out['is_bad'] = (out['jacobian_ratio'] < band_floor) | (out['jacobian_ratio'] > band_ceiling)
    out['bad_tier'] = np.where(out['is_bad'], 1, 0)
    return out


def parse_shell_quality_report(report_path):
    """
    Parse an LS-DYNA shell-element quality-check TEXT report (e.g.
    shell_report.k -- a human-readable report, NOT a .k element/selection
    card) into a plain sorted list of failing EIDs, for use as
    repair_distorted_shells' elements_bad_shell_df seed list (after
    filtering elements_all_shell_df down to these EIDs).

    Real report format (confirmed against HBM_F/F05's own shell_report.k):
        SHELL ELEMENT QUALITY CHECK SUMMARY:
        ------------------------------------
        ...
        Failed Elements list:
        (Element_ID     Value     )
        1500691         0.325
        1538079         0.318
        ...
    Only the "Failed Elements list" section's "EID  value" lines are
    parsed -- everything above it (file name, quality-name/min/max/
    allowable/violated-count summary table) is ignored. This is a plain
    regex line scan (not read_elements_from_kfile, which expects LS-DYNA
    *KEYWORD card syntax -- this file has none), matching the leading-
    integer-then-decimal-value pattern LS-DYNA's report writer uses.

    Inputs:
        report_path (str): Path to the shell quality-check report file.

    Returns:
        list[int]: Sorted, de-duplicated EIDs from the report's failed-
            elements list. Empty list if the file has no such section
            (e.g. every shell passed).
    """
    with open(report_path, 'r') as f:
        text = f.read()

    marker = text.find('Failed Elements list')
    body = text[marker:] if marker >= 0 else text

    bad_eids = set()
    for m in re.finditer(r'^\s*(\d+)\s+-?\d+(?:\.\d+)?\s*$', body, re.MULTILINE):
        bad_eids.add(int(m.group(1)))
    return sorted(bad_eids)


def read_part_name_map(csv_path):
    """
    Read a morph_info.csv-style part-metadata file into a plain PID ->
    part_name dict, for callers wanting to select/exclude elements by
    PART NAME (e.g. filtering out '_Null' void-fill parts, see
    get_null_part_ids below) rather than by raw PID number alone.

    Expected columns (confirmed against the real HBM_F/F05 morph_info.csv):
    an unnamed row-index column, then 'part_name', 'part_id',
    'section_id', 'section_type', 'material_id', 'material_type',
    'material_name'. Only 'part_id'/'part_name' are used here -- the
    section/material columns are ignored (this codebase has no other use
    for them).

    *** ENCODING FALLBACK ***
    Some morph_info.csv files are Excel/Windows-originated and contain
    non-UTF-8 bytes (e.g. a non-breaking space, 0xA0, pasted into a
    'material_name' literature citation such as "...biomechanics 37.1
    (2004)..."), which raise UnicodeDecodeError under pandas' default
    UTF-8 read. To stay robust against this without silently mangling
    genuinely UTF-8 files, we first try UTF-8 and only fall back to
    'cp1252' (correctly handles Windows/Excel single-byte extended
    characters like NBSP, smart quotes, em-dashes) and finally 'latin-1'
    (ISO-8859-1, which can decode any byte 0-255 and therefore always
    succeeds) if cp1252 itself still fails for some reason.

    Inputs:
        csv_path (str): Path to the morph_info.csv-style file.

    Returns:
        dict[int, str]: part_id -> part_name. Empty dict if the file has
            no rows.
    """
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding='cp1252')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='latin-1')
    return dict(zip(df['part_id'].astype(int), df['part_name'].astype(str)))


def get_null_part_ids(part_name_map, suffix='_Null'):
    """
    Return the set of part_ids whose part_name ends with `suffix`
    (default '_Null') -- these are void-fill/filler parts, NOT real
    anatomical geometry (confirmed with the user: e.g. R_Trapezius_Null,
    L_Latissimus_Dorsi_Null in the real HBM_F/F05 mesh are gap-fill
    shells wrapping the trapezius/latissimus dorsi muscles, not the
    muscles themselves). See the NULL_OFF setting in compare_model_
    morphs_multipart_hpc_sparse.py for how this set is used to toggle
    these parts' inclusion/exclusion from an entire repair run.

    Inputs:
        part_name_map (dict[int, str]): From read_part_name_map.
        suffix (str): Case-sensitive suffix to match (default '_Null').

    Returns:
        set[int]: part_ids whose name ends with `suffix`. Empty set if
            none match.
    """
    return {pid for pid, name in part_name_map.items() if name.endswith(suffix)}


def get_bone_part_ids(part_name_map, keywords=('bone',)):
    """
    Return the set of part_ids whose part_name contains ANY of `keywords`
    as a case-insensitive substring (default: just 'bone') -- auto-
    detects bone parts directly from morph_info.csv's part-name column,
    the SAME "explicit, auditable, name-pattern" philosophy already used
    by get_null_part_ids (suffix match, just above) and classify_
    priority_tiers (keyword match, further below). Confirmed against the
    real HBM_F/F05 morph_info.csv naming convention: every bone part name
    observed there contains the literal substring 'bone' (e.g.
    'Maxilla_Cortical_Bone', 'Mandible_Trabecular_Bone', 'Skull_Cap_
    Cortical_Bone').

    IMPORTANT: this is used ONLY to identify bone for (1) the immovable
    "hard" reference surface in bone-only penetration detection/repair,
    and (2) tier-0 priority classification in the all-PID penetration
    phase -- it is explicitly NEVER used to exclude bone from solid/
    shell jacobian repair (per explicit user decision: bone is repaired
    exactly like every other part, just like everything else it stays
    in scope unless separately restricted, e.g. via TARGET_PIDS_ONLY in
    the driver script).

    Inputs:
        part_name_map (dict[int, str]): From read_part_name_map.
        keywords (tuple[str] | list[str]): Case-insensitive substrings;
            a part matches if its name contains ANY of them. Default
            just ('bone',) -- extend (e.g. add 'tooth') if a model's
            naming convention needs it, without touching this function.

    Returns:
        set[int]: part_ids whose name contains any keyword. Empty set if
            none match (e.g. part_name_map is empty or no name matched).
    """
    keywords_lower = [str(kw).lower() for kw in keywords]
    return {
        pid for pid, name in part_name_map.items()
        if any(kw in str(name).lower() for kw in keywords_lower)
    }


def compute_pid_reference_volumes(healthy_elements_df, nodes_df):
    """
    Compute each PID's MEDIAN element volume from a set of already-
    filtered "healthy" (positive-volume, non-distorted) elements -- the
    "what's a normal-sized element in this part" reference baseline used
    by flag_large_volume_outliers below. Callers should compute this from
    the mesh's ORIGINAL (pre-repair) state so a repair can never bias its
    own reference.

    Median (not mean) is used deliberately: a part can already contain a
    few naturally larger elements (e.g. a coarse transition region), and
    a mean would let those drag the whole reference up, masking a
    genuinely new, repair-caused outlier. Median is robust against that.

    Inputs:
        healthy_elements_df (pd.DataFrame): Pre-filtered SOLID elements
            considered healthy/representative (caller's choice of
            filter -- typically classify_solid_badness's ~is_bad rows on
            the ORIGINAL, pre-repair node table).
        nodes_df (pd.DataFrame): Node table matching healthy_elements_df
            (typically the ORIGINAL, pre-repair node table).

    Returns:
        dict[int, float]: PID -> median volume, for every PID with at
            least one healthy element with a valid (non-NaN, positive)
            volume. A PID entirely absent from healthy_elements_df (e.g.
            100% of its elements started bad) simply has no key --
            callers must treat a missing PID as "no reference available"
            rather than assuming 0/NaN means something.
    """
    if healthy_elements_df.empty:
        return {}
    vol_df = compute_element_volumes(healthy_elements_df, nodes_df)
    vol_df = vol_df[vol_df['volume'].notna() & (vol_df['volume'] > 0)]
    if vol_df.empty:
        return {}
    return vol_df.groupby('PID')['volume'].median().to_dict()


def flag_large_volume_outliers(elements_df, nodes_df, pid_reference_volumes, ratio_threshold,
                                original_nodes_df=None):
    """
    Report-only check: flag every element whose CURRENT volume exceeds
    ratio_threshold times its own PID's reference median volume (see
    compute_pid_reference_volumes) -- catches a technically-valid
    (positive volume, non-distorted) repair result that is nevertheless
    unreasonably larger than every OTHER element in the same part, e.g. a
    repair that satisfied the volume_epsilon > 0 constraint by growing
    one Hex8 far beyond its neighbors' normal size instead of a modest,
    locally-proportioned fix. Never blocks/changes anything -- purely a
    human-review flag (see ENABLE_VOLUME_SANITY_CHECK's own docstring in
    compare_model_morphs_multipart_hpc_sparse.py for how this is wired
    into the pipeline and written to a dedicated log/CSV).

    Only POSITIVE-volume elements are ever flagged here -- a still-
    negative volume is already caught (and prioritized far higher) by
    the normal classify_solid_badness 'bad_tier' == 2 check; this
    function is strictly about "valid but suspiciously oversized", never
    about validity itself.

    *** BEFORE/AFTER CONTEXT (optional, via original_nodes_df) ***
    Without original_nodes_df, this function can only report "this
    element is currently N times larger than a typical element in its
    PID" -- it CANNOT say whether this run's repair caused that, because
    only one node table (the current/final one) is ever looked at. This
    was a real source of confusion: an element gets "touched" (included
    in this check at all) merely by referencing >=1 node that moved even
    a tiny amount (see the 1e-9 displacement threshold at this function's
    call site) -- it does NOT mean the element itself changed size
    meaningfully, let alone that repair is responsible for its size.
    When original_nodes_df (the ORIGINAL, pre-repair node table) is
    supplied, this function additionally computes each flagged element's
    volume on THAT table and adds:
        'volume_before' : volume computed on original_nodes_df. NaN if
                           unavailable (e.g. a referenced node is somehow
                           missing from original_nodes_df -- should not
                           normally happen).
        'ratio_before'  : volume_before / pid_reference_volume (directly
                           comparable to 'ratio' -- lets a human see "was
                           this already ~40x before, or did it start
                           small and only just cross the line now").
        'growth_ratio'  : volume / volume_before (how much THIS RUN
                           changed the element's size, independent of
                           the reference-volume threshold entirely). NaN
                           if volume_before is NaN, zero, or negative (a
                           non-positive volume_before means the element
                           was itself invalid/inverted before repair, so
                           a single growth multiple isn't a meaningful
                           number -- see pre_existing_outlier instead).
        'pre_existing_outlier' : True if the element ALREADY exceeded
                           ratio_threshold on the ORIGINAL mesh -- i.e.
                           this run's repair did NOT create the
                           oversized condition, it simply touched a node
                           this element happens to reference. False if
                           volume_before was at/below the threshold
                           (repair genuinely grew it past the
                           threshold) OR if volume_before is unavailable/
                           non-positive (conservatively treated as "not
                           confirmed pre-existing" rather than silently
                           asserting it).

    Inputs:
        elements_df (pd.DataFrame): SOLID elements to check (typically
            just the elements actually TOUCHED by this run's repair --
            i.e. referencing at least one moved node -- on the FINAL,
            post-repair node table, so a pre-existing large element
            elsewhere in the same part that repair never touched is not
            repeatedly flagged every run).
        nodes_df (pd.DataFrame): Node table matching elements_df
            (typically the FINAL, post-repair node table).
        pid_reference_volumes (dict[int, float]): From
            compute_pid_reference_volumes, computed on the ORIGINAL
            (pre-repair) mesh.
        ratio_threshold (float): Flag if volume > ratio_threshold *
            pid_reference_volumes[PID].
        original_nodes_df (pd.DataFrame, optional): The ORIGINAL,
            pre-repair node table (same 'NodeID'/'x'/'y'/'z' schema as
            nodes_df). If given, adds the four before/after columns
            described above. If omitted (default), behavior is
            unchanged from before this parameter existed.

    Returns:
        pd.DataFrame: columns ['EID', 'PID', 'volume',
            'pid_reference_volume', 'ratio'], plus ['volume_before',
            'ratio_before', 'growth_ratio', 'pre_existing_outlier'] when
            original_nodes_df is given, one row per flagged element,
            sorted by 'ratio' descending (worst first). Empty (same
            columns) if none exceed the threshold, elements_df is empty,
            or no PID in elements_df has a reference volume available.
    """
    extra_cols = ['volume_before', 'ratio_before', 'growth_ratio', 'pre_existing_outlier']
    base_cols = ['EID', 'PID', 'volume', 'pid_reference_volume', 'ratio']
    empty_result = pd.DataFrame(columns=base_cols + extra_cols if original_nodes_df is not None else base_cols)
    if elements_df.empty:
        return empty_result

    vol_df = compute_element_volumes(elements_df, nodes_df)
    vol_df = vol_df[vol_df['volume'].notna() & (vol_df['volume'] > 0)].copy()
    if vol_df.empty:
        return empty_result

    vol_df['pid_reference_volume'] = vol_df['PID'].map(pid_reference_volumes)
    vol_df = vol_df[vol_df['pid_reference_volume'].notna() & (vol_df['pid_reference_volume'] > 0)]
    if vol_df.empty:
        return empty_result

    vol_df['ratio'] = vol_df['volume'] / vol_df['pid_reference_volume']
    outliers = vol_df.loc[vol_df['ratio'] > ratio_threshold, base_cols].copy()
    outliers = outliers.sort_values('ratio', ascending=False).reset_index(drop=True)

    if original_nodes_df is not None and not outliers.empty:
        outlier_elements_df = elements_df[elements_df['EID'].isin(outliers['EID'])]
        vol_before_df = compute_element_volumes(outlier_elements_df, original_nodes_df)
        vol_before_df = vol_before_df[['EID', 'volume']].rename(columns={'volume': 'volume_before'})
        outliers = outliers.merge(vol_before_df, on='EID', how='left')

        has_valid_before = outliers['volume_before'].notna() & (outliers['volume_before'] > 0)
        outliers['ratio_before'] = np.where(
            has_valid_before, outliers['volume_before'] / outliers['pid_reference_volume'], np.nan
        )
        outliers['growth_ratio'] = np.where(
            has_valid_before, outliers['volume'] / outliers['volume_before'], np.nan
        )
        # NaN-safe: comparing NaN > threshold is already False in numpy, so an
        # unavailable/non-positive volume_before conservatively yields False
        # (see docstring) rather than raising or requiring an extra .fillna().
        outliers['pre_existing_outlier'] = outliers['ratio_before'] > ratio_threshold
    elif original_nodes_df is not None:
        for col in extra_cols:
            outliers[col] = pd.Series(dtype=bool if col == 'pre_existing_outlier' else float)

    return outliers


def _batched_tri3_jacobian_ratio_value_and_grad(point_batch):
    """
    Vectorized value + analytic gradient of the tri3 jacobian_ratio metric
    (see compute_shell_jacobian's docstring for the formula's own
    derivation/validation against real LS-DYNA output -- this is the
    SLSQP/trust-constr-facing counterpart, providing the gradient needed
    by repair_distorted_shells, exactly as _batched_tet_value_and_grad is
    the gradient-providing counterpart to compute_element_volumes' tet4
    branch).

    A tri3 is an AFFINE map (no argmin/branching, unlike quad4's corner
    selection), so this needs no softmin smoothing variant -- the same
    exact value/gradient are used for both the SLSQP-facing constraint
    and the exact floor/final-check reporting, exactly like
    _batched_hexlike_centroid_value_and_grad's centroid term.

    DERIVATION (verified against central finite differences to ~1e-3
    relative error via a standalone check during development -- see
    _self_check_analytic_gradient's docstring for why a jitter point,
    not x0 itself, is used):
        Let a = p1-p0, b = p2-p0, C = cross(a,b) (twice the true area,
        i.e. compute_shell_jacobian's j_surf as a VECTOR, not yet reduced
        to its norm).
        j_surf = |C|. Using the standard scalar-triple-product identity
        (the SAME one _batched_tet_value_and_grad/_batched_hexlike_value_
        and_grad already rely on for solids' corner Jacobians):
            d(j_surf)/da = (b x C) / j_surf
            d(j_surf)/db = (C x a) / j_surf
        (chain rule then distributes these to p0/p1/p2 via a=p1-p0,
        b=p2-p0: d/dp1 = d/da, d/dp2 = d/db, d/dp0 = -(d/da + d/db)).

        perimeter = |p1-p0| + |p2-p1| + |p0-p2|; for a generic edge
        e = p_b - p_a, d|e|/dp_a = -e/|e|, d|e|/dp_b = e/|e| (elementary
        unit-vector gradient of a vector norm).
        s_eq = perimeter/3; j_equil = s_eq^2 * sqrt(3)/2 (this is fully
        determined by perimeter, i.e. by the CURRENT, not frozen, element
        shape -- matching the tri3/tet4 convention in
        lsdyna_helper_functions_NewFunctions.py of comparing against an
        ideal reference shape sized to the ACTUAL current element, not a
        stale baseline).

        jratio = j_surf / j_equil; quotient rule combines the two above:
            d(jratio)/dp_i = d(j_surf)/dp_i / j_equil
                              - jratio * d(j_equil)/dp_i / j_equil

    Inputs:
        point_batch (np.ndarray): shape (M, 3, 3); p0, p1, p2 (n1, n2, n3).

    Returns:
        tuple:
            values (np.ndarray): shape (M,); jacobian_ratio per element
                (matches compute_shell_jacobian's tri3 branch).
            slot_grad (np.ndarray): shape (M, 3, 3); analytic gradient of
                `values[m]` with respect to each of its 3 corner points.
    """
    p0 = point_batch[:, 0, :]
    p1 = point_batch[:, 1, :]
    p2 = point_batch[:, 2, :]

    a = p1 - p0
    b = p2 - p0
    C = np.cross(a, b, axis=-1)
    j_surf = np.linalg.norm(C, axis=-1)
    j_surf_safe = np.where(j_surf > 1e-12, j_surf, 1.0)

    grad_a_jsurf = np.cross(b, C, axis=-1) / j_surf_safe[:, None]
    grad_b_jsurf = np.cross(C, a, axis=-1) / j_surf_safe[:, None]
    grad_p1_jsurf = grad_a_jsurf
    grad_p2_jsurf = grad_b_jsurf
    grad_p0_jsurf = -(grad_a_jsurf + grad_b_jsurf)

    e1 = p1 - p0  # p0 -> p1
    e2 = p2 - p1  # p1 -> p2
    e3 = p0 - p2  # p2 -> p0
    len1 = np.linalg.norm(e1, axis=-1)
    len2 = np.linalg.norm(e2, axis=-1)
    len3 = np.linalg.norm(e3, axis=-1)
    len1_safe = np.where(len1 > 1e-12, len1, 1.0)
    len2_safe = np.where(len2 > 1e-12, len2, 1.0)
    len3_safe = np.where(len3 > 1e-12, len3, 1.0)
    u1 = e1 / len1_safe[:, None]
    u2 = e2 / len2_safe[:, None]
    u3 = e3 / len3_safe[:, None]

    perimeter = len1 + len2 + len3
    grad_p0_perim = -u1 + u3
    grad_p1_perim = u1 - u2
    grad_p2_perim = u2 - u3

    s_eq = perimeter / 3.0
    j_equil = (s_eq ** 2) * (np.sqrt(3.0) / 2.0)
    j_equil_safe = np.where(j_equil > 1e-12, j_equil, 1.0)
    # d(j_equil)/dp = (2*s_eq/3) * (sqrt(3)/2) * d(perimeter)/dp
    #               = (perimeter*sqrt(3)/9) * d(perimeter)/dp
    dj_equil_dperim_coeff = (perimeter * np.sqrt(3.0) / 9.0)
    grad_p0_jequil = dj_equil_dperim_coeff[:, None] * grad_p0_perim
    grad_p1_jequil = dj_equil_dperim_coeff[:, None] * grad_p1_perim
    grad_p2_jequil = dj_equil_dperim_coeff[:, None] * grad_p2_perim

    with np.errstate(divide='ignore', invalid='ignore'):
        jratio = np.where(j_equil > 1e-12, j_surf / j_equil_safe, np.nan)
    jratio_safe = np.where(np.isnan(jratio), 0.0, jratio)

    def _combine(grad_jsurf, grad_jequil):
        return grad_jsurf / j_equil_safe[:, None] - jratio_safe[:, None] * grad_jequil / j_equil_safe[:, None]

    grad_p0 = _combine(grad_p0_jsurf, grad_p0_jequil)
    grad_p1 = _combine(grad_p1_jsurf, grad_p1_jequil)
    grad_p2 = _combine(grad_p2_jsurf, grad_p2_jequil)

    slot_grad = np.stack([grad_p0, grad_p1, grad_p2], axis=1)  # (M, 3, 3)
    return jratio, slot_grad


def _frozen_quad_ref_normal(coords, group_idx):
    """
    Per-element FROZEN reference normal for the quad4 jacobian_ratio
    formula (see compute_shell_jacobian's docstring for the full formula
    and its validation story), calibrated ONCE from a baseline
    configuration (coords, typically the repair zone's ORIGINAL,
    pre-optimization geometry) and then held fixed for the entire
    optimization -- exactly the same "freeze a non-essential smoothing/
    convention knob at the baseline" pattern already used for
    _frozen_softmin_beta (hex8/penta6's softmin sharpness).

    WHY FREEZING IS CORRECT HERE (not just convenient): a general quad4
    is not planar, so there is no single, unambiguous "the" Jacobian sign
    convention the way there is for a true 3-D solid's determinant (see
    compute_shell_jacobian's own docstring) -- n_ref exists PURELY to give
    the four corner-area-vector dot products a CONSISTENT sign/scale
    convention, it is not itself a directly LS-DYNA-meaningful physical
    quantity (unlike, say, the perimeter/side reference in the SAME
    formula, which DOES need to track the CURRENT, not frozen, element
    size -- see the side/perimeter terms in
    _batched_quad_jacobian_ratio_value_and_grad, which are NOT frozen).
    Differentiating through normalize(cross(diagonal1, diagonal2))'s own
    dependence on the live coordinates would add substantial complexity
    for a term whose only purpose is sign/scale bookkeeping, not
    physical accuracy -- freezing it, like beta, avoids that complexity
    without changing what the constraint actually enforces in any way
    that matters (the corner values it selects from are still fully live/
    differentiated; only the projection DIRECTION is frozen).

    Inputs:
        coords (np.ndarray): shape (K, 3); the FULL zone-local coordinate
            array (baseline/original, not mid-optimization) -- same
            convention as _frozen_softmin_beta's `coords` parameter.
        group_idx (np.ndarray): shape (M, 4); zone-local indices of each
            quad4 element's 4 corner points, in the group's row order.

    Returns:
        np.ndarray: shape (M, 3); frozen unit reference normal per
            element. Zero vector (not NaN) for a degenerate element whose
            diagonals are parallel/zero-length, matching compute_shell_
            jacobian's own defensive zero-fallback for that edge case.
    """
    if len(group_idx) == 0:
        return np.zeros((0, 3))
    pts = coords[group_idx]  # (M, 4, 3)
    p0, p1, p2, p3 = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    n_ref = np.cross(p2 - p0, p3 - p1, axis=-1)
    n_ref_norm = np.linalg.norm(n_ref, axis=-1, keepdims=True)
    return np.where(n_ref_norm > 1e-12, n_ref / n_ref_norm, 0.0)


def _quad_corner_jacobians_value_and_grad(point_batch, n_ref):
    """
    Shared low-level helper for BOTH the hard (argmin) and soft (softmin)
    quad4 jacobian-ratio functions below: computes all 4 corners' raw
    (uncalibrated, un-normalized-by-side) corner-Jacobian VALUES and their
    FULL analytic gradients with respect to all 4 slot points, using a
    FROZEN per-element reference normal (see _frozen_quad_ref_normal).
    Factored out as one shared implementation (rather than each of the
    hard/soft callers re-deriving the same corner formula, as the
    hex8/penta6 code does for ITS hard/soft pair) specifically to avoid
    the hard and soft variants' corner formulas ever silently drifting
    apart from each other -- a real correctness risk this shared-helper
    design eliminates by construction.

    DERIVATION (verified against central finite differences during
    development): each corner's value is a scalar triple product
    jac_c = dot(cross(u_c, v_c), n_ref) for a fixed pair of edge vectors
    (u_c, v_c) per corner (see compute_shell_jacobian's corner_area_vecs
    for exactly which edge pairs), with n_ref held CONSTANT (see
    _frozen_quad_ref_normal). Using the cyclic scalar-triple-product
    identity dot(cross(u,v),n) = dot(u, cross(v,n)) = dot(v, cross(n,u)):
        d(jac_c)/du_c = cross(v_c, n_ref)
        d(jac_c)/dv_c = cross(n_ref, u_c)
    then distributed to whichever of p0..p3 each edge vector's two
    endpoints are (every edge vector is a simple difference of two of the
    four corner points, so this is elementary chain rule, no product rule
    needed beyond the two-argument cross product itself).

    Inputs:
        point_batch (np.ndarray): shape (M, 4, 3); p0..p3 (n1..n4).
        n_ref (np.ndarray): shape (M, 3); FROZEN reference normal per
            element (see _frozen_quad_ref_normal) -- must NOT be
            recomputed from point_batch's current (possibly mid-
            optimization) coordinates.

    Returns:
        tuple:
            corner_jacs (np.ndarray): shape (M, 4); this element's 4
                corner-Jacobian values, in the SAME order as compute_
                shell_jacobian's corner_area_vecs (corner1..corner4).
            corner_slot_grad (np.ndarray): shape (M, 4, 4, 3); analytic
                gradient of corner_jacs[:, c] with respect to each of the
                4 slot points (zero for a slot that corner doesn't touch).
    """
    p0 = point_batch[:, 0, :]
    p1 = point_batch[:, 1, :]
    p2 = point_batch[:, 2, :]
    p3 = point_batch[:, 3, :]

    u1, v1 = p1 - p0, p3 - p0
    u2, v2 = p1 - p0, p2 - p1
    u3, v3 = p2 - p3, p2 - p1
    u4, v4 = p2 - p3, p3 - p0

    jac1 = np.einsum('ij,ij->i', np.cross(u1, v1, axis=-1), n_ref)
    jac2 = np.einsum('ij,ij->i', np.cross(u2, v2, axis=-1), n_ref)
    jac3 = np.einsum('ij,ij->i', np.cross(u3, v3, axis=-1), n_ref)
    jac4 = np.einsum('ij,ij->i', np.cross(u4, v4, axis=-1), n_ref)
    corner_jacs = np.stack([jac1, jac2, jac3, jac4], axis=1)  # (M, 4)

    gu1, gv1 = np.cross(v1, n_ref, axis=-1), np.cross(n_ref, u1, axis=-1)
    gu2, gv2 = np.cross(v2, n_ref, axis=-1), np.cross(n_ref, u2, axis=-1)
    gu3, gv3 = np.cross(v3, n_ref, axis=-1), np.cross(n_ref, u3, axis=-1)
    gu4, gv4 = np.cross(v4, n_ref, axis=-1), np.cross(n_ref, u4, axis=-1)

    m = point_batch.shape[0]
    corner_slot_grad = np.zeros((m, 4, 4, 3))
    # Corner 1: u=p1-p0, v=p3-p0
    corner_slot_grad[:, 0, 1, :] += gu1
    corner_slot_grad[:, 0, 3, :] += gv1
    corner_slot_grad[:, 0, 0, :] += -(gu1 + gv1)
    # Corner 2: u=p1-p0, v=p2-p1
    corner_slot_grad[:, 1, 0, :] += -gu2
    corner_slot_grad[:, 1, 1, :] += gu2 - gv2
    corner_slot_grad[:, 1, 2, :] += gv2
    # Corner 3: u=p2-p3, v=p2-p1
    corner_slot_grad[:, 2, 2, :] += gu3 + gv3
    corner_slot_grad[:, 2, 3, :] += -gu3
    corner_slot_grad[:, 2, 1, :] += -gv3
    # Corner 4: u=p2-p3, v=p3-p0
    corner_slot_grad[:, 3, 2, :] += gu4
    corner_slot_grad[:, 3, 3, :] += gv4 - gu4
    corner_slot_grad[:, 3, 0, :] += -gv4

    return corner_jacs, corner_slot_grad


def _quad_side_squared_value_and_grad(point_batch):
    """
    Value + analytic gradient of the quad4 formula's "ideal reference
    side, squared" term (side = perimeter/4; see compute_shell_jacobian's
    docstring for why this equals a perfect square's own corner-Jacobian
    value exactly) -- ALWAYS computed from the CURRENT (live,
    mid-optimization) point_batch, never frozen, since a real quality
    metric must track the actual current element size at every iteration
    (matching the tri3/tet4 formulas' own ideal-reference convention; see
    _batched_tri3_jacobian_ratio_value_and_grad's docstring for the
    parallel tri3 derivation, and _frozen_quad_ref_normal's docstring for
    why n_ref, unlike this term, IS safe to freeze).

    Inputs:
        point_batch (np.ndarray): shape (M, 4, 3); p0..p3.

    Returns:
        tuple:
            side_sq (np.ndarray): shape (M,); side^2 where side =
                perimeter/4.
            slot_grad (np.ndarray): shape (M, 4, 3); analytic gradient of
                side_sq with respect to each of the 4 corner points.
    """
    p0 = point_batch[:, 0, :]
    p1 = point_batch[:, 1, :]
    p2 = point_batch[:, 2, :]
    p3 = point_batch[:, 3, :]

    e1 = p1 - p0
    e2 = p2 - p1
    e3 = p3 - p2
    e4 = p0 - p3
    len1 = np.linalg.norm(e1, axis=-1)
    len2 = np.linalg.norm(e2, axis=-1)
    len3 = np.linalg.norm(e3, axis=-1)
    len4 = np.linalg.norm(e4, axis=-1)
    len1_safe = np.where(len1 > 1e-12, len1, 1.0)
    len2_safe = np.where(len2 > 1e-12, len2, 1.0)
    len3_safe = np.where(len3 > 1e-12, len3, 1.0)
    len4_safe = np.where(len4 > 1e-12, len4, 1.0)
    u1 = e1 / len1_safe[:, None]
    u2 = e2 / len2_safe[:, None]
    u3 = e3 / len3_safe[:, None]
    u4 = e4 / len4_safe[:, None]

    perimeter = len1 + len2 + len3 + len4
    side = perimeter / 4.0
    side_sq = side ** 2

    grad_p0_perim = -u1 + u4
    grad_p1_perim = u1 - u2
    grad_p2_perim = u2 - u3
    grad_p3_perim = u3 - u4

    # side_sq = (perimeter/4)^2 -> d(side_sq)/dp = (perimeter/8) * d(perimeter)/dp
    coeff = (perimeter / 8.0)[:, None]
    slot_grad = np.stack([
        coeff * grad_p0_perim,
        coeff * grad_p1_perim,
        coeff * grad_p2_perim,
        coeff * grad_p3_perim,
    ], axis=1)  # (M, 4, 3)
    return side_sq, slot_grad


def _combine_quad_ratio(corner_val, corner_grad, point_batch):
    """
    Shared final assembly step for BOTH the hard and soft quad4
    jacobian-ratio functions: combines an already-selected (hard argmin
    OR soft softmin) per-element corner value/gradient with the
    perimeter-based ideal-side normalization and the empirical LS-DYNA
    calibration (see QUAD4_CALIBRATION_SCALE/_OFFSET's own docstring),
    via the standard quotient rule:
        raw_ratio      = corner_val / side_sq
        raw_ratio_grad = corner_grad/side_sq - corner_val*side_sq_grad/side_sq^2
        jratio         = SCALE * raw_ratio + OFFSET      (affine, so its
                         gradient is simply SCALE * raw_ratio_grad)

    Inputs:
        corner_val (np.ndarray): shape (M,); hard-min or soft-min corner
            Jacobian value per element.
        corner_grad (np.ndarray): shape (M, 4, 3); its analytic gradient.
        point_batch (np.ndarray): shape (M, 4, 3); p0..p3 (needed to
            compute the LIVE, not frozen, side_sq term).

    Returns:
        tuple: (values (M,), slot_grad (M, 4, 3)) -- the full, calibrated
            quad4 jacobian_ratio and its analytic gradient.
    """
    side_sq, side_sq_grad = _quad_side_squared_value_and_grad(point_batch)
    side_sq_safe = np.where(side_sq > 1e-12, side_sq, 1.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        raw_ratio = np.where(side_sq > 1e-12, corner_val / side_sq_safe, np.nan)
    raw_ratio_safe = np.where(np.isnan(raw_ratio), 0.0, raw_ratio)

    raw_ratio_grad = (
        corner_grad / side_sq_safe[:, None, None]
        - raw_ratio_safe[:, None, None] * side_sq_grad / side_sq_safe[:, None, None]
    )

    values = QUAD4_CALIBRATION_SCALE * raw_ratio + QUAD4_CALIBRATION_OFFSET
    slot_grad = QUAD4_CALIBRATION_SCALE * raw_ratio_grad
    return values, slot_grad


def _batched_quad_jacobian_ratio_value_and_grad(point_batch, n_ref):
    """
    Vectorized value + analytic gradient of the FULL, calibrated quad4
    jacobian_ratio metric (see compute_shell_jacobian's docstring for the
    formula/validation story) using the EXACT (hard) corner-wise minimum
    -- the quad4 analogue of _batched_hexlike_value_and_grad, used for the
    non-regression floor baseline and the final exact pass/fail check
    (NOT for the SLSQP-facing constraint itself -- see the softmin variant
    below for that, for the same non-smooth-argmin-kink reason documented
    in _batched_hexlike_softmin_value_and_grad's docstring).

    Inputs:
        point_batch (np.ndarray): shape (M, 4, 3); p0..p3 (n1..n4).
        n_ref (np.ndarray): shape (M, 3); FROZEN reference normal (see
            _frozen_quad_ref_normal) -- NOT recomputed from point_batch.

    Returns:
        tuple:
            values (np.ndarray): shape (M,); calibrated jacobian_ratio
                using the hard corner-wise minimum (matches compute_
                shell_jacobian's quad4 branch when point_batch is the
                un-moved baseline geometry).
            slot_grad (np.ndarray): shape (M, 4, 3); analytic gradient of
                `values[m]` with respect to each of its 4 corner points.
    """
    corner_jacs, corner_slot_grad = _quad_corner_jacobians_value_and_grad(point_batch, n_ref)
    m = point_batch.shape[0]
    argmin_corner = np.argmin(corner_jacs, axis=1)  # (M,)
    row_idx = np.arange(m)
    jmin = corner_jacs[row_idx, argmin_corner]
    jmin_grad = corner_slot_grad[row_idx, argmin_corner]  # (M, 4, 3)
    return _combine_quad_ratio(jmin, jmin_grad, point_batch)


def _frozen_quad_softmin_beta(coords, group_idx, n_ref, k=_SOFTMIN_SHARPNESS):
    """
    Per-element softmin sharpness for the quad4 corner-Jacobian selection
    -- the quad4 analogue of _frozen_softmin_beta, same rationale and
    same "calibrate once from the baseline, freeze for the whole
    optimization" requirement (recomputing beta from mid-optimization
    coordinates would silently corrupt the analytic gradient -- see that
    function's docstring for the full explanation, which applies
    identically here).

    Inputs:
        coords (np.ndarray): shape (K, 3); FULL zone-local baseline
            coordinate array.
        group_idx (np.ndarray): shape (M, 4); zone-local indices of each
            quad4 element's 4 corner points.
        n_ref (np.ndarray): shape (M, 3); FROZEN reference normal (see
            _frozen_quad_ref_normal), computed from this SAME baseline.
        k (float): Sharpness constant, default _SOFTMIN_SHARPNESS (shared
            with the hex8/penta6 solid softmin for consistency).

    Returns:
        np.ndarray: shape (M,); frozen per-element softmin beta.
    """
    if len(group_idx) == 0:
        return np.zeros(0)
    pts = coords[group_idx]  # (M, 4, 3)
    corner_jacs, _ = _quad_corner_jacobians_value_and_grad(pts, n_ref)
    scale = np.maximum(np.ptp(corner_jacs, axis=1), 1e-9)
    return k / scale


def _batched_quad_jacobian_ratio_softmin_value_and_grad(point_batch, n_ref, beta):
    """
    Smooth (soft-min) stand-in for _batched_quad_jacobian_ratio_value_and_
    grad's hard corner-wise minimum, used ONLY as the SLSQP/trust-constr-
    facing constraint during optimization -- the quad4 analogue of
    _batched_hexlike_softmin_value_and_grad; same log-sum-exp softmin
    construction and same rationale (the hard argmin's non-smooth kink
    causes oscillation instead of convergence -- see that function's
    docstring for the full explanation).

    Inputs:
        point_batch (np.ndarray): shape (M, 4, 3); p0..p3.
        n_ref (np.ndarray): shape (M, 3); FROZEN reference normal (see
            _frozen_quad_ref_normal).
        beta (np.ndarray): shape (M,); FROZEN per-element softmin
            sharpness (see _frozen_quad_softmin_beta).

    Returns:
        tuple: (values (M,), slot_grad (M, 4, 3)) -- same contract as
            _batched_quad_jacobian_ratio_value_and_grad, but using the
            smooth softmin blend instead of the hard corner-wise minimum.
    """
    corner_jacs, corner_slot_grad = _quad_corner_jacobians_value_and_grad(point_batch, n_ref)
    v_min = corner_jacs.min(axis=1, keepdims=True)
    beta_col = beta[:, None]
    shifted = corner_jacs - v_min  # (M, 4), >= 0
    unnorm_w = np.exp(-beta_col * shifted)
    z_sum = unnorm_w.sum(axis=1, keepdims=True)
    softmin_val = (v_min - np.log(z_sum) / beta_col)[:, 0]
    weights = unnorm_w / z_sum  # (M, 4), rows sum to 1

    softmin_grad = np.einsum('mc,mcsj->msj', weights, corner_slot_grad)  # (M, 4, 3)
    return _combine_quad_ratio(softmin_val, softmin_grad, point_batch)


def _solid_face_node_lists(nodes, row=None):
    """
    Return topology faces (node lists) for supported solid elements.
    """
    if len(nodes) == 4:
        return [
            [nodes[0], nodes[1], nodes[2]],
            [nodes[0], nodes[1], nodes[3]],
            [nodes[0], nodes[2], nodes[3]],
            [nodes[1], nodes[2], nodes[3]],
        ]
    if len(nodes) == 6:
        return [
            [nodes[0], nodes[1], nodes[4]],
            [nodes[2], nodes[3], nodes[5]],
            [nodes[0], nodes[1], nodes[2], nodes[3]],
            [nodes[1], nodes[2], nodes[5], nodes[4]],
            [nodes[3], nodes[0], nodes[4], nodes[5]],
        ]
    if len(nodes) == 8:
        return [
            [nodes[0], nodes[1], nodes[2], nodes[3]],
            [nodes[4], nodes[5], nodes[6], nodes[7]],
            [nodes[0], nodes[1], nodes[5], nodes[4]],
            [nodes[1], nodes[2], nodes[6], nodes[5]],
            [nodes[2], nodes[3], nodes[7], nodes[6]],
            [nodes[3], nodes[0], nodes[4], nodes[7]],
        ]
    _warn_unsupported_element('solid', _element_id_for_warning(row), len(nodes), [4, 6, 8])
    return []


def _collect_boundary_nodes(elements_df, element_node_lists=None):
    """
    Return node IDs that lie on the exterior boundary of a solid mesh block.

    Inputs:
        elements_df (pd.DataFrame): Elements from read_elements_from_kfile.
        element_node_lists (list[list[int]], optional): Precomputed unique
            node IDs per row (e.g. from _unique_node_lists_fast), reused to
            avoid a second full pass over elements_df when the caller
            already has this from earlier preprocessing.
    """
    if element_node_lists is None:
        element_node_lists = _unique_node_lists_fast(elements_df)

    face_counts = {}
    for nodes in element_node_lists:
        for face_nodes in _solid_face_node_lists(nodes):
            face_key = _sorted_key(face_nodes)
            face_counts[face_key] = face_counts.get(face_key, 0) + 1

    boundary_nodes = set()
    for face_key, count in face_counts.items():
        if count == 1:
            boundary_nodes.update(face_key)

    return boundary_nodes


def build_solid_node_adjacency(solid_elements_df):
    """
    One-time precompute mapping every node ID to the set of SOLID element
    row-positions (into solid_elements_df, matching _unique_node_lists_
    fast's row order) that reference it -- lets repair_distorted_shells
    quickly find which solid elements are TOUCHED (share at least one
    node) by a shell repair pocket's free/constrained node zone, so those
    solid elements can be added to the pocket's constrained zone with a
    non-regression floor on their volume (see find_touching_solid_
    elements and repair_distorted_shells for how this is used).

    WHY THIS EXISTS (confirmed as a REAL, not hypothetical, concern on the
    actual HBM_F mesh): 78.7% of all shell-referenced nodes ALSO touch a
    solid element, and 95.7% of the specific nodes touched by this mesh's
    433 bad shell elements touch a solid element too -- i.e. moving a
    shell repair pocket's free nodes will, in the overwhelming majority
    of real cases, also move an underlying solid element's face (e.g. a
    tri3 skin shell directly overlaying a tet4 muscle solid, or a quad4
    over a hex8). Since solids are fully repaired BEFORE shells begin
    (confirmed workflow), an UNPROTECTED shell repair could silently
    re-invert/re-break an already-fixed solid element -- exactly the same
    "non-regression floor for untouched neighbours" principle solid
    repair already applies to ITS OWN neighbouring solid elements (see
    _build_floor_and_bad_mask), just extended ACROSS element types here.

    Built ONCE, before the shell-repair phase begins (analogous to how
    bone_node_to_pids is precomputed once in the driver script, and
    critically UNLIKE repair_inverted_elements' own internal nid_to_elem_
    indices, which is rebuilt from scratch on every single call) -- since
    solid_elements_df is the FULL ~2.4-million-element solid table,
    rebuilding this per shell pocket / per escalation attempt would be
    needlessly repeated work across however many shell pockets this mesh
    splits into.

    Inputs:
        solid_elements_df (pd.DataFrame): The FULL solid-only elements
            table (e.g. elements_df filtered to ETYPE == 'SOLID', or
            whatever table repair_inverted_elements itself was run
            against) -- NOT a pocket-restricted subset.

    Returns:
        tuple:
            solid_element_node_lists (list[list[int]]): One entry per row
                of solid_elements_df (same order), each that row's unique
                real node IDs -- same contract as _unique_node_lists_fast,
                returned here so callers don't need to recompute it
                separately.
            node_to_solid_indices (dict[int, set[int]]): NodeID -> set of
                row-positions (indices into solid_element_node_lists /
                solid_elements_df) of every solid element referencing
                that node.
    """
    solid_element_node_lists = _unique_node_lists_fast(solid_elements_df)
    node_to_solid_indices = {}
    for idx, nids in enumerate(solid_element_node_lists):
        for nid in nids:
            node_to_solid_indices.setdefault(nid, set()).add(idx)
    return solid_element_node_lists, node_to_solid_indices


def find_touching_solid_elements(node_ids, node_to_solid_indices):
    """
    Given a set of node IDs (e.g. a shell repair pocket's free-or-
    constrained zone nodes), return the row-positions of every solid
    element that references at least one of them, per the precomputed
    mapping from build_solid_node_adjacency.

    Inputs:
        node_ids (Iterable[int]): Node IDs to check.
        node_to_solid_indices (dict[int, set[int]]): From
            build_solid_node_adjacency.

    Returns:
        set[int]: Row-positions (into the same solid_elements_df/
            solid_element_node_lists build_solid_node_adjacency was
            called on) of every touched solid element. Empty set if none
            of node_ids touch any solid element.
    """
    touched = set()
    for nid in node_ids:
        touched.update(node_to_solid_indices.get(nid, set()))
    return touched


# ============================== CROSS-PART PENETRATION DETECTION ===========
# Detects a SOFT-tissue node (e.g. muscle) that has ended up on the wrong
# ("inside") side of a HARD/protected part's (e.g. bone) outer surface --
# a defect distinct from element inversion/distortion (which is purely
# per-element geometry): this is a CROSS-part contact violation, checked
# with the same "never move the hard/protected geometry" rule already
# established by BONE_PIDS (see the driver script). DETECTION ONLY for
# now (see build_hard_surface_index/detect_soft_hard_penetrations'
# docstrings) -- no function in this section moves any node; they only
# report which soft nodes/elements violate the hard surface, for the
# driver to print/export. Shared by BOTH the .k and .feb pipelines,
# exactly like every other function in this file.
#
# WHY A SURFACE-BASED LOCAL PLANE TEST, NOT A FULL WATERTIGHT INSIDE/
# OUTSIDE TEST (ray-casting, winding number): those require a genuinely
# closed, consistently-wound, non-self-intersecting surface to be
# reliable, and real anatomical FE meshes are not guaranteed to be that
# clean everywhere (small gaps, T-junctions, etc. are common). The
# approach here instead extracts each hard part's own boundary faces with
# a genuine outward normal, then classifies any nearby soft node using
# only the LOCAL plane of its single nearest face -- this degrades
# gracefully (simply not flagging anything, rather than silently giving a
# wrong global answer) wherever the hard surface is imperfect elsewhere.
# See build_hard_surface_index's own docstring for exactly how each face's
# outward direction is determined (differs for SOLID vs SHELL hard parts).
#
# WHY NOT A BARE POINT CLOUD: a point cloud alone has no orientation --
# there is no way to tell "inside" from "outside" purely from points, only
# from an oriented SURFACE (points + normals).
def _face_normal_and_centroid_batch(face_pts):
    """
    Vectorized face centroid + normal for a batch of homogeneous-length
    (all-triangle OR all-quadrilateral) faces -- the shared low-level
    geometry primitive for both the SOLID (per-face-flip) and SHELL
    (global-volume-sign-flip) branches of build_hard_surface_index.

    Inputs:
        face_pts (np.ndarray, shape (M, 3, 3) or (M, 4, 3)): Corner
            coordinates per face, M faces at once, ALL the same node
            count (3 or 4) -- callers group faces by length first (see
            build_hard_surface_index) rather than padding a mixed-length
            batch, avoiding any ambiguity about which "slot" is real vs
            padding.

    Returns:
        tuple:
            centroids (np.ndarray, shape (M, 3)): Mean of the face's
                corners.
            unit_normals (np.ndarray, shape (M, 3)): Unit-length normal,
                NOT yet outward-oriented (caller applies that -- this
                function only computes the raw undirected line of
                action). NaN row for a degenerate/zero-area face.
            raw_normals (np.ndarray, shape (M, 3)): The SAME normal
                direction, but NOT unit-normalized -- its magnitude is
                proportional to the face's own area (exactly 2x the
                triangle area for a tri3, via the standard cross-product
                area-vector identity). Needed (not just the unit version)
                for an AREA-WEIGHTED sum, e.g. the closed-surface signed-
                volume sign check in build_hard_surface_index's SHELL
                branch -- using the unit normal there would incorrectly
                weight every face equally regardless of its true size.
    """
    n_per_face = face_pts.shape[1]
    p0, p1, p2 = face_pts[:, 0], face_pts[:, 1], face_pts[:, 2]
    if n_per_face == 3:
        centroids = (p0 + p1 + p2) / 3.0
        raw_normals = np.cross(p1 - p0, p2 - p0)
    elif n_per_face == 4:
        p3 = face_pts[:, 3]
        centroids = (p0 + p1 + p2 + p3) / 4.0
        # Diagonal method: robust for slightly non-planar quads (common
        # on a curved anatomical surface), unlike picking one arbitrary
        # triangulation.
        raw_normals = np.cross(p2 - p0, p3 - p1)
    else:
        raise ValueError(f"_face_normal_and_centroid_batch only supports 3- or 4-node faces, got {n_per_face}.")

    norms = np.linalg.norm(raw_normals, axis=1, keepdims=True)
    zero_mask = (norms.ravel() == 0)
    safe_norms = np.where(norms == 0, 1.0, norms)
    unit_normals = raw_normals / safe_norms
    unit_normals[zero_mask] = np.nan

    return centroids, unit_normals, raw_normals


def _extract_boundary_faces_for_solid_pid(pid_solid_elements_df):
    """
    Return ONE solid PID's own exterior boundary faces -- faces that
    appear in exactly ONE of this part's own elements (its outer skin,
    INCLUDING wherever it borders a DIFFERENT PID or open space -- both
    look identical from inside this PID's own element set alone, which is
    exactly what penetration detection needs: a muscle resting against a
    bone's outer surface should see that surface, regardless of whether
    anything else happens to be modeled beyond it).

    Reuses the SAME face-topology/uniqueness-counting logic as
    _collect_boundary_nodes (via _solid_face_node_lists/_sorted_key,
    already proven at full multi-million-element mesh scale for the
    boundary-growth pre-bias feature), but returns the actual FACE node-
    ID tuples (in ORIGINAL, not sorted, order -- needed for a real cross-
    product normal) plus which element row owns each boundary face
    (needed to flip that face's normal outward via build_hard_surface_
    index's per-face centroid-flip method) -- _collect_boundary_nodes
    only returns the flat set of boundary NODE IDs, which is not enough
    information for either of those.

    Inputs:
        pid_solid_elements_df (pd.DataFrame): SOLID elements of ONE PID
            only (e.g. elements_df filtered to PID == some hard bone PID
            and ETYPE == 'SOLID').

    Returns:
        tuple:
            boundary_face_node_ids (list[tuple[int, ...]]): One entry per
                boundary face (3- or 4-node tuple, original winding).
            owning_row_idx (list[int]): Row-position (into
                pid_solid_elements_df) of the element that owns each
                boundary face, same order/length as boundary_face_node_ids.
    """
    element_node_lists = _unique_node_lists_fast(pid_solid_elements_df)

    face_counts = {}
    face_owner = {}
    face_original = {}
    for row_idx, nodes in enumerate(element_node_lists):
        for face_nodes in _solid_face_node_lists(nodes):
            face_key = _sorted_key(face_nodes)
            face_counts[face_key] = face_counts.get(face_key, 0) + 1
            face_owner[face_key] = row_idx
            face_original[face_key] = face_nodes

    boundary_face_node_ids = []
    owning_row_idx = []
    for face_key, count in face_counts.items():
        if count == 1:
            boundary_face_node_ids.append(tuple(face_original[face_key]))
            owning_row_idx.append(face_owner[face_key])

    return boundary_face_node_ids, owning_row_idx


def build_hard_surface_index(elements_df, nodes_df, hard_pids,
                              _precomputed_id_to_idx=None, _precomputed_coords_arr=None):
    """
    Build a searchable, OUTWARD-ORIENTED surface representation of every
    'hard' (never-move, e.g. bone) part, for penetration detection (see
    detect_soft_hard_penetrations). Built ONCE per run from hard_pids'
    CURRENT node positions (normally the post-jacobian-repair table),
    then queried repeatedly -- hard parts are NEVER moved by this
    feature, matching the user's explicit requirement that soft tissue
    is what gets pushed out, never bone.

    ORIENTATION METHOD (differs by element type, since a SOLID part has
    real volume to flip a face normal away from, but a SHELL part -- e.g.
    a cortical bone shell wrapping a trabecular bone solid interior, a
    common real construction in this codebase's HBM meshes -- does not):
      - SOLID hard PIDs: each boundary face's raw cross-product normal is
        flipped, if necessary, to point AWAY from its OWNING element's
        own centroid. This is a robust, PER-FACE decision that is always
        correct for any non-degenerate element, regardless of the part's
        overall shape (including long/non-convex bones like a femur
        shaft, where a single whole-part centroid would get the sign
        wrong in many places -- using each face's OWN owning element's
        local centroid avoids that entirely).
      - SHELL hard PIDs: shells have no owning volume element to flip
        away from, so this uses ONE GLOBAL flip decision per PID
        instead: the standard closed-surface signed-volume formula
        (divergence theorem, area-weighted via each face's RAW/non-unit
        normal -- see _face_normal_and_centroid_batch) is summed across
        all of this PID's own faces (relative to the part's own local
        centroid, not the global coordinate origin, to reduce numerical
        bias from any small non-closure); if the result is negative,
        EVERY face normal for that PID is flipped. This assumes the
        part's own shell winding is already LOCALLY self-consistent
        (adjacent faces agree with each other even if the whole part
        happens to be wound inside-out) -- true for a properly modeled
        closed anatomical shell even if not perfectly watertight. This
        is NOT a full per-edge winding-consistency repair; a scattered/
        inconsistent mesh would need that separately (out of scope for
        this detect-only rollout).

    Inputs:
        elements_df (pd.DataFrame): FULL element table (SOLID + SHELL).
        nodes_df (pd.DataFrame): CURRENT node coordinates (NodeID, x, y, z).
        hard_pids (set[int] | Iterable[int]): PIDs to treat as hard/
            protected (e.g. BONE_PIDS from the driver script).
        _precomputed_id_to_idx, _precomputed_coords_arr: Internal perf
            hook -- see _vectorized_element_corner_points' own docstring
            for the exact contract (both-or-neither). Lets build_all_
            part_surfaces (which calls this function ONCE PER PART at
            ~700-part whole-body scale) skip rebuilding an O(total mesh
            nodes) index on every single call -- a real, confirmed
            production bottleneck. None (default) preserves the EXACT
            original behavior for every existing caller.

    Returns:
        dict | None: None if hard_pids is empty, or no boundary faces
            could be extracted from it (e.g. every listed PID is missing
            from elements_df) -- callers (detect_soft_hard_penetrations)
            must handle this explicitly rather than assume a non-empty
            index. Otherwise:
                'kdtree' (scipy.spatial.cKDTree): built over face
                    centroids, one entry per row of the arrays below.
                'face_centroids' (np.ndarray, shape (F, 3))
                'face_normals' (np.ndarray, shape (F, 3)): unit outward
                    normals, aligned with face_centroids.
                'face_pid' (np.ndarray, shape (F,)): which hard PID each
                    face belongs to.
                'hard_pids' (set[int]): the input, normalized to a set.
    """
    from scipy.spatial import cKDTree

    hard_pids = set(int(p) for p in hard_pids)
    if not hard_pids:
        return None

    if _precomputed_id_to_idx is not None or _precomputed_coords_arr is not None:
        if _precomputed_id_to_idx is None or _precomputed_coords_arr is None:
            raise ValueError(
                "_precomputed_id_to_idx and _precomputed_coords_arr must both be "
                "provided together, or both left as None."
            )
        id_to_idx = _precomputed_id_to_idx
        coords_arr = _precomputed_coords_arr
    else:
        node_ids_arr = nodes_df['NodeID'].to_numpy()
        coords_arr = nodes_df[['x', 'y', 'z']].to_numpy(dtype=float)
        id_to_idx = pd.Series(np.arange(len(node_ids_arr)), index=node_ids_arr)

    def _gather_face_group(face_node_tuples):
        """Homogeneous-length (all-3-node or all-4-node) face group ->
        (centroids, unit_normals, raw_normals, row_valid), skipping (via
        row_valid) any face referencing a node ID missing from nodes_df."""
        arr = np.array(face_node_tuples, dtype=np.int64)
        flat_pos = id_to_idx.reindex(arr.ravel())
        missing = flat_pos.isna().to_numpy().reshape(arr.shape)
        row_valid = ~missing.any(axis=1)
        positions = flat_pos.fillna(0).to_numpy().astype(np.int64).reshape(arr.shape)
        pts = coords_arr[positions]
        centroids, unit_normals, raw_normals = _face_normal_and_centroid_batch(pts)
        row_valid = row_valid & ~np.isnan(unit_normals).any(axis=1)
        return centroids, unit_normals, raw_normals, row_valid

    all_centroids, all_normals, all_pid = [], [], []

    for pid in sorted(hard_pids):
        pid_df = elements_df[elements_df['PID'] == pid]
        if pid_df.empty:
            continue
        solid_df = pid_df[pid_df['ETYPE'] == 'SOLID']
        shell_df = pid_df[pid_df['ETYPE'] == 'SHELL']

        # ---- SOLID: per-face flip away from owning element's centroid ----
        if not solid_df.empty:
            boundary_faces, owning_idx = _extract_boundary_faces_for_solid_pid(solid_df)
            if boundary_faces:
                _, _, _, groups = _vectorized_element_corner_points(
                    solid_df, nodes_df,
                    _precomputed_id_to_idx=id_to_idx, _precomputed_coords_arr=coords_arr,
                )
                elem_centroids = np.full((len(solid_df), 3), np.nan)
                for count, (row_indices, pts, valid_mask) in groups.items():
                    grp_centroids = pts.mean(axis=1)
                    elem_centroids[row_indices[valid_mask]] = grp_centroids[valid_mask]

                owning_idx_arr = np.array(owning_idx)
                faces_by_len = {}
                for i, face in enumerate(boundary_faces):
                    faces_by_len.setdefault(len(face), []).append(i)

                for flen, idxs in faces_by_len.items():
                    idxs_arr = np.array(idxs)
                    faces_this_len = [boundary_faces[i] for i in idxs]
                    centroids, unit_normals, _, row_valid = _gather_face_group(faces_this_len)
                    owners_this_len = owning_idx_arr[idxs_arr]
                    owner_c = elem_centroids[owners_this_len]
                    ok = row_valid & ~np.isnan(owner_c).any(axis=1)
                    if not ok.any():
                        continue
                    flip = np.einsum('ij,ij->i', unit_normals[ok], centroids[ok] - owner_c[ok]) < 0
                    normals_ok = unit_normals[ok].copy()
                    normals_ok[flip] *= -1.0
                    all_centroids.append(centroids[ok])
                    all_normals.append(normals_ok)
                    all_pid.append(np.full(int(ok.sum()), pid))

        # ---- SHELL: one global flip decision from the closed-volume sign ----
        if not shell_df.empty:
            shell_node_lists = _unique_node_lists_fast(shell_df)
            faces_by_len = {}
            for nl in shell_node_lists:
                faces_by_len.setdefault(len(nl), []).append(tuple(nl))

            pid_centroid_parts, pid_normal_parts, pid_raw_parts = [], [], []
            for flen, faces_this_len in faces_by_len.items():
                if flen not in (3, 4):
                    # Unsupported shell node count (should not occur for
                    # tri3/quad4) -- skip rather than crash a detect-only
                    # pass; this PID's surface is simply incomplete.
                    continue
                centroids, unit_normals, raw_normals, row_valid = _gather_face_group(faces_this_len)
                pid_centroid_parts.append(centroids[row_valid])
                pid_normal_parts.append(unit_normals[row_valid])
                pid_raw_parts.append(raw_normals[row_valid])

            if pid_centroid_parts:
                pid_centroids = np.concatenate(pid_centroid_parts, axis=0)
                pid_normals = np.concatenate(pid_normal_parts, axis=0)
                pid_raw_normals = np.concatenate(pid_raw_parts, axis=0)

                # Area-weighted (via RAW, non-unit normals -- see
                # _face_normal_and_centroid_batch's docstring) closed-
                # surface signed-volume SIGN check, relative to this
                # part's own local centroid (reduces bias from any small
                # non-closure vs. using the global coordinate origin).
                ref_point = pid_centroids.mean(axis=0)
                shifted = pid_centroids - ref_point
                total_signed_vol = np.einsum('ij,ij->i', shifted, pid_raw_normals).sum()
                if total_signed_vol < 0:
                    pid_normals = -pid_normals

                all_centroids.append(pid_centroids)
                all_normals.append(pid_normals)
                all_pid.append(np.full(len(pid_centroids), pid))

    if not all_centroids:
        return None

    face_centroids = np.concatenate(all_centroids, axis=0)
    face_normals = np.concatenate(all_normals, axis=0)
    face_pid = np.concatenate(all_pid, axis=0)

    return {
        'kdtree': cKDTree(face_centroids),
        'face_centroids': face_centroids,
        'face_normals': face_normals,
        'face_pid': face_pid,
        'hard_pids': hard_pids,
    }


def detect_soft_hard_penetrations(elements_df, nodes_df, hard_surface_index,
                                   tolerance_mm=0.0, search_radius_mm=10.0,
                                   exclude_soft_pids=None, exclude_pid_pairs=None):
    """
    Detect SOFT-tissue nodes that have penetrated INTO a 'hard' (e.g.
    bone) part's outer surface, using the pre-built hard_surface_index
    from build_hard_surface_index.

    METHOD: for every node referenced by a NON-hard-PID ("soft") element,
    EXCLUDING any node that is ALSO referenced by a hard-PID element (an
    intentional weld/attachment point -- e.g. a tendon node welded
    directly onto a bone node -- NOT a penetration defect, per the user's
    explicit requirement), find the nearest hard-surface FACE CENTROID
    via hard_surface_index['kdtree'], then compute that node's SIGNED
    perpendicular distance to the INFINITE PLANE containing that face:
    dot(node - face_centroid, face_unit_normal). A negative signed
    distance means the node sits on the far/inside side of that LOCAL
    surface patch -- i.e. penetrating.

    A node is only ever flagged if BOTH:
      - the nearest-centroid distance itself is within search_radius_mm
        (otherwise the nearest hard face is too far away for its local
        plane to be a meaningful inside/outside proxy at that location --
        the node just isn't near any hard part at all), AND
      - the computed penetration depth exceeds tolerance_mm (a small
        dead-zone to avoid flagging harmless floating-point noise right
        at a true, intentional contact surface).

    TWO KINDS OF DELIBERATE EXCLUSION (both opt-in, both applied BEFORE
    the geometric check -- an excluded node/PID never even gets a
    nearest-face query, so it can never appear in either returned
    DataFrame, not merely be filtered out afterward):
      - exclude_soft_pids: entire soft PIDs to skip outright -- e.g.
        void-fill/filler geometry (same concept as NULL_OFF's '_Null'
        shells) that is not real anatomy and would otherwise show up as
        large, meaningless "penetrations" purely from occupying the same
        space as bone by construction.
      - exclude_pid_pairs: specific (soft_PID, hard_PID) pairs where
        overlap is GENUINELY EXPECTED anatomy, not a defect -- e.g. tooth
        roots embedded in jaw bone, or the spinal cord passing through
        the skull's foramen magnum (a real anatomical opening that this
        function's per-part boundary-face extraction has no way to know
        is intentionally open, since it only sees "this PID's own
        boundary faces", not which openings are deliberate). A soft node
        is excluded ONLY when ITS specific soft PID is paired with the
        SPECIFIC hard PID its nearest face belongs to -- the same soft
        node is still checked normally against every OTHER hard PID.

    Inputs:
        elements_df (pd.DataFrame): FULL element table (SOLID + SHELL).
        nodes_df (pd.DataFrame): CURRENT node coordinates (NodeID, x, y, z).
        hard_surface_index (dict | None): From build_hard_surface_index.
            If None (e.g. hard_pids was empty), both returned DataFrames
            are empty (correct schema, zero rows) -- callers should check
            for this and print a clear "nothing to check" message rather
            than silently doing nothing.
        tolerance_mm (float): Minimum penetration depth (mm) before a
            node is flagged. Default 0.0 (flag any amount, however small).
        search_radius_mm (float): Maximum nearest-hard-face-centroid
            distance (mm) for a soft node to be checked at all. Default
            10.0 -- tune to roughly your model's local element size scale.
        exclude_soft_pids (set[int] | None): Soft PIDs to skip entirely
            (e.g. void-fill parts). None/empty = no exclusion.
        exclude_pid_pairs (set[tuple[int, int]] | None): (soft_PID,
            hard_PID) pairs representing genuinely-expected anatomical
            overlap -- see docstring above. None/empty = no exclusion.

    Returns:
        tuple:
            node_level_df (pd.DataFrame): One row per PENETRATING soft
                node, sorted by depth_mm descending (worst first).
                Columns: 'NodeID', 'hard_PID' (nearest offending part),
                'depth_mm' (positive = how far inside),
                'nearest_face_distance_mm'.
            element_level_df (pd.DataFrame): One row per SOFT element
                that owns at least one penetrating node. Columns: 'EID',
                'PID', 'ETYPE', 'n_penetrating_nodes', 'max_depth_mm'.
            Both are empty (correct schema, zero rows) if nothing was
            found to be penetrating, or if hard_surface_index is None.
    """
    empty_node_df = pd.DataFrame(columns=['NodeID', 'hard_PID', 'depth_mm', 'nearest_face_distance_mm'])
    empty_elem_df = pd.DataFrame(columns=['EID', 'PID', 'ETYPE', 'n_penetrating_nodes', 'max_depth_mm'])

    if hard_surface_index is None:
        return empty_node_df, empty_elem_df

    exclude_soft_pids = set(int(p) for p in (exclude_soft_pids or ()))
    exclude_pid_pairs = set((int(s), int(h)) for s, h in (exclude_pid_pairs or ()))

    hard_pids = hard_surface_index['hard_pids']
    hard_df = elements_df[elements_df['PID'].isin(hard_pids)]
    soft_df = elements_df[~elements_df['PID'].isin(hard_pids)]
    if exclude_soft_pids:
        soft_df = soft_df[~soft_df['PID'].isin(exclude_soft_pids)]
    if soft_df.empty:
        return empty_node_df, empty_elem_df

    hard_node_ids = _flat_referenced_node_ids_fast(hard_df)
    soft_node_ids_all = _flat_referenced_node_ids_fast(soft_df)
    # Exclude welded/shared attachment nodes -- see docstring.
    candidate_node_ids = np.array(sorted(soft_node_ids_all - hard_node_ids), dtype=np.int64)
    if candidate_node_ids.size == 0:
        return empty_node_df, empty_elem_df

    node_ids_arr = nodes_df['NodeID'].to_numpy()
    coords_arr = nodes_df[['x', 'y', 'z']].to_numpy(dtype=float)
    id_to_idx = pd.Series(np.arange(len(node_ids_arr)), index=node_ids_arr)
    pos = id_to_idx.reindex(candidate_node_ids)
    valid = ~pos.isna().to_numpy()
    candidate_node_ids = candidate_node_ids[valid]
    candidate_coords = coords_arr[pos.to_numpy()[valid].astype(int)]
    if candidate_node_ids.size == 0:
        return empty_node_df, empty_elem_df

    kdtree = hard_surface_index['kdtree']
    dist, face_idx = kdtree.query(candidate_coords, k=1)

    face_centroids = hard_surface_index['face_centroids'][face_idx]
    face_normals = hard_surface_index['face_normals'][face_idx]
    face_pid = hard_surface_index['face_pid'][face_idx]

    signed_dist = np.einsum('ij,ij->i', candidate_coords - face_centroids, face_normals)
    depth = -signed_dist  # positive = penetrating (on the inside of the local plane)

    within_radius = dist <= search_radius_mm
    penetrating = within_radius & (depth > tolerance_mm)

    if exclude_pid_pairs and penetrating.any():
        # Per-candidate-node soft PID, looked up ONCE via the same
        # NodeID->soft-PID membership already available from soft_df's own
        # node lists (a node may belong to >1 soft PID in rare cases; ANY
        # membership matching an excluded pair is enough to exclude that
        # node against THAT specific hard PID -- see docstring).
        soft_node_lists_for_pid = _unique_node_lists_fast(soft_df)
        node_to_soft_pids = {}
        for nl, pid in zip(soft_node_lists_for_pid, soft_df['PID'].tolist()):
            for nid in nl:
                node_to_soft_pids.setdefault(nid, set()).add(int(pid))
        excluded_mask = np.zeros(len(candidate_node_ids), dtype=bool)
        for i in np.nonzero(penetrating)[0]:
            nid = int(candidate_node_ids[i])
            hpid = int(face_pid[i])
            node_soft_pids = node_to_soft_pids.get(nid, set())
            if any((sp, hpid) in exclude_pid_pairs for sp in node_soft_pids):
                excluded_mask[i] = True
        penetrating = penetrating & ~excluded_mask

    if not penetrating.any():
        return empty_node_df, empty_elem_df

    node_level_df = pd.DataFrame({
        'NodeID': candidate_node_ids[penetrating],
        'hard_PID': face_pid[penetrating],
        'depth_mm': depth[penetrating],
        'nearest_face_distance_mm': dist[penetrating],
    }).sort_values('depth_mm', ascending=False).reset_index(drop=True)

    penetrating_node_set = set(node_level_df['NodeID'].tolist())
    depth_by_node = dict(zip(node_level_df['NodeID'], node_level_df['depth_mm']))

    # Same "one pass, build per-row hits via a plain dict/set" pattern as
    # build_solid_node_adjacency, already proven at full multi-million-
    # element scale elsewhere in this file -- NOT a per-row pandas
    # `.iloc`/iterrows() lookup (see eid_arr/pid_arr/etype_arr below,
    # plain numpy arrays indexed positionally instead).
    soft_node_lists = _unique_node_lists_fast(soft_df)
    eid_arr = soft_df['EID'].to_numpy()
    pid_arr = soft_df['PID'].to_numpy()
    etype_arr = soft_df['ETYPE'].to_numpy()

    elem_rows = []
    for i, nl in enumerate(soft_node_lists):
        hits = [n for n in nl if n in penetrating_node_set]
        if hits:
            elem_rows.append({
                'EID': int(eid_arr[i]),
                'PID': int(pid_arr[i]),
                'ETYPE': etype_arr[i],
                'n_penetrating_nodes': len(hits),
                'max_depth_mm': max(depth_by_node[n] for n in hits),
            })

    element_level_df = pd.DataFrame(elem_rows) if elem_rows else empty_elem_df

    return node_level_df, element_level_df


# ============================== ALL-PID-PAIR PENETRATION (generalization) ===
# Generalizes detect_soft_hard_penetrations/build_hard_surface_index from
# "soft tissue vs. one fixed BONE_PIDS set" to "ANY part vs. ANY other
# part" -- per explicit user requirement: "we want NO PENETRATION between
# any PID, full stop." detect_soft_hard_penetrations/build_hard_surface_
# index/repair_soft_hard_penetrations above are left COMPLETELY UNCHANGED
# (still used as-is for the simpler bone-only case) -- these are NEW,
# ADDITIVE functions built for the general case, reusing the same proven
# building blocks (build_hard_surface_index itself already accepts an
# arbitrary PID set, so it is called here once PER INDIVIDUAL PART rather
# than once for a fixed bone set).
#
# WHO MOVES WHEN TWO NON-BONE PARTS OVERLAP (explicit user decision): a
# PRIORITY TIER ranking (see classify_priority_tiers) -- the higher-
# priority (stiffer) part's surface is treated as the FIXED reference,
# exactly like bone was before; the lower-priority part's nodes are what
# move. For two parts in the SAME tier (no inherent stiffness difference
# either way), a deterministic tiebreak (lower PID number is fixed) is
# used instead of a genuine two-way simultaneous-moving-surface solve --
# a deliberate engineering simplification (confirmed acceptable: user
# said "I just need the general shape of things to be preserved, even if
# you need to move more than 1 element/node") that reuses the EXACT same
# validated one-side-fixed repair_soft_hard_penetrations engine, avoiding
# an unvalidated new two-way solver.
def classify_priority_tiers(part_name_map, bone_pids, tier_keyword_map=None, default_tier=3):
    """
    Classify every PID in part_name_map into a numeric PRIORITY TIER (0 =
    highest priority/stiffest, never yields; higher numbers = lower
    priority/softer, yields first) via case-insensitive keyword matching
    against each part's name -- the SAME "explicit, auditable, name-
    pattern" philosophy already used for BONE_PIDS/get_null_part_ids, not
    a hidden heuristic. bone_pids is authoritative for tier 0 regardless
    of name (it is the user's own existing, already-reviewed list).

    DEFAULT 5-TIER SCHEME (confirmed with the user against the real F05
    part-name inventory: 641 / 703 parts matched by name, remainder
    default into tier 3 -- spine endplates, meniscus, ankle joints,
    tectorial membrane, a couple of named anatomical "connections"):
        0: Bone (from bone_pids, authoritative -- not name-matched here)
        1: Cartilage, Disc, Annulus, Meniscus
        2: Tendon, Ligament, Capsule, CNS (spinal cord/dura)
        3: Muscle, Organ, VS (vascular structures) -- also the DEFAULT
           for any unmatched part name
        4: Fascia, Flesh, Skin

    A part matching keywords from MULTIPLE tiers uses the LOWEST
    (highest-priority) matching tier number -- e.g. a hypothetical
    "Muscle_Tendon_Junction" would land in tier 2 (Tendon), not tier 3.

    Inputs:
        part_name_map (dict[int, str]): From read_part_name_map.
        bone_pids (set[int]): Authoritative tier-0 PIDs (e.g. BONE_PIDS).
        tier_keyword_map (dict[int, list[str]] | None): Override the
            default keyword-per-tier scheme above. Tier 0 (bone) is
            ALWAYS driven by bone_pids, never by keywords, regardless of
            this parameter -- do not include a tier-0 entry here.
        default_tier (int): Tier assigned to any part matching no
            keyword and not in bone_pids. Default 3 (Muscle/Organ tier
            -- confirmed with the user as the right default for the real
            F05 model's ~62 unmatched names).

    Returns:
        tuple:
            pid_to_tier (dict[int, int]): Every PID in part_name_map,
                mapped to its tier.
            unclassified_pids (set[int]): PIDs that matched no keyword
                and were assigned default_tier purely by fallback -- for
                the caller to optionally review/report.
    """
    if tier_keyword_map is None:
        tier_keyword_map = {
            1: ['cartilage', 'disc', 'annulus', 'meniscus'],
            2: ['tendon', 'ligament', 'capsule', 'cns'],
            4: ['fascia', 'flesh', 'skin'],
            # tier 3 (muscle/organ/vs) is the default_tier fallback, not
            # listed here, so it never "wins" over a more specific match
            # purely due to keyword-list ORDER -- see the loop below.
        }

    bone_pids = set(int(p) for p in bone_pids)
    pid_to_tier = {}
    unclassified_pids = set()

    for pid, name in part_name_map.items():
        pid = int(pid)
        if pid in bone_pids:
            pid_to_tier[pid] = 0
            continue
        name_lower = str(name).lower()
        matched_tiers = [
            tier for tier, keywords in tier_keyword_map.items()
            if any(kw in name_lower for kw in keywords)
        ]
        if matched_tiers:
            pid_to_tier[pid] = min(matched_tiers)
        else:
            pid_to_tier[pid] = default_tier
            unclassified_pids.add(pid)

    return pid_to_tier, unclassified_pids


def _group_elements_by_pid(elements_df):
    """
    Group elements_df by PID ONCE (a single pandas groupby pass) and
    return a plain dict[int, pd.DataFrame] -- shared by build_all_part_
    surfaces/find_overlapping_pid_pairs/detect_all_pid_penetrations so
    NONE of them ever re-filters the FULL element table per individual
    PID (a REAL, confirmed production bottleneck: with ~700 real parts
    and ~3 million elements, doing elements_df[elements_df['PID']==pid]
    once per PID costs O(700 x 3,000,000) row-scans in total -- this
    single groupby instead costs O(3,000,000) ONCE, then each PID's own
    slice is a cheap dict lookup).

    Returns:
        dict[int, pd.DataFrame]: PID -> that PID's own element rows.
            A PID with zero elements simply has no key (callers must
            use .get(pid) and treat a missing key as "no elements").
    """
    return {int(pid): sub_df for pid, sub_df in elements_df.groupby('PID')}


def build_all_part_surfaces(elements_df, nodes_df, pids, _pid_groups=None):
    """
    Build a PER-PART oriented boundary-surface index for EVERY pid in
    `pids`, by calling build_hard_surface_index ONCE PER INDIVIDUAL PID
    (reusing its exact, already-validated per-face SOLID/global-flip
    SHELL orientation logic completely unchanged -- see that function's
    own docstring) and caching the results in a dict. This is the
    all-pairs generalization's equivalent of calling build_hard_surface_
    index once with the whole BONE_PIDS set -- here it is called many
    times, once per part, so any TWO parts' surfaces can be looked up
    independently later (needed because which part is "the fixed
    reference surface" varies per PID-pair, per classify_priority_tiers).

    *** PERFORMANCE (real, confirmed bottleneck at ~700-part whole-body
    scale) *** -- pre-groups elements_df by PID ONCE (_group_elements_by_
    pid) and pre-builds the NodeID->row-position index ONCE, then passes
    both through to every one of the ~700 build_hard_surface_index calls
    via that function's own _precomputed_id_to_idx/_precomputed_coords_
    arr hook -- WITHOUT this, each of those ~700 calls independently (a)
    re-filters the full multi-million-row elements_df for just its own
    PID, AND (b) rebuilds an O(total mesh nodes) index from scratch,
    which measured as a genuine 15+ minute stall on a real ~1.4M-node,
    ~3M-element, ~700-part mesh before this fix.

    Inputs:
        elements_df (pd.DataFrame): FULL element table (SOLID + SHELL).
        nodes_df (pd.DataFrame): CURRENT node coordinates.
        pids (Iterable[int]): Every PID to build a surface for.
        _pid_groups (dict[int, pd.DataFrame] | None): Internal perf hook
            -- if the caller already has _group_elements_by_pid's own
            output (e.g. the driver script calling this alongside find_
            overlapping_pid_pairs/detect_all_pid_penetrations in the
            same pass), pass it here to skip regrouping. None (default)
            groups internally.

    Returns:
        dict[int, dict]: PID -> build_hard_surface_index's own return
            shape for that single PID (or simply ABSENT from this dict
            if that PID had no usable boundary faces, e.g. zero elements
            -- callers must use .get(pid) and handle None/missing).
    """
    pids = sorted(set(int(p) for p in pids))
    pid_groups = _pid_groups if _pid_groups is not None else _group_elements_by_pid(elements_df)

    node_ids_arr = nodes_df['NodeID'].to_numpy()
    coords_arr = nodes_df[['x', 'y', 'z']].to_numpy(dtype=float)
    id_to_idx = pd.Series(np.arange(len(node_ids_arr)), index=node_ids_arr)

    empty_df = elements_df.iloc[0:0]
    surfaces = {}
    for pid in pids:
        pid_df = pid_groups.get(pid, empty_df)
        if pid_df.empty:
            continue
        idx = build_hard_surface_index(
            pid_df, nodes_df, {pid},
            _precomputed_id_to_idx=id_to_idx, _precomputed_coords_arr=coords_arr,
        )
        if idx is not None:
            surfaces[pid] = idx
    return surfaces


def find_overlapping_pid_pairs(elements_df, nodes_df, pids, padding_mm=0.0, _pid_groups=None):
    """
    Cheap BROAD-PHASE filter: axis-aligned bounding box (AABB) overlap
    test across every pair of PIDs in `pids`, so the expensive per-face
    nearest-surface check (build_all_part_surfaces + the actual
    penetration query) is only ever run for PID pairs that could
    plausibly touch at all -- with hundreds of real anatomical parts,
    the overwhelming majority of pairs (e.g. a skull part vs. a foot
    part) have AABBs nowhere near each other, so this prunes the
    O(n_parts^2) candidate space down to a small, genuinely-adjacent
    subset before any real geometry work happens.

    Inputs:
        elements_df (pd.DataFrame): FULL element table (SOLID + SHELL).
        nodes_df (pd.DataFrame): CURRENT node coordinates.
        pids (Iterable[int]): Every PID to consider.
        padding_mm (float): Expands each PID's own AABB by this much
            before testing overlap -- should be >= the search_radius_mm
            the caller will later use for the real penetration check, so
            a genuinely-close-but-not-quite-overlapping pair is never
            pruned away here and then missed entirely downstream.
        _pid_groups (dict[int, pd.DataFrame] | None): Internal perf hook
            -- see build_all_part_surfaces' own docstring for the exact
            contract. None (default) groups internally.

    Returns:
        list[tuple[int, int]]: (pid_a, pid_b) with pid_a < pid_b, one
            entry per overlapping pair. Order is not otherwise
            meaningful. Empty list if pids has fewer than 2 usable
            entries.
    """
    pids = sorted(set(int(p) for p in pids))
    if len(pids) < 2:
        return []

    pid_groups = _pid_groups if _pid_groups is not None else _group_elements_by_pid(elements_df)
    node_ids_arr = nodes_df['NodeID'].to_numpy()
    coords_arr = nodes_df[['x', 'y', 'z']].to_numpy(dtype=float)
    id_to_idx = pd.Series(np.arange(len(node_ids_arr)), index=node_ids_arr)

    n = len(pids)
    mins = np.full((n, 3), np.nan)
    maxs = np.full((n, 3), np.nan)
    for i, pid in enumerate(pids):
        pid_df = pid_groups.get(pid)
        if pid_df is None or pid_df.empty:
            continue
        node_ids = _flat_referenced_node_ids_fast(pid_df)
        if not node_ids:
            continue
        pos = id_to_idx.reindex(list(node_ids))
        valid = ~pos.isna().to_numpy()
        if not valid.any():
            continue
        pts = coords_arr[pos.to_numpy()[valid].astype(int)]
        mins[i] = pts.min(axis=0)
        maxs[i] = pts.max(axis=0)

    valid_rows = ~np.isnan(mins).any(axis=1)
    valid_idx = np.nonzero(valid_rows)[0]

    pairs = []
    # Outer loop over each valid PID; inner comparison against every
    # LATER valid PID is fully vectorized (numpy broadcast), not a
    # nested Python loop -- O(n) outer iterations, each O(n) numpy work,
    # trivial even at n ~ 700 real parts (confirmed: well under a
    # second in practice).
    for pos_i in range(len(valid_idx)):
        i = valid_idx[pos_i]
        later = valid_idx[pos_i + 1:]
        if len(later) == 0:
            continue
        cond1 = np.all(mins[i] - padding_mm <= maxs[later], axis=1)
        cond2 = np.all(mins[later] - padding_mm <= maxs[i], axis=1)
        overlap = cond1 & cond2
        for j in later[overlap]:
            pairs.append((pids[i], pids[int(j)]))

    return pairs


def detect_all_pid_penetrations(elements_df, nodes_df, pid_to_tier, pid_surfaces,
                                 candidate_pairs, tolerance_mm=0.0, search_radius_mm=10.0,
                                 exclude_pid_pairs=None, containment_threshold=0.85,
                                 containment_min_checkable_nodes=5, _pid_groups=None):
    """
    The all-pairs generalization of detect_soft_hard_penetrations: for
    every candidate (pid_a, pid_b) pair (from find_overlapping_pid_pairs
    -- NOT re-derived here, so the expensive broad-phase step is only
    ever done once even if this function is called repeatedly across
    escalation rounds), determine which part is the FIXED reference
    surface (the higher-priority/lower-tier part; a same-tier pair uses
    a deterministic PID-number tiebreak -- see this module's own "WHO
    MOVES" comment above classify_priority_tiers) and check the OTHER
    part's nodes against it, using the EXACT SAME nearest-face signed-
    distance method as detect_soft_hard_penetrations (self-contained
    here, not calling that function, to keep the original bone-only path
    completely untouched).

    WELD EXCLUSION: a node referenced by BOTH pid_a's and pid_b's own
    elements (an intentional shared/attachment node between exactly
    those two parts) is excluded from THAT PAIR's check -- it may still
    be checked normally against any OTHER, unrelated part.

    CONTAINMENT AUTO-EXCLUSION (critical -- confirmed necessary via a
    real whole-body test run, not theoretical): many real anatomical
    pairs are INTENTIONALLY, ENTIRELY nested one inside the other by
    design -- e.g. trabecular (spongy) bone filling the interior of its
    own cortical (shell) bone, a disc nucleus pulposus inside its own
    annulus fibrosus, or flesh/fascia inside skin. These are NOT
    penetration defects; the inner part is SUPPOSED to sit fully inside
    the outer one. Naively flagging "node is inside the other part's
    surface" would mark THOUSANDS of perfectly normal interior nodes as
    violations and, if repaired, would try to shove an entire bone's
    trabecular core out through its own cortical shell -- destroying the
    model. To tell genuine nesting apart from a real LOCAL overlap
    defect (e.g. two adjacent organs pressed slightly into each other at
    a shared boundary), this function groups all directions by moving
    PID and computes the UNION containment fraction across EVERY fixed
    part currently paired against that one moving part -- not just one
    pair in isolation. This matters because a single "inner" structure
    is often enclosed by MULTIPLE separately-named outer parts in real
    anatomy (e.g. a skull's trabecular diploe layer sandwiched between
    six separate cortical plates, or a vertebra's trabecular core
    wrapped by BOTH its own body AND arch cortical shells) -- no single
    one of those outer parts alone would reach containment_threshold,
    but the union of all of them together does (confirmed necessary via
    a real whole-body test run: without the union step, only single-
    cortical-bone nesting like patella/scapula/clavicle was caught, and
    the multi-cortical skull/vertebra cases still produced thousands of
    false-positive violations). For each moving PID, a node counts as
    "inside" if it lands inside ANY of that PID's currently-paired fixed
    surfaces it is NOT welded to (using the exact same nearest-face
    signed-distance sign and per-direction weld exclusion as Pass 3's
    own violation flagging -- a node shared with a specific fixed part,
    e.g. a tied cortical/trabecular mesh interface, sits right ON that
    part's surface where the sign is numerically borderline, and was
    confirmed via a real whole-body test run to otherwise DILUTE an
    obviously-nested part's containment fraction far below threshold if
    left in).

    A NEIGHBOR-LEVEL (not per-node) relevance gate also decides whether a
    given fixed part gets to vote on containment AT ALL: it must have at
    least containment_min_checkable_nodes of moving_pid's own nodes
    within search_radius_mm of it (unwelded), otherwise it is skipped
    entirely for this moving PID. This was confirmed necessary via a
    real whole-body test run for a DIFFERENT failure mode than weld
    dilution: a broad-phase candidate pair can include a totally
    UNRELATED, distant part purely from loose bounding-box coincidence
    (e.g. torso flesh vs. a pelvis bone -- essentially none of the
    flesh's own nodes are anywhere near that bone), and without this
    gate that irrelevant neighbor's far-field sign values (geometrically
    meaningless at that range) would wrongly dilute a genuinely-nested
    neighbor's clean signal elsewhere in the same group. Deliberately a
    per-NEIGHBOR gate rather than a per-NODE one: a per-node radius gate
    was tried first and found to break thick, genuinely fully-nested
    soft tissue (e.g. flesh sandwiched entirely inside skin), where the
    vast majority of the inner part's own nodes are legitimately MORE
    than search_radius_mm from the boundary -- restricting the sign test
    itself to only nearby nodes would throw away that reliable deep-
    interior majority and keep only the numerically ambiguous near-
    boundary slice. Once a neighbor PASSES the gate, it votes using its
    FULL, unrestricted sign test over every one of moving_pid's own
    nodes (however deep), so a truly nested part's reliable interior is
    never discarded. A node with no relevant, unwelded direction at all
    is excluded from the fraction entirely (neither counted as inside
    nor outside). If the resulting union fraction is >= containment_threshold,
    EVERY (fixed, moving) direction sharing that moving PID is
    classified as expected whole-part nesting together: NO violations
    are emitted for any of them (not even ones that would have passed
    the normal search_radius_mm/tolerance_mm gate), and each is recorded
    in the returned nested_pairs_df for audit/reporting instead. A real
    partial/boundary overlap (only nodes near the shared surface are
    actually inside any neighbour) will have a MUCH lower union fraction
    and is unaffected by this check -- it is still detected and flagged
    normally. A moving part with only ONE fixed neighbour reduces
    exactly to the simple single-pair case.

    MINIMUM SAMPLE SIZE (containment_min_checkable_nodes): a tiny
    checkable-node count makes the fraction statistically meaningless --
    a single stray node that happens to be embedded in a neighbour looks
    IDENTICAL to "100% contained" (1 out of 1) even though it is exactly
    the kind of genuine, isolated defect this whole function exists to
    catch. If a moving PID has fewer than this many checkable nodes
    against the surfaces it is paired with, it is NEVER classified as
    nested regardless of its fraction -- confirmed necessary via this
    module's own synthetic ground-truth test (single-node test
    violations were being swallowed as "100% contained" before this
    guard was added). Real anatomical parts in this pipeline always have
    hundreds to thousands of nodes, so this guard has no effect on
    genuine whole-body nesting cases -- it only protects small/edge-case
    parts.

    Inputs:
        elements_df (pd.DataFrame): FULL element table (SOLID + SHELL).
        nodes_df (pd.DataFrame): CURRENT node coordinates.
        pid_to_tier (dict[int, int]): From classify_priority_tiers.
        pid_surfaces (dict[int, dict]): From build_all_part_surfaces.
        candidate_pairs (list[tuple[int, int]]): From find_overlapping_
            pid_pairs -- the broad-phase-pruned pair list to actually
            check.
        tolerance_mm, search_radius_mm (float): Same meaning as detect_
            soft_hard_penetrations' own parameters.
        exclude_pid_pairs (set[tuple[int, int]] | None): (PID_a, PID_b)
            pairs to skip entirely (checked in EITHER order) -- same
            EXPECTED_OVERLAP_PAIRS concept/list as the bone-only path.
        containment_threshold (float): Fraction (0-1) of the moving
            part's own candidate nodes that must land inside the fixed
            part's surface before that (fixed, moving) direction is
            classified as expected whole-part nesting rather than a
            genuine overlap defect -- see CONTAINMENT AUTO-EXCLUSION
            above. Default 0.85 -- tuned to tolerate a normal amount of
            local mesh/surface noise while still catching near-total
            containment (real whole-body test data showed nested pairs
            landing at 95-100%, and genuine partial/boundary overlaps
            landing well under half, so 0.85 leaves comfortable margin
            on both sides).
        containment_min_checkable_nodes (int): Dual-purpose threshold --
            see MINIMUM SAMPLE SIZE and the neighbor-level relevance gate
            above. (1) The minimum number of a moving PID's own nodes
            that must be within search_radius_mm of a given fixed
            neighbor before that neighbor is even allowed to vote on
            containment at all (filters out spurious, geometrically
            irrelevant broad-phase candidates). (2) The minimum total
            checkable node count a moving PID must have (across all its
            voting neighbors combined) before it is ELIGIBLE for
            containment classification at all (protects against tiny/
            single-node statistical flukes). Default 5.
        _pid_groups (dict[int, pd.DataFrame] | None): Internal perf hook
            -- see build_all_part_surfaces' own docstring for the exact
            contract (avoids re-filtering the full multi-million-row
            elements_df once per candidate pair -- a real, confirmed
            bottleneck at whole-body scale, since a single large part
            can appear in dozens of candidate pairs). None (default)
            groups internally. Per-PID node-ID sets are ALSO cached
            internally across pairs (a PID's own referenced-node-ID set
            never changes between pairs within one call), so a part
            touching many neighbours only pays that cost once too.

    Returns:
        tuple:
            violations_df (pd.DataFrame): One row per violating node,
                columns: 'NodeID' (the node that is penetrating),
                'moving_PID' (its own part), 'fixed_PID' (the part
                surface it penetrated into -- this is the part that must
                NOT move when repairing this specific violation),
                'depth_mm', 'nearest_face_distance_mm', 'same_tier'
                (bool -- True if this pair's fixed/moving assignment
                came from the PID-number tiebreak rather than a genuine
                priority difference, useful for reporting). Empty
                (correct schema, zero rows) if nothing violates. NEVER
                contains rows for a direction classified as nested --
                see CONTAINMENT AUTO-EXCLUSION above.
            nested_pairs_df (pd.DataFrame): One row per (fixed_PID,
                moving_PID) direction classified as expected whole-part
                nesting and excluded from violations_df entirely.
                Columns: 'fixed_PID', 'moving_PID', 'containment_
                fraction', 'same_tier'. Empty (correct schema, zero
                rows) if nothing was classified as nested. For audit/
                reporting only -- callers should NOT attempt to repair
                anything listed here.
    """
    empty_df = pd.DataFrame(columns=[
        'NodeID', 'moving_PID', 'fixed_PID', 'depth_mm', 'nearest_face_distance_mm', 'same_tier'
    ])
    empty_nested_df = pd.DataFrame(columns=[
        'fixed_PID', 'moving_PID', 'containment_fraction', 'same_tier'
    ])
    exclude_pid_pairs = set(exclude_pid_pairs or ())
    pid_groups = _pid_groups if _pid_groups is not None else _group_elements_by_pid(elements_df)
    node_ids_arr = nodes_df['NodeID'].to_numpy()
    coords_arr = nodes_df[['x', 'y', 'z']].to_numpy(dtype=float)
    id_to_idx = pd.Series(np.arange(len(node_ids_arr)), index=node_ids_arr)
    # Cache each PID's own referenced-node-ID set across pairs -- a part
    # touching many neighbours (common for large parts) would otherwise
    # recompute the SAME set on every one of those pairs.
    node_id_set_cache = {}

    def _node_ids_for(pid):
        if pid not in node_id_set_cache:
            pid_df = pid_groups.get(pid)
            node_id_set_cache[pid] = _flat_referenced_node_ids_fast(pid_df) if pid_df is not None and not pid_df.empty else set()
        return node_id_set_cache[pid]

    def _excluded(a, b):
        return (a, b) in exclude_pid_pairs or (b, a) in exclude_pid_pairs

    # ---- Pass 1: resolve (fixed, moving, same_tier) direction for every
    # candidate pair (tier comparison / PID tiebreak, exactly as before). ----
    directions_list = []
    for pid_a, pid_b in candidate_pairs:
        if _excluded(pid_a, pid_b):
            continue
        tier_a = pid_to_tier.get(pid_a, 3)
        tier_b = pid_to_tier.get(pid_b, 3)
        if tier_a < tier_b:
            directions_list.append((pid_a, pid_b, False))  # (fixed, moving, same_tier)
        elif tier_b < tier_a:
            directions_list.append((pid_b, pid_a, False))
        else:
            # Same tier: deterministic tiebreak for WHICH one repair
            # treats as fixed (lower PID number) -- see this module's
            # own "WHO MOVES" comment above classify_priority_tiers.
            fixed_pid, moving_pid = (pid_a, pid_b) if pid_a < pid_b else (pid_b, pid_a)
            directions_list.append((fixed_pid, moving_pid, True))

    # ---- Pass 2: group directions by moving_PID and classify containment
    # using the UNION of every fixed part currently paired against that
    # moving part -- NOT just one pair at a time. This is essential for
    # real anatomy where a single "inner" structure is enclosed by
    # MULTIPLE separately-named outer parts (e.g. a skull's trabecular
    # diploe layer sandwiched between six separate cortical plates, or a
    # vertebra's trabecular core wrapped by BOTH its own body AND arch
    # cortical shells): no single one of those outer parts alone reaches
    # containment_threshold, but the union of all of them together does.
    # Each direction's per-node (dist, depth) arrays are cached here and
    # REUSED in Pass 3 below, so nothing is ever KD-tree-queried twice.
    moving_groups = collections.defaultdict(list)
    for fixed_pid, moving_pid, same_tier in directions_list:
        moving_groups[moving_pid].append((fixed_pid, same_tier))

    nested_directions = set()
    nested_rows = []
    cached = {}  # (fixed_pid, moving_pid) -> (ids_arr, dist, depth, welded)

    for moving_pid, fixed_list in moving_groups.items():
        moving_node_ids = _node_ids_for(moving_pid)
        if not moving_node_ids:
            continue
        ids_arr = np.array(sorted(moving_node_ids), dtype=np.int64)
        pos = id_to_idx.reindex(ids_arr)
        valid = ~pos.isna().to_numpy()
        ids_arr = ids_arr[valid]
        if ids_arr.size == 0:
            continue
        coords = coords_arr[pos.to_numpy()[valid].astype(int)]

        inside_any = np.zeros(len(ids_arr), dtype=bool)
        checkable_any = np.zeros(len(ids_arr), dtype=bool)
        for fixed_pid, same_tier in fixed_list:
            surface = pid_surfaces.get(fixed_pid)
            if surface is None:
                continue
            dist, face_idx = surface['kdtree'].query(coords, k=1)
            face_centroids = surface['face_centroids'][face_idx]
            face_normals = surface['face_normals'][face_idx]
            signed_dist = np.einsum('ij,ij->i', coords - face_centroids, face_normals)
            depth = -signed_dist

            # WELD EXCLUSION (per direction): a node shared between THIS
            # specific moving/fixed pair (e.g. a tied cortical/trabecular
            # mesh interface, common in real anatomy) sits right ON the
            # fixed part's own surface, where the signed-distance sign is
            # numerically borderline/noisy -- confirmed via a real whole-
            # body test run to otherwise DILUTE the containment fraction
            # of a genuinely fully-nested part far below threshold (e.g.
            # coxal trabecular/cortical measured ~73% instead of ~100%
            # once its many shared interface nodes were wrongly counted
            # as "50/50 borderline" instead of excluded). Excluded from
            # BOTH the numerator and denominator here, same as Pass 3's
            # own violation-flagging weld exclusion below.
            fixed_node_ids = _node_ids_for(fixed_pid)
            if fixed_node_ids:
                welded = np.isin(ids_arr, np.fromiter(fixed_node_ids, dtype=np.int64, count=len(fixed_node_ids)))
            else:
                welded = np.zeros(len(ids_arr), dtype=bool)
            cached[(fixed_pid, moving_pid)] = (ids_arr, dist, depth, welded)

            # NEIGHBOR-LEVEL RELEVANCE GATE: only let this fixed_pid vote
            # on containment AT ALL if it has a MEANINGFUL number of
            # moving_pid's nodes genuinely near it (>= containment_min_
            # checkable_nodes within search_radius_mm) -- this is
            # deliberately a per-NEIGHBOR gate, not a per-NODE gate (a
            # per-node radius gate was tried and found to break thick,
            # genuinely fully-nested soft tissue, e.g. flesh sandwiched
            # entirely inside skin, where the vast majority of the
            # INNER part's own nodes are legitimately MORE than
            # search_radius_mm from the boundary -- restricting the
            # sign test itself to only near nodes throws away the
            # reliable deep-interior majority and keeps only the
            # numerically ambiguous near-boundary slice). A neighbor
            # that FAILS this gate (essentially no nodes anywhere near
            # it -- e.g. a torso flesh part vs. an unrelated pelvis bone
            # that only entered the SAME broad-phase group via a loose,
            # coincidental bounding-box overlap) is skipped entirely so
            # its far-field, geometrically-meaningless sign values can
            # never dilute a genuinely nested neighbor's clean signal
            # elsewhere in the same group. A neighbor that PASSES this
            # gate votes using its FULL, unrestricted sign test (every
            # one of moving_pid's own nodes, however deep), so a truly
            # nested part's reliable deep-interior nodes are never
            # thrown away.
            near_count = int(((dist <= search_radius_mm) & (~welded)).sum())
            if near_count < containment_min_checkable_nodes:
                continue

            not_welded = ~welded
            checkable_any |= not_welded
            inside_any |= (depth > 0) & not_welded

        # CONTAINMENT AUTO-EXCLUSION -- see this function's own docstring.
        # Computed over all of moving_pid's own CHECKABLE (unwelded, from
        # a relevant neighbor) nodes. MINIMUM SAMPLE SIZE guard: a tiny
        # checkable count is never eligible, regardless of its fraction
        # -- see this function's own docstring.
        n_checkable = int(checkable_any.sum())
        containment_fraction = float(inside_any[checkable_any].mean()) if n_checkable else 0.0
        if n_checkable >= containment_min_checkable_nodes and containment_fraction >= containment_threshold:
            for fixed_pid, same_tier in fixed_list:
                nested_directions.add((fixed_pid, moving_pid))
                nested_rows.append({
                    'fixed_PID': fixed_pid, 'moving_PID': moving_pid,
                    'containment_fraction': containment_fraction, 'same_tier': same_tier,
                })

    # ---- Pass 3: flag genuine violations for every non-nested direction,
    # reusing Pass 2's cached distances AND weld masks (never re-queried
    # or recomputed). ----
    all_rows = []
    for fixed_pid, moving_pid, same_tier in directions_list:
        if (fixed_pid, moving_pid) in nested_directions:
            continue
        cached_entry = cached.get((fixed_pid, moving_pid))
        if cached_entry is None:
            continue
        ids_arr, dist, depth, welded = cached_entry
        not_welded = ~welded

        penetrating = not_welded & (dist <= search_radius_mm) & (depth > tolerance_mm)
        if not penetrating.any():
            continue

        all_rows.append(pd.DataFrame({
            'NodeID': ids_arr[penetrating],
            'moving_PID': moving_pid,
            'fixed_PID': fixed_pid,
            'depth_mm': depth[penetrating],
            'nearest_face_distance_mm': dist[penetrating],
            'same_tier': same_tier,
        }))

    nested_pairs_df = (
        pd.DataFrame(nested_rows, columns=['fixed_PID', 'moving_PID', 'containment_fraction', 'same_tier'])
        if nested_rows else empty_nested_df
    )
    if not all_rows:
        return empty_df, nested_pairs_df
    result = pd.concat(all_rows, ignore_index=True)
    result = result.sort_values('depth_mm', ascending=False).reset_index(drop=True)
    return result, nested_pairs_df


def group_all_pid_penetration_violations_into_pockets(violations_df, elements_df_all, expansion_layers):
    """
    Split detect_all_pid_penetrations' combined violations table into
    independent repair POCKETS. Grouping happens in TWO stages:
      1. By (fixed_PID, moving_PID) pair FIRST -- every pocket must have
         exactly ONE consistent "who is protected, who moves" story
         (repair_soft_hard_penetrations, reused unchanged for this
         all-pairs case, needs a single fixed reference surface per
         call), so violations against different fixed/moving pairs are
         NEVER merged, even if geometrically close.
      2. WITHIN each (fixed_PID, moving_PID) group, further split by
         NODE ADJACENCY (reusing group_bad_penetration_nodes_into_
         pockets on that group's moving-PID element subset) -- so two
         far-apart violations against the same two parts still solve as
         separate, smaller, faster problems, exactly like every other
         pocket-splitting step in this codebase.

    Inputs:
        violations_df (pd.DataFrame): From detect_all_pid_penetrations.
        elements_df_all (pd.DataFrame): FULL element table (SOLID +
            SHELL) -- used to build each group's own moving-PID element
            subset for the node-adjacency sub-split.
        expansion_layers (int): Same meaning as group_bad_elements_into_
            pockets' own parameter -- must be at least as large as the
            largest free_node_layers + constraint_layers any escalation
            rung could reach.

    Returns:
        list[dict]: One entry per pocket, each:
            {'fixed_PID': int, 'moving_PID': int,
             'node_ids': list[int], 'pocket_size': int}
            Sorted by descending pocket_size (largest/likely-slowest
            first, matching every other pocket list in this codebase).
            Empty list if violations_df is empty.
    """
    if violations_df.empty:
        return []

    pockets = []
    for (fixed_pid, moving_pid), group in violations_df.groupby(['fixed_PID', 'moving_PID']):
        moving_elements_df = elements_df_all[elements_df_all['PID'] == moving_pid]
        node_id_groups = group_bad_penetration_nodes_into_pockets(
            group['NodeID'].unique().tolist(), moving_elements_df, expansion_layers=expansion_layers
        )
        for node_ids in node_id_groups:
            pockets.append({
                'fixed_PID': int(fixed_pid),
                'moving_PID': int(moving_pid),
                'node_ids': node_ids,
                'pocket_size': len(node_ids),
            })

    pockets.sort(key=lambda p: p['pocket_size'], reverse=True)
    return pockets


def group_bad_penetration_nodes_into_pockets(penetrating_node_ids, soft_elements_df, expansion_layers):
    """
    Split a (possibly large, whole-model) list of penetrating soft NodeIDs
    into independent "pockets" -- the penetration-repair analogue of
    group_bad_elements_into_pockets, seeded from NODE IDs directly
    (detect_soft_hard_penetrations' own node_level_df output) rather than
    element rows, since a penetration defect is fundamentally a per-NODE
    violation, not a per-element one.

    SIMPLER THAN group_bad_shells_into_pockets' solid-aware version, ON
    PURPOSE: that function needs extra "claim every node of any touched
    solid" logic because solids CAN move during shell repair, so two
    shell pockets touching different corners of the SAME solid could
    otherwise be solved in unsafe parallel ignorance of each other. HARD
    (bone) elements here NEVER move (by this entire feature's explicit
    design), so two penetration pockets are always safe to solve
    independently even if they happen to reference the same hard part or
    even the same hard element/face -- ordinary SOFT-element-only node
    adjacency is sufficient, exactly like group_bad_elements_into_pockets'
    plain (non-solid-aware) version.

    Inputs:
        penetrating_node_ids (Iterable[int]): Penetrating soft NodeIDs
            (e.g. node_level_df['NodeID'] from detect_soft_hard_
            penetrations, already filtered/excluded as desired).
        soft_elements_df (pd.DataFrame): ALL non-hard (soft) elements
            (SOLID + SHELL) -- used only to build soft-to-soft node
            adjacency for the reach/expansion step, never itself split.
        expansion_layers (int): Adjacency rings per seed node, chosen the
            SAME way as group_bad_elements_into_pockets' own parameter --
            at least as large as the largest free_node_layers +
            constraint_layers any escalation attempt could use.

    Returns:
        list[list[int]]: penetrating_node_ids split into pockets (each a
            plain list of NodeIDs), sorted by descending pocket size.
            Empty list if penetrating_node_ids is empty.
    """
    seed_ids = sorted(set(int(n) for n in penetrating_node_ids))
    if not seed_ids:
        return []

    all_element_nodes = _unique_node_lists_fast(soft_elements_df)
    nid_to_elem_indices = {}
    for elem_idx, unique_nids in enumerate(all_element_nodes):
        for nid in unique_nids:
            nid_to_elem_indices.setdefault(nid, set()).add(elem_idx)

    n_seed = len(seed_ids)
    parent = list(range(n_seed))

    def _find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    node_owner = {}
    for idx, nid in enumerate(seed_ids):
        reach = _expand_node_region(
            {nid}, nid_to_elem_indices, all_element_nodes, expansion_layers
        )
        for touched_nid in reach:
            owner = node_owner.get(touched_nid)
            if owner is None:
                node_owner[touched_nid] = idx
            else:
                _union(idx, owner)

    groups = {}
    for idx in range(n_seed):
        root = _find(idx)
        groups.setdefault(root, []).append(seed_ids[idx])

    pockets = list(groups.values())
    pockets.sort(key=len, reverse=True)
    return pockets


# PENETRATION-REPAIR-SPECIFIC continuation/tuning constants -- separate
# from _CONTINUATION_STEP/_MAX_CONTINUATION_STAGES (those are in VOLUME/
# JACOBIAN units, not mm, so reusing them directly would be meaningless).
# Sized from real observed F05/CaseA penetration depths (0-8mm) -- a 1mm
# per-stage ramp comfortably covers the worst real case seen so far
# within a handful of stages while keeping each individual step small.
_PENETRATION_CONTINUATION_STEP_MM = 1.0
_PENETRATION_MAX_CONTINUATION_STAGES = 20


def repair_soft_hard_penetrations(penetrating_node_ids_pocket, soft_elements_df,
                                   hard_surface_index, hard_node_ids, nodes_df,
                                   max_displacement=2.0, tolerance_mm=0.0,
                                   search_radius_mm=10.0,
                                   free_node_layers=1, constraint_layers=1,
                                   quality_volume_epsilon=1e-3, quality_shell_epsilon=0.05):
    """
    Push penetrating SOFT-tissue nodes (e.g. muscle) back outside a 'hard'
    (e.g. bone) part's surface, via the SAME scipy SLSQP/trust-constr
    constrained-optimization machinery already validated for solid-volume
    (repair_inverted_elements) and shell-jacobian (repair_distorted_
    shells) repair -- this is that same family's PENETRATION sibling.
    Hard/bone geometry is NEVER a free variable and never even appears in
    the constrained zone as anything other than a FIXED reference surface
    (hard_surface_index) -- by construction, this function cannot move a
    single hard/bone node, matching the user's explicit requirement.

    *** WHY THIS CANNOT "BOW IN" THE MUSCLE (the user's explicit concern)
    *** every SOFT solid/shell element touching the free/constrained zone
    -- not just the penetrating nodes' own elements, but every neighbour
    pulled in by free_node_layers/constraint_layers -- gets a hard,
    non-negotiable NON-REGRESSION constraint on its OWN quality metric
    (volume/Jacobian for solids, jacobian_ratio for shells), exactly the
    SAME min(baseline, epsilon) relaxed floor rule repair_inverted_
    elements already uses for ITS OWN neighbour elements (see that
    function's _build_floor_and_bad_mask docstring) -- NOT a stricter
    "never move at all from its exact starting value" rule. This was a
    REAL mistake caught during real-data testing (see this function's own
    module-level history): an earlier version used the element's raw
    baseline value directly as the floor (i.e., as if epsilon were
    infinite), which is FAR stricter than solid/shell repair's own
    neighbour rule -- on a real 276-node F05 shoulder/humerus pocket,
    escalating free_node_layers under that stricter rule made the worst
    constraint violation get MEASURABLY WORSE (attempt 1's worst -3.2 vs.
    attempt 3's worst -18.2), because pulling MORE already-healthy
    elements into the zone under an unnecessarily rigid floor only added
    MORE hard-to-satisfy constraints, not more genuine room to move. The
    correct rule (matching solid/shell repair exactly): floor_i =
    min(baseline_i, quality_epsilon) -- a healthy element (baseline far
    above epsilon) gets enormous headroom to shift as part of a natural,
    coordinated local bulge; only an element ALREADY at/near the epsilon
    threshold is held tightly to its current value. This is what actually
    prevents "bowing in": not a rigid freeze, but the SAME generous-but-
    safe non-regression contract already proven correct elsewhere in this
    file -- the same "smooth, not creased" outcome smooth_repair_zone_
    boundary was built to guarantee for jacobian repair, achieved here
    directly through the constraint structure instead of a separate
    smoothing pass.

    ZONE CONSTRUCTION:
      1. Seed nodes = penetrating_node_ids_pocket (one pocket from
         group_bad_penetration_nodes_into_pockets).
      2. free_nids_set / constraint_nids_set expand via SOFT-element-only
         node adjacency (soft_elements_df) -- hard/bone connectivity is
         NEVER used to grow these sets. Any node also referenced by a
         hard-PID element (a welded/attachment point) is explicitly
         EXCLUDED from ever becoming free, even if it is topologically
         reachable from a penetrating node via a shared soft element --
         moving a genuine weld point would separate it from its bone
         anchor, creating a new crack/gap defect instead of fixing one.
      3. The repair zone = every SOFT solid/shell element (mixed, exactly
         like repair_distorted_shells' own mixed solid+shell handling)
         touching the constrained node set.
      4. Every free-or-constrained-zone node within search_radius_mm of
         SOME hard face gets its own PENETRATION constraint row (see
         CONSTRAINT ROWS below) -- nodes with no nearby hard face at all
         get no such row (not applicable there).

    CONSTRAINT ROWS:
      - Element quality (solid volume/Jacobian, shell jacobian_ratio):
        RELAXED non-regression floor = min(baseline_i, quality_epsilon)
        (hard/soft-baseline blended for fully-pinned elements, IDENTICAL
        rule/rationale to repair_inverted_elements' own group8_has_free
        handling) -- see "WHY THIS CANNOT BOW IN" above for why this must
        be the RELAXED form, not the element's raw baseline value.
      - Penetration depth, per protected node: signed distance to its
        OWN nearest hard face (frozen at the pocket's baseline geometry,
        exactly like this file's other "frozen reference" conventions --
        hard geometry never moves, so this correspondence stays valid
        for the whole solve) must be >= a per-row floor:
          - penetrating (bad-list) nodes: ramped via continuation from
            their own baseline depth up to +tolerance_mm (see
            _PENETRATION_CONTINUATION_STEP_MM).
          - every other protected node: min(baseline_signed_distance,
            tolerance_mm) -- pure non-regression, preventing this pocket
            from relieving one penetration by creating a NEW one nearby.
        This constraint is LINEAR in the free node's own position (the
        reference face/normal is frozen), so its analytic gradient is a
        simple constant vector -- no softmin approximation needed here,
        unlike the volume/Jacobian rows.

    Inputs:
        penetrating_node_ids_pocket (Iterable[int]): One pocket's worth of
            penetrating soft NodeIDs (must-fix list for this call).
        soft_elements_df (pd.DataFrame): ALL soft (non-hard-PID) elements,
            SOLID + SHELL mixed (bad + neighbours) -- used for adjacency
            expansion AND as the repair zone's own element table.
        hard_surface_index (dict): From build_hard_surface_index -- the
            SAME frozen reference surface detect_soft_hard_penetrations
            uses; queried here for the nearest hard face per zone node.
        hard_node_ids (set[int]): Every NodeID referenced by any hard-PID
            element (e.g. from _flat_referenced_node_ids_fast on the hard
            elements table) -- used to exclude welded nodes from the free
            set (see ZONE CONSTRUCTION point 2).
        nodes_df (pd.DataFrame): CURRENT node table (NodeID, x, y, z) --
            normally the post-solid-and-shell-repair table.
        max_displacement (float): Max per-axis movement for any free node.
        tolerance_mm (float): Target clearance past the hard surface for
            previously-penetrating nodes (same meaning/units as detect_
            soft_hard_penetrations' own parameter -- pass the SAME value
            used for detection so "fixed" genuinely means "no longer
            flagged by the same check").
        search_radius_mm (float): Same meaning as detect_soft_hard_
            penetrations' own parameter -- how far a zone node's nearest
            hard face may be before it's considered "not applicable".
        free_node_layers, constraint_layers (int): Soft-element adjacency
            expansion depth (same meaning as repair_inverted_elements).
        quality_volume_epsilon (float): RELAXED non-regression threshold
            for solid (hex8/penta6/tet4) neighbour elements -- same
            meaning/default scale as repair_inverted_elements' own
            volume_epsilon parameter (a tiny positive "barely valid"
            value, NOT a target -- see "WHY THIS CANNOT BOW IN" above).
        quality_shell_epsilon (float): RELAXED non-regression threshold
            for shell (quad4/tri3) neighbour elements' jacobian_ratio --
            deliberately far BELOW shell_band_floor (e.g. 0.3), since
            this is only a "don't let it collapse to near-degenerate"
            backstop, not a quality target (repair_distorted_shells'
            own band-floor target is a completely separate, later phase).

    Returns:
        tuple:
            - updated_nodes_df (pd.DataFrame): Repaired node table
              (NodeID, x, y, z) -- hard/bone nodes always unchanged BY
              CONSTRUCTION (never part of any coordinate array this
              function writes to).
            - info (dict): 'n_free_nodes', 'n_repair_zone_elements',
              'n_penetration_rows', 'n_still_penetrating',
              'n_neighbor_regressions', 'max_neighbor_regression'
              (analogous to repair_inverted_elements' own fields of the
              same name -- see its docstring for the real production bug
              this guards against -- generalized here to cover BOTH this
              function's own protected-quantity families: every element-
              quality row, which unlike this function's penetration rows
              has NO 'bad' concept at all here and so is always non-
              regression-only, AND every non-bad/neighbour penetration
              row), 'max_displacement_achieved', 'converged'.
    """
    from scipy.optimize import minimize, Bounds, OptimizeResult
    from scipy.sparse import coo_matrix, identity as sparse_identity

    if max_displacement <= 0:
        raise ValueError("max_displacement must be > 0.")
    if free_node_layers < 0 or constraint_layers < 0:
        raise ValueError("free_node_layers and constraint_layers must be >= 0.")

    bad_nids_set = set(int(n) for n in penetrating_node_ids_pocket) - set(hard_node_ids)
    if not bad_nids_set:
        unchanged_nodes_df = nodes_df[['NodeID', 'x', 'y', 'z']].copy()
        info = {
            'n_free_nodes': 0, 'n_repair_zone_elements': 0, 'n_penetration_rows': 0,
            'n_still_penetrating': 0, 'max_displacement_achieved': 0.0, 'converged': True,
        }
        return unchanged_nodes_df, info

    all_soft_element_nodes = _unique_node_lists_fast(soft_elements_df)
    nid_to_soft_elem_indices = {}
    for elem_idx, unique_nids in enumerate(all_soft_element_nodes):
        for nid in unique_nids:
            nid_to_soft_elem_indices.setdefault(nid, set()).add(elem_idx)

    free_nids_set = _expand_node_region(
        bad_nids_set, nid_to_soft_elem_indices, all_soft_element_nodes, free_node_layers
    ) - set(hard_node_ids)  # never let a welded node become free -- see docstring point 2
    constraint_nids_set = _expand_node_region(
        free_nids_set, nid_to_soft_elem_indices, all_soft_element_nodes, constraint_layers
    )
    free_nids_list = sorted(free_nids_set)

    if not free_nids_list:
        # Every seed node in this pocket turned out to be welded (excluded)
        # -- nothing this function is allowed to move.
        unchanged_nodes_df = nodes_df[['NodeID', 'x', 'y', 'z']].copy()
        info = {
            'n_free_nodes': 0, 'n_repair_zone_elements': 0, 'n_penetration_rows': 0,
            'n_still_penetrating': len(bad_nids_set), 'max_displacement_achieved': 0.0,
            'converged': False,
        }
        return unchanged_nodes_df, info

    repair_indices = [
        elem_idx
        for elem_idx, elem_nodes in enumerate(all_soft_element_nodes)
        if set(elem_nodes) & constraint_nids_set
    ]
    repair_zone_node_lists = [all_soft_element_nodes[i] for i in repair_indices]
    repair_zone_eids = soft_elements_df['EID'].to_numpy()[repair_indices]
    repair_zone_etypes = soft_elements_df['ETYPE'].to_numpy()[repair_indices]

    zone_nids_set = set(free_nids_set)
    for nodes in repair_zone_node_lists:
        zone_nids_set.update(nodes)
    zone_nids_list = sorted(zone_nids_set)
    zone_nid_to_local = {nid: i for i, nid in enumerate(zone_nids_list)}

    node_coord = nodes_df.set_index('NodeID')[['x', 'y', 'z']]
    zone_coords0 = node_coord.loc[zone_nids_list].values.astype(float)

    free_zone_idx = np.array([zone_nid_to_local[nid] for nid in free_nids_list], dtype=int)
    free_orig_coords = zone_coords0[free_zone_idx].copy()

    zone_local_to_freevar = np.full(len(zone_nids_list), -1, dtype=int)
    for var_idx, nid in enumerate(free_nids_list):
        zone_local_to_freevar[zone_nid_to_local[nid]] = var_idx

    # ---- Element quality (solid + shell mixed) non-regression groups ----
    # Grouping/padding convention IDENTICAL to repair_inverted_elements/
    # repair_distorted_shells (penta6 ridge node repeated into slots 6-8).
    # Separate dict keys for SHELL quad4 vs SOLID tet4 (both have 4 unique
    # nodes, but need different value/gradient formulas) so no ambiguity
    # is possible.
    tri_rows, quad_rows, tet4_rows, penta6_rows, hex8_rows = [], [], [], [], []
    for eid, etype, nodes in zip(repair_zone_eids, repair_zone_etypes, repair_zone_node_lists):
        n = len(nodes)
        local = [zone_nid_to_local[nid] for nid in nodes]
        if etype == 'SHELL':
            if n == 3:
                tri_rows.append(local)
            elif n == 4:
                quad_rows.append(local)
            else:
                raise ValueError(f"Unsupported SHELL element node count in penetration repair zone: {n}.")
        else:
            if n == 4:
                tet4_rows.append(local)
            elif n == 6:
                penta6_rows.append(local[:5] + [local[5], local[5], local[5]])
            elif n == 8:
                hex8_rows.append(local)
            else:
                raise ValueError(f"Unsupported SOLID element node count in penetration repair zone: {n}.")

    tri_idx = np.array(tri_rows, dtype=int) if tri_rows else np.zeros((0, 3), dtype=int)
    quad_idx = np.array(quad_rows, dtype=int) if quad_rows else np.zeros((0, 4), dtype=int)
    group4_idx = np.array(tet4_rows, dtype=int) if tet4_rows else np.zeros((0, 4), dtype=int)
    group6_idx = np.array(penta6_rows, dtype=int) if penta6_rows else np.zeros((0, 8), dtype=int)
    group8_idx = np.array(hex8_rows, dtype=int) if hex8_rows else np.zeros((0, 8), dtype=int)

    group8_has_free = (np.any(zone_local_to_freevar[group8_idx] != -1, axis=1)
                        if len(group8_idx) else np.zeros(0, dtype=bool))
    group6_has_free = (np.any(zone_local_to_freevar[group6_idx] != -1, axis=1)
                        if len(group6_idx) else np.zeros(0, dtype=bool))

    group8_beta = _frozen_softmin_beta(zone_coords0, group8_idx, 8)
    group6_beta = _frozen_softmin_beta(zone_coords0, group6_idx, 4)
    quad_n_ref = _frozen_quad_ref_normal(zone_coords0, quad_idx)
    quad_beta = _frozen_quad_softmin_beta(zone_coords0, quad_idx, quad_n_ref)

    # Baseline (pre-optimization) values -- these feed the RELAXED
    # min(baseline, epsilon) floor construction below, not used directly
    # as the floor themselves (see "WHY THIS CANNOT BOW IN" in the
    # docstring). hard-vs-soft baseline blending for fully-pinned
    # elements mirrors repair_inverted_elements' own group8_has_free rule
    # exactly (same rationale: a fully-pinned element's true hard-min
    # can never move, so the softer proxy baseline carries zero risk
    # there, while a free element needs the real hard-min baseline to
    # make its non-regression guarantee meaningful).
    v8_base = (np.where(
        group8_has_free,
        _batched_hexlike_value_and_grad(zone_coords0[group8_idx], 8)[0],
        _batched_hexlike_softmin_value_and_grad(zone_coords0[group8_idx], 8, group8_beta)[0],
    ) if len(group8_idx) else np.zeros(0))
    v8c_base = (_batched_hexlike_centroid_value_and_grad(zone_coords0[group8_idx])[0]
                if len(group8_idx) else np.zeros(0))
    v6_base = (np.where(
        group6_has_free,
        _batched_hexlike_value_and_grad(zone_coords0[group6_idx], 4)[0],
        _batched_hexlike_softmin_value_and_grad(zone_coords0[group6_idx], 4, group6_beta)[0],
    ) if len(group6_idx) else np.zeros(0))
    v6c_base = (_batched_hexlike_centroid_value_and_grad(zone_coords0[group6_idx])[0]
                if len(group6_idx) else np.zeros(0))
    v4_base = (_batched_tet_value_and_grad(zone_coords0[group4_idx])[0]
               if len(group4_idx) else np.zeros(0))
    quad_base = (_batched_quad_jacobian_ratio_value_and_grad(zone_coords0[quad_idx], quad_n_ref)[0]
                 if len(quad_idx) else np.zeros(0))
    tri_base = (_batched_tri3_jacobian_ratio_value_and_grad(zone_coords0[tri_idx])[0]
                if len(tri_idx) else np.zeros(0))

    # RELAXED non-regression floor = min(baseline, epsilon) -- NOT the
    # element's raw baseline value (see "WHY THIS CANNOT BOW IN" in the
    # docstring for why the raw-baseline version is a real, previously-
    # caught mistake: it needlessly tightens as more healthy neighbours
    # are pulled into the zone, fighting the escalation ladder instead of
    # helping it). Solid-family rows use quality_volume_epsilon; shell-
    # family rows use quality_shell_epsilon (different natural scales).
    quality_floor_parts = [
        np.minimum(v8_base, quality_volume_epsilon),
        np.minimum(v8c_base, quality_volume_epsilon),
        np.minimum(v6_base, quality_volume_epsilon),
        np.minimum(v6c_base, quality_volume_epsilon),
        np.minimum(v4_base, quality_volume_epsilon),
        np.minimum(quad_base, quality_shell_epsilon),
        np.minimum(tri_base, quality_shell_epsilon),
    ]
    n_quality_constraints = sum(len(p) for p in quality_floor_parts)

    # ---- Penetration constraint rows (frozen nearest-hard-face) ----
    # Every zone node (free set) within search_radius_mm of a hard face --
    # constrained (non-free) zone nodes need no penetration row (they
    # can't move, so their own depth can't regress from THIS pocket's
    # solve; they still get their ELEMENT quality protection above).
    kdtree = hard_surface_index['kdtree']
    pen_query_coords = zone_coords0[free_zone_idx]
    pen_dist, pen_face_idx = kdtree.query(pen_query_coords, k=1)
    pen_within_radius = pen_dist <= search_radius_mm

    pen_face_centroids = hard_surface_index['face_centroids'][pen_face_idx]
    pen_face_normals = hard_surface_index['face_normals'][pen_face_idx]
    pen_baseline_signed = np.einsum(
        'ij,ij->i', pen_query_coords - pen_face_centroids, pen_face_normals
    )

    pen_free_local_idx = np.nonzero(pen_within_radius)[0]  # indices into free_zone_idx/free_nids_list
    n_pen_rows = len(pen_free_local_idx)
    pen_is_bad = np.array([free_nids_list[i] in bad_nids_set for i in pen_free_local_idx], dtype=bool)
    pen_baseline = pen_baseline_signed[pen_free_local_idx]
    pen_face_normals_sel = pen_face_normals[pen_free_local_idx]
    # Final (end-of-optimization) floor per penetration row: bad rows ramp
    # to +tolerance_mm; neighbour rows get a pure non-regression floor.
    pen_final_floor = np.where(pen_is_bad, tolerance_mm, np.minimum(pen_baseline, tolerance_mm))

    n_constraints = n_quality_constraints + n_pen_rows

    print(
        f"  Penetration repair zone: {len(free_nids_list)} free node(s), "
        f"{len(group8_idx)} hex8 + {len(group6_idx)} penta6 + {len(group4_idx)} tet4 + "
        f"{len(quad_idx)} quad4 + {len(tri_idx)} tri3 protected element(s), "
        f"{n_pen_rows} node(s) with an active penetration constraint "
        f"(free_node_layers={free_node_layers}, constraint_layers={constraint_layers})."
    )

    low_bounds = free_orig_coords - max_displacement
    high_bounds = free_orig_coords + max_displacement
    x0 = np.minimum(np.maximum(free_orig_coords, low_bounds), high_bounds).flatten()
    orig_x0 = free_orig_coords.flatten()
    n_free_vars = len(orig_x0)

    dense_jacobian_size = n_constraints * n_free_vars
    use_sparse_solver = dense_jacobian_size >= _DENSE_JACOBIAN_SIZE_THRESHOLD
    print(
        f"  Solver dispatch: n_free_vars={n_free_vars}, n_constraints={n_constraints} "
        f"-> estimated dense Jacobian size={dense_jacobian_size:,} "
        f"(threshold={_DENSE_JACOBIAN_SIZE_THRESHOLD:,}) -> "
        f"using {'trust-constr (sparse)' if use_sparse_solver else 'SLSQP (dense)'}."
    )

    def _zone_coords_from_x(x):
        coords = zone_coords0.copy()
        coords[free_zone_idx] = x.reshape(-1, 3)
        return coords

    def objective(x):
        return float(np.sum((x - orig_x0) ** 2))

    def objective_jac(x):
        return 2.0 * (x - orig_x0)

    def objective_hess(x):
        return 2.0 * sparse_identity(n_free_vars, format='csr')

    if use_sparse_solver:
        bounds = Bounds(orig_x0 - max_displacement, orig_x0 + max_displacement)
    else:
        bounds = [(orig_x0[j] - max_displacement, orig_x0[j] + max_displacement)
                  for j in range(n_free_vars)]

    def _quality_parts(coords, soft_variant):
        """Mirrors repair_inverted_elements/repair_distorted_shells'
        _constraint_parts_hard/_soft -- soft_variant selects the
        softmin-approximated corner-min rows (SLSQP/trust-constr-facing)
        vs. the exact hard rows (baseline/final-check only)."""
        parts = []
        if len(group8_idx):
            if soft_variant:
                v8, g8 = _batched_hexlike_softmin_value_and_grad(coords[group8_idx], 8, group8_beta)
            else:
                v8, g8 = _batched_hexlike_value_and_grad(coords[group8_idx], 8)
            parts.append((v8, g8, group8_idx))
            v8c, g8c = _batched_hexlike_centroid_value_and_grad(coords[group8_idx])
            parts.append((v8c, g8c, group8_idx))
        if len(group6_idx):
            if soft_variant:
                v6, g6 = _batched_hexlike_softmin_value_and_grad(coords[group6_idx], 4, group6_beta)
            else:
                v6, g6 = _batched_hexlike_value_and_grad(coords[group6_idx], 4)
            parts.append((v6, g6, group6_idx))
            v6c, g6c = _batched_hexlike_centroid_value_and_grad(coords[group6_idx])
            parts.append((v6c, g6c, group6_idx))
        if len(group4_idx):
            v4, g4 = _batched_tet_value_and_grad(coords[group4_idx])
            parts.append((v4, g4, group4_idx))
        if len(quad_idx):
            if soft_variant:
                vq, gq = _batched_quad_jacobian_ratio_softmin_value_and_grad(coords[quad_idx], quad_n_ref, quad_beta)
            else:
                vq, gq = _batched_quad_jacobian_ratio_value_and_grad(coords[quad_idx], quad_n_ref)
            parts.append((vq, gq, quad_idx))
        if len(tri_idx):
            vt, gt = _batched_tri3_jacobian_ratio_value_and_grad(coords[tri_idx])
            parts.append((vt, gt, tri_idx))
        return parts

    def constraint_vec(x):
        coords = _zone_coords_from_x(x)
        rows = []
        offset = 0
        for v, g, idx in _quality_parts(coords, soft_variant=True):
            n_rows = len(v)
            rows.append(v - quality_floor_vec[offset:offset + n_rows])
            offset += n_rows
        if n_pen_rows:
            free_coords_now = coords[free_zone_idx[pen_free_local_idx]]
            signed_now = np.einsum('ij,ij->i', free_coords_now - pen_face_centroids[pen_free_local_idx], pen_face_normals_sel)
            rows.append(signed_now - pen_floor_vec)
        if not rows:
            return np.zeros(0)
        return np.concatenate(rows)

    def constraint_jac(x):
        coords = _zone_coords_from_x(x)
        rows_list, cols_list, vals_list = [], [], []
        row_offset = 0
        for v, g, idx in _quality_parts(coords, soft_variant=True):
            r, c, val = _scatter_group_jacobian(g, idx, zone_local_to_freevar, row_offset)
            rows_list.append(r); cols_list.append(c); vals_list.append(val)
            row_offset += len(v)
        if n_pen_rows:
            # Linear constraint: d(signed_dist)/dx = the frozen face
            # normal itself, in that node's own 3 columns only.
            slot_idx = free_zone_idx[pen_free_local_idx].reshape(-1, 1)  # (n_pen_rows, 1), ZONE-local
            slot_grad = pen_face_normals_sel.reshape(-1, 1, 3)
            r, c, val = _scatter_group_jacobian(slot_grad, slot_idx, zone_local_to_freevar, row_offset)
            rows_list.append(r); cols_list.append(c); vals_list.append(val)
            row_offset += n_pen_rows
        if not rows_list:
            empty = coo_matrix((0, n_free_vars))
            return empty.toarray() if not use_sparse_solver else empty.tocsr()
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        vals = np.concatenate(vals_list)
        J = coo_matrix((vals, (rows, cols)), shape=(n_constraints, n_free_vars))
        return J.tocsr() if use_sparse_solver else J.toarray()

    quality_floor_vec = np.concatenate(quality_floor_parts) if n_quality_constraints else np.zeros(0)
    pen_floor_vec = pen_final_floor if n_pen_rows else np.zeros(0)

    # Per-row "real regression, not solver noise" tolerance for the
    # post-solve neighbour-regression check below -- same role/rationale
    # as repair_inverted_elements' own reuse of volume_epsilon for this
    # purpose, just split by family since this function mixes two
    # differently-scaled quality metrics (solid volume vs shell
    # jacobian_ratio) in one flat constraint vector. Row order MUST match
    # quality_floor_parts' own concatenation order above.
    quality_regression_tol_vec = (
        np.concatenate([
            np.full(len(v8_base), quality_volume_epsilon),
            np.full(len(v8c_base), quality_volume_epsilon),
            np.full(len(v6_base), quality_volume_epsilon),
            np.full(len(v6c_base), quality_volume_epsilon),
            np.full(len(v4_base), quality_volume_epsilon),
            np.full(len(quad_base), quality_shell_epsilon),
            np.full(len(tri_base), quality_shell_epsilon),
        ]) if n_quality_constraints else np.zeros(0)
    )

    zone_extent = float(np.ptp(zone_coords0, axis=0).max()) if len(zone_coords0) else 1.0
    jitter_scale = max(zone_extent * 1e-3, 1e-6)
    _self_check_analytic_gradient(constraint_vec, constraint_jac, x0, n_free_vars, n_constraints,
                                   jitter_scale=jitter_scale)

    iteration_state = {'count': 0, 'last_xk': None, 'early_stop_reason': None}

    def _make_stage_callback():
        history = []
        move_history = []
        max_theoretical_move = max_displacement * np.sqrt(3.0)

        def _callback(intermediate_result):
            xk = intermediate_result.x if hasattr(intermediate_result, 'x') else intermediate_result
            iteration_state['count'] += 1
            iteration_state['last_xk'] = xk.copy()
            cvals = constraint_vec(xk)
            worst = float(cvals.min()) if len(cvals) else 0.0
            move = float(np.max(np.abs(xk - orig_x0))) if len(xk) else 0.0
            history.append(worst)
            move_history.append(move)
            if iteration_state['count'] % 1 == 0:
                print(f"    iter {iteration_state['count']:5d}: worst constraint={worst: .6e}, max move={move:.4f}")

            if len(history) >= _FEASIBLE_STALL_WINDOW:
                recent = history[-_FEASIBLE_STALL_WINDOW:]
                recent_moves = move_history[-_FEASIBLE_STALL_WINDOW:]
                if min(recent) >= 0 and (max(recent_moves) - min(recent_moves)) < 1.0e-3:
                    print(
                        f"    -> feasible and stable: every constraint has been satisfied "
                        f"(worst >= 0) and stable (swing < 1.0e-03) over the last "
                        f"{_FEASIBLE_STALL_WINDOW} iterations. Stopping this stage early."
                    )
                    return True

            if len(history) >= _STALL_WINDOW:
                recent = history[-_STALL_WINDOW:]
                if recent[0] != 0 and abs((recent[-1] - recent[0]) / recent[0]) < 0.02 and recent[-1] < 0:
                    print(
                        f"    -> stalled: worst constraint has not improved by more than "
                        f"2% over the last {_STALL_WINDOW} iterations (still violated at "
                        f"{recent[-1]:.4f}). Stopping this stage early."
                    )
                    return True

            if len(move_history) >= _BOUND_SATURATION_WINDOW:
                recent_moves = move_history[-_BOUND_SATURATION_WINDOW:]
                mean_move = float(np.mean(recent_moves))
                worst_now = history[-1]
                if (mean_move >= _BOUND_SATURATION_RATIO * max_theoretical_move
                        and worst_now < -_BOUND_SATURATION_MIN_VIOLATION):
                    print(
                        f"    -> diverging: mean max-move ({mean_move:.4f}) has stayed within "
                        f"90% of the theoretical bound-diagonal limit ({max_theoretical_move:.4f}) "
                        f"over the last {_BOUND_SATURATION_WINDOW} iterations while still "
                        f"meaningfully violated ({worst_now:.4f}) -- this zone/attempt cannot "
                        f"satisfy the constraint within the current displacement bound. "
                        f"Stopping this stage early."
                    )
                    return True
            return False

        return _callback

    # Continuation schedule over the PENETRATION rows only (element
    # quality rows are never ramped -- their floor is already correct via
    # each element's own baseline, see quality_floor_parts above).
    if n_pen_rows and pen_is_bad.any():
        gap = np.maximum(pen_final_floor[pen_is_bad] - pen_baseline[pen_is_bad], 0.0)
        max_gap = float(gap.max())
    else:
        max_gap = 0.0
    n_stages = (
        int(np.clip(np.ceil(max_gap / _PENETRATION_CONTINUATION_STEP_MM), 1, _PENETRATION_MAX_CONTINUATION_STAGES))
        if max_gap > 0 else 1
    )

    print(f"  Running {'trust-constr (sparse Jacobian)' if use_sparse_solver else 'SLSQP (dense Jacobian)'} optimisation...")
    if n_stages > 1:
        print(f"  Worst penetration gap to target: {max_gap:.3f}mm -> using {n_stages} continuation stages.")

    opt_result = None
    for stage in range(1, n_stages + 1):
        alpha = stage / n_stages
        if n_pen_rows and pen_is_bad.any():
            pen_floor_vec = np.where(
                pen_is_bad, pen_baseline + alpha * (pen_final_floor - pen_baseline), pen_final_floor
            )
        is_final_stage = (stage == n_stages)
        if n_stages > 1:
            print(f"  -- continuation stage {stage}/{n_stages} (alpha={alpha:.2f}) --")
        if use_sparse_solver:
            opt_result = _run_trust_constr_stage(
                objective, x0, objective_jac, objective_hess, bounds,
                constraint_vec, constraint_jac, _make_stage_callback(),
                _FINAL_STAGE_MAXITER if is_final_stage else _INTERMEDIATE_STAGE_MAXITER,
                iteration_state,
            )
        else:
            try:
                opt_result = minimize(
                    objective, x0, jac=objective_jac, method='SLSQP', bounds=bounds,
                    constraints=[{'type': 'ineq', 'fun': constraint_vec, 'jac': constraint_jac}],
                    callback=_make_stage_callback(),
                    options={
                        'maxiter': _FINAL_STAGE_MAXITER if is_final_stage else _INTERMEDIATE_STAGE_MAXITER,
                        'ftol': _SLSQP_FTOL, 'disp': is_final_stage,
                    },
                )
            except StopIteration as e:
                last_xk = iteration_state['last_xk']
                if last_xk is None:
                    last_xk = x0
                opt_result = OptimizeResult(
                    x=last_xk, success=False, status=-1,
                    message=f"stopped early via callback: {e}",
                    fun=float(objective(last_xk)), nit=iteration_state['count'],
                )
        x0 = opt_result.x

    final_coords = _zone_coords_from_x(opt_result.x)

    updated_nodes_df = nodes_df[['NodeID', 'x', 'y', 'z']].copy().set_index('NodeID')
    final_free_coords = final_coords[free_zone_idx]
    updated_nodes_df.loc[free_nids_list, ['x', 'y', 'z']] = final_free_coords
    updated_nodes_df = updated_nodes_df.reset_index()

    # Post-repair validation. THREE independent checks (same real-bug
    # rationale as repair_inverted_elements' own n_still_negative/
    # n_neighbor_regressions split -- see that function's docstring):
    #
    # 1) n_still_penetrating: the user's original "must fix" bad-list
    #    (pen_is_bad) penetration rows -- unchanged from before this fix.
    #
    # 2) Element-quality neighbour regressions: EVERY quality row this
    #    function protects (solid volume + shell jacobian_ratio) has NO
    #    "bad" concept here at all -- they are ALWAYS non-regression-only
    #    (see quality_floor_parts above) -- so a diverging/early-stopped
    #    solve could silently push one past its own floor while every
    #    tracked penetrating node still happened to clear. Recomputed via
    #    the EXACT hard (non-softmin) formulas, exactly like
    #    repair_inverted_elements' final_parts / _constraint_parts_hard.
    #
    # 3) Non-bad ("neighbour") penetration rows: a protected node that
    #    was NOT itself on the penetrating list also has its own
    #    non-regression floor (pen_final_floor for ~pen_is_bad rows) --
    #    same blind spot, different constraint family. Uses the exact
    #    same strict '<' convention as n_still_penetrating (no extra
    #    buffer -- this constraint is exactly linear, so there is no
    #    softmin-approximation slack to buffer against, unlike the
    #    quality rows above).
    if n_quality_constraints:
        final_quality_vals = np.concatenate(
            [v for v, g, idx in _quality_parts(final_coords, soft_variant=False)]
        )
        quality_shortfall = quality_floor_vec - final_quality_vals
        n_quality_regressions = int(np.sum(quality_shortfall > quality_regression_tol_vec))
        max_quality_regression = float(quality_shortfall.max())
    else:
        n_quality_regressions = 0
        max_quality_regression = 0.0

    if n_pen_rows:
        final_free_pen_coords = final_coords[free_zone_idx[pen_free_local_idx]]
        final_signed = np.einsum(
            'ij,ij->i', final_free_pen_coords - pen_face_centroids[pen_free_local_idx], pen_face_normals_sel
        )
        # Strict '<' to match detect_soft_hard_penetrations' own convention
        # exactly (depth > tolerance_mm, i.e. signed_dist < tolerance_mm --
        # NOT '<='), so a node that lands EXACTLY on the target boundary
        # (touching, the correct outcome for tolerance_mm=0.0's "just
        # resolve to touching" semantics) is correctly reported as fixed,
        # not falsely flagged as still-penetrating due to an off-by-
        # equality mismatch between this internal check and re-running
        # detection on the repaired mesh (confirmed via a real ground-
        # truth test: a node landing at exactly signed_dist=0.0 was
        # wrongly counted here before this fix, even though a fresh
        # detect_soft_hard_penetrations call on the same repaired mesh
        # correctly did NOT flag it).
        n_still_penetrating = int(np.sum(final_signed[pen_is_bad] < tolerance_mm)) if pen_is_bad.any() else 0
        pen_neighbor_mask = ~pen_is_bad
        if pen_neighbor_mask.any():
            pen_neighbor_shortfall = pen_final_floor[pen_neighbor_mask] - final_signed[pen_neighbor_mask]
            n_pen_neighbor_regressions = int(np.sum(pen_neighbor_shortfall > 0))
            max_pen_neighbor_regression = float(pen_neighbor_shortfall.max())
        else:
            n_pen_neighbor_regressions = 0
            max_pen_neighbor_regression = 0.0
    else:
        n_still_penetrating = 0
        n_pen_neighbor_regressions = 0
        max_pen_neighbor_regression = 0.0

    n_neighbor_regressions = n_quality_regressions + n_pen_neighbor_regressions
    max_neighbor_regression = max(max_quality_regression, max_pen_neighbor_regression)

    displacements = np.linalg.norm(
        opt_result.x.reshape(-1, 3) - orig_x0.reshape(-1, 3), axis=1
    )
    info = {
        'n_free_nodes': len(free_nids_list),
        'n_repair_zone_elements': len(repair_indices),
        'n_penetration_rows': n_pen_rows,
        'n_still_penetrating': n_still_penetrating,
        'n_neighbor_regressions': n_neighbor_regressions,
        'max_neighbor_regression': max_neighbor_regression,
        'max_displacement_achieved': float(displacements.max()) if len(displacements) else 0.0,
        # NOTE the AND-of-ORs structure here (not opt_result.success OR
        # (...)): a solver can report success=True (its OWN soft/
        # approximate constraint tolerance satisfied) while the
        # independent HARD recheck still finds a real neighbor
        # regression -- e.g. a real, non-hypothetical gap between the
        # SLSQP-facing constraint value and this exact post-solve
        # recompute. n_neighbor_regressions == 0 is therefore a
        # mandatory gate, never overridable by opt_result.success alone;
        # opt_result.success is only consulted (via the OR) as a
        # substitute for the n_still_penetrating == 0 check, preserving
        # this fork's trust-constr solver's own well-understood quirk of
        # reporting success=False on essentially every deliberately-
        # early-stopped-but-genuinely-fine run (see repair_inverted_
        # elements' own 'converged' comment for the full story --
        # provably identical to the simpler `S or (X==0 and Y==0)` form
        # in every case except this one, which it deliberately closes).
        'converged': (n_neighbor_regressions == 0) and (n_still_penetrating == 0 or bool(opt_result.success)),
    }
    return updated_nodes_df, info


def build_shell_floor_ceiling(quad_baseline, quad_is_bad, tri_baseline, tri_is_bad,
                               solid_baseline_volumes,
                               band_floor=0.5, band_ceiling=0.7, volume_epsilon=1e-6):
    """
    Build the per-constraint-row (floor, ceiling, baseline) vectors for a
    MIXED shell-repair zone -- the shell analogue of repair_inverted_
    elements' own _build_floor_and_bad_mask, generalized to a TWO-SIDED
    band target (floor AND ceiling) instead of solids' floor-only target,
    and additionally covering neighbouring SOLID elements touched via
    shared nodes (see build_solid_node_adjacency/find_touching_solid_
    elements -- confirmed a REAL, common case on the actual HBM_F/F05
    mesh: shells directly overlay solids, e.g. a tri3 skin over a tet4
    muscle, or a quad4 over a hex8, sharing nodes with them).

    *** WHY A TWO-SIDED BAND, UNLIKE SOLIDS' FLOOR-ONLY TARGET *** (user's
    explicit requirement): a solid's volume/Jacobian has no natural upper
    bound to avoid -- more positive is unambiguously safer, so
    repair_inverted_elements only ever needs volume_epsilon as a MINIMUM.
    Shells are different: the user specifically wants bad shells pushed
    into [0.5, 0.7] as a genuine BAND, not "as high as possible" -- so
    every bad shell constraint row needs BOTH a floor (value >= 0.5) AND
    a ceiling (value <= 0.7) as independent inequality constraints.

    *** NON-REGRESSION GENERALIZATION FOR NEIGHBOUR SHELLS *** (mirrors
    _build_floor_and_bad_mask's own floor = min(baseline, volume_epsilon)
    rule, extended to both sides of a band instead of one side of a
    floor): a neighbour (non-bad) shell pulled into the zone only for
    non-regression protection gets:
        floor_i   = min(baseline_i, band_floor)
        ceiling_i = max(baseline_i, band_ceiling)
    so a neighbour shell that ALREADY sits inside [0.5, 0.7] is held
    there (never forced further, never allowed to drift out); one that
    starts BELOW 0.5 is only required not to drop any lower than it
    already was (exactly like the solid non-regression floor); one that
    starts ABOVE 0.7 (a very "ideal", near-perfect-square/equilateral
    shell) is only required not to rise any higher than it already was --
    this is the natural two-sided analogue of "don't force improvement,
    only forbid regression", now with two directions to regress in.

    *** SOLID NEIGHBOURS: FLOOR-ONLY, NO CEILING *** (identical treatment
    to repair_inverted_elements' own neighbour-solid handling, since
    solids were ALREADY fully repaired in the earlier solid-repair phase
    before shell repair ever begins -- confirmed workflow): a touched
    solid element gets ONLY
        floor_i = min(baseline_volume_i, volume_epsilon)
    -- no ceiling row at all, matching the fact that solids have no
    two-sided target concept anywhere else in this codebase. This is what
    protects an already-fixed solid element's face from being silently
    re-inverted/re-broken as a side effect of moving a shared node to fix
    the shell sitting on top of it.

    Inputs:
        quad_baseline (np.ndarray): shape (Nq,); each quad4 shell's
            baseline jacobian_ratio value (at the zone's ORIGINAL,
            pre-optimization geometry) -- e.g. from
            _batched_quad_jacobian_ratio_value_and_grad's hard variant.
        quad_is_bad (np.ndarray[bool]): shape (Nq,); True for quads on the
            user's "must fix" bad-shell list.
        tri_baseline (np.ndarray): shape (Nt,); each tri3 shell's baseline
            jacobian_ratio (from _batched_tri3_jacobian_ratio_value_and_
            grad).
        tri_is_bad (np.ndarray[bool]): shape (Nt,); True for tris on the
            bad-shell list.
        solid_baseline_volumes (np.ndarray): shape (Ns,); each touched
            neighbour solid element's baseline volume (from the EXISTING
            compute_element_volumes-equivalent batched formulas -- always
            treated as non-bad/floor-only, since solids are already fully
            repaired before shell repair begins).
        band_floor (float): Minimum acceptable jacobian_ratio for a bad
            shell. Default 0.5 (per user's explicit requirement).
        band_ceiling (float): Maximum acceptable jacobian_ratio for a bad
            shell. Default 0.7 (per user's explicit requirement).
        volume_epsilon (float): Non-regression floor value for solid
            neighbours, matching repair_inverted_elements' own parameter
            of the same name/meaning.

    Returns:
        dict: {
            'quad_floor'   : np.ndarray (Nq,),
            'quad_ceiling' : np.ndarray (Nq,),
            'tri_floor'    : np.ndarray (Nt,),
            'tri_ceiling'  : np.ndarray (Nt,),
            'solid_floor'  : np.ndarray (Ns,),  # no 'solid_ceiling' key --
                                                  # solids are floor-only,
                                                  # by design (see above).
        }
        Every array is aligned index-for-index with its corresponding
        input baseline/is_bad array (same row order, same length) -- the
        caller (repair_distorted_shells) is responsible for combining
        these into the actual stacked constraint vector alongside
        whichever value/gradient functions produced quad_baseline/
        tri_baseline/solid_baseline_volumes in the first place.
    """
    if band_floor >= band_ceiling:
        raise ValueError(
            f"band_floor ({band_floor}) must be strictly less than "
            f"band_ceiling ({band_ceiling})."
        )

    quad_floor = np.where(quad_is_bad, band_floor, np.minimum(quad_baseline, band_floor))
    quad_ceiling = np.where(quad_is_bad, band_ceiling, np.maximum(quad_baseline, band_ceiling))

    tri_floor = np.where(tri_is_bad, band_floor, np.minimum(tri_baseline, band_floor))
    tri_ceiling = np.where(tri_is_bad, band_ceiling, np.maximum(tri_baseline, band_ceiling))

    solid_floor = np.minimum(solid_baseline_volumes, volume_epsilon)

    return {
        'quad_floor': quad_floor,
        'quad_ceiling': quad_ceiling,
        'tri_floor': tri_floor,
        'tri_ceiling': tri_ceiling,
        'solid_floor': solid_floor,
    }


def _expand_node_region(seed_nodes, nid_to_elem_indices, element_nodes, layers):
    """
    Expand a node set by element adjacency for a fixed number of layers.
    """
    if layers <= 0 or not seed_nodes:
        return set(seed_nodes)

    expanded = set(seed_nodes)
    frontier = set(seed_nodes)
    for _ in range(layers):
        touched_elem_indices = set()
        for nid in frontier:
            touched_elem_indices.update(nid_to_elem_indices.get(nid, set()))

        next_frontier = set()
        for elem_idx in touched_elem_indices:
            next_frontier.update(element_nodes[elem_idx])

        next_frontier -= expanded
        if not next_frontier:
            break

        expanded.update(next_frontier)
        frontier = next_frontier

    return expanded


def group_bad_elements_into_pockets(elements_bad_df, elements_all_df, expansion_layers):
    """
    Split a (possibly large, possibly whole-model) list of bad elements into
    independent "pockets" -- maximal groups of bad elements that are close
    enough together (by node adjacency) that repairing them could interact,
    versus pockets that are far enough apart on the mesh that they can be
    handed to completely separate repair_inverted_elements() calls with zero
    risk of one pocket's node movement affecting another pocket's elements.

    WHY THIS EXISTS: repair_inverted_elements solves every bad element it is
    given in ONE combined SLSQP call. That is correct regardless of how
    spread out the bad elements are, but SciPy's SLSQP internally scales
    WORSE than linearly with problem size (free-variable/constraint-row
    count). For a real whole-model file with 100+ parts, bad elements are
    typically scattered in many small, mutually-uninfluential clusters (e.g.
    a few bad hexes in part A's corner, a few more in part Z's interior,
    with thousands of unrelated good elements and dozens of other parts in
    between). Solving all of them in one giant combined SLSQP call pays for
    a single large, slow problem for no correctness benefit, when solving
    each pocket as its own small, fast, independent problem gives the exact
    same final result.

    ALGORITHM (union-find over bad elements, one pass, no pairwise O(n^2)
    scan):
      1. For each bad element (by row position in elements_bad_df), expand
         ONLY that element's own node set outward by `expansion_layers`
         topological adjacency rings (same _expand_node_region mechanism
         repair_inverted_elements itself uses to build free/constrained
         zones) to get that element's own local "reach".
      2. Track which bad element first "claims" each node touched by any
         reach computed so far. If a later bad element's reach touches a
         node already claimed by an earlier one, union the two bad elements
         into the same pocket (transitively -- if A shares reach with B, and
         B shares reach with C, all three end up in ONE pocket even if A and
         C never directly overlap).
      3. Every bad element ends up in exactly one pocket; pockets with no
         shared reach to any other pocket are returned as fully independent
         groups.

    CHOOSING expansion_layers: this must be chosen GENEROUSLY -- at least as
    large as the LARGEST free_node_layers + constraint_layers combination
    that any escalation-ladder attempt could ever use for these elements.
    Pockets are computed ONCE, up front, before any repair attempt runs; if
    expansion_layers were too small, two pockets that look independent at
    attempt #1's small zone size could actually start to touch once a later,
    more aggressive escalation attempt widens the zone -- solving them as
    separate problems at that point would be WRONG (each pocket's solve
    would be blind to the other's simultaneous movement of a now-shared
    neighborhood). Too large a value only costs some potential speed-up
    (more bad elements merged into one pocket than strictly necessary) --
    never correctness. When in doubt, err large.

    Inputs:
        elements_bad_df (pd.DataFrame): Candidate bad elements (may span
            many different parts/PIDs if elements_all_df is a merged
            multi-part table from read_multi_part_mesh).
        elements_all_df (pd.DataFrame): The FULL mesh element table (used
            only to build node-adjacency for the reach/expansion step --
            never itself split or modified).
        expansion_layers (int): Number of adjacency rings to expand each
            individual bad element's own node set by, for the overlap test.
            See "CHOOSING expansion_layers" above.

    Returns:
        list[pd.DataFrame]: elements_bad_df split into pockets (each a
            row-subset DataFrame, same columns as the input), sorted by
            descending pocket size (largest/most bad elements first) so a
            caller looping over pockets tackles the biggest, most-likely-
            slowest pocket first. Empty list if elements_bad_df is empty.
    """
    if elements_bad_df.empty:
        return []

    all_element_nodes = _unique_node_lists_fast(elements_all_df)
    nid_to_elem_indices = {}
    for elem_idx, unique_nids in enumerate(all_element_nodes):
        for nid in unique_nids:
            nid_to_elem_indices.setdefault(nid, set()).add(elem_idx)

    bad_node_lists = _unique_node_lists_fast(elements_bad_df)
    n_bad = len(bad_node_lists)

    # Union-find (disjoint-set) over bad-element row positions, with simple
    # path compression (no union-by-rank -- n_bad is expected to be small
    # enough, up to the low thousands, that this is not a bottleneck).
    parent = list(range(n_bad))

    def _find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    # node_owner records which bad element's reach FIRST touched a given
    # node. Any subsequent bad element whose reach touches an already-owned
    # node gets unioned with that node's owner -- this is what merges two
    # bad elements into the same pocket even if their OWN node sets never
    # directly intersect, only their expanded reaches do.
    node_owner = {}
    for idx, nodes in enumerate(bad_node_lists):
        reach = _expand_node_region(
            set(nodes), nid_to_elem_indices, all_element_nodes, expansion_layers
        )
        for nid in reach:
            owner = node_owner.get(nid)
            if owner is None:
                node_owner[nid] = idx
            else:
                _union(idx, owner)

    groups = {}
    for idx in range(n_bad):
        root = _find(idx)
        groups.setdefault(root, []).append(idx)

    pockets = [elements_bad_df.iloc[idxs].copy() for idxs in groups.values()]
    pockets.sort(key=len, reverse=True)
    return pockets


def group_bad_shells_into_pockets(elements_bad_shell_df, elements_all_shell_df,
                                    solid_element_node_lists, node_to_solid_indices,
                                    expansion_layers):
    """
    Shell analogue of group_bad_elements_into_pockets, extended to ALSO
    guarantee no two returned pockets could ever independently touch the
    SAME solid element -- a correctness requirement group_bad_elements_
    into_pockets' own shell-to-shell-only reach does NOT provide (see
    "WHY EXTRA SOLID-AWARENESS IS NEEDED" below). This guarantee is
    specifically what makes it SAFE to solve shell pockets in PARALLEL
    the same way solid pockets already are -- see repair_distorted_
    shells' own protection of touching solids, and the driver script's
    shell parallel-worker path, which both depend on this function (not
    the plain shell-only group_bad_elements_into_pockets) for splitting.

    WHY EXTRA SOLID-AWARENESS IS NEEDED (a real correctness gap, not
    hypothetical): repair_distorted_shells finds and protects every solid
    element that shares a node with a shell pocket's FREE node set (see
    build_solid_node_adjacency/find_touching_solid_elements), holding
    each one to a non-regression floor. If two DIFFERENT shell pockets
    each independently touch a DIFFERENT node of the SAME solid element
    (entirely possible -- a shell-to-shell-only reach has no way to see
    this coming), and those two pockets were solved in PARALLEL from the
    same original mesh (as solid pockets already safely are), each
    pocket's optimizer would only ever validate ITS OWN change to that
    solid in isolation -- neither one would ever see the OTHER pocket's
    simultaneous move of a different node on the SAME solid element.
    Merging both results afterward could silently produce a solid element
    whose COMBINED (both moves applied together) shape was never actually
    validated by either optimizer, defeating the whole non-regression
    guarantee. Solid pockets do not have this problem because their own
    pocket-splitting (group_bad_elements_into_pockets) already uses
    solid-to-solid adjacency directly; shells need this separate,
    solid-aware version because shell-to-shell adjacency alone cannot see
    this risk.

    ALGORITHM (same union-find core as group_bad_elements_into_pockets,
    with one crucial addition): for each bad shell, after expanding its
    own shell-to-shell reach by `expansion_layers` (exactly as before),
    ALSO find every solid element that reach touches (via find_touching_
    solid_elements) and add EVERY ONE OF THAT SOLID'S OWN NODES (not just
    the specific node the reach happened to touch) into the claimed node
    set used for the union-find step below. This is what makes two shell
    pockets that would touch even DIFFERENT nodes of the SAME solid
    element get correctly merged into ONE combined pocket: the first
    pocket to touch ANY node of that solid "claims" ALL of that solid's
    nodes as part of its reach, so a second pocket touching any OTHER
    node of the same solid is detected as already-owned and unioned in --
    exactly as if the two shells had directly shared a node themselves.

    Inputs:
        elements_bad_shell_df (pd.DataFrame): Candidate bad shells (may
            span many different parts/PIDs).
        elements_all_shell_df (pd.DataFrame): The FULL shell element
            table (used only to build shell-to-shell node-adjacency for
            the reach/expansion step -- never itself split or modified).
        solid_element_node_lists, node_to_solid_indices: From
            build_solid_node_adjacency(solid_elements_all_df) -- the SAME
            precomputed solid adjacency repair_distorted_shells itself
            uses; passed straight through, never rebuilt here.
        expansion_layers (int): Same meaning/guidance as group_bad_
            elements_into_pockets' own parameter -- must be at least as
            large as the largest free_node_layers + constraint_layers
            combination any shell escalation-ladder attempt could reach
            (see that function's own "CHOOSING expansion_layers" section
            for the full rationale, which applies identically here).

    Returns:
        list[pd.DataFrame]: elements_bad_shell_df split into pockets
            (each a row-subset DataFrame, same columns as the input),
            sorted by descending pocket size. Empty list if elements_
            bad_shell_df is empty. GUARANTEE: no two returned pockets
            share a node (as before) AND no two returned pockets ever
            touch the same solid element (new) -- making it safe to
            solve every pocket fully independently, including in
            parallel, exactly like group_bad_elements_into_pockets'
            own guarantee for solids.
    """
    if elements_bad_shell_df.empty:
        return []

    all_shell_nodes = _unique_node_lists_fast(elements_all_shell_df)
    nid_to_shell_indices = {}
    for elem_idx, unique_nids in enumerate(all_shell_nodes):
        for nid in unique_nids:
            nid_to_shell_indices.setdefault(nid, set()).add(elem_idx)

    bad_node_lists = _unique_node_lists_fast(elements_bad_shell_df)
    n_bad = len(bad_node_lists)

    # Union-find (disjoint-set) over bad-shell row positions -- identical
    # mechanics to group_bad_elements_into_pockets' own union-find.
    parent = list(range(n_bad))

    def _find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    node_owner = {}
    for idx, nodes in enumerate(bad_node_lists):
        reach = _expand_node_region(
            set(nodes), nid_to_shell_indices, all_shell_nodes, expansion_layers
        )
        # Extend the reach with every node of every solid element this
        # reach touches -- see docstring's ALGORITHM section for why the
        # FULL solid node set (not just the touched node) must be
        # included here, not merely the specific node reach happened to
        # land on.
        touching_solid_idx = find_touching_solid_elements(reach, node_to_solid_indices)
        for solid_idx in touching_solid_idx:
            reach.update(solid_element_node_lists[solid_idx])

        for nid in reach:
            owner = node_owner.get(nid)
            if owner is None:
                node_owner[nid] = idx
            else:
                _union(idx, owner)

    groups = {}
    for idx in range(n_bad):
        root = _find(idx)
        groups.setdefault(root, []).append(idx)

    pockets = [elements_bad_shell_df.iloc[idxs].copy() for idxs in groups.values()]
    pockets.sort(key=len, reverse=True)
    return pockets


def _self_check_analytic_gradient(constraint_vec, constraint_jac, x0,
                                   n_free_vars, n_constraints,
                                   jitter_scale=1e-3,
                                   sample_size=5, rtol=1e-3, seed=0):
    """
    Runtime safety net for the hand-derived analytic constraint Jacobian
    used by repair_inverted_elements: spot-checks a random sample of
    (constraint row, free variable column) entries against a central
    finite-difference estimate on the SAME zone/grouping/scatter machinery
    actually used during optimization (not just the underlying math in
    isolation, which was separately verified during development to ~1e-10).

    This protects against integration bugs (wrong slot order, wrong repeat
    pattern, wrong scatter target column) even though the closed-form
    formulas themselves are correct, since a bug in the surrounding batching/
    indexing code could still silently feed SciPy a wrong gradient.

    The check point is x0 plus a small random jitter (jitter_scale), NOT x0
    itself. Locally regular/symmetric mesh regions (e.g. an extruded block
    of near-identical hexes) can make several of the sampled corner
    Jacobians tie almost exactly, which makes _batched_hexlike_value_and_
    grad's per-element argmin-corner selection -- a genuine subgradient of
    a min() function -- fall right on a non-smooth kink. A finite
    difference straddling such a kink can land on a different corner than
    the analytic one-sided value and disagree sharply despite BOTH being
    individually valid, which is not a bug. Jittering off that measure-zero
    tie set avoids false alarms; a real indexing/formula bug is not
    tie-dependent and still reproduces at the jittered point.

    Raises:
        RuntimeError: If the sampled relative error exceeds `rtol`, so a
            broken gradient is never silently trusted by the optimizer.
    """
    if n_constraints == 0 or n_free_vars == 0:
        return

    rng = np.random.default_rng(seed)
    sample_rows = rng.choice(n_constraints, size=min(sample_size, n_constraints), replace=False)
    sample_cols = rng.choice(n_free_vars, size=min(sample_size, n_free_vars), replace=False)

    x_check = x0 + rng.normal(scale=jitter_scale, size=x0.shape)

    analytic_J = constraint_jac(x_check)
    eps = 1e-6
    max_rel_err = 0.0

    for col in sample_cols:
        x_p = x_check.copy(); x_p[col] += eps
        x_m = x_check.copy(); x_m[col] -= eps
        fd = (constraint_vec(x_p) - constraint_vec(x_m)) / (2.0 * eps)
        for row in sample_rows:
            # constraint_jac may now return a scipy.sparse matrix (see
            # this fork's constraint_jac docstring) instead of a dense
            # ndarray -- (row, col) scalar indexing works on both, but
            # sparse indexing can return a numpy scalar wrapped
            # differently depending on scipy version, so float() it
            # explicitly to guarantee a plain Python float for the abs()/
            # max() comparisons below.
            a = float(analytic_J[row, col])
            f = fd[row]
            denom = max(abs(a), abs(f), 1e-8)
            max_rel_err = max(max_rel_err, abs(a - f) / denom)

    if max_rel_err > rtol:
        raise RuntimeError(
            "Analytic constraint Jacobian failed its runtime self-check "
            f"against finite differences (max relative error {max_rel_err:.3e} "
            f"> tolerance {rtol:.1e}). Aborting rather than trusting a "
            "possibly-incorrect gradient."
        )

    print(f"  Analytic gradient self-check passed (max rel. error {max_rel_err:.2e}).")


# Continuation (staged-target) schedule constants for repair_inverted_
# elements: a severely inverted element (e.g. baseline Jacobian -30) cannot
# be pushed straight to +volume_epsilon in a single SLSQP call -- that is
# far outside the region where SLSQP's internal linearized QP subproblem
# is a valid local approximation, and reliably fails immediately ("Positive
# directional derivative for linesearch", zero movement) on real meshes.
# _CONTINUATION_STEP caps how much any one stage may raise the WORST
# bad-element's floor; _MAX_CONTINUATION_STAGES bounds worst-case runtime
# for pathologically deep inversions. Intermediate stages use a smaller
# maxiter (they only need to make progress before handing off to the next,
# closer target); the FINAL stage (the true, user-facing target) gets the
# full budget.
_CONTINUATION_STEP = 2.0
_MAX_CONTINUATION_STAGES = 20
_INTERMEDIATE_STAGE_MAXITER = 150
_FINAL_STAGE_MAXITER = 1000

# Shell analogue of _CONTINUATION_STEP: repair_distorted_shells' bad-shell
# jacobian_ratio rows are ramped from their own baseline to the true
# [band_floor, band_ceiling] target (see that function's docstring for why
# BOTH sides of the band can need ramping, unlike solids' floor-only case).
# The jacobian_ratio metric lives on an O(1) scale (roughly -1 to 1 in
# practice), not solids' O(volume) scale, so this step is sized to the
# band's own width (0.2 by default) rather than reusing _CONTINUATION_STEP
# (2.0) verbatim -- a gap that size would almost always collapse to a
# single stage even for the worst real bad shells observed (~0.84 max
# gap in HBM_F/F05), which is fine, but a smaller step keeps the ladder
# meaningful if a future band width or a worse mesh ever needs more than
# one stage.
_SHELL_CONTINUATION_STEP = 0.3
# SLSQP's ftol governs precision of the OBJECTIVE (sum of squared node
# displacement) between iterations, not the constraint satisfaction margin
# directly -- but a very tight ftol (e.g. 1e-9) was observed on the real
# mesh to burn 100+ extra iterations purely refining the worst constraint
# from ~1e-5 down to ~1e-10 past the point of any practical benefit (target
# is volume_epsilon=1e-3 on a mesh with ~10-unit-scale elements, so ~1e-6
# headroom is already 1000x tighter than needed). 1e-6 cut this
# "polishing past the point of usefulness" substantially in testing while
# still leaving every constraint comfortably satisfied.
# NOTE: _SLSQP_FTOL itself is kept here (unused by this fork's trust-constr
# solver) purely for reference/comparison against the reference
# implementation's tuning rationale above -- see _TRUST_CONSTR_* below for
# the tolerances this fork's solver actually uses.
_SLSQP_FTOL = 1e-6

# trust-constr's convergence criteria are NOT the same quantities as
# SLSQP's ftol (see above) -- there is no direct equivalent, so these are
# a fresh, conservative starting point (not yet tuned against a completed
# real-cluster run at the time this fork was built) rather than a
# principled conversion of the SLSQP value:
#   gtol        : first-order KKT (Lagrangian-gradient) optimality
#                 tolerance -- SciPy's own default is 1e-8; loosened
#                 slightly here in the same spirit as _SLSQP_FTOL's
#                 rationale (target margin is volume_epsilon=1e-3 on a
#                 ~10-unit-scale mesh, so extreme first-order precision is
#                 not needed in practice).
#   xtol        : trust-region step-size tolerance (stops when successive
#                 iterates stop moving meaningfully) -- kept at SciPy's
#                 own default (1e-8); tighter than gtol/barrier_tol
#                 deliberately, since a genuinely stalled step (not just a
#                 loose KKT residual) is a stronger signal that no further
#                 useful progress is possible.
#   barrier_tol : interior-point barrier subproblem tolerance -- matched
#                 to gtol for consistency; trust-constr's barrier
#                 parameter roughly governs how tightly the inequality
#                 constraints (our volume/Jacobian floor) are enforced
#                 near the boundary of feasibility.
# IMPORTANT: these are a reasonable starting point, not a validated final
# choice -- recalibrate against real timing/convergence behavior once this
# fork has been run against actual production pockets on the cluster (the
# whole reason this fork exists: a pocket SLSQP could not even complete
# one iteration on).
_TRUST_CONSTR_GTOL = 1e-6
_TRUST_CONSTR_XTOL = 1e-8
_TRUST_CONSTR_BARRIER_TOL = 1e-6

# Stall detection / early exit for a SINGLE continuation-stage SLSQP call.
# On a harder real mesh (a bigger/more severely tangled bad-element region),
# SLSQP can reach a genuine local KKT point of the CURRENT stage's
# linearized problem where the worst constraint stops improving entirely
# (confirmed via a live diagnostic: gradient magnitude ~7 at the stalled
# row, i.e. NOT a flat/degenerate region, and NO free variable at its
# max_displacement bound -- so it is a real, structural local-optimum
# tension between this row and other active constraints sharing the same
# free nodes, not a trivial bug) while still having 100+ maxiter budget
# left -- silently burning many minutes per wasted iteration on a large
# zone. Rather than exhausting the full per-stage iteration budget once
# this happens, the progress callback watches a rolling window of the
# worst-constraint value and raises StopIteration (scipy's SLSQP driver
# catches this from a callback and returns the last iterate with
# success=False -- verified empirically) once the window's relative swing
# drops below a small threshold WHILE the violation is still meaningfully
# non-zero (this second guard avoids falsely stopping late-stage
# fine-polishing near zero, which legitimately fluctuates at a tiny,
# already-good-enough scale). The next continuation stage (or the
# escalation ladder's next, larger-zone attempt) then picks up from
# wherever this stage's best-effort point left off.
_STALL_WINDOW = 10
_STALL_MIN_VIOLATION = 0.01
_STALL_RELATIVE_SWING = 0.02

# EARLY-EXIT ONCE FEASIBLE AND STABLE (this fork's trust-constr-specific
# addition -- NOT present in the reference SLSQP implementation, which
# does not need it). Discovered from a real measurement: on a SMALL,
# already-validated problem (SharedNodes, 165 free nodes -- SLSQP solves
# this in well under a minute), trust-constr's constraint became FEASIBLE
# (worst constraint >= 0, i.e. every repair-zone element already clears
# its target) within the first ~5 iterations of each continuation stage,
# but then kept iterating for 50-100+ MORE iterations per stage (182.5s
# total, vs SLSQP's few seconds) -- because trust-constr is an
# interior-point method that keeps refining toward its OWN internal
# optimality criteria (gtol/xtol/barrier_tol; see those constants above)
# even once our actual real-world goal (every constraint safely >=
# volume_epsilon) is already met and stable. That extra "polishing" buys
# nothing we need -- exactly the same "stop chasing precision we don't
# need" rationale as _SLSQP_FTOL's own comment above, just for a
# DIFFERENT solver's DIFFERENT (much tighter) default stopping criteria.
# The callback below therefore also stops early once every constraint has
# been simultaneously satisfied (worst >= 0) for a full rolling window
# AND that window's absolute swing has settled below a small threshold --
# using an ABSOLUTE (not relative) swing bound here, unlike
# _STALL_RELATIVE_SWING above, since `worst` can itself be arbitrarily
# close to zero in this feasible regime, which would make a relative
# comparison degenerate.
_FEASIBLE_STALL_WINDOW = 10
_FEASIBLE_STALL_ABS_SWING = 1e-3

# DIVERGENCE / BOUND-SATURATION early-stop (this fork's addition, found
# from a REAL production run -- "Run3" -- on the actual HBM mesh, not a
# hypothetical). Two of twelve pockets (87 and 74 bad elements, both
# routed to the SLSQP branch) never stalled in the _STALL_WINDOW sense
# above -- their worst-constraint value kept swinging by huge amounts
# every iteration (e.g. -2.5 -> -13 -> -180 -> -1148 over ~40 iterations),
# so the small-swing stall check never triggered -- while `max move`
# (the largest single free-node displacement norm) repeatedly approached
# 13.8564, which is EXACTLY max_displacement * sqrt(3): free nodes kept
# getting pushed toward all 3 axes' bounds simultaneously. This is the
# textbook signature of an infeasible bound-constrained subproblem: the
# solver keeps pushing free nodes as far as the box allows, trying (and
# failing) to satisfy a constraint that has no feasible point within the
# current (too-tight, attempt-1) zone -- burning through the FULL maxiter
# budget rather than escalating to a larger/less-constrained attempt.
# Each runaway attempt tied up a worker slot for a long time, and shortly
# after, FOUR concurrent pockets (including two healthy ones making good
# progress, confirmed by their own logs) all stopped writing within the
# same 4-second window -- consistent with the OS/cgroup OOM-killer
# sweeping multiple sibling processes at once once memory pressure from
# the runaway pockets became severe enough.
# This check is a SEPARATE, independent condition from the stall check
# above: it looks at `move` (a fixed, dimensionally meaningful quantity --
# max_displacement is known at solve time, unlike the constraint's own
# problem-dependent numeric scale) rather than trying to threshold the
# constraint's own (highly variable-magnitude) value. Declares divergence
# once `move`'s MEAN over a rolling window has stayed at/near the
# theoretical worst-case diagonal bound (max_displacement * sqrt(3)) WHILE
# the constraint remains meaningfully violated. MEAN (not a strict "every
# single iteration" min) was deliberately chosen after replaying pocket
# 2's own real recorded iteration log through both formulations offline:
# `move` genuinely oscillates a bit even while diverging (e.g. dipping to
# ~11.7 between runs of ~13.86), so requiring EVERY iteration in the
# window to individually clear a near-100% threshold fired too late (or
# not at all over the recorded data); the mean-based version reliably
# fired by iteration ~37 on that same real data -- well before the
# iteration-70 point where the process was actually killed -- while still
# requiring sustained (not single-iteration-fluke) proximity to the
# bound. This combination (meaningfully violated + persistently near the
# displacement bound on average) should never occur during genuine,
# healthy convergence: every previously-validated successful pocket
# settles to a `move` well below this theoretical maximum long before
# finishing (see Run3's own pockets 1/3/5/6 logs). Bailing out fast here
# hands off to the escalation ladder's next attempt (larger free/
# constraint layers, more displacement room, boundary growth) far sooner
# than exhausting the iteration budget would, reducing both wall-clock
# time and the peak-memory window that appears to have triggered the
# mass-kill.
_BOUND_SATURATION_WINDOW = 10
_BOUND_SATURATION_RATIO = 0.90   # mean(move) over the window >= ratio * max_displacement*sqrt(3)
_BOUND_SATURATION_MIN_VIOLATION = 0.1  # still meaningfully violated, not just noisy-near-zero

# HYBRID DISPATCH (per user request): this fork's original purpose was
# tackling pockets too large for SLSQP's dense-Jacobian factorization cost
# (see repair_inverted_elements' docstring below for the full discovery).
# But paying trust-constr's per-iteration overhead (interior-point method,
# looser default stopping criteria even after the _FEASIBLE_STALL_WINDOW
# fix above) on EVERY pocket -- including ones SLSQP already solves in
# well under a minute -- is wasteful. Each continuation-stage solve now
# picks its solver based on an estimate of the reference (SLSQP) dense
# Jacobian's element count (n_constraints * n_free_vars) -- the quantity
# directly responsible for SLSQP's per-iteration cost, since its Fortran
# internals factor/solve using the FULL dense matrix every iteration
# regardless of true sparsity.
# Two REAL measurements bracket this threshold (no intermediate sizes
# measured yet):
#   - SharedNodes (validated OK with SLSQP, well under a minute):
#     n_free_vars=495, n_constraints=1252 -> dense size ~= 619,740
#   - Run2's stalled 218-element pocket (100% CPU confirmed busy, SLSQP
#     made ZERO progress on even continuation stage 1's first iteration):
#     n_free_vars=3309, n_constraints=4128 -> dense size ~= 13,659,552
# 2,000,000 is picked as a conservative starting point roughly in between
# these two real data points, clearly documented as an ESTIMATE pending
# recalibration against a broader range of real production pocket sizes
# on the cluster -- the whole reason this fork exists.
_DENSE_JACOBIAN_SIZE_THRESHOLD = 2_000_000

# NUMERICAL-OVERFLOW SAFETY NET FOR TRUST-CONSTR (real user report: a
# large real shell pocket got stuck partway through its LAST escalation
# attempt, with the console showing
#   RuntimeWarning: overflow encountered in scalar multiply
#     discriminant = b*b - 4*a*c
# from scipy's OWN internal trust-region step calculation, deep inside
# `scipy/optimize/_trustregion_constr/qp_subproblem.py`). This happens
# INSIDE a single call to `scipy.optimize.minimize(..., method=
# 'trust-constr', ...)` -- BEFORE control ever returns to our own
# per-iteration callback. This is a CRITICAL gap the existing stall/
# feasible/diverging early-stop checks (_STALL_WINDOW,
# _FEASIBLE_STALL_WINDOW, _BOUND_SATURATION_WINDOW -- all callback-based)
# CANNOT catch, since they only ever run BETWEEN scipy's own outer
# iterations: if scipy's internal QP subproblem never successfully
# completes even one outer iteration once corrupted by this overflow, our
# callback may never fire again -- a genuine "stuck forever" failure mode
# on a sufficiently large/ill-conditioned pocket, not just "slow".
#
# FIX: wrap ONLY the `minimize(..., method='trust-constr', ...)` call in
# a `warnings.catch_warnings()` block that promotes JUST this specific
# overflow message to a raised exception, then fall back to the last-
# known-good iterate (iteration_state['last_xk'], recorded by the
# trust-constr callback on every iteration it DID manage to complete --
# or x0 if the overflow struck before even one callback call), building a
# synthetic OptimizeResult exactly like the SLSQP branch's existing
# StopIteration handler already does. This converts an UNBOUNDED, "may
# never return" failure into a BOUNDED, "this stage/attempt failed
# cleanly, hand off to the next one" failure -- the same graceful-
# degradation contract every other early-stop condition already provides.
_TRUST_CONSTR_OVERFLOW_MESSAGE = "overflow encountered in scalar multiply"


def _run_trust_constr_stage(objective, x0, objective_jac, objective_hess, bounds,
                             constraint_vec, constraint_jac, callback, maxiter,
                             iteration_state):
    """
    Run ONE trust-constr continuation-stage solve with the numerical-
    overflow safety net described above -- shared by repair_inverted_
    elements and repair_distorted_shells so the fallback/logging behavior
    can never drift between the two.

    Returns a scipy OptimizeResult -- either trust-constr's own normal
    return value, or (if the overflow guard tripped) a synthetic one
    built from iteration_state['last_xk'] (or x0), with success=False and
    a message explaining what happened, mirroring the SLSQP branch's own
    StopIteration fallback shape.
    """
    from scipy.optimize import minimize, NonlinearConstraint, OptimizeResult
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'error', message=_TRUST_CONSTR_OVERFLOW_MESSAGE, category=RuntimeWarning
            )
            return minimize(
                objective,
                x0,
                jac=objective_jac,
                hess=objective_hess,
                method='trust-constr',
                bounds=bounds,
                constraints=[NonlinearConstraint(constraint_vec, 0, np.inf, jac=constraint_jac)],
                callback=callback,
                options={
                    'maxiter': maxiter,
                    'gtol': _TRUST_CONSTR_GTOL,
                    'xtol': _TRUST_CONSTR_XTOL,
                    'barrier_tol': _TRUST_CONSTR_BARRIER_TOL,
                    'sparse_jacobian': True,
                    'verbose': 0,
                },
            )
    except RuntimeWarning as e:
        last_xk = iteration_state['last_xk']
        if last_xk is None:
            last_xk = x0
        print(
            f"    -> NUMERICAL OVERFLOW detected inside trust-constr's internal "
            f"solve ({e}) -- this pocket/zone is large or ill-conditioned enough "
            f"to exceed float64 range. Stopping this stage early (falling back to "
            f"the last good iterate at iteration {iteration_state['count']}) and "
            f"handing off to the next stage/attempt, exactly like a stall/"
            f"feasible/diverging early-stop -- see _TRUST_CONSTR_OVERFLOW_MESSAGE's "
            f"own comment for why this can't be caught by the normal callback path."
        )
        return OptimizeResult(
            x=last_xk, success=False, status=-1,
            message=f"stopped early: numerical overflow in trust-constr ({e})",
            fun=float(objective(last_xk)), nit=iteration_state['count'],
        )


def repair_inverted_elements(elements_bad_df, elements_all_df, nodes_df,
                              max_displacement=2.0, volume_epsilon=1e-6,
                              free_node_layers=2, constraint_layers=2,
                              boundary_growth=0.0, growth_stiffness=0.25):
    """
#TODO: TEST, THIS WAS CREATED BY COPILOT

    Repair inverted (negative-volume) solid elements by adjusting node
    positions using scipy SLSQP constrained nonlinear optimization.

    Nodes belonging to inverted elements are always free variables. The free
    set can optionally be expanded by topological layers so interior failures
    can be fixed by coordinated motion of nearby support nodes, not only by
    moving the failed element nodes themselves.

    The constrained zone is similarly layer-expanded around free nodes so the
    optimizer enforces positive Jacobian/volume on a neighborhood, reducing
    the chance of creating new nearby tangles.

    For deeply interior bad hexes, boundary_growth can bias exterior boundary
    nodes in the free set to move outward from the bad-region centroid, which
    creates additional room for interior untangling while preserving all IDs.

    Performance: earlier versions of this function (a) rebuilt a copy of the
    ENTIRE mesh's coordinate array inside every single per-element constraint
    closure, and (b) registered one scalar SciPy constraint dict per
    repair-zone element with no analytic gradient, forcing SLSQP to
    finite-difference each element's constraint separately (an added factor
    of ~n_free_variables calls per optimizer iteration). Both made runtime
    scale with the FULL part's node/element count, not the repair zone's.
    This version (1) restricts every array to the repair-zone-local node
    subset, and (2) evaluates all repair-zone constraints in one batched,
    analytically-differentiated vector call (see
    _batched_hexlike_value_and_grad / _batched_tet_value_and_grad), so cost
    now scales with repair-zone size only. A runtime self-check
    (_self_check_analytic_gradient) cross-validates the analytic gradient
    against finite differences before optimization starts.

    *** SPARSE/TRUST-CONSTR SOLVER (this file's fork-specific change) ***
    The reference (non-sparse) version of this function hands SciPy's
    SLSQP a DENSE constraint Jacobian (shape n_constraints x n_free_vars)
    every iteration, even though the underlying constraint structure is
    naturally sparse -- each constraint row (one repair-zone element's
    volume/Jacobian check) only ever depends on that element's own ~8
    corner nodes (~24 free variables), regardless of how many thousands of
    total free variables exist in the whole pocket. SLSQP's Fortran
    implementation requires a dense Jacobian and internally factors/solves
    using the FULL dense matrix every iteration -- cost that scales far
    worse than linearly with problem size. This was measured directly on a
    real production pocket (1103 free nodes, 4128 constrained elements,
    i.e. a (4128 x 3309) dense Jacobian, ~13.7 million entries): SLSQP made
    ZERO progress -- not even one completed iteration logged -- after many
    CPU-minutes, while smaller pockets (hundreds of free nodes) on the same
    mesh converged in well under a minute.

    This version instead builds the SAME analytically-correct Jacobian
    values in SPARSE (CSR) format -- see constraint_jac below, which now
    returns a scipy.sparse matrix instead of calling `.toarray()` on it --
    and uses `scipy.optimize.minimize(..., method='trust-constr', ...)`,
    which accepts a sparse constraint Jacobian natively and performs
    sparse matrix-vector operations whose cost scales with the TRUE number
    of non-zero entries (a small, fixed ~24 per constraint row), not the
    full dense size. The minimized objective (sum of squared node
    displacement, optionally plus a boundary-growth term) is a pure
    quadratic in x, so its Hessian is EXACT, CONSTANT, and diagonal
    (2*Identity, or 2*(1+growth_stiffness)*Identity with boundary growth
    active) -- see objective_hess below -- handed to trust-constr directly
    rather than requiring an iterative quasi-Newton approximation for the
    objective (SLSQP itself does not use/need a Hessian at all, since it
    solves a sequence of linearized QP subproblems -- this is a genuinely
    different algorithm, not merely a re-tuned SLSQP). The CONSTRAINTS'
    contribution to the Lagrangian Hessian is left to SciPy's own BFGS
    quasi-Newton approximation (NonlinearConstraint's default), which is
    standard practice and avoids needing to hand-derive second derivatives
    of the volume/Jacobian formulas.

    *** HYBRID SOLVER DISPATCH (per-pocket, this file's fork-specific
    addition) *** Despite the above, trust-constr is NOT strictly better
    across the board: on an already-validated SMALL case (SharedNodes, 165
    free nodes), it took ~4x longer than SLSQP even after tuning its early-
    stop behaviour (see _FEASIBLE_STALL_WINDOW), because its interior-point
    method carries real per-iteration overhead that dense SLSQP's much
    simpler QP subproblem does not. Since this fork's raison d'etre (huge
    pockets SLSQP cannot progress on at all) and small validated pockets
    (where SLSQP is proven fast) can BOTH show up in the same run's list of
    pockets, EVERY continuation-stage solve now independently picks its
    solver based on an estimate of the reference dense Jacobian's element
    count (n_constraints * n_free_vars -- see _DENSE_JACOBIAN_SIZE_
    THRESHOLD's comment for the two real data points bracketing the
    threshold): pockets below the threshold use the SAME SLSQP call shape
    as the reference implementation (dense Jacobian, list-of-tuples
    bounds, StopIteration-based callback, no Hessian); pockets at/above it
    use this fork's trust-constr solver (sparse Jacobian, Bounds object,
    bool-returning callback, exact Hessian). All surrounding logic (zone
    construction, floor/continuation schedule, ID preservation) is IDENTICAL
    regardless of which branch a given pocket takes.

    All surrounding logic -- zone construction, free/constrained node
    expansion, the non-regression floor, the bad-vs-neighbour element
    distinction, and the continuation-stage schedule for severely inverted
    elements -- is UNCHANGED from the reference implementation; only the
    solver backend inside the continuation loop differs. Node/element ID
    preservation and every other correctness guarantee are identical BY
    CONSTRUCTION, since the surrounding code that builds/reads back
    updated_nodes_df is untouched.

    Validity constraint: uses _min_valid_jacobian for every repair-zone
    element, which reuses the SAME reduced-integration volume formulas as
    compute_element_volumes (tet4/penta6/hex8) AND the SAME corner-Jacobian
    sampling as compute_element_distortion (penta6/hex8). This means the
    optimizer is constrained to avoid BOTH a negative centroid volume
    (inversion) AND a local corner sign-change (bow-tie tangling), unlike
    the previous version of this function, which only checked a planar
    tet-decomposition volume and could leave the repaired mesh with a
    deceptively "positive volume" but still-tangled element.

    Non-regression floor for neighbour elements: only elements present in
    elements_bad_df are held to the strict volume_epsilon target. Every
    other repair-zone element (pulled in solely to keep the moving free
    nodes from creating a NEW tangle nearby) is instead constrained to
    "no worse than its own current value" -- i.e. its floor is
    min(baseline_value, volume_epsilon), not volume_epsilon outright. Real
    biomechanical/anatomical hex meshes commonly have plenty of already-
    working elements whose min-corner-Jacobian is small or already negative
    (see is_distorted in compute_element_distortion) despite a perfectly
    valid, positive centroid volume; forcing ALL of those to also clear
    volume_epsilon is (a) outside this function's job -- fixing unrelated,
    pre-existing element quality is not what was asked -- and (b) was found
    to make the constraint set infeasible right at the start for real
    meshes, which starved SLSQP of any descent direction at all (immediate
    "Singular matrix E in LSQ subproblem" / "Positive directional
    derivative" failures with zero movement), not just slow convergence.

    Node ordering convention matches compute_element_volumes:
        Tet4  : n1-n2-n3 base CCW from below, n4 apex.
        Penta6: n1-n2-n3-n4 quad base (CCW from below); n5 ridge point over
                n1-n2, n6 ridge point over n3-n4 (n6 repeated in NID6-8).
        Hex8  : n1-n4 bottom CCW, n5-n8 top; verticals n1-n5 .. n4-n8.

    Inputs:
        elements_bad_df  (pd.DataFrame): Inverted elements from read_elements_from_kfile.
        elements_all_df  (pd.DataFrame): All elements in the part (bad + neighbours).
        nodes_df         (pd.DataFrame): Node table from read_nodes_from_kfile;
                                         columns NodeID, x, y, z.
        max_displacement (float): Max movement allowed per coordinate axis for any
                                  free node (same units as coordinates). Default 2.0.
        volume_epsilon   (float): Minimum acceptable signed volume after repair.
                                  Default 1e-6.
        free_node_layers (int): Number of adjacency layers used to expand the
                                free-node set beyond bad-element nodes.
        constraint_layers (int): Number of adjacency layers used to expand the
                                 constrained validation region around free nodes.
        boundary_growth  (float): Outward pre-bias magnitude applied to free
                                  nodes that are also on the outer part
                                  boundary. Zero disables outward bias.
        growth_stiffness (float): Relative objective weight for staying close
                                  to the outward-bias target when
                                  boundary_growth > 0.

    Returns:
        tuple:
            - updated_nodes_df (pd.DataFrame): Repaired node table (NodeID, x, y, z).
            - result (OptimizeResult): Full scipy result object.
            - info (dict): Summary with keys 'n_free_nodes',
              'n_repair_zone_elements', 'n_still_negative',
              'max_displacement_achieved', 'converged'.
    """
    from scipy.optimize import minimize, Bounds, OptimizeResult
    from scipy.sparse import coo_matrix, identity as sparse_identity

    if max_displacement <= 0:
        raise ValueError("max_displacement must be > 0.")
    if volume_epsilon < 0:
        raise ValueError("volume_epsilon must be >= 0.")
    if free_node_layers < 0 or constraint_layers < 0:
        raise ValueError("free_node_layers and constraint_layers must be >= 0.")
    if boundary_growth < 0:
        raise ValueError("boundary_growth must be >= 0.")
    if growth_stiffness < 0:
        raise ValueError("growth_stiffness must be >= 0.")

    # Seed nodes: all unique nodes referenced by inverted elements. Uses the
    # fast NumPy-array-based row scan (not DataFrame.iterrows()) since this
    # can run on tens of thousands of elements per escalation-ladder attempt.
    bad_element_node_lists = _unique_node_lists_fast(elements_bad_df)
    bad_nids_set = set()
    for nodes in bad_element_node_lists:
        bad_nids_set.update(nodes)

    if not bad_nids_set:
        unchanged_nodes_df = nodes_df[['NodeID', 'x', 'y', 'z']].copy()
        info = {
            'n_free_nodes': 0,
            'n_repair_zone_elements': 0,
            'n_still_negative': 0,
            'max_displacement_achieved': 0.0,
            'converged': True,
        }
        return unchanged_nodes_df, None, info

    # Pre-compute full-model element node lists and node->element adjacency
    # (needed to expand the free/constrained node regions by topological
    # layers below).
    all_element_nodes = _unique_node_lists_fast(elements_all_df)
    nid_to_elem_indices = {}
    for elem_idx, unique_nids in enumerate(all_element_nodes):
        for nid in unique_nids:
            nid_to_elem_indices.setdefault(nid, set()).add(elem_idx)

    # Expand free nodes and constrained zone for interior-failure robustness.
    free_nids_set = _expand_node_region(
        bad_nids_set, nid_to_elem_indices, all_element_nodes, free_node_layers
    )
    constraint_nids_set = _expand_node_region(
        free_nids_set, nid_to_elem_indices, all_element_nodes, constraint_layers
    )
    free_nids_list = sorted(free_nids_set)

    # Repair zone: every element that shares at least one node with constraint nodes.
    repair_indices = [
        elem_idx
        for elem_idx, elem_nodes in enumerate(all_element_nodes)
        if set(elem_nodes) & constraint_nids_set
    ]
    repair_zone_node_lists = [all_element_nodes[i] for i in repair_indices]

    # Which repair-zone elements are actually on the user's "must fix" bad
    # list (elements_bad_df), vs. incidental neighbours pulled in only to
    # keep the zone from tangling further. Real biomechanical/anatomical
    # hex meshes routinely have plenty of already-working elements whose
    # min-corner-Jacobian is small or even negative (see is_distorted in
    # compute_element_distortion) despite a perfectly fine, positive
    # centroid volume -- these are pre-existing, unrelated to the reported
    # inversions, and are NOT this function's job to "fix" (see floor_vec
    # below).
    bad_eids_set = set(int(e) for e in elements_bad_df['EID'].tolist())
    repair_zone_eids = elements_all_df['EID'].to_numpy()[repair_indices]

    # PERFORMANCE: restrict every coordinate array to only the nodes actually
    # touched by the repair zone ("zone" nodes), instead of the FULL mesh's
    # node table. A bad-element neighborhood is typically a tiny fraction of
    # a part with tens of thousands of nodes; rebuilding/copying the entire
    # mesh's coordinate array on every constraint/objective evaluation (as an
    # earlier version of this function did via a full-size `_build_coords`)
    # makes each call O(total mesh nodes) instead of O(repair-zone nodes),
    # which otherwise dominates runtime once SciPy needs many evaluations.
    zone_nids_set = set(free_nids_set)
    for nodes in repair_zone_node_lists:
        zone_nids_set.update(nodes)
    zone_nids_list = sorted(zone_nids_set)
    zone_nid_to_local = {nid: i for i, nid in enumerate(zone_nids_list)}

    node_coord = nodes_df.set_index('NodeID')[['x', 'y', 'z']]
    zone_coords0 = node_coord.loc[zone_nids_list].values.astype(float)  # (K, 3), K << N

    free_zone_idx = np.array([zone_nid_to_local[nid] for nid in free_nids_list], dtype=int)
    free_orig_coords = zone_coords0[free_zone_idx].copy()

    # Map every zone-local node to its free-variable block index, or -1 if
    # that node is a fixed anchor (present only to validate a neighbouring
    # element, never itself moved).
    zone_local_to_freevar = np.full(len(zone_nids_list), -1, dtype=int)
    for var_idx, nid in enumerate(free_nids_list):
        zone_local_to_freevar[zone_nid_to_local[nid]] = var_idx

    # Group repair-zone elements by unique-node count (4/6/8) so their
    # constraint values AND analytic gradients can be evaluated in batched,
    # vectorized NumPy calls instead of one Python closure per element (the
    # other major cost of the earlier implementation: SLSQP finite-differenced
    # each of those per-element closures separately, multiplying the call
    # count by the number of free variables). Also track, per group and in
    # the SAME row order, whether each element is on the "must fix" bad list
    # (see bad_mask_vec/floor_vec below).
    group_rows = {4: [], 6: [], 8: []}
    group_is_bad = {4: [], 6: [], 8: []}
    for eid, nodes in zip(repair_zone_eids, repair_zone_node_lists):
        n = len(nodes)
        if n not in group_rows:
            raise ValueError(
                f"Unsupported element node count in repair zone: found {n} "
                "unique real nodes; supported counts are 4, 6, 8."
            )
        local = [zone_nid_to_local[nid] for nid in nodes]
        if n == 6:
            local = local[:5] + [local[5], local[5], local[5]]
        group_rows[n].append(local)
        group_is_bad[n].append(int(eid) in bad_eids_set)

    group4_idx = np.array(group_rows[4], dtype=int) if group_rows[4] else np.zeros((0, 4), dtype=int)
    group6_idx = np.array(group_rows[6], dtype=int) if group_rows[6] else np.zeros((0, 8), dtype=int)
    group8_idx = np.array(group_rows[8], dtype=int) if group_rows[8] else np.zeros((0, 8), dtype=int)
    # Each hex8/penta6 element contributes TWO constraint rows (corner-min
    # AND centroid, see _build_floor_and_bad_mask); tet4 contributes one.
    n_constraints = len(group4_idx) + 2 * len(group6_idx) + 2 * len(group8_idx)

    group4_is_bad = np.array(group_is_bad[4], dtype=bool)
    group6_is_bad = np.array(group_is_bad[6], dtype=bool)
    group8_is_bad = np.array(group_is_bad[8], dtype=bool)

    # Per-element "can this element's geometry possibly change at all?" mask
    # (True iff at least one of its slots maps to a free variable). Used by
    # _build_floor_and_bad_mask to choose between a strict hard-baseline
    # floor (for elements that CAN move, where it is a real, achievable
    # non-regression guarantee) and a softmin-baseline floor (for elements
    # that CANNOT move at all, where using the hard baseline would create a
    # permanently-unsatisfiable constraint -- see that function's
    # docstring). Only matters for the corner-min row: the centroid and
    # tet4 rows use a single exact (non-approximated) function in both the
    # hard and soft constraint variants, so there is no baseline/runtime
    # mismatch to guard against there.
    group8_has_free = (np.any(zone_local_to_freevar[group8_idx] != -1, axis=1)
                        if len(group8_idx) else np.zeros(0, dtype=bool))
    group6_has_free = (np.any(zone_local_to_freevar[group6_idx] != -1, axis=1)
                        if len(group6_idx) else np.zeros(0, dtype=bool))

    # Frozen (never recomputed mid-optimization) softmin sharpness, one value
    # per constrained element, calibrated from the BASELINE geometry only.
    # See _batched_hexlike_softmin_value_and_grad's docstring for why this
    # must stay fixed.
    group8_beta = _frozen_softmin_beta(zone_coords0, group8_idx, 8)
    group6_beta = _frozen_softmin_beta(zone_coords0, group6_idx, 4)

    def _build_floor_and_bad_mask():
        """
        Per-constraint-row (floor, is_bad, baseline) vectors, in the exact
        row order _constraint_parts_hard/_constraint_parts_soft concatenate:
        for each of group8 then group6, the CORNER-min row followed by its
        CENTROID row, then group4 (tet4, single row, corner==centroid by
        construction).

        floor_vec is the FINAL (end-of-optimization target) minimum
        acceptable value for each repair-zone element:
          - "must fix" bad-list elements: volume_epsilon (the user's target).
          - all other (neighbour/support) elements: min(their OWN baseline
            value, volume_epsilon), i.e. a pure non-regression bar. This
            deliberately does NOT force pre-existing, unrelated distortion
            elsewhere in the mesh to improve -- only requires it not get any
            WORSE than it already is. Without this, real meshes with plenty
            of already-working-but-naturally-skewed elements (small or
            negative min-corner-Jacobian, positive volume) make the combined
            constraint set infeasible near x0, which is what previously
            caused SLSQP to fail immediately (singular LSQ subproblem /
            positive directional derivative) with zero movement.

        Both the corner-min AND centroid rows get their own independently-
        computed floor from the SAME rule above (applied to that row's own
        baseline value) -- a neighbour element's corner-min floor and
        centroid floor are usually different numbers, both non-regression
        bars relative to their own respective baselines.

        baseline_vec (the raw, un-clamped value at zone_coords0 for every
        row) is also returned so the caller can build a CONTINUATION
        schedule for bad-list rows: severely inverted elements (e.g.
        baseline -30) cannot jump straight to +volume_epsilon in a single
        SLSQP call -- that is far outside the region where SLSQP's internal
        linearized QP subproblem is a remotely valid approximation, and was
        observed to fail immediately ("Positive directional derivative for
        linesearch", zero movement) on the real mesh. Ramping the bad-row
        floor from baseline_vec up to volume_epsilon over several
        SLSQP calls (each warm-started from the previous one) keeps every
        individual step small enough for SLSQP to handle.

        IMPORTANT (hard-vs-soft baseline choice for the corner-min row): the
        corner-min row's baseline must be computed differently depending on
        whether the element can move at all:
          - Elements with >= 1 free node (group{8,6}_has_free): use the
            EXACT hard-min baseline. This is a real, achievable target --
            since the element CAN move, SLSQP finding a feasible solution
            means the TRUE (not approximated) corner-min genuinely did not
            regress. Using a softer/looser baseline here would only give a
            SOFT non-regression guarantee (softmin(final) >= softmin(base))
            which does NOT mathematically imply the true hard-min didn't
            regress, because the softmin-vs-hardmin approximation gap can
            itself change as the geometry moves. This was caught via a
            real-mesh diagnostic AFTER a first (all-soft-baseline) fix: 2
            neighbour elements (never on the user's bad list) had their
            TRUE corner-min cross slightly negative post-repair (e.g.
            +0.048 -> -0.016) even though their smoothed proxy still
            nominally satisfied its own (softer, weaker) non-regression
            bar -- a real, if minor, regression that a hard baseline
            prevents outright.
          - Elements with ZERO free nodes (fully pinned, cannot move under
            ANY circumstances): use the SOFTMIN baseline instead. For these
            rows the true hard-min is PROVABLY frozen at its baseline value
            forever (no free node means no possible coordinate change), so
            there is zero risk in using the softer baseline -- nothing can
            regress regardless of which floor style is chosen. Using the
            hard baseline here, by contrast, would create a permanently
            unsatisfiable constraint: softmin(x0) < hardmin(baseline) purely
            from the smoothing gap (never a real defect), and since the row
            can never move, this false violation can NEVER be corrected.
            This was caught via an earlier real-mesh diagnostic: 11
            unrelated neighbour elements (e.g. EID 2366230) came out
            "stuck" (zero gradient AND already violated at x0) purely
            because of this mismatch, which made the whole zone look
            falsely infeasible to SLSQP from the very first iteration.
        The centroid and tet4 rows need NO such blending: both use a single
        EXACT (non-approximated, no argmin/softmin involved) function in
        both the hard and soft constraint variants, so there is no
        baseline/runtime gap to guard against for those rows regardless of
        whether the element can move.
        """
        floor_parts, mask_parts, baseline_parts = [], [], []
        if len(group8_idx):
            v8_base_hard, _ = _batched_hexlike_value_and_grad(zone_coords0[group8_idx], 8)
            v8_base_soft, _ = _batched_hexlike_softmin_value_and_grad(zone_coords0[group8_idx], 8, group8_beta)
            v8_base = np.where(group8_has_free, v8_base_hard, v8_base_soft)
            floor_parts.append(np.where(group8_is_bad, volume_epsilon, np.minimum(v8_base, volume_epsilon)))
            mask_parts.append(group8_is_bad)
            baseline_parts.append(v8_base)
            v8c_base, _ = _batched_hexlike_centroid_value_and_grad(zone_coords0[group8_idx])
            floor_parts.append(np.where(group8_is_bad, volume_epsilon, np.minimum(v8c_base, volume_epsilon)))
            mask_parts.append(group8_is_bad)
            baseline_parts.append(v8c_base)
        if len(group6_idx):
            v6_base_hard, _ = _batched_hexlike_value_and_grad(zone_coords0[group6_idx], 4)
            v6_base_soft, _ = _batched_hexlike_softmin_value_and_grad(zone_coords0[group6_idx], 4, group6_beta)
            v6_base = np.where(group6_has_free, v6_base_hard, v6_base_soft)
            floor_parts.append(np.where(group6_is_bad, volume_epsilon, np.minimum(v6_base, volume_epsilon)))
            mask_parts.append(group6_is_bad)
            baseline_parts.append(v6_base)
            v6c_base, _ = _batched_hexlike_centroid_value_and_grad(zone_coords0[group6_idx])
            floor_parts.append(np.where(group6_is_bad, volume_epsilon, np.minimum(v6c_base, volume_epsilon)))
            mask_parts.append(group6_is_bad)
            baseline_parts.append(v6c_base)
        if len(group4_idx):
            v4_base, _ = _batched_tet_value_and_grad(zone_coords0[group4_idx])
            floor_parts.append(np.where(group4_is_bad, volume_epsilon, np.minimum(v4_base, volume_epsilon)))
            mask_parts.append(group4_is_bad)
            baseline_parts.append(v4_base)
        if not floor_parts:
            return np.zeros(0), np.zeros(0, dtype=bool), np.zeros(0)
        return np.concatenate(floor_parts), np.concatenate(mask_parts), np.concatenate(baseline_parts)

    final_floor_vec, bad_mask_vec, baseline_vec = _build_floor_and_bad_mask()
    floor_vec = final_floor_vec.copy()  # mutated stage-by-stage; see continuation loop below


    # Optional outward pre-bias for free nodes on the global outer boundary.
    target_free_coords = free_orig_coords.copy()
    n_boundary_grow_nodes = 0
    if boundary_growth > 0:
        boundary_nodes = _collect_boundary_nodes(elements_all_df, all_element_nodes)
        bad_centroid = node_coord.loc[sorted(bad_nids_set)].values.astype(float).mean(axis=0)
        free_nid_to_local = {nid: i for i, nid in enumerate(free_nids_list)}

        for nid in sorted(free_nids_set & boundary_nodes):
            local_idx = free_nid_to_local[nid]
            p = free_orig_coords[local_idx]
            direction = p - bad_centroid
            norm = float(np.linalg.norm(direction))
            if norm == 0.0:
                continue
            target_free_coords[local_idx] = p + (boundary_growth * direction / norm)
            n_boundary_grow_nodes += 1

    low_bounds = free_orig_coords - max_displacement
    high_bounds = free_orig_coords + max_displacement
    target_free_coords = np.minimum(np.maximum(target_free_coords, low_bounds), high_bounds)

    x0 = target_free_coords.flatten()
    orig_x0 = free_orig_coords.flatten()
    target_x0 = target_free_coords.flatten()
    n_free_vars = len(orig_x0)

    # HYBRID DISPATCH decision (see _DENSE_JACOBIAN_SIZE_THRESHOLD's
    # docstring/comment above): small pockets use the reference SLSQP
    # solver (dense Jacobian, proven fast on already-validated cases);
    # large pockets use this fork's trust-constr solver (sparse Jacobian
    # + exact objective Hessian), the whole reason this fork exists.
    dense_jacobian_size = n_constraints * n_free_vars
    use_sparse_solver = dense_jacobian_size >= _DENSE_JACOBIAN_SIZE_THRESHOLD
    print(
        f"  Solver dispatch: n_free_vars={n_free_vars}, n_constraints={n_constraints} "
        f"-> estimated dense Jacobian size={dense_jacobian_size:,} "
        f"(threshold={_DENSE_JACOBIAN_SIZE_THRESHOLD:,}) "
        f"-> using {'trust-constr (sparse)' if use_sparse_solver else 'SLSQP (dense)'}."
    )

    def _zone_coords_from_x(x):
        coords = zone_coords0.copy()   # (K, 3): zone-local only, NOT full mesh
        coords[free_zone_idx] = x.reshape(-1, 3)
        return coords

    def objective(x):
        stay_term = np.sum((x - orig_x0) ** 2)
        if boundary_growth > 0 and growth_stiffness > 0 and n_boundary_grow_nodes > 0:
            grow_term = np.sum((x - target_x0) ** 2)
            return float(stay_term + growth_stiffness * grow_term)
        return float(stay_term)

    def objective_jac(x):
        grad = 2.0 * (x - orig_x0)
        if boundary_growth > 0 and growth_stiffness > 0 and n_boundary_grow_nodes > 0:
            grad += 2.0 * growth_stiffness * (x - target_x0)
        return grad

    # trust-constr requires a scipy.optimize.Bounds object; SLSQP requires
    # a plain list of (low, high) tuples, one per variable -- same actual
    # bounds (per-axis box around each free node's original position),
    # just a different container type per solver's API.
    if use_sparse_solver:
        bounds = Bounds(orig_x0 - max_displacement, orig_x0 + max_displacement)
    else:
        bounds = [(orig_x0[j] - max_displacement, orig_x0[j] + max_displacement)
                  for j in range(n_free_vars)]

    def _constraint_parts_hard(coords):
        """One (values, slot_grad, slot_idx) tuple per constraint row, using
        EXACT (non-smooth) criteria throughout: for group8/group6, both the
        hard corner-min row AND the centroid-volume row (see
        _build_floor_and_bad_mask for why both are required); values/
        slot_grad are batched over that whole group at once. Used only for
        the floor baseline and the final post-optimization pass/fail report
        -- NOT for the SLSQP-facing constraint itself (see
        _constraint_parts_soft)."""
        parts = []
        if len(group8_idx):
            v8, g8 = _batched_hexlike_value_and_grad(coords[group8_idx], 8)
            parts.append((v8, g8, group8_idx))
            v8c, g8c = _batched_hexlike_centroid_value_and_grad(coords[group8_idx])
            parts.append((v8c, g8c, group8_idx))
        if len(group6_idx):
            v6, g6 = _batched_hexlike_value_and_grad(coords[group6_idx], 4)
            parts.append((v6, g6, group6_idx))
            v6c, g6c = _batched_hexlike_centroid_value_and_grad(coords[group6_idx])
            parts.append((v6c, g6c, group6_idx))
        if len(group4_idx):
            v4, g4 = _batched_tet_value_and_grad(coords[group4_idx])
            parts.append((v4, g4, group4_idx))
        return parts

    def _constraint_parts_soft(coords):
        """Same shape/contract as _constraint_parts_hard, but the corner-min
        rows for group8/group6 use the smooth softmin proxy (frozen
        per-element beta; see _batched_hexlike_softmin_value_and_grad)
        instead of the hard min. This is what SLSQP actually optimizes
        against: softmin(v) <= min(v) always, so it's a safe (never-too-
        lenient), smoother stand-in that avoids the argmin "kink" which was
        causing SLSQP to oscillate instead of converging. The centroid rows
        need no smoothing (already an exact smooth polynomial, no argmin
        involved), so they're identical between the hard and soft variants.
        Tet4 has a single scalar value (no corner-min involved), so it's
        identical to the hard version too."""
        parts = []
        if len(group8_idx):
            v8, g8 = _batched_hexlike_softmin_value_and_grad(coords[group8_idx], 8, group8_beta)
            parts.append((v8, g8, group8_idx))
            v8c, g8c = _batched_hexlike_centroid_value_and_grad(coords[group8_idx])
            parts.append((v8c, g8c, group8_idx))
        if len(group6_idx):
            v6, g6 = _batched_hexlike_softmin_value_and_grad(coords[group6_idx], 4, group6_beta)
            parts.append((v6, g6, group6_idx))
            v6c, g6c = _batched_hexlike_centroid_value_and_grad(coords[group6_idx])
            parts.append((v6c, g6c, group6_idx))
        if len(group4_idx):
            v4, g4 = _batched_tet_value_and_grad(coords[group4_idx])
            parts.append((v4, g4, group4_idx))
        return parts

    def constraint_vec(x):
        parts = _constraint_parts_soft(_zone_coords_from_x(x))
        if not parts:
            return np.zeros(0)
        return np.concatenate([v for v, _, _ in parts]) - floor_vec

    def constraint_jac(x):
        """
        Returns a SPARSE (CSR) matrix when routed to trust-constr, or a
        dense ndarray (via `.toarray()`) when routed to SLSQP -- see
        HYBRID DISPATCH / `use_sparse_solver` above. trust-constr accepts
        a sparse constraint Jacobian natively (see NonlinearConstraint
        below) and performs sparse matrix-vector operations internally,
        whose cost scales with the TRUE number of non-zero entries (each
        constraint row only ever touches ~24 free variables -- one
        repair-zone element's own corner nodes -- regardless of how many
        thousands of free variables exist in the whole pocket), instead
        of the full (n_constraints x n_free_vars) dense size SLSQP
        requires. _self_check_analytic_gradient below works unmodified
        against EITHER return type: both dense ndarrays and scipy sparse
        matrices support scalar (row, col) indexing, returning the same
        numeric value.
        """
        parts = _constraint_parts_soft(_zone_coords_from_x(x))
        if not parts:
            empty = coo_matrix((0, n_free_vars))
            return empty.toarray() if not use_sparse_solver else empty.tocsr()

        rows_list, cols_list, vals_list = [], [], []
        row_offset = 0
        for _, slot_grad, slot_idx in parts:
            r, c, v = _scatter_group_jacobian(slot_grad, slot_idx, zone_local_to_freevar, row_offset)
            rows_list.append(r)
            cols_list.append(c)
            vals_list.append(v)
            row_offset += slot_grad.shape[0]

        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        vals = np.concatenate(vals_list)
        J = coo_matrix((vals, (rows, cols)), shape=(n_constraints, n_free_vars))
        return J.tocsr() if use_sparse_solver else J.toarray()

    def objective_hess(x):
        """
        EXACT, CONSTANT Hessian of the objective (sum of squared node
        displacement, optionally plus a boundary-growth term) -- both
        terms are pure quadratics in x, so their second derivative is a
        constant, diagonal matrix regardless of x. Returned as a sparse
        matrix (trust-constr accepts sparse Hessians directly, and an
        identity-scaled diagonal is the cheapest possible representation
        -- no dense (n_free_vars x n_free_vars) matrix is ever built).
        Unlike SLSQP (which never uses/needs a Hessian at all -- it solves
        a sequence of linearized QP subproblems), trust-constr uses this
        directly for the objective's contribution to the Lagrangian
        Hessian; the constraints' contribution is left to SciPy's own BFGS
        quasi-Newton approximation (NonlinearConstraint's default `hess`),
        which is standard practice and avoids hand-deriving second
        derivatives of the volume/Jacobian formulas.
        """
        scale = 2.0
        if boundary_growth > 0 and growth_stiffness > 0 and n_boundary_grow_nodes > 0:
            scale += 2.0 * growth_stiffness
        return scale * sparse_identity(n_free_vars, format='csr')

    print(
        f"  Repair zone: {len(free_nids_list)} free nodes, "
        f"{n_constraints} constrained elements (of {len(zone_nids_list)} zone nodes). "
        f"(free_node_layers={free_node_layers}, constraint_layers={constraint_layers})"
    )
    if boundary_growth > 0:
        print(
            f"  Boundary growth pre-bias: {n_boundary_grow_nodes} boundary nodes "
            f"moved outward by up to {boundary_growth}."
        )

    zone_extent = float(np.ptp(zone_coords0, axis=0).max()) if len(zone_coords0) else 1.0
    jitter_scale = max(zone_extent * 1e-3, 1e-6)
    _self_check_analytic_gradient(constraint_vec, constraint_jac, x0, n_free_vars, n_constraints,
                                   jitter_scale=jitter_scale)

    iteration_state = {'count': 0, 'last_xk': None, 'early_stop_reason': None}

    def _make_stage_callback_slsqp():
        """
        SLSQP branch: tracks a rolling window of the worst-constraint
        value and raises StopIteration to signal early-stop under either
        of two conditions (see _STALL_WINDOW's and _BOUND_SATURATION_
        WINDOW's docstrings/comments above). History intentionally does
        NOT carry over between stages: each stage targets a different
        (tighter) floor, so a plateau at the END of the previous stage
        must not immediately re-trigger a stall in the fresh stage. No
        "feasible and stable" early-stop here -- that condition exists
        only to counteract trust-constr's much looser default stopping
        behaviour (see _FEASIBLE_STALL_WINDOW's comment); SLSQP does not
        need it.

        *** CRITICAL FIX (found via a REAL production crash, "Run5") ***
        Earlier versions of this docstring claimed StopIteration raised
        here is "caught by SciPy's SLSQP driver" -- this was WRONG, and
        the mistake had gone undetected because it had only ever been
        exercised via multiprocessing.Pool.imap_unordered (MAX_CONCURRENT_
        POCKETS > 1): a StopIteration escaping a worker process, once
        propagated back through the POOL'S OWN internal generator, gets
        swallowed by Python's generator/iterator protocol (PEP 479) and
        mistaken for "the iterator is exhausted" rather than surfacing as
        a real error -- which is what silently produced the "N pocket(s)
        NEVER RETURNED A RESULT" messages in Run3/Run4's parallel runs
        (a real bug, just not the one those messages described). The
        FIRST time this code path ran with MAX_CONCURRENT_POCKETS=1 (no
        Pool at all -- see Run5), the exact same StopIteration propagated
        with NOTHING to catch it -- straight out of scipy.optimize.minimize,
        out of repair_inverted_elements, and crashed the entire script
        (confirmed via the real traceback: .../_slsqp_py.py line 445,
        `callback(np.copy(x))`, with no surrounding try/except for
        StopIteration in that scipy version). This callback therefore now
        records the LAST xk and the early-stop reason into iteration_state
        (a plain dict shared with the enclosing function, not returned by
        minimize()) BEFORE raising, so the call site below can catch the
        StopIteration itself and reconstruct a proper best-effort
        OptimizeResult from that recorded xk -- instead of relying on
        scipy to do it, which it does not.
        """
        history = []
        move_history = []
        max_theoretical_move = max_displacement * np.sqrt(3.0)

        def _callback(xk):
            iteration_state['count'] += 1
            iteration_state['last_xk'] = xk.copy()
            worst = float(constraint_vec(xk).min()) if n_constraints else 0.0
            move = float(np.linalg.norm((xk - orig_x0).reshape(-1, 3), axis=1).max()) if n_free_vars else 0.0
            print(f"    iter {iteration_state['count']:4d}: worst constraint={worst: .6e}, max move={move:.4f}")

            history.append(worst)
            if len(history) > _STALL_WINDOW:
                history.pop(0)
            if (len(history) == _STALL_WINDOW and worst < -_STALL_MIN_VIOLATION
                    and (max(history) - min(history)) < _STALL_RELATIVE_SWING * abs(worst)):
                print(
                    f"    -> stalled: worst constraint has not improved by more than "
                    f"{_STALL_RELATIVE_SWING:.0%} over the last {_STALL_WINDOW} iterations "
                    f"(still violated at {worst:.4f}). Stopping this stage early and "
                    f"handing off to the next stage/attempt."
                )
                iteration_state['early_stop_reason'] = 'stalled: no meaningful improvement in worst constraint'
                raise StopIteration(iteration_state['early_stop_reason'])

            move_history.append(move)
            if len(move_history) > _BOUND_SATURATION_WINDOW:
                move_history.pop(0)
            mean_move = sum(move_history) / len(move_history)
            if (len(move_history) == _BOUND_SATURATION_WINDOW and worst < -_BOUND_SATURATION_MIN_VIOLATION
                    and mean_move >= _BOUND_SATURATION_RATIO * max_theoretical_move):
                print(
                    f"    -> diverging: mean max-move ({mean_move:.4f}) has stayed within "
                    f"{_BOUND_SATURATION_RATIO:.0%} of the theoretical bound-diagonal limit "
                    f"({max_theoretical_move:.4f}) over the last {_BOUND_SATURATION_WINDOW} "
                    f"iterations while still meaningfully violated ({worst:.4f}) -- this zone/"
                    f"attempt cannot satisfy the constraint within the current displacement "
                    f"bound. Stopping this stage early and handing off to the next, less-"
                    f"constrained stage/attempt rather than exhausting the iteration budget."
                )
                iteration_state['early_stop_reason'] = 'diverging: pinned at displacement bound while still violated'
                raise StopIteration(iteration_state['early_stop_reason'])

        return _callback

    def _make_stage_callback_trust_constr():
        """
        Fresh callback per continuation stage: tracks a rolling window of
        the worst-constraint value and returns True -- trust-constr's
        documented signal to terminate the CURRENT solve early -- under
        EITHER of two conditions (see _STALL_WINDOW's and
        _FEASIBLE_STALL_WINDOW's docstrings/comments above):
          1. Stalled while VIOLATED: the window's swing collapses to
             near-zero WHILE the constraint is still meaningfully unmet.
          2. Feasible and stable (this fork's trust-constr-specific
             addition): every constraint has been simultaneously
             SATISFIED for a full window AND has stopped changing
             meaningfully -- stops the interior-point method's further
             "polishing" once our actual goal is already met, rather than
             letting it grind toward its own much tighter internal
             optimality tolerance.
          3. Diverging / pinned at the displacement bound (see
            _BOUND_SATURATION_WINDOW's comment above) -- observed on the
            SLSQP branch in real production use; included here too since
            trust-constr's own bounds mechanics could suffer the same
            infeasible-subproblem pathology on a large pocket.
        History intentionally does NOT carry over between stages: each
        stage targets a different (tighter) floor, so a plateau at the
        END of the previous stage must not immediately re-trigger either
        early-stop condition in the fresh stage.

        NOTE: trust-constr's callback signature is `callback(xk, state) ->
        bool` (return True to stop) -- a genuine API difference from
        SLSQP's `callback(xk)`, which instead signals early-stop by
        raising StopIteration. `state` (a scipy OptimizeResult-like object
        with its own iteration/constraint-violation info) is accepted but
        deliberately unused here, so this callback's printed diagnostics
        stay identical in format to the reference SLSQP implementation's
        (independently recomputed via constraint_vec(xk), not read from
        trust-constr's own internal state) -- this keeps existing log-
        watching/parsing workflows unaffected by the solver swap.
        """
        history = []
        move_history = []
        max_window = max(_STALL_WINDOW, _FEASIBLE_STALL_WINDOW, _BOUND_SATURATION_WINDOW)
        max_theoretical_move = max_displacement * np.sqrt(3.0)

        def _callback(xk, state):
           iteration_state['count'] += 1
           iteration_state['last_xk'] = xk.copy()
           worst = float(constraint_vec(xk).min()) if n_constraints else 0.0
           move = float(np.linalg.norm((xk - orig_x0).reshape(-1, 3), axis=1).max()) if n_free_vars else 0.0
           print(f"    iter {iteration_state['count']:4d}: worst constraint={worst: .6e}, max move={move:.4f}")

           history.append(worst)
           if len(history) > max_window:
               history.pop(0)
           move_history.append(move)
           if len(move_history) > max_window:
               move_history.pop(0)

           stall_window_vals = history[-_STALL_WINDOW:]
           if (len(stall_window_vals) == _STALL_WINDOW and worst < -_STALL_MIN_VIOLATION
                   and (max(stall_window_vals) - min(stall_window_vals)) < _STALL_RELATIVE_SWING * abs(worst)):
               print(
                   f"    -> stalled: worst constraint has not improved by more than "
                   f"{_STALL_RELATIVE_SWING:.0%} over the last {_STALL_WINDOW} iterations "
                   f"(still violated at {worst:.4f}). Stopping this stage early and "
                   f"handing off to the next stage/attempt."
               )
               return True

           feas_window_vals = history[-_FEASIBLE_STALL_WINDOW:]
           if (len(feas_window_vals) == _FEASIBLE_STALL_WINDOW and min(feas_window_vals) >= 0.0
                   and (max(feas_window_vals) - min(feas_window_vals)) < _FEASIBLE_STALL_ABS_SWING):
               print(
                   f"    -> feasible and stable: every constraint has been "
                   f"satisfied (worst >= 0) and stable (swing < "
                   f"{_FEASIBLE_STALL_ABS_SWING:.1e}) over the last "
                   f"{_FEASIBLE_STALL_WINDOW} iterations. Stopping this stage "
                   f"early rather than over-polishing past trust-constr's own "
                   f"much tighter internal optimality tolerance."
               )
               return True

           move_window_vals = move_history[-_BOUND_SATURATION_WINDOW:]
           mean_move = sum(move_window_vals) / len(move_window_vals) if move_window_vals else 0.0
           if (len(move_window_vals) == _BOUND_SATURATION_WINDOW and worst < -_BOUND_SATURATION_MIN_VIOLATION
                   and mean_move >= _BOUND_SATURATION_RATIO * max_theoretical_move):
               print(
                   f"    -> diverging: mean max-move ({mean_move:.4f}) has stayed within "
                   f"{_BOUND_SATURATION_RATIO:.0%} of the theoretical bound-diagonal limit "
                   f"({max_theoretical_move:.4f}) over the last {_BOUND_SATURATION_WINDOW} "
                   f"iterations while still meaningfully violated ({worst:.4f}) -- this zone/"
                   f"attempt cannot satisfy the constraint within the current displacement "
                   f"bound. Stopping this stage early and handing off to the next, less-"
                   f"constrained stage/attempt rather than exhausting the iteration budget."
               )
               return True

           return False

        return _callback

    _make_stage_callback = _make_stage_callback_trust_constr if use_sparse_solver else _make_stage_callback_slsqp

    # Continuation schedule: ramp bad-list rows' floor from their own
    # baseline up to volume_epsilon over several solver calls (see
    # _CONTINUATION_STEP's docstring/comment above). Neighbour/support rows
    # are never ramped -- their non-regression floor is already satisfied
    # at x0 by construction (_build_floor_and_bad_mask), so no continuation
    # is needed for them.
    if np.any(bad_mask_vec):
        bad_gap = np.maximum(final_floor_vec[bad_mask_vec] - baseline_vec[bad_mask_vec], 0.0)
        max_gap = float(bad_gap.max())
    else:
        bad_gap = np.zeros(0)
        max_gap = 0.0
    n_stages = int(np.clip(np.ceil(max_gap / _CONTINUATION_STEP), 1, _MAX_CONTINUATION_STAGES)) if max_gap > 0 else 1

    print(f"  Running {'trust-constr (sparse Jacobian)' if use_sparse_solver else 'SLSQP (dense Jacobian)'} optimisation...")
    if n_stages > 1:
        print(f"  Worst bad-element gap to target: {max_gap:.3f} -> using {n_stages} continuation stages.")

    opt_result = None
    for stage in range(1, n_stages + 1):
        alpha = stage / n_stages
        if np.any(bad_mask_vec):
            floor_vec[bad_mask_vec] = baseline_vec[bad_mask_vec] + alpha * bad_gap
        is_final_stage = (stage == n_stages)
        if n_stages > 1:
            print(f"  -- continuation stage {stage}/{n_stages} (alpha={alpha:.2f}) --")
        if use_sparse_solver:
            # Delegates to _run_trust_constr_stage (see that function's own
            # docstring, right after _DENSE_JACOBIAN_SIZE_THRESHOLD) so the
            # numerical-overflow safety net is shared, not duplicated,
            # between this function and repair_distorted_shells.
            opt_result = _run_trust_constr_stage(
                objective, x0, objective_jac, objective_hess, bounds,
                constraint_vec, constraint_jac, _make_stage_callback(),
                _FINAL_STAGE_MAXITER if is_final_stage else _INTERMEDIATE_STAGE_MAXITER,
                iteration_state,
            )
        else:
            # SLSQP branch, WITH a fix this fork adds on top of the
            # reference implementation's call shape (no `hess=`, since
            # SLSQP never uses second-derivative information).
            #
            # *** CRITICAL: StopIteration from the callback must be
            # caught HERE, not assumed to be handled by scipy *** (see
            # _make_stage_callback_slsqp's docstring above for the full
            # story -- a real production crash, "Run5", proved this
            # scipy version's SLSQP driver does NOT catch it internally).
            # iteration_state['last_xk'] was recorded by the callback on
            # every iteration BEFORE it ever raises, so a caught
            # StopIteration here can still reconstruct a valid, usable
            # best-effort OptimizeResult (marked success=False, exactly
            # like scipy's own early-exit results elsewhere in this
            # ladder) instead of losing the stage's progress and crashing
            # the entire script.
            try:
                opt_result = minimize(
                    objective,
                    x0,
                    jac=objective_jac,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[{'type': 'ineq', 'fun': constraint_vec, 'jac': constraint_jac}],
                    callback=_make_stage_callback(),
                    options={
                        'maxiter': _FINAL_STAGE_MAXITER if is_final_stage else _INTERMEDIATE_STAGE_MAXITER,
                        'ftol': _SLSQP_FTOL,
                        'disp': is_final_stage,
                    },
                )
            except StopIteration as e:
                last_xk = iteration_state['last_xk']
                if last_xk is None:
                    # Should be unreachable (the callback always runs at
                    # least once before it can raise), but fail safe
                    # rather than crash if scipy's internals ever change.
                    last_xk = x0
                opt_result = OptimizeResult(
                    x=last_xk, success=False, status=-1,
                    message=f"stopped early via callback: {e}",
                    fun=float(objective(last_xk)), nit=iteration_state['count'],
                )
        x0 = opt_result.x  # warm-start the next (closer-to-final-target) stage

    final_coords = _zone_coords_from_x(opt_result.x)
    final_parts = _constraint_parts_hard(final_coords)

    # Preserve every node's exact original row order/coordinates except free
    # nodes, whose coordinates are overwritten with the optimizer's result.
    updated_nodes_df = nodes_df[['NodeID', 'x', 'y', 'z']].copy().set_index('NodeID')
    final_free_coords = final_coords[free_zone_idx]
    updated_nodes_df.loc[free_nids_list, ['x', 'y', 'z']] = final_free_coords
    updated_nodes_df = updated_nodes_df.reset_index()

    # Post-repair validation. Two INDEPENDENT checks, both against the exact
    # same final_floor_vec the optimizer was actually asked to satisfy:
    #
    # 1) n_still_negative: the user's original "must fix" bad-list elements
    #    (bad_mask_vec) -- did we actually hit volume_epsilon for these?
    #
    # 2) n_neighbor_regressions (added after a real production bug: a
    #    pocket can log "diverging ... cannot satisfy the constraint" and
    #    then STILL report converged=True/still_bad=0, because bad_mask_vec
    #    elements happened to clear while a NEIGHBOUR element silently blew
    #    through its own non-regression floor instead -- invisible to any
    #    check until the final whole-mesh report, far too late to retry.
    #    Confirmed on a real HPC run: EID 1243817/1243828 went from healthy
    #    (+4.75/+0.49) to negative (-0.0507/-0.0101) as untracked collateral
    #    from exactly this blind spot). Neighbour/support elements are still
    #    intentionally held to a non-regression floor rather than an
    #    absolute bar (see _build_floor_and_bad_mask) -- a pre-existing,
    #    unrelated low/negative baseline elsewhere in the mesh must NOT
    #    count as a regression here, since final_floor_vec for those rows
    #    already equals their own baseline. volume_epsilon is reused as the
    #    "real regression, not solver noise" tolerance since it is already
    #    this function's own definition of an acceptable margin above zero.
    if final_parts and len(bad_mask_vec):
        all_final_vals = np.concatenate([v for v, _, _ in final_parts])
        n_still_negative = int(np.sum(all_final_vals[bad_mask_vec] <= 0))
        neighbor_mask = ~bad_mask_vec
        neighbor_shortfall = final_floor_vec[neighbor_mask] - all_final_vals[neighbor_mask]
        neighbor_regressed = neighbor_shortfall > volume_epsilon
        n_neighbor_regressions = int(np.sum(neighbor_regressed))
        max_neighbor_regression = float(neighbor_shortfall.max()) if len(neighbor_shortfall) else 0.0
    else:
        n_still_negative = 0
        n_neighbor_regressions = 0
        max_neighbor_regression = 0.0

    displacements = np.linalg.norm(
        opt_result.x.reshape(-1, 3) - orig_x0.reshape(-1, 3), axis=1
    )
    info = {
        'n_free_nodes': len(free_nids_list),
        'n_repair_zone_elements': n_constraints,
        'n_still_negative': n_still_negative,
        'n_neighbor_regressions': n_neighbor_regressions,
        'max_neighbor_regression': max_neighbor_regression,
        'max_displacement_achieved': float(displacements.max()) if len(displacements) else 0.0,
        # 'converged' = our independently re-verified goal (n_still_negative
        # == 0 AND no neighbour floor regressions), with opt_result.success
        # consulted only as a substitute for the n_still_negative == 0 leg --
        # see the AND-of-ORs note below for why it is NEVER allowed to
        # override a real detected neighbour regression on its own.
        # The reference SLSQP implementation used opt_result.success alone,
        # which was a valid proxy THERE because SLSQP's early-stop-via-
        # callback is a rare stall-only fallback (see _STALL_WINDOW), not the
        # normal exit path -- a real SLSQP success generally did mean "goal
        # achieved". This fork's trust-constr solver, however, is
        # DELIBERATELY told to stop early via callback on EVERY stage once
        # feasible-and-stable (see _FEASIBLE_STALL_WINDOW) -- this is now the
        # NORMAL, fast, intended exit path (4x+ faster than letting
        # trust-constr grind to its own much tighter internal tolerance),
        # which makes opt_result.success == False on essentially every
        # successful run, not just genuine failures. Falling back to the
        # independently-recomputed n_still_negative/n_neighbor_regressions
        # (via the EXACT hard-min constraint check above, not trust-constr's
        # own internal feasibility notion) restores a meaningful 'converged'
        # signal regardless of which early-stop path was taken -- and is a
        # STRICTLY stronger correctness check than trusting the solver's
        # self-reported flag in the first place. Requiring BOTH conditions
        # (not just n_still_negative == 0) is what makes the escalation
        # ladder correctly retry with the next, less-constrained attempt
        # instead of prematurely declaring success while a neighbour
        # element was left in a newly-broken state.
        #
        # NOTE the AND-of-ORs structure (not `opt_result.success OR (...)`,
        # which this function used prior to this tightening): a solver can
        # report success=True (its OWN soft/approximate constraint tolerance
        # satisfied) while the independent HARD recheck above still finds a
        # real neighbor regression -- e.g. a real, non-hypothetical gap
        # between the solver-facing constraint value and this exact
        # post-solve recompute. n_neighbor_regressions == 0 is therefore a
        # mandatory gate, never overridable by opt_result.success alone;
        # opt_result.success is only consulted (via the OR) as a substitute
        # for the n_still_negative == 0 check. Algebraically identical to
        # the old `S or (X==0 and Y==0)` form in every case except this one,
        # which it deliberately closes (mirrored in
        # repair_soft_hard_penetrations and repair_distorted_shells).
        'converged': (n_neighbor_regressions == 0) and (n_still_negative == 0 or bool(opt_result.success)),
    }
    return updated_nodes_df, opt_result, info


def build_node_to_node_adjacency(elements_df):
    """
    Build a NodeID -> set of other NodeIDs sharing at least one common
    element -- used ONLY for Laplacian-style smoothing (see
    smooth_repair_zone_boundary below), not for the repair zone's own
    free/constrained expansion (which uses _expand_node_region /
    element-to-node adjacency instead).

    For hex8/penta6/tet4 elements this connects EVERY pair of nodes
    within the SAME element -- a conservative superset of true mesh
    edges (e.g. it also connects a hex8's body-diagonal corners). This
    is intentional for smoothing: it pulls each free node toward its
    whole local neighborhood's average position, which produces a
    gentler, more forgiving blend than strict edge-only adjacency would.

    Inputs:
        elements_df (pd.DataFrame): elements to build adjacency from
            (typically a small, local repair-zone subset, not a full
            multi-million-element mesh).

    Returns:
        dict[int, set[int]]: NodeID -> set of co-element NodeIDs.
    """
    node_lists = _unique_node_lists_fast(elements_df)
    adjacency = {}
    for nodes in node_lists:
        for nid in nodes:
            others = adjacency.setdefault(nid, set())
            others.update(n for n in nodes if n != nid)
    return adjacency


def smooth_repair_zone_boundary(elements_all_df, nodes_before_df, nodes_after_df,
                                 hard_displacement_cap, max_smoothing_displacement,
                                 blend_ring_layers=2, max_iterations=3,
                                 initial_blend_factor=0.35, min_blend_factor=0.05):
    """
    OPTIONAL, PURELY COSMETIC post-repair smoothing pass -- reduces the
    sharp local crease that repair_inverted_elements/repair_distorted_
    shells can leave behind at the boundary between a repair zone's moved
    ("free") nodes and its untouched surroundings. Both repair functions'
    own objective is ONLY `sum((x - x0)^2)` (minimum total displacement)
    -- there is NO smoothness/curvature term at all, so the solver's
    cheapest valid fix can visibly kink even though every volume/Jacobian
    constraint is satisfied. This function is a separate, opt-in follow-
    up step -- it never runs inside the constrained optimizer itself, and
    is safe to skip entirely (repair correctness never depends on it).

    ALGORITHM: identify every node whose position actually changed during
    repair ("core moved" nodes -- an exact diff between nodes_before_df/
    nodes_after_df, not the optimizer's full theoretical free-node set,
    since not every free node necessarily needed to move), then expand
    outward by blend_ring_layers adjacency rings (_expand_node_region,
    same mechanism the repair zone itself was built from) to include a
    ring of previously-FROZEN neighbor nodes in the smoothing -- this is
    what actually blends the transition across a wider region instead of
    just relaxing the already-moved nodes back toward their frozen
    neighbors (which would partially undo the repair instead of smoothing
    it). Runs damped Jacobi-style Laplacian smoothing (every smoothable
    node moves toward the average position of every node sharing an
    element with it, see build_node_to_node_adjacency) for up to
    max_iterations sweeps.

    WHY A SEPARATE, TIGHT max_smoothing_displacement BUDGET (measured from
    each node's POST-REPAIR position, NOT its original pre-repair
    position): confirmed via direct testing that unconstrained multi-sweep
    Laplacian smoothing does not just soften a repair's own artificial
    seam -- on a curved anatomical surface, it also flattens the mesh's
    REAL, pre-existing curvature nearby (a real measurement: 4 sweeps at
    blend_factor=0.5 moved one ring node 3.6 mm, on a mesh where the
    ACTUAL repair itself only needed 0.89 mm -- clearly no longer a small
    cosmetic nudge, but a meaningful reshape of real anatomy). Clamping
    each candidate to within max_smoothing_displacement of that SAME
    node's post-repair starting position (a small, separate budget,
    independent of however much of the true hard_displacement_cap the
    repair itself already used) is what keeps this pass genuinely
    cosmetic -- hard_displacement_cap (measured from the ORIGINAL
    pre-repair position) is still enforced as a belt-and-suspenders outer
    bound, but should rarely be the binding constraint once max_smoothing_
    displacement is set sensibly small relative to it.

    SAFETY (never allowed to regress the repair or exceed either
    displacement budget): after every tentative sweep, every element
    touching the smoothed zone is re-classified with classify_solid_
    badness. Any element that was ALREADY bad immediately after repair is
    excluded from this check (smoothing is not expected to fix what the
    optimizer itself could not); every element that WAS valid must STAY
    valid. If a sweep would violate this, its blend factor is halved and
    retried from the last safe state; once the blend factor drops below
    min_blend_factor, smoothing simply stops and the last safe (possibly
    zero-sweep) result is returned -- cosmetic smoothing is never worth
    risking the actual repair guarantee.

    Inputs:
        elements_all_df (pd.DataFrame): SOLID elements to validate
            against (typically the same table the repair call itself
            used, e.g. a pocket's local solids_df) -- restricting this to
            a reasonably local subset keeps every classify_solid_badness
            call cheap regardless of overall mesh size.
        nodes_before_df (pd.DataFrame): node table BEFORE repair (used
            both to find which nodes moved, and as the origin for the
            outer hard_displacement_cap check).
        nodes_after_df (pd.DataFrame): node table immediately AFTER a
            successful repair -- the starting point for smoothing, AND
            the origin for the primary max_smoothing_displacement check.
        hard_displacement_cap (float): true Euclidean-radius OUTER bound
            (e.g. HARD_DISPLACEMENT_CAP_MM), measured from each node's
            ORIGINAL pre-repair position -- a belt-and-suspenders safety
            net, not the primary smoothing-strength control.
        max_smoothing_displacement (float): the PRIMARY, small budget for
            how far smoothing alone may move any node, measured from that
            node's POST-REPAIR position -- keep this small (e.g. a
            fraction of hard_displacement_cap) so smoothing stays a
            cosmetic nudge, not a reshape.
        blend_ring_layers (int): how many adjacency rings beyond the
            core moved nodes are also allowed to move during smoothing
            (default 2).
        max_iterations (int): maximum smoothing sweeps (default 3).
        initial_blend_factor (float): starting blend strength per sweep
            (default 0.35 -- each smoothable node moves 35% of the way
            from its current position to its neighbors' average); halved
            on any rejected sweep.
        min_blend_factor (float): stop smoothing once the halved blend
            factor drops below this (default 0.05).

    Returns:
        tuple:
            smoothed_nodes_df (pd.DataFrame): node table after smoothing
                (identical to nodes_after_df if no safe smoothing move
                was ever found).
            n_sweeps_applied (int): how many sweeps were actually
                accepted (0 if smoothing never found a safe move, e.g.
                nothing moved during repair, or every attempted sweep
                was unsafe even at the smallest tried blend factor).
    """
    before_coord = nodes_before_df.set_index('NodeID')[['x', 'y', 'z']]
    after_coord = nodes_after_df.set_index('NodeID')[['x', 'y', 'z']]
    common_ids = before_coord.index.intersection(after_coord.index)
    diff = (before_coord.loc[common_ids] - after_coord.loc[common_ids])
    moved_mask = (diff['x'] != 0) | (diff['y'] != 0) | (diff['z'] != 0)
    core_moved_ids = set(common_ids[moved_mask].tolist())
    if not core_moved_ids:
        return nodes_after_df, 0  # repair didn't move anything -- nothing to smooth

    all_element_node_lists = _unique_node_lists_fast(elements_all_df)
    nid_to_elem_indices = {}
    for idx, nl in enumerate(all_element_node_lists):
        for nid in nl:
            nid_to_elem_indices.setdefault(nid, set()).add(idx)

    smoothable_ids = _expand_node_region(
        core_moved_ids, nid_to_elem_indices, all_element_node_lists, blend_ring_layers
    )
    zone_elem_indices = set()
    for nid in smoothable_ids:
        zone_elem_indices.update(nid_to_elem_indices.get(nid, set()))
    if not zone_elem_indices:
        return nodes_after_df, 0
    zone_elements_df = elements_all_df.iloc[sorted(zone_elem_indices)].reset_index(drop=True)

    node_adjacency = build_node_to_node_adjacency(zone_elements_df)
    smoothable_ids = sorted(nid for nid in smoothable_ids if nid in node_adjacency)
    if not smoothable_ids:
        return nodes_after_df, 0

    baseline_status = classify_solid_badness(zone_elements_df, nodes_after_df)
    baseline_bad_eids = set(baseline_status.loc[baseline_status['is_bad'], 'EID'].tolist())

    # Fixed reference for the PRIMARY (tight) smoothing budget -- each
    # node's position right after repair, captured once and never updated
    # across sweeps, so the smoothing-only budget cannot be "spent" once
    # per sweep and compound into a much larger cumulative move.
    post_repair_coord = after_coord

    current_nodes_df = nodes_after_df.copy()
    n_sweeps_applied = 0
    blend_factor = initial_blend_factor

    while n_sweeps_applied < max_iterations and blend_factor >= min_blend_factor:
        current_pos = current_nodes_df.set_index('NodeID')[['x', 'y', 'z']]
        proposed = current_pos.copy()
        moved_any = False

        for nid in smoothable_ids:
            neighbors = node_adjacency[nid]
            neighbor_pos = current_pos.loc[list(neighbors)].to_numpy(dtype=float)
            target = neighbor_pos.mean(axis=0)
            cur = current_pos.loc[nid].to_numpy(dtype=float)
            candidate = cur + blend_factor * (target - cur)

            # PRIMARY guard: stay within a small, dedicated smoothing-only
            # budget of this node's POST-REPAIR position -- keeps this
            # pass a cosmetic nudge regardless of how many sweeps run or
            # how strong the local curvature pulls the Laplacian target.
            if nid in post_repair_coord.index:
                post_pos = post_repair_coord.loc[nid].to_numpy(dtype=float)
                direction = candidate - post_pos
                dist = np.linalg.norm(direction)
                if dist > max_smoothing_displacement and dist > 1e-12:
                    candidate = post_pos + direction * (max_smoothing_displacement / dist)

            # Belt-and-suspenders OUTER guard: never exceed the true global
            # hard cap from the node's ORIGINAL pre-repair position either
            # (should rarely bind given the tighter guard above).
            if nid in before_coord.index:
                origin = before_coord.loc[nid].to_numpy(dtype=float)
                direction2 = candidate - origin
                dist2 = np.linalg.norm(direction2)
                if dist2 > hard_displacement_cap and dist2 > 1e-12:
                    candidate = origin + direction2 * (hard_displacement_cap / dist2)

            if not np.allclose(candidate, cur):
                moved_any = True
            proposed.loc[nid] = candidate

        if not moved_any:
            break

        proposed_df = proposed.reset_index()
        trial_status = classify_solid_badness(zone_elements_df, proposed_df)
        newly_bad = set(trial_status.loc[trial_status['is_bad'], 'EID'].tolist()) - baseline_bad_eids

        if newly_bad:
            blend_factor *= 0.5
            continue

        current_nodes_df = proposed_df
        n_sweeps_applied += 1

    return current_nodes_df, n_sweeps_applied


def repair_distorted_shells(elements_bad_shell_df, elements_all_shell_df,
                             solid_element_node_lists, node_to_solid_indices,
                             nodes_df,
                             max_displacement=2.0, band_floor=0.5, band_ceiling=0.7,
                             volume_epsilon=1e-6,
                             free_node_layers=2, constraint_layers=2,
                             boundary_growth=0.0, growth_stiffness=0.25):
    """
    Repair distorted SHELL elements (tri3/quad4) by pushing their
    compute_shell_jacobian jacobian_ratio into a two-sided [band_floor,
    band_ceiling] target band, reusing the SAME scipy SLSQP/trust-constr
    hybrid constrained-optimization machinery validated for solids in
    repair_inverted_elements. This is that function's SHELL sibling --
    a separate function, NOT a modification of it; solids keep their own,
    unchanged, floor-only repair path.

    *** ASSUMED WORKFLOW ORDER (confirmed with user): call this ONLY
    after ALL solid pockets have already been fully repaired. ***
    solid_element_node_lists / node_to_solid_indices (from
    build_solid_node_adjacency, called ONCE by the driver on the FULL,
    already-repaired solid element table) are read-only lookups used
    here purely to find which solid elements a shell pocket's free nodes
    touch -- this function never re-optimizes solids, it only PROTECTS
    them (floor-only, non-regression) from being silently re-broken by a
    shell repair moving a node they share (confirmed a REAL, common case
    on the real mesh: 95.7% of nodes touched by the actual bad shells
    also touch a solid element -- see build_solid_node_adjacency's own
    docstring).

    ZONE CONSTRUCTION (mirrors repair_inverted_elements' pattern,
    generalized across element types):
      1. Seed nodes = every unique node referenced by
         elements_bad_shell_df.
      2. free_nids_set / constraint_nids_set expand via SHELL-to-SHELL
         adjacency only (elements_all_shell_df), exactly like the solid
         function's own free_node_layers/constraint_layers knobs -- solid
         connectivity is never used to grow the free/constrained node
         sets, only to find which solids to PROTECT afterward (step 4).
      3. The shell repair zone = every SHELL element (from
         elements_all_shell_df) touching constraint_nids_set.
      4. Touching SOLID elements = every solid element (via
         node_to_solid_indices) sharing a node with free_nids_set ONLY
         (not the wider constraint zone) -- a solid element can only be
         put at risk if one of ITS OWN nodes can actually move, so basing
         this on the free set alone avoids pulling in solids that could
         never be affected anyway.
      5. Every node referenced by either the shell repair zone or the
         touching solid elements becomes a zone-local coordinate (free if
         already in free_nids_set, otherwise a fixed anchor needed only
         to evaluate its element's own value/gradient) -- exactly like
         repair_inverted_elements' own repair_zone_node_lists handling.

    CONSTRAINT ROWS, per element group (mirrors _build_floor_and_bad_mask/
    _constraint_parts_hard/_constraint_parts_soft, generalized to a
    TWO-SIDED band for shells -- see build_shell_floor_ceiling's own
    docstring for the underlying floor/ceiling construction rule):
      - quad4: ONE jacobian_ratio value per element (hard argmin variant
        for the baseline/final-check, softmin variant for the SLSQP-
        facing constraint -- see _batched_quad_jacobian_ratio_value_and_
        grad / _batched_quad_jacobian_ratio_softmin_value_and_grad),
        contributing TWO constraint rows: (value - floor) >= 0 and
        (ceiling - value) >= 0.
      - tri3: ONE jacobian_ratio value per element (single exact affine
        formula, no hard/soft distinction needed -- see
        _batched_tri3_jacobian_ratio_value_and_grad), same two-row
        (floor, ceiling) treatment.
      - touching solids (group8/group6/group4): IDENTICAL corner-min +
        centroid (or single tet4) rows to repair_inverted_elements' own
        floor-only treatment -- reuses the EXISTING _batched_hexlike_*/
        _batched_tet_value_and_grad functions verbatim, no new solid
        math. Floor-only, no ceiling (see build_shell_floor_ceiling's
        docstring for why), and never on the "bad" list (solids are
        already fully repaired before this function is ever called).

    CONTINUATION SCHEDULE: unlike solids (whose bad-element floor only
    ever needs to be RAISED, from a very negative baseline up to
    +volume_epsilon), a bad shell's baseline can sit on EITHER side of
    the band -- too low (needs its floor raised) OR too high (needs its
    ceiling lowered) -- so both the floor and ceiling for BAD rows are
    ramped LINEARLY from their own baseline value to the true target
    (band_floor/band_ceiling) over however many stages
    _SHELL_CONTINUATION_STEP requires. Neighbour (non-bad) shell rows and
    ALL solid rows are never ramped -- their non-regression floor/ceiling
    is already satisfied at the baseline by construction (build_shell_
    floor_ceiling), exactly like repair_inverted_elements' own neighbour
    treatment.

    boundary_growth is NOT YET SUPPORTED for shells (see the ValueError
    raised below if a caller passes a nonzero value) -- solids' outward
    pre-bias relies on _collect_boundary_nodes, which is solid-FACE-
    topology-specific (via _solid_face_node_lists) and has no shell-EDGE
    equivalent built/validated yet. The parameter is kept in this
    function's signature purely so a caller's escalation-ladder-building
    code (which already knows how to construct these attempt dicts for
    repair_inverted_elements) can pass the same shape of arguments
    without special-casing shells, as long as it always leaves this one
    at its default of 0.0.

    Inputs:
        elements_bad_shell_df (pd.DataFrame): The user's "must fix" bad
            shell elements (e.g. parsed from shell_report.k), tri3/quad4
            mixed, columns include 'EID', 'PID', 'NID1'..'NID4'.
        elements_all_shell_df (pd.DataFrame): ALL shell elements in the
            part/model (bad + neighbours) -- used for shell-to-shell
            adjacency expansion. Must NOT include any solid rows.
        solid_element_node_lists, node_to_solid_indices: From
            build_solid_node_adjacency(solid_elements_all_df), called
            ONCE by the caller on the FULL (already fully repaired) solid
            element table -- NOT rebuilt here.
        nodes_df (pd.DataFrame): Node table (NodeID, x, y, z) -- the
            SAME, already solid-repaired node table the solid phase
            produced.
        max_displacement (float): Max per-axis movement for any free node.
        band_floor, band_ceiling (float): Two-sided jacobian_ratio target
            band for bad shells. Defaults 0.5/0.7 per the user's explicit
            requirement.
        volume_epsilon (float): Non-regression floor for touched solid
            neighbours (same meaning as repair_inverted_elements' own
            parameter).
        free_node_layers, constraint_layers (int): Shell-to-shell
            adjacency expansion depth (same meaning as
            repair_inverted_elements).
        boundary_growth, growth_stiffness (float): See "NOT YET
            SUPPORTED" note above -- must stay at 0.0 for shells.

    Returns:
        tuple:
            - updated_nodes_df (pd.DataFrame): Repaired node table
              (NodeID, x, y, z).
            - result (OptimizeResult): Full scipy result object (None if
              there was nothing to repair).
            - info (dict): Summary with keys 'n_free_nodes',
              'n_repair_zone_elements', 'n_touching_solid_elements',
              'n_still_out_of_band' (shell analogue of solids'
              'n_still_negative'), 'n_neighbor_regressions',
              'max_neighbor_regression' (analogous to repair_inverted_
              elements' own fields of the same name -- see its docstring
              for the real production bug this guards against --
              generalized here to cover every OTHER quad/tri row's own
              [floor, ceiling] band AND every solid_floor row, none of
              which are in the tracked bad-shell list),
              'max_displacement_achieved', 'converged'.
    """
    from scipy.optimize import minimize, Bounds, OptimizeResult
    from scipy.sparse import coo_matrix, identity as sparse_identity

    if max_displacement <= 0:
        raise ValueError("max_displacement must be > 0.")
    if band_floor >= band_ceiling:
        raise ValueError(f"band_floor ({band_floor}) must be strictly less than band_ceiling ({band_ceiling}).")
    if volume_epsilon < 0:
        raise ValueError("volume_epsilon must be >= 0.")
    if free_node_layers < 0 or constraint_layers < 0:
        raise ValueError("free_node_layers and constraint_layers must be >= 0.")
    if boundary_growth < 0:
        raise ValueError("boundary_growth must be >= 0.")
    if boundary_growth > 0:
        raise NotImplementedError(
            "boundary_growth is not yet supported for shell repair (no shell-"
            "edge boundary detector has been built/validated -- see "
            "_collect_boundary_nodes, which is solid-face-topology-specific). "
            "Leave boundary_growth at its default of 0.0 for shell pockets."
        )
    if growth_stiffness < 0:
        raise ValueError("growth_stiffness must be >= 0.")

    # Seed nodes: all unique nodes referenced by bad shells.
    bad_shell_node_lists = _unique_node_lists_fast(elements_bad_shell_df)
    bad_nids_set = set()
    for nodes in bad_shell_node_lists:
        bad_nids_set.update(nodes)

    if not bad_nids_set:
        unchanged_nodes_df = nodes_df[['NodeID', 'x', 'y', 'z']].copy()
        info = {
            'n_free_nodes': 0,
            'n_repair_zone_elements': 0,
            'n_touching_solid_elements': 0,
            'n_still_out_of_band': 0,
            'max_displacement_achieved': 0.0,
            'converged': True,
        }
        return unchanged_nodes_df, None, info

    # Shell-only node adjacency (used ONLY to expand free/constrained node
    # sets -- solid connectivity is deliberately never used for this, see
    # docstring point 2).
    all_shell_node_lists = _unique_node_lists_fast(elements_all_shell_df)
    nid_to_shell_elem_indices = {}
    for elem_idx, unique_nids in enumerate(all_shell_node_lists):
        for nid in unique_nids:
            nid_to_shell_elem_indices.setdefault(nid, set()).add(elem_idx)

    free_nids_set = _expand_node_region(
        bad_nids_set, nid_to_shell_elem_indices, all_shell_node_lists, free_node_layers
    )
    constraint_nids_set = _expand_node_region(
        free_nids_set, nid_to_shell_elem_indices, all_shell_node_lists, constraint_layers
    )
    free_nids_list = sorted(free_nids_set)

    # Shell repair zone: every shell element touching the constrained node set.
    shell_repair_indices = [
        elem_idx
        for elem_idx, elem_nodes in enumerate(all_shell_node_lists)
        if set(elem_nodes) & constraint_nids_set
    ]
    shell_repair_zone_node_lists = [all_shell_node_lists[i] for i in shell_repair_indices]
    bad_eids_set = set(int(e) for e in elements_bad_shell_df['EID'].tolist())
    shell_repair_zone_eids = elements_all_shell_df['EID'].to_numpy()[shell_repair_indices]

    # Touching solids: based on the FREE node set only (see docstring
    # point 4) -- a solid element can only be put at risk if one of its
    # own nodes can actually move.
    touching_solid_indices = sorted(find_touching_solid_elements(free_nids_set, node_to_solid_indices))
    touching_solid_node_lists = [solid_element_node_lists[i] for i in touching_solid_indices]

    # PERFORMANCE: restrict every coordinate array to only the nodes
    # actually touched by the shell repair zone or the protected solid
    # neighbours (see repair_inverted_elements' own comment on this same
    # pattern for the full rationale).
    zone_nids_set = set(free_nids_set)
    for nodes in shell_repair_zone_node_lists:
        zone_nids_set.update(nodes)
    for nodes in touching_solid_node_lists:
        zone_nids_set.update(nodes)
    zone_nids_list = sorted(zone_nids_set)
    zone_nid_to_local = {nid: i for i, nid in enumerate(zone_nids_list)}

    node_coord = nodes_df.set_index('NodeID')[['x', 'y', 'z']]
    zone_coords0 = node_coord.loc[zone_nids_list].values.astype(float)  # (K, 3), K << N

    free_zone_idx = np.array([zone_nid_to_local[nid] for nid in free_nids_list], dtype=int)
    free_orig_coords = zone_coords0[free_zone_idx].copy()

    zone_local_to_freevar = np.full(len(zone_nids_list), -1, dtype=int)
    for var_idx, nid in enumerate(free_nids_list):
        zone_local_to_freevar[zone_nid_to_local[nid]] = var_idx

    # Group shell repair-zone elements by unique-node count (3 tri / 4
    # quad), tracking is_bad in the same row order (mirrors
    # repair_inverted_elements' group_rows/group_is_bad construction).
    shell_group_rows = {3: [], 4: []}
    shell_group_is_bad = {3: [], 4: []}
    for eid, nodes in zip(shell_repair_zone_eids, shell_repair_zone_node_lists):
        n = len(nodes)
        if n not in shell_group_rows:
            raise ValueError(
                f"Unsupported SHELL element node count in repair zone: found {n} "
                "unique real nodes; supported counts are 3, 4."
            )
        local = [zone_nid_to_local[nid] for nid in nodes]
        shell_group_rows[n].append(local)
        shell_group_is_bad[n].append(int(eid) in bad_eids_set)

    tri_idx = np.array(shell_group_rows[3], dtype=int) if shell_group_rows[3] else np.zeros((0, 3), dtype=int)
    quad_idx = np.array(shell_group_rows[4], dtype=int) if shell_group_rows[4] else np.zeros((0, 4), dtype=int)
    tri_is_bad = np.array(shell_group_is_bad[3], dtype=bool)
    quad_is_bad = np.array(shell_group_is_bad[4], dtype=bool)

    # Group touching solid neighbours by unique-node count (4/6/8),
    # IDENTICAL grouping/padding convention to repair_inverted_elements
    # (penta6's ridge node repeated into slots 6-8) -- these are always
    # floor-only, never on any "bad" list (solids are already repaired).
    solid_group_rows = {4: [], 6: [], 8: []}
    for nodes in touching_solid_node_lists:
        n = len(nodes)
        if n not in solid_group_rows:
            raise ValueError(
                f"Unsupported SOLID element node count among shell-pocket "
                f"neighbours: found {n} unique real nodes; supported counts "
                "are 4, 6, 8."
            )
        local = [zone_nid_to_local[nid] for nid in nodes]
        if n == 6:
            local = local[:5] + [local[5], local[5], local[5]]
        solid_group_rows[n].append(local)

    group4_idx = np.array(solid_group_rows[4], dtype=int) if solid_group_rows[4] else np.zeros((0, 4), dtype=int)
    group6_idx = np.array(solid_group_rows[6], dtype=int) if solid_group_rows[6] else np.zeros((0, 8), dtype=int)
    group8_idx = np.array(solid_group_rows[8], dtype=int) if solid_group_rows[8] else np.zeros((0, 8), dtype=int)

    n_constraints = (2 * len(quad_idx) + 2 * len(tri_idx)
                     + 2 * len(group8_idx) + 2 * len(group6_idx) + len(group4_idx))

    group8_has_free = (np.any(zone_local_to_freevar[group8_idx] != -1, axis=1)
                        if len(group8_idx) else np.zeros(0, dtype=bool))
    group6_has_free = (np.any(zone_local_to_freevar[group6_idx] != -1, axis=1)
                        if len(group6_idx) else np.zeros(0, dtype=bool))

    # Frozen (baseline-only) conventions: quad4's reference normal/softmin
    # sharpness, and the solid groups' softmin sharpness -- ALL calibrated
    # once from zone_coords0 and held fixed through optimization (see
    # _frozen_quad_ref_normal / _frozen_quad_softmin_beta / _frozen_
    # softmin_beta's own docstrings for why this is correct, not just
    # convenient).
    quad_n_ref = _frozen_quad_ref_normal(zone_coords0, quad_idx)
    quad_beta = _frozen_quad_softmin_beta(zone_coords0, quad_idx, quad_n_ref)
    group8_beta = _frozen_softmin_beta(zone_coords0, group8_idx, 8)
    group6_beta = _frozen_softmin_beta(zone_coords0, group6_idx, 4)

    # Baseline (pre-optimization) values -- HARD variant for quad/tri (per
    # build_shell_floor_ceiling's own docstring), hard-vs-soft blended for
    # the solid corner-min rows (same has_free-dependent blending rule as
    # repair_inverted_elements' _build_floor_and_bad_mask, for the exact
    # same reason: a fully-pinned solid element's true corner-min can
    # never move, so using the softer baseline there creates zero risk
    # and avoids a permanently-unsatisfiable constraint).
    quad_baseline = (_batched_quad_jacobian_ratio_value_and_grad(zone_coords0[quad_idx], quad_n_ref)[0]
                     if len(quad_idx) else np.zeros(0))
    tri_baseline = (_batched_tri3_jacobian_ratio_value_and_grad(zone_coords0[tri_idx])[0]
                    if len(tri_idx) else np.zeros(0))

    v8_base = (np.where(
        group8_has_free,
        _batched_hexlike_value_and_grad(zone_coords0[group8_idx], 8)[0],
        _batched_hexlike_softmin_value_and_grad(zone_coords0[group8_idx], 8, group8_beta)[0],
    ) if len(group8_idx) else np.zeros(0))
    v8c_base = (_batched_hexlike_centroid_value_and_grad(zone_coords0[group8_idx])[0]
                if len(group8_idx) else np.zeros(0))
    v6_base = (np.where(
        group6_has_free,
        _batched_hexlike_value_and_grad(zone_coords0[group6_idx], 4)[0],
        _batched_hexlike_softmin_value_and_grad(zone_coords0[group6_idx], 4, group6_beta)[0],
    ) if len(group6_idx) else np.zeros(0))
    v6c_base = (_batched_hexlike_centroid_value_and_grad(zone_coords0[group6_idx])[0]
                if len(group6_idx) else np.zeros(0))
    v4_base = (_batched_tet_value_and_grad(zone_coords0[group4_idx])[0]
               if len(group4_idx) else np.zeros(0))

    # Order MUST match the order constraint_vec/constraint_jac's
    # 'solid_floor'-kind parts are concatenated in below.
    solid_baseline_volumes = np.concatenate([v8_base, v8c_base, v6_base, v6c_base, v4_base])

    band = build_shell_floor_ceiling(
        quad_baseline, quad_is_bad, tri_baseline, tri_is_bad,
        solid_baseline_volumes, band_floor=band_floor, band_ceiling=band_ceiling,
        volume_epsilon=volume_epsilon,
    )
    final_quad_floor, final_quad_ceiling = band['quad_floor'], band['quad_ceiling']
    final_tri_floor, final_tri_ceiling = band['tri_floor'], band['tri_ceiling']
    solid_floor_vec = band['solid_floor']  # never ramped -- already correct via min(baseline, volume_epsilon)

    # Mutable, per-stage floor/ceiling for the two-sided band rows;
    # neighbour (non-bad) rows are already at their final, never-ramped
    # value from the start (see docstring's CONTINUATION SCHEDULE section).
    quad_floor_vec = final_quad_floor.copy()
    quad_ceiling_vec = final_quad_ceiling.copy()
    tri_floor_vec = final_tri_floor.copy()
    tri_ceiling_vec = final_tri_ceiling.copy()

    # boundary_growth is guaranteed 0.0 past the validation above, so the
    # objective is the plain "stay close to original position" quadratic
    # -- no outward-bias target term needed (unlike repair_inverted_
    # elements, which must support boundary_growth > 0).
    low_bounds = free_orig_coords - max_displacement
    high_bounds = free_orig_coords + max_displacement
    x0 = np.minimum(np.maximum(free_orig_coords, low_bounds), high_bounds).flatten()
    orig_x0 = free_orig_coords.flatten()
    n_free_vars = len(orig_x0)

    dense_jacobian_size = n_constraints * n_free_vars
    # *** SHELL-SPECIFIC: ALWAYS use trust-constr, never dense SLSQP ***
    # (unlike repair_inverted_elements' solid dispatch, which genuinely
    # picks per-pocket based on size). Confirmed via a REAL, whole-mesh
    # production run (real F05 data, 202 shell pockets): pockets dispatched
    # to dense SLSQP got COMPLETELY STUCK (zero node movement at all,
    # scipy reporting "Positive directional derivative for linesearch"
    # right at iteration 1) in 149 / 187 cases (80%) -- while EVERY one of
    # the 15 pockets dispatched to trust-constr (by virtue of being large
    # enough to cross _DENSE_JACOBIAN_SIZE_THRESHOLD) made at least SOME
    # real progress, none got stuck at zero. Root cause: a bad shell's
    # constraint is TWO-SIDED (floor AND ceiling simultaneously), unlike a
    # solid's one-sided floor -- when several topologically-coupled shells
    # in one pocket need to move in different/conflicting directions to
    # each land inside a narrow band, SLSQP's dense active-set QP
    # linearization frequently has no feasible descent direction and
    # fails immediately; trust-constr's interior-point method does not
    # share this failure mode. This makes the SLSQP branch actively
    # harmful for shells (not just slower) regardless of pocket size, so
    # dense_jacobian_size is still computed/logged for visibility but
    # never used to route to SLSQP here.
    use_sparse_solver = True
    print(
        f"  Solver dispatch: n_free_vars={n_free_vars}, n_constraints={n_constraints} "
        f"-> estimated dense Jacobian size={dense_jacobian_size:,} "
        f"-> using trust-constr (sparse) -- ALWAYS forced for shells, "
        f"regardless of size (see this function's dispatch comment for why)."
    )

    def _zone_coords_from_x(x):
        coords = zone_coords0.copy()
        coords[free_zone_idx] = x.reshape(-1, 3)
        return coords

    def objective(x):
        return float(np.sum((x - orig_x0) ** 2))

    def objective_jac(x):
        return 2.0 * (x - orig_x0)

    def objective_hess(x):
        return 2.0 * sparse_identity(n_free_vars, format='csr')

    if use_sparse_solver:
        bounds = Bounds(orig_x0 - max_displacement, orig_x0 + max_displacement)
    else:
        bounds = [(orig_x0[j] - max_displacement, orig_x0[j] + max_displacement)
                  for j in range(n_free_vars)]

    def _constraint_parts_hard(coords):
        """One (kind, values, slot_grad, slot_idx) tuple per constraint
        GROUP -- kind 'quad'/'tri' rows expand into TWO constraint rows
        each (floor, ceiling) below; kind 'solid_floor' rows expand into
        ONE (floor only). Uses EXACT (non-smooth) values throughout --
        for the non-regression baseline and the final pass/fail report,
        NOT the SLSQP-facing constraint (see _constraint_parts_soft)."""
        parts = []
        if len(quad_idx):
            v, g = _batched_quad_jacobian_ratio_value_and_grad(coords[quad_idx], quad_n_ref)
            parts.append(('quad', v, g, quad_idx))
        if len(tri_idx):
            v, g = _batched_tri3_jacobian_ratio_value_and_grad(coords[tri_idx])
            parts.append(('tri', v, g, tri_idx))
        if len(group8_idx):
            v8, g8 = _batched_hexlike_value_and_grad(coords[group8_idx], 8)
            parts.append(('solid_floor', v8, g8, group8_idx))
            v8c, g8c = _batched_hexlike_centroid_value_and_grad(coords[group8_idx])
            parts.append(('solid_floor', v8c, g8c, group8_idx))
        if len(group6_idx):
            v6, g6 = _batched_hexlike_value_and_grad(coords[group6_idx], 4)
            parts.append(('solid_floor', v6, g6, group6_idx))
            v6c, g6c = _batched_hexlike_centroid_value_and_grad(coords[group6_idx])
            parts.append(('solid_floor', v6c, g6c, group6_idx))
        if len(group4_idx):
            v4, g4 = _batched_tet_value_and_grad(coords[group4_idx])
            parts.append(('solid_floor', v4, g4, group4_idx))
        return parts

    def _constraint_parts_soft(coords):
        """Same shape/contract as _constraint_parts_hard, but quad4/hex8/
        penta6 corner-min-style rows use their smooth softmin proxy
        instead of the hard min (see _batched_quad_jacobian_ratio_softmin_
        value_and_grad / _batched_hexlike_softmin_value_and_grad's own
        docstrings). tri3/centroid/tet4 rows are exact affine/polynomial
        formulas already, so they are identical between hard and soft."""
        parts = []
        if len(quad_idx):
            v, g = _batched_quad_jacobian_ratio_softmin_value_and_grad(coords[quad_idx], quad_n_ref, quad_beta)
            parts.append(('quad', v, g, quad_idx))
        if len(tri_idx):
            v, g = _batched_tri3_jacobian_ratio_value_and_grad(coords[tri_idx])
            parts.append(('tri', v, g, tri_idx))
        if len(group8_idx):
            v8, g8 = _batched_hexlike_softmin_value_and_grad(coords[group8_idx], 8, group8_beta)
            parts.append(('solid_floor', v8, g8, group8_idx))
            v8c, g8c = _batched_hexlike_centroid_value_and_grad(coords[group8_idx])
            parts.append(('solid_floor', v8c, g8c, group8_idx))
        if len(group6_idx):
            v6, g6 = _batched_hexlike_softmin_value_and_grad(coords[group6_idx], 4, group6_beta)
            parts.append(('solid_floor', v6, g6, group6_idx))
            v6c, g6c = _batched_hexlike_centroid_value_and_grad(coords[group6_idx])
            parts.append(('solid_floor', v6c, g6c, group6_idx))
        if len(group4_idx):
            v4, g4 = _batched_tet_value_and_grad(coords[group4_idx])
            parts.append(('solid_floor', v4, g4, group4_idx))
        return parts

    def constraint_vec(x):
        coords = _zone_coords_from_x(x)
        parts = _constraint_parts_soft(coords)
        rows = []
        solid_offset = 0
        for kind, v, g, idx in parts:
            if kind == 'quad':
                rows.append(v - quad_floor_vec)
                rows.append(quad_ceiling_vec - v)
            elif kind == 'tri':
                rows.append(v - tri_floor_vec)
                rows.append(tri_ceiling_vec - v)
            else:
                n_rows = len(v)
                rows.append(v - solid_floor_vec[solid_offset:solid_offset + n_rows])
                solid_offset += n_rows
        if not rows:
            return np.zeros(0)
        return np.concatenate(rows)

    def constraint_jac(x):
        """Dense (SLSQP) or sparse CSR (trust-constr) Jacobian -- see
        repair_inverted_elements' own constraint_jac for the full
        dense-vs-sparse rationale, identical here. Each 'quad'/'tri' part
        contributes TWO scattered row-blocks (floor: +gradient, ceiling:
        -gradient, since d(ceiling - v)/dx = -dv/dx); each 'solid_floor'
        part contributes ONE (matching constraint_vec's row layout
        exactly, so row_offset stays in lockstep between the two)."""
        coords = _zone_coords_from_x(x)
        parts = _constraint_parts_soft(coords)
        if not parts:
            empty = coo_matrix((0, n_free_vars))
            return empty.toarray() if not use_sparse_solver else empty.tocsr()

        rows_list, cols_list, vals_list = [], [], []
        row_offset = 0
        for kind, v, g, idx in parts:
            r, c, val = _scatter_group_jacobian(g, idx, zone_local_to_freevar, row_offset)
            rows_list.append(r); cols_list.append(c); vals_list.append(val)
            row_offset += len(v)
            if kind in ('quad', 'tri'):
                r2, c2, val2 = _scatter_group_jacobian(-g, idx, zone_local_to_freevar, row_offset)
                rows_list.append(r2); cols_list.append(c2); vals_list.append(val2)
                row_offset += len(v)

        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        vals = np.concatenate(vals_list)
        J = coo_matrix((vals, (rows, cols)), shape=(n_constraints, n_free_vars))
        return J.tocsr() if use_sparse_solver else J.toarray()

    print(
        f"  Shell repair zone: {len(free_nids_list)} free nodes, "
        f"{len(quad_idx)} quad4 + {len(tri_idx)} tri3 constrained shells, "
        f"{len(touching_solid_indices)} protected neighbour solids "
        f"(free_node_layers={free_node_layers}, constraint_layers={constraint_layers})"
    )

    zone_extent = float(np.ptp(zone_coords0, axis=0).max()) if len(zone_coords0) else 1.0
    jitter_scale = max(zone_extent * 1e-3, 1e-6)
    _self_check_analytic_gradient(constraint_vec, constraint_jac, x0, n_free_vars, n_constraints,
                                   jitter_scale=jitter_scale)

    iteration_state = {'count': 0, 'last_xk': None, 'early_stop_reason': None}

    def _make_stage_callback_slsqp():
        """SLSQP branch -- IDENTICAL mechanics/rationale to
        repair_inverted_elements' own _make_stage_callback_slsqp
        (including the StopIteration-must-be-caught-by-the-caller fix),
        just closing over THIS function's constraint_vec/n_constraints/
        n_free_vars/orig_x0/max_displacement/iteration_state instead."""
        history = []
        move_history = []
        max_theoretical_move = max_displacement * np.sqrt(3.0)

        def _callback(xk):
            iteration_state['count'] += 1
            iteration_state['last_xk'] = xk.copy()
            worst = float(constraint_vec(xk).min()) if n_constraints else 0.0
            move = float(np.linalg.norm((xk - orig_x0).reshape(-1, 3), axis=1).max()) if n_free_vars else 0.0
            print(f"    iter {iteration_state['count']:4d}: worst constraint={worst: .6e}, max move={move:.4f}")

            history.append(worst)
            if len(history) > _STALL_WINDOW:
                history.pop(0)
            if (len(history) == _STALL_WINDOW and worst < -_STALL_MIN_VIOLATION
                    and (max(history) - min(history)) < _STALL_RELATIVE_SWING * abs(worst)):
                print(
                    f"    -> stalled: worst constraint has not improved by more than "
                    f"{_STALL_RELATIVE_SWING:.0%} over the last {_STALL_WINDOW} iterations "
                    f"(still violated at {worst:.4f}). Stopping this stage early and "
                    f"handing off to the next stage/attempt."
                )
                iteration_state['early_stop_reason'] = 'stalled: no meaningful improvement in worst constraint'
                raise StopIteration(iteration_state['early_stop_reason'])

            move_history.append(move)
            if len(move_history) > _BOUND_SATURATION_WINDOW:
                move_history.pop(0)
            mean_move = sum(move_history) / len(move_history)
            if (len(move_history) == _BOUND_SATURATION_WINDOW and worst < -_BOUND_SATURATION_MIN_VIOLATION
                    and mean_move >= _BOUND_SATURATION_RATIO * max_theoretical_move):
                print(
                    f"    -> diverging: mean max-move ({mean_move:.4f}) has stayed within "
                    f"{_BOUND_SATURATION_RATIO:.0%} of the theoretical bound-diagonal limit "
                    f"({max_theoretical_move:.4f}) over the last {_BOUND_SATURATION_WINDOW} "
                    f"iterations while still meaningfully violated ({worst:.4f}). Stopping this "
                    f"stage early and handing off to the next, less-constrained stage/attempt "
                    f"rather than exhausting the iteration budget."
                )
                iteration_state['early_stop_reason'] = 'diverging: pinned at displacement bound while still violated'
                raise StopIteration(iteration_state['early_stop_reason'])

        return _callback

    def _make_stage_callback_trust_constr():
        """trust-constr branch -- IDENTICAL mechanics/rationale to
        repair_inverted_elements' own _make_stage_callback_trust_constr,
        just closing over THIS function's locals instead."""
        history = []
        move_history = []
        max_window = max(_STALL_WINDOW, _FEASIBLE_STALL_WINDOW, _BOUND_SATURATION_WINDOW)
        max_theoretical_move = max_displacement * np.sqrt(3.0)

        def _callback(xk, state):
            iteration_state['count'] += 1
            iteration_state['last_xk'] = xk.copy()
            worst = float(constraint_vec(xk).min()) if n_constraints else 0.0
            move = float(np.linalg.norm((xk - orig_x0).reshape(-1, 3), axis=1).max()) if n_free_vars else 0.0
            print(f"    iter {iteration_state['count']:4d}: worst constraint={worst: .6e}, max move={move:.4f}")

            history.append(worst)
            if len(history) > max_window:
                history.pop(0)
            move_history.append(move)
            if len(move_history) > max_window:
                move_history.pop(0)

            stall_window_vals = history[-_STALL_WINDOW:]
            if (len(stall_window_vals) == _STALL_WINDOW and worst < -_STALL_MIN_VIOLATION
                    and (max(stall_window_vals) - min(stall_window_vals)) < _STALL_RELATIVE_SWING * abs(worst)):
                print(
                    f"    -> stalled: worst constraint has not improved by more than "
                    f"{_STALL_RELATIVE_SWING:.0%} over the last {_STALL_WINDOW} iterations "
                    f"(still violated at {worst:.4f}). Stopping this stage early and "
                    f"handing off to the next stage/attempt."
                )
                return True

            feas_window_vals = history[-_FEASIBLE_STALL_WINDOW:]
            if (len(feas_window_vals) == _FEASIBLE_STALL_WINDOW and min(feas_window_vals) >= 0.0
                    and (max(feas_window_vals) - min(feas_window_vals)) < _FEASIBLE_STALL_ABS_SWING):
                print(
                    f"    -> feasible and stable: every constraint has been "
                    f"satisfied (worst >= 0) and stable (swing < "
                    f"{_FEASIBLE_STALL_ABS_SWING:.1e}) over the last "
                    f"{_FEASIBLE_STALL_WINDOW} iterations. Stopping this stage "
                    f"early rather than over-polishing past trust-constr's own "
                    f"much tighter internal optimality tolerance."
                )
                return True

            move_window_vals = move_history[-_BOUND_SATURATION_WINDOW:]
            mean_move = sum(move_window_vals) / len(move_window_vals) if move_window_vals else 0.0
            if (len(move_window_vals) == _BOUND_SATURATION_WINDOW and worst < -_BOUND_SATURATION_MIN_VIOLATION
                    and mean_move >= _BOUND_SATURATION_RATIO * max_theoretical_move):
                print(
                    f"    -> diverging: mean max-move ({mean_move:.4f}) has stayed within "
                    f"{_BOUND_SATURATION_RATIO:.0%} of the theoretical bound-diagonal limit "
                    f"({max_theoretical_move:.4f}) over the last {_BOUND_SATURATION_WINDOW} "
                    f"iterations while still meaningfully violated ({worst:.4f}). Stopping this "
                    f"stage early and handing off to the next, less-constrained stage/attempt "
                    f"rather than exhausting the iteration budget."
                )
                return True

            return False

        return _callback

    _make_stage_callback = _make_stage_callback_trust_constr if use_sparse_solver else _make_stage_callback_slsqp

    # Continuation schedule: ramp BAD quad4/tri3 rows' floor AND ceiling
    # linearly from their own baseline to the true target over several
    # solver calls (see docstring's CONTINUATION SCHEDULE section for why
    # BOTH sides can need ramping here, unlike solids). Neighbour shell
    # rows and all solid rows are never ramped -- already correct at
    # their final value from the start.
    def _bad_ramp_gap(final_vec, baseline_vec, is_bad):
        if not len(is_bad) or not np.any(is_bad):
            return np.zeros(0)
        return final_vec[is_bad] - baseline_vec[is_bad]

    quad_floor_gap = _bad_ramp_gap(final_quad_floor, quad_baseline, quad_is_bad)
    quad_ceiling_gap = _bad_ramp_gap(final_quad_ceiling, quad_baseline, quad_is_bad)
    tri_floor_gap = _bad_ramp_gap(final_tri_floor, tri_baseline, tri_is_bad)
    tri_ceiling_gap = _bad_ramp_gap(final_tri_ceiling, tri_baseline, tri_is_bad)

    all_gaps = [g for g in (quad_floor_gap, quad_ceiling_gap, tri_floor_gap, tri_ceiling_gap) if len(g)]
    max_gap = float(np.concatenate([np.abs(g) for g in all_gaps]).max()) if all_gaps else 0.0
    n_stages = (int(np.clip(np.ceil(max_gap / _SHELL_CONTINUATION_STEP), 1, _MAX_CONTINUATION_STAGES))
                if max_gap > 0 else 1)

    print(f"  Running {'trust-constr (sparse Jacobian)' if use_sparse_solver else 'SLSQP (dense Jacobian)'} optimisation...")
    if n_stages > 1:
        print(f"  Worst bad-shell band gap: {max_gap:.3f} -> using {n_stages} continuation stages.")

    opt_result = None
    for stage in range(1, n_stages + 1):
        alpha = stage / n_stages
        if len(quad_idx) and np.any(quad_is_bad):
            quad_floor_vec[quad_is_bad] = quad_baseline[quad_is_bad] + alpha * quad_floor_gap
            quad_ceiling_vec[quad_is_bad] = quad_baseline[quad_is_bad] + alpha * quad_ceiling_gap
        if len(tri_idx) and np.any(tri_is_bad):
            tri_floor_vec[tri_is_bad] = tri_baseline[tri_is_bad] + alpha * tri_floor_gap
            tri_ceiling_vec[tri_is_bad] = tri_baseline[tri_is_bad] + alpha * tri_ceiling_gap
        is_final_stage = (stage == n_stages)
        if n_stages > 1:
            print(f"  -- continuation stage {stage}/{n_stages} (alpha={alpha:.2f}) --")
        if use_sparse_solver:
            # Delegates to _run_trust_constr_stage (shared with repair_
            # inverted_elements -- see that function's own docstring,
            # right after _DENSE_JACOBIAN_SIZE_THRESHOLD) so the
            # numerical-overflow safety net is identical, not duplicated,
            # between solids and shells.
            opt_result = _run_trust_constr_stage(
                objective, x0, objective_jac, objective_hess, bounds,
                constraint_vec, constraint_jac, _make_stage_callback(),
                _FINAL_STAGE_MAXITER if is_final_stage else _INTERMEDIATE_STAGE_MAXITER,
                iteration_state,
            )
        else:
            try:
                opt_result = minimize(
                    objective,
                    x0,
                    jac=objective_jac,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[{'type': 'ineq', 'fun': constraint_vec, 'jac': constraint_jac}],
                    callback=_make_stage_callback(),
                    options={
                        'maxiter': _FINAL_STAGE_MAXITER if is_final_stage else _INTERMEDIATE_STAGE_MAXITER,
                        'ftol': _SLSQP_FTOL,
                        'disp': is_final_stage,
                    },
                )
            except StopIteration as e:
                last_xk = iteration_state['last_xk']
                if last_xk is None:
                    last_xk = x0
                opt_result = OptimizeResult(
                    x=last_xk, success=False, status=-1,
                    message=f"stopped early via callback: {e}",
                    fun=float(objective(last_xk)), nit=iteration_state['count'],
                )
        x0 = opt_result.x  # warm-start the next (closer-to-final-target) stage

    final_coords = _zone_coords_from_x(opt_result.x)
    final_parts = _constraint_parts_hard(final_coords)

    updated_nodes_df = nodes_df[['NodeID', 'x', 'y', 'z']].copy().set_index('NodeID')
    final_free_coords = final_coords[free_zone_idx]
    updated_nodes_df.loc[free_nids_list, ['x', 'y', 'z']] = final_free_coords
    updated_nodes_df = updated_nodes_df.reset_index()

    # Post-repair validation. TWO independent checks (same real-bug
    # rationale as repair_inverted_elements' own n_still_negative/
    # n_neighbor_regressions split -- see that function's docstring):
    #
    # 1) n_still_out_of_band: the user's original "must fix" bad shell
    #    list -- unchanged from before this fix.
    #
    # 2) n_neighbor_regressions: every OTHER row this pocket's own
    #    constraints protect -- non-bad quad/tri rows' own [floor,
    #    ceiling] band, AND every solid_floor row (solids have NO "bad"
    #    concept in this function -- pulled-in solid elements are always
    #    non-regression-only) -- did the optimizer's own diverging/
    #    early-stop path silently push one of THESE past its own
    #    protective floor/ceiling while the tracked bad-shell list
    #    happened to clear? volume_epsilon (already this function's own
    #    "smallest meaningful margin" parameter, used to build
    #    solid_floor's own floor) is reused as the "real regression, not
    #    solver noise" buffer for BOTH quality families here -- an
    #    equally appropriate noise floor for a 0-1 shell ratio metric as
    #    it is for solid volume.
    n_still_out_of_band = 0
    all_vals_list, all_floor_list, all_ceiling_list, all_is_bad_list = [], [], [], []
    solid_offset = 0
    for kind, v, g, idx in final_parts:
        n_rows = len(v)
        all_vals_list.append(v)
        if kind == 'quad' and len(quad_idx):
            n_still_out_of_band += int(np.sum((v[quad_is_bad] < band_floor) | (v[quad_is_bad] > band_ceiling)))
            all_floor_list.append(quad_floor_vec)
            all_ceiling_list.append(quad_ceiling_vec)
            all_is_bad_list.append(quad_is_bad)
        elif kind == 'tri' and len(tri_idx):
            n_still_out_of_band += int(np.sum((v[tri_is_bad] < band_floor) | (v[tri_is_bad] > band_ceiling)))
            all_floor_list.append(tri_floor_vec)
            all_ceiling_list.append(tri_ceiling_vec)
            all_is_bad_list.append(tri_is_bad)
        else:  # 'solid_floor' -- floor only, no ceiling, never "bad" here
            all_floor_list.append(solid_floor_vec[solid_offset:solid_offset + n_rows])
            all_ceiling_list.append(np.full(n_rows, np.inf))
            all_is_bad_list.append(np.zeros(n_rows, dtype=bool))
            solid_offset += n_rows

    if all_vals_list:
        all_final_vals = np.concatenate(all_vals_list)
        all_floor = np.concatenate(all_floor_list)
        all_ceiling = np.concatenate(all_ceiling_list)
        all_is_bad = np.concatenate(all_is_bad_list)
        neighbor_mask = ~all_is_bad
        floor_shortfall = all_floor[neighbor_mask] - all_final_vals[neighbor_mask]
        ceiling_shortfall = all_final_vals[neighbor_mask] - all_ceiling[neighbor_mask]
        combined_shortfall = np.maximum(floor_shortfall, ceiling_shortfall)
        n_neighbor_regressions = int(np.sum(combined_shortfall > volume_epsilon))
        max_neighbor_regression = float(combined_shortfall.max()) if len(combined_shortfall) else 0.0
    else:
        n_neighbor_regressions = 0
        max_neighbor_regression = 0.0

    displacements = np.linalg.norm(
        opt_result.x.reshape(-1, 3) - orig_x0.reshape(-1, 3), axis=1
    )
    info = {
        'n_free_nodes': len(free_nids_list),
        'n_repair_zone_elements': n_constraints,
        'n_touching_solid_elements': len(touching_solid_indices),
        'n_still_out_of_band': n_still_out_of_band,
        'n_neighbor_regressions': n_neighbor_regressions,
        'max_neighbor_regression': max_neighbor_regression,
        'max_displacement_achieved': float(displacements.max()) if len(displacements) else 0.0,
        # NOTE the AND-of-ORs structure (not opt_result.success OR
        # (...)): n_neighbor_regressions == 0 is a MANDATORY gate here,
        # never overridable by opt_result.success alone -- a solver can
        # report success=True (its own soft/approximate constraint
        # tolerance satisfied) while this independent HARD recheck still
        # finds a real neighbor regression (e.g. the softmin-vs-hard-min
        # approximation gap). opt_result.success is only consulted (via
        # the inner OR) as a substitute for n_still_out_of_band == 0,
        # preserving repair_inverted_elements' own documented rationale
        # for why that flag is needed at all (this fork's trust-constr
        # solver reports success=False on essentially every
        # deliberately-early-stopped-but-genuinely-fine run) -- see that
        # function's own 'converged' comment for the full story.
        # Provably identical to the simpler `S or (X==0 and Y==0)` form
        # in every case except this one, which it deliberately closes.
        'converged': (n_neighbor_regressions == 0) and (n_still_out_of_band == 0 or bool(opt_result.success)),
    }
    return updated_nodes_df, opt_result, info
