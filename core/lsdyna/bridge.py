"""
Bridge into the client's own LS-DYNA parsing engine
(GUARDS_Software/hbm_guards/api/finite_element/lsdyna).

Why this file exists
---------------------
hbm.dyn's included mesh decks use LS-DYNA's classic fixed-width columns
with NO guaranteed whitespace between adjacent fields once IDs grow past
7 digits (e.g. "300045110218168102181631021817010218170" packs a pid and
four node ids together with no separator). A naive ``line.split()`` parser
silently corrupts these lines. The client's ``KeywordProcessor`` +
``cards_lsdyna`` keyword field-width registry already parses this
correctly, so we import and drive that engine directly rather than
re-deriving column widths ourselves.

The only piece of their engine we can't use as-is is the compiled Cython
accelerator ``hbm_guards.api.finite_element.parse_line_fixed_width`` (a
``.pyx`` with no prebuilt ``.pyd`` in this environment, and no C compiler
available to build one). We register a pure-Python module in
``sys.modules`` under that same import path implementing the identical
documented contract (slice by ``field_length``, convert by
``ParameterType``) so the rest of the client's real engine -- the state
machine, the keyword field-width registry, *INCLUDE resolution -- runs
unmodified.

We also bypass ``KeywordProcessor``'s end-of-keyword consolidation step
for repeating cards (``*ELEMENT_SHELL``, ``*ELEMENT_SOLID``, ...): that
step is designed to hand rows off to the client's SQL import pipeline and
isn't in their own special-cased list for element keywords, so it
collapses repeated cards down to just the last one. Their own code notes
elements are instead handled through a separate "legacy bulk insert path".
We read ``current_card.parameters`` directly after each processed line
instead, which is the same correctly-parsed per-line dict, just without
that consolidation step.

--------------------------------------------------------------------------
Moved from: app/client_parser_bridge.py (2026-09-18), as part of the
            core/ vendoring pass (AGENTS.md section 4.1 layout).
Change from the original: GUARDS_SOFTWARE_ROOT is now overridable via the
            GUARDS_SOFTWARE_ROOT environment variable, instead of being a
            hardcoded absolute path -- this project's own path convention
            (see GUARDS_DATA_ROOT for the HBM/PPE data root) applied
            consistently. Defaults to the same path as before when the
            env var is unset, so existing behaviour is unchanged.
``app/client_parser_bridge.py`` re-exports from here for backward
compatibility during the migration; new code should import from here.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Iterator, Tuple

_DEFAULT_GUARDS_SOFTWARE_ROOT = Path(r"C:\Users\lhudson\Desktop\Projects\GUARDS_Software")


def _guards_software_root() -> Path:
    override = os.environ.get("GUARDS_SOFTWARE_ROOT")
    return Path(override) if override else _DEFAULT_GUARDS_SOFTWARE_ROOT


GUARDS_SOFTWARE_ROOT = _guards_software_root()

_bridged = False


def ensure_bridged() -> None:
    """Make the client's hbm_guards LS-DYNA parsing engine importable."""
    global _bridged
    if _bridged:
        return

    root_str = str(_guards_software_root())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    _install_fixed_width_shim()
    _bridged = True


def _install_fixed_width_shim() -> None:
    """Provide a pure-Python stand-in for the uncompiled Cython module."""
    module_name = "hbm_guards.api.finite_element.parse_line_fixed_width"
    if module_name in sys.modules:
        return

    from hbm_guards.api.finite_element.lsdyna.parameter import ParameterType

    def parse_line_fixed_width(line: str, parameters):
        parsed = {}
        start = 0
        length = len(line)
        for parameter in parameters:
            end = min(start + parameter.field_length, length)
            raw = line[start:end] if start < length else ""
            parsed[parameter.variable] = _convert_field(raw, parameter.type)
            start = end
        return parsed

    def _convert_field(raw: str, param_type: "ParameterType"):
        if param_type == ParameterType.INTEGER:
            value = raw.strip()
            try:
                return int(value) if value else 0
            except ValueError:
                return 0
        if param_type == ParameterType.FLOAT:
            value = raw.strip()
            try:
                return float(value) if value else 0.0
            except ValueError:
                return 0.0
        return raw.rstrip()

    shim = types.ModuleType(module_name)
    shim.parse_line_fixed_width = parse_line_fixed_width
    sys.modules[module_name] = shim


def make_keyword_processor():
    """Return a fresh client ``KeywordProcessor`` instance."""
    ensure_bridged()
    from hbm_guards.api.finite_element.lsdyna.keyword_processor import KeywordProcessor

    return KeywordProcessor()


def resolve_includes(main_dyna_path) -> list:
    """Recursively resolve *INCLUDE paths using the client's own resolver."""
    ensure_bridged()
    from hbm_guards.api.db.utils.dyna_include_path_resolver import (
        DynaIncludePathResolver,
    )

    return DynaIncludePathResolver.resolve_includes(str(main_dyna_path))


def iter_wanted_cards(
    file_path, wanted_prefixes: Tuple[str, ...]
) -> Iterator[Tuple[str, dict]]:
    """
    Stream ``(keyword, parameters)`` for every data card belonging to a
    keyword section whose upper-cased name starts with one of
    ``wanted_prefixes`` (e.g. ``("*NODE", "*ELEMENT_SHELL")``).

    Uses the client's real ``KeywordProcessor`` state machine and
    per-keyword fixed-width card formats for every line (including
    section headers, so internal state stays consistent), but only feeds
    card lines through the parser -- and only yields results -- for
    sections we actually want, skipping the expensive per-line parsing
    entirely for sections (e.g. ``*ELEMENT_SOLID``) we don't need.
    """
    kp = make_keyword_processor()
    current_keyword = None
    section_wanted = False

    with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if not line:
                continue

            if line.startswith("*"):
                current_keyword = line.upper()
                section_wanted = current_keyword.startswith(wanted_prefixes)
                kp.process_next(line)
                continue

            if line.startswith("$"):
                continue

            if not section_wanted:
                continue

            kp.process_next(line)
            card = kp.current_card
            if card is not None:
                yield current_keyword, card.parameters
