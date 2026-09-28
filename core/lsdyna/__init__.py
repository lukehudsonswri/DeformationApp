"""Bridge into the client's LS-DYNA parsing engine.

See ``bridge.py`` for the full rationale (fixed-width columns, the Cython
accelerator shim, and why we bypass KeywordProcessor's card consolidation
for element keywords).
"""

from .bridge import (
    GUARDS_SOFTWARE_ROOT,
    ensure_bridged,
    make_keyword_processor,
    resolve_includes,
    iter_wanted_cards,
)

__all__ = [
    "GUARDS_SOFTWARE_ROOT",
    "ensure_bridged",
    "make_keyword_processor",
    "resolve_includes",
    "iter_wanted_cards",
]
