"""Body-site resolution: which part ids are "flesh" (solid) and "skin"
(shell) for a named anatomical site, in a given HBM model.

**Torso only, per explicit instruction** — head is out of scope (no head
PPE exists in this project). The multi-site auto-detection design in
AGENTS.md section 1.3 (shape-signature shortlist + rigid-CPD fit-score
decision) is deferred, not built, until a second site is actually needed;
building it now against a single site would be speculative complexity with
nothing to validate it against. Site selection today is simply always
``"torso"``.

Why this is a hardcoded pid table, not a `*PART`-title lookup
----------------------------------------------------------------------------
The original plan was to match `*PART` title cards (e.g. ``Thorax_Flesh``)
in the source deck. That doesn't work against this project's own data
anymore: ``config/hbm_normalize.py``'s normalized ``HBM/<model>/Elements.k``
files deliberately strip every `*PART` title card -- geometry only, no
materials, no part names, per instruction. So there are no titles left to
match against at the point site resolution actually runs (only at
`config/hbm_normalize.py`'s one-time extraction step, which doesn't need
site information -- it's told which whole MODEL to build, not which site
within it).

Confirmed against the original (pre-normalization) ``I-PREDICT_v1.0_Main.dyn``
`*PART` cards, and against direct element pid extraction in the actual
built ``F05_Standing`` model (34,980 Thorax_Flesh hex8 solids, 11,158
Thorax_Skin quad4 shells -- see ``tests/test_hbm_models.py``):

    Thorax_Flesh (solid) = pid 2000500
    Thorax_Skin  (shell) = pid 2000501

These pids are a property of the I-PREDICT model FAMILY (confirmed
identical across F05/F50/M50/M95 -- same topology, different anthropometry),
not something re-derived per model instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from config.hbm_models import HbmModel

# (solid/flesh pids, shell/skin pids) per site, for the I-PREDICT model
# family. Add a row here (and a corresponding regression test) before
# using a new site name -- resolve_site() deliberately raises rather than
# guessing for anything not listed.
_SITE_PIDS = {
    "torso": ((2000500,), (2000501,)),
}


@dataclass(frozen=True)
class SiteResolution:
    """Which part ids make up a site's flesh (solid) and skin (shell)."""

    site: str
    model_key: str
    solid_pids: Tuple[int, ...]
    shell_pids: Tuple[int, ...]


def resolve_site(model: HbmModel, site: str) -> SiteResolution:
    """Return the solid ("flesh") and shell ("skin") part ids for ``site``
    in ``model``.

    Args:
        model: an ``HbmModel`` from ``config.hbm_models.discover_hbm_models``
            (unused today beyond carrying the model key through into the
            result -- pids are a model-FAMILY constant, not model-specific
            -- but kept as a parameter so a future model family with
            different pids has an obvious place to branch on
            ``model.family`` without changing every call site).
        site: site name, e.g. ``"torso"``.

    Returns:
        SiteResolution with the resolved pids.

    Raises:
        KeyError: if ``site`` isn't in the known-sites table. Deliberately
            does not fall back to guessing -- an unresolvable site must
            fail loudly, not silently produce an empty/wrong part set.
    """
    if site not in _SITE_PIDS:
        raise KeyError(f"Unknown site '{site}' -- known sites: {sorted(_SITE_PIDS)}")

    solid_pids, shell_pids = _SITE_PIDS[site]
    return SiteResolution(
        site=site,
        model_key=model.key,
        solid_pids=solid_pids,
        shell_pids=shell_pids,
    )


def known_sites() -> Tuple[str, ...]:
    """Every site this project currently knows how to resolve."""
    return tuple(_SITE_PIDS.keys())
