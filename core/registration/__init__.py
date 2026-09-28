"""Rigid registration primitives: Kabsch, PCA pre-alignment, and rigid CPD.

Vendored (and, for CPD, rewritten CPU-only) from
``fe_personalization/registration`` -- see each module's own provenance
header for exactly what changed and why.

Typical PPE seating sequence (AGENTS.md section 2.2):
    1. ``pca_prealign.pca_affine_align`` / ``should_apply_prealign`` /
       ``apply_pca_alignment`` -- coarse orientation fix.
    2. ``rigid_cpd.RigidCPD`` -- fine registration (CPD needs (1) to have
       already gotten close; see rigid_cpd.py's module docstring and
       ``tests/_check_rigid_cpd.py`` for why).
    3. ``rigid.rigid_align`` / ``rigid.kabsch_rotation`` -- lightweight
       closed-form alternative when point correspondence is already known
       (no CPD needed), or as a fallback.
"""

from .rigid import kabsch_rotation, rigid_align
from .pca_prealign import (
    PCAPreAlignDiagnostics,
    pca_affine_align,
    should_apply_prealign,
    apply_pca_alignment,
)
from .rigid_cpd import RigidCPD, CPDConfig, CPDResult

__all__ = [
    "kabsch_rotation",
    "rigid_align",
    "PCAPreAlignDiagnostics",
    "pca_affine_align",
    "should_apply_prealign",
    "apply_pca_alignment",
    "RigidCPD",
    "CPDConfig",
    "CPDResult",
]
