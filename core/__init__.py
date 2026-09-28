"""Vendored / adapted infrastructure this project builds on.

Nothing in ``core/`` is original to DeformationApp. Every module here is
either copied from an external project with a provenance header (see
AGENTS.md "Vendoring policy"), or a thin adapter around the client's own
LS-DYNA parsing engine (``core/lsdyna``).

Subpackages:
    lsdyna       -- bridge into the client's KeywordProcessor (GUARDS_Software)
    mesh         -- multi-format mesh I/O (.k / .inp / .feb / .stl / .obj),
                    vendored from fe_personalization/mesh_io
    registration -- Kabsch / PCA pre-align / rigid CPD, vendored (and, for
                    CPD, rewritten CPU-only) from fe_personalization/registration
    repair       -- mesh quality + negative-Jacobian / shell-distortion
                    repair, vendored from ForSarah/Sparse
"""
