"""Smoke test for the vendored core.repair (Sparse mesh-repair engine)
against real production HBM data -- one of this project's regression
fixtures (AGENTS.md section 6 / VERIFICATION.md section 7).

Run with:
    .venv\\Scripts\\python.exe -m pytest tests/test_core_repair_smoke.py -v
"""
from pathlib import Path

import pandas as pd
import pytest

from core.repair import (
    read_nodes_from_kfile,
    read_all_elements_from_kfile,
    compute_shell_jacobian,
    classify_shell_badness,
    compute_element_distortion,
    compute_element_volumes,
    classify_solid_badness,
    build_solid_node_adjacency,
    find_touching_solid_elements,
)

# This is real, large (untracked/gitignored) case data that lives in the
# original DeformationApp working tree, not this worktree's own cases/
# (git worktrees only carry tracked files -- see AGENTS.md section 4.3).
# Skip rather than fail if it isn't present on this machine.
_F05_COMBINED_K = Path(
    r"C:\Users\lhudson\Desktop\Projects\GUARDS\DeformationApp"
    r"\cases\F05_Torso_ArmoredPlate\torso_F05_standing_full.k"
)

pytestmark = pytest.mark.skipif(
    not _F05_COMBINED_K.exists(), reason="F05 case data not present on this machine"
)


def test_solid_and_shell_quality_on_real_f05_case():
    nodes_df = read_nodes_from_kfile(str(_F05_COMBINED_K))
    all_elems = read_all_elements_from_kfile(str(_F05_COMBINED_K))

    solids = all_elems[all_elems["ETYPE"] == "SOLID"].copy()
    shells = all_elems[all_elems["ETYPE"] == "SHELL"].copy()

    # Known topology (see config/sites.py): 11,412 skin nodes /
    # 11,158 skin shells shared across the whole I-PREDICT v1.0 family.
    assert len(shells) == 11158

    vol_df = compute_element_volumes(solids, nodes_df)
    n_inverted = int((vol_df["volume"] < 0).sum())
    assert n_inverted == 0, f"{n_inverted} inverted solid elements in a supposedly clean source mesh"

    dist_df = compute_element_distortion(solids, nodes_df)
    n_tangled = int(dist_df["is_distorted"].sum())
    assert n_tangled == 0, f"{n_tangled} locally-tangled solid elements"

    bad_solid = classify_solid_badness(solids, nodes_df)
    assert int((bad_solid["bad_tier"] > 0).sum()) == 0

    jac_df = compute_shell_jacobian(shells, nodes_df)
    assert jac_df["jacobian_ratio"].min() > 0, "a shell element has non-positive jacobian_ratio"

    # Baseline finding (2026-09-18): 38 of 11,158 shell elements already sit
    # outside a [0.4, 0.95] jacobian_ratio band in the SOURCE mesh, before
    # any decimation. Not a failure -- just the known baseline this
    # regression test pins, so Goal 3 remeshing work can tell whether it
    # made shell quality better or worse.
    bad_shell = classify_shell_badness(shells, nodes_df, band_floor=0.4, band_ceiling=0.95)
    assert int(bad_shell["is_bad"].sum()) == 38


def test_skin_flesh_connectivity_on_real_f05_case():
    """Every skin (shell) node must also belong to at least one flesh
    (solid) element -- the "maintain connectivity" requirement in AGENTS.md
    Goal 3. This is the actual mechanism Sparse's repair functions rely on
    to protect flesh elements when a touching skin shell is repaired.
    """
    nodes_df = read_nodes_from_kfile(str(_F05_COMBINED_K))
    all_elems = read_all_elements_from_kfile(str(_F05_COMBINED_K))
    solids = all_elems[all_elems["ETYPE"] == "SOLID"].copy()
    shells = all_elems[all_elems["ETYPE"] == "SHELL"].copy()

    _, node_to_solid = build_solid_node_adjacency(solids)
    skin_node_ids = pd.unique(shells[["NID1", "NID2", "NID3", "NID4"]].values.ravel())
    skin_node_ids = skin_node_ids[skin_node_ids != 0]

    orphaned = [nid for nid in skin_node_ids if not node_to_solid.get(nid)]
    assert not orphaned, f"{len(orphaned)} skin nodes have no touching flesh solid"

    touching = find_touching_solid_elements(skin_node_ids.tolist(), node_to_solid)
    assert len(touching) > 0
