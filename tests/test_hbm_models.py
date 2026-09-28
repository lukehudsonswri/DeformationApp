"""Regression tests for config.hbm_models -- the simple runtime registry
that scans clean ``HBM/<model_name>/{Nodes,Elements}.k`` folders.

This is deliberately NOT a test of the messy raw source data (that's
tests/test_hbm_normalize.py, which covers config._hbm_raw_source and
config.hbm_normalize). This module has zero knowledge of the raw source
layout -- it only reads whatever clean model folders already exist under
HBM/, which is exactly what should be tested here: does discovery work
correctly against the CLEAN layout, independent of how that layout was
produced.
"""
from pathlib import Path

import pytest

from config.hbm_models import discover_hbm_models


@pytest.fixture
def clean_hbm_root(tmp_path):
    """A synthetic HBM/ with a couple of clean model folders, so this test
    doesn't depend on any specific model having been built yet (per
    instruction, only F05_Standing is built today, and that may change).
    """
    root = tmp_path / "HBM"
    for name, nodes, elements in [
        ("F05_Standing", "*KEYWORD\n*NODE\n1,0,0,0,0,0\n*END\n", "*KEYWORD\n*ELEMENT_SOLID\n1,2000500,1,1,1,1,1,1,1,1\n*END\n"),
        ("M50_Seated", "*KEYWORD\n*NODE\n1,0,0,0,0,0\n*END\n", "*KEYWORD\n*ELEMENT_SHELL\n1,2000501,1,1,1,1\n*END\n"),
        ("unrelated_notes", None, None),  # folder with no Nodes.k/Elements.k -- must be skipped
    ]:
        d = root / name
        d.mkdir(parents=True)
        if nodes is not None:
            (d / "Nodes.k").write_text(nodes)
        if elements is not None:
            (d / "Elements.k").write_text(elements)
    # A stray file directly under HBM/ (not a directory) must not crash discovery.
    (root / "README.txt").write_text("not a model")
    return root


def test_discovers_only_folders_with_both_files(clean_hbm_root):
    models = discover_hbm_models(clean_hbm_root)
    assert set(models.keys()) == {"F05_Standing", "M50_Seated"}


def test_family_and_posture_parsed_from_folder_name(clean_hbm_root):
    models = discover_hbm_models(clean_hbm_root)
    f05 = models["F05_Standing"]
    assert f05.family == "F05"
    assert f05.posture == "Standing"

    m50 = models["M50_Seated"]
    assert m50.family == "M50"
    assert m50.posture == "Seated"


def test_paths_point_at_the_actual_files(clean_hbm_root):
    models = discover_hbm_models(clean_hbm_root)
    f05 = models["F05_Standing"]
    assert f05.nodes_path == clean_hbm_root / "F05_Standing" / "Nodes.k"
    assert f05.elements_path == clean_hbm_root / "F05_Standing" / "Elements.k"
    assert f05.nodes_path.is_file()
    assert f05.elements_path.is_file()


def test_missing_hbm_root_returns_empty_not_error(tmp_path):
    assert discover_hbm_models(tmp_path / "does_not_exist") == {}


def test_unrecognized_folder_name_still_discovered_with_none_family_posture(tmp_path):
    """A model folder whose name doesn't match any known family/posture
    token is still a usable model (per instruction: any HBM model dropped
    in should work) -- family/posture are best-effort display metadata,
    not a requirement for the folder to be recognized.
    """
    root = tmp_path / "HBM"
    d = root / "SomeNewModel"
    d.mkdir(parents=True)
    (d / "Nodes.k").write_text("*KEYWORD\n*NODE\n*END\n")
    (d / "Elements.k").write_text("*KEYWORD\n*END\n")
    models = discover_hbm_models(root)
    assert "SomeNewModel" in models
    assert models["SomeNewModel"].family is None
    assert models["SomeNewModel"].posture is None


# ---------------------------------------------------------------------------
# Integration check against the real, built F05_Standing model, if present.
# Skipped (not failed) if it hasn't been built on this machine yet -- per
# instruction, models are built on demand, not pre-populated.
# ---------------------------------------------------------------------------
_REAL_HBM_ROOT = Path(r"C:\Users\lhudson\Desktop\Projects\GUARDS\DeformationApp-agent\HBM")


@pytest.mark.skipif(
    not (_REAL_HBM_ROOT / "F05_Standing" / "Nodes.k").is_file(),
    reason="F05_Standing has not been built yet (config.hbm_normalize.build_model_by_key)",
)
def test_real_f05_standing_has_expected_element_counts():
    """F05_Standing was verified (2026-09-18) against the client's own
    authoritative parser to have exactly 34,980 Thorax_Flesh (pid 2000500)
    solid elements and 11,158 Thorax_Skin (pid 2000501) shell elements,
    matching independently-known counts from this project's case data.
    This test re-checks that the ACTUALLY BUILT files on disk still match,
    using only a plain split(",") (the format this project's own writer
    produces, per config/hbm_normalize.py) rather than the client parser,
    since that's what real consumers of this file should be able to rely on.
    """
    models = discover_hbm_models(_REAL_HBM_ROOT)
    model = models["F05_Standing"]

    n_2000500 = 0
    n_2000501 = 0
    section = None
    with open(model.elements_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("*"):
                section = line
                continue
            parts = line.split(",")
            pid = int(parts[1])
            if section == "*ELEMENT_SOLID" and pid == 2000500:
                n_2000500 += 1
            elif section == "*ELEMENT_SHELL" and pid == 2000501:
                n_2000501 += 1

    assert n_2000500 == 34980
    assert n_2000501 == 11158
