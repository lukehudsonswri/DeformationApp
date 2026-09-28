"""Regression tests for config.hbm_normalize / config._hbm_raw_source --
building a clean HBM/<model_name>/ folder from the client's messy original
source export.

Runs against the real (large, gitignored) source tree in the sibling
DeformationApp working copy. Skipped, not failed, when unavailable.

Does NOT re-run the (~2 minute) full F05_Standing build on every test
session -- that's already been done once for real (see
tests/test_hbm_models.py's integration check, which reads the result).
This file instead builds a small synthetic raw-source tree to test the
normalization LOGIC quickly, plus one narrow real-data check for the
*NODE 6-field requirement that was only discovered by hitting it for real.
"""
from pathlib import Path

import pytest

from config._hbm_raw_source import RawSourceModel
from config.hbm_normalize import build_hbm_model_folder

_REAL_SOURCE_ROOT = Path(r"C:\Users\lhudson\Desktop\Projects\GUARDS\DeformationApp\HBM")


def _write_source_deck(path: Path, lines):
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def synthetic_source_model(tmp_path):
    """A tiny hand-built raw source deck pair, standing in for the messy
    original export -- exercises the SAME extraction logic
    (build_hbm_model_folder) without needing the real multi-hundred-MB
    files for a fast, deterministic unit test.
    """
    nodes_deck = tmp_path / "src_nodes.k"
    elements_deck = tmp_path / "src_elements.k"
    _write_source_deck(
        nodes_deck,
        [
            "*KEYWORD",
            "*NODE",
            "       1     1.00000     2.00000     3.00000       0       0",
            "       2     4.00000     5.00000     6.00000       0       0",
            "*END",
        ],
    )
    _write_source_deck(
        elements_deck,
        [
            "*KEYWORD",
            "*ELEMENT_SOLID",
            "       1 2000500       1       1       1       1       1       1       1       1",
            "*ELEMENT_SHELL",
            "       2 2000501       1       2       1       2",
            "*END",
        ],
    )
    return RawSourceModel(
        key="TestModel",
        family="TEST",
        posture="Seated",
        source_dir=tmp_path,
        nodes_decks=(nodes_deck,),
        elements_decks=(elements_deck,),
        properties_deck=None,
    )


def test_build_writes_clean_nodes_and_elements_files(synthetic_source_model, tmp_path):
    out_root = tmp_path / "HBM_out"
    result = build_hbm_model_folder(synthetic_source_model, out_root)

    assert result.n_nodes == 2
    assert result.n_solid_elements == 1
    assert result.n_shell_elements == 1
    assert result.nodes_path == out_root / "TestModel" / "Nodes.k"
    assert result.elements_path == out_root / "TestModel" / "Elements.k"
    assert result.nodes_path.is_file()
    assert result.elements_path.is_file()


def test_written_nodes_file_has_six_fields_per_line(synthetic_source_model, tmp_path):
    """The client's real *NODE card format requires exactly 6 comma fields
    (nid,x,y,z,tc,rc) -- its delimited parser has no tolerance for missing
    trailing fields. This was discovered empirically while building
    F05_Standing (an earlier 4-field writer raised IndexError on re-parse)
    -- pin it here so it can't silently regress.
    """
    out_root = tmp_path / "HBM_out"
    result = build_hbm_model_folder(synthetic_source_model, out_root)

    lines = [
        line for line in result.nodes_path.read_text().splitlines() if line and not line.startswith("*")
    ]
    assert len(lines) == 2
    for line in lines:
        assert len(line.split(",")) == 6


def test_written_elements_have_pid_but_no_materials(synthetic_source_model, tmp_path):
    """Per explicit instruction: elements/nodes only, no material models.
    Confirm the pid (topological tag, not a material) IS kept, and that
    nothing resembling a *MAT_/*SECTION_/*PART card appears anywhere.
    """
    out_root = tmp_path / "HBM_out"
    result = build_hbm_model_folder(synthetic_source_model, out_root)
    text = result.elements_path.read_text()

    assert "2000500" in text  # the solid's pid is kept
    assert "2000501" in text  # the shell's pid is kept
    for forbidden in ("*MAT_", "*SECTION_", "*PART", "*HOURGLASS"):
        assert forbidden not in text


def test_solid_and_shell_written_to_separate_keyword_blocks(synthetic_source_model, tmp_path):
    out_root = tmp_path / "HBM_out"
    result = build_hbm_model_folder(synthetic_source_model, out_root)
    text = result.elements_path.read_text()

    # Exactly one *ELEMENT_SOLID and one *ELEMENT_SHELL keyword line -- not
    # one repeated per element (an earlier draft did this; wasteful/wrong
    # shape for a normal LS-DYNA deck).
    assert text.count("*ELEMENT_SOLID") == 1
    assert text.count("*ELEMENT_SHELL") == 1


@pytest.mark.skipif(not _REAL_SOURCE_ROOT.is_dir(), reason="HBM/ raw source data not present on this machine")
def test_build_model_by_key_rejects_unknown_key(tmp_path):
    from config.hbm_normalize import build_model_by_key

    with pytest.raises(KeyError):
        build_model_by_key("NotARealModel", _REAL_SOURCE_ROOT, tmp_path)


# ---------------------------------------------------------------------------
# The generic, no-naming-convention-required folder scanner
# (build_hbm_model_from_folder) -- what actually answers "if I drop in a new
# model folder with arbitrary file naming, will it get extracted correctly".
# config._hbm_raw_source's hardcoded patterns do NOT answer that question --
# they only match the two specific layouts already in the client's existing
# HBM/ export.
# ---------------------------------------------------------------------------


def test_generic_scan_finds_every_k_and_dyn_file_regardless_of_naming(tmp_path):
    """No naming convention required: files named nothing like
    "Nodes"/"Elements" at all are still fully scanned."""
    from config.hbm_normalize import build_hbm_model_from_folder

    src = tmp_path / "src"
    src.mkdir()
    (src / "whatever_export_1.k").write_text("*KEYWORD\n*NODE\n1,1,2,3,0,0\n2,4,5,6,0,0\n*END\n")
    (src / "some_other_file.dyn").write_text(
        "*KEYWORD\n*ELEMENT_SOLID\n1,500,1,1,2,2,1,1,2,2\n*END\n"
    )
    (src / "irrelevant_boundary_conditions.dyn").write_text("*KEYWORD\n*INCLUDE\nwhatever.k\n*END\n")

    result = build_hbm_model_from_folder("AnyNameModel", src, tmp_path / "out", progress=None)
    assert result.n_nodes == 2
    assert result.n_solid_elements == 1


def test_generic_scan_raises_on_conflicting_node_coordinates(tmp_path):
    """The same node id appearing in two files with DIFFERENT coordinates
    is ambiguous source data -- must fail loudly, never silently guess.
    """
    from config.hbm_normalize import build_hbm_model_from_folder

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.k").write_text("*KEYWORD\n*NODE\n1,0.0,0.0,0.0,0,0\n*END\n")
    (src / "b.k").write_text("*KEYWORD\n*NODE\n1,99.0,99.0,99.0,0,0\n*END\n")

    with pytest.raises(ValueError, match="DIFFERENT coordinates"):
        build_hbm_model_from_folder("ConflictTest", src, tmp_path / "out", progress=None)


def test_generic_scan_dedupes_identical_node_duplicates(tmp_path):
    """The same node id in two files with the SAME coordinates (e.g. a
    shared boundary node between two decks) is expected and must be
    silently deduplicated, not an error.
    """
    from config.hbm_normalize import build_hbm_model_from_folder

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.k").write_text("*KEYWORD\n*NODE\n1,5.0,5.0,5.0,0,0\n2,6.0,6.0,6.0,0,0\n*END\n")
    (src / "b.k").write_text("*KEYWORD\n*NODE\n1,5.0,5.0,5.0,0,0\n*END\n")

    result = build_hbm_model_from_folder("DedupTest", src, tmp_path / "out", progress=None)
    assert result.n_nodes == 2


def test_generic_scan_raises_on_duplicate_element_id(tmp_path):
    """Unlike nodes, a duplicate element id is NEVER expected/benign --
    always ambiguous source data, must raise.
    """
    from config.hbm_normalize import build_hbm_model_from_folder

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.k").write_text("*KEYWORD\n*ELEMENT_SOLID\n1,100,1,1,1,1,1,1,1,1\n*END\n")
    (src / "b.k").write_text("*KEYWORD\n*ELEMENT_SOLID\n1,200,1,1,1,1,1,1,1,1\n*END\n")

    with pytest.raises(ValueError, match="appears in more than one"):
        build_hbm_model_from_folder("ElemConflictTest", src, tmp_path / "out", progress=None)


def test_generic_scan_raises_on_folder_with_no_deck_files(tmp_path):
    from config.hbm_normalize import build_hbm_model_from_folder

    src = tmp_path / "empty_src"
    src.mkdir()
    (src / "readme.txt").write_text("not a deck file")

    with pytest.raises(FileNotFoundError):
        build_hbm_model_from_folder("EmptyTest", src, tmp_path / "out", progress=None)


def test_generic_scan_parses_family_posture_from_key(tmp_path):
    from config.hbm_normalize import build_hbm_model_from_folder

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.k").write_text("*KEYWORD\n*NODE\n1,0,0,0,0,0\n*END\n")

    result = build_hbm_model_from_folder("M50_Standing", src, tmp_path / "out", progress=None)
    assert result.family == "M50"
    assert result.posture == "Standing"


@pytest.mark.skipif(
    not (_REAL_SOURCE_ROOT / "M50_Standing_Skin").is_dir(),
    reason="M50_Standing_Skin raw source data not present on this machine",
)
def test_generic_scan_matches_known_counts_on_real_messy_folder(tmp_path):
    """Run the generic scanner directly against the REAL, messy
    M50_Standing_Skin source folder -- as a stand-in for "the user drops in
    a folder with all the .k files and cards for a model" -- and confirm it
    reproduces the exact counts independently verified earlier (2026-09-18):
    zero Thorax_Flesh (pid 2000500) solids (confirmed absent from this
    model), and exactly 11,158 Thorax_Skin (pid 2000501) shells (same
    shared topology as F05_Standing). This is the real proof the generic,
    no-naming-convention path works -- not just the synthetic unit tests
    above.
    """
    from config.hbm_normalize import build_hbm_model_from_folder

    source_dir = _REAL_SOURCE_ROOT / "M50_Standing_Skin"
    result = build_hbm_model_from_folder("M50_Standing_GenericScanTest", source_dir, tmp_path / "out", progress=None)

    n_2000500 = 0
    n_2000501 = 0
    section = None
    with open(result.elements_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("*"):
                section = line
                continue
            pid = int(line.split(",")[1])
            if section == "*ELEMENT_SOLID" and pid == 2000500:
                n_2000500 += 1
            elif section == "*ELEMENT_SHELL" and pid == 2000501:
                n_2000501 += 1

    assert n_2000500 == 0
    assert n_2000501 == 11158

