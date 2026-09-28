"""Tests for postprocess.gui_case -- assembling and discovering the single
self-contained GUI-case artifact main_2.py produces for app/viewer_app.py.
"""
from pathlib import Path

import numpy as np
import pytest

from postprocess.gui_case import build_gui_case, discover_gui_cases

_SAMPLE_LOG = """\
*Step  = 0
*Time  = 0
*Data  = x;y;z;ux;uy;uz
1 0 0 0 0 0 0
2 1 0 0 0 0 0
3 1 1 0 0 0 0
*Step  = 1
*Time  = 1.0342
*Data  = x;y;z;ux;uy;uz
1 0 0 0 0.01 0.02 0.03
2 1 0 0 0.04 0.05 0.06
3 1 1 0 0.07 0.08 0.09
"""


@pytest.fixture
def case_dir(tmp_path: Path) -> Path:
    case_name = "F05_Standing_torso_armored_plate_preliminary"

    (tmp_path / f"{case_name}_node_displacement.txt").write_text(_SAMPLE_LOG)

    np.savez_compressed(
        tmp_path / f"{case_name}_node_map.npz",
        torso_node_ids=np.array([1001, 1002, 1003], dtype=np.int64),
        n_torso=3,
        hbm_model_key="F05_Standing",
        site_name="torso",
        ppe_key="armored_plate",
    )

    np.savez_compressed(
        tmp_path / f"{case_name}_plate_render.npz",
        nodes_rest=np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [11.0, 1.0, 0.0], [10.0, 1.0, 0.0]]),
        boundary_faces=np.array([[0, 1, 2, 3]], dtype=np.int64),
        push_direction=np.array([1.0, 0.0, 0.0]),
        total_travel_mm=4.35,
        final_time=1.0342,
        hbm_model_key="F05_Standing",
        site_name="torso",
        ppe_key="armored_plate",
    )
    return tmp_path


def test_build_gui_case_writes_expected_file(case_dir: Path):
    out_path = build_gui_case("F05_Standing_torso_armored_plate_preliminary", case_dir)
    assert out_path.is_file()
    assert out_path.name == "F05_Standing_torso_armored_plate_preliminary_gui_case.npz"


def test_gui_case_contains_all_solved_steps_and_plate_geometry(case_dir: Path):
    out_path = build_gui_case("F05_Standing_torso_armored_plate_preliminary", case_dir)
    with np.load(out_path) as data:
        assert data["times"].tolist() == pytest.approx([0.0, 1.0342])
        assert data["displacement"].shape == (2, 3, 3)
        assert np.array_equal(data["node_ids"], [1001, 1002, 1003])
        assert data["plate_nodes_rest"].shape == (4, 3)
        assert data["plate_boundary_faces"].shape == (1, 4)
        assert data["push_direction"].tolist() == pytest.approx([1.0, 0.0, 0.0])
        assert float(data["total_travel_mm"]) == pytest.approx(4.35)
        assert float(data["final_time"]) == pytest.approx(1.0342)
        assert str(data["hbm_model_key"]) == "F05_Standing"
        assert str(data["site_name"]) == "torso"
        assert str(data["ppe_key"]) == "armored_plate"


def test_build_gui_case_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_gui_case("does_not_exist", tmp_path)


def test_discover_gui_cases_finds_built_case(case_dir: Path):
    build_gui_case("F05_Standing_torso_armored_plate_preliminary", case_dir)
    found = discover_gui_cases(case_dir)
    assert len(found) == 1
    assert found[0].hbm_model_key == "F05_Standing"
    assert found[0].site_name == "torso"
    assert found[0].ppe_key == "armored_plate"


def test_discover_gui_cases_empty_dir_returns_empty_list(tmp_path: Path):
    assert discover_gui_cases(tmp_path) == []


def test_discover_gui_cases_nonexistent_dir_returns_empty_list(tmp_path: Path):
    assert discover_gui_cases(tmp_path / "does_not_exist") == []
