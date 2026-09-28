"""Tests for app.gui_case_loader's time interpolation between real solved
steps.
"""
from pathlib import Path

import numpy as np
import pytest

from app.gui_case_loader import GuiCase, displacement_at_time, gather_indices_for_target, load_gui_case


def _make_case() -> GuiCase:
    return GuiCase(
        hbm_model_key="F05_Standing",
        site_name="torso",
        ppe_key="armored_plate",
        node_ids=np.array([1001, 1002], dtype=np.int64),
        times=np.array([0.0, 0.5, 1.0]),
        displacement=np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ]
        ),
        plate_nodes_rest=np.zeros((1, 3)),
        plate_boundary_faces=np.zeros((0, 4), dtype=np.int64),
        push_direction=np.array([1.0, 0.0, 0.0]),
        total_travel_mm=4.35,
        final_time=1.0,
    )


def test_displacement_at_exact_step_matches_that_step():
    case = _make_case()
    result = displacement_at_time(case, 0.5)
    assert np.allclose(result, [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])


def test_displacement_interpolates_between_steps():
    case = _make_case()
    result = displacement_at_time(case, 0.25)
    assert np.allclose(result, [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])


def test_displacement_clamps_below_first_time():
    case = _make_case()
    result = displacement_at_time(case, -1.0)
    assert np.allclose(result, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


def test_displacement_clamps_above_last_time():
    case = _make_case()
    result = displacement_at_time(case, 2.0)
    assert np.allclose(result, [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])


def test_load_gui_case_reads_all_fields(tmp_path: Path):
    path = tmp_path / "case_gui_case.npz"
    np.savez_compressed(
        path,
        hbm_model_key="F05_Standing",
        site_name="torso",
        ppe_key="armored_plate",
        node_ids=np.array([1001, 1002], dtype=np.int64),
        times=np.array([0.0, 1.0]),
        displacement=np.zeros((2, 2, 3)),
        plate_nodes_rest=np.zeros((1, 3)),
        plate_boundary_faces=np.zeros((0, 4), dtype=np.int64),
        push_direction=np.array([1.0, 0.0, 0.0]),
        total_travel_mm=4.35,
        final_time=1.0,
        case_name="F05_Standing_torso_armored_plate_preliminary",
        source_node_log="dummy.txt",
    )
    case = load_gui_case(path)
    assert case.hbm_model_key == "F05_Standing"
    assert case.node_ids.tolist() == [1001, 1002]


def test_gather_indices_reorders_and_subsets_correctly():
    """Target region (e.g. skin-only) is usually a reordered SUBSET of the
    case's own node set (e.g. skin+flesh) -- must gather correctly, not
    just assume alignment (a real bug caught here: broadcasting a
    (47544, 3) displacement array straight onto an (11412, 3) skin mesh).
    """
    case_node_ids = np.array([500, 501, 502, 503], dtype=np.int64)
    target_node_ids = np.array([503, 501])  # subset, reordered
    idx = gather_indices_for_target(case_node_ids, target_node_ids)
    assert idx.tolist() == [3, 1]


def test_gather_indices_missing_id_returns_negative_one():
    case_node_ids = np.array([500, 501], dtype=np.int64)
    target_node_ids = np.array([501, 999])
    idx = gather_indices_for_target(case_node_ids, target_node_ids)
    assert idx.tolist() == [1, -1]
