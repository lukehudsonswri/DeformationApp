"""Tests for postprocess.project_displacement -- mapping a solved case's
local FEBio node numbering back onto original HBM node ids, across every
solved step (for the GUI's animated slider).
"""
from pathlib import Path

import numpy as np
import pytest

from postprocess.project_displacement import project_case_all_steps

_SAMPLE_LOG = """\
*Step  = 0
*Time  = 0
*Data  = x;y;z;ux;uy;uz
1 0 0 0 0 0 0
2 1 0 0 0 0 0
3 1 1 0 0 0 0
4 5 5 5 0 0 0
*Step  = 1
*Time  = 1.0342
*Data  = x;y;z;ux;uy;uz
1 0 0 0 0.01 0.02 0.03
2 1 0 0 0.04 0.05 0.06
3 1 1 0 0.07 0.08 0.09
4 5.5 5.1 5.2 0.5 0.1 0.2
"""
# node 4 is a plate node (n_torso=3), not part of the HBM body -- its
# large displacement must NOT leak into the projected torso-only field.


@pytest.fixture
def case_files(tmp_path: Path):
    log_path = tmp_path / "case_node_displacement.txt"
    log_path.write_text(_SAMPLE_LOG)

    map_path = tmp_path / "case_node_map.npz"
    np.savez_compressed(
        map_path,
        torso_node_ids=np.array([1001, 1002, 1003], dtype=np.int64),  # original HBM ids
        n_torso=3,
        hbm_model_key="F05_Standing",
        site_name="torso",
        ppe_key="armored_plate",
    )
    return log_path, map_path


def test_project_all_steps_returns_every_step_in_order(case_files):
    log_path, map_path = case_files
    result = project_case_all_steps(log_path, map_path)

    assert np.array_equal(result.node_ids, [1001, 1002, 1003])
    assert result.times.tolist() == pytest.approx([0.0, 1.0342])
    assert result.displacement.shape == (2, 3, 3)
    # step 0: all zero displacement
    assert np.allclose(result.displacement[0], 0.0)
    # step 1, local node 2 (original id 1002) -> ux,uy,uz = 0.04, 0.05, 0.06
    idx = list(result.node_ids).index(1002)
    assert result.displacement[1, idx].tolist() == pytest.approx([0.04, 0.05, 0.06])


def test_project_all_steps_excludes_plate_nodes(case_files):
    log_path, map_path = case_files
    result = project_case_all_steps(log_path, map_path)
    # node 4 (plate, displacement 0.5/0.1/0.2) must be excluded entirely
    assert np.all(np.abs(result.displacement) < 0.5)


def test_missing_uxyz_columns_raises(tmp_path: Path):
    log_path = tmp_path / "bad_log.txt"
    log_path.write_text("*Step  = 0\n*Time  = 0\n*Data  = x;y;z\n1 0 0 0\n")
    map_path = tmp_path / "bad_map.npz"
    np.savez_compressed(map_path, torso_node_ids=np.array([1], dtype=np.int64), n_torso=1)
    with pytest.raises(ValueError):
        project_case_all_steps(log_path, map_path)


def test_non_contiguous_torso_ids_raises(tmp_path: Path):
    """If the log's torso-range node ids skip a number, our 1..n_torso
    ordering assumption (see assemble_case.py) is violated -- must fail
    loudly rather than silently mis-map displacement to the wrong node.
    """
    log_path = tmp_path / "gap_log.txt"
    log_path.write_text(
        "*Step  = 0\n*Time  = 1.0\n*Data  = x;y;z;ux;uy;uz\n"
        "1 0 0 0 0 0 0\n"
        "3 1 1 0 0 0 0\n"  # missing id 2
    )
    map_path = tmp_path / "gap_map.npz"
    np.savez_compressed(map_path, torso_node_ids=np.array([1001, 1002, 1003], dtype=np.int64), n_torso=3)
    with pytest.raises(ValueError):
        project_case_all_steps(log_path, map_path)
