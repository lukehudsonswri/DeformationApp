"""Tests for postprocess.node_log -- parsing FEBio's plain-text
<Output><logfile><node_data> output (AGENTS.md section 2.5 /
febio/assemble_case.py's Output block).
"""
from pathlib import Path

import numpy as np
import pytest

from postprocess.node_log import last_step, read_node_log

_SAMPLE_LOG = """\
*Step  = 0
*Time  = 0
*Data  = x;y;z;ux;uy;uz
1 0 0 0 0 0 0
2 1 0 0 0 0 0
3 1 1 0 0 0 0
*Step  = 1
*Time  = 0.5
*Data  = x;y;z;ux;uy;uz
1 0 0 0 0 0 0
2 1 0 0 0.1 0 0
3 1 1 0 0.2 0 0
*Step  = 2
*Time  = 1.0
*Data  = x;y;z;ux;uy;uz
1 0 0 0 0 0 0
2 1.2 0 0 0.2 0 0
3 1.4 1 0 0.4 0 0
"""


@pytest.fixture
def sample_log_path(tmp_path: Path) -> Path:
    p = tmp_path / "sample_node_displacement.txt"
    p.write_text(_SAMPLE_LOG)
    return p


def test_reads_all_steps_in_order(sample_log_path: Path):
    steps = read_node_log(sample_log_path)
    assert [s.step for s in steps] == [0, 1, 2]
    assert [s.time for s in steps] == pytest.approx([0.0, 0.5, 1.0])


def test_fields_parsed_from_data_header(sample_log_path: Path):
    steps = read_node_log(sample_log_path)
    assert steps[0].fields == ["x", "y", "z", "ux", "uy", "uz"]


def test_node_ids_and_values_shape(sample_log_path: Path):
    steps = read_node_log(sample_log_path)
    step1 = steps[1]
    assert np.array_equal(step1.node_ids, [1, 2, 3])
    assert step1.values.shape == (3, 6)
    # ux column (index 3 of fields -> column 3 of values) for node 3 is 0.2
    ux_col = step1.fields.index("ux")
    assert step1.values[2, ux_col] == pytest.approx(0.2)


def test_last_step_returns_final_block(sample_log_path: Path):
    final = last_step(sample_log_path)
    assert final.step == 2
    assert final.time == pytest.approx(1.0)
    ux_col = final.fields.index("ux")
    assert final.values[:, ux_col].tolist() == pytest.approx([0.0, 0.2, 0.4])


def test_empty_log_raises(tmp_path: Path):
    p = tmp_path / "empty.txt"
    p.write_text("")
    with pytest.raises(ValueError):
        read_node_log(p)
