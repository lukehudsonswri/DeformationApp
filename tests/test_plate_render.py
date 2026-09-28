"""Tests for app.plate_render -- reconstructing the rigid PPE's position
at any load-curve time from its seated rest geometry + push direction.
"""
from pathlib import Path

import numpy as np
import pytest

from app.plate_render import load_plate_render, plate_points_at_time


@pytest.fixture
def render_path(tmp_path: Path) -> Path:
    path = tmp_path / "case_plate_render.npz"
    np.savez_compressed(
        path,
        nodes_rest=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        boundary_faces=np.array([[0, 1, 1, 0]], dtype=np.int64),
        push_direction=np.array([1.0, 0.0, 0.0]),
        total_travel_mm=10.0,
        final_time=1.05,
    )
    return path


def test_load_plate_render_reads_all_fields(render_path):
    plate = load_plate_render(render_path)
    assert plate.nodes_rest.shape == (2, 3)
    assert plate.total_travel_mm == pytest.approx(10.0)
    assert plate.final_time == pytest.approx(1.05)


def test_position_at_t0_is_rest_position(render_path):
    plate = load_plate_render(render_path)
    pts = plate_points_at_time(plate, 0.0)
    assert np.allclose(pts, plate.nodes_rest)


def test_position_at_t1_is_full_travel(render_path):
    plate = load_plate_render(render_path)
    pts = plate_points_at_time(plate, 1.0)
    expected = plate.nodes_rest + np.array([10.0, 0.0, 0.0])
    assert np.allclose(pts, expected)


def test_position_is_linear_at_half_travel(render_path):
    plate = load_plate_render(render_path)
    pts = plate_points_at_time(plate, 0.5)
    expected = plate.nodes_rest + np.array([5.0, 0.0, 0.0])
    assert np.allclose(pts, expected)


def test_position_clamps_past_t1(render_path):
    """extend=CONSTANT: overshoot past t=1 (the final solved step can be
    slightly past 1.0, see febio/loadcurve.py) must not overshoot travel.
    """
    plate = load_plate_render(render_path)
    pts_at_1 = plate_points_at_time(plate, 1.0)
    pts_overshoot = plate_points_at_time(plate, 1.05)
    assert np.allclose(pts_at_1, pts_overshoot)


def test_position_clamps_below_t0():
    path_data = dict(
        nodes_rest=np.array([[0.0, 0.0, 0.0]]),
        boundary_faces=np.zeros((0, 4), dtype=np.int64),
        push_direction=np.array([0.0, 1.0, 0.0]),
        total_travel_mm=5.0,
        final_time=1.0,
    )
    from app.plate_render import PlateRenderData

    plate = PlateRenderData(**path_data)
    pts = plate_points_at_time(plate, -0.5)
    assert np.allclose(pts, plate.nodes_rest)
