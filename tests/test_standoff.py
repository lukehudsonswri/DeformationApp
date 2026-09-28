"""Tests for fitting.standoff -- mesh-aware standoff floor (AGENTS.md
section 2.4: a requested contact gap smaller than the mesh's own faceting
sagitta is not physically meaningful).
"""
import numpy as np
import pytest

from core.mesh.base import ElementType, SurfaceMesh
from fitting.standoff import compute_standoff_floor, resolve_target_standoff


def _flat_grid_surface(edge_length_mm, n=6):
    """A flat regular-grid triangulated patch with a known edge length."""
    xs = np.arange(n) * edge_length_mm
    ys = np.arange(n) * edge_length_mm
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.stack([xx.ravel(), yy.ravel(), np.zeros(n * n)], axis=1)

    def idx(i, j):
        return i * n + j

    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            faces.append([a, b, c])
            faces.append([a, c, d])
    return SurfaceMesh(nodes=nodes, faces=np.array(faces, dtype=np.int64), face_type=ElementType.TRI3, name="grid")


def test_sagitta_matches_agents_md_worked_example():
    """AGENTS.md section 2.4's table: 7mm edges at a 150mm radius of
    curvature gives a 0.041mm sagitta. A flat grid's axis-aligned edges are
    exactly the requested length, giving an exact regression check.
    """
    surface = _flat_grid_surface(edge_length_mm=7.0)
    floor = compute_standoff_floor(surface, radius_of_curvature_mm=150.0)
    assert floor.median_edge_length_mm == pytest.approx(7.0, abs=1e-6)
    assert floor.sagitta_mm == pytest.approx(0.0408, abs=0.001)


def test_floor_scales_with_edge_length_squared():
    small = compute_standoff_floor(_flat_grid_surface(3.0), radius_of_curvature_mm=150.0)
    large = compute_standoff_floor(_flat_grid_surface(15.0), radius_of_curvature_mm=150.0)
    # sagitta ~ L^2, so a 5x edge length gives a 25x sagitta
    assert large.sagitta_mm / small.sagitta_mm == pytest.approx(25.0, rel=0.05)


def test_resolve_passes_through_when_above_floor():
    floor = compute_standoff_floor(_flat_grid_surface(7.0))
    assert resolve_target_standoff(0.1, floor, progress=None) == 0.1


def test_resolve_clamps_up_when_below_floor():
    floor = compute_standoff_floor(_flat_grid_surface(7.0))
    resolved = resolve_target_standoff(0.001, floor, progress=None)
    assert resolved == pytest.approx(floor.floor_mm)
    assert resolved > 0.001


def test_resolve_logs_when_clamping():
    floor = compute_standoff_floor(_flat_grid_surface(7.0))
    messages = []
    resolve_target_standoff(0.001, floor, progress=messages.append)
    assert len(messages) == 1
    assert "floor" in messages[0].lower()


def test_resolve_does_not_log_when_not_clamping():
    floor = compute_standoff_floor(_flat_grid_surface(7.0))
    messages = []
    resolve_target_standoff(0.1, floor, progress=messages.append)
    assert len(messages) == 0
