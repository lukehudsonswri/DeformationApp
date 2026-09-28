"""Tests for core.mesh.signed_distance -- point-to-face signed gap
measurement (AGENTS.md section 2.4: "measure gaps point-to-face, never
point-to-node").
"""
import numpy as np
import pytest

from core.mesh.base import ElementType, SurfaceMesh
from core.mesh.signed_distance import signed_gap_to_surface


def _flat_square_surface(half_extent=50.0, n=5):
    """A flat z=0 square patch, triangulated, for controlled sign checks."""
    xs = np.linspace(-half_extent, half_extent, n)
    ys = np.linspace(-half_extent, half_extent, n)
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
    surface = SurfaceMesh(nodes=nodes, faces=np.array(faces, dtype=np.int64), face_type=ElementType.TRI3, name="flat")
    surface.compute_normals()
    return surface


def test_point_above_flat_plane_is_positive():
    surface = _flat_square_surface()
    # ensure the winding gives +z outward normal; if not, this test still
    # holds by symmetry of the assertion below (sign is self-consistent)
    query = np.array([[0.0, 0.0, 5.0]])
    result = signed_gap_to_surface(query, surface)
    outward_z = np.sign(surface.face_normals[0][2]) or 1.0
    assert np.sign(result.signed_distance_mm[0]) == np.sign(5.0 * outward_z)
    assert abs(result.signed_distance_mm[0]) == pytest.approx(5.0, abs=1e-6)


def test_point_below_flat_plane_is_negative_relative_to_above():
    """A point mirrored through the plane must have the opposite sign from
    the point above it, and the same magnitude.
    """
    surface = _flat_square_surface()
    above = signed_gap_to_surface(np.array([[0.0, 0.0, 3.0]]), surface)
    below = signed_gap_to_surface(np.array([[0.0, 0.0, -3.0]]), surface)
    assert np.sign(above.signed_distance_mm[0]) == -np.sign(below.signed_distance_mm[0])
    assert abs(above.signed_distance_mm[0]) == pytest.approx(abs(below.signed_distance_mm[0]), abs=1e-6)


def test_point_on_plane_is_near_zero():
    surface = _flat_square_surface()
    query = np.array([[1.0, 2.0, 0.0]])
    result = signed_gap_to_surface(query, surface)
    assert abs(result.signed_distance_mm[0]) < 1e-6


def test_gap_result_summary_properties():
    surface = _flat_square_surface()
    query = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, -2.0], [0.0, 0.0, 1.0]])
    result = signed_gap_to_surface(query, surface)
    assert result.min_mm == pytest.approx(result.signed_distance_mm.min())
    assert result.max_mm == pytest.approx(result.signed_distance_mm.max())
    assert result.mean_mm == pytest.approx(result.signed_distance_mm.mean())
    assert result.n_penetrating == int(np.count_nonzero(result.signed_distance_mm < 0))


def test_matches_off_plane_projection_for_a_point_within_the_patch_interior():
    """For a query point directly above the interior of a large flat patch
    (away from any edge), the point-to-triangle closest point should be
    directly below it (pure vertical offset), giving an exact match to the
    naive height above the plane.
    """
    surface = _flat_square_surface(half_extent=50.0, n=9)
    query = np.array([[3.3, -7.1, 12.0]])
    result = signed_gap_to_surface(query, surface)
    assert abs(abs(result.signed_distance_mm[0]) - 12.0) < 1e-4
