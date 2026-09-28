"""Tests for core.registration.point_to_plane_icp.

See that module's docstring for why point-to-plane ICP replaced rigid CPD
as the local seating refinement for the rigid armored plate: both CPD and
plain point-to-point ICP were tested directly against real F05 torso data
and diverged (60-70mm / ~45mm mean distance from a 13mm seed); point-to-plane
converged monotonically to 5.3mm. These tests cover the synthetic recovery
guarantee that real-data regression in ``tests/test_seat_plate.py`` builds
on.
"""
import numpy as np
import pytest

from core.registration.point_to_plane_icp import ICPConfig, point_to_plane_icp


def _asymmetric_points_and_normals(nx=25, ny=25, half_extent=100.0):
    """A deliberately *asymmetric* curved surface (mixed quadratic + cubic
    terms, different coefficients per term) -- unlike a paraboloid of
    revolution, this shape has no rotational symmetry, so a rigid
    registration has one well-defined global optimum, making it a fair test
    of whether ICP can recover an arbitrary known transform exactly.
    """
    xs = np.linspace(-half_extent, half_extent, nx)
    ys = np.linspace(-half_extent, half_extent, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zz = 0.0015 * xx ** 2 + 0.0035 * yy ** 2 + 0.001 * xx * yy + 0.00002 * xx ** 3
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    # analytic gradient of z(x,y) -> outward normal (-dz/dx, -dz/dy, 1), normalized
    dzdx = 0.003 * xx + 0.001 * yy + 0.00006 * xx ** 2
    dzdy = 0.007 * yy + 0.001 * xx
    normals = np.stack([-dzdx.ravel(), -dzdy.ravel(), np.ones(nx * ny)], axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return points, normals


def _paraboloid_points_and_normals(nx=25, ny=25, half_extent=100.0, curvature=0.002):
    """A curved point cloud (not a mesh -- ICP only needs points+normals)
    with analytic outward normals. This shape is a paraboloid of
    revolution -- rotationally symmetric about its own axis -- which is
    deliberately used by ``test_symmetric_surface_leaves_residual_ambiguity``
    to demonstrate the same kind of correspondence ambiguity found with the
    real plate/torso data (see module docstring), on a controlled shape.
    """
    xs = np.linspace(-half_extent, half_extent, nx)
    ys = np.linspace(-half_extent, half_extent, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zz = curvature * (xx ** 2 + yy ** 2)
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    # analytic surface normal of z = c(x^2+y^2): gradient (-2cx, -2cy, 1), normalized
    nx_ = -2 * curvature * points[:, 0]
    ny_ = -2 * curvature * points[:, 1]
    nz_ = np.ones(len(points))
    normals = np.stack([nx_, ny_, nz_], axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return points, normals


def _rotation_from_axis_angle(axis, degrees):
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(degrees)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def test_recovers_small_known_perturbation_on_asymmetric_surface():
    """Perturb an asymmetric curved point cloud by a small known rigid
    transform, then verify ICP (registering the perturbed cloud back onto
    the original) removes almost all of the induced error. Unlike a
    paraboloid of revolution, this shape has a unique registration optimum.
    """
    target_points, target_normals = _asymmetric_points_and_normals()

    R_true = _rotation_from_axis_angle(np.array([0.1, 1.0, 0.2]), degrees=4.0)
    t_true = np.array([3.0, -2.0, 1.5])
    source = target_points @ R_true.T + t_true  # perturbed

    result = point_to_plane_icp(source, target_points, target_normals, ICPConfig(max_iterations=50))

    assert result.final_mean_dist_mm < 0.5
    assert result.converged


def test_symmetric_surface_leaves_residual_ambiguity():
    """A paraboloid-of-revolution patch has a rotational symmetry the
    plate/torso pairing shares in spirit (a locally similar-looking patch
    nearby is nearly as good a fit) -- ICP should still *improve* on the
    perturbation and remain stable (not diverge, unlike CPD/point-to-point
    ICP on the real data), but need not fully recover the exact ground
    truth transform. This documents the same limitation found on real data,
    on a controlled synthetic shape.
    """
    target_points, target_normals = _paraboloid_points_and_normals()

    R_true = _rotation_from_axis_angle(np.array([0.1, 1.0, 0.2]), degrees=4.0)
    t_true = np.array([3.0, -2.0, 1.5])
    source = target_points @ R_true.T + t_true
    starting_dist = np.linalg.norm(source - target_points, axis=1).mean()

    result = point_to_plane_icp(source, target_points, target_normals, ICPConfig(max_iterations=50))

    assert result.final_mean_dist_mm < starting_dist
    assert result.converged


def test_apply_reproduces_transformed_points():
    target_points, target_normals = _paraboloid_points_and_normals()
    R_true = _rotation_from_axis_angle(np.array([0.0, 0.0, 1.0]), degrees=2.0)
    t_true = np.array([1.0, 0.5, -0.5])
    source = target_points @ R_true.T + t_true

    result = point_to_plane_icp(source, target_points, target_normals, ICPConfig(max_iterations=50))
    reapplied = result.apply(source)
    np.testing.assert_allclose(reapplied, result.transformed_points, atol=1e-8)


def test_composed_rotation_is_proper():
    target_points, target_normals = _paraboloid_points_and_normals()
    source = target_points + np.array([5.0, 0.0, 0.0])

    result = point_to_plane_icp(source, target_points, target_normals)
    R = result.rotation
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)


def test_pure_translation_case_converges_to_near_zero():
    """A pure lateral offset (no rotation needed) should converge cleanly --
    a baseline sanity check before trusting the harder mixed-perturbation case.
    """
    target_points, target_normals = _paraboloid_points_and_normals()
    source = target_points + np.array([2.0, 0.0, 0.0])

    result = point_to_plane_icp(source, target_points, target_normals, ICPConfig(max_iterations=50))
    assert result.final_mean_dist_mm < 0.1
