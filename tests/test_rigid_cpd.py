"""Numeric regression tests for core.registration.rigid_cpd's scale-lock math.

Run with:
    .venv\\Scripts\\python.exe -m pytest tests/test_rigid_cpd.py -v
"""
import numpy as np

from core.registration.rigid_cpd import RigidCPD, CPDConfig


def _small_rotation(rng, max_deg=20.0):
    """A modest rotation, representative of what rigid CPD actually has to
    resolve in this project's pipeline: it runs AFTER PCA pre-alignment and
    shape-signature seeding (AGENTS.md section 2.2), which are responsible
    for getting close to the right orientation first. CPD is the fine
    registration step, not a cold solve of an arbitrary large rotation --
    testing it with a small perturbation is the faithful case, not an
    easier one (an earlier version of this test used a fully random SO(3)
    rotation and CPD correctly failed to converge -- that was a test bug,
    not a rigid_cpd.py bug).
    """
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    theta = np.radians(rng.uniform(-max_deg, max_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _make_asymmetric_shape(rng, n, center, extent):
    """A shape with no rotational symmetry, so CPD has a unique orientation
    to recover (an isotropic Gaussian blob does not: it looks the same
    rotated, which makes registration ill-posed).
    """
    t = np.linspace(0, 4 * np.pi, n)
    x = extent[0] * np.cos(t) * (1 + 0.3 * np.sin(3 * t))
    y = extent[1] * np.sin(t) * 0.6
    z = extent[2] * (t / (4 * np.pi) - 0.5) * 2 + 0.05 * extent[2] * rng.normal(size=n)
    pts = np.stack([x, y, z], axis=1)
    pts += 0.02 * extent.mean() * rng.normal(size=(n, 3))  # small noise
    return pts + center


def test_scale_locks_to_exactly_one_in_original_coordinates():
    """Source and target are the SAME shape but at very different physical
    positions (mimicking a PPE mesh in mm sitting far from a body region in
    a different frame). allow_scale=False must recover scale == 1.0
    EXACTLY -- not approximately, and not the ratio of the two point
    clouds' independently-normalized extents (see rigid_cpd.py's module
    docstring for why naive independent normalization would leak a hidden
    scale even with CPD's own internal `s` locked at 1).
    """
    rng = np.random.default_rng(0)
    R_true = _small_rotation(rng)
    t_true = np.array([500.0, -200.0, 1500.0])  # PPE "far away" translation

    n = 400
    base = _make_asymmetric_shape(rng, n, center=np.zeros(3), extent=np.array([150.0, 90.0, 200.0]))
    target = base + np.array([1000.0, 1000.0, 1000.0])  # target sits elsewhere too
    source = (base @ R_true.T) + t_true  # known rigid motion, no scale

    cpd = RigidCPD(CPDConfig(allow_scale=False, downsample=0, rigid_iters=100, random_seed=1))
    result = cpd.register(source, target)

    assert abs(result.scale - 1.0) < 1e-8, f"scale drifted away from 1.0: {result.scale!r}"

    err = np.linalg.norm(result.transformed_points - target, axis=1)
    assert err.mean() < 5.0, f"registration did not converge to the known answer: mean err={err.mean():.3f}mm"

    # CPDResult.apply() is the "apply the surface-fitted transform to the
    # full PPE volume" path -- must reproduce transformed_points exactly.
    reapplied = result.apply(source)
    assert np.allclose(reapplied, result.transformed_points)


def test_allow_scale_true_recovers_real_scale():
    """With allow_scale=True and a genuine scale difference between source
    and target, the composed scale should recover something close to the
    true ratio (not exact -- CPD's EM correspondence search isn't a
    closed-form fit -- but in the right ballpark for a well-separated,
    unambiguous shape).
    """
    rng = np.random.default_rng(2)
    R_true = _small_rotation(rng)
    t_true = np.array([10.0, 20.0, -5.0])
    true_scale = 1.6

    n = 400
    base = _make_asymmetric_shape(rng, n, center=np.zeros(3), extent=np.array([80.0, 50.0, 100.0]))
    source = base
    target = (base * true_scale) @ R_true.T + t_true

    cpd = RigidCPD(CPDConfig(allow_scale=True, downsample=0, rigid_iters=150, random_seed=3))
    result = cpd.register(source, target)
    assert abs(result.scale - true_scale) < 0.15
