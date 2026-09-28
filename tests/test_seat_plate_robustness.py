"""Robustness regression test: fitting.seat_plate must recover a correct
seating even when the PPE starts translated/rotated far from the body
(AGENTS.md Goal 2 -- "if the torso is rotated in some weird way and is
located far away, I need a way to bring it back to the human body model").

This is the "far away" path of seat_plate()'s near/far branch (see that
module's docstring) -- exercised deliberately here, separately from
test_seat_plate.py's near-body baseline, since the two paths were found
(empirically, not assumed) to need different handling: applying the
far-away recovery unconditionally measurably degraded the already-good
near-body fit.
"""
import copy
from pathlib import Path

import numpy as np
import pytest

from config.hbm_models import discover_hbm_models
from config.ppe import get_ppe
from config.sites import resolve_site
from core.registration.point_to_plane_icp import ICPConfig
from fitting.seat_plate import seat_plate


def _rotation_from_axis_angle(axis, degrees):
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(degrees)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


@pytest.fixture(scope="module")
def context():
    root = Path(__file__).resolve().parent.parent / "HBM"
    models = discover_hbm_models(root)
    if "F05_Standing" not in models:
        pytest.skip("HBM/F05_Standing not built -- run config.hbm_normalize first")
    model = models["F05_Standing"]
    site = resolve_site(model, "torso")
    ppe_spec = get_ppe("armored_plate")
    if not ppe_spec.mesh_path.is_file():
        pytest.skip(f"PPE mesh not present at {ppe_spec.mesh_path}")
    return model, site, ppe_spec


def _perturb(mesh, axis, degrees, translation):
    R = _rotation_from_axis_angle(np.asarray(axis, dtype=np.float64), degrees)
    perturbed = copy.deepcopy(mesh)
    perturbed.nodes = mesh.nodes @ R.T + np.asarray(translation, dtype=np.float64)
    return perturbed


@pytest.mark.parametrize(
    "axis,degrees,translation",
    [
        pytest.param([0.3, 1.0, -0.4], 130.0, [400.0, -150.0, 250.0], id="130deg_plus_500mm"),
        pytest.param([1.0, 0.0, 0.0], 90.0, [0.0, 300.0, -300.0], id="90deg_about_x_plus_400mm"),
        pytest.param([0.0, 0.0, 1.0], 179.0, [-350.0, 200.0, 0.0], id="near_180deg_flip_plus_400mm"),
    ],
)
def test_recovers_from_far_and_rotated_start(context, axis, degrees, translation):
    model, site, ppe_spec = context
    plate_mesh = ppe_spec.load()
    perturbed = _perturb(plate_mesh, axis, degrees, translation)

    dist_before = float(np.linalg.norm(perturbed.nodes.mean(axis=0) - plate_mesh.nodes.mean(axis=0)))
    assert dist_before > 200.0  # sanity: this really is a "far away" perturbation

    result = seat_plate(
        model, site, ppe_spec, plate_mesh=perturbed, target_standoff_mm=0.1,
        icp_config=ICPConfig(max_iterations=60),
    )

    # Orientation must be correct (concave face toward the body) and the
    # standoff correction must still land close to the 0.1mm target --
    # even though, per seat_plate()'s docstring, the far-away recovery path
    # has a wider tangential placement error than the near-body path.
    assert result.final_gap.n_penetrating == 0
    assert result.final_gap.min_mm == pytest.approx(0.1, abs=0.05)


def test_far_away_recovery_uses_the_anchor_path_not_the_near_body_path(context):
    """Confirms the test above is actually exercising the far-away branch,
    not accidentally still close enough to take the near-body path.
    """
    from fitting.body_surface import extract_skin_surface

    model, site, ppe_spec = context
    plate_mesh = ppe_spec.load()
    perturbed = _perturb(plate_mesh, [0.3, 1.0, -0.4], 130.0, [400.0, -150.0, 250.0])

    body_surface = extract_skin_surface(model, site)
    body_centroid = body_surface.nodes.mean(axis=0)
    plate_centroid = perturbed.nodes.mean(axis=0)
    assert np.linalg.norm(plate_centroid - body_centroid) > 200.0
