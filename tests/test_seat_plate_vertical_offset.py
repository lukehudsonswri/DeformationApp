"""Tests for fitting.seat_plate's ``vertical_offset_mm`` -- letting a user
explicitly move the plate up/down the torso (e.g. "it's sitting too low,
over the stomach, when it should cover the chest").

A real design tension was found and fixed here: point-to-plane ICP's own
tangential translation is large even for the well-seated baseline case
(~88mm out of ~90mm total, on real data) -- a symptom of the aperture
problem (core.registration.point_to_plane_icp), not a deliberate
correction. Left unclamped, it silently undoes a user's requested
reposition (a +100mm request was measured collapsing to ~12mm of net
movement). Fixed by discarding ICP's tangential translation component
whenever ``vertical_offset_mm != 0`` -- keeping its rotation and
normal-direction correction, which is what lets the plate still sit flush
against the body's local curvature at the new height. The no-offset path
is completely unchanged (verified: baseline regression values in
test_seat_plate.py are untouched by this feature).
"""
from pathlib import Path

import numpy as np
import pytest

from config.hbm_models import discover_hbm_models
from config.ppe import get_ppe
from config.sites import resolve_site
from core.registration.point_to_plane_icp import ICPConfig
from fitting.seat_plate import seat_plate


@pytest.fixture(scope="module")
def context():
    root = Path(__file__).resolve().parent.parent
    models = discover_hbm_models(root / "HBM")
    if "F05_Standing" not in models:
        pytest.skip("HBM/F05_Standing not built")
    model = models["F05_Standing"]
    site = resolve_site(model, "torso")
    ppe_spec = get_ppe("armored_plate")
    if not ppe_spec.mesh_path.is_file():
        pytest.skip(f"PPE mesh not present at {ppe_spec.mesh_path}")
    return model, site, ppe_spec


def _concave_centroid(result):
    return result.transformed_plate_nodes[result.plate_orientation.concave_face_node_ids].mean(axis=0)


def test_raising_and_lowering_move_the_plate_the_requested_relative_distance(context):
    """+100mm and -80mm requests should differ from each other by ~180mm
    along the body's superior/inferior (Z) axis -- checked relative to each
    other (not to the undisturbed baseline, which has its own large,
    ICP-driven tangential settling that makes it a moving target -- see
    module docstring).
    """
    model, site, ppe_spec = context
    cfg = ICPConfig(max_iterations=60)

    raised = seat_plate(model, site, ppe_spec, target_standoff_mm=0.1, icp_config=cfg, vertical_offset_mm=100.0)
    lowered = seat_plate(model, site, ppe_spec, target_standoff_mm=0.1, icp_config=cfg, vertical_offset_mm=-80.0)

    delta_z = _concave_centroid(raised)[2] - _concave_centroid(lowered)[2]
    # body_up_axis = [0, 0, -1] here, so "raised" (more superior) has more
    # negative Z than "lowered" -- delta should be close to -(100 - (-80)) = -180
    assert delta_z == pytest.approx(-180.0, abs=5.0)


def test_raised_plate_still_seats_with_negligible_penetration(context):
    """The tangential clamp trades some fit precision for preserving the
    user's requested height -- final gap should still land close to the
    target standoff, with any residual penetration below the mesh's own
    geometric resolution floor (~0.065mm on this mesh, see fitting.standoff).
    """
    model, site, ppe_spec = context
    result = seat_plate(
        model, site, ppe_spec, target_standoff_mm=0.1, icp_config=ICPConfig(max_iterations=60), vertical_offset_mm=100.0
    )
    assert result.final_gap.min_mm > -0.1
    assert result.final_gap.min_mm == pytest.approx(0.1, abs=0.15)


def test_lowered_plate_seats_cleanly_with_zero_penetration(context):
    model, site, ppe_spec = context
    result = seat_plate(
        model, site, ppe_spec, target_standoff_mm=0.1, icp_config=ICPConfig(max_iterations=60), vertical_offset_mm=-80.0
    )
    assert result.final_gap.n_penetrating == 0
    assert result.final_gap.min_mm == pytest.approx(0.1, abs=1e-2)


def test_zero_offset_is_a_no_op_matching_the_default_path(context):
    """vertical_offset_mm=0.0 (the default) must take the exact same
    (unclamped) code path as calling seat_plate with no offset argument at
    all -- this is what keeps test_seat_plate.py's regression values valid.
    """
    model, site, ppe_spec = context
    cfg = ICPConfig(max_iterations=60)
    a = seat_plate(model, site, ppe_spec, target_standoff_mm=0.1, icp_config=cfg)
    b = seat_plate(model, site, ppe_spec, target_standoff_mm=0.1, icp_config=cfg, vertical_offset_mm=0.0)
    np.testing.assert_allclose(a.transformed_plate_nodes, b.transformed_plate_nodes)
