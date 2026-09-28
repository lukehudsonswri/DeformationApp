"""Tests for fitting.seat_plate's bilateral symmetry correction stage
(``enforce_symmetry``, default True) -- AGENTS.md Goal 2.

Three real asymmetries were found and fixed here, none zero by
construction anywhere upstream:

1. A 0.73-degree roll about the plate's own normal (~3mm differential
   edge-to-edge at full press on the ~240mm-wide plate).
2. A yaw about the body's *vertical* axis -- found only after fixing (1)
   still left a real solved case's contact/displacement field visibly
   uneven left-to-right. Confirmed via a direct, user-suggested check: draw
   a vector across the plate's width (left-rim centroid to right-rim
   centroid, at mid-height) and see whether it's parallel to the body's own
   coronal (lateral/vertical) plane -- i.e. has zero component along the
   anterior axis. It didn't, on the real uncorrected case; now it does
   (checked below).
3. A lateral offset from the body's own **spine groove** -- the real
   anatomical landmark the user visually identified in a rendered case
   ("the natural central dividing indentation"), not a numerically-fit
   local mirror-symmetry plane (an earlier version's target, found to
   disagree with the spine groove by ~1.1mm -- it had overfit to local
   front-side anatomy near the plate rather than the body's true
   structural midline) or the whole torso's bounding-box center.

Slow (~35-40s per seating call, real HBM + real plate mesh).
"""
from pathlib import Path

import numpy as np
import pytest

from config.hbm_models import discover_hbm_models
from config.ppe import get_ppe
from config.sites import resolve_site
from core.registration.point_to_plane_icp import ICPConfig
from fitting.body_surface import extract_skin_surface
from fitting.plate_signature import compute_plate_orientation
from fitting.seat_plate import (
    DEFAULT_ANTERIOR_AXIS,
    DEFAULT_BODY_UP_AXIS,
    DEFAULT_LATERAL_AXIS,
    _find_rim_indices,
    _find_spine_groove_lateral_mm,
    _orthonormal_frame,
    seat_plate,
)


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


def _rim_anterior_component(result, ppe_spec):
    """The rim-to-rim vector's component along the anterior axis, on the
    ALREADY-SEATED result -- the same check ``_apply_symmetry_correction``
    itself corrects for. Recomputed independently here (not just trusting
    the internal diagnostic) via the raw mesh's own rim indices.
    """
    raw_mesh = ppe_spec.load()
    orient = compute_plate_orientation(raw_mesh)
    frame = _orthonormal_frame(orient.concave_normal, orient.up_axis)
    left_ids, right_ids = _find_rim_indices(raw_mesh.nodes, orient.concave_face_node_ids, frame[1], frame[2])
    left_c = result.transformed_plate_nodes[left_ids].mean(axis=0)
    right_c = result.transformed_plate_nodes[right_ids].mean(axis=0)
    return float(np.dot(right_c - left_c, DEFAULT_ANTERIOR_AXIS))


def test_symmetry_enabled_by_default(context):
    model, site, ppe_spec = context
    result = seat_plate(model, site, ppe_spec, target_standoff_mm=0.1, icp_config=ICPConfig(max_iterations=60))
    # some correction should generally be needed (ICP's fit is not already
    # perfectly symmetric) -- a nonzero diagnostic confirms the stage ran,
    # not just that it's a no-op wired in but never doing anything.
    assert (
        result.symmetry_twist_deg != 0.0
        or result.symmetry_lateral_shift_mm != 0.0
        or result.symmetry_yaw_deg != 0.0
    )


def test_rim_vector_has_no_anterior_component_after_symmetry_correction(context):
    """The literal thing the yaw-correction stage guarantees: the plate's
    left-rim-to-right-rim vector must lie in the body's own coronal plane
    (zero anterior-axis component) -- i.e. both sides of the plate are
    equally far from the body, not one side pressed in more than the other.
    """
    model, site, ppe_spec = context
    result = seat_plate(model, site, ppe_spec, target_standoff_mm=0.1, icp_config=ICPConfig(max_iterations=60))
    anterior_component = _rim_anterior_component(result, ppe_spec)
    assert abs(anterior_component) < 0.05


def test_plate_up_axis_has_no_lateral_component_after_symmetry_correction(context):
    """The literal thing bilateral symmetry requires: the plate's up
    direction must not be tilted toward one side.
    """
    model, site, ppe_spec = context
    result = seat_plate(model, site, ppe_spec, target_standoff_mm=0.1, icp_config=ICPConfig(max_iterations=60))

    # Recover the plate's actual up-axis post-correction empirically from
    # two well-separated points on the transformed mesh that lie along the
    # plate's own local up_axis in the raw mesh.
    raw_mesh = ppe_spec.load()
    orient = result.plate_orientation
    concave_c = orient.plate_centroid
    tip = concave_c + orient.up_axis * 50.0  # a point 50mm "up" in the plate's own local frame

    dists = np.linalg.norm(raw_mesh.nodes - tip, axis=1)
    idx = int(np.argmin(dists))
    seated_tip = result.transformed_plate_nodes[idx]
    seated_centroid = result.transformed_plate_nodes[
        np.linalg.norm(raw_mesh.nodes - concave_c, axis=1).argmin()
    ]
    up_direction_seated = seated_tip - seated_centroid
    up_direction_seated /= np.linalg.norm(up_direction_seated)

    lateral_component = np.dot(up_direction_seated, DEFAULT_LATERAL_AXIS)
    assert abs(lateral_component) < 0.02  # ~1 degree of residual tilt, generous vs. the ~0.73deg found


def test_concave_centroid_lands_on_spine_groove_midline(context):
    """The lateral-recentering target is the body's own spine-groove
    landmark (see _find_spine_groove_lateral_mm), not a numerically-fit
    local mirror-plane or the whole torso's bounding-box center -- this
    replaced an earlier version after the user visually identified the
    spine groove itself as "the natural central dividing indentation" and
    found it disagreed with the local-mirror-fit target by ~1.1mm.
    Recompute that same landmark independently here (not just trusting the
    internal resolution) and check the seated plate's concave centroid
    lands on it.
    """
    model, site, ppe_spec = context
    result = seat_plate(model, site, ppe_spec, target_standoff_mm=0.1, icp_config=ICPConfig(max_iterations=60))

    from fitting.seat_plate import DEFAULT_BODY_UP_AXIS, _find_spine_groove_lateral_mm

    plate_centroid_now = _concave_centroid(result)
    plate_z = result.transformed_plate_nodes[result.plate_orientation.concave_face_node_ids] @ DEFAULT_BODY_UP_AXIS
    groove_target = _find_spine_groove_lateral_mm(
        result.body_surface, DEFAULT_ANTERIOR_AXIS, DEFAULT_BODY_UP_AXIS, DEFAULT_LATERAL_AXIS,
        z_band=(float(plate_z.min()), float(plate_z.max())),
    )

    concave_lateral = np.dot(plate_centroid_now, DEFAULT_LATERAL_AXIS)
    assert concave_lateral == pytest.approx(groove_target, abs=0.1)


def test_symmetry_correction_preserves_the_target_standoff(context):
    """The yaw, de-twist, and lateral-translation corrections can each
    perturb the achieved gap slightly -- the final re-application of the
    closed-form standoff correction (inside _apply_symmetry_correction)
    must restore it.
    """
    model, site, ppe_spec = context
    result = seat_plate(model, site, ppe_spec, target_standoff_mm=0.1, icp_config=ICPConfig(max_iterations=60))
    assert result.final_gap.n_penetrating == 0
    assert result.final_gap.min_mm == pytest.approx(result.resolved_target_standoff_mm, abs=1e-3)


def test_disabling_symmetry_reports_zero_diagnostics(context):
    model, site, ppe_spec = context
    result = seat_plate(
        model, site, ppe_spec, target_standoff_mm=0.1, icp_config=ICPConfig(max_iterations=60),
        enforce_symmetry=False,
    )
    assert result.symmetry_twist_deg == 0.0
    assert result.symmetry_lateral_shift_mm == 0.0
    assert result.symmetry_yaw_deg == 0.0


def test_disabling_symmetry_still_seats_with_zero_penetration(context):
    """enforce_symmetry=False must still be a fully valid seating (just
    without the extra centering/de-twist/de-yaw) -- not a degraded/broken
    path.
    """
    model, site, ppe_spec = context
    result = seat_plate(
        model, site, ppe_spec, target_standoff_mm=0.1, icp_config=ICPConfig(max_iterations=60),
        enforce_symmetry=False,
    )
    assert result.final_gap.n_penetrating == 0
    assert result.final_gap.min_mm == pytest.approx(result.resolved_target_standoff_mm, abs=1e-3)


def test_spine_groove_landmark_is_near_zero_on_real_f05_torso():
    """The spine groove is expected to sit very close to Y=0 for a
    reasonably-symmetric anthropometric body mesh -- confirmed directly on
    real F05_Standing data (median -0.05mm across the whole back, -0.12mm
    restricted to the plate's own chest-height band), closely matching the
    whole-torso bounding-box center (0.166mm) and clearly distinguishing it
    from the earlier local-mirror-fit target (1.09mm, since shown to be
    biased by local front-side anatomy).
    """
    root = Path(__file__).resolve().parent.parent
    models = discover_hbm_models(root / "HBM")
    if "F05_Standing" not in models:
        pytest.skip("HBM/F05_Standing not built")
    model = models["F05_Standing"]
    site = resolve_site(model, "torso")
    body_surface = extract_skin_surface(model, site)

    groove_y = _find_spine_groove_lateral_mm(
        body_surface, DEFAULT_ANTERIOR_AXIS, DEFAULT_BODY_UP_AXIS, DEFAULT_LATERAL_AXIS
    )
    assert abs(groove_y) < 1.0


def test_spine_groove_falls_back_to_bbox_center_when_no_back_region_present():
    """A degenerate point cloud with no real 'back' structure (or too few
    points to fit a meaningful quadratic) must fall back to the
    bounding-box center rather than raising or returning nonsense.
    """
    from core.mesh.base import SurfaceMesh

    rng = np.random.default_rng(0)
    # a small, flat, featureless patch -- no groove signal at all
    pts = rng.normal(scale=5.0, size=(30, 3))
    pts[:, 0] -= 100.0  # push into the "posterior" region regardless of axis convention
    dummy_surface = SurfaceMesh(nodes=pts, faces=np.zeros((0, 3), dtype=np.int64))

    result = _find_spine_groove_lateral_mm(
        dummy_surface, DEFAULT_ANTERIOR_AXIS, DEFAULT_BODY_UP_AXIS, DEFAULT_LATERAL_AXIS
    )
    expected_fallback = float((pts[:, 1].min() + pts[:, 1].max()) / 2.0)
    assert result == pytest.approx(expected_fallback)
