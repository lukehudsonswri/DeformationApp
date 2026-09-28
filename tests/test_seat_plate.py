"""End-to-end regression test for fitting.seat_plate -- shape-signature
seed rotation + local point-to-plane ICP seating of the rigid armored plate
against F05_Standing's torso (AGENTS.md Goal 2).

Slow (~35-40s: loads the 619k-node plate mesh and parses the torso skin
surface from the full HBM deck) -- kept to a single end-to-end test plus a
couple of cheap assertions on its result, rather than re-running the whole
pipeline per assertion.
"""
from pathlib import Path

import numpy as np
import pytest

from config.hbm_models import discover_hbm_models
from config.ppe import get_ppe
from config.sites import resolve_site
from core.registration.point_to_plane_icp import ICPConfig
from fitting.plate_signature import compute_plate_orientation
from fitting.seat_plate import seat_plate


@pytest.fixture(scope="module")
def seating_result():
    root = Path(__file__).resolve().parent.parent / "HBM"
    models = discover_hbm_models(root)
    if "F05_Standing" not in models:
        pytest.skip("HBM/F05_Standing not built -- run config.hbm_normalize first")
    model = models["F05_Standing"]
    site = resolve_site(model, "torso")
    ppe_spec = get_ppe("armored_plate")
    if not ppe_spec.mesh_path.is_file():
        pytest.skip(f"PPE mesh not present at {ppe_spec.mesh_path}")

    return seat_plate(
        model, site, ppe_spec, target_standoff_mm=0.1, icp_config=ICPConfig(max_iterations=60)
    )


def test_seed_rotation_is_a_proper_rotation(seating_result):
    R = seating_result.seed_rotation
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)


def test_icp_meaningfully_improves_on_the_seed(seating_result):
    """Regression values from direct verification during development: the
    shipped PPE/plate.inp is already roughly seated against F05_Standing
    (~13.2mm mean nearest-neighbor distance after the signature seed
    rotation alone); local point-to-plane ICP refines this to ~5.3mm and
    stays stable there (not diverging, unlike CPD/point-to-point ICP tried
    directly against this same data -- see core.registration.point_to_plane_icp).
    """
    assert seating_result.pre_icp_mean_dist_mm == pytest.approx(13.19, abs=0.5)
    assert seating_result.icp_result.final_mean_dist_mm == pytest.approx(5.30, abs=0.5)
    assert seating_result.icp_result.final_mean_dist_mm < seating_result.pre_icp_mean_dist_mm


def test_transformed_plate_preserves_node_count(seating_result):
    original_count = get_ppe("armored_plate").load().nodes.shape[0]
    assert seating_result.transformed_plate_nodes.shape == (original_count, 3)


def test_concave_face_points_toward_body_after_seating(seating_result):
    """After the full seat_plate() pipeline, the plate's (now-transformed)
    concave face should still face the torso -- i.e. seating didn't
    accidentally flip the plate. Recomputes orientation on the transformed
    geometry directly, rather than trusting the pre-transform value.
    """
    from core.mesh.base import ElementType, VolumeMesh

    transformed_mesh = VolumeMesh(
        nodes=seating_result.transformed_plate_nodes,
        element_groups={ElementType.HEX8: get_ppe("armored_plate").load().element_groups[ElementType.HEX8]},
        element_type=ElementType.HEX8,
        name="transformed_plate",
    )
    orient_after = compute_plate_orientation(transformed_mesh)
    concave_centroid = transformed_mesh.nodes[orient_after.concave_face_node_ids].mean(axis=0)
    body_centroid = seating_result.body_surface.nodes.mean(axis=0)
    direction_to_body = body_centroid - concave_centroid
    direction_to_body /= np.linalg.norm(direction_to_body)
    assert np.dot(orient_after.concave_normal, direction_to_body) > 0.5


def test_final_min_gap_matches_target_standoff_precisely(seating_result):
    """The closed-form standoff correction (AGENTS.md section 2.4 step 2)
    should land the minimum signed point-to-face gap on the *real* (not
    locally-cropped) body skin almost exactly at the resolved target
    standoff, with zero penetrating nodes.
    """
    gap = seating_result.final_gap
    assert gap.min_mm == pytest.approx(seating_result.resolved_target_standoff_mm, abs=1e-3)
    assert gap.n_penetrating == 0


def test_standoff_floor_is_below_the_default_target(seating_result):
    """F05_Standing's torso skin (median edge ~6.2mm) gives a floor well
    under the 0.1mm default target -- the default should not need clamping
    for this mesh.
    """
    assert seating_result.standoff_floor.floor_mm < 0.1
    assert seating_result.resolved_target_standoff_mm == pytest.approx(0.1)


def test_unreasonably_tight_target_gets_clamped_to_the_mesh_floor():
    """Requesting a standoff far below what this mesh can geometrically
    resolve (AGENTS.md section 2.4) must be raised, not silently honored.
    """
    root = Path(__file__).resolve().parent.parent / "HBM"
    models = discover_hbm_models(root)
    if "F05_Standing" not in models:
        pytest.skip("HBM/F05_Standing not built")
    model = models["F05_Standing"]
    site = resolve_site(model, "torso")
    ppe_spec = get_ppe("armored_plate")
    if not ppe_spec.mesh_path.is_file():
        pytest.skip(f"PPE mesh not present at {ppe_spec.mesh_path}")

    result = seat_plate(
        model, site, ppe_spec, target_standoff_mm=1e-6, icp_config=ICPConfig(max_iterations=60)
    )
    assert result.resolved_target_standoff_mm == pytest.approx(result.standoff_floor.floor_mm)
    assert result.resolved_target_standoff_mm > 1e-6
    assert result.final_gap.min_mm == pytest.approx(result.resolved_target_standoff_mm, abs=1e-3)
