"""Tests for febio.assemble_case -- structural checks only (node/element
counts, contact surface non-emptiness, XML well-formedness, and Output
scoping). Does NOT solve the case in FEBio -- see
febio/build_preliminary_case.py's module docstring for confirmation that
the assembled cases do solve to normal termination.
"""
from pathlib import Path

import numpy as np
import pytest

from config.hbm_models import discover_hbm_models
from config.ppe import get_ppe
from config.sites import resolve_site
from core.mesh.base import ElementType
from core.registration.point_to_plane_icp import ICPConfig
from febio.assemble_case import assemble_torso_plate_case
from febio.loadcurve import derive_load_curve
from fitting.body_volume import extract_torso_volume
from fitting.seat_plate import seat_plate


@pytest.fixture(scope="module")
def assembled_case(tmp_path_factory):
    root = Path(__file__).resolve().parent.parent
    models = discover_hbm_models(root / "HBM")
    if "F05_Standing" not in models:
        pytest.skip("HBM/F05_Standing not built")
    model = models["F05_Standing"]
    site = resolve_site(model, "torso")
    ppe_spec = get_ppe("armored_plate")
    if not ppe_spec.mesh_path.is_file():
        pytest.skip(f"PPE mesh not present at {ppe_spec.mesh_path}")

    seating = seat_plate(model, site, ppe_spec, target_standoff_mm=0.1, icp_config=ICPConfig(max_iterations=60))
    torso = extract_torso_volume(model, site)
    lc = derive_load_curve(target_standoff_mm=seating.resolved_target_standoff_mm, indentation_mm=4.25)
    concave_normal_final = seating.plate_orientation.concave_normal @ seating.seed_rotation @ seating.icp_result.rotation
    concave_normal_final /= np.linalg.norm(concave_normal_final)

    plate_mesh = ppe_spec.load()
    plate_hexes = plate_mesh.element_groups[ElementType.HEX8]

    out_path = tmp_path_factory.mktemp("febio_case") / "test_case.feb"
    assemble_torso_plate_case(
        torso=torso,
        plate_nodes=seating.transformed_plate_nodes,
        plate_hexes=plate_hexes,
        plate_concave_face_node_ids=seating.plate_orientation.concave_face_node_ids,
        load_curve=lc,
        into_body_direction=concave_normal_final,
        out_path=out_path,
    )
    return out_path, torso, plate_hexes


def test_case_file_is_well_formed_xml(assembled_case):
    import xml.etree.ElementTree as ET

    out_path, _, _ = assembled_case
    tree = ET.parse(out_path)
    assert tree.getroot().tag == "febio_spec"


def test_node_and_element_counts_match_inputs(assembled_case):
    from core.mesh.febio import get_febio_info

    out_path, torso, plate_hexes = assembled_case
    info = get_febio_info(str(out_path))
    counts = {es["name"]: es["count"] for es in info["element_sets"]}
    assert counts["TorsoSkin"] == len(torso.skin_quads)
    assert counts["TorsoFlesh"] == len(torso.flesh_hexes)
    assert counts["Plate"] == len(plate_hexes)


def test_contact_secondary_surface_is_nonempty(assembled_case):
    """A real regression this test would have caught early: if the
    concave-face node id filter or the plate's boundary-face extraction
    were broken, this surface would silently come out empty and contact
    could never engage regardless of geometry.
    """
    out_path, _, _ = assembled_case
    txt = out_path.read_text(encoding="ISO-8859-1")
    secondary_section = txt.split('<Surface name="SlidingElastic1Secondary">')[1].split("</Surface>")[0]
    assert secondary_section.count("<quad4") > 1000


def test_free_boundary_and_skin_nodes_are_disjoint_by_construction(assembled_case):
    """Documents a real, previously-latent bug found during investigation:
    ``TorsoVolume.free_boundary_node_ids`` is *always* disjoint from the
    skin's own node set by construction (it's computed as flesh-surface
    minus skin), so any later code trying to find "free boundary nodes
    that are also skin nodes" (e.g. to apply a shell zero-rotation BC) will
    always find an empty set -- silently, not an error. Confirmed harmless
    for this case (equation count was unaffected by adding a rotation BC on
    the skin's own boundary edges instead), but worth keeping this
    assumption explicit and tested rather than tripping over it again.
    """
    _, torso, _ = assembled_case
    skin_node_idx = set(np.unique(torso.skin_quads).tolist())
    free_idx = set(torso.free_boundary_node_ids.tolist())
    assert skin_node_idx.isdisjoint(free_idx)


def test_node_data_log_is_scoped_to_torso_only(assembled_case):
    """A real bug found and fixed here: the node_data logfile output was
    previously unscoped, so FEBio logged every node in the model --
    including the rigid plate's own ~619k nodes -- at every solved step,
    even though postprocess.project_displacement only ever reads the
    torso's nodes. A real solved case produced a ~780MB text file as a
    result, ~93% of it discarded downstream. Fixed by adding a
    "TorsoNodes" NodeSet (ids 1..n_torso) and referencing it via
    node_set="TorsoNodes" on the node_data block -- confirmed directly
    (a tiny synthetic multi-domain case) that FEBio 4.12 honors this.
    """
    out_path, torso, _ = assembled_case
    txt = out_path.read_text(encoding="ISO-8859-1")
    assert '<NodeSet name="TorsoNodes">' in txt
    assert 'node_set="TorsoNodes"' in txt
    assert "<node_data" in txt

    node_set_section = txt.split('<NodeSet name="TorsoNodes">')[1].split("</NodeSet>")[0]
    ids_in_set = {int(x) for x in node_set_section.replace("\n", "").replace("\t", "").split(",") if x}
    assert ids_in_set == set(range(1, len(torso.nodes) + 1))

