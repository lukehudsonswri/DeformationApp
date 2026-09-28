"""Regression tests for postprocess.export_state -- the GUI's "Export
Current State" button (app/viewer_app.py). Uses small synthetic HBM/PPE
fixtures, not the real ~1.79M-element production decks.
"""
from pathlib import Path

import numpy as np
import pytest

from app.gui_case_loader import GuiCase
from app.hbm_reader import BodyMesh
from app.plate_render import PlateRenderData
from config.hbm_models import HbmModel
from config.sites import SiteResolution
from core.mesh.base import ElementType, VolumeMesh
from postprocess.export_state import build_fe_model_with_ppe, export_meshes


@pytest.fixture
def hbm_model(tmp_path: Path) -> HbmModel:
    """A tiny synthetic HBM model: one hex8 flesh element (8 nodes) whose
    top face (4 of those 8 nodes) also forms one skin shell quad -- same
    "skin nodes are a subset of flesh nodes" structure as the real torso
    (see fitting/body_volume.py's module docstring) -- plus one unrelated
    context node/element elsewhere in the body that must pass through
    completely untouched.

    Nodes.k/Elements.k are written in a deliberately NOT-simple-comma-only
    style (a leading ``$`` comment line, like the real M50_Standing
    folder) specifically to catch a real regression: an earlier version of
    export_state.py assumed every HBM folder was in
    config/hbm_normalize.py's own always-comma format and used a naive
    ``line.split(",")`` reader, which crashed immediately on M50's more
    raw LS-DYNA-style deck.
    """
    folder = tmp_path / "TinyModel"
    folder.mkdir()

    nodes_path = folder / "Nodes.k"
    nodes_path.write_text(
        "*KEYWORD\n"
        "$ a comment line, like M50_Standing's real Elements.k/Nodes.k\n"
        "*NODE\n"
        "100,0,0,0,0,0\n"
        "101,1,0,0,0,0\n"
        "102,1,1,0,0,0\n"
        "103,0,1,0,0,0\n"
        "104,0,0,1,0,0\n"
        "105,1,0,1,0,0\n"
        "106,1,1,1,0,0\n"
        "107,0,1,1,0,0\n"
        "999,50,50,50,0,0\n"  # unrelated context node, elsewhere in the body
        "*END\n"
    )

    elements_path = folder / "Elements.k"
    elements_path.write_text(
        "*KEYWORD\n"
        "$ a comment line\n"
        "*ELEMENT_SOLID\n"
        "1,2000500,100,101,102,103,104,105,106,107\n"
        "*ELEMENT_SHELL\n"
        "1,2000501,104,105,106,107\n"
        "2,9999,999,999,999,999\n"  # unrelated context element, must pass through unchanged
        "*END\n"
    )

    return HbmModel(
        key="TinyModel",
        family=None,
        posture=None,
        folder=folder,
        nodes_path=nodes_path,
        elements_path=elements_path,
    )


@pytest.fixture
def site() -> SiteResolution:
    return SiteResolution(site="torso", model_key="TinyModel", solid_pids=(2000500,), shell_pids=(2000501,))


@pytest.fixture
def body(hbm_model: HbmModel) -> BodyMesh:
    return BodyMesh(
        node_ids=np.array([100, 101, 102, 103, 104, 105, 106, 107, 999], dtype=np.int64),
        xyz=np.array(
            [
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
                [50, 50, 50],
            ],
            dtype=np.float64,
        ),
        shell_pid=np.array([2000501, 9999], dtype=np.int64),
        shell_conn=np.array([[4, 5, 6, 7], [8, 8, 8, 8]], dtype=np.int64),
    )


@pytest.fixture
def case() -> GuiCase:
    # torso node set is skin+flesh combined = ids 100..107 (8 nodes); only
    # the top-face skin nodes (104..107) get real displacement here so the
    # "deformed = rest + disp" arithmetic is easy to check by hand.
    node_ids = np.array([100, 101, 102, 103, 104, 105, 106, 107], dtype=np.int64)
    disp_t1 = np.zeros((8, 3))
    disp_t1[4:8] = [0.0, 0.0, 1.0]  # nodes 104-107 move +1 in Z at t=1
    return GuiCase(
        hbm_model_key="TinyModel",
        site_name="torso",
        ppe_key="tiny_plate",
        node_ids=node_ids,
        times=np.array([0.0, 1.0]),
        displacement=np.stack([np.zeros((8, 3)), disp_t1], axis=0),
        plate_nodes_rest=np.zeros((1, 3)),
        plate_boundary_faces=np.zeros((0, 4), dtype=np.int64),
        push_direction=np.array([0.0, 0.0, -1.0]),
        total_travel_mm=2.0,
        final_time=1.0,
    )


@pytest.fixture
def plate() -> PlateRenderData:
    # A single hex8 "plate" element sitting well clear of the body/torso
    # node-id range (so its own new node ids can't accidentally collide).
    return PlateRenderData(
        nodes_rest=np.array(
            [
                [10, 10, 10], [11, 10, 10], [11, 11, 10], [10, 11, 10],
                [10, 10, 11], [11, 10, 11], [11, 11, 11], [10, 11, 11],
            ],
            dtype=np.float64,
        ),
        boundary_faces=np.zeros((0, 4), dtype=np.int64),
        push_direction=np.array([0.0, 0.0, -1.0]),
        total_travel_mm=2.0,
        final_time=1.0,
    )


@pytest.fixture(autouse=True)
def fake_ppe(monkeypatch):
    """Avoid loading the real ~619k-node PPE/plate.inp -- a tiny one hex8
    element mesh is enough to exercise the export logic.
    """
    fake_mesh = VolumeMesh(
        nodes=np.array(
            [
                [10, 10, 10], [11, 10, 10], [11, 11, 10], [10, 11, 10],
                [10, 10, 11], [11, 10, 11], [11, 11, 11], [10, 11, 11],
            ],
            dtype=np.float64,
        ),
        element_groups={ElementType.HEX8: np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)},
    )

    class _FakePpeSpec:
        key = "tiny_plate"

        def load(self):
            return fake_mesh

    monkeypatch.setattr("postprocess.export_state.get_ppe", lambda key: _FakePpeSpec())


def test_export_meshes_writes_deformed_skin_flesh_and_ppe(hbm_model, site, case, plate, tmp_path):
    out_dir = tmp_path / "out"
    paths = export_meshes(hbm_model, site, case, plate, "tiny_plate", t=1.0, out_dir=out_dir)

    assert set(paths) == {"torso_skin", "torso_flesh", "ppe"}
    for p in paths.values():
        assert p.is_file()

    skin_text = paths["torso_skin"].read_text()
    # node 104 (rest z=1) should be deformed to z=2 at t=1.0 (disp +1 in Z)
    assert "104,0.0,0.0,2.0,0,0" in skin_text

    flesh_text = paths["torso_flesh"].read_text()
    assert "*ELEMENT_SOLID" in flesh_text
    assert ",2000500," in flesh_text  # original flesh pid preserved

    ppe_text = paths["ppe"].read_text()
    assert "*ELEMENT_SOLID" in ppe_text


def test_export_meshes_does_not_touch_context_node(hbm_model, site, case, plate, tmp_path):
    out_dir = tmp_path / "out"
    paths = export_meshes(hbm_model, site, case, plate, "tiny_plate", t=1.0, out_dir=out_dir)
    # node 999 (context, outside the torso) must never appear in the torso exports
    for key in ("torso_skin", "torso_flesh"):
        assert "999," not in paths[key].read_text()


def test_build_fe_model_patches_only_torso_nodes(hbm_model, body, case, plate, tmp_path):
    out_path = tmp_path / "combined.k"
    build_fe_model_with_ppe(hbm_model, body, case, plate, "tiny_plate", t=1.0, out_path=out_path)

    text = out_path.read_text()
    lines = {ln.split(",")[0]: ln for ln in text.splitlines() if ln and ln[0].isdigit()}

    # node 104 moved from z=1 to z=2 (disp +1 in Z at t=1.0)
    assert lines["104"].split(",")[3] == "2.0"
    # node 100 (flesh-only, no displacement applied in this fixture) unchanged
    assert lines["100"].split(",")[1:4] == ["0.0", "0.0", "0.0"]
    # unrelated context node 999 completely untouched
    assert lines["999"].split(",")[1:4] == ["50.0", "50.0", "50.0"]


def test_build_fe_model_new_ppe_part_has_unique_pid_and_eid(hbm_model, body, case, plate, tmp_path):
    """The real bug this guards against: computing the new part's pid/eid
    from only ONE element block (e.g. just *ELEMENT_SOLID) can collide
    with a higher id that only appears in the OTHER block (*ELEMENT_SHELL
    contains pid 9999 in this fixture, higher than the solid block's own
    2000500 -- a naive "max solid pid + 1" would not be unique).
    """
    out_path = tmp_path / "combined.k"
    build_fe_model_with_ppe(hbm_model, body, case, plate, "tiny_plate", t=1.0, out_path=out_path)

    text = out_path.read_text()
    all_pids = set()
    all_eids = set()
    section = None
    for line in text.splitlines():
        if line in ("*ELEMENT_SOLID", "*ELEMENT_SHELL"):
            section = line
            continue
        if section and line and line[0].isdigit():
            fields = line.split(",")
            all_eids.add(int(fields[0]))
            all_pids.add(int(fields[1]))

    # original pids: 2000500 (flesh), 2000501 (skin), 9999 (context) -- the
    # new PPE part's pid must not collide with any of them, in particular
    # not with 9999 (the highest), which a solid-block-only max would miss.
    new_part_pid = max(all_pids)
    assert new_part_pid not in (2000500, 2000501, 9999)
    assert new_part_pid == 2000502  # max(2000500, 2000501, 9999) + 1

    original_eids = {1, 2}
    new_eids = all_eids - original_eids
    assert len(new_eids) == 1  # exactly the one new PPE hex8 element
