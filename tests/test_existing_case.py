"""Tests for the "already at a good standoff -> skip the fit" check and for
adopting an externally built ``.feb`` (febio.existing_case), plus the
main_2-side handling of a strap-driven plate. Synthetic data only, except the
single seat_plate test, which needs the real HBM/PPE and skips without them.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from core.mesh.base import ElementType, SurfaceMesh
from app.plate_render import PlateRenderData, StrapRenderData, plate_points_at_time, strap_points_at_time
from febio.existing_case import patch_feb_outputs, read_existing_case, read_strap_geometry
from fitting.standoff import check_standoff
from postprocess.gui_case import build_gui_case
from postprocess.plate_from_feb import ExternallyDrivenPlateError, read_plate_render_from_feb
from postprocess.rigid_body_log import read_rigid_body_travel, read_rigid_body_trajectory

_FEB = """\
<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="4.0">
	<Control>
		<time_steps>4</time_steps>
		<step_size>0.25</step_size>
	</Control>
	<Material>
		<material id="1" name="Skin" type="Ogden"/>
		<material id="2" name="PlateRigid" type="rigid body"/>
	</Material>
	<Mesh>
		<Nodes name="AllNodes">
			<node id="1">0,0,0</node>
			<node id="2">10,0,0</node>
			<node id="3">10,10,0</node>
			<node id="4">0,10,0</node>
			<node id="5">0,0,0.5</node>
			<node id="6">10,0,0.5</node>
			<node id="7">10,10,0.5</node>
			<node id="8">0,10,0.5</node>
		</Nodes>
		<Elements type="quad4" name="TorsoSkin">
			<elem id="1">1,2,3,4</elem>
		</Elements>
		<Elements type="quad4" name="Plate">
			<elem id="2">5,6,7,8</elem>
		</Elements>
		<Surface name="Primary">
			<quad4 id="1">1,2,3,4</quad4>
		</Surface>
		<Surface name="Secondary">
			<quad4 id="1">5,6,7,8</quad4>
		</Surface>
		<SurfacePair name="C">
			<primary>Primary</primary>
			<secondary>Secondary</secondary>
		</SurfacePair>
	</Mesh>
	<MeshDomains>
		<ShellDomain name="TorsoSkin" mat="Skin" type="elastic-shell"/>
		<ShellDomain name="Plate" mat="PlateRigid" type="rigid-shell"/>
	</MeshDomains>
	<Rigid>
		<rigid_bc name="PlateOnlyX" type="rigid_fixed">
			<rb>PlateRigid</rb>
			<Rx_dof>0</Rx_dof>
			<Ry_dof>1</Ry_dof>
			<Rz_dof>1</Rz_dof>
			<Ru_dof>1</Ru_dof>
			<Rv_dof>1</Rv_dof>
			<Rw_dof>1</Rw_dof>
		</rigid_bc>
	</Rigid>
	<Output>
		<logfile>
			<rigid_body_data data="x;y;z;Fx" delim="," file="runs/plate.txt">2</rigid_body_data>
		</logfile>
	</Output>
</febio_spec>
"""

# Same plate, plus one strap quad (nodes 9-12, all above the 4 torso nodes).
_FEB_STRAP = _FEB.replace(
    '\t\t</Nodes>',
    '\t\t\t<node id="9">20,0,0.5</node>\n\t\t\t<node id="10">30,0,0.5</node>\n'
    '\t\t\t<node id="11">30,10,0.5</node>\n\t\t\t<node id="12">20,10,0.5</node>\n\t\t</Nodes>',
).replace(
    '\t\t<Surface name="Primary">',
    '\t\t<Elements type="quad4" name="Strap">\n\t\t\t<elem id="3">9,10,11,12</elem>\n\t\t</Elements>\n'
    '\t\t<Surface name="Primary">',
)

_PLATE_LOG = """\
*Step  = 0
*Time  = 0
*Data  = x;y;z;Fx
2,10,0,0,0
*Step  = 1
*Time  = 1
*Data  = x;y;z;Fx
2,7.5,0,0,0
"""

_STRAP_LOG = """\
*Step  = 0
*Time  = 0
*Data  = x;y;z;ux;uy;uz
9 20 0 0.5 0 0 0
10 30 0 0.5 0 0 0
11 30 10 0.5 0 0 0
12 20 10 0.5 0 0 0
*Step  = 1
*Time  = 1
*Data  = x;y;z;ux;uy;uz
12 20 10 0.5 -2 0 0
9 20 0 0.5 -2 0 0
10 30 0 0.5 -1 0 0
11 30 10 0.5 -1 0 0
"""

_NODE_LOG = """\
*Step  = 0
*Time  = 0
*Data  = x;y;z;ux;uy;uz
1 0 0 0 0 0 0
2 0 0 0 0 0 0
*Step  = 1
*Time  = 1
*Data  = x;y;z;ux;uy;uz
1 0 0 0 0.1 0 0
2 0 0 0 0.2 0 0
"""


def _flat_surface(half_extent=50.0):
    nodes = np.array(
        [[-half_extent, -half_extent, 0.0], [half_extent, -half_extent, 0.0],
         [half_extent, half_extent, 0.0], [-half_extent, half_extent, 0.0]]
    )
    surface = SurfaceMesh(
        nodes=nodes, faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64), face_type=ElementType.TRI3, name="flat"
    )
    surface.compute_normals()
    return surface


@pytest.fixture
def feb_path(tmp_path: Path) -> Path:
    path = tmp_path / "case.feb"
    path.write_bytes(_FEB.encode("ISO-8859-1"))
    return path


def test_check_standoff_accepts_gap_inside_band_and_rejects_outside():
    surface = _flat_surface()
    outward = np.sign(surface.face_normals[0][2])
    pts = np.array([[0.0, 0.0, 0.5 * outward], [5.0, 5.0, 0.8 * outward]])

    ok = check_standoff(pts, surface, (0.0, 1.0))
    assert ok.is_acceptable
    assert ok.gap.min_mm == pytest.approx(0.5)

    assert not check_standoff(pts, surface, (0.0, 0.3)).is_acceptable  # too far off the skin
    assert not check_standoff(pts, surface, (0.6, 1.0)).is_acceptable  # closer than the band allows
    assert not check_standoff(-pts, surface, (0.0, 1.0)).is_acceptable  # penetrating


def test_read_existing_case_finds_plate_surface_axes_and_log(feb_path: Path):
    case = read_existing_case(ET.parse(feb_path).getroot(), feb_path)
    assert case.plate_node_ids.tolist() == [5, 6, 7, 8]
    assert case.concave_node_ids.tolist() == [5, 6, 7, 8]
    assert case.plate_elements_local.tolist() == [[0, 1, 2, 3]]
    assert case.free_axes == [0]
    assert case.plate_log_file == "runs/plate.txt"
    assert case.final_time == pytest.approx(1.0)


def test_patch_adds_torso_nodeset_and_node_data_and_is_idempotent(feb_path: Path):
    case = read_existing_case(ET.parse(feb_path).getroot(), feb_path)
    changed, plate_log = patch_feb_outputs(feb_path, 4, case)
    assert changed
    assert plate_log == "plate.txt"  # existing plate log is reused (folder prefix dropped), not duplicated

    root = ET.parse(feb_path).getroot()
    node_set = root.find("Mesh/NodeSet[@name='TorsoNodes']")
    assert [int(i) for i in node_set.text.replace("\n", "").replace("\t", "").split(",")] == [1, 2, 3, 4]
    node_data = root.find("Output/logfile/node_data")
    assert node_data.get("data") == "x;y;z;ux;uy;uz"
    assert node_data.get("file") == "case_node_displacement.txt"
    assert node_data.get("node_set") == "TorsoNodes"
    assert len(root.findall("Output/logfile/rigid_body_data")) == 1
    assert root.find("Output/logfile/rigid_body_data").get("file") == "plate.txt"  # no runs/ folder needed
    assert feb_path.with_name("case.feb.bak").is_file()

    changed_again, _ = patch_feb_outputs(feb_path, 4, case)
    assert not changed_again
    assert len(ET.parse(feb_path).getroot().findall("Output/logfile/node_data")) == 1


def test_patch_adds_plate_log_and_logfile_when_missing(tmp_path: Path):
    text = _FEB.replace(
        '\t\t<logfile>\n\t\t\t<rigid_body_data data="x;y;z;Fx" delim="," file="runs/plate.txt">2</rigid_body_data>\n\t\t</logfile>\n',
        "",
    )
    path = tmp_path / "bare.feb"
    path.write_bytes(text.encode("ISO-8859-1"))
    case = read_existing_case(ET.parse(path).getroot(), path)
    assert case.plate_log_file is None

    _, plate_log = patch_feb_outputs(path, 4, case)
    assert plate_log == "bare_plate_rigid_body.txt"
    root = ET.parse(path).getroot()
    assert root.find("Output/logfile/node_data") is not None
    rb = root.find("Output/logfile/rigid_body_data")
    assert rb.text == "2" and rb.get("file") == plate_log


def test_patch_preserves_crlf_line_endings(tmp_path: Path):
    path = tmp_path / "crlf.feb"
    path.write_bytes(_FEB.replace("\n", "\r\n").encode("ISO-8859-1"))
    case = read_existing_case(ET.parse(path).getroot(), path)
    patch_feb_outputs(path, 4, case)
    raw = path.read_bytes()
    assert raw.count(b"\n") == raw.count(b"\r\n")


def test_read_rigid_body_travel_handles_comma_delimited_log(tmp_path: Path):
    log = tmp_path / "plate.txt"
    log.write_text(_PLATE_LOG)
    assert read_rigid_body_travel(log, np.array([-1.0, 0.0, 0.0])) == pytest.approx(2.5)


def test_plate_from_feb_flags_externally_driven_plate(feb_path: Path):
    with pytest.raises(ExternallyDrivenPlateError):
        read_plate_render_from_feb(feb_path)


def test_build_gui_case_uses_sidecar_and_plate_log_for_strap_driven_plate(feb_path: Path):
    cases_dir = feb_path.parent
    (cases_dir / "case_node_displacement.txt").write_text(_NODE_LOG)
    (cases_dir / "runs").mkdir()
    (cases_dir / "runs" / "plate.txt").write_text(_PLATE_LOG)
    np.savez_compressed(
        cases_dir / "case_node_map.npz",
        torso_node_ids=np.array([1001, 1002], dtype=np.int64),
        n_torso=2,
        hbm_model_key="F05_Standing",
        site_name="torso",
        ppe_key="armored_plate",
    )
    np.savez_compressed(
        cases_dir / "case_plate_render.npz",
        nodes_rest=np.zeros((4, 3)),
        boundary_faces=np.array([[0, 1, 2, 3]], dtype=np.int64),
        push_direction=np.array([-1.0, 0.0, 0.0]),
        total_travel_mm=0.0,
        final_time=1.0,
        plate_log_file="runs/plate.txt",
        hbm_model_key="F05_Standing",
        site_name="torso",
        ppe_key="armored_plate",
    )

    out_path = build_gui_case("case", cases_dir)
    with np.load(out_path) as data:
        assert float(data["total_travel_mm"]) == pytest.approx(2.5)
        assert data["push_direction"].tolist() == [-1.0, 0.0, 0.0]


def test_seat_plate_skips_fit_when_plate_already_in_acceptable_band():
    from config.hbm_models import discover_hbm_models
    from config.ppe import get_ppe
    from config.sites import resolve_site
    from fitting.seat_plate import seat_plate

    models = discover_hbm_models(Path(__file__).resolve().parent.parent / "HBM")
    if "F05_Standing" not in models:
        pytest.skip("HBM/F05_Standing not built")
    ppe_spec = get_ppe("armored_plate")
    if not ppe_spec.mesh_path.is_file():
        pytest.skip(f"PPE mesh not present at {ppe_spec.mesh_path}")

    model = models["F05_Standing"]
    site = resolve_site(model, "torso")
    plate_mesh = ppe_spec.load()

    # A band wide enough to contain any gap forces the skip path.
    skipped = seat_plate(model, site, ppe_spec, plate_mesh=plate_mesh, accept_standoff_range_mm=(-1e4, 1e4))
    assert skipped.fit_skipped
    np.testing.assert_array_equal(skipped.transformed_plate_nodes, plate_mesh.nodes)
    np.testing.assert_array_equal(skipped.seed_rotation, np.eye(3))

    # A non-zero vertical offset is an explicit request to move the plate, so the band is ignored.
    moved = seat_plate(
        model, site, ppe_spec, plate_mesh=plate_mesh, accept_standoff_range_mm=(-1e4, 1e4), vertical_offset_mm=5.0
    )
    assert not moved.fit_skipped

def test_read_rigid_body_trajectory_returns_times_and_positions(tmp_path: Path):
    log = tmp_path / "plate.txt"
    log.write_text(_PLATE_LOG)
    times, positions = read_rigid_body_trajectory(log)
    assert times.tolist() == [0.0, 1.0]
    assert positions.tolist() == [[10.0, 0.0, 0.0], [7.5, 0.0, 0.0]]


def test_plate_points_follow_logged_trajectory_when_present():
    base = dict(
        nodes_rest=np.zeros((1, 3)), boundary_faces=np.zeros((0, 4), dtype=np.int64),
        push_direction=np.array([-1.0, 0.0, 0.0]), total_travel_mm=99.0, final_time=1.0,
    )
    traj = PlateRenderData(**base, trajectory_times=np.array([0.0, 1.0]), trajectory_offsets=np.array([[0.0, 0, 0], [-2.0, 0, 0]]))
    np.testing.assert_allclose(plate_points_at_time(traj, 0.5), [[-1.0, 0.0, 0.0]])
    np.testing.assert_allclose(plate_points_at_time(traj, 5.0), [[-2.0, 0.0, 0.0]])  # clamped
    # without a trajectory the old straight-line model is unchanged
    np.testing.assert_allclose(plate_points_at_time(PlateRenderData(**base), 0.5), [[-49.5, 0.0, 0.0]])


def test_strap_points_interpolate_between_solved_steps():
    strap = StrapRenderData(
        nodes_rest=np.zeros((1, 3)), faces=np.zeros((0, 4), dtype=np.int64),
        times=np.array([0.0, 1.0]), displacement=np.array([[[0.0, 0, 0]], [[4.0, 0, 0]]]),
    )
    np.testing.assert_allclose(strap_points_at_time(strap, 0.25), [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(strap_points_at_time(strap, 9.0), [[4.0, 0.0, 0.0]])


def test_read_strap_geometry_finds_strap_block_and_ignores_plate_and_torso(tmp_path: Path):
    path = tmp_path / "strap.feb"
    path.write_bytes(_FEB_STRAP.encode("ISO-8859-1"))
    root = ET.parse(path).getroot()
    case = read_existing_case(root, path)
    assert case.plate_domain_name == "Plate"
    straps = read_strap_geometry(root, path, case.plate_domain_name, 4)
    assert straps.node_ids.tolist() == [9, 10, 11, 12]
    assert straps.log_ids.tolist() == [9, 10, 11, 12]
    assert straps.faces_local.tolist() == [[0, 1, 2, 3]]
    assert straps.nodes_rest[1].tolist() == [30.0, 0.0, 0.5]


def test_read_strap_geometry_returns_none_without_straps(feb_path: Path):
    root = ET.parse(feb_path).getroot()
    case = read_existing_case(root, feb_path)
    assert read_strap_geometry(root, feb_path, case.plate_domain_name, 4) is None


def test_patch_adds_strap_logging_only_when_straps_given(tmp_path: Path):
    path = tmp_path / "strap.feb"
    path.write_bytes(_FEB_STRAP.encode("ISO-8859-1"))
    case = read_existing_case(ET.parse(path).getroot(), path)

    patch_feb_outputs(path, 4, case)
    root = ET.parse(path).getroot()
    assert root.find("Mesh/NodeSet[@name='StrapNodes']") is None
    assert [e.get("file") for e in root.findall("Output/logfile/node_data")] == ["strap_node_displacement.txt"]

    changed, _ = patch_feb_outputs(path, 4, case, strap_node_ids=np.array([9, 10, 11, 12]))
    assert changed
    root = ET.parse(path).getroot()
    assert root.find("Mesh/NodeSet[@name='StrapNodes']").text.replace("\n", "").replace("\t", "") == "9,10,11,12"
    strap_data = [e for e in root.findall("Output/logfile/node_data") if e.get("node_set") == "StrapNodes"]
    assert [e.get("file") for e in strap_data] == ["strap_strap_displacement.txt"]

    changed_again, _ = patch_feb_outputs(path, 4, case, strap_node_ids=np.array([9, 10, 11, 12]))
    assert not changed_again


def _write_case_files(cases_dir: Path, with_straps: bool) -> None:
    (cases_dir / "case_node_displacement.txt").write_text(_NODE_LOG)
    (cases_dir / "runs").mkdir(exist_ok=True)
    (cases_dir / "runs" / "plate.txt").write_text(_PLATE_LOG)
    np.savez_compressed(
        cases_dir / "case_node_map.npz",
        torso_node_ids=np.array([1001, 1002], dtype=np.int64),
        n_torso=2, hbm_model_key="F05_Standing", site_name="torso", ppe_key="armored_plate",
    )
    extra = {}
    if with_straps:
        (cases_dir / "case_strap_displacement.txt").write_text(_STRAP_LOG)
        extra = dict(
            strap_node_ids=np.array([9, 10, 11, 12], dtype=np.int64),
            strap_log_ids=np.array([9, 10, 11, 12], dtype=np.int64),
            strap_nodes_rest=np.array([[20.0, 0, 0.5], [30, 0, 0.5], [30, 10, 0.5], [20, 10, 0.5]]),
            strap_faces=np.array([[0, 1, 2, 3]], dtype=np.int64),
            strap_log_file="case_strap_displacement.txt",
        )
    np.savez_compressed(
        cases_dir / "case_plate_render.npz",
        nodes_rest=np.zeros((4, 3)), boundary_faces=np.array([[0, 1, 2, 3]], dtype=np.int64),
        push_direction=np.array([-1.0, 0.0, 0.0]), total_travel_mm=0.0, final_time=1.0,
        plate_log_file="runs/plate.txt", hbm_model_key="F05_Standing", site_name="torso",
        ppe_key="armored_plate", **extra,
    )


def test_gui_case_stores_plate_trajectory_and_straps_when_present(feb_path: Path):
    _write_case_files(feb_path.parent, with_straps=True)
    with np.load(build_gui_case("case", feb_path.parent)) as data:
        assert data["plate_traj_times"].tolist() == [0.0, 1.0]
        assert data["plate_traj_offsets"].tolist() == [[0.0, 0.0, 0.0], [-2.5, 0.0, 0.0]]
        assert data["strap_nodes_rest"].shape == (4, 3)
        assert data["strap_faces"].tolist() == [[0, 1, 2, 3]]
        assert data["strap_times"].tolist() == [0.0, 1.0]
        # the log lists node 12 first at step 1; displacement must come back in id order
        assert data["strap_displacement"][1][:, 0].tolist() == [-2.0, -1.0, -1.0, -2.0]


def test_gui_case_has_no_strap_arrays_without_strap_geometry(feb_path: Path):
    _write_case_files(feb_path.parent, with_straps=False)
    with np.load(build_gui_case("case", feb_path.parent)) as data:
        assert "plate_traj_times" in data.files
        assert not any(k.startswith("strap_") for k in data.files)


def test_gui_case_strap_log_missing_raises(feb_path: Path):
    _write_case_files(feb_path.parent, with_straps=True)
    (feb_path.parent / "case_strap_displacement.txt").unlink()
    with pytest.raises(FileNotFoundError):
        build_gui_case("case", feb_path.parent)

def test_patch_flattens_all_logfile_paths_but_leaves_other_text_alone(tmp_path: Path):
    text = _FEB.replace(
        '<rigid_body_data data="x;y;z;Fx" delim="," file="runs/plate.txt">',
        '<node_data data="Rx" file="runs\\sub/buckle.txt">1</node_data>\n'
        '\t\t\t<rigid_body_data data="x;y;z;Fx" delim="," file="runs/plate.txt">',
    )
    path = tmp_path / "flat.feb"
    path.write_bytes(text.encode("ISO-8859-1"))
    case = read_existing_case(ET.parse(path).getroot(), path)
    patch_feb_outputs(path, 4, case)

    files = [e.get("file") for e in ET.parse(path).getroot().findall("Output/logfile/*")]
    assert files == ["buckle.txt", "plate.txt", "flat_node_displacement.txt"]
    assert not (tmp_path / "runs").exists()

def test_strap_log_ids_are_file_positions_when_node_ids_are_sparse(tmp_path: Path):
    """FEBio logs nodes under their position in the file, not their id (real pilot: ids 666643+ logged as 157525+)."""
    sparse = _FEB_STRAP
    for old, new in (("9", "609"), ("10", "610"), ("11", "611"), ("12", "612")):
        sparse = sparse.replace(f'<node id="{old}">', f'<node id="{new}">')
    sparse = sparse.replace("9,10,11,12", "609,610,611,612")
    path = tmp_path / "sparse.feb"
    path.write_bytes(sparse.encode("ISO-8859-1"))
    root = ET.parse(path).getroot()
    case = read_existing_case(root, path)
    straps = read_strap_geometry(root, path, case.plate_domain_name, 4)
    assert straps.node_ids.tolist() == [609, 610, 611, 612]
    assert straps.log_ids.tolist() == [9, 10, 11, 12]


def test_gui_case_matches_strap_log_rows_by_logged_id_not_node_id(feb_path: Path):
    _write_case_files(feb_path.parent, with_straps=True)
    cases_dir = feb_path.parent
    # sidecar says the strap nodes are file ids 609..612, logged as positions 9..12 (in a scrambled node order)
    side = dict(np.load(cases_dir / "case_plate_render.npz"))
    side["strap_node_ids"] = np.array([609, 610, 611, 612], dtype=np.int64)
    np.savez_compressed(cases_dir / "case_plate_render.npz", **side)
    with np.load(build_gui_case("case", cases_dir)) as data:
        assert data["strap_displacement"][1][:, 0].tolist() == [-2.0, -1.0, -1.0, -2.0]

    side["strap_log_ids"] = np.array([157525, 157526, 157527, 157528], dtype=np.int64)
    np.savez_compressed(cases_dir / "case_plate_render.npz", **side)
    with pytest.raises(ValueError, match="different nodes"):
        build_gui_case("case", cases_dir)