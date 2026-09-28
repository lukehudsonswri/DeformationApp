"""Regression tests for config.ppe -- the PPE registry (AGENTS.md section
1.4). Rigid armored plate only, per explicit instruction.
"""
import pytest

from config.ppe import get_ppe, known_ppe


def test_known_ppe_is_armored_plate_only():
    assert known_ppe() == ("armored_plate",)


def test_get_unknown_ppe_raises():
    with pytest.raises(KeyError):
        get_ppe("not_a_real_ppe")


def test_armored_plate_spec_is_rigid_and_path_exists():
    spec = get_ppe("armored_plate")
    assert spec.is_rigid is True
    assert spec.mesh_path.is_file(), f"expected PPE mesh at {spec.mesh_path}"
    assert spec.mesh_path.suffix.lower() == ".inp"


@pytest.mark.skipif(not get_ppe("armored_plate").mesh_path.is_file(), reason="PPE/plate.inp not present")
def test_armored_plate_loads_expected_geometry():
    """Cross-check against counts already independently verified earlier
    in this project (loaded via core.mesh directly, and via the smoke-test
    module): 619,098 nodes, 562,037 hex8 solid elements.
    """
    from core.mesh.base import ElementType

    mesh = get_ppe("armored_plate").load()
    assert mesh.nodes.shape[0] == 619098
    assert mesh.element_groups[ElementType.HEX8].shape[0] == 562037


def test_stl_ppe_raises_not_implemented(tmp_path):
    """STL tetrahedralization is designed but not implemented -- must fail
    loudly and clearly, not silently mishandle an STL PPE.
    """
    from config.ppe import PpeSpec

    stl_path = tmp_path / "fake.stl"
    stl_path.write_text("not a real stl, never read")
    spec = PpeSpec(key="fake", display_name="Fake", mesh_path=stl_path, is_rigid=True)
    with pytest.raises(NotImplementedError):
        spec.load()
