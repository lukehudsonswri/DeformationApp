"""Regression tests for remesh.febio_smoke_test -- the ground-truth mesh
validity check (AGENTS.md section 3.3 / VERIFICATION.md section 2.1):
"a simple FEBio problem setup that must start without errors".

These tests need a working febio4 executable (see config/febio_solver.py)
and the F05 case data (real, large, gitignored -- see
tests/test_core_repair_smoke.py for the same data-availability pattern).
Both are skipped, not failed, when unavailable.
"""
from pathlib import Path

import numpy as np
import pytest

from config.febio_solver import find_febio_executable
from core.mesh import load_mesh
from core.mesh.base import ElementType, VolumeMesh
from remesh.febio_smoke_test import run_febio_smoke_test

_F05_COMBINED_K = Path(
    r"C:\Users\lhudson\Desktop\Projects\GUARDS\DeformationApp"
    r"\cases\F05_Torso_ArmoredPlate\torso_F05_standing_full.k"
)

pytestmark = [
    pytest.mark.skipif(find_febio_executable() is None, reason="febio4 executable not found"),
    pytest.mark.skipif(not _F05_COMBINED_K.exists(), reason="F05 case data not present on this machine"),
]


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("febio_smoke")


def test_clean_combined_mesh_passes(workdir):
    """The real F05 combined flesh(hex8)+skin(quad4) mesh, independently
    confirmed clean by core.repair (zero inverted, zero locally-tangled
    elements across all 34,980 solid + 11,158 shell elements), must pass.

    This is also a regression test for a real bug found while building
    this module: with ``shell_normal_nodal=1``, FEBio reported a spurious
    "Negative jacobian ... during domain initialization" for a solid
    element that has nothing wrong with it by any measure (see
    febio_smoke_test.py's module docstring for the full investigation).
    If this test starts failing again, check that setting first.
    """
    mesh = load_mesh(str(_F05_COMBINED_K))
    result = run_febio_smoke_test(mesh, workdir, name="clean_f05")
    assert result.passed, result.message


def test_inverted_solid_element_is_rejected(workdir):
    """A deliberately inverted hex8 element (two nodes transposed) must be
    caught -- this is the actual point of the smoke test.
    """
    mesh = load_mesh(str(_F05_COMBINED_K))
    hex_arr = mesh.element_groups[ElementType.HEX8].copy()
    hex_arr[0, [0, 1]] = hex_arr[0, [1, 0]]  # swap n1, n2 -> inverted
    mesh.element_groups[ElementType.HEX8] = hex_arr

    result = run_febio_smoke_test(mesh, workdir, name="corrupted_f05")
    assert not result.passed
    assert "jacobian" in result.message.lower()


def test_solid_only_subset_passes():
    """A small, hand-verified-clean subset (isolation regression case for
    the shell-interaction bug above): pure hex8, no shells at all.
    """
    mesh = load_mesh(str(_F05_COMBINED_K))
    hex_arr = mesh.element_groups[ElementType.HEX8]
    solid_only = VolumeMesh(nodes=mesh.nodes, element_groups={ElementType.HEX8: hex_arr})

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        result = run_febio_smoke_test(solid_only, Path(td), name="solid_only")
        assert result.passed, result.message
