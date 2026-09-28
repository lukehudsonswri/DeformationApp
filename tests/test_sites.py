"""Regression tests for config.sites -- torso-only body-site pid
resolution (AGENTS.md section 1.2; head is explicitly out of scope).
"""
from pathlib import Path

import pytest

from config.hbm_models import HbmModel, discover_hbm_models
from config.sites import resolve_site, known_sites

_REAL_HBM_ROOT = Path(r"C:\Users\lhudson\Desktop\Projects\GUARDS\DeformationApp-agent\HBM")


def _fake_model(key="F05_Standing"):
    return HbmModel(key=key, family="F05", posture="Standing", folder=Path("."), nodes_path=Path("."), elements_path=Path("."))


def test_known_sites_is_torso_only():
    assert known_sites() == ("torso",)


def test_resolve_torso_returns_expected_pids():
    result = resolve_site(_fake_model(), "torso")
    assert result.solid_pids == (2000500,)
    assert result.shell_pids == (2000501,)
    assert result.site == "torso"
    assert result.model_key == "F05_Standing"


def test_resolve_unknown_site_raises():
    with pytest.raises(KeyError):
        resolve_site(_fake_model(), "head")


@pytest.mark.skipif(
    not (_REAL_HBM_ROOT / "F05_Standing" / "Elements.k").is_file(),
    reason="F05_Standing has not been built yet",
)
def test_resolved_torso_pids_match_real_element_counts():
    """Cross-check resolve_site's pids against the ACTUAL built
    F05_Standing model: the resolved Thorax_Flesh/Thorax_Skin pids must
    match real, present elements -- not just be plausible-looking numbers.
    Counts (34,980 solid / 11,158 shell) were independently verified via
    the client's authoritative parser (see tests/test_hbm_models.py).
    """
    models = discover_hbm_models(_REAL_HBM_ROOT)
    model = models["F05_Standing"]
    result = resolve_site(model, "torso")

    n_solid = 0
    n_shell = 0
    section = None
    with open(model.elements_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("*"):
                section = line
                continue
            pid = int(line.split(",")[1])
            if section == "*ELEMENT_SOLID" and pid in result.solid_pids:
                n_solid += 1
            elif section == "*ELEMENT_SHELL" and pid in result.shell_pids:
                n_shell += 1

    assert n_solid == 34980
    assert n_shell == 11158
