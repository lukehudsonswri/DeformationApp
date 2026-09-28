"""Tests for fitting.body_surface -- torso skin surface extraction with
vertex normals, used as the CPD/ICP registration target (AGENTS.md Goal 2).
"""
import numpy as np
import pytest

from config.hbm_models import discover_hbm_models
from config.sites import resolve_site
from fitting.body_surface import extract_skin_surface

HBM_ROOT = None  # resolved via conftest fixture below


@pytest.fixture(scope="module")
def f05_standing():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "HBM"
    models = discover_hbm_models(root)
    if "F05_Standing" not in models:
        pytest.skip("HBM/F05_Standing not built -- run config.hbm_normalize first")
    return models["F05_Standing"]


def test_torso_skin_surface_has_expected_node_count(f05_standing):
    site = resolve_site(f05_standing, "torso")
    surface = extract_skin_surface(f05_standing, site)
    # Regression value, independently cross-checked via a raw manual parse
    # of Elements.k/Nodes.k earlier in this project.
    assert surface.nodes.shape == (11412, 3)
    assert surface.faces.shape[1] == 3  # triangulated (from quad4 shells)


def test_torso_skin_normals_point_away_from_flesh(f05_standing):
    """The skin's outward normal should point away from the underlying
    flesh (solid) part -- a real, independently-computed sanity check
    (not just "normals exist"), since LS-DYNA shell winding conventions
    could in principle produce inward-facing normals if the source data
    were unusual.
    """
    from core.lsdyna.bridge import iter_wanted_cards

    site = resolve_site(f05_standing, "torso")
    surface = extract_skin_surface(f05_standing, site)

    solid_pids = set(site.solid_pids)
    referenced = set()
    for _kw, params in iter_wanted_cards(f05_standing.elements_path, ("*ELEMENT_SOLID",)):
        if params.get("pid") in solid_pids:
            for i in range(1, 9):
                v = params.get(f"n{i}")
                if v is not None:
                    referenced.add(v)

    coords = []
    for _kw, params in iter_wanted_cards(f05_standing.nodes_path, ("*NODE",)):
        if params["nid"] in referenced:
            coords.append((params["x"], params["y"], params["z"]))
    flesh_centroid = np.array(coords).mean(axis=0)

    to_flesh = flesh_centroid - surface.nodes
    to_flesh /= np.linalg.norm(to_flesh, axis=1, keepdims=True)
    dots = np.sum(surface.vertex_normals * to_flesh, axis=1)
    # Regression value: mean ~-0.70, ~98.7% of vertices negative.
    assert dots.mean() < -0.5
    assert (dots < 0).mean() > 0.9


def test_unknown_site_pids_raise():
    from config.sites import SiteResolution

    site = SiteResolution(site="torso", model_key="F05_Standing", solid_pids=(999999999,), shell_pids=(999999999,))
    from config.hbm_models import HbmModel
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "HBM"
    models = discover_hbm_models(root)
    if "F05_Standing" not in models:
        pytest.skip("HBM/F05_Standing not built")
    with pytest.raises(ValueError):
        extract_skin_surface(models["F05_Standing"], site)
