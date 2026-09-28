"""Extract a body region's skin surface from a normalized HBM model as a
``SurfaceMesh`` with computed vertex normals, for use as the CPD/PCA-prealign
registration target (AGENTS.md Goal 2 / section 2.2).

Reads ``HBM/<model>/Elements.k`` and ``Nodes.k`` via the client's own
``KeywordProcessor`` bridge (``core.lsdyna.bridge.iter_wanted_cards``), not
``core.mesh.lsdyna.load_lsdyna`` -- per that module's own docstring, its
general-purpose parser is for PPE `.k` files; HBM decks need the client's
fixed-width-aware engine once node/element ids run past 7 digits (F05_Standing
has ~2.4M elements). Two lightweight streaming passes, not a full ``VolumeMesh``
build: (1) collect the site's shell element connectivity (and the referenced
node ids) from Elements.k, (2) collect only those referenced nodes' coordinates
from Nodes.k. This avoids ever materializing the model's other ~2.4M solid
elements or the ~97% of nodes outside the site.
"""
from __future__ import annotations

import numpy as np

from config.hbm_models import HbmModel
from config.sites import SiteResolution
from core.lsdyna.bridge import iter_wanted_cards
from core.mesh.base import ElementType, SurfaceMesh


def extract_skin_surface(model: HbmModel, site: SiteResolution) -> SurfaceMesh:
    """Build a triangulated ``SurfaceMesh`` (with vertex normals) of the
    site's skin (shell) part(s) in ``model``.

    Only the skin is extracted -- the flesh (solid) is not part of the
    contact surface a PPE is seated against, and pulling its ~35k hex8
    elements in as well would be wasted work for this use case.
    """
    shell_pids = set(site.shell_pids)
    quads: list[list[int]] = []  # node ids (original LS-DYNA ids), 4 per element
    referenced_ids: set = set()
    for _keyword, params in iter_wanted_cards(model.elements_path, ("*ELEMENT_SHELL",)):
        if "eid" not in params or params.get("pid") not in shell_pids:
            continue
        n = [params.get(f"n{i}") for i in range(1, 5)]
        if any(v is None for v in n):
            continue
        quads.append(n)
        referenced_ids.update(n)

    if not quads:
        raise ValueError(
            f"no *ELEMENT_SHELL cards found for site '{site.site}' pids {site.shell_pids} "
            f"in {model.elements_path}"
        )

    coords: dict = {}
    for _keyword, params in iter_wanted_cards(model.nodes_path, ("*NODE",)):
        nid = params.get("nid")
        if nid in referenced_ids:
            coords[nid] = (params["x"], params["y"], params["z"])

    missing = referenced_ids - coords.keys()
    if missing:
        raise ValueError(
            f"{len(missing)} node id(s) referenced by site '{site.site}' shell elements "
            f"are missing from {model.nodes_path} (e.g. {sorted(missing)[:5]})"
        )

    id_to_index = {nid: i for i, nid in enumerate(coords)}
    nodes = np.array([coords[nid] for nid in coords], dtype=np.float64)

    faces = []
    for quad in quads:
        idx = [id_to_index[nid] for nid in quad]
        faces.append([idx[0], idx[1], idx[2]])
        faces.append([idx[0], idx[2], idx[3]])
    faces_arr = np.array(faces, dtype=np.int64)

    surface = SurfaceMesh(
        nodes=nodes,
        faces=faces_arr,
        face_type=ElementType.TRI3,
        name=f"{model.key}_{site.site}_skin",
    )
    surface.compute_normals()
    return surface
