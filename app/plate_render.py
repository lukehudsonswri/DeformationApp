"""Rigid PPE rendering for the GUI (``app.viewer_app``).

Reads the ``<case>_plate_render.npz`` sidecar
(``febio.build_preliminary_case.build_case``) and reconstructs the seated
rigid plate's position at any point along its solved travel -- **without**
needing to log the plate's own nodes in the FEBio solve at all. A rigid
body's motion here is a pure translation (see ``febio/assemble_case.py``'s
``PlateDriveX/Y/Z`` rigid BCs): the plate's node-by-node "displacement"
isn't meaningful to log/animate individually, and is fully described by one
direction vector + one distance, both already known before the solve even
runs (from ``fitting.seat_plate``'s seating result and
``febio.loadcurve``'s derived load curve).

Motion model: at load-curve time ``t`` (the same units as the node log's
own ``*Time`` values -- FEBio's ``STATIC`` analysis time is the load-curve
parameter here, not physical time), the plate has moved
``push_direction * total_travel_mm * min(t, 1.0)`` from its seated rest
position -- linear from ``t=0`` to ``t=1``, then constant (matching
``LoadCurveSpec.extend = "CONSTANT"``, see ``febio/loadcurve.py``).

Sign derivation (matching the actual applied FEBio motion, not just a
plausible-looking guess): ``febio/assemble_case.py`` prescribes the rigid
body's displacement as ``dv = -push_direction`` (a unit-vector coefficient)
times the load curve's own value, which is itself ``-total_travel_mm`` at
``t=1`` (see ``febio/loadcurve.py``'s ``LoadCurveSpec.points``). The two
minus signs cancel: net displacement at ``t=1`` is
``+push_direction * total_travel_mm`` -- i.e. the plate moves INTO the
body along ``push_direction`` (its concave-normal / "into body" direction)
as ``t`` increases, not away from it. An earlier version of this function
had the sign backwards (moving the plate away from the body as load
increased); caught by comparing this derivation against
``febio/assemble_case.py``'s own comment deriving the same thing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PlateRenderData:
    nodes_rest: np.ndarray  # (N, 3) seated, pre-press (t=0) node positions
    boundary_faces: np.ndarray  # (M, 4) 0-based quad indices into nodes_rest
    push_direction: np.ndarray  # (3,) unit vector, points INTO the body
    total_travel_mm: float
    final_time: float


def load_plate_render(path: Path) -> PlateRenderData:
    with np.load(path) as data:
        return PlateRenderData(
            nodes_rest=data["nodes_rest"],
            boundary_faces=data["boundary_faces"],
            push_direction=data["push_direction"],
            total_travel_mm=float(data["total_travel_mm"]),
            final_time=float(data["final_time"]),
        )


def plate_points_at_time(plate: PlateRenderData, t: float) -> np.ndarray:
    """The plate's node positions at load-curve time ``t``."""
    load_fraction = float(np.clip(t, 0.0, 1.0))
    offset = plate.push_direction * (plate.total_travel_mm * load_fraction)
    return plate.nodes_rest + offset
