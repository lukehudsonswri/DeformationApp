"""Mesh-aware standoff floor and closed-form standoff correction
(AGENTS.md section 2.4).

A target contact gap in the 1e-1 to 1e-3 mm range is the right *intent*,
but the achievable floor is set by the mesh, not by preference: a faceted
surface dips below the true curved surface by the sagitta of its elements,
so a requested gap smaller than that is not physically meaningful. See
AGENTS.md section 2.4's sagitta table (7mm skin edges -> ~0.041mm sagitta
-> ~0.082mm floor at a 2x-sagitta safety margin, on a ~150mm torso radius
of curvature).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from core.mesh.base import SurfaceMesh
from core.mesh.signed_distance import SignedGapResult, signed_gap_to_surface

# AGENTS.md section 2.4: measured torso radius of curvature used for the
# sagitta estimate. This is a property of human torso anatomy at the scale
# PPE seating cares about, not of any one mesh -- exposed as a parameter so
# a different body region could override it, not re-derived per mesh.
DEFAULT_RADIUS_OF_CURVATURE_MM = 150.0
DEFAULT_FLOOR_MULTIPLE = 2.0


@dataclass
class StandoffFloor:
    median_edge_length_mm: float
    sagitta_mm: float
    floor_mm: float


def _edge_lengths(surface: SurfaceMesh) -> np.ndarray:
    """All unique triangle edge lengths in ``surface``."""
    faces = surface.faces
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    p0 = surface.nodes[edges[:, 0]]
    p1 = surface.nodes[edges[:, 1]]
    return np.linalg.norm(p1 - p0, axis=1)


def compute_standoff_floor(
    surface: SurfaceMesh,
    radius_of_curvature_mm: float = DEFAULT_RADIUS_OF_CURVATURE_MM,
    floor_multiple: float = DEFAULT_FLOOR_MULTIPLE,
) -> StandoffFloor:
    """Smallest standoff that is geometrically meaningful on ``surface``.

    Sagitta of a chord of length ``L`` on a circle of radius ``R``:
    ``h ~= L^2 / (8R)`` for ``L << R``. Verified against AGENTS.md's own
    worked numbers: a 7mm median edge at R=150mm gives h=0.0408mm, matching
    the documented 0.041mm.
    """
    median_edge = float(np.median(_edge_lengths(surface)))
    sagitta = median_edge ** 2 / (8.0 * radius_of_curvature_mm)
    floor = floor_multiple * sagitta
    return StandoffFloor(median_edge_length_mm=median_edge, sagitta_mm=sagitta, floor_mm=floor)


def resolve_target_standoff(
    requested_standoff_mm: float,
    floor: StandoffFloor,
    progress=print,
) -> float:
    """Clamp ``requested_standoff_mm`` up to the mesh's floor if needed,
    logging (not silently accepting) when the request was unachievable.
    """
    if requested_standoff_mm >= floor.floor_mm:
        return requested_standoff_mm
    if progress:
        progress(
            f"requested standoff {requested_standoff_mm:.4f}mm is below the mesh-aware "
            f"floor {floor.floor_mm:.4f}mm (median edge {floor.median_edge_length_mm:.2f}mm, "
            f"sagitta {floor.sagitta_mm:.4f}mm) -- raising to the floor"
        )
    return floor.floor_mm


@dataclass
class StandoffCheck:
    """Result of ``check_standoff``: the measured gap and whether it is
    already inside the acceptable band (so no re-fit is needed).
    """

    gap: SignedGapResult
    min_ok_mm: float
    max_ok_mm: float
    is_acceptable: bool


def check_standoff(
    concave_points: np.ndarray,
    body_surface: SurfaceMesh,
    acceptable_range_mm: Tuple[float, float],
) -> StandoffCheck:
    """Decide whether a plate is already at a usable standoff from the body.

    Uses the same measure ``seat_plate`` drives to its target: the minimum
    signed point-to-face gap of the plate's body-facing surface.

    Args:
        concave_points: (N, 3) body-facing plate surface points in mm.
        body_surface: skin ``SurfaceMesh`` with outward-facing normals.
        acceptable_range_mm: (lo, hi) inclusive band for the minimum signed
            gap -- ``lo`` bounds penetration, ``hi`` bounds how far off the
            skin the plate may sit.

    Returns:
        ``StandoffCheck`` carrying the full gap result and the verdict.
    """
    lo, hi = acceptable_range_mm
    gap = signed_gap_to_surface(concave_points, body_surface)
    return StandoffCheck(gap=gap, min_ok_mm=lo, max_ok_mm=hi, is_acceptable=lo <= gap.min_mm <= hi)
