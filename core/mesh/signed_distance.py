"""Signed point-to-face distance against a triangulated surface.

AGENTS.md section 2.4: "Measure gaps point-to-face, never point-to-node."
At a 0.1mm target standoff, a point-to-*node* query on a 7mm mesh can be
wrong by tens of microns -- and it's exactly what made an existing case
(M50) look like 0.281mm clearance when it was actually 3.22mm embedded
(VERIFICATION.md section 5.2). Point-to-*face* (nearest point on the
nearest triangle, not nearest vertex) plus a sign from the face's own
outward normal is the measure that actually distinguishes penetration from
clearance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.mesh.base import SurfaceMesh


@dataclass
class SignedGapResult:
    """Signed point-to-face distances for a batch of query points against
    one target surface.

    Positive = clear (query point is on the outward side of its nearest
    triangle's face normal). Negative = penetrating.
    """

    signed_distance_mm: np.ndarray  # (N,)
    closest_points: np.ndarray  # (N, 3)
    triangle_ids: np.ndarray  # (N,)

    @property
    def min_mm(self) -> float:
        return float(self.signed_distance_mm.min())

    @property
    def max_mm(self) -> float:
        return float(self.signed_distance_mm.max())

    @property
    def mean_mm(self) -> float:
        return float(self.signed_distance_mm.mean())

    @property
    def n_penetrating(self) -> int:
        return int(np.count_nonzero(self.signed_distance_mm < 0.0))


def signed_gap_to_surface(query_points: np.ndarray, surface: SurfaceMesh) -> SignedGapResult:
    """Signed point-to-face distance from ``query_points`` to ``surface``.

    Uses ``trimesh``'s AABB-tree-accelerated point-to-triangle nearest
    point query (``mesh.nearest.on_surface``) -- a true point-to-*triangle*
    closest point (with barycentric clamping to the triangle's interior/
    edges/vertices as needed), not a nearest-vertex approximation. Sign
    comes from ``surface.face_normals`` at the matched triangle -- the same
    normals already validated in ``fitting.body_surface`` (outward, away
    from flesh; mean dot with the direction-to-flesh-centroid ~-0.70).
    """
    if surface.face_normals is None:
        surface.compute_normals()

    tm = surface.to_trimesh()
    closest, distances, triangle_ids = tm.nearest.on_surface(np.asarray(query_points, dtype=np.float64))

    normals = surface.face_normals[triangle_ids]
    signed = np.sum((np.asarray(query_points, dtype=np.float64) - closest) * normals, axis=1)

    return SignedGapResult(
        signed_distance_mm=signed,
        closest_points=closest,
        triangle_ids=triangle_ids,
    )
