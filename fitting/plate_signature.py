"""Shape-signature detection for the rigid armored-plate PPE.

Per AGENTS.md Goal 2 / section 2.2: before CPD registration can seat the
plate against the torso, we need to know (a) which of its two large faces
is the concave, body-facing surface (the convex face "caves out", the
concave face "caves in" and is supposed to sit up against the human body
model), and (b) which end is the "top" -- a shorter trapezoid edge -- versus
the "bottom", a straight, full-width edge.

This module is deliberately specific to the single rigid-plate PPE in
scope right now (no generic multi-PPE signature framework). If/when more
PPE are added, each will need its own signature routine written against
its own known geometric quirks, exactly as AGENTS.md describes.

Method
------
1. PCA on the plate's surface node cloud gives three orthogonal axes. The
   plate is a thin curved shell, so the *smallest*-variance axis is the
   through-thickness direction; the other two span the plate's own local
   "in-plane" (u, v) frame.
2. Every outward-facing boundary triangle is bucketed into "+thickness
   side", "-thickness side", or "edge/rim" by the dot product of its
   normal with the thickness axis. The two side buckets are the plate's
   two large faces; the rim bucket (thin edge band) is discarded.
3. For each side, curvature sign is read from how face-centroid height
   above the local best-fit plane (measured along that side's own average
   outward normal) varies with in-plane radial distance from the side's
   centroid:
       cov(r^2, h) > 0  -> centre sits *behind* the rim along +normal
                           (a bowl/dish -- concave, "caves in")
       cov(r^2, h) < 0  -> centre sits *ahead of* the rim along +normal
                           (a dome -- convex, "caves out")
   (Worked example, see module docstring test: a bowl opening toward its
   own outward normal has its rim higher than its centre along that
   normal, i.e. height *increases* with radius -> positive covariance.)
4. Top/bottom is resolved on the concave face's 2D (u, v) footprint: try
   both in-plane axes as the candidate "up" direction, and compare the
   in-plane width (extent along the *other* axis) of a narrow band at
   each end (~1/10th of the axis's range). The axis with the larger
   relative width difference is the true trapezoid taper axis; the
   narrower end is "top".
"""
from dataclasses import dataclass

import numpy as np

from core.mesh.base import VolumeMesh


@dataclass
class PlateOrientation:
    """Result of :func:`compute_plate_orientation`.

    All vectors are unit-length, in the plate mesh's own (untransformed)
    coordinate frame.
    """

    plate_centroid: np.ndarray          # (3,) centroid of ALL volume nodes
    thickness_axis: np.ndarray          # (3,) unit vector, smallest-variance PCA axis
    concave_normal: np.ndarray          # (3,) unit outward normal of the concave face
                                         # (points from the plate toward where the
                                         # body is supposed to sit)
    convex_normal: np.ndarray           # (3,) unit outward normal of the convex face
    concave_face_node_ids: np.ndarray   # node ids of the concave face's surface points
    convex_face_node_ids: np.ndarray    # node ids of the convex face's surface points
    up_axis: np.ndarray                 # (3,) unit vector, bottom -> top (trapezoid taper)
    concavity_score: float              # cov(r^2, h) of the concave face (diagnostic, > 0)
    taper_ratio: float                  # narrow-end width / wide-end width on up_axis (diagnostic, < 1)


def _side_curvature_score(centroids: np.ndarray, normal: np.ndarray, in_plane_axes: np.ndarray) -> float:
    """cov(r^2, h) for one face-side's triangle centroids.

    ``h`` is height along ``normal`` relative to the side's own centroid;
    ``r`` is the in-plane radial distance from that same centroid.
    """
    c0 = centroids.mean(axis=0)
    rel = centroids - c0
    h = rel @ normal
    uv = rel @ in_plane_axes.T
    r2 = (uv ** 2).sum(axis=1)
    # cov(r2, h) via the standard two-variable covariance formula.
    return float(np.mean((r2 - r2.mean()) * (h - h.mean())))


def _taper_axis_and_up(points_uv: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, float]:
    """Given an (N, 2) in-plane point cloud (columns = the two candidate
    in-plane axes), decide which axis is the up/down (trapezoid taper)
    axis and which end is "top" (narrower). Returns (sign_vector, ratio)
    where sign_vector is a length-2 array picking out the up axis with the
    correct sign (e.g. [0, 1] or [0, -1]), and ratio = narrow/wide width
    (< 1, the smaller the more trapezoidal).

    Uses narrow (~1/``n_bins``) end bands rather than broad thirds: a
    trapezoid taper is often concentrated in just the last ~10% of an
    edge's length, and taking ``ptp`` over a broad third would let the
    still-full-width interior of that third mask the true taper at its
    very tip.
    """
    best = None
    for axis_idx in (0, 1):
        other_idx = 1 - axis_idx
        v = points_uv[:, axis_idx]
        lo, hi = v.min(), v.max()
        band = (hi - lo) / n_bins
        if band <= 0:
            continue
        near_mask = v <= lo + band
        far_mask = v >= hi - band
        if near_mask.sum() < 3 or far_mask.sum() < 3:
            continue
        width_near = np.ptp(points_uv[near_mask, other_idx])
        width_far = np.ptp(points_uv[far_mask, other_idx])
        wide, narrow = max(width_near, width_far), min(width_near, width_far)
        if wide <= 0:
            continue
        rel_diff = (wide - narrow) / wide
        sign = +1.0 if width_far < width_near else -1.0
        candidate = (rel_diff, axis_idx, sign, narrow / wide)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        # Degenerate/near-rectangular fallback: pick axis 1, arbitrary sign.
        return np.array([0.0, 1.0]), 1.0
    _, axis_idx, sign, ratio = best
    sign_vector = np.zeros(2)
    sign_vector[axis_idx] = sign
    return sign_vector, ratio


def compute_plate_orientation(mesh: VolumeMesh, side_normal_dot_threshold: float = 0.7) -> PlateOrientation:
    """Detect the concave (body-facing) surface and top/bottom axis of the
    rigid armored plate.

    Parameters
    ----------
    mesh:
        The plate's volume mesh (as loaded via ``config.ppe.get_ppe(...).load()``).
    side_normal_dot_threshold:
        Boundary faces whose normal has ``|dot(normal, thickness_axis)|``
        below this are treated as rim/edge faces and excluded from the
        concave/convex classification.
    """
    surface = mesh.extract_surface()
    if surface.num_faces == 0:
        raise ValueError("plate mesh has no extractable boundary surface")
    if surface.face_normals is None:
        surface.compute_normals()

    plate_centroid = mesh.nodes.mean(axis=0)

    # PCA on the surface node cloud: smallest-variance axis = thickness.
    surf_nodes = surface.nodes[np.unique(surface.faces)]
    centered = surf_nodes - surf_nodes.mean(axis=0)
    cov = centered.T @ centered / len(centered)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
    thickness_axis = eigvecs[:, 0]
    in_plane_axes = eigvecs[:, 1:].T  # (2, 3), rows = the two in-plane axes

    face_centroids = surface.nodes[surface.faces].mean(axis=1)
    dots = surface.face_normals @ thickness_axis
    pos_mask = dots >= side_normal_dot_threshold
    neg_mask = dots <= -side_normal_dot_threshold
    if pos_mask.sum() < 3 or neg_mask.sum() < 3:
        raise ValueError(
            "could not separate plate surface into two large faces -- check "
            "side_normal_dot_threshold or the input mesh"
        )

    score_pos = _side_curvature_score(face_centroids[pos_mask], thickness_axis, in_plane_axes)
    score_neg = _side_curvature_score(face_centroids[neg_mask], -thickness_axis, in_plane_axes)

    if score_pos > score_neg:
        concave_mask, convex_mask = pos_mask, neg_mask
        concave_normal, convex_normal = thickness_axis, -thickness_axis
        concavity_score = score_pos
    else:
        concave_mask, convex_mask = neg_mask, pos_mask
        concave_normal, convex_normal = -thickness_axis, thickness_axis
        concavity_score = score_neg

    concave_face_node_ids = np.unique(surface.faces[concave_mask])
    convex_face_node_ids = np.unique(surface.faces[convex_mask])

    concave_points = surface.nodes[concave_face_node_ids]
    uv = (concave_points - concave_points.mean(axis=0)) @ in_plane_axes.T
    sign_vector, taper_ratio = _taper_axis_and_up(uv)
    up_axis = sign_vector @ in_plane_axes
    up_axis = up_axis / np.linalg.norm(up_axis)

    return PlateOrientation(
        plate_centroid=plate_centroid,
        thickness_axis=thickness_axis,
        concave_normal=concave_normal,
        convex_normal=convex_normal,
        concave_face_node_ids=concave_face_node_ids,
        convex_face_node_ids=convex_face_node_ids,
        up_axis=up_axis,
        concavity_score=concavity_score,
        taper_ratio=taper_ratio,
    )
