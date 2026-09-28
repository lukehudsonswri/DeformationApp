"""Point-to-plane ICP -- rigid surface-to-surface refinement.

Built for, and empirically validated against, a failure mode discovered
while implementing PPE seating (AGENTS.md Goal 2 / section 2.2): registering
a near-flat rigid plate's concave face against a *much larger* local patch
of torso skin is a classic ICP "aperture problem". Both ``pycpd``'s
probabilistic rigid CPD (``core.registration.rigid_cpd``) and plain
point-to-point trimmed ICP were tested directly against real data (F05
torso + the shipped armored plate, already seeded to a good ~13mm mean
nearest-neighbor distance by ``fitting.plate_signature`` + a signature-based
seed rotation) and both **diverged**: CPD's fitted pose landed at 60-70mm
mean distance, and point-to-point ICP oscillated and settled around 45mm --
both *worse* than doing nothing. The plate's concave surface is nearly flat
(23mm of out-of-plane deviation over a 320x240mm footprint) while a
correspondingly-sized local patch of the torso's skin has much more genuine
curvature (up to ~230mm of relief once cropped to a comparable footprint);
point-to-point correspondence has no way to distinguish "the true matching
patch" from "a nearby, differently-curved patch that also looks locally
plausible", so free rigid search wanders.

Point-to-plane ICP fixes this because its cost term, ``(p - q)*n``, imposes
*zero* penalty for a correspondence that's off purely tangentially (sliding
along the target's local tangent plane) -- so an ambiguous tangential
degree of freedom is left where it started (at whatever the seed placed it)
instead of being actively (and, per the above, wrongly) pulled somewhere
else. Verified: from the same 13mm seed, converges monotonically to 5.3mm
in under 10 iterations and stays there (vs. 45-70mm for the alternatives).

This does not replace ``core.registration.rigid_cpd`` in the codebase --
CPD may still be the right tool for PPE shapes with enough real 3D
structure (not near-flat) that point-to-point/probabilistic correspondence
has a genuine, unambiguous shape signal to lock onto. It replaces CPD
specifically for the rigid armored plate's local seating refinement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class ICPConfig:
    max_iterations: int = 30
    trim_fraction: float = 0.1  # discard the worst-matched fraction of correspondences each iteration
    convergence_rotation_deg: float = 0.01
    convergence_translation_mm: float = 0.001


@dataclass
class ICPResult:
    """Composed rigid transform across all ICP iterations.

    Row-vector convention throughout this project: ``transformed = points @
    rotation + translation``.
    """

    transformed_points: np.ndarray
    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)
    iterations: int
    converged: bool
    final_mean_dist_mm: float

    def apply(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64) @ self.rotation + self.translation


def _rodrigues(r: np.ndarray) -> np.ndarray:
    """Standard (column-vector) rotation matrix for a small rotation vector ``r``."""
    theta = np.linalg.norm(r)
    if theta < 1e-14:
        return np.eye(3)
    k = r / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def point_to_plane_icp(
    source: np.ndarray,
    target_points: np.ndarray,
    target_normals: np.ndarray,
    config: Optional[ICPConfig] = None,
) -> ICPResult:
    """Rigidly register ``source`` onto ``target_points`` (with per-point
    outward ``target_normals``) by minimizing point-to-plane distance.

    Each iteration: find nearest-neighbor correspondences, drop the worst
    ``trim_fraction`` (robustness against source points with no genuine
    match nearby -- e.g. the plate's rim), solve the linearized
    small-rotation normal-equations system for an incremental rigid step,
    and compose it into the running total transform.
    """
    cfg = config or ICPConfig()
    source = np.asarray(source, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    target_normals = np.asarray(target_normals, dtype=np.float64)

    tree = cKDTree(target_points)
    current = source.copy()
    M_total = np.eye(3)
    t_total = np.zeros(3)
    converged = False
    iterations_run = 0

    for iteration in range(cfg.max_iterations):
        iterations_run = iteration + 1
        dist, idx = tree.query(current)
        keep_thresh = np.percentile(dist, 100.0 * (1.0 - cfg.trim_fraction))
        keep = dist <= keep_thresh

        P = current[keep]
        Q = target_points[idx[keep]]
        N = target_normals[idx[keep]]

        A = np.empty((len(P), 6))
        A[:, 0:3] = np.cross(P, N)
        A[:, 3:6] = N
        b = np.sum((Q - P) * N, axis=1)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        r, t_step = x[0:3], x[3:6]

        R_step = _rodrigues(r)  # column-vector rotation
        M_step = R_step.T  # row-vector-convention equivalent

        current = current @ M_step + t_step
        t_total = t_total @ M_step + t_step
        M_total = M_total @ M_step

        rot_deg = np.degrees(np.linalg.norm(r))
        trans_mm = float(np.linalg.norm(t_step))
        if rot_deg < cfg.convergence_rotation_deg and trans_mm < cfg.convergence_translation_mm:
            converged = True
            break

    final_dist, _ = tree.query(current)
    return ICPResult(
        transformed_points=current,
        rotation=M_total,
        translation=t_total,
        iterations=iterations_run,
        converged=converged,
        final_mean_dist_mm=float(final_dist.mean()),
    )
