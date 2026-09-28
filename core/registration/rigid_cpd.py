"""
CPU-only rigid Coherent Point Drift (CPD) registration.

Adapted from ``fe_personalization/registration/cpd.py``'s
``GPUCPDRegistration``, but reimplemented against ``pycpd.RigidRegistration``
directly instead of ``torchcpd``, per the CPU-only decision recorded in
AGENTS.md section 5 (Q6): the original ``GPUCPDRegistration.register()``
imports ``torch`` unconditionally at the top of the method -- even for the
rigid-only stage this project uses -- so it could not be vendored as-is
under that decision. This module is a fresh, minimal implementation, not a
verbatim copy.

Kept from cpd.py's design:
  * normalize-to-zero-mean/unit-RMS-distance before registration, and
    denormalize after -- keeps pycpd's variance-based EM well-conditioned
    regardless of the mesh's absolute scale (mm);
  * voxel/random downsampling for point budgets, with the fitted transform
    re-applied exactly (not RBF-interpolated -- see below) to the full
    point set.

What's different, and why:
  * ``pycpd.RigidRegistration`` is the sole backend (no torch/torchcpd).
  * ``pycpd.RigidRegistration`` always re-estimates scale every iteration
    and has **no constructor flag to disable it**. AGENTS.md section 2.2
    requires ``allow_scale=False`` always for PPE ("PPE is physical
    hardware and must not be resized"), so ``_lock_scale`` monkey-patches
    the registration instance's ``update_transform`` to force ``s = 1``
    every iteration -- the same technique cpd.py itself uses elsewhere to
    patch a torchcpd variance bug, applied here for a different reason.
  * The composed rigid transform (R, t, s) is recovered algebraically in
    ORIGINAL (mm) coordinates and applied as an exact matrix transform to
    the full point set -- never via RBF/TPS interpolation. cpd.py's
    ``_interpolate_to_full`` is appropriate for its deformable morphing use
    case, but an RBF fit through a rigid registration's sampled points is
    only *approximately* rigid; AGENTS.md section 2.3 is explicit that a
    rigid PPE must receive an exact rigid transform, not an RBF
    approximation of one. See ``_compose_original_frame_transform`` for the
    derivation.
  * When ``allow_scale=False``, source and target are normalized by the
    SAME scale factor (the source's own RMS distance), not independent
    per-cloud scales. This matters: independent per-cloud normalization
    means locking CPD's internal scale at 1.0 only guarantees no resizing
    in a rescaled frame where source and target may already differ in
    size -- it does **not** guarantee ``s = 1`` in real mm once denormalized.
    A shared normalization scale makes "locked at 1 internally" and
    "s = 1 in original coordinates" the same statement. See the module's
    self-test in ``tests/`` for a numeric check of this.

See AGENTS.md "Vendoring policy" and section 2.2 / section 5 Q6.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _downsample_points_with_idx(
    points: np.ndarray,
    n: int,
    method: str = "voxel",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Downsample points and return (downsampled_points, indices).

    method:
      - "random": uniform random sampling
      - "voxel": voxel grid sampling for better spatial coverage

    Ported near-verbatim from cpd.py's module-level helper of the same
    name/behaviour (pure numpy, no torch involved there either).
    """
    if n <= 0 or points.shape[0] <= n:
        return points, None

    if rng is None:
        rng = np.random.default_rng()

    method = (method or "random").lower()

    if method == "random":
        idx = rng.choice(points.shape[0], n, replace=False)
        return points[idx], idx

    if method == "voxel":
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        extents = maxs - mins
        extents = np.where(extents < 1e-9, 1e-9, extents)
        volume = float(np.prod(extents))

        voxel_size = (volume / n) ** (1.0 / 3.0) if volume > 0 else 0.0
        if not np.isfinite(voxel_size) or voxel_size <= 0:
            voxel_size = extents.max() / (n ** (1.0 / 3.0) + 1e-9)
            if voxel_size <= 0:
                voxel_size = 1.0

        coords = np.floor((points - mins) / voxel_size).astype(np.int64)
        voxel_map = {}
        for i, c in enumerate(coords):
            key = (int(c[0]), int(c[1]), int(c[2]))
            if key not in voxel_map:
                voxel_map[key] = i

        idx = np.fromiter(voxel_map.values(), dtype=np.int64)

        if idx.size > n:
            idx = rng.choice(idx, n, replace=False)
        elif idx.size < n:
            remaining = np.setdiff1d(np.arange(points.shape[0]), idx, assume_unique=False)
            if remaining.size > 0:
                extra = rng.choice(remaining, min(n - idx.size, remaining.size), replace=False)
                idx = np.concatenate([idx, extra])

        return points[idx], idx

    logger.warning("Unknown downsample method '%s', using random sampling.", method)
    idx = rng.choice(points.shape[0], n, replace=False)
    return points[idx], idx


@dataclass
class CPDConfig:
    """Configuration for rigid CPD registration.

    Only the rigid stage is implemented -- affine/deformable/projection
    (present in cpd.py's GPUCPDRegistration) are out of scope here: PPE
    fitting in this project is rigid-only by design (AGENTS.md section 2.3).
    """

    rigid_iters: int = 60
    tolerance: float = 1e-6
    outlier_weight: float = 0.05
    allow_scale: bool = False  # AGENTS.md 2.2: always False for PPE

    downsample: int = 6000  # 0 = no limit
    downsample_method: str = "voxel"
    random_seed: Optional[int] = 42

    normalize: bool = True


@dataclass
class CPDResult:
    """Result of rigid CPD registration.

    ``rotation``/``translation``/``scale`` are expressed in the SAME
    (original, e.g. mm) coordinate frame the inputs were given in, so they
    can be re-applied to any other point set from the same source object
    (e.g. fit on a PPE's surface, then applied to its full volume mesh) via
    ``points @ (scale * rotation) + translation``.
    """

    transformed_points: np.ndarray  # full-resolution transformed source points
    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)
    scale: float
    iterations: int = 0
    converged: bool = False

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply this fitted rigid transform to an arbitrary point set."""
        return np.asarray(points, dtype=np.float64) @ (self.scale * self.rotation) + self.translation


def _normalize(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Center at zero mean and rescale to unit RMS distance."""
    center = points.mean(axis=0)
    centered = points - center
    rms = float(np.sqrt((centered ** 2).sum(axis=1).mean()))
    if rms < 1e-10:
        rms = 1.0
    return centered / rms, center, rms


def _lock_scale(reg) -> None:
    """Monkey-patch a pycpd RigidRegistration instance so scale stays 1.0.

    pycpd's ``update_transform`` always re-estimates ``self.s`` (there is no
    constructor argument to disable it). This recomputes the translation
    with scale locked at 1, using pycpd's own formula
    (``t = muX - s * R^T @ muY``) with ``s`` forced to 1 -- not just
    discarding the fitted scale afterwards, which would leave the EM
    correspondences computed against an unlocked scale during iteration.
    """
    original_update_transform = reg.update_transform

    def _locked_update_transform():
        original_update_transform()
        P, X, Y, Np = reg.P, reg.X, reg.Y, reg.Np
        muX = np.divide(np.sum(np.dot(P, X), axis=0), Np)
        muY = np.divide(np.sum(np.dot(np.transpose(P), Y), axis=0), Np)
        reg.s = 1.0
        reg.t = np.transpose(muX) - np.dot(np.transpose(reg.R), np.transpose(muY))

    reg.update_transform = _locked_update_transform


def _compose_original_frame_transform(
    s_hat: float,
    R_hat: np.ndarray,
    t_hat: np.ndarray,
    src_center: np.ndarray,
    src_scale: float,
    tgt_center: np.ndarray,
    tgt_scale: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Un-normalize a CPD-fitted (s, R, t) into original (mm) coordinates.

    Registration ran on ``Y_norm = (Y - src_center) / src_scale`` against
    ``X_norm = (X - tgt_center) / tgt_scale`` and fitted
    ``TY = s_hat * (Y_norm @ R_hat) + t_hat``. Substituting the
    normalizations and collecting terms gives a single equivalent transform
    in ORIGINAL coordinates:

        TY_original = Y @ (s_full * R_hat) + t_full

        s_full = s_hat * tgt_scale / src_scale
        t_full = t_hat * tgt_scale + tgt_center - src_center @ (s_full * R_hat)

    This is the exact rigid map -- applying it to any point set from the
    source object (not just the ones used to fit it) is mathematically
    equivalent to normalizing, transforming, and denormalizing that point
    set the same way registration did.
    """
    s_full = float(s_hat) * tgt_scale / src_scale
    t_full = t_hat * tgt_scale + tgt_center - src_center @ (s_full * R_hat)
    return R_hat, np.asarray(t_full, dtype=np.float64).reshape(3), s_full


class RigidCPD:
    """CPU-only rigid CPD registration (pycpd backend).

    Registers a source point cloud onto a target point cloud via rigid
    (rotation + translation, optionally uniform scale) Coherent Point
    Drift. Deliberately does not implement the affine/deformable/projection
    stages of cpd.py's GPUCPDRegistration -- this project's PPE fitting is
    rigid-only (AGENTS.md section 2.3).
    """

    def __init__(self, config: Optional[CPDConfig] = None):
        self.config = config or CPDConfig()

    def register(self, source: np.ndarray, target: np.ndarray, callback=None) -> CPDResult:
        """Register ``source`` (N, 3) onto ``target`` (M, 3).

        Returns a ``CPDResult`` with the FULL-resolution transformed source
        points (even when registration itself ran on a downsampled subset)
        and the fitted (rotation, translation, scale) in original
        coordinates, so the same rigid transform can be applied to points
        that were never part of the input (e.g. the PPE's full volume mesh,
        when ``source`` was only its surface) via ``CPDResult.apply``.
        """
        from pycpd import RigidRegistration

        cfg = self.config
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        rng = np.random.default_rng(cfg.random_seed) if cfg.random_seed is not None else None

        source_ds, source_ds_idx = _downsample_points_with_idx(
            source, cfg.downsample, method=cfg.downsample_method, rng=rng
        )
        target_ds, _ = _downsample_points_with_idx(
            target, cfg.downsample, method=cfg.downsample_method, rng=rng
        )

        if cfg.normalize:
            source_norm, src_center, src_scale = _normalize(source_ds)
            if cfg.allow_scale:
                target_norm, tgt_center, tgt_scale = _normalize(target_ds)
            else:
                # Shared scale (see module docstring): locking CPD's
                # internal s at 1.0 then genuinely means s=1 in original
                # mm, not just in some frame where source and target were
                # independently rescaled to unit RMS first.
                tgt_center = target_ds.mean(axis=0)
                tgt_scale = src_scale
                target_norm = (target_ds - tgt_center) / tgt_scale
        else:
            source_norm, src_center, src_scale = source_ds, np.zeros(3), 1.0
            target_norm, tgt_center, tgt_scale = target_ds, np.zeros(3), 1.0

        reg = RigidRegistration(
            X=target_norm,
            Y=source_norm,
            max_iterations=cfg.rigid_iters,
            tolerance=cfg.tolerance,
            w=cfg.outlier_weight,
        )
        if not cfg.allow_scale:
            _lock_scale(reg)

        if callback is not None:
            TY, (s_hat, R_hat, t_hat) = reg.register(callback=callback)
        else:
            TY, (s_hat, R_hat, t_hat) = reg.register()

        R_full, t_full, s_full = _compose_original_frame_transform(
            s_hat, R_hat, t_hat, src_center, src_scale, tgt_center, tgt_scale
        )

        if not cfg.allow_scale and abs(s_full - 1.0) > 1e-8:
            # Should be unreachable given the shared-scale normalization
            # above; kept as a hard check because a silent scale creep on
            # PPE geometry is exactly the failure this module exists to
            # prevent (AGENTS.md section 2.2/2.3).
            raise AssertionError(
                f"scale lock failed: composed s={s_full!r} != 1.0 with allow_scale=False"
            )

        # source_ds_idx only records which points were used to FIT
        # (R, t, s); the composed transform is then applied EXACTLY to
        # every point of `source`, never RBF-interpolated (see module
        # docstring: an RBF fit through sampled rigid displacements is
        # only approximately rigid, which is not acceptable for PPE).
        transformed_full = source @ (s_full * R_full) + t_full

        return CPDResult(
            transformed_points=transformed_full,
            rotation=np.asarray(R_full, dtype=np.float64),
            translation=np.asarray(t_full, dtype=np.float64).reshape(3),
            scale=float(s_full),
            iterations=int(getattr(reg, "iteration", 0)),
            converged=True,
        )
