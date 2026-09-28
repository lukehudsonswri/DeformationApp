"""
PCA-based prealignment with sign-flip search and reflection rejection.

This is a port of the standalone `pca_affine_align()` (Morphing Codes/SurfaceNICP.py)
that solves the orientation ambiguity problem when template and target principal
axes happen to be anti-parallel — which causes downstream CPD/affine to converge
to a 180°-flipped solution.

Approach:
  1. Center both point clouds at their respective centroids.
  2. Compute principal axes (right singular vectors) of each.
  3. Try all 8 combinations of axis sign flips on the source basis.
  4. For each, build a rotation+per-axis-scale transform that maps source PCA
     frame onto target PCA frame, with the matched extents.
  5. Reject any combination with negative determinant (would invert the mesh).
  6. Score remaining candidates by mean nearest-neighbor distance on a small
     subsample, pick the best.

--------------------------------------------------------------------------
Vendored from: C:\\Users\\lhudson\\Documents\\Morphing\\fe_personalization\\registration\\pca_prealign.py
Vendored on:   2026-09-18, unmodified.
Local changes: none yet.
Used by (AGENTS.md section 2.2): PCA pre-align is step 4 of the PPE seating
sequence, gated by ``should_apply_prealign`` before the rigid CPD step in
core/registration/rigid_cpd.py. For the armored plate this replaces
pca_prealign's chamfer-only symmetry-breaking with the deterministic
concave-face / trapezoid-top shape signature (fitting/signature.py) as the
seed orientation -- ``orientation_penalty_mm`` here is then a secondary,
not primary, disambiguator.
See AGENTS.md "Vendoring policy".
--------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PCAPreAlignDiagnostics:
    """Diagnostics returned alongside the alignment transform.

    These let the caller decide whether the alignment is meaningful and safe
    to apply, vs. risk worsening downstream stages with unnecessary rotation.
    """
    # Cosine between principal long axes (template's first PC vs. target's first PC).
    # Close to +1: already aligned; close to -1: 180° flipped (the case we MUST fix);
    # near 0: orthogonal (probably degenerate or PCA picked the wrong long axis).
    axis_cosine: float
    # Per-axis extent scale chosen (target_extent / source_extent in PCA frames).
    # Sane muscle range is roughly [0.5, 2.0]; far outliers indicate the source
    # and target shapes are too dissimilar for PCA-based extent matching to be safe.
    scale_per_axis: np.ndarray  # (3,)
    # Sign flips that won the candidate scoring. Useful for logging/debugging.
    chosen_flips: np.ndarray  # (3,) of +/-1
    # Mean nearest-target distance in the scoring subsample, after applying the
    # winning candidate. Smaller is better.
    sample_mean_dist: float
    # Rotation angle (degrees) of the winning candidate's transform vs. identity.
    # Useful to verify the orientation penalty kept the candidate away from a flip.
    winner_rotation_deg: float = 0.0


def pca_affine_align(
    src: np.ndarray,
    tgt: np.ndarray,
    sample_size: int = 800,
    rng_seed: int = 0,
    include_scale: bool = True,
    orientation_penalty_mm: float = 5.0,
    axial_profile_weight_mm: float = 0.0,
    axial_profile_bins: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PCAPreAlignDiagnostics]:
    """Compute a translation+rotation+per-axis-scale that aligns `src` to `tgt`.

    Args:
        src: (N, 3) source point cloud.
        tgt: (M, 3) target point cloud.
        sample_size: number of source points used to score each candidate orientation.
        rng_seed: deterministic seed for the scoring subsample.
        orientation_penalty_mm: weight (in mm) of an orientation prior added to
            each candidate's chamfer score: ``score = chamfer + λ·(1 − cos θ)``,
            where θ is the rotation angle the candidate would apply to the source.
            Breaks symmetry on near-symmetric parts (long muscles, fibula, patella):
            without it, chamfer is roughly equal for flipped vs. unflipped, so the
            winner is essentially noise. A genuine 180° flip in the MRI target
            still wins because its chamfer advantage exceeds the penalty (max 2·λ).
            Set to 0.0 to disable. Default 5.0 mm.
        axial_profile_weight_mm: optional weight for matching end-to-end radial
            profiles along the target long axis. This breaks proximal/distal
            ambiguity for anatomy like tibia where nearest-neighbor chamfer can
            be good with the ends swapped. Default 0.0 preserves legacy behavior.
        axial_profile_bins: number of equal-count long-axis bins used by the
            optional axial profile score.

    Returns:
        (T, src_centroid, tgt_centroid, diagnostics). Apply via:
            aligned = (src - src_centroid) @ T + tgt_centroid
        T has det(T) > 0 (no reflections). `diagnostics` lets the caller decide
        whether to actually apply this — see `should_apply_prealign()`.

    The transform also matches per-axis EXTENTS in the PCA frames, which keeps
    elongated meshes (long muscles, bones) from being over- or under-stretched
    on their long axis when the bbox aspect differs.
    """
    src = np.asarray(src, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)

    sc = src.mean(axis=0)
    tc = tgt.mean(axis=0)
    Xs = src - sc
    Xt = tgt - tc

    _, _, Vs = np.linalg.svd(Xs, full_matrices=False)
    _, _, Vt = np.linalg.svd(Xt, full_matrices=False)

    Xs_pca = Xs @ Vs.T
    Xt_pca = Xt @ Vt.T
    ext_src = Xs_pca.max(0) - Xs_pca.min(0)
    ext_tgt = Xt_pca.max(0) - Xt_pca.min(0)
    scale_extents = ext_tgt / np.maximum(ext_src, 1e-12)
    # When include_scale=False we do rotation+translation only, leaving scale
    # for the downstream affine/CPD stage. This avoids subtly shifting where
    # the volumetric optimizer concentrates displacement.
    scale = scale_extents if include_scale else np.ones(3, dtype=np.float64)

    rng = np.random.RandomState(rng_seed)
    n_sample = min(sample_size, len(src))
    sample_idx = rng.choice(len(src), size=n_sample, replace=False)
    Xs_sample = Xs[sample_idx]
    target_profile = (
        _axial_radial_profile(tgt, Vt[0], tc, axial_profile_bins)
        if axial_profile_weight_mm > 0.0
        else None
    )

    best_T = None
    best_score = np.inf
    best_chamfer = np.inf
    best_angle_deg = 0.0
    best_flips = None
    for flips in range(8):
        sign = np.array([(-1) ** ((flips >> k) & 1) for k in range(3)], dtype=np.float64)
        Vs_signed = Vs * sign[:, None]
        T = Vs_signed.T @ np.diag(scale) @ Vt
        if np.linalg.det(T) <= 0:
            continue
        out = Xs_sample @ T + tc
        # Mean nearest-target-vertex distance (good ranking proxy)
        d2 = ((out[:, None, :] - tgt[None, :, :]) ** 2).sum(axis=-1)
        d = np.sqrt(d2.min(axis=1))
        chamfer = float(d.mean())
        # Rotation angle the candidate would apply to the source. When
        # include_scale=False this is exact (T is a rotation). When scale is
        # baked in, the trace formula is a first-order approximation that's
        # still a good orientation proxy for the typical [0.5, 2.0] scale range.
        cos_theta = float(np.clip((np.trace(T) - 1.0) / 2.0, -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_theta)))
        penalty = orientation_penalty_mm * (1.0 - np.cos(np.radians(angle_deg)))
        profile_penalty = 0.0
        if target_profile is not None:
            profile = _axial_radial_profile(
                Xs @ T + tc, Vt[0], tc, axial_profile_bins
            )
            profile_penalty = axial_profile_weight_mm * float(
                np.mean(np.abs(profile - target_profile))
            )
        score = chamfer + penalty + profile_penalty
        if score < best_score:
            best_score = score
            best_chamfer = chamfer
            best_angle_deg = angle_deg
            best_T = T
            best_flips = sign

    if best_T is None:
        raise RuntimeError(
            "PCA prealign: no orientation-preserving alignment found "
            "(all 8 sign-flip candidates had det <= 0). This should be impossible."
        )

    # Cosine between source and target longest principal axes (in original frames,
    # before any sign flips). +1 = aligned, -1 = anti-parallel (180° flip case),
    # 0 = orthogonal.
    axis_cosine = float(np.dot(Vs[0], Vt[0]))

    # For diagnostics, surface the natural per-axis extent ratio even when we
    # didn't apply it — that's the more useful "are these meshes the same shape?"
    # signal regardless of whether scale was baked into T.
    diag = PCAPreAlignDiagnostics(
        axis_cosine=axis_cosine,
        scale_per_axis=scale_extents.copy(),
        chosen_flips=best_flips.copy(),
        sample_mean_dist=best_chamfer,
        winner_rotation_deg=best_angle_deg,
    )

    logger.info(
        "PCA prealign: axis_cosine=%.3f, chose flips=%s, "
        "per-axis scale=[%.3f, %.3f, %.3f], "
        "winner: chamfer=%.2fmm rot=%.1f° (penalty λ=%.1fmm, axial λ=%.1fmm)",
        axis_cosine,
        best_flips.tolist(),
        scale[0], scale[1], scale[2],
        best_chamfer, best_angle_deg, orientation_penalty_mm,
        axial_profile_weight_mm,
    )
    return best_T, sc, tc, diag


def _axial_radial_profile(
    points: np.ndarray,
    axis: np.ndarray,
    center: np.ndarray,
    bins: int,
) -> np.ndarray:
    """Normalized radial-width profile along ``axis`` in equal-count bins."""
    bins = max(2, int(bins))
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    pts = np.asarray(points, dtype=np.float64)
    rel = pts - np.asarray(center, dtype=np.float64)
    axial = rel @ axis
    radial = np.linalg.norm(rel - axial[:, None] * axis, axis=1)
    edges = np.quantile(axial, np.linspace(0.0, 1.0, bins + 1))
    profile = np.zeros(bins, dtype=np.float64)
    for i in range(bins):
        if i == bins - 1:
            mask = (axial >= edges[i]) & (axial <= edges[i + 1])
        else:
            mask = (axial >= edges[i]) & (axial < edges[i + 1])
        if np.any(mask):
            profile[i] = float(np.quantile(radial[mask], 0.9))
    mean = float(profile.mean())
    if mean <= 1e-12:
        return profile
    return profile / mean


def should_apply_prealign(
    diag: PCAPreAlignDiagnostics,
    pre_mean_dist: float,
    post_mean_dist: float,
    *,
    axis_misaligned_thresh: float = 0.5,
    large_improvement_factor: float = 10.0,
    scale_min: float = 0.5,
    scale_max: float = 2.0,
) -> Tuple[bool, str]:
    """Decide whether to actually apply the PCA prealign transform.

    Two cases warrant applying:
      (a) Long principal axes are meaningfully misaligned
          (axis_cosine < axis_misaligned_thresh, default 0.5 ≈ 60° off).
          This is the 180° flip / orthogonal-axis case.
      (b) The prealign would yield a LARGE relative improvement
          (post < pre / large_improvement_factor, default 10×). This
          catches situations where axes are aligned but template and
          target are in completely different coordinate frames — e.g.
          bone templates that sit near the origin while targets sit
          a meter away. Centroid alignment alone is translation-only,
          but PCA also corrects any small rotation difference.

    Skip when axes are aligned AND the relative improvement is small —
    that's the case where applying PCA can nudge well-aligned meshes
    into worse downstream local minima.
    """
    s = diag.scale_per_axis
    if not (scale_min <= s.min() and s.max() <= scale_max):
        return False, (
            f"reject: scale per axis out of range [{scale_min}, {scale_max}]: "
            f"{s.tolist()}"
        )

    if diag.axis_cosine < axis_misaligned_thresh:
        return True, (
            f"apply: axis_cosine={diag.axis_cosine:.3f} < "
            f"{axis_misaligned_thresh} (long axes meaningfully misaligned)"
        )

    # Relative-improvement test: only fires when prealign offers an
    # order-of-magnitude better starting position than no prealign at all.
    if pre_mean_dist > 0 and post_mean_dist * large_improvement_factor < pre_mean_dist:
        return True, (
            f"apply: post={post_mean_dist:.2f} vs pre={pre_mean_dist:.2f} — "
            f">{large_improvement_factor:g}× improvement"
        )

    return False, (
        f"skip: axis_cosine={diag.axis_cosine:.3f} aligned and "
        f"post={post_mean_dist:.2f} not >{large_improvement_factor:g}× "
        f"better than pre={pre_mean_dist:.2f}"
    )


def apply_pca_alignment(
    points: np.ndarray, T: np.ndarray, src_centroid: np.ndarray, tgt_centroid: np.ndarray
) -> np.ndarray:
    """Apply the alignment from `pca_affine_align` to a point set.

    Common usage: align the FULL template mesh (interior + surface) using a
    transform fit only on its outer surface — interior nodes ride along.
    """
    return (points - src_centroid) @ T + tgt_centroid
