"""
Rigid alignment utilities.

Implements Kabsch algorithm and rigid transform operations.

--------------------------------------------------------------------------
Vendored from: C:\\Users\\lhudson\\Documents\\Morphing\\fe_personalization\\registration\\rigid.py
Vendored on:   2026-09-18, unmodified.
Local changes: none. ``use_gpu`` already defaults to CPU (``None`` ->
               ``False``) and the torch import is guarded by a top-level
               try/except, so this file has zero hard torch dependency --
               consistent with the CPU-only CPD decision (AGENTS.md
               section 5, Q6).
See AGENTS.md "Vendoring policy".
--------------------------------------------------------------------------
"""

import numpy as np
from typing import Tuple, Optional

# Optional PyTorch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def kabsch_rotation(P: np.ndarray, Q: np.ndarray, use_gpu: Optional[bool] = None) -> np.ndarray:
    """
    Compute the optimal rotation matrix using the Kabsch algorithm.

    Finds R that minimizes ||P @ R - Q||^2

    Args:
        P: (N, 3) source points (centered)
        Q: (N, 3) target points (centered)
        use_gpu: Whether to use GPU acceleration (None = auto-detect).
                 Note: For single Kabsch calls on 3x3 covariance matrices,
                 CPU is typically faster. GPU is beneficial for batched operations.

    Returns:
        (3, 3) optimal rotation matrix
    """
    # Auto-detect: CPU is actually faster for single small matrix SVD
    # GPU overhead (data transfer, kernel launch) exceeds computation time
    # Only use GPU if explicitly requested
    if use_gpu is None:
        use_gpu = False  # CPU is faster for single Kabsch

    if use_gpu and TORCH_AVAILABLE:
        return _kabsch_rotation_gpu(P, Q)

    # CPU fallback
    # Covariance matrix
    H = P.T @ Q

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Rotation matrix
    R = Vt.T @ U.T

    # Ensure proper rotation (det(R) = +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


def _kabsch_rotation_gpu(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """GPU-accelerated Kabsch rotation using PyTorch SVD."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    P_t = torch.tensor(P, dtype=torch.float64, device=device)
    Q_t = torch.tensor(Q, dtype=torch.float64, device=device)

    # Covariance matrix
    H = P_t.T @ Q_t

    # SVD
    U, S, Vt = torch.linalg.svd(H)

    # Rotation matrix
    R = Vt.T @ U.T

    # Ensure proper rotation (det(R) = +1)
    if torch.linalg.det(R) < 0:
        Vt_fixed = Vt.clone()
        Vt_fixed[-1, :] *= -1
        R = Vt_fixed.T @ U.T

    return R.cpu().numpy()


def rigid_align(
    source: np.ndarray,
    target: np.ndarray,
    allow_scale: bool = False,
    use_gpu: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute optimal rigid (or similarity) transform from source to target.

    Finds R, t, s such that: target ≈ s * (source @ R) + t

    Args:
        source: (N, 3) source point cloud
        target: (N, 3) target point cloud (must have same N)
        allow_scale: If True, also optimize for uniform scale
        use_gpu: Whether to use GPU acceleration (None = auto-detect)

    Returns:
        (aligned_source, R, t, s) where:
            aligned_source: transformed source points
            R: (3, 3) rotation matrix
            t: (3,) translation vector
            s: scale factor (1.0 if allow_scale=False)
    """
    assert source.shape == target.shape, "Source and target must have same shape"

    # Center both point sets
    src_center = source.mean(axis=0)
    tgt_center = target.mean(axis=0)

    src_centered = source - src_center
    tgt_centered = target - tgt_center

    # Compute rotation (with GPU if available)
    R = kabsch_rotation(src_centered, tgt_centered, use_gpu=use_gpu)

    # Compute scale if requested
    if allow_scale:
        # Optimal scale: s = trace(R^T @ H) / ||P||^2
        H = src_centered.T @ tgt_centered
        rotated = src_centered @ R
        s = np.sum(rotated * tgt_centered) / (np.sum(src_centered ** 2) + 1e-10)
        s = max(0.1, min(10.0, s))  # Clamp to reasonable range
    else:
        s = 1.0

    # Compute translation
    t = tgt_center - s * (src_center @ R)

    # Apply transform
    aligned = s * (source @ R) + t

    return aligned, R, t, s


def apply_rigid_transform(
    points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    s: float = 1.0,
    center: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply rigid transform to points.

    Args:
        points: (N, 3) point cloud
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        s: scale factor
        center: Optional center of rotation (if None, uses origin)

    Returns:
        (N, 3) transformed points
    """
    if center is not None:
        points = points - center

    transformed = s * (points @ R)

    if center is not None:
        transformed = transformed + center

    return transformed + t


def compute_rigid_error(
    source: np.ndarray,
    target: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    s: float = 1.0,
) -> float:
    """
    Compute RMSE after applying rigid transform.

    Args:
        source: (N, 3) source points
        target: (N, 3) target points
        R, t, s: Transform parameters

    Returns:
        Root mean squared error
    """
    transformed = s * (source @ R) + t
    diff = transformed - target
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))


def procrustes_analysis(
    source: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Perform Procrustes analysis (rigid alignment with scale).

    Args:
        source: (N, 3) source points
        target: (N, 3) target points

    Returns:
        (aligned_source, disparity) where disparity is the sum of squared differences
    """
    aligned, R, t, s = rigid_align(source, target, allow_scale=True)
    disparity = np.sum((aligned - target) ** 2)
    return aligned, disparity


class RigidAligner:
    """
    Class-based rigid alignment interface.

    Provides a consistent API for rigid alignment operations.
    """

    def __init__(self, allow_scale: bool = False, use_gpu: Optional[bool] = None):
        """
        Initialize aligner.

        Args:
            allow_scale: Whether to allow uniform scaling
            use_gpu: Whether to use GPU acceleration (None = auto-detect)
        """
        self.allow_scale = allow_scale
        self.use_gpu = use_gpu
        self.R = None
        self.t = None
        self.s = 1.0

    def align(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> dict:
        """
        Align source to target.

        Args:
            source: (N, 3) source point cloud
            target: (N, 3) target point cloud

        Returns:
            dict with 'aligned', 'rotation', 'translation', 'scale'
        """
        aligned, R, t, s = rigid_align(source, target, self.allow_scale, use_gpu=self.use_gpu)

        self.R = R
        self.t = t
        self.s = s

        return {
            'aligned': aligned,
            'rotation': R,
            'translation': t,
            'scale': s,
        }

    def transform(self, points: np.ndarray) -> np.ndarray:
        """
        Apply the learned transform to new points.

        Args:
            points: (M, 3) points to transform

        Returns:
            (M, 3) transformed points
        """
        if self.R is None:
            raise ValueError("Must call align() first")

        return self.s * (points @ self.R) + self.t


def generalized_procrustes(
    shapes: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
    allow_scale: bool = False,
    use_gpu: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generalized Procrustes Analysis for multi-shape alignment.

    Args:
        shapes: (M, N, 3) array of M shapes, each with N points
        max_iter: Maximum iterations
        tol: Convergence tolerance
        allow_scale: Whether to allow scaling
        use_gpu: Whether to use GPU acceleration (None = auto-detect)

    Returns:
        (aligned_shapes, mean_shape) both as numpy arrays
    """
    M, N, D = shapes.shape
    assert D == 3, "Expected 3D points"

    # Initialize mean shape
    mean_shape = shapes.mean(axis=0)
    mean_shape -= mean_shape.mean(axis=0)  # Center

    aligned = np.zeros_like(shapes)

    for it in range(max_iter):
        # Align each shape to current mean
        for i in range(M):
            aligned[i], _, _, _ = rigid_align(shapes[i], mean_shape, allow_scale=allow_scale, use_gpu=use_gpu)

        # Compute new mean
        new_mean = aligned.mean(axis=0)
        new_mean -= new_mean.mean(axis=0)  # Center

        # Check convergence
        diff = np.linalg.norm(new_mean - mean_shape) / (np.linalg.norm(mean_shape) + 1e-12)

        mean_shape = new_mean

        if diff < tol:
            break

    return aligned, mean_shape
