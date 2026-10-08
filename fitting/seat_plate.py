"""Rigid seating of a PPE against a body region: shape-signature seed
rotation + coarse translation recovery + local point-to-plane ICP
refinement + closed-form standoff correction + bilateral symmetry
correction (AGENTS.md Goal 2).

Sequence:

1. Extract the body region's skin surface (``fitting.body_surface``) and
   the PPE's concave (body-facing) surface (``fitting.plate_signature``).
2. Offset the skin surface outward along its own vertex normals by the
   (mesh-floor-resolved, ``fitting.standoff``) target standoff -- ICP then
   drives the plate's concave surface onto this offset shell directly.
3. Seed a rigid rotation from the two shape signatures (plate's
   concave-normal/up-axis pair -> a *fixed anatomical anchor's*
   local-outward-normal/superior-axis pair -- not a lookup near the PPE's
   own current position, which would be meaningless if the PPE starts far
   away). This resolves orientation deterministically, with no free-form
   search needed to get the concave face facing the right way, regardless
   of the PPE's starting rotation.
4. Coarse translation: move the plate so its concave surface sits just
   outside that same anatomical anchor. This is what actually solves "the
   PPE arrived in an unrelated coordinate system, far from the body" --
   verified directly against a synthetic ~130-degree rotation + ~500mm
   translation of the real armored plate, converging to the same final
   seating as starting from the already-close shipped file.
5. Crop the offset target to a local neighborhood of the (now-oriented,
   now-nearby) plate -- registering a small patch against an entire
   torso's worth of skin is ill-posed (see
   ``core.registration.point_to_plane_icp`` module docstring for the
   empirical failure this avoids).
6. Point-to-plane ICP refinement of the plate's concave surface onto the
   cropped local target.
7. Closed-form standoff correction (``_apply_standoff_correction``):
   translate along the plate's current concave-normal direction until the
   minimum signed gap to the *real* (uncropped) body skin equals the
   resolved target standoff.
8. Bilateral symmetry correction (``_apply_symmetry_correction``): correct
   yaw about the body's vertical axis (checked via a rim-to-rim vector
   across the plate's width -- it should lie entirely in the body's
   coronal plane), remove any lateral roll the ICP fit introduced, and
   recenter the plate on the body's own local left-right midline, then
   re-apply step 7's standoff correction. Real, measurable asymmetries
   (0.67 degrees of yaw, 0.73 degrees of roll, ~1mm of lateral offset, on
   real data) were found and fixed here -- see that function's docstring
   for the full writeup. Skippable via ``enforce_symmetry=False``.
9. Apply the composed transform to the PPE's full volume mesh (not just its
   concave surface) -- every stage above is rigid, applied identically to
   the full node set alongside the surface subset driving the next stage.

Why not CPD (``core.registration.rigid_cpd``) for step 6, despite AGENTS.md's
original plan calling for it: tested directly against real data (F05 torso +
the shipped armored plate) and it diverged -- see the point-to-plane ICP
module docstring for the full empirical writeup. CPD remains available in
the codebase for PPE shapes with enough unambiguous 3D structure that
correspondence search has a genuine shape signal to lock onto; it just isn't
the right tool for a near-flat rigid plate against a broadly-similar-curved
patch of torso.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree

from config.hbm_models import HbmModel
from config.ppe import PpeSpec
from config.sites import SiteResolution
from core.mesh.base import SurfaceMesh, VolumeMesh
from core.mesh.signed_distance import SignedGapResult, signed_gap_to_surface
from core.registration.point_to_plane_icp import ICPConfig, ICPResult, point_to_plane_icp
from fitting.body_surface import extract_skin_surface
from fitting.plate_signature import PlateOrientation, compute_plate_orientation
from fitting.standoff import StandoffFloor, check_standoff, compute_standoff_floor, resolve_target_standoff

# F05_Standing export frame convention (AGENTS.md 2.1): superior is -Z. This
# is a property of the *export frame*, not the anatomy -- re-derive (don't
# assume) for any other model export.
DEFAULT_BODY_UP_AXIS = np.array([0.0, 0.0, -1.0])

# Anterior direction for the same export frame: the shipped, already-seated
# PPE/plate.inp sits at X in [75, 107], just beyond the torso skin's own
# X range of [-147, 88.7] -- i.e. the skin's own most-anterior extent is
# ~X=88.7, right where the known-good plate touches it. +X is anterior.
# Used only as a *default anchor* for coarse recovery when a PPE starts far
# from the body (see `_default_anchor_point`) -- re-derive per model/site,
# same caveat as DEFAULT_BODY_UP_AXIS.
DEFAULT_ANTERIOR_AXIS = np.array([1.0, 0.0, 0.0])

# Left-right (medial-lateral) direction for the same export frame -- the
# remaining axis once up (Z) and anterior (X) are fixed. Confirmed directly:
# the torso skin's own Y range (-155.3 to 155.7mm) is nearly symmetric about
# 0 (bbox center 0.166mm, mean 0.067mm, median 0.125mm -- all within
# ~0.2mm of each other and of zero), consistent with an anthropometric body
# mesh that is itself bilaterally symmetric. Used by the final symmetry
# correction stage (`_apply_symmetry_correction`) -- re-derive, same caveat
# as the other two axes, for any other model export.
DEFAULT_LATERAL_AXIS = np.array([0.0, 1.0, 0.0])


@dataclass
class SeatingResult:
    """Everything a caller needs to inspect or apply a seating fit."""

    plate_orientation: PlateOrientation
    body_surface: SurfaceMesh
    offset_target_points: np.ndarray
    seed_rotation: np.ndarray  # (3, 3), row-vector convention: v_row @ seed_rotation
    icp_result: ICPResult
    transformed_plate_nodes: np.ndarray  # full PPE mesh, after all stages
    pre_icp_mean_dist_mm: float  # concave -> local target, right after the seed rotation
    standoff_floor: StandoffFloor
    resolved_target_standoff_mm: float  # target_standoff_mm, raised to the mesh floor if needed
    final_gap: SignedGapResult  # concave surface vs. the real (uncropped) body skin, after standoff correction
    symmetry_twist_deg: float  # de-twist rotation applied by the symmetry stage (0.0 if disabled)
    symmetry_lateral_shift_mm: float  # lateral recentering applied by the symmetry stage (0.0 if disabled)
    symmetry_yaw_deg: float  # rim-vector yaw correction applied by the symmetry stage (0.0 if disabled)
    fit_skipped: bool = False  # True when the plate already had an acceptable standoff and was left untouched


def _mean_nearest_dist(points: np.ndarray, target: np.ndarray) -> float:
    tree = cKDTree(target)
    d, _ = tree.query(points)
    return float(d.mean())


def _orthonormal_frame(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Rows = an orthonormal right-handed basis: ``primary`` (normalized),
    the component of ``secondary`` orthogonal to it (normalized), and their
    cross product.
    """
    e0 = primary / np.linalg.norm(primary)
    e1 = secondary - np.dot(secondary, e0) * e0
    n1 = np.linalg.norm(e1)
    if n1 < 1e-8:
        # secondary is (near-)parallel to primary -- fall back to an
        # arbitrary vector not parallel to e0.
        fallback = np.array([1.0, 0.0, 0.0]) if abs(e0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = fallback - np.dot(fallback, e0) * e0
        n1 = np.linalg.norm(e1)
    e1 = e1 / n1
    e2 = np.cross(e0, e1)
    return np.stack([e0, e1, e2], axis=0)


def seed_rotation_from_signature(
    plate_orientation: PlateOrientation,
    target_body_normal: np.ndarray,
    target_up_axis: np.ndarray,
) -> np.ndarray:
    """Rotation ``R`` (row-vector convention, ``v_row @ R``) mapping the
    plate's own (concave_normal, up_axis) frame onto a target frame where
    the concave face points *toward* the body (i.e. anti-parallel to the
    body's own outward surface normal) and the plate's top end points along
    the body's superior axis.

    Both source and target frames are built the same way (an orthonormal
    right-handed basis from a primary + secondary vector), so the rotation
    mapping one onto the other is guaranteed proper (``det(R) = 1``, no
    reflection) regardless of the input vectors' exact orthogonality.
    """
    source_frame = _orthonormal_frame(plate_orientation.concave_normal, plate_orientation.up_axis)
    target_frame = _orthonormal_frame(-target_body_normal, target_up_axis)
    # source_frame @ R = target_frame (row i: (row i of source) @ R = row i
    # of target); source_frame is orthogonal so its inverse is its
    # transpose.
    return source_frame.T @ target_frame


def _local_body_normal(body_surface: SurfaceMesh, near_point: np.ndarray, k: int = 50) -> np.ndarray:
    """Average outward normal of the ``k`` body-surface vertices nearest
    ``near_point`` -- a local estimate of "which way does the body face
    here", used only to seed the initial rotation (ICP refines the rest).
    """
    tree = cKDTree(body_surface.nodes)
    k = min(k, len(body_surface.nodes))
    _, idx = tree.query(near_point, k=k)
    normal = body_surface.vertex_normals[idx].mean(axis=0)
    return normal / np.linalg.norm(normal)


def _default_anchor_point(
    body_surface: SurfaceMesh, anterior_axis: np.ndarray = DEFAULT_ANTERIOR_AXIS
) -> Tuple[np.ndarray, np.ndarray]:
    """The body surface's own most-anterior point, and its local outward
    normal there -- a coarse, PPE-position-independent anchor for both the
    orientation seed and the coarse translation recovery.

    This is what actually makes seating robust to a PPE that starts far
    away (AGENTS.md Goal 2: "if the torso is rotated in some weird way and
    is located far away, I need a way to bring it back"): using the PPE's
    own (possibly nonsensical, if it's far away) current position to look
    up "the nearest body surface point" -- as ``_local_body_normal`` does --
    would seed the wrong orientation entirely if that nearest point happens
    to be e.g. the back of the torso instead of the front. Anchoring on a
    fixed anatomical reference instead removes that dependency.
    """
    centroid = body_surface.nodes.mean(axis=0)
    proj = (body_surface.nodes - centroid) @ anterior_axis
    idx = int(np.argmax(proj))
    return body_surface.nodes[idx], body_surface.vertex_normals[idx]


def _coarse_translate_to_anchor(
    full_points: np.ndarray,
    concave_ids: np.ndarray,
    anchor_point: np.ndarray,
    anchor_normal: np.ndarray,
    clearance_mm: float = 10.0,
) -> np.ndarray:
    """Translate (no rotation) so the plate's concave-surface centroid sits
    ``clearance_mm`` outward from ``anchor_point`` along ``anchor_normal``.

    Purely a coarse recovery step -- gets a PPE that started arbitrarily far
    away into the local ICP/crop's capture basin. Refined precisely by the
    ICP + standoff-correction stages that follow.
    """
    target = anchor_point + anchor_normal * clearance_mm
    current_centroid = full_points[concave_ids].mean(axis=0)
    return full_points + (target - current_centroid)


def _crop_indices_near(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Indices of ``points`` within ``radius`` of ``center``.

    Registering a small PPE patch against the *entire* torso skin is
    ill-posed for both CPD and ICP (see this module's docstring). Cropping
    the target to a local neighborhood fixes it -- but the radius must be
    an *additive* margin on top of the plate's own half-extent, not a
    multiple of it: a first attempt using ``1.75 * plate_radius`` (plate
    half-diagonal ~195mm) gave a ~342mm radius, comparable to the entire
    torso's own extents (Y span 311mm, Z span 536mm), and captured 11,185
    of 11,412 skin points -- i.e. cropped almost nothing.
    """
    tree = cKDTree(points)
    idx = tree.query_ball_point(center, r=radius)
    if len(idx) < 10:
        raise ValueError(
            f"only {len(idx)} target points within {radius:.1f}mm of {center} -- "
            f"the PPE may be too far from the body region for local cropping to "
            f"find a plausible neighborhood (try a larger radius, or check the "
            f"seed orientation)"
        )
    return np.asarray(idx, dtype=np.int64)


def _apply_standoff_correction(
    full_points: np.ndarray,
    concave_ids: np.ndarray,
    direction: np.ndarray,
    body_surface: SurfaceMesh,
    target_standoff_mm: float,
    max_iters: int = 10,
    tolerance_mm: float = 1e-4,
) -> tuple[np.ndarray, SignedGapResult]:
    """Closed-form translation along ``direction`` so the concave surface's
    minimum signed gap to ``body_surface`` equals ``target_standoff_mm``
    (AGENTS.md section 2.4, step 2: "translate the PPE along its
    body-facing normal until the minimum signed gap equals the target
    standoff. Closed-form, no iteration on the solver.").

    A single step assumes the extremal point's local body normal is exactly
    anti-parallel to ``direction`` (true by construction right after
    seating, approximately true after that) -- so this iterates a handful
    of times to correct for the approximation rather than solving a general
    optimization. Each step is a plain translation (no rotation), so it
    stays an exact rigid transform throughout.
    """
    current = full_points
    gap = signed_gap_to_surface(current[concave_ids], body_surface)
    for _ in range(max_iters):
        delta = gap.min_mm - target_standoff_mm
        if abs(delta) <= tolerance_mm:
            break
        current = current + delta * direction
        gap = signed_gap_to_surface(current[concave_ids], body_surface)
    return current, gap


def _project_out_axis(vector: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """``vector`` with its component along (unit) ``axis`` removed."""
    return vector - np.dot(vector, axis) * axis


def _signed_angle_about_axis(v_from: np.ndarray, v_to: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle (radians, right-hand rule about unit ``axis``) to rotate
    ``v_from`` onto ``v_to`` -- both assumed already (at least approximately)
    perpendicular to ``axis``, as they are every place this is used below.
    """
    v_from_n = v_from / np.linalg.norm(v_from)
    v_to_n = v_to / np.linalg.norm(v_to)
    cos_a = np.clip(np.dot(v_from_n, v_to_n), -1.0, 1.0)
    sin_a = np.dot(np.cross(v_from_n, v_to_n), axis)
    return float(np.arctan2(sin_a, cos_a))


def _rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix ``R`` (row-vector convention, ``v_row @ R``, matching
    every other rotation in this module) that rotates any vector by
    ``angle`` radians about unit ``axis`` (right-hand rule), via the
    coordinate-free vector form of Rodrigues' rotation formula applied to
    each standard basis vector in turn (``R``'s row ``i`` = rotated ``e_i``,
    since rotation is linear).
    """
    k = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    rows = []
    for e in np.eye(3):
        rows.append(e * c + np.cross(k, e) * s + k * np.dot(k, e) * (1.0 - c))
    return np.stack(rows, axis=0)


def _find_rim_indices(
    plate_nodes_local: np.ndarray,
    concave_face_node_ids: np.ndarray,
    up_axis_local: np.ndarray,
    lateral_axis_local: np.ndarray,
    band_fraction: float = 0.1,
    rim_fraction: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Node indices (absolute, into the plate's own full node array -- same
    space as ``concave_face_node_ids``) of the concave face's left-rim and
    right-rim points, restricted to a narrow vertical band around
    mid-height (avoids the trapezoid taper at the top -- see
    ``fitting.plate_signature`` -- skewing the comparison).

    The vector connecting these two rim centroids is what
    ``_apply_symmetry_correction``'s yaw stage checks: it should lie
    entirely in the body's coronal (up/lateral) plane if the plate squarely
    faces the body -- any component along the anterior axis means one side
    sits closer to (or further from) the body than the other, i.e. the
    plate is yawed about the body's vertical axis, not just laterally
    off-center. A user-suggested check (draw a vector across the plate's
    width and see whether it's parallel to the body's own coronal plane) --
    verified directly against a real solved case: the previous
    "de-twist + lateral recentering" symmetry stage (rotating only about
    the plate's own *normal*) does not, and cannot, correct this, since a
    yaw about the *vertical* axis is a different rotational degree of
    freedom entirely.
    """
    concave_pts = plate_nodes_local[concave_face_node_ids]
    up_coord = concave_pts @ up_axis_local
    lateral_coord = concave_pts @ lateral_axis_local

    up_lo, up_hi = up_coord.min(), up_coord.max()
    band_lo = up_lo + (0.5 - band_fraction / 2) * (up_hi - up_lo)
    band_hi = up_lo + (0.5 + band_fraction / 2) * (up_hi - up_lo)
    band_mask = (up_coord >= band_lo) & (up_coord <= band_hi)

    band_ids = concave_face_node_ids[band_mask]
    lateral_in_band = lateral_coord[band_mask]
    lat_lo, lat_hi = lateral_in_band.min(), lateral_in_band.max()
    rim_width = lat_hi - lat_lo
    left_mask = lateral_in_band <= lat_lo + rim_fraction * rim_width
    right_mask = lateral_in_band >= lat_hi - rim_fraction * rim_width

    return band_ids[left_mask], band_ids[right_mask]


def _local_mirror_symmetry_offset(
    points: np.ndarray, axis: np.ndarray, sample_frac: float = 0.3, seed: int = 0
) -> float:
    """The offset ``o`` (scalar along unit ``axis``) such that reflecting
    ``points`` about the plane ``{x : x . axis = o}`` best overlaps
    ``points`` with itself -- i.e. ``points``' own true mirror-symmetry
    plane along ``axis``, found by minimizing mean squared nearest-neighbor
    distance between the reflected and original point sets.

    Used (instead of a naive bounding-box center) to resolve the body's own
    lateral midline for the symmetry-correction stage below.  Verified
    directly on real data: the *whole-torso* skin's own bbox-center is a
    poor proxy for local symmetry right under the plate (self-mirror
    residual ~25 mm^2 there) -- a *local* patch's own fitted mirror plane
    fits far better (residual ~0.4 mm^2) and is what should actually be
    used as the plate's lateral seating target.
    """
    rng = np.random.default_rng(seed)
    if sample_frac < 1.0:
        n = max(1, int(len(points) * sample_frac))
        idx = rng.choice(len(points), size=n, replace=False)
        pts = points[idx]
    else:
        pts = points
    tree = cKDTree(pts)
    coord = pts @ axis

    def _residual(o: float) -> float:
        reflected = pts + 2.0 * (o - coord)[:, None] * axis[None, :]
        d, _ = tree.query(reflected, k=1)
        return float(np.mean(d ** 2))

    res = minimize_scalar(_residual, bounds=(coord.min(), coord.max()), method="bounded",
                           options={"xatol": 1e-4})
    return float(res.x)


def _find_spine_groove_lateral_mm(
    body_surface: SurfaceMesh,
    anterior_axis: np.ndarray,
    body_up_axis: np.ndarray,
    lateral_axis: np.ndarray,
    z_band: Optional[Tuple[float, float]] = None,
    posterior_fraction: float = 0.3,
    search_half_width_mm: float = 30.0,
    n_bands: int = 12,
    min_valid_bands: int = 3,
) -> float:
    """Find the body's own true bilateral symmetry line via the **spine
    groove** -- the visible indentation running down the back where the
    skin dips slightly toward the front (anterior) between the erector
    spinae muscle bulges on either side, directly over the spinous
    processes.

    This replaces an earlier version of the lateral-recentering target
    (a numerically-fitted local mirror-symmetry plane near the plate,
    ``_local_mirror_symmetry_offset`` / ``_resolve_local_lateral_target``)
    after the user visually identified the spine groove itself, in a
    rendered case, as "the natural central dividing indentation" and
    pointed out the plate's center of mass was not aligned with it.
    Measured directly: the local-mirror-fit target was off from the spine
    groove by **~1.1mm** -- it had overfit to some local asymmetry near the
    plate's own position (front/anterior side: pecs, sternum) rather than
    reflecting the body's true structural midline. The spine groove is a
    real, unambiguous, bony/structural landmark (unlike a numerically-fit
    plane or a bounding-box center, which have no direct anatomical
    meaning) -- confirmed to closely match the whole-torso bounding-box
    center too (within ~0.2mm), unlike the local-fit target.

    Method: restrict to the posterior ``posterior_fraction`` of the body's
    own anterior-axis range (the "back"), bin by height into ``n_bands``
    slices (restricted to ``z_band`` if given -- e.g. the seated plate's
    own vertical extent, so the groove is measured at the height that
    actually matters, not averaged over the whole torso including the neck
    or waist where its exact lateral position may drift slightly). Within
    each band, restrict to a ``search_half_width_mm`` window around the
    coordinate origin (avoids picking up shoulder-blade/erector-spinae
    bulges as a false groove) and fit a quadratic ``X(Y) = a*Y^2 + b*Y + c``
    -- the groove is a **local maximum** in anterior coordinate (the skin
    dips toward the front there relative to the muscle on either side), so
    only concave-down fits (``a < 0``) with a vertex inside the search
    window are kept. Returns the median across valid bands (robust to the
    occasional band picking up an off-midline feature instead).

    Falls back to the whole-torso bounding-box center (logged) if fewer
    than ``min_valid_bands`` bands produce a usable estimate -- e.g. a
    body region/mesh where the back isn't captured, or too coarse a mesh
    for the quadratic fit to be meaningful.
    """
    x = body_surface.nodes @ anterior_axis
    y = body_surface.nodes @ lateral_axis
    z = body_surface.nodes @ body_up_axis

    x_lo, x_hi = x.min(), x.max()
    back_mask = x < x_lo + posterior_fraction * (x_hi - x_lo)
    back_x, back_y, back_z = x[back_mask], y[back_mask], z[back_mask]

    z_lo, z_hi = z_band if z_band is not None else (back_z.min(), back_z.max())

    groove_ys = []
    for i in range(n_bands):
        band_lo = z_lo + i * (z_hi - z_lo) / n_bands
        band_hi = z_lo + (i + 1) * (z_hi - z_lo) / n_bands
        band_mask = (back_z >= band_lo) & (back_z < band_hi)
        if band_mask.sum() < 20:
            continue
        by, bx = back_y[band_mask], back_x[band_mask]
        mid_mask = np.abs(by) < search_half_width_mm
        if mid_mask.sum() < 8:
            continue
        by, bx = by[mid_mask], bx[mid_mask]
        a, b, _c = np.polyfit(by, bx, 2)
        if a >= -1e-8:  # not concave-down -- no groove signal in this band
            continue
        vertex_y = -b / (2 * a)
        if abs(vertex_y) > search_half_width_mm:
            continue
        groove_ys.append(vertex_y)

    if len(groove_ys) < min_valid_bands:
        fallback = float((y.min() + y.max()) / 2.0)
        print(
            f"[seat_plate] spine groove detection found only {len(groove_ys)} valid height "
            f"band(s) (< {min_valid_bands}) -- falling back to whole-torso bbox-center "
            f"({fallback:.4f}mm) for the lateral symmetry target."
        )
        return fallback

    return float(np.median(groove_ys))


def _apply_symmetry_correction(
    full_points: np.ndarray,
    concave_ids: np.ndarray,
    left_rim_ids: np.ndarray,
    right_rim_ids: np.ndarray,
    up_axis_current: np.ndarray,
    concave_normal_current: np.ndarray,
    body_up_axis: np.ndarray,
    anterior_axis: np.ndarray,
    lateral_axis: np.ndarray,
    target_lateral_mm: float,
    body_surface: SurfaceMesh,
    target_standoff_mm: float,
) -> tuple[np.ndarray, float, float, float, SignedGapResult, np.ndarray, np.ndarray]:
    """Final seating stage: make the plate sit bilaterally symmetric on the
    torso, so results (contact pressure, strain) come out symmetric across
    the midline, not just "close" to the body.

    A real asymmetry was found and fixed here. Point-to-plane ICP's own
    rotation (stage 4, needed to conform the plate to the body's *local*
    curvature) is a general 3D rotation -- nothing about it is constrained
    to preserve the plate's lateral centering, keep its up-axis strictly out
    of the lateral direction, or keep its two sides equidistant from the
    body. Measured directly on real solved cases: a **0.73-degree roll**
    about the plate's own normal (tilts the ~240mm-wide plate by ~3mm
    differential, edge to edge, at full press), a **1.0mm lateral offset**
    from the torso's own midline, and -- found only after the first two
    fixes still left the solved contact/displacement field visibly uneven
    left-to-right -- a **0.67-degree yaw** about the body's *vertical* axis
    (one side of the plate sits measurably closer to the body than the
    other; confirmed directly via the rim-vector check below). None of
    these three are zero by construction anywhere upstream.

    Three corrections, in order:

    0. **Yaw correction** (about ``body_up_axis``, the one rotational
       degree of freedom the de-twist stage below cannot reach, since that
       one only rotates about the plate's *normal*): take the concave
       face's left-rim and right-rim centroids at mid-height
       (``_find_rim_indices`` -- a narrow band avoids the trapezoid taper
       at the top skewing the comparison), and rotate about
       ``body_up_axis`` (pivoting at the rim-vector's own midpoint) until
       the vector connecting them has **zero component along
       ``anterior_axis``** -- i.e. until it lies entirely in the body's own
       coronal (up/lateral) plane, meaning both sides of the plate are
       equally far from the body. This is exactly "draw a vector across the
       plate's width and check it's parallel to the Y-Z (lateral/vertical)
       plane" -- confirmed by direct investigation to be a real, separate
       effect from the roll the de-twist stage already corrected: on a real
       solved case, the roll-only correction left an uncorrected ~0.67
       degree yaw that the earlier two corrections cannot reach at all.
    1. **De-twist rotation**, about the plate's own *current* body-facing
       normal (``concave_normal_current``, recomputed after the yaw stage
       above rotates it slightly) -- the target "up" direction is
       ``body_up_axis`` projected into the plane perpendicular to the
       normal (the closest the plate's up-axis can get to *global* up,
       given the normal direction ICP already fit), not simply re-imposing
       ``body_up_axis`` outright, which would fight the normal-direction
       fit instead of just removing the lateral tilt component.
    2. **Lateral recentering**, a pure translation along ``lateral_axis`` so
       the plate's concave-face centroid lands exactly on
       ``target_lateral_mm`` (see ``_find_spine_groove_lateral_mm`` for why
       this should be the body's own spine-groove landmark, not a
       numerically-fit local mirror-plane or the whole torso's own
       bounding-box center).

    All three are rigid (rotation about a fixed axis/point, then
    translation), so ``_apply_standoff_correction`` is re-run afterward to
    restore the exact target standoff -- each correction can perturb the
    achieved gap slightly.

    Verified end to end (see ``tests/test_seat_plate_symmetry.py``): after
    this stage, the rim-to-rim vector has ~0 anterior component, the
    plate's up-axis has ~0 lateral component, and its concave-face centroid
    lands within a few hundredths of a mm of ``target_lateral_mm``, on real
    F05_Standing + armored_plate data.
    """
    concave_normal_current = concave_normal_current / np.linalg.norm(concave_normal_current)
    up_axis_current = up_axis_current / np.linalg.norm(up_axis_current)

    # --- stage 0: yaw correction about body_up_axis, from the rim-to-rim vector ---
    left_centroid = full_points[left_rim_ids].mean(axis=0)
    right_centroid = full_points[right_rim_ids].mean(axis=0)
    rim_vector = right_centroid - left_centroid
    rim_pivot = (left_centroid + right_centroid) / 2.0

    anterior_component = np.dot(rim_vector, anterior_axis)
    target_rim_vector = rim_vector - anterior_component * anterior_axis
    yaw_angle = _signed_angle_about_axis(rim_vector, target_rim_vector, body_up_axis)

    R_yaw = _rotation_about_axis(body_up_axis, yaw_angle)
    current = (full_points - rim_pivot) @ R_yaw + rim_pivot
    concave_normal_current = concave_normal_current @ R_yaw
    concave_normal_current /= np.linalg.norm(concave_normal_current)
    up_axis_current = up_axis_current @ R_yaw
    up_axis_current /= np.linalg.norm(up_axis_current)

    # --- stage 1: de-twist rotation about the (post-yaw) current normal ---
    desired_up = _project_out_axis(body_up_axis, concave_normal_current)
    desired_up /= np.linalg.norm(desired_up)
    twist_angle = _signed_angle_about_axis(up_axis_current, desired_up, concave_normal_current)

    pivot = current[concave_ids].mean(axis=0)
    R_detwist = _rotation_about_axis(concave_normal_current, twist_angle)
    current = (current - pivot) @ R_detwist + pivot
    # de-twist rotates about concave_normal_current itself, so the normal
    # direction is unchanged (rotating a vector about itself is identity);
    # up_axis_current becomes exactly desired_up by construction of twist_angle.
    up_axis_current = desired_up

    # --- stage 2: lateral recentering ---
    concave_centroid = current[concave_ids].mean(axis=0)
    lateral_shift = target_lateral_mm - np.dot(concave_centroid, lateral_axis)
    current = current + lateral_shift * lateral_axis

    # --- stage 3: closed-form standoff correction, restored after the above ---
    current, final_gap = _apply_standoff_correction(
        current, concave_ids, concave_normal_current, body_surface, target_standoff_mm
    )
    return (
        current,
        float(np.degrees(yaw_angle)),
        float(np.degrees(twist_angle)),
        float(lateral_shift),
        final_gap,
        concave_normal_current,
        up_axis_current,
    )


def seat_plate(
    model: HbmModel,
    site: SiteResolution,
    ppe_spec: PpeSpec,
    plate_mesh: Optional[VolumeMesh] = None,
    target_standoff_mm: float = 0.1,
    body_up_axis: np.ndarray = DEFAULT_BODY_UP_AXIS,
    anterior_axis: np.ndarray = DEFAULT_ANTERIOR_AXIS,
    lateral_axis: np.ndarray = DEFAULT_LATERAL_AXIS,
    icp_config: Optional[ICPConfig] = None,
    seating_margin_mm: float = 75.0,
    far_away_threshold_mm: float = 200.0,
    vertical_offset_mm: float = 0.0,
    enforce_symmetry: bool = True,
    target_lateral_mm: Optional[float] = None,
    accept_standoff_range_mm: Optional[Tuple[float, float]] = None,
) -> SeatingResult:
    """Rigidly seat ``ppe_spec`` (a rigid plate) against ``site`` in
    ``model``, at ``target_standoff_mm`` outward from the skin.

    Robust to the PPE starting in an arbitrary pose -- rotated any way,
    translated arbitrarily far from the body (AGENTS.md Goal 2). Two paths,
    chosen by ``far_away_threshold_mm``:

    - **Already near the body** (the common case: a PPE placed by a human
      or a previous fitting run, in roughly the right spot): orientation is
      seeded from the *local* body normal nearest the PPE's own current
      position, and no coarse translation runs. This is the precise path,
      verified end to end against real data.
    - **Far away** (``dist(plate_centroid, body_centroid) >
      far_away_threshold_mm``): orientation is seeded from a fixed
      anatomical anchor instead (``_default_anchor_point`` -- the body
      surface's own most-anterior point), and the plate is coarsely
      translated there before the local ICP stage runs. This recovers a
      usable seating (correct concave-face orientation; the closed-form
      standoff correction still lands the closest-approach point precisely
      on target) but with a wider tangential placement error than the
      near-body path, since the fixed anchor is only an approximate
      anatomical stand-in, not a per-PPE landmark. Verified against a real
      ~130-degree rotation + ~500mm translation of the shipped plate.

    Do not remove the near/far branch to "simplify" this to always using
    the anatomical anchor -- that was tried, and directly degraded the
    already-good near-body case (ICP's known aperture-problem sensitivity
    to starting position turned a ~65mm anchor-vs-truth offset into a worse
    overall fit; caught by this module's own regression tests).

    ``plate_mesh`` can be passed in already-loaded (e.g. by a caller that
    also needs it for other purposes) to avoid loading it twice; otherwise
    it is loaded from ``ppe_spec``.

    ``seating_margin_mm`` bounds how far off the seeded pose is allowed to
    be from the truth before the local ICP target neighborhood no longer
    contains the correct body patch -- see ``_crop_indices_near``. The
    default (75mm) is generous for the plate's own footprint (~195mm
    half-diagonal).

    ``vertical_offset_mm`` moves the plate along ``body_up_axis`` before
    the local ICP/standoff stages re-conform it to the body surface at the
    new height -- e.g. if the shipped PPE file sits low (over the stomach)
    when it should cover the chest, pass a positive value to raise it.
    Positive = toward superior (up), negative = toward inferior (down),
    per ``body_up_axis``'s convention for this model/site. This is
    independent of the near/far path: it always applies, right before
    cropping, so the crop radius (``seating_margin_mm``) doesn't need to
    grow just because of a deliberate, user-requested repositioning.

    ``enforce_symmetry`` (default ``True``) runs a final symmetry-correction
    stage after standoff correction: corrects yaw about the body's vertical
    axis (rim-to-rim vector check), removes any lateral roll ICP's fit
    introduced (so the plate's up-axis has no lateral tilt), and recenters
    the plate's concave-face centroid on ``target_lateral_mm`` (default:
    ``None`` -> auto-derived from the body's own **spine groove** -- a real
    anatomical landmark (the indentation over the spinous processes,
    between the erector spinae muscle bulges), not a numerically-fit plane
    or a bounding-box center. An earlier version used a local
    mirror-symmetry-plane fit near the plate instead; found (directly, from
    user feedback on a rendered case) to disagree with the spine groove by
    ~1.1mm, having overfit to local front-side anatomy rather than the
    body's true structural midline. See ``_find_spine_groove_lateral_mm``
    for the full rationale and method). Set to ``False`` to reproduce the
    pre-symmetry-correction seating exactly (e.g. for comparing against
    older cases, or a PPE that is intentionally meant to sit off-center).

    ``accept_standoff_range_mm`` (default ``None`` = always fit) is a
    ``(lo, hi)`` band for the plate's minimum signed gap to the skin. If the
    PPE as loaded already sits inside it, seating (seed rotation, ICP,
    standoff and symmetry correction) is skipped entirely and the plate is
    returned untouched with ``fit_skipped=True``. Ignored when
    ``vertical_offset_mm`` is non-zero, since that is an explicit request to
    move the plate.
    """
    if not ppe_spec.is_rigid:
        raise NotImplementedError(
            f"seat_plate only supports rigid PPE (no RBF, no scale) -- "
            f"'{ppe_spec.key}' is not marked rigid"
        )

    body_surface = extract_skin_surface(model, site)
    standoff_floor = compute_standoff_floor(body_surface)
    resolved_standoff_mm = resolve_target_standoff(target_standoff_mm, standoff_floor)

    offset_points = body_surface.nodes + body_surface.vertex_normals * resolved_standoff_mm
    offset_normals = body_surface.vertex_normals  # unchanged by a small outward point offset

    plate_mesh = plate_mesh if plate_mesh is not None else ppe_spec.load()
    orient = compute_plate_orientation(plate_mesh)

    current_full = plate_mesh.nodes.copy()
    concave_ids = orient.concave_face_node_ids

    if accept_standoff_range_mm is not None and vertical_offset_mm == 0.0:
        existing = check_standoff(current_full[concave_ids], body_surface, accept_standoff_range_mm)
        if existing.is_acceptable:
            identity = np.eye(3)
            return SeatingResult(
                plate_orientation=orient,
                body_surface=body_surface,
                offset_target_points=offset_points,
                seed_rotation=identity,
                icp_result=ICPResult(
                    transformed_points=current_full[concave_ids],
                    rotation=identity,
                    translation=np.zeros(3),
                    iterations=0,
                    converged=True,
                    final_mean_dist_mm=existing.gap.mean_mm,
                ),
                transformed_plate_nodes=current_full,
                pre_icp_mean_dist_mm=existing.gap.mean_mm,
                standoff_floor=standoff_floor,
                resolved_target_standoff_mm=resolved_standoff_mm,
                final_gap=existing.gap,
                symmetry_twist_deg=0.0,
                symmetry_lateral_shift_mm=0.0,
                symmetry_yaw_deg=0.0,
                fit_skipped=True,
            )

    # Rim indices for the yaw-correction stage (_apply_symmetry_correction)
    # -- computed once, in the plate's own local/untransformed frame, since
    # they're just a fixed subset of concave_ids that any world-space copy
    # of the plate's node array can be indexed with later.
    local_frame = _orthonormal_frame(orient.concave_normal, orient.up_axis)
    left_rim_ids, right_rim_ids = _find_rim_indices(
        plate_mesh.nodes, concave_ids, local_frame[1], local_frame[2]
    )

    # --- stage 1: signature-seeded rotation, about the plate's centroid ---
    # Prefer the plate's own current position for the local body-normal
    # lookup when it's already roughly on the body (preserves the precise
    # result verified for an already-well-placed PPE); fall back to a
    # fixed anatomical anchor -- and a coarse translation, stage 2 -- only
    # when it is not, since that anchor was found to be ~65mm off the true
    # target and, combined with ICP's known aperture-problem sensitivity to
    # starting position (see core.registration.point_to_plane_icp), that
    # was enough to visibly degrade an already-good fit when applied
    # unconditionally (caught by this module's own regression tests, not
    # assumed safe).
    body_centroid = body_surface.nodes.mean(axis=0)
    dist_to_body = float(np.linalg.norm(orient.plate_centroid - body_centroid))
    is_far_away = dist_to_body > far_away_threshold_mm

    if is_far_away:
        anchor_point, anchor_normal = _default_anchor_point(body_surface, anterior_axis)
    else:
        anchor_point, anchor_normal = orient.plate_centroid, _local_body_normal(body_surface, orient.plate_centroid)

    seed_R = seed_rotation_from_signature(orient, anchor_normal, body_up_axis)
    seed_center = orient.plate_centroid
    current_full = (current_full - seed_center) @ seed_R + seed_center
    current_concave = current_full[concave_ids]

    # --- stage 2: coarse translation recovery (far-away PPE only) ---
    if is_far_away:
        current_full = _coarse_translate_to_anchor(current_full, concave_ids, anchor_point, anchor_normal)
        current_concave = current_full[concave_ids]

    # --- stage 2b: user-requested vertical adjustment ---
    # Moves the plate along the body's superior/inferior axis before the
    # local ICP/standoff stages re-conform it to the surface at the new
    # height -- this is how a user corrects "the plate is seated too low/
    # high" (e.g. sitting on the stomach instead of the chest) without
    # needing to touch the PPE file itself. Positive = toward superior
    # (up the torso, per `body_up_axis`), negative = toward inferior (down).
    if vertical_offset_mm != 0.0:
        current_full = current_full + vertical_offset_mm * body_up_axis
        current_concave = current_full[concave_ids]

    pre_icp_dist = _mean_nearest_dist(current_concave, offset_points)

    # --- stage 3: crop the target to a local neighborhood of the plate ---
    plate_radius = float(np.linalg.norm(current_concave - current_concave.mean(axis=0), axis=1).max())
    crop_radius = plate_radius + seating_margin_mm
    crop_idx = _crop_indices_near(offset_points, current_concave.mean(axis=0), crop_radius)
    local_target_points = offset_points[crop_idx]
    local_target_normals = offset_normals[crop_idx]

    # --- stage 4: point-to-plane ICP, fit on the concave surface, applied to the full mesh ---
    icp_result = point_to_plane_icp(current_concave, local_target_points, local_target_normals, icp_config)

    if vertical_offset_mm != 0.0:
        # Discard ICP's tangential translation, keeping only its rotation
        # and normal-direction correction. Verified necessary: ICP's own
        # tangential motion is large even in the well-seated baseline case
        # (~88mm out of ~90mm total, on real data) -- that's a symptom of
        # the aperture problem (see core.registration.point_to_plane_icp),
        # not a deliberate correction, and it will happily undo a
        # user-requested reposition (a real +100mm vertical_offset_mm was
        # measured collapsing to ~12mm of net movement before this clamp).
        # Only applied when the caller actually asked to move the plate --
        # leaving the well-tested no-offset path completely unchanged.
        pre_icp_normal = orient.concave_normal @ seed_R
        pre_icp_normal /= np.linalg.norm(pre_icp_normal)
        normal_translation = np.dot(icp_result.translation, pre_icp_normal) * pre_icp_normal
        current_full = current_full @ icp_result.rotation + normal_translation
    else:
        current_full = icp_result.apply(current_full)

    # --- stage 5: closed-form standoff correction (AGENTS.md 2.4 step 2) ---
    # rotate (not translate) the original concave normal through the seed
    # rotation and ICP stages, to get the plate's *current* body-facing
    # direction (the coarse translation in stage 2 has no rotation
    # component, so it doesn't affect this).
    current_concave_normal = orient.concave_normal @ seed_R @ icp_result.rotation
    current_concave_normal /= np.linalg.norm(current_concave_normal)
    current_full, final_gap = _apply_standoff_correction(
        current_full, concave_ids, current_concave_normal, body_surface, resolved_standoff_mm
    )

    # --- stage 6: bilateral symmetry correction (see _apply_symmetry_correction) ---
    symmetry_twist_deg = 0.0
    symmetry_lateral_shift_mm = 0.0
    symmetry_yaw_deg = 0.0
    if enforce_symmetry:
        current_up_axis = orient.up_axis @ seed_R @ icp_result.rotation
        current_up_axis /= np.linalg.norm(current_up_axis)

        if target_lateral_mm is None:
            # Spine-groove landmark (see _find_spine_groove_lateral_mm) --
            # a real anatomical feature, not a numerically-fit plane or
            # bbox-center. Restricted to roughly the plate's own vertical
            # extent (established by standoff correction above; unaffected
            # by the small yaw/de-twist/lateral corrections still to come,
            # since none of them meaningfully change height) so the groove
            # is measured at the height that actually matters, not averaged
            # over the whole torso. This target does not depend on the
            # plate's own lateral position, so (unlike an earlier version
            # that fit a local mirror-plane near the plate) it needs no
            # fixed-point iteration to stay self-consistent -- a single
            # pass is correct.
            plate_z = current_full[concave_ids] @ body_up_axis
            resolved_target_lateral_mm = _find_spine_groove_lateral_mm(
                body_surface, anterior_axis, body_up_axis, lateral_axis,
                z_band=(float(plate_z.min()), float(plate_z.max())),
            )
        else:
            resolved_target_lateral_mm = target_lateral_mm

        current_full, symmetry_yaw_deg, symmetry_twist_deg, symmetry_lateral_shift_mm, final_gap, _, _ = _apply_symmetry_correction(
            current_full,
            concave_ids,
            left_rim_ids,
            right_rim_ids,
            current_up_axis,
            current_concave_normal,
            body_up_axis,
            anterior_axis,
            lateral_axis,
            resolved_target_lateral_mm,
            body_surface,
            resolved_standoff_mm,
        )

    return SeatingResult(
        plate_orientation=orient,
        body_surface=body_surface,
        offset_target_points=offset_points,
        seed_rotation=seed_R,
        icp_result=icp_result,
        transformed_plate_nodes=current_full,
        pre_icp_mean_dist_mm=pre_icp_dist,
        standoff_floor=standoff_floor,
        resolved_target_standoff_mm=resolved_standoff_mm,
        final_gap=final_gap,
        symmetry_twist_deg=symmetry_twist_deg,
        symmetry_lateral_shift_mm=symmetry_lateral_shift_mm,
        symmetry_yaw_deg=symmetry_yaw_deg,
    )
