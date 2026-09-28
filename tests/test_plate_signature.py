"""Tests for fitting.plate_signature -- the rigid-plate concave-face and
top/bottom shape-signature detector (AGENTS.md Goal 2 / section 2.2).
"""
import numpy as np
import pytest

from core.mesh.base import ElementType, VolumeMesh
from fitting.plate_signature import compute_plate_orientation


def _build_paraboloid_shell(nx=11, ny=11, half_extent=5.0, curvature=0.02, thickness=0.5):
    """A single-hex-layer curved shell shaped like a shallow paraboloid bowl,
    z = curvature * (x^2 + y^2), extruded by ``thickness`` along +z.

    The *top* face (larger z) is concave: its own outward normal (+z-ish)
    sees the rim higher than the centre (a bowl, opening upward). The
    *bottom* face (smaller z) is convex by the same construction (a dome,
    bulging downward away from centre). This gives ground truth to check
    ``compute_plate_orientation`` against, independent of the large real
    plate mesh.
    """
    xs = np.linspace(-half_extent, half_extent, nx)
    ys = np.linspace(-half_extent, half_extent, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    r2 = xx ** 2 + yy ** 2
    z_bottom = curvature * r2
    z_top = z_bottom + thickness

    bottom_nodes = np.stack([xx.ravel(), yy.ravel(), z_bottom.ravel()], axis=1)
    top_nodes = np.stack([xx.ravel(), yy.ravel(), z_top.ravel()], axis=1)
    nodes = np.concatenate([bottom_nodes, top_nodes], axis=0)
    n_grid = nx * ny

    def idx(i, j):
        return i * ny + j

    elements = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            b00, b10, b11, b01 = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            elements.append([b00, b10, b11, b01, b00 + n_grid, b10 + n_grid, b11 + n_grid, b01 + n_grid])

    elements = np.array(elements, dtype=np.int64)
    return VolumeMesh(
        nodes=nodes,
        element_groups={ElementType.HEX8: elements},
        element_type=ElementType.HEX8,
        name="synthetic_paraboloid_shell",
    )


def test_synthetic_paraboloid_top_face_is_concave():
    mesh = _build_paraboloid_shell()
    orient = compute_plate_orientation(mesh)

    # thickness axis should be ~z (the shell is thin along z, wide in x/y)
    assert abs(abs(orient.thickness_axis[2]) - 1.0) < 0.05

    # concave face is the top layer (z close to `thickness`): its outward
    # normal should point mostly +z, matching the "bowl opens upward" setup.
    assert orient.concave_normal[2] > 0.9
    assert orient.convex_normal[2] < -0.9
    assert orient.concavity_score > 0

    concave_z = mesh.nodes[orient.concave_face_node_ids][:, 2]
    convex_z = mesh.nodes[orient.convex_face_node_ids][:, 2]
    assert concave_z.mean() > convex_z.mean()  # concave = top (larger z) layer


def test_synthetic_paraboloid_orientation_is_curvature_direction_invariant():
    """Flip the shell upside-down (mirror z): the *bottom* layer becomes the
    concave one now, and the detector must follow the surface, not an
    absolute z convention.
    """
    mesh = _build_paraboloid_shell()
    mesh.nodes[:, 2] *= -1.0
    orient = compute_plate_orientation(mesh)

    assert orient.concave_normal[2] < -0.9
    concave_z = mesh.nodes[orient.concave_face_node_ids][:, 2]
    convex_z = mesh.nodes[orient.convex_face_node_ids][:, 2]
    assert concave_z.mean() < convex_z.mean()


def test_synthetic_flat_plate_has_no_reliable_taper():
    """A flat rectangular plate (no trapezoid taper) should report a taper
    ratio close to 1 (no meaningful narrowing at either end).
    """
    mesh = _build_paraboloid_shell(curvature=0.0)
    orient = compute_plate_orientation(mesh)
    assert orient.taper_ratio > 0.95


@pytest.fixture(scope="module")
def real_plate_mesh():
    from config.ppe import get_ppe

    spec = get_ppe("armored_plate")
    if not spec.mesh_path.is_file():
        pytest.skip(f"PPE mesh not present at {spec.mesh_path}")
    return spec.load()


def test_real_plate_concave_convex_faces_are_comparable_size(real_plate_mesh):
    orient = compute_plate_orientation(real_plate_mesh)
    n_concave = len(orient.concave_face_node_ids)
    n_convex = len(orient.convex_face_node_ids)
    # Two large faces of one thin plate should be similarly sized surface
    # patches (within ~5%), not e.g. one face vs. a sliver of rim faces.
    assert abs(n_concave - n_convex) / max(n_concave, n_convex) < 0.05


def test_real_plate_has_a_genuine_trapezoid_taper(real_plate_mesh):
    """Regression value from direct inspection (see session notes): the
    plate narrows from ~240mm to ~160mm width at one end -- a real,
    pronounced taper, not measurement noise.
    """
    orient = compute_plate_orientation(real_plate_mesh)
    assert orient.taper_ratio == pytest.approx(0.658, abs=0.01)


def test_real_plate_concave_face_points_toward_current_torso_position(real_plate_mesh):
    """The shipped PPE/plate.inp is already roughly seated against
    HBM/F05_Standing (verified separately: ~151mm centroid distance,
    overlapping bounding boxes) -- its concave face should already point
    toward the torso, i.e. no flip is needed for this specific pairing.
    """
    torso_centroid = np.array([-19.4628245, 0.0931017080, -986.914772])
    orient = compute_plate_orientation(real_plate_mesh)
    concave_centroid = real_plate_mesh.nodes[orient.concave_face_node_ids].mean(axis=0)
    direction_to_torso = torso_centroid - concave_centroid
    direction_to_torso /= np.linalg.norm(direction_to_torso)
    assert np.dot(orient.concave_normal, direction_to_torso) > 0.5
