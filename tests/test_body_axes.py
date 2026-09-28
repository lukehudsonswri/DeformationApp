"""Tests for fitting.body_axes -- deriving per-model anterior/up/lateral
axes from real anatomical landmarks (sternum vs. thoracic spine), instead
of assuming a fixed global-axis convention.
"""
from pathlib import Path

import numpy as np
import pytest

from config.hbm_models import HbmModel
from fitting.body_axes import derive_torso_axes


def _write_synthetic_model(tmp_path: Path, *, anterior_dir, up_dir, flip_element_order=False) -> HbmModel:
    """A tiny synthetic model with just the landmark elements
    derive_torso_axes needs: one sternum node (shell), one spine node per
    thoracic level (shell, T1..T12), placed so the resulting anterior/up
    axes are known exactly, to check the math independent of real data.

    T1..T12 are placed evenly along a straight line (no spinal curvature
    in this synthetic fixture) symmetric about the origin, so their mean
    (the "spine centroid" derive_torso_axes computes) lands exactly on the
    origin -- keeping the expected anterior/up vectors exact, not off by a
    small residual from an uneven synthetic placement.
    """
    anterior_dir = np.asarray(anterior_dir, dtype=np.float64)
    anterior_dir /= np.linalg.norm(anterior_dir)
    up_dir = np.asarray(up_dir, dtype=np.float64)
    up_dir /= np.linalg.norm(up_dir)

    # Body origin at (0,0,0); sternum sits 100mm anterior of the spine.
    sternum_xyz = anterior_dir * 100.0

    t_pids = list(range(3000021, 3000055, 3))  # T1..T12, 12 levels
    n_levels = len(t_pids)
    # Evenly spaced from t=+1 (T1, superior-most) down to t=-1 (T12,
    # inferior-most), symmetric about the origin -- fracs sum to exactly 0.
    fracs = np.linspace(1.0, -1.0, n_levels)
    pid_xyz = {pid: up_dir * 100.0 * frac for pid, frac in zip(t_pids, fracs)}

    nodes = {1: sternum_xyz}
    pid_to_node_id = {}
    for i, pid in enumerate(t_pids):
        node_id = 2 + i
        nodes[node_id] = pid_xyz[pid]
        pid_to_node_id[pid] = node_id

    nodes_path = tmp_path / "Nodes.k"
    with open(nodes_path, "w") as f:
        f.write("*KEYWORD\n*NODE\n")
        for nid, xyz in nodes.items():
            f.write(f"{nid},{float(xyz[0])!r},{float(xyz[1])!r},{float(xyz[2])!r},0,0\n")
        f.write("*END\n")

    elements_path = tmp_path / "Elements.k"
    element_lines = [
        # sternum: one degenerate shell quad, all corners at the same node (fine -- only node ids matter here)
        "1,2000048,1,1,1,1",
    ]
    for i, pid in enumerate(t_pids):
        node_id = pid_to_node_id[pid]
        element_lines.append(f"{i+2},{pid},{node_id},{node_id},{node_id},{node_id}")
    if flip_element_order:
        element_lines = element_lines[::-1]

    with open(elements_path, "w") as f:
        f.write("*KEYWORD\n*ELEMENT_SHELL\n")
        f.write("\n".join(element_lines) + "\n")
        f.write("*END\n")

    return HbmModel(
        key="SyntheticModel", family=None, posture=None,
        folder=tmp_path, nodes_path=nodes_path, elements_path=elements_path,
    )


def test_derives_correct_anterior_and_up_for_standing_like_convention(tmp_path):
    # F05_Standing-like convention: anterior=+X, up=-Z
    model = _write_synthetic_model(tmp_path, anterior_dir=[1, 0, 0], up_dir=[0, 0, -1])
    axes = derive_torso_axes(model)

    assert np.allclose(axes.anterior, [1, 0, 0], atol=1e-6)
    assert np.allclose(axes.up, [0, 0, -1], atol=1e-6)
    assert np.allclose(axes.lateral, np.cross([1, 0, 0], [0, 0, -1]), atol=1e-6)


def test_derives_flipped_axes_for_a_different_convention(tmp_path):
    """The actual bug this module fixes: a model whose real anterior/up
    directions are simply DIFFERENT from F05_Standing's -- must be
    detected as such, not silently forced to the same (wrong) answer.
    """
    model = _write_synthetic_model(tmp_path, anterior_dir=[-1, 0, 0], up_dir=[0, 0, 1])
    axes = derive_torso_axes(model)

    assert np.allclose(axes.anterior, [-1, 0, 0], atol=1e-6)
    assert np.allclose(axes.up, [0, 0, 1], atol=1e-6)
    # anterior must NOT match the "standard" +X convention
    assert np.dot(axes.anterior, [1, 0, 0]) < 0


def test_lateral_is_unit_and_orthogonal_to_anterior_and_up(tmp_path):
    model = _write_synthetic_model(tmp_path, anterior_dir=[0.6, 0, 0.8], up_dir=[0, 0, -1])
    axes = derive_torso_axes(model)

    assert np.isclose(np.linalg.norm(axes.lateral), 1.0)
    assert np.isclose(np.dot(axes.lateral, axes.anterior), 0.0, atol=1e-9)
    assert np.isclose(np.dot(axes.lateral, axes.up), 0.0, atol=1e-9)


def test_element_order_in_file_does_not_matter(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    model_a = _write_synthetic_model(tmp_path / "a", anterior_dir=[1, 0, 0], up_dir=[0, 0, -1])
    model_b = _write_synthetic_model(
        tmp_path / "b", anterior_dir=[1, 0, 0], up_dir=[0, 0, -1], flip_element_order=True
    )
    axes_a = derive_torso_axes(model_a)
    axes_b = derive_torso_axes(model_b)
    assert np.allclose(axes_a.anterior, axes_b.anterior)
    assert np.allclose(axes_a.up, axes_b.up)


def test_missing_landmark_pids_raises_clear_error(tmp_path):
    nodes_path = tmp_path / "Nodes.k"
    nodes_path.write_text("*KEYWORD\n*NODE\n1,0,0,0,0,0\n*END\n")
    elements_path = tmp_path / "Elements.k"
    # no sternum/vertebra pids at all -- e.g. a non-I-PREDICT model
    elements_path.write_text("*KEYWORD\n*ELEMENT_SHELL\n1,9999999,1,1,1,1\n*END\n")
    model = HbmModel(
        key="NoSkeleton", family=None, posture=None,
        folder=tmp_path, nodes_path=nodes_path, elements_path=elements_path,
    )
    with pytest.raises(ValueError, match="landmark group"):
        derive_torso_axes(model)


def test_cache_dir_writes_and_is_reused_without_rescanning(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model = _write_synthetic_model(model_dir, anterior_dir=[1, 0, 0], up_dir=[0, 0, -1])
    cache_dir = tmp_path / "cache"

    axes_first = derive_torso_axes(model, cache_dir=cache_dir)
    cache_file = cache_dir / "body_axes_cache_SyntheticModel.json"
    assert cache_file.exists()

    # Corrupt the source files so a real recompute would fail loudly --
    # proves the second call is served entirely from the cache.
    model.elements_path.write_text("*KEYWORD\n*END\n")
    axes_second = derive_torso_axes(model, cache_dir=cache_dir)
    assert np.allclose(axes_first.anterior, axes_second.anterior)
    assert np.allclose(axes_first.up, axes_second.up)
    assert np.allclose(axes_first.lateral, axes_second.lateral)


def test_force_recompute_bypasses_cache(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model = _write_synthetic_model(model_dir, anterior_dir=[1, 0, 0], up_dir=[0, 0, -1])
    cache_dir = tmp_path / "cache"

    derive_torso_axes(model, cache_dir=cache_dir)

    # Rewrite the model with a different (still valid) convention, then
    # force a recompute -- the cache must not mask the change.
    model2 = _write_synthetic_model(model_dir, anterior_dir=[0, 1, 0], up_dir=[0, 0, -1])
    axes = derive_torso_axes(model2, cache_dir=cache_dir, force_recompute=True)
    assert np.allclose(axes.anterior, [0, 1, 0], atol=1e-6)


def test_stale_cache_version_is_ignored(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model = _write_synthetic_model(model_dir, anterior_dir=[1, 0, 0], up_dir=[0, 0, -1])
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "body_axes_cache_SyntheticModel.json").write_text(
        '{"version": 0, "anterior": [0, 0, 1], "up": [1, 0, 0], "lateral": [0, 1, 0]}'
    )

    axes = derive_torso_axes(model, cache_dir=cache_dir)
    assert np.allclose(axes.anterior, [1, 0, 0], atol=1e-6)

