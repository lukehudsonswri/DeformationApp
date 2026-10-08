"""Read a FEBio ``<rigid_body_data>`` logfile to recover how a rigid plate
actually moved (for plates driven by straps/springs, whose motion is not
prescribed in the ``.feb``).

Same ``*Step`` / ``*Time`` / ``*Data`` layout as ``postprocess.node_log``,
but rows may be comma-separated (``delim=","``) or whitespace-separated, and
the first column is the rigid body id rather than a node id. The plate in
this pipeline has its rotations locked, so its position alone describes its
motion.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def read_rigid_body_trajectory(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Logged ``(times, positions)`` of the rigid body's reference point.

    Args:
        path: rigid-body log with ``x``, ``y`` and ``z`` among its ``*Data`` fields.

    Returns:
        times: (K,) solver times, in logged order.
        positions: (K, 3) reference-point positions in mm.

    Raises:
        ValueError: if the log has no rows or no ``x;y;z`` columns.
    """
    fields: list = []
    time = 0.0
    times: list = []
    positions: list = []
    for raw in Path(path).read_text(encoding="ISO-8859-1").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("*Time"):
            time = float(line.split("=", 1)[1])
        elif line.startswith("*Data"):
            fields = [f.strip() for f in line.split("=", 1)[1].split(";")]
        elif not line.startswith("*"):
            if not {"x", "y", "z"} <= set(fields):
                raise ValueError(f"{path}: rigid-body log has no x;y;z columns (has {fields})")
            row = [float(v) for v in line.replace(",", " ").split()]
            positions.append([row[1 + fields.index(c)] for c in ("x", "y", "z")])
            times.append(time)
    if not positions:
        raise ValueError(f"no rigid-body rows found in {path}")
    return np.asarray(times, dtype=np.float64), np.asarray(positions, dtype=np.float64)


def read_rigid_body_travel(path: Path, direction: np.ndarray) -> float:
    """Signed distance the rigid body moved along ``direction``, first to last logged row.

    Args:
        path: rigid-body log (see ``read_rigid_body_trajectory``).
        direction: (3,) unit vector to project the net movement onto.

    Returns:
        Final minus first position, projected on ``direction`` (mm).
    """
    _, positions = read_rigid_body_trajectory(path)
    return float(np.dot(positions[-1] - positions[0], np.asarray(direction, dtype=np.float64)))
