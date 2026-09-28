"""Parse FEBio's plain-text ``<Output><logfile><node_data>`` output.

We use this instead of the binary ``.xplt`` plot file: the installed
``febio-python`` package's xplt reader cannot correctly parse FEBio 4.12's
plot file version (53) -- it only recognizes up to version 52, and even
patching that version check does not correctly locate the (per-state
compressed) STATE section; a real, unresolved bug/format-drift in that
third-party library. FEBio's own plain-text logfile output is a first-class,
documented, uncompressed alternative that sidesteps this entirely --
``febio/assemble_case.py`` requests one (``data="x;y;z;ux;uy;uz"``) on every
case it writes.

File format (confirmed empirically against a real solved case -- see
``tests/test_node_log.py``)::

    *Step  = 0
    *Time  = 0
    *Data  = x;y;z;ux;uy;uz
    1 0 0 0 0 0 0
    2 1 0 0 0 0 0
    ...
    *Step  = 1
    *Time  = 1
    *Data  = x;y;z;ux;uy;uz
    1 0 0 0 0 0 0
    ...

One ``*Step``/``*Time``/``*Data`` header followed by one row per node
(``<node_id> <values...>``, whitespace-separated, node id is FEBio's own
1-based numbering from the ``.feb`` that was solved) repeated per solved
step, in step order.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class NodeLogStep:
    step: int
    time: float
    fields: List[str]  # e.g. ["x", "y", "z", "ux", "uy", "uz"]
    node_ids: np.ndarray  # (N,) int64, FEBio's own 1-based node ids
    values: np.ndarray  # (N, len(fields)) float64


def read_node_log(path: Path) -> List[NodeLogStep]:
    """Parse every ``*Step`` block in a FEBio ``node_data`` logfile."""
    path = Path(path)
    text = path.read_text(encoding="ISO-8859-1")

    steps: List[NodeLogStep] = []
    step_num = None
    step_time = None
    fields: List[str] = []
    rows: List[List[float]] = []

    def _flush():
        if step_num is None:
            return
        arr = np.asarray(rows, dtype=np.float64)
        steps.append(
            NodeLogStep(
                step=step_num,
                time=step_time,
                fields=list(fields),
                node_ids=arr[:, 0].astype(np.int64) if len(arr) else np.zeros(0, dtype=np.int64),
                values=arr[:, 1:] if len(arr) else np.zeros((0, len(fields)), dtype=np.float64),
            )
        )

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("*Step"):
            _flush()
            step_num = int(line.split("=", 1)[1].strip())
            rows = []
        elif line.startswith("*Time"):
            step_time = float(line.split("=", 1)[1].strip())
        elif line.startswith("*Data"):
            fields = [f.strip() for f in line.split("=", 1)[1].strip().split(";")]
        else:
            rows.append([float(x) for x in line.split()])

    _flush()
    if not steps:
        raise ValueError(f"no '*Step' blocks found in node log: {path}")
    return steps


def last_step(path: Path) -> NodeLogStep:
    """The final (fully converged, or overshoot-settled) solved step."""
    return read_node_log(path)[-1]
