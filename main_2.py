"""
DeformationApp -- Stage 2 entry point: build the GUI-ready result for a
solved case.

Workflow (per AGENTS.md):

    main_1.py  -- define HBM model + PPE, seat it (ICP/CPD), write a .feb
    (you)      -- open the .feb in FEBio Studio, set/check boundary
                  conditions, solve it
    main_2.py  -- (this file) reads the solved case's outputs and builds
                  the single, self-contained "GUI case" file
                  app/viewer_app.py reads to visualize the plate pressing
                  into the chest

Run this *after* you've solved a case written by ``main_1.py`` (in FEBio
Studio, or ``febio4.exe -i <path>.feb``) -- the solve produces a plain-text
node-displacement logfile (``<case>_node_displacement.txt``, requested by
default in ``febio/assemble_case.py``'s Output/logfile block) alongside the
usual ``.xplt``; see that module's docstring for why we read the text log
instead of the ``.xplt``.

Edit the USER SETTINGS block below, then run:

    .venv\\Scripts\\python.exe main_2.py

Only ``CASE_NAME`` needs to change between runs -- the base filename
``main_1.py`` used (e.g. ``"F05_Standing_torso_armored_plate_preliminary"``,
its default auto-generated name). Everything main_2.py needs is found
automatically next to it in ``CASES_DIR`` by that same base name:
``<CASE_NAME>_node_displacement.txt`` (written by the FEBio solve),
``<CASE_NAME>_node_map.npz`` and ``<CASE_NAME>_plate_render.npz`` (both
written by ``main_1.py`` alongside the ``.feb``).
"""
from pathlib import Path

import numpy as np

from postprocess.gui_case import build_gui_case

ROOT = Path(__file__).resolve().parent

# ─── USER SETTINGS ──────────────────────────────────────────────────────────
# The base filename (no extension) main_1.py used for this case -- e.g. its
# printed "wrote: .../cases_generated/<CASE_NAME>.feb" line. main_2.py finds
# <CASE_NAME>_node_displacement.txt, _node_map.npz, and _plate_render.npz
# automatically from this one name.
CASE_NAME = "F05_Seated_torso_armored_plate_preliminary"

CASES_DIR = ROOT / "cases_generated"
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"1) reading solved outputs for '{CASE_NAME}' from {CASES_DIR}...")
    out_path = build_gui_case(CASE_NAME, CASES_DIR)

    with np.load(out_path) as data:
        n_steps = len(data["times"])
        n_nodes = len(data["node_ids"])
        max_u = float(np.linalg.norm(data["displacement"][-1], axis=1).max())
        final_time = float(data["final_time"])

    print(
        f"   {n_steps} solved steps, {n_nodes} torso nodes, "
        f"final_time={final_time:.4f}, max|u| at final step={max_u:.3f}mm"
    )
    print("wrote:", out_path, " size(MB):", out_path.stat().st_size / 1e6)
    print()
    print("Open the GUI (app/viewer_app.py) and select this HBM model / PPE")
    print("from the dropdowns to view it.")


if __name__ == "__main__":
    main()
