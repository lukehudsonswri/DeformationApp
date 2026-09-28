"""
DeformationApp -- Stage 1 entry point: seat the PPE and assemble a FEBio case.

Edit the USER SETTINGS block below, then run:

    .venv\\Scripts\\python.exe main_1.py

This seats the selected PPE against the selected HBM model/site and writes
a FEBio case to ``cases_generated/``. Open/edit/solve the ``.feb`` in FEBio
Studio (or ``febio4.exe``) yourself -- once you have a solved case, run
``main_2.py`` to project the resulting deformation onto the full HBM model
for viewing in the GUI (``app/viewer_app.py``).

See AGENTS.md for the full pipeline design and VERIFICATION.md for how
results are checked.
"""
from pathlib import Path

from febio.build_preliminary_case import build_case

# ─── USER SETTINGS ──────────────────────────────────────────────────────────
HBM_MODEL_KEY = "F05_Seated"   # folder name under HBM/ -- see config/hbm_models.py
SITE = "torso"                   # only "torso" exists today -- see config/sites.py
PPE_KEY = "armored_plate"        # see config/ppe.py for the registry

INDENTATION_MM = 8            # how far the PPE presses in past first contact
TARGET_STANDOFF_MM = 0.1         # rest gap for contact detection (AGENTS.md 2.4).
                                  # Must stay > 0; below the mesh's own sagitta
                                  # floor it gets raised automatically (logged).
INITIAL_CONTACT_MM = 0.5         # target travel on the FIRST solved step, to
                                  # register as engaged contact (AGENTS.md 2.4 /
                                  # febio/loadcurve.py). Deliberately separate
                                  # from TARGET_STANDOFF_MM -- do not shrink this
                                  # to match a tight standoff, or the first step
                                  # may be too small for the mesh to resolve as
                                  # real contact (this broke convergence once).

# Move the seated plate up/down the torso before it's fit to the body --
# e.g. if it lands too low (over the stomach) when it should cover the
# chest. Positive = superior (up), negative = inferior (down). 0 = leave
# the plate wherever its own file/previous fit already placed it.
VERTICAL_OFFSET_MM = 10.0

OUTPUT_PATH = None                # None = auto-named under cases_generated/
HBM_ROOT = Path("HBM")
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    build_case(
        hbm_model_key=HBM_MODEL_KEY,
        site_name=SITE,
        ppe_key=PPE_KEY,
        target_standoff_mm=TARGET_STANDOFF_MM,
        indentation_mm=INDENTATION_MM,
        initial_contact_mm=INITIAL_CONTACT_MM,
        vertical_offset_mm=VERTICAL_OFFSET_MM,
        output_path=OUTPUT_PATH,
        hbm_root=HBM_ROOT,
    )


if __name__ == "__main__":
    main()
