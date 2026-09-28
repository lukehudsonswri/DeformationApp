# DeformationApp

Seats a rigid PPE (currently: an armored torso plate) against a human body
model, writes a FEBio case, and (once you've solved it) projects the result
onto the full HBM model for viewing in the GUI. See `AGENTS.md` for the
full pipeline design and rationale, and `VERIFICATION.md` for how results
are checked.

## Setup (one-time)

This worktree shares its Python environment with the main repo checkout --
there is no separate `.venv` here. From this folder:

```powershell
cd C:\Users\lhudson\Desktop\Projects\GUARDS\DeformationApp-agent
C:\Users\lhudson\Desktop\Projects\GUARDS\DeformationApp\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

(If that `.venv` doesn't exist yet, create it once from the main repo:
`cd C:\Users\lhudson\Desktop\Projects\GUARDS\DeformationApp; python -m venv .venv`.)

**Every command below assumes you're using *that* `.venv`'s Python, not
whatever plain `python` resolves to on your PATH** (usually a different,
system-wide install without this project's packages -- running a bare
`python somefile.py` will fail with `ModuleNotFoundError` for PySide6,
numpy, etc.). Either:

- prefix every command with the full venv path, as shown throughout this
  README: `C:\...\DeformationApp\.venv\Scripts\python.exe <script>.py`, or
- activate the venv once per PowerShell window, after which plain `python`
  resolves correctly for the rest of that session:
  ```powershell
  C:\Users\lhudson\Desktop\Projects\GUARDS\DeformationApp\.venv\Scripts\Activate.ps1
  python viewer_app.py
  ```


## Run: three stages

### Stage 1 -- `main_1.py`: seat the PPE and write a FEBio case

```powershell
cd C:\Users\lhudson\Desktop\Projects\GUARDS\DeformationApp-agent
C:\Users\lhudson\Desktop\Projects\GUARDS\DeformationApp\.venv\Scripts\python.exe main_1.py
```

**Takes about 2-3 minutes** (no tests run): ~50-90s to seat the plate against
the body, ~60-65s to extract the torso skin+flesh mesh, then a few seconds
to write the `.feb`. Progress prints as it goes, e.g.:

```
1) seating 'armored_plate' against 'torso' on 'F05_Standing'...
   done in 51.4s, final gap min=0.1000mm
2) extracting torso volume (skin+flesh)...
   done in 65.2s: 47544 nodes, 11158 skin quads, 34980 flesh hexes
3) deriving load curve...
   total_travel=4.3500mm step_size=0.114943 time_steps=9
4) assembling .feb...
wrote: ...\cases_generated\F05_Standing_torso_armored_plate_preliminary.feb
wrote: ...\cases_generated\F05_Standing_torso_armored_plate_preliminary_node_map.npz
```

Open the written `.feb` in FEBio Studio to run the solve, or run it directly
with `febio4.exe -i <path>.feb`. This case solves to normal termination as
generated -- no manual editing needed (see `AGENTS.md` §2.4/§2.5 for the
load-curve bug that used to prevent this, now fixed). Contact
pressure/strain currently read lower than the gold-standard reference's own
values -- this is expected from the current whole-torso mesh's resolution
under the contact patch, not a sign of a modeling error; see
`VERIFICATION.md` §5.3.

The solve produces the usual `.xplt`, plus a plain-text
`<case>_node_displacement.txt` (requested automatically -- see
`febio/assemble_case.py`'s `Output/logfile` block, scoped to the torso's
own ~47.5k nodes only -- not the plate's, which the projection step never
reads -- so the text log stays around ~50-60MB rather than the ~780MB it
was before that scoping fix). Stage 2 reads that text file, not the
`.xplt` -- see below for why.

### Stage 2 -- `main_2.py`: build the GUI case file

Once you've solved the `.feb` (so `<case>_node_displacement.txt` exists next
to it):

```powershell
C:\Users\lhudson\Desktop\Projects\GUARDS\DeformationApp\.venv\Scripts\python.exe main_2.py
```

Edit only `CASE_NAME` at the top of `main_2.py` -- the base filename
`main_1.py` printed (e.g. `"F05_Standing_torso_armored_plate_preliminary"`).
Everything else (`<CASE_NAME>_node_displacement.txt`, `_node_map.npz`,
`_plate_render.npz`) is found automatically next to it in `cases_generated/`.

This reads every solved step's displacement (not just the final one), maps
it from the FEBio case's own local node numbering back onto the original
HBM node ids (via the `<case>_node_map.npz` sidecar `main_1.py` wrote),
combines it with the plate's rest geometry/motion (`_plate_render.npz`),
and writes one self-contained `<CASE_NAME>_gui_case.npz` -- the file
`app/viewer_app.py` scans `cases_generated/` for and reads directly. You
don't need to point the GUI at it manually; see below.

**Why not read the `.xplt` directly?** The installed `febio-python`
package's xplt reader cannot parse FEBio 4.12's plot file version (53) -- it
only recognizes up to version 52, and even patching that version check does
not correctly locate the (per-state-compressed) STATE section: a real,
unresolved bug/format-drift in that third-party library, not something in
our control. FEBio's own plain-text logfile output is a first-class,
documented, uncompressed alternative that sidesteps this entirely.

## What you can edit

Everything you'd normally change for Stage 1 lives in one block at the top
of `main_1.py`:

```python
HBM_MODEL_KEY = "F05_Standing"   # folder name under HBM/
SITE = "torso"                   # only "torso" exists today
PPE_KEY = "armored_plate"        # see config/ppe.py

INDENTATION_MM = 4.25            # how far the PPE presses in past first contact
TARGET_STANDOFF_MM = 0.1         # rest gap for contact detection
INITIAL_CONTACT_MM = 0.5         # target travel on the FIRST solved step (must
                                  # be large enough for the mesh to resolve as
                                  # real contact -- keep this separate from
                                  # TARGET_STANDOFF_MM, see AGENTS.md 2.4)

VERTICAL_OFFSET_MM = 0.0         # move the seated plate up/down the torso
                                  # before it's fit to the body -- e.g. if it
                                  # lands too low (over the stomach) when it
                                  # should cover the chest. + = superior (up),
                                  # - = inferior (down).

OUTPUT_PATH = None                # None = auto-named under cases_generated/
```

Edit the block, save, re-run `main_1.py`. Nothing else needs to change for
the common case of "try a different standoff/indentation" or "nudge the
plate's height." Stage 2's `main_2.py` has its own small settings block
(just the solved `.feb` path).

### Adding a new HBM model

Drop a folder of `.k`/`.dyn` files anywhere and use
`config.hbm_normalize.build_hbm_model_from_folder(...)` to parse it down to
plain `HBM/<name>/{Nodes.k,Elements.k}` (elements+nodes only, no materials) --
see that function's docstring. Point `HBM_MODEL_KEY` at the resulting folder
name.

### Adding a new PPE

Register it in `config/ppe.py`'s `_PPE_REGISTRY` (a mesh path + whether it's
rigid). `.k`/`.inp` are supported today; `.stl` raises `NotImplementedError`
(tetrahedralization not yet built).

## Stage 3 -- the GUI: pick an HBM model + PPE, scrub the solved deformation

```powershell
cd C:\Users\lhudson\Desktop\Projects\GUARDS\DeformationApp-agent
C:\Users\lhudson\Desktop\Projects\GUARDS\DeformationApp\.venv\Scripts\python.exe viewer_app.py
```

`viewer_app.py` at the repo root is a thin launcher (same convention as
`main_1.py`/`main_2.py`) for the real implementation in `app/viewer_app.py`.
It opens a PySide6 + PyVista window with two dropdowns at the top ("Human
Body Model", "PPE"). It scans `cases_generated/*_gui_case.npz` (written by
`main_2.py`) and offers every HBM-model/PPE combination that has one --
there's nothing to point it at manually. "Refresh case list" re-scans
`cases_generated/` if you've just finished a `main_2.py` run without
restarting the GUI.

Once a case is picked, the slider scrubs through that case's real solved
FEBio steps (linearly interpolated between them, not a synthetic curve).
The window has three views, all sharing one slider:

- **Main view** (the whole window): the torso context body + target skin
  region, plain-colored geometry only -- freely rotate/pan/zoom to inspect
  the deforming region without clutter.
- **Top-left inset**: the same target region colored by displacement
  magnitude (fixed color scale, frontal camera), with a scalar bar.
- **Bottom-left inset**: the target region plus the rigid PPE approaching
  it, from an oblique angle chosen specifically so the PPE's motion into
  the body is visible (a straight-on view would make that motion nearly
  invisible on screen).

Both insets are separate embedded viewports in the same render window (not
floating windows) and are independently rotatable/zoomable, just like the
main view.

### Export Current State

The "Export Current State" button (top toolbar) exports whatever
deformation the slider is currently showing -- 20% load, 100% load,
anywhere in between -- back into real mesh files, in two phases (shown as
they complete in the status bar):

1. **"Meshes Exported"** -- three small, standalone LS-DYNA `.k` files
   under `cases_generated/<hbm_model>_<ppe>_export_<pct>pct/`:
   `torso_skin.k`, `torso_flesh.k` (both deformed to the current slider
   position), and `ppe.k` (the rigid PPE at its corresponding position).
2. **"FE Model Built with PPE"** -- one more file in that same folder,
   `<hbm_model>_with_ppe.k`: the *entire* HBM model (every other node and
   part completely untouched) with the torso's coordinates patched to the
   current deformed state and the PPE added as a brand-new part. This is
   the one to open if you want the deformed torso back in full anatomical
   context, ready for further FE work.

This reads the full HBM deck fresh each time (`postprocess/export_state.py`),
so it takes real time -- expect roughly 1-3 minutes depending on model size,
not an instant click. The button is disabled while it runs, and a progress
bar (bottom-left) tracks both phases combined as one overall 0-100%.

If you'd rather launch it from a Python one-liner (e.g. embedding it
elsewhere): `from app import viewer_app; viewer_app.main()`.

## Running the test suite

```powershell
C:\Users\lhudson\Desktop\Projects\GUARDS\DeformationApp\.venv\Scripts\python.exe -m pytest tests/
```

This takes **~20-25 minutes**, not 2-3 -- most of the suite exercises the
real pipeline against real production data (the real 619k-node plate mesh,
the real HBM deck) rather than mocks, and several test files each pay that
cost independently (no cross-file cache). You don't need to run this for
routine use of `main_1.py`/`main_2.py`; it exists to catch regressions, not
to produce output you need.

## Repository layout

- `main_1.py`, `main_2.py`, `viewer_app.py` -- the three entry points (see
  "Run: three stages" above).
- `config/` -- HBM model registry, PPE registry, body-site pid tables, FEBio
  solver discovery.
- `core/` -- vendored mesh I/O, rigid registration primitives (point-to-plane
  ICP, Kabsch, PCA pre-align, rigid CPD), LS-DYNA parsing bridge, mesh repair.
- `fitting/` -- the actual seating pipeline: shape signature, per-model
  anterior/up/lateral axis derivation from skeletal landmarks
  (`body_axes.py`), body surface/volume extraction, point-to-plane ICP
  seating, standoff correction.
- `febio/` -- load-curve derivation and FEBio case assembly.
- `remesh/` -- the FEBio-acceptance smoke test gate (`febio_smoke_test.py`):
  every mesh this project produces must pass a minimal FEBio solve attempt
  before being considered usable, not just this project's own quality checks.
- `postprocess/` -- solved-case node-displacement log parsing, projection
  back onto original HBM node ids, the combined GUI-case builder
  (`main_2.py`), and the "Export Current State" mesh/FE-model writer.
- `app/` -- the PySide6/PyVista GUI viewer (dropdown-driven, see "Stage 3"
  above).
- `HBM/`, `PPE/` -- model and PPE data (gitignored; large binary files).
- `febio_bin/` -- packaged FEBio4 solver (gitignored).
- `tests/` -- regression suite, mostly run against real data.
- `cases/` -- early reference case(s), kept for historical comparison (see
  `VERIFICATION.md` §5.2/§7).
- `cases_generated/` -- output of `main_1.py`/`main_2.py` (gitignored).
- `cache/` -- per-HBM-model parsed-body cache the GUI uses to skip
  re-parsing the ~200MB `Elements.k` deck on every launch (gitignored).

See `AGENTS.md` for the detailed design/rationale behind every piece above,
and `VERIFICATION.md` for acceptance criteria and known caveats in the
existing example case under `cases/`.
