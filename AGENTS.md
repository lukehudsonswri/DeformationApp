# AGENTS.md — DeformationApp Rebuild Plan

Plan of record for rebuilding `DeformationApp` into a single, maintained,
GUI-driven pipeline for **(human body model × body site × PPE)** contact
simulations in FEBio.

This document is the authoritative task list. `VERIFICATION.md` holds the
acceptance criteria that every simulation produced by this codebase must be
checked against.

---

## 0. Context and current state

### What exists today

| Area | State |
|---|---|
| `app/` | Three near-duplicate viewer stacks (`viewer_app.py`, `_f05`, `_m50`), three `hbm_reader*` variants, two `plate_model*` variants |
| `pipeline/` | 12 single-purpose CLI scripts, reasonable logic, no orchestration, hardcoded region/plate assumptions |
| `main.py` / `main_f05.py` / `main_m50.py` | Three entry points that differ only by which viewer they import |
| Repo size | **1.48 GiB** of loose git objects; `.fsm` (70 MB each), `.inp` (57–61 MB), `.npz`, `hbm.dyn` (44 MB) are all committed |
| Root directory | Contains ~420 MB of stray model/output files (`NewHex_close.xplt`, `I-PREDICT_v0.12_*.k`, `_skipped.face`, `lspost.*`) |

### The workflow this codebase supports

The pipeline is **not** an end-to-end FEA runner and is not intended to become
one. It is a preprocessor + postprocessor wrapped around a manual solve:

```
PPE file ──► detect site ──┬──► orient ──► CPD fit ──► seat to standoff ──┐
                           │                                              │
HBM decks ──► extract site ┴──► decimate/remesh ──► repair ───────────────┤
                                                                          ▼
                                                                     .feb case
                                                                          │
                                                          YOU solve in FEBio Studio
                                                                          │
                                          GUI viewer ◄── metrics ◄── .xplt results
```

The PPE drives site selection (§1.3), which is what tells the HBM side which
flesh + skin parts to extract.

Goal 1–3 automate everything on either side of the manual solve, and remove
every remaining hand-editing step that can be derived from data.

### External codebases we draw on

| Source | Path | What we take |
|---|---|---|
| **Morphing** (`fe_personalization`) | `C:\Users\lhudson\Documents\Morphing` | CPD, PCA pre-alignment, RBF/TPS morphing, multi-format mesh IO, gear-fitting phases |
| **Sparse** | `C:\Users\lhudson\Desktop\Projects\ForSarah\Sparse` | Negative-Jacobian / shell-distortion repair, penetration repair, solid↔shell connectivity |
| **Plate_Skin_Deformation** | `C:\Users\lhudson\Desktop\Projects\GUARDS\Plate_Skin_Deformation` | Gold-standard solved case; `AGENTS_STRAIN.md` acceptance bands; structured remesh + projection |
| **GUARDS_Software** | `C:\Users\lhudson\Desktop\Projects\GUARDS_Software` | Client LS-DYNA `KeywordProcessor` (already bridged) |

**Vendoring policy.** Morphing and Sparse are separate projects, not installable
dependencies. Required modules are **copied into this repo** under `core/` with a
provenance header naming the source file, original commit/date, and every local
change. We do not import across project directories at runtime (the existing
`GUARDS_SOFTWARE_ROOT` absolute-path bridge is the one exception, and it gets an
env-var override).

---

## Goal 1 — Configuration-driven model, site and PPE selection

**Outcome:** one entry point. The user picks a human body model and a PPE file;
the pipeline **infers the body site from the PPE geometry** and derives
everything downstream. `main_f05.py` and `main_m50.py` cease to exist.

### 1.1 HBM model registry — `config/hbm_models.py` — **implemented, revised**

**Layout, per explicit instruction:** `HBM/` contains one self-contained
subfolder per model (e.g. `HBM/F05_Standing/`), each with exactly two files —
`Nodes.k` and `Elements.k` — geometry only (node coordinates; element id +
part id + connectivity). No shared properties deck, no cross-model file
sharing, nothing implicit. `main.py` names which model folder to use; there
is no registry-side notion of "family/posture combinations that exist" to
maintain.

`discover_hbm_models(hbm_root) -> Dict[str, HbmModel]` scans `hbm_root` for
subfolders containing both files; `HbmModel` carries `key` (the folder
name), best-effort `family`/`posture` (parsed from the folder name, for
GUI display only — an unrecognized name is still a fully usable model),
and `nodes_path`/`elements_path`.

> **Revised (2026-09-23):** a raw-dropped-in folder (e.g. a client export
> named `HBM/F05_Seated/` holding `I-PREDICT_v1.0_F05_Nodes.k` /
> `I-PREDICT_v1.0_Morphing_F05_Elements.k`, not the clean `Nodes.k`/
> `Elements.k` this module's own writer produces) used to raise a `KeyError`
> — real bug, reported directly ("there is an F05_Seated directory..." /
> "it searches for any files in there that have 'Elements' or 'Element' or
> 'Nodes' or 'Node' in the directory"). `discover_hbm_models` now tries the
> exact clean filenames first (fast path, unchanged for every
> already-normalized model), then falls back **independently per file** to
> a case-insensitive substring search (`_find_by_substring`) for
> `"node"`/`"element"` among `.k`/`.dyn` files in that folder, picking the
> largest match if more than one exists. No separate normalization step is
> needed for this to work: `core.lsdyna.bridge.iter_wanted_cards` (used
> everywhere else in this pipeline) already parses the client's raw,
> fixed-width export format directly — confirmed on `F05_Seated`'s own raw
> files, including correctly skipping `*ELEMENT_BEAM_ORIENTATION` cards and
> matching `F05_Standing`'s known-good Thorax_Flesh/Thorax_Skin element
> counts (34,980 / 11,158) exactly.

**Building that clean layout from the client's original, messier export is
a separate, one-time, on-demand step** — `config/hbm_normalize.py`, backed
by `config/_hbm_raw_source.py` (internal-only). **Models are built one at a
time, as needed** — there is no "build every model the source tree happens
to contain" batch mode; that was tried and explicitly rejected.

Two build entry points, for two different situations:

1. **`build_model_by_key(key, source_hbm_root, out_hbm_root)`** — for a
   model already present in the client's *existing* `HBM/` export
   (`config._hbm_raw_source` has two hardcoded layout patterns
   reverse-engineered from that specific tree: a shared-properties-deck
   `01_Model` pattern, and a `*_{Bone,Skin}_Nodes.k` standalone-tree
   pattern). Raises `KeyError` for anything that doesn't match one of
   those two patterns.
2. **`build_hbm_model_from_folder(key, source_dir, out_hbm_root)`** — **the
   general answer to "if I drop in a folder with all the .k files and
   cards for a model, will it get stripped down automatically."** No
   naming convention required at all: every `.k`/`.dyn` file directly
   under `source_dir` (recursively) is scanned for `*NODE` /
   `*ELEMENT_SOLID` / `*ELEMENT_SHELL` cards, regardless of filename —
   `*INCLUDE` lines are ignored rather than resolved, since an included
   sub-file is expected to already be sitting in the same dropped-in
   folder and gets picked up directly. Family/posture are parsed
   best-effort from `key` for GUI display, same as the runtime registry.
   **This is the one that should be used for any genuinely new model.**

   Validates for a real failure mode rather than silently guessing: the
   same node id appearing twice with *different* coordinates across
   scanned files raises `ValueError` (ambiguous source data); the same
   node id with *identical* coordinates is silently deduplicated (a
   legitimate shared boundary node between two decks); any duplicate
   element id at all raises `ValueError` (never expected/benign for
   elements, unlike nodes).

   **Verified against real data, not just synthetic tests:** run directly
   on the client's actual messy `M50_Standing_Skin/` folder (6 files,
   inconsistent naming including a genuine `Bone_Element.k` singular/plural
   typo — see below) with zero knowledge of that folder's naming
   convention, it reproduced the exact same Thorax_Flesh-absent /
   Thorax_Skin-present counts (0 and 11,158 respectively) already
   independently confirmed via the hardcoded path and the raw client
   parser. See `tests/test_hbm_normalize.py`.

If a new HBM model is ever dropped into a folder, `build_hbm_model_from_folder`
extracts exactly that model's geometry into its own `HBM/<name>/` folder,
nothing more — no advance knowledge of its internal file layout needed.

**Only `F05_Standing` is built today** (2,435,778 solid + 559,351 shell
elements; confirmed exactly 34,980 Thorax_Flesh (pid 2000500) solids and
11,158 Thorax_Skin (pid 2000501) shells, matching independently-known counts
from this project's case data). Its normalized files are ~211 MB
(`Elements.k`) + ~56 MB (`Nodes.k`), gitignored like every other large
model/case artifact.

**What normalization keeps and drops, per explicit instruction ("I only need
the elements and nodes... no material models"):**

| Kept | Dropped |
|---|---|
| `*NODE` (id + xyz) | `*MAT_*`, `*SECTION_*`, `*HOURGLASS_*` |
| `*ELEMENT_SOLID` / `*ELEMENT_SHELL` (id + **pid** + connectivity) | `*PART` title cards, `*INCLUDE`, comments |

The part id is kept on every element — not a material, but the topological
tag ("this hex is Thorax_Flesh" vs. "this hex is Skull_Trabecular_Bone")
that site resolution (§1.2) needs and has no other way to get.

**Output format is comma-separated free-field, not fixed-width** — this
project's own writer controls the layout, so the fixed-width-column
ambiguity that motivated bridging the client's parser in the first place
(ids abutting past 7 digits) simply doesn't apply to files this module
produces; a plain `line.split(",")` reads them back exactly, at any
magnitude.

**Two card-shape requirements found empirically, not assumed** — by writing
files and feeding them back through the client's own parser until it
stopped raising:
- `*NODE` needs **exactly 6 comma fields** (`nid,x,y,z,tc,rc`) — the
  parser's delimited reader does a bare `split(",")[idx]` with zero
  tolerance for missing trailing fields. The otherwise-unused `tc`/`rc`
  constraint flags are written as `0`.
- `*ELEMENT_SHELL` is exactly 6 fields (`eid,pid,n1..n4`); the single-line
  8-node `*ELEMENT_SOLID` form is exactly 10 fields (`eid,pid,n1..n8`).

**Verified end to end** for `F05_Standing`: re-parsing the written files
with the same authoritative client parser and comparing against the
original source deck gave an exact node-coordinate match and exact
element-connectivity match on sampled nodes/elements. See
`tests/test_hbm_models.py` and `tests/test_hbm_normalize.py`.

**A real, non-obvious finding surfaced while building this, worth keeping**
(now historical — it shaped the source-side extraction logic, even though
the runtime registry no longer needs to know it): the client's original
"Bone" decks (used only during extraction, never seen at runtime) are not
shell-only — they contain real skeletal bone solid elements *in addition to*
bone surface shells, discovered only by a full per-file scan, not a
first-keyword check. Separately: Thorax_Flesh (pid 2000500) is a real,
populated part with 34,980 hex8 elements in the seated M50/F05 source decks
— but the *standing*-posture M50 source export never included it. Getting a
standing-posture M50 flesh mesh (if/when that model is built) is therefore
not a same-posture anthropometry morph — it needs re-posing a seated flesh
mesh into a standing configuration (hips/knees/spine actually re-articulate),
a substantially larger problem than anything currently scoped in §2. Not
relevant to `F05_Standing` (today's only built model), which has full
native standing-posture flesh.

### 1.2 Body-site registry and flesh+skin resolution — `config/sites.py`

Today `region_specs.py` hardcodes `solid_pids=(2000500,)` /
`shell_pids=(2000501,)`. Replace with a lookup driven by known part ids.

> **Revised from the original plan.** The original plan was to match `*PART`
> title cards (`Thorax_Flesh`, etc.) in the source deck. That doesn't work
> anymore: §1.1's normalized `HBM/<model>/Elements.k` files deliberately
> strip every `*PART` title card (no materials, no part names — geometry
> only, per instruction). So there are no titles left to match against at
> the point site resolution actually runs. Site resolution must therefore
> work off a **hardcoded, verified pid table**, not a title lookup — title
> matching only ever applied to the messy original source, and only during
> the one-time `config/hbm_normalize.py` extraction step (which doesn't
> need it either, since it's told which model KEY to build, not which
> site).

- `resolve_site(model, site_name) -> SiteResolution` returning the solid
  ("flesh") PIDs and shell ("skin") PIDs for that site, from a small,
  explicit table — confirmed against the original `I-PREDICT_v1.0_Main.dyn`
  `*PART` cards before normalization stripped them:

  | Site | Flesh (solid) | Skin (shell) |
  |---|---|---|
  | `torso` | `Thorax_Flesh` — **2000500** | `Thorax_Skin` — **2000501** |

- **Torso only, per explicit instruction** — head is out of scope (no head
  PPE exists), so the earlier two-row table and the multi-site detection
  apparatus below are dropped down to torso-only. Re-add a row (and the
  detection logic in §1.3) only if/when a second site is actually needed.
- **This pid is a property of the I-PREDICT model FAMILY, confirmed
  identical across F05/F50/M50/M95** (same topology, different
  anthropometry) — not something re-derived per model.
- **Flesh and skin are always pulled together.** This is a standing
  assumption, not a user choice — a site resolves to a `(solid_pids,
  shell_pids)` pair or it is not usable for that model (see §1.1's
  `F05_Standing` — the one model built today — which has both).

### 1.3 Site selection — simplified: torso only, no detection needed yet

Originally scoped as automatic PPE-driven site detection (shape signature +
CPD fit scoring) to distinguish e.g. a chest plate from a helmet. **Out of
scope for now, per explicit instruction: only the torso and the rigid plate
PPE are in play** — there is no second site or second PPE shape to
distinguish from, so building a detector would be speculative complexity
with nothing to validate it against.

**Current behaviour: site is always `"torso"`.** `config/pairings.py` (§1.5)
hardcodes this rather than calling any detection logic. If a genuinely
different-shaped PPE or a second body site is introduced later, revisit this
section — the original Tier-1-descriptors/Tier-2-fit-score design (pose-
invariant shape descriptors to shortlist, then a rigid-CPD chamfer score to
decide) is still the right approach *then*, just not needed *now*.

### 1.4 PPE registry and multi-format loader — `config/ppe.py`, `core/mesh/io.py`

- Accept `.k`, `.inp`, `.stl` (and `.feb`, free from the same loader).
- Back it with the Morphing `mesh_io` package, which already covers every one
  of these:

  | Format | Module | Reader / writer |
  |---|---|---|
  | LS-DYNA `.k` | `mesh_io/lsdyna.py` | `load_lsdyna` / `save_lsdyna` |
  | Abaqus `.inp` | `mesh_io/abaqus.py` | `load_abaqus` / `save_abaqus` |
  | FEBio `.feb` | `mesh_io/febio.py` | `load_febio` / `save_febio` |
  | STL / OBJ | `mesh_io/surface.py` | `load_stl` / `save_stl` |
  | dispatch | `mesh_io/__init__.py` | `load_mesh(path, format=None)` |

- `.stl` is surface-only, so it is **tetrahedralised into a solid on import**
  (same `tetgen` constrained-Delaunay path as §3.2, which preserves every input
  boundary vertex and facet exactly). This keeps every PPE a volume mesh
  regardless of source format, so downstream fitting, material assignment and
  `.feb` writing have exactly one code path. Requirements: the STL must be
  watertight and manifold — validate on import and fail loudly with the
  specific defect rather than feeding a bad surface to `tetgen`.
- `PpeSpec` dataclass: `key`, `display_name`, `mesh_path`, `format`,
  `is_rigid`, plus the placement hints consumed by Goal 2 (§2.1) and the
  detection descriptors consumed by §1.3.

### 1.5 Linked pairing library — `config/pairings.py`

The "linked library" requirement: a PPE is meaningful only against a body site,
so the valid combinations and their fitting parameters live in one table.

```python
CasePairing(
    hbm_key="M50_Standing",
    site="torso",                # auto-detected by §1.3, user-overridable
    ppe_key="armored_plate",
    fit=FitParams(target_standoff_mm=..., allow_scale=False, backend="cpd"),
    solve=SolveParams(indentation_mm=4.25, initial_contact_mm=0.5),
)
```

- The table is the **ground truth that site detection is validated against**: a
  known pairing tells us what §1.3 *should* have inferred, so detection can be
  regression-tested rather than trusted blindly.
- A pairing supplies the fitting and solve parameters once the site is known;
  detection picks the row, it does not invent parameters.
- A pairing resolves to a deterministic case directory under `cases/`, so
  re-running is idempotent and results are browsable.
- Existing `cases/F05_Torso_ArmoredPlate` and `cases/M50_Torso_ArmoredPlate`
  become the first two entries and serve as regression fixtures.

---

## Goal 2 — Seating PPE onto the body with CPD + RBF

**Outcome:** given an arbitrarily posed/located PPE mesh, produce a rigidly
transformed PPE that sits against the body region at a controlled standoff,
with the correct face toward the body — plus the derived load curve and time
stepping.

### 2.1 Deterministic orientation from PPE shape signature — **implemented, verified**

PCA alone leaves an 8-fold sign ambiguity, and chamfer scoring on a
near-symmetric plate picks the wrong flip. Morphing's `pca_affine_align`
already mitigates this with an `orientation_penalty_mm` term, but for PPE we
can resolve it outright from known shape features.

Implemented in `fitting/plate_signature.py` as `compute_plate_orientation(mesh)
-> PlateOrientation` (fields: `concave_normal`, `convex_normal`,
`concave_face_node_ids`, `convex_face_node_ids`, `thickness_axis`, `up_axis`,
`concavity_score`, `taper_ratio`). Specific to the single rigid-plate PPE in
scope; each future PPE gets its own signature routine against its own known
geometric quirks, per this same pattern.

| Feature | Rule | Implementation |
|---|---|---|
| Concave face → body | The more concave side seats against the body; the convex side faces out | PCA on the surface node cloud finds the through-thickness axis (smallest-variance eigenvector). Boundary faces are bucketed into the two large sides by their normal's dot product with that axis. For each side, `cov(r², h)` — face-centroid height `h` along the side's own outward normal vs. in-plane radial distance `r` from the side's centroid — is **positive for concave** (a bowl: the rim sits higher than the centre along the outward normal) and **negative for convex** (a dome: the centre sits highest). Verified both on a synthetic paraboloid shell (ground truth by construction) and on the real plate mesh. |
| Trapezoid end → superior | Top of the plate is shorter and trapezoidal | On the concave face's 2D in-plane footprint, try both in-plane axes as the candidate up/down direction; for each, compare the in-plane width of a narrow (~10%) band at each end. **A broad one-third split was tried first and failed** — the real plate's taper is concentrated in only the last ~10% of its length, so a wide end-band's `ptp()` was dominated by the still-full-width interior and washed out the real signal (measured taper ratio 0.99 instead of the true 0.66). Narrow end-bands fixed this. |
| Straight end → inferior | Bottom edge is a straight line | Confirmed implicitly: the axis/end not chosen as the taper end is the wide, ~constant-width (straight) end. |

Verified on the real `PPE/plate.inp`: concave/convex faces are near-equal in
size (38,638 vs. 38,544 surface nodes), `taper_ratio ≈ 0.658` (a genuine ~34%
narrowing, confirmed by direct width-vs-height binning: 159.5 mm at the
narrow end vs. 240.1 mm at the wide end), and the concave face already points
toward `HBM/F05_Standing`'s torso (`dot(concave_normal, direction_to_torso) ≈
0.79`) — this specific shipped file needs no flip for that pairing.
- Body-side target frame from the region: anterior direction (global `X` in the
  I-PREDICT v1.0 frame — sternum mean `X ≈ 94.1` anterior vs. T6 mean
  `X ≈ 216.3` posterior) and superior direction (in the F05 standing export
  frame, **up = −Z**, confirmed two ways: skin `Y`-width tapering to ~108 mm at
  the neck opening, and the coordinate convention itself — full-body `Z` spans
  `[−1414, +75]` (≈ standing height), and thorax skin (`Z ∈ [−1239, −683]`)
  sits at 12–49% of the way down from the `Z`-minimum end, matching head→mid-
  torso proportions. This is a Z-down (SAE J211-style) frame: **more negative
  Z is more superior**, not less negative — the opposite of a naive "Z-up"
  assumption, and worth re-checking explicitly for any new model export.
- **These axis conventions must be re-derived per model, not assumed** — the
  standing export frame differs from the v1.0 deck frame (`Z ∈ [−1239, −683]`
  vs `[12, 490]`). This was originally planned as a `detect_body_axes(region_mesh)`
  landmark/width-profiling helper, but was not actually built until a real bug
  forced the issue — see **§2.1b** below for what shipped instead
  (`fitting/body_axes.py`'s `derive_torso_axes`) and why.
- The descriptor framework generalises: more complex PPE (carriers with straps)
  adds its own descriptors to the same registry.

### 2.1b Per-model anterior/up axis derivation — `fitting/body_axes.py` — **implemented, verified (2026-09-23)**

**The bug.** Adding a second raw-dropped-in model, `F05_Seated` (§1.1's
`build_hbm_model_from_folder` path), and running it through the full
pipeline seated the plate on the **posterior** side of the torso — reported
directly by testing it end to end, not caught by any unit test (every
existing `tests/test_seat_plate*.py` test only ever exercises
`F05_Standing`-shaped synthetic fixtures against `seat_plate()`'s own
hardcoded `DEFAULT_ANTERIOR_AXIS`/`DEFAULT_BODY_UP_AXIS` defaults, so a
model with a genuinely different real-world coordinate convention was never
covered).

**Root cause.** Those defaults (`(1,0,0)` anterior, `(0,0,-1)` up) were
derived once, by hand, specifically for `F05_Standing`'s own export frame,
then silently reused for every model `build_case()` seats a plate against.
Different raw HBM exports are not guaranteed to share a coordinate
convention just because they're the same client/model family — confirmed
quantitatively (below), not just visually.

**Fix.** `derive_torso_axes(model) -> BodyAxes` computes each model's own
anterior/up/lateral axes directly from real skeletal landmarks already
present in every I-PREDICT v1.0 family deck, instead of assuming a fixed
global convention:

| Axis | Landmarks (part ids, validated against `HBM/01_Model/I-PREDICT_v1.0_Main.dyn`'s real `*PART` cards) | Method |
|---|---|---|
| Anterior | Sternum (`Sternum_Cortical_Bone` pid 2000048 shell, `Sternum_Trabecular_Bone` pid 2000049 solid) vs. the full thoracic vertebral column (`T1..T12_Vertebral_Body_Cortical_Bone`, pids 3000021..3000054 step 3) | `sternum_centroid − spine_centroid`, normalized |
| Up (superior) | T1 (topmost thoracic level) vs. T12 (bottommost) | `T1_centroid − T12_centroid`, Gram-Schmidt orthogonalized against anterior (the thoracic spine has real kyphotic curvature, so this raw vector is not perfectly perpendicular to anterior on its own) |
| Lateral | — | `cross(anterior, up)`, normalized |

A distal landmark (patella, also found at this point — `R/L_Patella_Cortical_Bone`
pids 6100011/7100011) was deliberately **rejected** as a candidate: a seated
pose's bent knee/hip would not reliably share the torso's own anterior
direction in world space, so only landmarks squarely within the torso
region being fitted are used.

**Verification, in order:**
1. Derived axes for `F05_Standing` (whose correct axes are independently
   known — the hardcoded defaults above) came out within **~5.7°** of those
   defaults — the small residual is real thoracic kyphosis, not a
   derivation bug.
2. Derived anterior for `F05_Seated` came out **~163° from F05_Standing's**
   (essentially flipped, mostly `-X` instead of `+X`) — a direct,
   quantitative reproduction of exactly the reported "posterior" symptom,
   not a subtle discrepancy. Lateral came out nearly identical between the
   two (~`+Y` both) — left/right convention is shared across the family
   even when anterior/up are not.
3. `tests/test_body_axes.py` (8 tests, synthetic fixtures): correct
   anterior/up recovery for both a `F05_Standing`-like and a deliberately
   *flipped* convention, unit-length/orthogonality of all three axes,
   element-file-order independence, a clear `ValueError` for a model
   missing the landmark part ids entirely, and the JSON on-disk cache
   (write/reuse/`force_recompute`/stale-version-ignored).
4. **End to end, on real data, after wiring into `build_case()`**: re-ran
   `main_1.py`'s equivalent for `F05_Seated` and confirmed geometrically
   — `dot(plate_centroid − spine_centroid, derived_anterior)` came out
   **+248mm** (anterior side) vs. what the old hardcoded axis would have
   given, **-317mm** (posterior side, i.e. the exact bug) computed against
   the same seated plate. Final standoff gap unaffected (0.1000mm, as
   before).
5. **F05_Standing regression check**: rebuilt with the new per-model-derived
   axes (isolating *only* the axis change, same code otherwise) and
   compared against a parallel build using the old hardcoded axes — plate
   centroid shift **0.195mm** (mean per-node shift 0.20mm, max 0.26mm),
   concave-normal direction change **0.0034°**, final gap min unchanged
   (0.09996 vs 0.09997mm). The ~5.7° seed-axis change is almost entirely
   absorbed by the existing standoff-correction + symmetry-correction
   stages, as expected — the already-solved, GUI-consumed production
   `F05_Standing` case did not need to be re-solved.

**Wiring.** `seat_plate()`'s own module-level `DEFAULT_*` constants and
function-signature defaults are **unchanged** (so none of the ~20+ existing
`tests/test_seat_plate*.py` tests, which call `seat_plate()` directly with
its own defaults, needed to change). `febio/build_preliminary_case.py`'s
`build_case()` — the actual production call site `main_1.py` uses — now
calls `derive_torso_axes(model, cache_dir=CACHE_DIR)` and passes the result
into `seat_plate(..., body_up_axis=, anterior_axis=, lateral_axis=)`
explicitly, for every model. `app/viewer_app.py`'s camera orientation code
(`_compute_frontal_camera`/`_compute_plate_view_camera`) does the same,
replacing its own `DEFAULT_BODY_UP_AXIS` import, so the GUI's default
camera "up" is also correct per-model, not just the seating.

**Performance.** A cold `derive_torso_axes()` call scans the model's full
elements deck (~200-240MB) in a single pass across all four landmark groups
at once (not four separate passes) — measured **~65-77s** per model. Since
these axes are static per model (never change between cases), results are
cached to a small JSON file under `cache/body_axes_cache_<model_key>.json`
(mirrors the pattern `app/hbm_reader.py` already uses for the much larger
full-body mesh `.npz` cache) — a warm call returns in a few milliseconds.



### 2.2 Registration: shape-signature seed → local point-to-plane ICP — **implemented, verified**

The original plan (PCA pre-align → rigid CPD, both vendored from Morphing)
turned out not to work for this specific PPE/body pairing — tested directly
against real data (F05 torso + the shipped armored plate), not assumed.
Both are still vendored and available (`core/registration/pca_prealign.py`,
`core/registration/rigid_cpd.py`), but the seating pipeline
(`fitting/seat_plate.py`) uses neither by default. What ships instead:

| Step | Source | Entry point |
|---|---|---|
| Body surface extraction | `fitting/body_surface.py` | `extract_skin_surface(model, site) -> SurfaceMesh` (with vertex normals). Reads the normalized `Elements.k`/`Nodes.k` via the client's `iter_wanted_cards` bridge, not `core.mesh.lsdyna.load_lsdyna` (see that module's own docstring on why: HBM decks need the client's fixed-width-aware parser past 7-digit ids). Verified against real F05_Standing torso: 11,412 skin nodes (matches independent manual verification), vertex normals point away from the flesh centroid (mean dot ≈ −0.70, 98.7% negative). |
| Shape signature seed rotation | `fitting/seat_plate.py` | `seed_rotation_from_signature(plate_orientation, target_body_normal, target_up_axis)` — builds two orthonormal frames (source: plate's `concave_normal`/`up_axis`; target: `-local_body_normal`/`body_up_axis`) and returns the rotation mapping one onto the other exactly (`det = 1` by construction, no reflection risk). Replaces PCA pre-align's blind 8-way sign-flip search: the correct orientation is *known*, not searched for. |
| Local point-to-plane ICP | `core/registration/point_to_plane_icp.py` | `point_to_plane_icp(source, target_points, target_normals, config)` → `ICPResult` (composed rigid transform, `.apply(points)`). Replaces rigid CPD as the local refinement stage. |
| Target cropping | `fitting/seat_plate.py` | `_crop_indices_near(points, center, radius)` — restricts the (much larger) torso skin target to a local neighborhood of the plate before registration. |

**Why CPD (and even plain point-to-point ICP) were dropped from this path —
found empirically, not by assumption.** Registering the plate's concave
surface (source) directly against `PPE/plate.inp` + `HBM/F05_Standing`'s
torso skin (already seeded to a good ~13.2mm mean nearest-neighbor distance
by the signature seed rotation alone) was tested three ways:

| Method | Result |
|---|---|
| Rigid CPD (`core/registration/rigid_cpd.py`, uncropped target) | PCA's own extent-matching scale came out `[1.79, 1.30, 10.83]` (correctly rejected by `should_apply_prealign` as nonsensical); CPD itself converged (`s=1.0` exactly, as designed) but to a **worse** pose — 69.9mm mean distance, up from the 13.2mm seed. |
| Rigid CPD, target cropped to a local neighborhood | Still diverged, 60–70mm, regardless of crop radius tested (0 to 150mm margin beyond the plate's own ~195mm half-diagonal). |
| Plain point-to-point trimmed ICP, same local crop | Diverged differently but just as badly: improved briefly (7.1mm at iteration 0) then ran away to ~45mm by iteration 10 and stayed there. |
| **Point-to-plane ICP, same local crop** | **Converged monotonically to 5.3mm and stayed stable** (verified at both 30 and 60 max iterations — identical result). |

The diagnosis: the plate's concave face is nearly flat (only ~23mm of
out-of-plane deviation over its ~320×240mm footprint), while a
comparably-sized local patch of the torso's skin has genuine curvature
(up to ~230mm of relief once cropped to a similar footprint). This is the
classic ICP/CPD "aperture problem" — point-to-point and probabilistic
correspondence have no way to tell "the true matching patch" from "a
nearby, differently-curved patch that also looks locally plausible", so
free rigid search (CPD's EM correspondence, or point-to-point ICP's nearest-
neighbor correspondence) actively wanders to a wrong optimum. Point-to-plane
ICP's cost term, `(p−q)·n`, imposes zero penalty for a purely-tangential
correspondence error, so the ambiguous tangential degree of freedom is left
where the seed put it instead of being pulled somewhere wrong — confirmed
with a synthetic paraboloid-of-revolution test (`tests/test_point_to_plane_icp.py::test_symmetric_surface_leaves_residual_ambiguity`)
showing the same qualitative behaviour (improves, converges, stops short of
exact recovery) on a controlled shape, versus a synthetic *asymmetric*
surface where the same algorithm recovers a known transform to <0.5mm
exactly, proving the algorithm itself is correct and the residual is a
genuine shape-identifiability limit, not a bug.

CPD is **not deleted** — it may be the right tool for a PPE with enough
distinct, non-ambiguous 3D structure that correspondence search has a real
shape signal to lock onto (unlike a near-flat plate against a broadly
similar-curved body patch). Any future PPE should get this same
CPD-vs-ICP comparison run against its own real data before picking a
default, rather than assuming either one.

`fitting/seat_plate.py`'s `seat_plate(model, site, ppe_spec, ...)` sequence:

1. Extract the body region's skin surface (`extract_skin_surface`) and
   offset it outward along its own vertex normals by `target_standoff_mm`.
2. Load the PPE and compute its shape signature (§2.1).
3. Seed rotation (`seed_rotation_from_signature`), applied about the
   plate's centroid, to both the plate's full node set and its concave
   surface.
4. Crop the offset target to a local neighborhood of the (now-oriented)
   plate (`_crop_indices_near`, additive margin — see that function's
   docstring for why a *multiplicative* radius fails: `1.75 × plate_radius`
   ≈ 342mm is comparable to the torso's own extents and crops almost
   nothing).
5. Point-to-plane ICP refinement of the plate's concave surface onto the
   cropped local target.
6. Apply the composed rigid transform to the **full** PPE mesh.

Verified end-to-end on real data (`tests/test_seat_plate.py`): seed rotation
is a proper rotation (no reflection); ICP improves 13.2mm → 5.3mm and stays
stable; the transformed plate's concave face still faces the torso
afterward (not accidentally flipped); the closest-approach point lands
under 1mm from the skin (consistent with §2.4's "as close as possible, a
little penetration is fine" seating goal — precise standoff/penetration
control is §2.4/§2.6's job, not this stage's).

Scale is never enabled: PPE is physical hardware and must not be resized.

**Why this replaces `fit_plate.py` and `snap_plate_pose.py`.** The existing
`fit_plate.py` registers torso-to-torso and reuses a known plate pose, which
only works when a reference case with that exact plate already exists;
`snap_plate_pose.py` then requires a human to have placed the plate by eye
first. The signature-seeded ICP path removes both constraints. Both scripts are
retained as fallbacks during migration and deleted once the new path
reproduces the existing `cases/` fits.

### 2.3 RBF — scope and limits

RBF/TPS (`morphing/rbf_morph.py`: `RBFMorpher.fit/transform`, `TPSMorpher`,
kernels `gaussian | multiquadric | thin_plate`) is used for:

- Propagating a fitted surface displacement into the PPE's full volume mesh
  when the PPE is deformable.
- Morphing the *body* region between anthropometries.

It is **not** used to deform the rigid plate. A rigid plate must receive a
rigid transform only; applying RBF to it would silently change its geometry and
invalidate the rigid-body material. The fitting API enforces this via
`PpeSpec.is_rigid`.

### 2.3b Robustness to a far-away/rotated starting pose — **implemented, verified**

Directly tested against the user's original scenario: "if the torso is
rotated in some weird way and is located far away, I need a way to bring it
back to the human body model." `seat_plate()` has two paths, chosen by
`far_away_threshold_mm` (default 200mm, comparing the plate's centroid to
the body surface's centroid):

- **Near-body** (the common case): orientation is seeded from the *local*
  body normal nearest the PPE's own current position (`_local_body_normal`)
  -- the precise path, verified end to end in `tests/test_seat_plate.py`.
- **Far away**: orientation is seeded from a fixed anatomical anchor instead
  (`_default_anchor_point` -- the body surface's own most-anterior point),
  and the plate is coarsely translated there (`_coarse_translate_to_anchor`)
  before the local ICP stage runs.

**A real, empirically-caught mistake worth recording**: the first
implementation used the anchor-based path *unconditionally*, on the theory
that it should be a no-op for an already-close PPE. It was not — applying it
to the already-good shipped `PPE/plate.inp` measurably degraded the fit
(ICP's `final_mean_dist_mm` went from the verified 5.3mm to ~51mm, and the
concave face no longer reliably pointed at the body). The anatomical anchor
turned out to sit ~65mm away from the true best position (the "most anterior
point of the *entire* torso skin" isn't exactly the same point as "where a
chest plate conventionally sits"), and ICP's own aperture-problem sensitivity
to starting position (§2.2) turned that 65mm offset into a genuinely worse
local minimum, not just a slower convergence. This was caught by the existing
`test_seat_plate.py` regression suite failing after the change — exactly the
kind of thing regression tests against real data are for. Fixed by only
taking the anchor path when the PPE is actually far away.

Verified in `tests/test_seat_plate_robustness.py` against three real
perturbations of the shipped plate (rotations of 90°, 130°, and a near-180°
flip, each combined with 400-500mm of translation): all three recover a
correct seating (concave face toward the body, zero penetrating nodes, final
minimum gap within 0.05mm of the 0.1mm target) via the far-away path. The
far-away path's *tangential* placement is less precise than the near-body
path's (the fixed anchor is an approximation, not a per-PPE landmark) — but
the standoff correction (§2.4) is a separate, robust closed-form step that
still lands the closest-approach point on target regardless.

### 2.3c Manual vertical repositioning — **implemented, verified**

Real user need: the shipped/seated plate can end up sitting lower than
intended (e.g. over the stomach, when it should cover the chest — caught
from a visual check of a rendered case, not from any automated metric).
`seat_plate(..., vertical_offset_mm=...)` moves the plate along
`body_up_axis` by the requested amount before the local ICP/standoff stages
re-conform it to the body surface at the new height. Positive = superior
(up), negative = inferior (down), per this model/site's `body_up_axis`
convention (§2.1 — re-derive per model, don't assume).

**A second real aperture-problem consequence, found and fixed here.**
Point-to-plane ICP's own tangential translation is *large* even in the
well-seated baseline case — measured at ~88mm out of ~90mm total on real
data. That's tolerable (even beneficial) when nothing constrains where the
plate should sit tangentially, but it actively **undoes** a deliberately
requested reposition: an unclamped `+100mm` request was measured collapsing
to only ~12mm of net movement, because ICP slid the plate back toward
whatever locally-comfortable patch it preferred. Fixed by discarding ICP's
tangential translation component — keeping only its rotation and
normal-direction correction — whenever `vertical_offset_mm != 0`. The
no-offset path (the default) is completely unchanged; this was a
deliberate, narrow scoping decision after confirming that clamping
tangential motion *unconditionally* would also alter the already-validated
baseline case's behavior (its own large tangential ICP correction
apparently still lands somewhere reasonable, but changing that
unconditionally was an unnecessary risk to already-tested behavior for a
problem that only matters when a caller explicitly asks to move the plate).

Verified on real data (`tests/test_seat_plate_vertical_offset.py`):
`+100mm` vs. `-80mm` requests differ from each other by the expected
~180mm; both still seat with the final gap close to the 0.1mm target
(the raised case has ~0.017mm of residual penetration on a single node —
below the mesh's own ~0.065mm geometric resolution floor, i.e. not a real,
resolvable defect); `vertical_offset_mm=0.0` is byte-for-byte identical to
omitting the argument.

### 2.3d Bilateral symmetry correction — **implemented, verified**

**A real asymmetry was found and fixed here — in three separate rounds,
since fixing the first two did not fully explain a solved case's still-
visibly-uneven contact/displacement field.** The user pointed out a solved
case (`F05_Standing_torso_armored_plate_final.feb`) where the plate did not
sit evenly on the chest when pressed in.

**Round 1 — roll + lateral offset.** Measured directly (Kabsch rigid-
transform recovery from the shipped, unseated plate mesh to the solved
case's actual node positions — first attempt gave a false ~10mm residual
from a `U`/`V` swap bug in the SVD formula, `R = U @ D @ Vt` instead of the
correct `R = Vt.T @ D @ U.T`; fixed and reverified against a synthetic
ground-truth transform before trusting it against real data):

- **A 0.73-degree roll** about the plate's own body-facing normal — on a
  ~240mm-wide plate this is a ~3mm differential penetration edge-to-edge at
  full press.
- **A 1.0mm lateral offset** from the torso's own left-right midline.

Root cause: point-to-plane ICP's fitted rotation (needed to conform the
plate to the body's *local* curvature) is a general 3D rotation — nothing
about the seating pipeline up to that point constrains it to preserve
lateral centering or keep the plate's up-axis free of lateral tilt. Seed
rotation (§2.1/§2.2) *does* align the plate's up-axis exactly to
`body_up_axis` (zero lateral component, by construction of the full 3×3
frame-to-frame rotation) — but ICP's subsequent refinement can undo that
alignment by any amount, in any direction, with nothing downstream to
catch or correct it.

**Round 2 — the user re-checked the corrected case and still saw uneven
contact.** Suggested exactly the right diagnostic: *draw a vector across
the plate's width and check whether it's parallel to the body's own
coronal (Y-Z) plane.* Implemented directly (`_find_rim_indices` +
`_apply_symmetry_correction`'s new stage 0): take the concave face's
left-rim and right-rim centroids at mid-height (a narrow vertical band
avoids the trapezoid taper at the top skewing the comparison), and check
the vector between them for a component along the anterior axis. Found: **a
0.67-degree yaw** about the body's *vertical* axis — a real, separate
rotational degree of freedom the roll-only correction (round 1) cannot
reach at all, since that one only rotates about the plate's own *normal*,
not the body's up axis.

**Round 3 — the lateral-recentering target itself was found to be wrong,
after the user visually identified the actual landmark it should match.**
An intermediate fix (before this one) switched the lateral-recentering
target from the whole-torso bounding-box center to a numerically-fitted
*local* mirror-symmetry plane near the plate (`_local_mirror_symmetry_offset`
— minimizes mean squared nearest-neighbor distance between a point cloud
and its own reflection), on the reasoning that it fit the local patch's own
self-mirror residual ~60x better (~0.4mm² vs. ~25mm² for the whole torso).
**The user then pointed directly at a rendered case and identified the
actual anatomical landmark this should align with: the spine groove** — the
visible indentation running down the back where the skin dips toward the
front between the erector spinae muscle bulges, directly over the spinous
processes — and observed the plate's center of mass was visibly left of it.
Measured directly: the local-mirror-fit target disagreed with the spine
groove by **~1.1mm** (local-fit ≈ +1.09mm vs. spine groove ≈ -0.05 to
-0.12mm) — it had overfit to some local front-side anatomy (pecs/sternum)
near the plate rather than the body's true structural midline. The spine
groove, by contrast, closely matches the much simpler whole-torso
bounding-box center (both within ~0.2mm of zero) — strong evidence the
local-mirror-fit approach was the wrong idea, not just imprecise.

**Fix**: `_find_spine_groove_lateral_mm` finds this landmark directly and
automatically — no manual measurement needed for any future seating. Method:
restrict to the posterior ~30% of the body's own anterior-axis range (the
"back"), bin by height into slices restricted to the plate's own vertical
extent (so the groove is measured at the height that matters, not averaged
over the whole torso including the neck or waist), and within each height
band fit a quadratic to the anterior-coordinate-vs-lateral-coordinate
profile in a narrow window around the midline — the groove is a *local
maximum* in anterior coordinate (skin dips toward the front there), so only
concave-down fits are kept. Returns the median across valid height bands
(robust to the occasional band picking up an off-midline feature instead);
falls back to the whole-torso bounding-box center (logged) if too few bands
produce a usable estimate. Because this target no longer depends on the
plate's own lateral position (only its vertical extent, essentially fixed
once standoff correction completes), the earlier fixed-point iteration
loop needed for the local-mirror-fit's self-consistency is no longer
needed — a single pass is correct, and seating is measurably faster as a
result (~54s vs. ~111s for a full `seat_plate()` call).

**Fix**: `fitting.seat_plate._apply_symmetry_correction`, run by default
(`enforce_symmetry=True`) right after standoff correction, in this order:

0. **Yaw correction** about `body_up_axis`, from the rim-to-rim vector
   check above — rotates (pivoting at the rim vector's own midpoint) until
   that vector has zero component along `anterior_axis`.
1. **De-twist rotation** about the plate's *current* body-facing normal
   (recomputed after the yaw stage rotates it slightly) — target "up" is
   `body_up_axis` projected into the plane perpendicular to the current
   normal (the closest the up-axis can get to true global up, given the
   normal direction already fit), not `body_up_axis` re-imposed outright.
2. **Lateral recentering** — pure translation along `lateral_axis`
   (`DEFAULT_LATERAL_AXIS = [0, 1, 0]` for this export frame) so the
   plate's concave-face centroid lands exactly on the body's own **spine
   groove** (see Round 3 above and `_find_spine_groove_lateral_mm`).
3. Standoff correction (§2.4) is re-applied afterward, since every stage
   above can perturb the achieved gap slightly.

Verified on real data (`tests/test_seat_plate_symmetry.py`, 9 tests): the
rim-to-rim vector has ~0 anterior component (confirmed on a freshly
regenerated case: `6e-10mm`, essentially exact), the plate's up-axis has ~0
lateral component, and its concave-face centroid lands on the spine-groove
target to within ~0.1mm; the standoff/zero-penetration guarantee (§2.4)
still holds exactly; `enforce_symmetry=False` reproduces the previous
(non-corrected) behavior exactly, for comparison or an intentionally
off-center PPE. All 15 pre-existing `seat_plate` regression tests
(baseline, vertical-offset, far/rotated robustness) pass unchanged with the
new stage on by default. Confirmed on a freshly regenerated real case: the
plate's concave centroid now lands at **Y ≈ -0.12mm** (was +1.09mm before
this fix) — negative, matching the user's own visual read ("move it a bit
more in the -Y direction").

A one-off manual fix (direct node-coordinate rewrite of the specific
already-solved `.feb`, not a GUI transform-dialog value — after repeated
attempts at handing the user Rotate/Translate dialog numbers ran into
unresolvable ambiguity about FEBio Studio's own pivot/Absolute-vs-Relative
semantics, direct node editing was used instead, verified byte-for-byte
identical outside the touched node block) was also given directly to the
user for an earlier already-solved case, before the spine-groove fix landed
in the pipeline itself.

### 2.4 Seating, standoff, and derived time stepping — **implemented, verified**

Target: *the closest we can get to the body without deformation* — which for
FEBio penalty/AUGLAG contact means a **small positive gap**, not zero and never
negative. Contact must be detected on the first increment without the plate
starting embedded.

**Implementation**, all in `fitting/`:

| Piece | Source | Entry point |
|---|---|---|
| Point-to-face signed gap | `core/mesh/signed_distance.py` | `signed_gap_to_surface(query_points, surface) -> SignedGapResult` (`.min_mm`, `.n_penetrating`, etc.). Uses `trimesh`'s AABB-tree point-to-triangle nearest point (`mesh.nearest.on_surface`, needs the `rtree` package — added to `requirements.txt`), signed via the matched triangle's own outward face normal. Verified: a flat-plane synthetic test confirms sign/magnitude exactly; a point-above-vs-mirrored-below pair confirms sign flips correctly. |
| Mesh-aware standoff floor | `fitting/standoff.py` | `compute_standoff_floor(surface, radius_of_curvature_mm=150.0) -> StandoffFloor`; `resolve_target_standoff(requested_mm, floor)` clamps up (with a logged reason) rather than silently accepting an unachievable target. Verified: reproduces AGENTS.md's own worked example exactly (7mm edge @ 150mm radius → 0.0408mm sagitta, matching the documented 0.041mm) and scales as edge²  (5× edge → 25× sagitta, confirmed to within 5%). On the real `F05_Standing` torso skin: median edge 6.23mm → floor 0.0648mm — comfortably below the 0.1mm default target. |
| Closed-form standoff correction | `fitting/seat_plate.py` | `_apply_standoff_correction(...)`, folded into `seat_plate()` as its final stage. Translates the plate along its *current* concave-normal direction (the original signature normal, rotated through both the seed and ICP stages — a direction vector only needs the rotation, not the translation, part of each stage) by `delta = current_min_signed_gap − target_standoff`, re-measuring and repeating up to 5 times (correcting for the extremal point's local body normal not being exactly anti-parallel to the translation direction — true by construction right after seating, only approximately true after that). Verified on real data: converges the minimum signed gap to **0.10000923mm** against a 0.1mm target (zero penetrating nodes), from a pre-correction ICP fit. |

`seat_plate()`'s full sequence is now: extract body surface → compute/resolve
standoff floor → seed rotation (§2.2) → local crop → point-to-plane ICP (§2.2)
→ closed-form standoff correction (this section). `SeatingResult` carries
`standoff_floor`, `resolved_target_standoff_mm`, and `final_gap` (a
`SignedGapResult` against the **real, uncropped** body skin — not the
locally-cropped ICP target — so the final check is against the whole torso,
not just the neighborhood used to fit it).

1. After CPD, compute the signed distance field from PPE inner surface to body
   skin (`clearance_metrics`).
2. Translate the PPE along its body-facing normal until the **minimum signed
   gap equals the target standoff**. Closed-form, no iteration on the solver.
3. Verify no initial penetration anywhere (`min_signed_mm > 0`).

> **This check is not theoretical.** The existing M50 case fails it: 309 plate
> nodes already interpenetrate the undeformed skin by up to 3.22 mm at rest
> (`VERIFICATION.md` §5.2). Seating must be validated with a **signed** measure
> along the travel axis — an unsigned nearest-point distance cannot tell
> penetration from clearance, and reports a reassuring 0.281 mm minimum for
> exactly the case that is 3.22 mm embedded.
>
> **Update:** re-checked with the now-implemented `signed_gap_to_surface`
> against the M50 case's plate and skin *paired from the same solved `.feb`*
> (not the standalone `torso_M50_standing_skin.k`, which turned out to be a
> different mesh entirely) and found **no penetration** — see
> `VERIFICATION.md` §5.2 for the full re-check. M50 is out of scope now, so
> this isn't chased further, but the "3.22mm" figure should not be taken at
> face value if M50 is ever revisited.

#### Choosing the standoff: bounded below by mesh faceting

A target gap in the 1e-1 to 1e-3 mm range is the right *intent*, but the
achievable floor is set by the mesh, not by preference. A faceted surface dips
below the true curved surface by the sagitta of its elements, so a requested gap
smaller than that is not physically meaningful — the "surface" is not defined to
that precision, and a point that clears a node may still penetrate a facet.

Measured on `torso_M50_standing_skin.k` (11,412 nodes / 11,158 shells): median
skin edge length **7.0 mm** (p5 2.9, p95 9.1), giving a sagitta of ~**0.041 mm**
at a ~150 mm torso radius of curvature.

| Skin edge length | Sagitta | Smallest meaningful gap (≈2× sagitta) |
|---|---|---|
| 3 mm | 0.0075 mm | 0.015 mm |
| 5 mm | 0.021 mm | 0.042 mm |
| **7 mm (current)** | **0.041 mm** | **0.082 mm** |
| 10 mm | 0.083 mm | 0.167 mm |
| 15 mm | 0.188 mm | 0.375 mm |
| 20 mm | 0.334 mm | 0.667 mm |

So on the current mesh:

- **1e-1 mm (0.1 mm) — use this.** 2.45× the sagitta: comfortably resolvable,
  tight enough to engage contact on step 1.
- **1e-2 mm** — 0.24× sagitta. Below geometric resolution; whether a given point
  clears or penetrates is decided by faceting, not by the seating.
- **1e-3 mm** — 0.02× sagitta. Meaningless at this discretisation.

`TARGET_STANDOFF_MM = 0.1` is therefore the default (§4.6).

**This couples Goal 2 to Goal 3.** Decimation makes elements larger and the
floor worse: at 15 mm edges the smallest meaningful gap is 0.375 mm, nearly 4×
the default standoff. `fitting/` must therefore compute the achievable floor
from the *actual* mesh it is handed and raise the standoff (with a warning) if
the requested value is below ~2× the local sagitta, rather than silently
accepting an unachievable target.

**Measure gaps point-to-face, never point-to-node** — **implemented**: see
`core/mesh/signed_distance.py` above.

**Load curve derivation — implemented, verified, bug found and fixed.**
The reference case's numbers are self-consistent and give the rule:

```
total_travel     = 4.25 mm          (NewHex_close.feb LoadCurve: <pt>1,-4.25</pt>)
initial_contact  = 0.5 mm           (first-contact increment)
step_size        = 0.5 / 4.25 = 0.117647  ≈ 0.1176   ← the reference's <step_size>
time_steps       = ceil(1 / step_size) = 9           ← the reference's <time_steps>
```

```python
step_size        = initial_contact_mm / total_travel_mm
time_steps       = ceil(1.0 / step_size)
total_travel_mm  = target_standoff_mm + indentation_mm
```

`febio/loadcurve.py`'s `derive_load_curve(target_standoff_mm, indentation_mm=4.25,
initial_contact_mm=0.5) -> LoadCurveSpec` implements this and emits the
`<LoadData>` points via `.points` (`[(0, 0), (1, -total_travel_mm)]`).

> **A real bug was found and fixed here.** An earlier version of this
> function used `target_standoff_mm` directly in place of `initial_contact_mm`
> in the `step_size` formula — i.e. it assumed "how precisely the plate is
> seated at rest" and "how far it should travel on the first solved step to
> establish meaningful contact" are the same number. They are not: the
> reference case's own construction only makes them look the same because
> its plate happened to sit off by exactly 0.5mm at rest, and 0.5mm was
> *also* a reasonable first-step size. Once this project started seating far
> more precisely (`target_standoff_mm = 0.1`, for geometric accuracy — see
> above), reusing it as the first-step size shrank the plate's actual
> first-step travel to ~0.1mm — far too small relative to the current
> whole-torso mesh's own ~3.5-7mm element edges to register as engaged
> contact on the first increment. This was the real cause of a full
> non-convergent solve (residuals stuck at ~1e-20, "No force acting on the
> system", repeated through every reformation/retry) that was otherwise
> investigated exhaustively (contact type/parameters, mesh scale, single- vs
> multi-axis rigid displacement, shell rotation dofs — none of these were
> the actual cause; see `febio/build_preliminary_case.py`'s module
> docstring for the full investigation history).
>
> **Fix**: `initial_contact_mm` is now a genuinely separate parameter
> (default 0.5mm, matching the reference), decoupled from
> `target_standoff_mm` (which remains purely a *seating* parameter).
> Verified: plugging the reference's own numbers back in
> (`target_standoff_mm=0.5, initial_contact_mm=0.5, indentation_mm=3.75`, so
> `total_travel_mm=4.25`) reproduces `step_size=0.1176` and `time_steps=9`
> exactly. With this project's own defaults
> (`target_standoff_mm=0.1, initial_contact_mm=0.5, indentation_mm=4.25` →
> `total_travel_mm=4.35`), `step_size=0.114943`, `time_steps=9` — still 9
> steps (not 44, as the buggy formula produced), now driven by a genuine
> 0.5mm first-step target instead of an accidental 0.1mm one. **Confirmed
> end-to-end**: `main.py`'s own generated case (no manual `.feb` editing)
> now solves to `N O R M A L T E R M I N A T I O N` in `febio4.exe`.
>
> Regression tests (`tests/test_loadcurve.py`) cover: the reference case
> reproduced exactly, our own defaults giving `time_steps=9`, the first-step
> travel landing near `initial_contact_mm` (not the standoff), and that
> `target_standoff_mm` alone no longer moves the first-step travel.

- The first solved step therefore always lands at the chosen initial-contact
  increment, which is what makes contact initiation well-conditioned.
- The rest of the `<Control>` block is ported verbatim from the gold standard
  (§4.1 below).

> **Load-factor overshoot — decided.** `ceil` means the solve runs
> *past* the load curve endpoint: `9 × 0.1176 = 1.0584`, a 5.84% overshoot.
> Both existing solved cases confirm this — `skin_displacement_F05_GUI.npz` and
> `skin_displacement_M50_GUI.npz` both record `state_time = 1.0584`. Whether
> the plate actually travels 4.25 mm or 4.50 mm depends on the load curve's
> `extend` mode (FEBio's default is CONSTANT, which clamps at the last point —
> so the final partial step is effectively a relaxation/settling step rather
> than extra travel). **`LoadCurveSpec.extend` is set explicitly to `CONSTANT`
> rather than relying on the default**, and the generated case should either
> accept the overshoot as a deliberate settling step or use
> `step_size = 1 / time_steps` to land exactly on 1.0. Do not leave this
> implicit — a silent 5.8% travel difference would invalidate cross-case
> comparison.

### 2.5 FEBio case assembly — `febio/` — **implemented, verified: solves to normal termination**

Removes the manual FEBio Studio setup that `apply_control_and_materials.py`
explicitly punts on ("Contact ... the plate's rigid body prescribed-motion BC,
and the load curve are intentionally NOT touched here").

Written from the gold standard, all values verbatim from `NewHex_close.feb`:

- `<Control>`: `STATIC`, `time_steps` and `step_size` from §2.4,
  `plot_level=PLOT_MAJOR_ITRS`; time stepper `max_retries=15`, `opt_iter=8`,
  `dtmin=0.0001`, `dtmax=0.3`, `cutback=0.5`; solver `non-symmetric`,
  `staggered`, `lstol=0.9`, `lsmin=0.01`, `lsiter=5`,
  `ls_check_jacobians=1`, `max_refs=15`, `dtol=0.01`, `etol=0.01`,
  full Newton, `mkl_dss`.
- `<Material>`: Ogden `Skin` (`density=1.1e-06`, `k=10`, `c1=0.05`, `m1=10`),
  Ogden `Flesh` (`density=1.1e-06`, `k=1.5`, `c1=0.003`, `m1=4`), rigid body
  for the PPE (`density=1`).
- `<Contact>`: `sliding-elastic`, `laugon=AUGLAG`, `tolerance=0.2`,
  `gaptol=0`, `penalty=1`, `auto_penalty=1`, `update_penalty=1`,
  `search_radius=3`, `fric_coeff=0`.
- `<Boundary>`: zero displacement on the artificial cut boundary — every outer
  face of the flesh volume not covered by a skin shell (neck, armholes, waist).
  This is the verified reference scheme (15,539 of 15,540 free nodes
  constrained; skin contact surface left free), carried over from
  `constrain_torso.py`.
- `<rigid_bc>`: prescribed displacement of the PPE along the seating axis
  driven by the load curve, rotations fixed.

**Implementation**: `fitting/body_volume.py` (`extract_torso_volume` — skin
shell + flesh solid sharing one node array, confirmed 100% node overlap on
real data; `hex8_boundary_faces` — orientation-corrected boundary quad
extraction, needed for FEBio contact facets, unlike the triangulated
extraction in `core/mesh/base.py`) and `febio/assemble_case.py`
(`assemble_torso_plate_case(...)` — combines a `TorsoVolume` and a
`SeatingResult`'s transformed plate into a full `.feb`). Runnable via
`febio/build_preliminary_case.py`.

**Resolved — the assembled case solves to normal termination.** It parses
successfully in FEBio 4.12 (`Reading file ... SUCCESS!`), the geometry is
independently verified correct from the written file's own node data (plate
concave surface ~1mm point-to-node / ~0.1mm point-to-face from the torso
skin, exactly matching the seating target), and — since the §2.4 load-curve
fix — solving it in `febio4.exe` reaches `N O R M A L T E R M I N A T I O N`
(9 nominal / 13 actual time steps including retries, final time 1.034, real
physical gap values of ~tens-of-microns during augmentation).

An earlier version of this pipeline diverged (residual stuck at ~1e-20,
"No force acting on the system", "maximum gap: 0.000000e+00" repeating
through every reformation/retry). This was investigated exhaustively before
the real cause was found: contact type, all individual contact parameters,
mesh scale in isolation, single- vs multi-axis prescribed displacement,
manual vs. auto penalty, standoff-exact vs. guaranteed-overlap start, and a
missing shell zero-rotation BC were all ruled out (not a FEBio version issue
either — the client's own solved M50 reference case ran to correct
convergence with the identical `febio4.exe` binary). The investigation also
reproduced, in isolation, a symptom that looked like it might be the cause:
adding *any* elastic (Ogden) material domain next to an otherwise-working
rigid body silently stopped that rigid body's prescribed displacement from
updating (a second disconnected *rigid* domain did not). **The actual root
cause was unrelated to any of this**: the load curve's step size formula
(§2.4) conflated the seating standoff (~0.1mm, deliberately tiny) with the
first-step engagement size, so the first solved step moved the plate only
~0.1mm into a mesh with ~3.5-7mm element edges — never registering as
contact. Fixing the load-curve formula alone (`febio/loadcurve.py`'s new
`initial_contact_mm` parameter) resolved the full pipeline's convergence
with no other changes. The "elastic domain breaks rigid body" symptom was
not re-isolated after the fix and is presumed to have been an artifact of
those specific minimal repro cases (which also used the too-small step
size), not a separate real bug — noted here rather than declared fully
explained, in case it resurfaces.

**Separately (not a convergence blocker): solved contact pressure/strain
reads lower than the gold-standard reference's own values.** This is most
likely a mesh-resolution effect, not a sign of insufficient indentation:
our whole-torso HBM flesh mesh has ~3.5mm through-thickness / ~7.3mm
in-plane elements under the contact patch, vs. the reference's
locally-refined mesh at ~0.85mm through-thickness (down to 0.43mm near the
surface) / ~3.7mm in-plane — roughly 4x finer directly under the indenter.
A coarser mesh averages stress/strain over a much larger element volume and
will report smaller peak values for the same underlying physical
deformation. Increasing indentation travel would not fix this — it would
just push more displacement through an already under-resolved mesh. The
real fix is local graded mesh refinement under the contact patch, which is
exactly what Goal 3 (below) is for, and has not yet been implemented.

---

## Goal 3 — Robust decimation and remeshing

**Outcome:** a whole-torso skin + flesh mesh, coarse enough to
solve quickly, with zero inverted elements and skin/flesh nodes still welded.

### 3.1 Why the structured route is not enough

`structured_remesh.py` (the gold standard's Stage 1) produces beautiful graded
hex8 meshes but requires **disc topology** — a single boundary loop. A full
torso is a "shirt": sphere-with-4-holes (neck, two armholes, waist). It fails
outright, which is why `crop_anterior.py` exists. But the stated requirement is
to **keep the entire torso** for strapped PPE, so cropping is off the table.

We keep both routes:

| Route | Use when | Output |
|---|---|---|
| **Structured** (`structured_remesh.py` port) | Region is a single patch / disc topology | Graded hex8 + quad4, best element quality |
| **Topology-agnostic** (`decimate_torso.py` evolved) | Whole torso, any number of openings | tet4 + tri3 |

Constants worth preserving from the structured route, and the reasoning behind
them: `NI=80`, `NJ=120`, `N_LAYERS=15`, `N_FINE_LAYERS=5`,
**`GROWTH_RATIO=1.1`** (lowered from 1.3 — at 1.3, layer 2 on a ~10 mm column
over the ribs came out ~0.36 mm and element 580805 inverted at every retried
timestep down to `dt=0.0002`; the destination geometry was physically
impossible, not a solver step-size problem),
`MIN_LAYER_THICKNESS_MM_SCHEDULE=[3.0, 0.667, 0.667, 0.667, 0.667]`,
`N_PROTECTED_LAYERS=5`, `MIN_THICKNESS_FLOOR_MM=10.0`,
`NORMAL_SMOOTH_ITERS=12`, `USE_HEX=True`.

### 3.2 Topology-agnostic route, hardened

Current `decimate_torso.py` already has the right shape:

1. Classify the flesh volume's outer surface into **skin** faces (covered by
   the shell part) and **free** faces (the artificial cut boundary).
2. Decimate each patch separately with
   `decimate_pro(..., boundary_vertex_deletion=False)`, so the shared edge loop
   between them is untouched and the patches re-merge watertight.
3. Constrained-Delaunay tetrahedralise the interior with `tetgen`, which
   preserves every boundary vertex and facet exactly — so flesh tets and skin
   triangles share nodes **by construction**, with no tolerance-based
   reattachment step.
4. Re-tag skin faces on the tetrahedralised boundary and emit
   `*ELEMENT_SHELL` + `*ELEMENT_SOLID`.

Hardening work:
- Make the reduction ratio adaptive to a target element count / target edge
  length instead of a flat `--reduction 0.85`.
- Enforce a minimum through-thickness element count so thin regions over the
  ribs do not collapse (the `GROWTH_RATIO` lesson, transplanted: check the
  *destination* geometry is physically compressible, not just that the element
  is valid at rest).
- Preserve feature edges and part boundaries during decimation.
- Emit the skin/flesh shared-node manifest as a first-class artifact so
  connectivity can be asserted in tests.

### 3.3 Quality gate and repair — port from Sparse

`ForSarah/Sparse/lsdyna_helper_functions_sparse.py` is a validated, pure
NumPy/SciPy/pandas repair engine (no mesh-library dependency), proven on
multi-million-element whole-body models. It is a far better backstop than the
current `untangle_mesh.py` / `febio_guided_untangle.py` pair.

**Detection**
- `compute_element_volumes(elements_df, nodes_df)` — signed volume, tet4 /
  penta6 / hex8, LS-DYNA reduced integration, vectorised. **Negative volume
  = fully/uniformly inverted element** — this is the one that catches a
  hex whose top and bottom faces are simply swapped.
- `compute_element_distortion(...)` — a **different, narrower** check:
  corner-to-corner Jacobian **sign change** (`jacobian_min`, `jacobian_max`,
  `jacobian_ratio`, `is_distorted`). Catches LOCAL tangling (e.g. a
  bow-tied quad face) that a uniformly-inverted element's volume sign alone
  would miss — but, symmetrically, does **not** catch a uniformly inverted
  element (no sign change to detect). **Both are needed together**; neither
  subsumes the other. Always `False`/not-applicable for tet4 (whose
  Jacobian is constant, so it can't locally tangle).
- `classify_solid_badness(...)` → `is_bad`, `bad_tier`.
- `compute_shell_jacobian(...)` / `classify_shell_badness(..., band_floor, band_ceiling)` — tri3 / quad4 jacobian ratio banding.

**Repair**
- `group_bad_elements_into_pockets(...)` / `group_bad_shells_into_pockets(...)`
  — union-find over bad elements, expanded by node adjacency, so pockets are
  independent and parallelisable. Shell pockets are guaranteed never to share a
  solid element.
- `repair_inverted_elements(elements_bad_df, elements_all_df, nodes_df, free_node_layers, constraint_layers, volume_epsilon, max_displacement, hard_node_ids=..., use_continuation=True, nonfree_elements=..., original_nodes_df=...)`
  — sparse-Jacobian `trust-constr` optimisation with analytic gradients and
  exact Hessian, hard per-axis displacement caps, and **non-regression
  constraints** so fixing one element cannot degrade its neighbours.
- `repair_distorted_shells(..., shell_band_floor, shell_band_ceiling, ...)` —
  same machinery for shell jacobian ratio, protecting touching solids.

**Connectivity preservation — the key requirement**
- `build_solid_node_adjacency(solid_elements_df)` → `(solid_element_node_lists, node_to_solid_indices)`.
- `find_touching_solid_elements(node_ids, node_to_solid_indices)`.
- Together these guarantee that moving a skin shell node applies a
  non-regression floor to every flesh solid element sharing that node — i.e.
  skin and flesh stay welded and neither degrades.

**Penetration** (relevant once straps and bone-adjacent PPE arrive)
- `build_hard_surface_index(elements_df, nodes_df, hard_pids)`,
  `detect_soft_hard_penetrations(...)`, `repair_soft_hard_penetrations(...)`,
  plus the all-PID generalisation (`classify_priority_tiers`,
  `find_overlapping_pid_pairs`, `detect_all_pid_penetrations`) with its
  containment auto-exclusion (≥85% nodal containment ⇒ intentional nesting, e.g.
  trabecular inside cortical bone).

**Escalation ladder** — `remesh/quality.py` runs these in order and stops as
soon as the mesh is clean:

1. Decimate/remesh with conservative settings.
2. Quality scan (Sparse detection functions above).
3. Pocket-based Sparse repair.
4. Local untangling (`untangle_mesh.py`'s Gauss-point-accurate
   `min_detJ_at_gauss`, which reproduces FEBio's own 2×2×2 quadrature rather
   than checking corners).
5. `febio_guided_untangle.py` if our own checks above still disagree with
   FEBio — uses FEBio itself as the oracle. Kept because of a documented
   finding: FEBio's reported element volume did **not** match any standard
   geometric interpretation of the literal node coordinates (verified
   against Gauss quadrature, two tet decompositions, VTK Verdict metrics,
   corner checks, a 9,261-point parametric grid scan, and several hex8 node
   orderings). When our metrics and FEBio's disagree, FEBio wins.
6. **The FEBio smoke test (`remesh/febio_smoke_test.py`) — the final,
   mandatory gate before any mesh is considered usable.** Every mesh this
   project produces must pass it. See §3.4 below.
7. Fail loudly with a report — never ship a mesh with inversions.

Also fold in `strip_isolated_nodes.py`: FEBio's own isolated-vertex removal
triggers an internal renumbering that has produced spurious negative-Jacobian
reports, so we remove unreferenced nodes ourselves first.

### 3.4 The FEBio smoke test — ground truth, not a geometric proxy

**Design instruction (verbatim):** *"All meshes should be checked with a
simple FEBio problem set-up and trying to run. If it starts up successfully
without issues, the mesh is good, otherwise there are problems."*

This is deliberately a *different kind of check* from everything in §3.3:
those are all geometric proxies for what FEBio itself will accept. This
check asks FEBio directly. It is necessary precisely because this project
has already documented (in `febio_guided_untangle.py`) that FEBio's own
notion of element validity does not always match any of our geometric
formulas — so the only fully authoritative answer to "will FEBio accept
this mesh" is to actually ask FEBio.

**FEBio binary.** Packaged directly into this repo at `febio_bin/` (not left
pointing at a sibling project's checkout) — a standalone copy of `febio4.exe`
plus its exact runtime DLL dependencies, originally assembled for the
Digital Twin project's own `ik_check/` (see `febio_bin/README.md` for how it
was built: recursive PE import-table resolution via `pefile`, not the whole
~300 MB FEBioStudio install). **Not committed to git** — ~146 MB of
third-party redistributable binaries; `.gitignore` excludes everything under
`febio_bin/` except its own `README.md`, matching the exact policy the
Digital Twin project uses for its own copy. Resolved via
`config/febio_solver.py`, `find_febio_executable()`: this repo's
`febio_bin/febio4.exe` first, then `$FEBIO_BIN_ROOT` override, then the
Digital Twin project's copy as a fallback (in case this repo's hasn't been
populated on a given machine), then `febio4` on PATH, then common
FEBioStudio install locations — same resolution philosophy as that
project's own `ik_check/build_hip_flexion_sim.py::_find_febio_executable`,
kept in sync.

**What the smoke test actually does (`remesh/febio_smoke_test.py`).** Every
node is fixed to zero displacement, zero load is applied, one static step is
run. This was verified empirically (not assumed) against febio4.exe
directly, using hand-built minimal cases:

- With every node fixed, FEBio reports `Nr of equations: 0` — no linear
  system is ever assembled. That does **not** mean element quality goes
  unchecked, though: a separate, earlier **"mesh initialization" phase
  checks every solid element's Jacobian sign unconditionally**, regardless
  of boundary conditions or equation count.
- A hex8 with two nodes transposed (a guaranteed inversion) was caught at
  that phase every time — `"Negative jacobian detected during mesh
  initialization."`, `"Model initialization failed"`, non-zero exit —
  in a fraction of a second, before the time-step loop even begins.
- This makes "fix everything, zero load" the ideal solid-element gate: it
  is essentially free (no real linear solve happens) and it is checked
  unconditionally, independent of whatever boundary conditions the real
  case will eventually use.

**Known limitation — shells are not reliably covered.** A hand-built
bow-tie (self-intersecting) quad4 converged normally under this scheme, and
still converged normally even under a real applied deformation. So this
smoke test is an *additional* gate for solid validity; `classify_shell_badness`
(§3.3) remains the primary shell quality check.

**A real false-positive was found and fixed while validating this against
production data — worth recording in full, because it is exactly the kind
of FEBio quirk this project has hit before.** Run against the real F05
combined flesh(hex8)+skin(quad4) mesh (46,138 elements; independently
confirmed clean — zero inverted, zero locally-tangled — by three separate
methods: `core.repair`'s centroid-volume check, its corner-Jacobian check,
and a hand-reproduction of FEBio's own 2×2×2 Gauss formula):

- Hex8-only (no shells) passed cleanly at full scale.
- Adding as few as **10** shell elements back caused FEBio to report a
  spurious `"Negative jacobian ... during domain initialization"` for a
  solid element with nothing wrong with it — one that passed cleanly both
  in total isolation and in an 18-element local neighbourhood.
- Root cause: `<shell_normal_nodal>1</shell_normal_nodal>` (nodal-averaged
  shell normals, which pull in geometry from every element sharing a node,
  including solid neighbours at the skin/flesh interface). Setting it to
  `0` (per-element normals) removed the false positive with no change to
  the actual geometry.
- The gold-standard reference case (`NewHex_close.feb`) *does* use
  `shell_normal_nodal=1` successfully — so this isn't "nodal normals are
  wrong" in general, just a narrow interaction this general-purpose gate
  must avoid triggering. `write_smoke_test_feb` always emits `0`.

Regression tests for all of the above live in `tests/test_febio_smoke_test.py`.

---

## Goal 4 — Cleanup, GUI, and documentation

### 4.1 Target layout

```
DeformationApp/
├── main.py                     # single entry point → GUI
├── README.md                   # how to install and run
├── AGENTS.md                   # this plan
├── VERIFICATION.md             # acceptance criteria
├── requirements.txt
├── config/
│   ├── hbm_models.py           # §1.1 discover clean HBM/<name>/{Nodes,Elements}.k folders
│   ├── hbm_normalize.py        # §1.1 on-demand: build one model's clean folder from raw source
│   ├── _hbm_raw_source.py      # §1.1 internal: locate a model in the messy original export
│   ├── sites.py                # §1.2 flesh+skin PID resolution per body site (hardcoded pid table)
│   ├── ppe.py                  # §1.4 PPE registry
│   ├── pairings.py             # §1.5 linked (hbm, site, ppe) table
│   └── febio_solver.py         # §3.4 find_febio_executable() (FEBIO_BIN_ROOT env var)
├── core/
│   ├── lsdyna/                 # client KeywordProcessor bridge (env-var root)
│   ├── mesh/                   # vendored Morphing mesh_io + mesh types
│   ├── registration/           # Kabsch, PCA pre-align, rigid CPD (core.registration)
│   └── repair/                 # vendored Sparse repair engine
├── fitting/                    # §1.3 site detection; §2 signature, CPD, seating, RBF
├── remesh/
│   └── febio_smoke_test.py     # §3.4 ground-truth "does FEBio accept this mesh" gate
├── febio/                      # §2.5 case writer (control/materials/contact/BC/loadcurve)
├── postprocess/                # .xplt → .npz, verification metrics
├── gui/                        # PySide6 app
├── cases/                      # per-case working dirs (large files gitignored)
├── febio_bin/                  # §3.4 bundled febio4.exe + DLLs (binaries gitignored)
├── HBM/                        # one self-contained folder per model, built on demand (§1.1)
│   └── F05_Standing/           # only model built today
│       ├── Nodes.k             # gitignored (large; regenerate via hbm_normalize)
│       └── Elements.k          # gitignored
└── tests/
```

### 4.2 Deletions and consolidations — **done (2026-09-22), see §4.7 for the full pass**

| Delete | Replaced by |
|---|---|
| `main_f05.py`, `main_m50.py` | `main_1.py` + `main_2.py`, HBM/PPE selection lives in `main_1.py`'s USER SETTINGS block + GUI dropdowns (`app/viewer_app.py`) rather than a separate `main.py` per the original plan -- deliberate divergence, since a solve happens in FEBio Studio between stages 1 and 2 |
| `app/viewer_app_f05.py`, `app/viewer_app_m50.py` | `app/viewer_app.py` (single parameterised viewer with HBM/PPE dropdowns) -- ended up staying under `app/`, not a separate `gui/` (see §4.7: `gui/` was an empty unused placeholder, deleted) |
| `app/hbm_reader_f05.py`, `app/hbm_reader_m50.py` | `app/hbm_reader.py` reader driven by `config.hbm_models.HbmModel` |
| `app/plate_model_f05.py` | `fitting/` + `config/ppe.py`'s PPE registry |
| `pipeline/region_specs.py` | `config/hbm_models.py` + `config/sites.py` |
| `pipeline/crop_anterior.py` | Retained only for disc-topology regions; not on the torso path -- deleted along with the rest of `pipeline/` since nothing on the current torso-only path needs it |
| `pipeline/fit_plate.py`, `pipeline/snap_plate_pose.py` | §2.2's point-to-plane ICP + symmetry correction (`fitting/seat_plate.py`) fully reproduces and improves on these; deleted |

`pipeline/` (all 12 scripts) and `gui/` (empty placeholder) were deleted in
their entirety -- see §4.7 for the full cleanup pass and confirmation that
nothing else referenced them.

Stray files removed from the working tree: `_skipped.face`, `_skipped.node`,
`lspost.cfile`, `lspost.msg`, `hbm.dyn`, `smoke_test.py` (replaced by
`tests/`). (`control.k`, `I-PREDICT_v0.12_*.k`, `NewHex_close.xplt` from the
original list were already gone before this pass.)

### 4.3 Repository size and what actually gets published

**Current: 1.48 GiB of loose objects.** `.gitignore` covers `*.k`, `*.feb`,
`*.xplt`, `*.log` but **not** `*.fsm`, `*.inp`, `*.npz`, `*.dyn` — so 70 MB
FEBio Studio session files and 60 MB plate meshes are committed.

**Intended published artifact set.** What goes to GitHub is only what the GUI
needs to render a body and animate a deformation on it — **body models plus
deformation maps, and nothing from the solver**.

| Publish | Do **not** publish |
|---|---|
| Source code, `README`, `AGENTS.md`, `VERIFICATION.md` | `.xplt` plot files |
| GUI-ready body geometry (`cache/*.npz`) | `.feb` case files |
| Deformation maps (`ResultsForGUI/*.npz`) | `.fsm` FEBio Studio sessions |
| Small config / PPE metadata | Raw HBM decks (`*.k`, `*.dyn`), `.inp` plate meshes, `.out`/`.log` |

This is very achievable — the publishable set is already tiny:

| Artifact | Size |
|---|---|
| `cache/hbm_body_cache_f05_standing.npz` | 27.6 MB |
| `cases_generated/F05_Standing_torso_armored_plate_preliminary_gui_case.npz` | 22.4 MB |
| **Total** | **~50 MB** |

So the published repo is **~50 MB against today's 1.48 GiB** — a ~97%
reduction, with no loss of GUI functionality. (This table originally also
listed `cache/hbm_body_cache.npz`, `cache/hbm_body_cache_m50_standing.npz`,
`cache/plate_surface_cache.npz`, and `cases/*/ResultsForGUI/*.npz` -- all
stale/orphaned artifacts from the pre-rewrite `app/` and old M50 prototype
case, deleted in §4.7's cleanup pass. The single `*_gui_case.npz` file
above is the actual current equivalent: it already bundles the projected
displacement, plate render data, and case metadata the GUI needs into one
file, via `postprocess/gui_case.py`.)

Key point: **the GUI never needs the raw HBM decks at runtime.** `cache/*.npz`
*is* the extracted, GUI-ready body (12–28 MB), produced once from the 100–230 MB
decks. The decks are a build-time input, not a published artifact. Note this
inverts the current `.gitignore`, which ignores `*.npz` and tracks `*.dyn` —
exactly backwards for this goal.

**Actions — non-destructive, done:**
- `.gitignore` rewritten around the table above: ignores `*.xplt`, `*.feb`,
  `*.fsm`, `*.k`, `*.dyn`, `*.inp`, `*.out`, `*.log`, `*.pyd`,
  `cases/**/jobs/`, with `!cache/*.npz` and
  `!cases/**/ResultsForGUI/*.npz` negations placed **after** the broad rules so
  they win. Verified with `git check-ignore -v`: the deformation maps and body
  caches all resolve as publishable, and every solver/deck file matches an
  ignore rule.

**Actions — still to do:**
- Move build-time bulk data behind a configurable `GUARDS_DATA_ROOT` so the
  repo never needs it present.
- Guarantee the deformation maps are reproducible from a solved `.xplt` via
  `postprocess/`, so publishing them is a convenience, not a dependency.

**Actions — destructive, yours to run and push:**

> **Critical: `.gitignore` does not untrack anything already committed.**
> Verified here — `git check-ignore` reports tracked files as *not* ignored
> even when a rule matches them, because ignore rules only apply to untracked
> paths. The new `.gitignore` prevents *future* large files; it does not remove
> the 1.48 GiB already in history. Two separate steps are needed:

1. `git rm --cached` the already-tracked solver/deck files, so they stop being
   tracked going forward (one commit; does not shrink history).
2. History rewrite (`git filter-repo --strip-blobs-bigger-than 5M`, or a fresh
   initial commit) to actually reclaim the space. This rewrites every commit
   hash and must be coordinated with anyone who has cloned the repo.

I have not run either, and have not committed or pushed anything.

### 4.4 GUI

One PySide6 + pyvistaqt window:
- Select an HBM model and a PPE file. The body site is **auto-detected**
  (§1.3) and shown pre-selected with its confidence and the runner-up score,
  and can be overridden.
- Case browser listing solved simulations found under `cases/`, switchable
  without restarting.
- The existing three-viewport layout is worth keeping: main body view, a
  displacement-magnitude inset, and a PPE-approach inset.
- Reuse the existing caching approach (`cache/*.npz`, ~20–25 s first parse then
  near-instant) keyed on model + site rather than on a hardcoded filename.
- A verification panel showing the `VERIFICATION.md` metrics for the selected
  case, pass/warn/fail coloured.

### 4.5 Two-stage pipeline: `main_1.py` → solve → `main_2.py` → GUI — **implemented and verified**

The entry point is split into two scripts, since a solve happens in
between them (in FEBio Studio or `febio4.exe`, outside this codebase's
control) and each half has its own settings:

- **`main_1.py`** (was `main.py`): seat the PPE, extract the torso volume,
  derive the load curve, assemble the `.feb` -- unchanged pipeline, just
  renamed for clarity now that there's a second stage.
- **`main_2.py`**: after you've solved the `.feb`, read the solved
  displacement and project it onto the full HBM model's own node ids, for
  the GUI (`app/viewer_app.py`) to display.

**Why not read the solved `.xplt` directly?** Investigated directly:
the installed `febio-python` (0.2.1) package's xplt reader only recognizes
plot file versions up to 52; FEBio 4.12 writes version 53. Patching the
version check alone is not enough -- traced further and confirmed the
STATE section's own framing/compression differs enough that the reader's
`search_block` can't locate it correctly even once the version gate is
bypassed (real, unresolved bug/format-drift in that third-party library,
not something under this project's control). **Fix**: `febio/assemble_case.py`
now requests FEBio's own first-class, documented plain-text
`<Output><logfile><node_data data="x;y;z;ux;uy;uz">` output on every case
(`<case>_node_displacement.txt`, one row per node per solved step) --
verified against a real solved case end to end. `main_2.py` reads this
instead of the `.xplt`.

> **A real bug found and fixed here: the log was scoped to the whole
> model, not just the torso.** An earlier version's `node_data` block had
> no `node_set` attribute, so FEBio logged *every* node -- including the
> rigid plate's own ~619k nodes -- at every solved step, even though
> `postprocess.project_displacement` only ever reads the torso's ~47.5k
> nodes (a rigid body's own "deformation" isn't meaningful to project onto
> the HBM). A real solved case produced a **~780MB text file**, ~93% of it
> plate-node rows nothing downstream ever reads -- caught when the user
> pointed out the file size directly. Fixed by adding a `TorsoNodes`
> `<NodeSet>` (ids `1..n_torso`) and referencing it via
> `node_set="TorsoNodes"` on the `node_data` block -- confirmed directly
> (a tiny synthetic multi-domain `.feb`) that FEBio 4.12 honors per-record
> `node_set` scoping on logfile output. Cuts the log to ~50-60MB, ~7% of
> its previous size. New regression test:
> `tests/test_assemble_case.py::test_node_data_log_is_scoped_to_torso_only`.

**Node-id mapping.** FEBio's own case file numbers nodes locally and
sequentially (torso nodes `1..n_torso`, plate nodes after -- see
`febio/assemble_case.py`), not by the original HBM node id. `fitting/body_volume.py`'s
`TorsoVolume` now also carries `node_ids` (the original LS-DYNA node id for
each local index, in order), and `build_case()` writes a sidecar
`<case>_node_map.npz` next to the `.feb` so `main_2.py`
(`postprocess/project_displacement.py`) can map the solved local
displacement back onto original HBM node ids without re-deriving anything.

**Verified end-to-end** on a freshly solved real case: 47,544 torso nodes
projected, `state_time=1.0345` (matches the `.log`'s own recorded time),
`max|u|=5.518mm` (consistent with VERIFICATION.md §5.2/§5.3's expectation
that peak skin displacement exceeds plate travel due to tissue bulging).
11 new unit tests (`tests/test_node_log.py`, `tests/test_project_displacement.py`),
all passing, using small synthetic fixtures (not the real ~890MB log --
real logs are large since FEBio's plain-text format is far less compact
than the binary `.xplt`; noted here in case that ever needs revisiting).

> **Known gap, not yet fixed: GUI full-body node-id mismatch.** Running
> the GUI (`app/viewer_app.py`) against a freshly projected displacement
> field surfaced a real problem: its cached full-body reference mesh
> (`cache/hbm_body_cache.npz`) was parsed from the *old* messy client
> source tree (`I-PREDICT_v0.12_90-90-90_*_split.k`, via
> `app/hbm_reader.py`'s `_parse_full_body`) and uses a **different node id
> numbering** than this pipeline's clean `HBM/<model>/{Nodes,Elements}.k`
> folders. Confirmed directly: only 1,387 of 49,006 target-region nodes in
> the cached mesh matched an id in the freshly projected 47,544-node
> displacement field ("47619 of 49006 target nodes have no solved
> displacement" printed by the GUI itself) -- the rest is coincidental
> overlap, not real alignment. Compounding this,
> `app/hbm_reader.py`'s hardcoded `TARGET_SKIN_PID = 3000451` does not even
> match `config/sites.py`'s own torso-skin pid (`2000501`) -- two entirely
> different PID numbering schemes from two different source exports of
> "the same" model. **Fix (not yet done)**: rebuild `app/hbm_reader.py`'s
> full-body parse to read *all* shells (no pid filter) directly from the
> same clean `HBM/<model>/{Nodes,Elements}.k` folder the rest of the
> pipeline already uses (same `iter_wanted_cards` approach as
> `fitting/body_volume.py`), which guarantees node-id alignment with
> `main_2.py`'s projected displacement by construction and removes the
> dependency on the old messy source tree + stale cache entirely. Tracked
> as `g4-gui-node-id-mismatch`.

> **Resolved (2026-09-22).** `app/hbm_reader.py` was rewritten exactly as
> described above: it now reads `HBM/<model_key>/{Nodes,Elements}.k`
> directly (via `core.lsdyna.bridge.iter_wanted_cards`, same as
> `fitting/body_volume.py`), with `extract_target_region()` taking a set of
> pids from `config.sites.resolve_site(...)` instead of a hardcoded PID.
> Per-model-key cache files (`cache/hbm_body_cache_<key>.npz`); the old
> single unkeyed `cache/hbm_body_cache.npz` was deleted as stale. This also
> resolves `g4-gui-node-id-mismatch`.
>
> A second, related bug was caught in the same pass: the case's own solved
> displacement covers the *whole* torso volume (skin+flesh, from
> `fitting.body_volume.TorsoVolume`), but the GUI's rendered target region
> is skin-only -- a reordered subset, not the same array. Adding
> `app/gui_case_loader.py`'s `gather_indices_for_target()` (dense id→index
> lookup, same pattern as `_node_id_to_index_map`) fixed a real broadcast
> error this surfaced immediately on first real GUI launch.
>
> **GUI fully implemented and verified working end-to-end**, dropdown
> selection through real solved-case rendering: `app/viewer_app.py` now has
> HBM-model/PPE dropdowns (populated by scanning
> `cases_generated/*_gui_case.npz` via `postprocess.gui_case.discover_gui_cases`,
> no filename parsing or separate registry needed) driving the three-view
> layout from §4.4 (main view, displacement-magnitude inset, PPE-approach
> inset) — restored to match the original prototype's layout per explicit
> user request, after an earlier single-view simplification. `main_2.py`
> takes just a case's base name and finds every sidecar file itself.
>
> Two more real, easy-to-miss bugs were caught by actually *launching* the
> GUI and inspecting the render (not just running unit tests against
> synthetic data):
> 1. The rigid PPE's analytic motion reconstruction (`app/plate_render.py`,
>    `plate_points_at_time`) had its sign backwards -- moving the plate
>    *away* from the body as load increased, not into it. Root cause: a
>    sign-cancellation in `febio/assemble_case.py`'s own applied
>    displacement (`dv = -push_direction`, times a load curve whose value
>    is itself `-total_travel_mm`) was not carried through correctly into
>    the GUI's independent reconstruction of that same motion. Fixed by
>    re-deriving the sign directly from `assemble_case.py`'s own comment,
>    not by guessing from what "looked right".
> 2. The main view rendered completely flat/washed out (no visible shading
>    or body contour) while the two insets looked normal. Root cause:
>    `self.plotter.clear()` (called every time a case loads, including the
>    very first launch) wipes the renderer's entire VTK light kit along
>    with its actors; with zero lights left, VTK silently falls back to a
>    single default headlight always aligned exactly with the view
>    direction, which saturates diffuse shading to ~100% everywhere. Fixed
>    by calling `self.plotter.enable_lightkit()` right after `clear()`.
>    (The insets were unaffected because they are each built in their own
>    throwaway off-screen `Plotter`, never cleared.)

> **Fixed (2026-09-22): body rendered upside-down by default on every
> launch.** All three views' camera "up" vector was computed as
> `up[np.argmax(body_extents)] = 1.0` -- "whichever global axis spans the
> largest body extent, assumed positive." That guess is wrong exactly half
> the time: this HBM export frame's real superior (head) direction is
> `-Z`, not `+Z` (see `fitting.seat_plate.DEFAULT_BODY_UP_AXIS`, already
> established and used elsewhere in the seating pipeline), so every render
> came up head-down until the user manually rotated it. Fixed by using
> `DEFAULT_BODY_UP_AXIS` directly instead of guessing from extents, in
> both `_compute_frontal_camera` and `_compute_plate_view_camera`.
> Verified visually on F05 (and the fix is model-family-generic, not
> F05-specific, since M50 shares the same export-frame convention).

> **Added (2026-09-22): "Export Current State" button.** Exports whatever
> deformation the slider currently shows (any load fraction, not just a
> real solved step) back into mesh files, per explicit user request:
> "no matter how much deformation is being viewed... export the meshes
> for the torso flesh, torso skin, and the PPE, and throw them back into
> the human body model." Two phases, matching the button's own status
> messages:
>
> 1. **"Meshes Exported"** (`postprocess.export_state.export_meshes`) --
>    three small, standalone `.k` files: deformed torso skin, deformed
>    torso flesh, PPE at its current position. Re-derives torso
>    connectivity via a fresh `fitting.body_volume.extract_torso_volume`
>    call (same one `main_1.py` uses -- deterministic or the same static
>    source files, so it reproduces the exact node order `case.node_ids`
>    already has, though the code still gathers explicitly by id rather
>    than assuming that, same defensive pattern as
>    `gather_indices_for_target`).
> 2. **"FE Model Built with PPE"** (`export_state.build_fe_model_with_ppe`)
>    -- ONE combined `.k` file: the entire HBM model, every other node and
>    part completely untouched, with the torso's coordinates patched to
>    the deformed state and the PPE appended as a new part. This is the
>    literal "throw them back into the human body model" deliverable.
>
> **Two real bugs caught building this, both by actually running it
> end-to-end against real HBM data rather than trusting the design on
> paper:**
> 1. *PID collision.* An early version computed the new PPE part's pid as
>    "max pid seen in the solid element block so far, + 1" and injected it
>    right before the `*ELEMENT_SHELL` keyword. This collided with a real
>    pid (7100046) that only appears later, in the *shell* block, which
>    happens to go higher than any solid pid on F05's real data. Fixed by
>    scanning the *entire* elements file for the true global max pid/eid
>    (or, in the final version below, folding this into the same single
>    read pass) before deciding the new part's numbering, and appending
>    the new part at the very end rather than mid-file.
> 2. *Format assumption.* The first working version assumed every
>    `HBM/<model>/{Nodes.k,Elements.k}` is in `config/hbm_normalize.py`'s
>    own simple always-comma format, and used a naive `line.split(",")`
>    reader for speed (byte-for-byte passthrough of untouched rows). This
>    crashed immediately on `M50_Standing`, whose folder instead holds a
>    more raw LS-DYNA-style deck (`$` comments, other card types like
>    `*ELEMENT_BEAM_ORIENTATION`, different field spacing) -- not
>    something a plain comma-split can safely skip over. Fixed by reading
>    via `core.lsdyna.bridge.iter_wanted_cards` (the same parser
>    `fitting.body_volume`/`app.hbm_reader` already use for these exact
>    files) instead, and writing output in one controlled, always-comma
>    format rather than trying to preserve each source file's own original
>    formatting. Verified against both F05 (~84s per full export) and M50
>    (~175s -- slower, since its deck has more card types to classify).
>
> `tests/test_export_state.py` (4 tests, synthetic fixtures including a
> deliberately M50-style `$`-commented deck and a pid layout that
> specifically re-triggers bug 1 above if it ever regresses) all pass.

> **Refined (2026-09-23): cosmetic GUI polish + export responsiveness.**
> Per explicit user feedback:
> - Dropdowns show a friendlier display string ("F05 - Standing", "Armored
>   Plate") while every actual lookup still uses the combo's stored raw
>   key (`QComboBox.addItem(display, userData=key)` / `.currentData()`),
>   never the display text -- so this is purely cosmetic, no behavior
>   change to case discovery/loading.
> - Window title -> "GUARDS PPE Viewer". Load label dropped its `(t=...)`
>   term (now just `Load / Closure: NN%   max|u|=X.Xmm`). Status bar
>   dropped node count/solved-step count/final_time (now just
>   `<hbm> / <ppe>  |  target: <site>`).
> - Added a `QProgressBar` (bottom-left, hidden except during an export)
>   driven by `postprocess.export_state`'s two functions each reporting
>   their own independent 0-1 `progress(message, fraction)`, blended by
>   `app.viewer_app._make_export_progress_fn` into one overall 0-100% bar
>   via fixed phase weights (35% / 65%) -- so neither export function
>   needs to know about the other's existence or the GUI's bar at all.
>
> **A real responsiveness bug was caught building the progress bar, not
> just a missing nice-to-have**: with only ~10 total progress callback
> invocations across a 160-180s export (each callback call is also what
> pumps the GUI's Qt event loop, via `QApplication.processEvents()`), the
> window went unresponsive enough for Windows to show "(Not Responding)"
> during the multi-minute stretches between calls -- most severely the
> ~90-150s single unbroken loop reading every element in the whole-body
> deck. Fixed by adding *optional*, fully backward-compatible periodic
> progress reporting inside both `fitting.body_volume.extract_torso_volume`
> (new `progress` kwarg, default `None` -- every existing caller
> unaffected) and `export_state.build_fe_model_with_ppe`'s own element
> loop: a cheap upfront raw-line count (not full card parsing) as a rough
> denominator, then a callback fired every ~0.5% of that count. Reduced
> the longest gap between UI updates from **63.6s -> 23.4s** on a real
> F05 export (not fully eliminated -- the shell/solid element loops don't
> have a cheap accurate total to report a smooth fraction against, so
> those two just ping periodically at a fixed fraction to keep the event
> loop pumped, rather than showing a moving number). Verified directly by
> instrumenting `QProgressBar.setValue` in a test harness and measuring
> the actual gaps between calls, not just eyeballing the running GUI.
>
> **Plate/PPE geometry is now always read from the actual solved `.feb`,
> not the pre-solve sidecar (2026-09-24).** Reported directly: a
> manually-repositioned `F05_Seated` case (fine-tuned in FEBio Studio
> before solving) rendered in the GUI with the plate back in its
> *original*, pre-adjustment position -- a real **162mm** discrepancy on
> that case, confirmed by direct comparison. Root cause:
> `postprocess/gui_case.py`'s `build_gui_case()` sourced the plate's rest
> geometry/prescribed motion from `<case>_plate_render.npz`, a sidecar
> `febio.build_preliminary_case.build_case()` writes **before the case is
> ever solved** -- nothing keeps it in sync with a later manual edit (or
> even a later `build_case()` re-run with different settings; also
> observed directly on `F05_Standing`).
>
> Fix: new `postprocess/plate_from_feb.py`'s `read_plate_render_from_feb()`
> reconstructs the plate's rest node coordinates + prescribed rigid
> displacement directly from the solved `.feb` -- the exact file FEBio
> actually consumed, so it's guaranteed correct regardless of any manual
> edit. This works because `febio/assemble_case.py` prescribes the plate
> as a rigid body with **translation-only** DOFs (rotation permanently
> locked via `PlateFixRotation`) -- its motion at every state is fully
> determined by two things, both plain XML: the rest `<Nodes>` coordinates
> and the `rigid_bc`/load-curve values.
>
> **Binary `.xplt` was investigated and deliberately rejected** as the
> read source (even though it was the first thing asked for) after direct
> testing showed it genuinely is not safely readable here: the installed
> `febio-python` xplt reader's version gate rejects FEBio 4.12's plot file
> version (53) outright; patching that gate to accept it (mapping 53 to
> the same reader used for 52) gets past the *version* check but then
> silently returns **zero states** -- traced to a second, independent bug
> in that library (it reads `<HDR_COMPRESSION>` but never passes it to its
> own state-decompression call); patching *that* too finally reaches real
> decompression, which then fails outright (`zlib.error: unknown
> compression method`) -- a further, unresolved format/library mismatch.
> Given the plate is provably a pure rigid translation (see above), the
> `.feb`'s own rest geometry + prescribed BC is exactly as faithful to
> "what was actually solved" with none of that risk, so it was used
> instead.
>
> **A second real wrinkle found only by testing against the actual
> manually-edited file, not a synthetic one**: opening/re-saving a
> generated `.feb` in FEBio Studio renames the plate's own `Elements`/
> `MeshDomains` block (e.g. `"Plate"` -> `"Part4"`) and can split node
> coordinates across **multiple** `<Nodes>` blocks (observed: Studio moved
> the plate's nodes into a separate `<Nodes name="Detached1">` block, its
> own auto-organization for a disconnected mesh island). Fixed by locating
> the plate's domain via its **rigid-body material** (`<material
> type="rigid body">`, matched through `<MeshDomains>`'s `mat=` attribute)
> rather than a hardcoded name, and by searching every `<Nodes>` block in
> the file, not just the first.
>
> `build_gui_case()` now prefers the solved `.feb` whenever present
> (`postprocess/plate_from_feb.py`), falling back to the old
> `plate_render.npz` sidecar only if the `.feb` is missing -- verified this
> fallback still passes every existing `tests/test_gui_case.py` test
> unchanged. Verified end to end on the real, manually-edited
> `F05_Seated` case: rebuilt `main_2.py`'s GUI-case output, confirmed the
> plate now sits with **~0mm gap** against the torso skin at rest (was
> floating 162mm away under the stale sidecar) and still on the anterior
> side per `fitting.body_axes` (dot product +171mm), plus a direct
> offscreen-render screenshot showing it seated correctly over the chest.

### 4.6 User-editable settings in `main_1.py` — **implemented**

The handful of knobs that get tuned per study live in one clearly-marked block
at the top of `main_1.py`, not scattered across config modules. Everything else is
derived.

```python
# ─── USER SETTINGS ──────────────────────────────────────────────────────────
HBM_MODEL_KEY = "F05_Standing"   # folder name under HBM/ -- config/hbm_models.py
SITE = "torso"                   # only "torso" exists today -- config/sites.py
PPE_KEY = "armored_plate"        # config/ppe.py registry

INDENTATION_MM      = 4.25   # how far the PPE presses in past first contact
TARGET_STANDOFF_MM  = 0.1    # rest gap for contact detection; see §2.4.
                             # Must stay > 0; auto-raised (logged) if below
                             # the mesh's own sagitta floor -- fitting/standoff.py.
INITIAL_CONTACT_MM  = 0.5    # target travel on the FIRST solved step -- see §2.4.
                             # Deliberately separate from TARGET_STANDOFF_MM.
VERTICAL_OFFSET_MM  = 0.0    # move the seated plate up/down before it's
                             # fit to the body -- §2.3c. + = superior (up).

OUTPUT_PATH = None           # None = auto-named under cases_generated/
HBM_ROOT    = Path("HBM")
# ──────────────────────────────────────────────────────────────────────────
```

- `INDENTATION_MM = 4.25` is the fixed default for every case so results stay
  comparable across anthropometries and PPE. **This matters: the existing cases
  do not currently share a travel distance** (F05 7.25 mm, M50 6.25 mm,
  reference 3.75 mm post-contact), which is why they are not comparable today.
  Standardising here and re-running is the fix. Note this is the *post-contact*
  indentation, not total travel -- see §2.4's worked example: with
  `TARGET_STANDOFF_MM = 0.1`, total travel is `4.35mm`, about 13% more
  post-contact travel than the reference's own `3.75mm` (a deliberate,
  acknowledged difference, not a bug -- kept as-is per user decision).
- `TARGET_STANDOFF_MM = 0.1` with `INITIAL_CONTACT_MM = 0.5` and
  `INDENTATION_MM = 4.25` gives `total_travel_mm = 4.35mm`,
  `step_size = 0.5/4.35 = 0.114943` → `time_steps = 9` (§2.4's
  `febio/loadcurve.py`).
- `INITIAL_CONTACT_MM = 0.5` is the parameter whose absence (conflated with
  `TARGET_STANDOFF_MM`) caused the original non-convergence bug -- see §2.4
  for the full writeup.
- `VERTICAL_OFFSET_MM` is the fix for exactly the kind of problem a visual
  check can catch that no automated metric would: a plate that renders
  sitting low on the stomach when it should cover the chest. See §2.3c for
  how this is implemented (and the real aperture-problem interaction with
  ICP that had to be fixed for it to actually work, not silently get undone).
- `main_1.py` calls `febio.build_preliminary_case.build_case(...)` with these
  settings — that function (not `main_1.py` itself) is where the actual
  seat → extract-torso-volume → derive-load-curve → assemble-case pipeline
  lives, so it stays callable from tests/other scripts without re-running
  `main_1.py`'s module-level code.
- The values actually used are printed as the pipeline runs (seating result,
  load curve numbers, output path), so a result is never ambiguous about
  what produced it.

> **Correction from measurement.** An earlier draft assumed the existing cases
> stand off from the body and that sub-mm seating would therefore *increase*
> deformation. Measurement shows the opposite: **both existing cases are already
> seated essentially in contact** (M50's minimum rest gap is 0.281 mm), so
> sub-mm seating changes their character very little. What *does* change is
> travel — F05 currently uses **7.25 mm** and M50 **6.25 mm**, neither matching
> the reference's 4.25 mm. Standardising on `INDENTATION_MM = 4.25` will produce
> **less** deformation than both current cases, and for the first time make them
> directly comparable. See `VERIFICATION.md` §5.2 for the measured numbers and
> for the initial-overlap defect found in the M50 seating.

### 4.7 Repository cleanup pass — **done (2026-09-22)**

With the GUI working end-to-end, the repo was swept for dead code so it
stays small and packageable (explicit user goal: eventually importable
into the client's own GUI codebase). Kept flat top-level folders per
explicit user decision (no nested namespace package) -- this pass only
removes what turned out to be genuinely unused, it does not restructure
what's left.

**Deleted (confirmed zero real importers first, not just "looks unused"):**
- `pipeline/` (12 scripts) and `gui/` (empty placeholder) -- both already
  called out for deletion in §4.2 above; superseded by `config/hbm_models.py`
  + `config/sites.py` + `fitting/` + `febio/` and by `app/` respectively.
- `app/client_parser_bridge.py` (dead re-export shim, self-documented as
  "removed once app/ is retired", only ever used by the now-deleted
  `pipeline/extract_region.py`) and `app/displacement_field.py` (old
  prototype's npz reader, fully superseded by `app/gui_case_loader.py`).
- The legacy single-step API in `postprocess/project_displacement.py`
  (`project_case_displacement`, `ProjectedDisplacement`,
  `write_displacement_field_npz`) -- kept "for backward compatibility"
  during the GUI rewrite, but nothing in the real pipeline calls them
  (only `project_case_all_steps` is used, by `postprocess/gui_case.py`).
  `_map_step_to_original_ids` and `project_case_all_steps` kept; test
  coverage for the removed functions' behavior (error paths, plate-node
  exclusion) preserved by rewriting the affected tests against the
  surviving API instead of just deleting the coverage.
- Stray root files already flagged in §4.2: `_skipped.face`, `_skipped.node`,
  `lspost.cfile`, `lspost.msg`, `hbm.dyn`, `smoke_test.py` (the last
  referenced a `rate_combo` widget that no longer exists post-rewrite --
  would have crashed if run).
- Unused `requirements.txt` entries: `matplotlib` (only used by the now-deleted
  `pipeline/` scripts) and `febio-python` (evaluated and explicitly rejected
  for xplt reading, per §4.5 above -- the decision not to use it doesn't
  require having it installed).
- Stale generated/cache data (all gitignored, regeneratable, not source):
  `cases_generated/*_final_symmetric*` (superseded by `_preliminary`, the
  current working case), `cache/hbm_body_cache.npz` (old unkeyed format,
  pre-`hbm_reader.py`-rewrite), `cache/plate_surface_cache.npz` (used only
  by the deleted `app/plate_mesh_reader.py`), `cache/hbm_body_cache_m50_standing.npz`
  (orphaned -- no corresponding `HBM/M50_Standing/` folder exists in this
  worktree at all).
- `cases/M50_Torso_ArmoredPlate/` reference data (per explicit user
  decision: F05 Standing only, for now) -- `cases/F05_Torso_ArmoredPlate`
  kept. `VERIFICATION.md` §5.2 and §7 updated accordingly (condensed the
  M50-specific forensic analysis into a shorter historical note; the
  F05-relevant lessons carried forward are still called out explicitly).

**Explicitly kept, not dead code despite looking unused on the currently
active path:** `core/registration/rigid_cpd.py` and `pca_prealign.py`
(CPD/PCA approach from the original Goal 2 exploration) -- per explicit
user decision, kept as a documented, tested alternative to
`point_to_plane_icp` (currently used by `fitting/seat_plate.py`), since CPD
"may be more effective... at some point." Both still have their own
passing regression tests (`tests/test_rigid_cpd.py`,
`core/registration/__init__.py`'s re-exports).

### 4.8 Documentation

- `README.md` — install, run, add a model, add a PPE, run a case end to end,
  where the manual FEBio Studio step sits.
- `VERIFICATION.md` — acceptance criteria (written; see that file).
- `AGENTS.md` — this plan.
- Module docstrings in the existing house style: *what it does*, then *why it
  is done this way*, with the evidence. That style is a genuine strength of the
  current `pipeline/` code and should survive the rewrite.

---

## 5. Decisions and remaining questions

Items 2–7 were resolved in review; item 1 is now moot (scope narrowed to
torso-only, no site detection needed); item 8 is a new open question raised
by building §1.1.

1. ~~**Site detection confidence margin**~~ — **MOOT.** Scope narrowed to
   torso + rigid plate only, per explicit instruction (no head PPE exists).
   §1.3's multi-site detection apparatus is deferred, not built, until a
   second site is actually needed — see §1.3 for the simplified
   always-`"torso"` current behaviour.
2. ~~**Head site parts**~~ — **MOOT**, same reason. `Thorax_Flesh` = 2000500,
   `Thorax_Skin` = 2000501 remain confirmed and in use.
3. **Target standoff** — **DECIDED: `TARGET_STANDOFF_MM = 0.1`.** FEBio
   penalty/AUGLAG contact wants a small *positive* gap (order 1e-1 mm), not
   zero. The floor is set by mesh faceting, not preference: on the current 7 mm
   skin mesh the sagitta is 0.041 mm, so 1e-2 and 1e-3 mm targets are below
   geometric resolution and meaningless (§2.4). 0.1 mm is ~2.5× sagitta —
   resolvable and tight enough to engage contact on step 1. `fitting/` must
   recompute this floor from whatever mesh it is actually given and warn if the
   requested standoff is unachievable, since decimation raises the floor
   (0.375 mm at 15 mm edges).
4. **Indentation target** — **DECIDED: fixed 4.25 mm for every case**, matching
   the reference, so cases stay directly comparable. Exposed as a user-editable
   setting at the top of `main.py` (§4.6) rather than buried in a config
   module, with per-pairing override still available for special cases.
5. **STL PPE** — **DECIDED: tetrahedralise into a solid on import** (§1.4), so
   every PPE is a volume mesh regardless of source format and there is one
   downstream code path. Requires watertight/manifold validation at load.
6. **GPU** — **DECIDED: CPU-only.** Use `pycpd`; do not take the `torch` /
   `torchcpd` CUDA dependency (~2.5 GB) for this project. Registration is on
   ~6000 voxel-downsampled points, which is fast on CPU, and the available GPU
   is an NVIDIA T600 (4 GB). When vendoring Morphing's `registration/cpd.py`,
   strip or guard the torch import path so the module does not require torch to
   load. `requirements.txt` already pins `pycpd==2.0.0`.
7. **Git history rewrite** — **DECIDED (scope).** The published artifact set is
   code + GUI-ready body models (`cache/*.npz`) + deformation maps
   (`ResultsForGUI/*.npz`) only — **no `.xplt`, `.feb`, or `.fsm`** (§4.3).
   That is ~68 MB vs today's 1.48 GiB. I will rewrite `.gitignore` to match,
   but **will not** run the history rewrite that actually reclaims the space,
   and will not commit or push. Still to confirm: `filter-repo` on the existing
   history vs. starting a fresh initial commit.
8. ~~**`M50_Torso_ArmoredPlate`'s flesh provenance**~~ — **RESOLVED, and it's
   simpler than the open question assumed.** Checked both solved `.feb`
   files directly: **neither** `cases/F05_Torso_ArmoredPlate` nor
   `cases/M50_Torso_ArmoredPlate` has a flesh `SolidDomain` at all — each is
   a deliberate skin-shell-only simulation (`ShellDomain` + Ogden `Skin`
   material + the rigid plate's own `SolidDomain`, nothing else), exactly as
   their own filenames say (`..._skin_only`). So there was no flesh to morph
   in from elsewhere; these two existing cases simply are not the same kind
   of simulation as the gold standard (`NewHex_close.feb`, which has a real
   `Flesh` + `Skin` pair). **This means `VERIFICATION.md` §5.2's comparison
   of these two cases against the gold standard's numbers was comparing two
   different physics setups** (shell-only stiffness vs. shell-over-solid) —
   corrected there. The two existing cases remain a valid pair to compare
   *against each other* (both skin-only), just not a stand-in for a full
   flesh+skin torso solve.

---

## 6. Sequencing

| Phase | Tasks | Gate |
|---|---|---|
| **0. Scaffold** | Package layout (§4.1); vendor Morphing + Sparse into `core/` with provenance | Imports clean; vendored code unit-tested in isolation |
| **1. Config** | §1.1, §1.2, §1.4, §1.5 | `cases/F05_Torso_ArmoredPlate` and `M50_Torso_ArmoredPlate` resolve purely from config |
| **2. Fitting** | §2.1–2.4, then §1.3 (site detection reuses the fit) | New fit reproduces the existing F05/M50 plate poses within tolerance; the armored plate is detected as `torso` against both models |
| **3. FEBio writer** | §2.5 | Generated `.feb` byte-compares on `<Control>`/`<Material>`/`<Contact>` against the gold standard |
| **4. Remesh** | §3.1–3.3 | Whole-torso mesh: 0 inversions, 0 orphans, skin/flesh connectivity asserted |
| **5. GUI** | §4.4 | Both existing cases browsable in one window |
| **6. Cleanup** | §4.2–4.3, §4.6–4.7 | No unused files; README accurate; repo size addressed |

Regression fixtures throughout: the two existing solved cases plus the
`Plate_Skin_Deformation` gold standard. Any change that alters their fitted
pose, generated `.feb` settings, or verification metrics must be justified.
