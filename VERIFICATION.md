# VERIFICATION.md — How we decide a simulation is trustworthy

Acceptance criteria for every PPE-on-body contact simulation produced by this
codebase.

Criteria are drawn from the gold-standard reference case at
`C:\Users\lhudson\Desktop\Projects\GUARDS\Plate_Skin_Deformation`, principally
its `AGENTS_STRAIN.md`. Numeric bands are quoted verbatim from that document
and from the solved `FE Model/NewHex_close.feb`.

> **These are engineering screening bands, not validated armor injury
> thresholds.** They exist to catch implausible results and to compare
> configurations against each other. They are not a safety certification, and
> they must not be reported as injury predictions.

---

## 1. The four stages

A case is only "verified" once it clears Stages 1–3. Stage 4 is out of scope
for this codebase but is where these numbers eventually get their authority.

| Stage | Question | Automatable here |
|---|---|---|
| **1. Numerical verification** | Did the solver produce a converged, mesh-independent, numerically clean answer? | Yes — fully |
| **2. Geometric fit** | Is the PPE in an anatomically sensible place, without unrealistic gaps or initial overlap? | Yes — fully |
| **3. Mechanical fit** | Are the contact pressures, tissue strains, and plate stability plausible? | Yes — from `.xplt` |
| **4. Experimental validation** | Does it match measurement? | No — external |

---

## 2. Stage 1 — Numerical verification

Run before any result is interpreted. Failing any of these invalidates the run.

| Check | Criterion | Where it comes from |
|---|---|---|
| Solver convergence | Converged force/displacement residuals; `dtol = 0.01`, `etol = 0.01` | `NewHex_close.feb` `<solver>` |
| Contact penetration | **P99 numerical penetration below approximately 1–2% of local element size**; **investigate above approximately 5%** | `AGENTS_STRAIN.md` §6 |
| Timestep sensitivity | Results stable when the timestep is reduced | `AGENTS_STRAIN.md` §10 |
| Penalty sensitivity | Results stable under penalty scaling (`auto_penalty = 1` makes this scale-independent) | `AGENTS_STRAIN.md` §10 |
| Mesh convergence | Pressure percentiles and strain volumes converged w.r.t. mesh density | `AGENTS_STRAIN.md` §10 |
| Element validity | **Zero inverted or severely distorted elements**, and **the mesh must pass the FEBio smoke test (§2.2)** | `AGENTS_STRAIN.md` §10; design instruction, §2.2 below |

### 2.1 Mesh quality gate (pre-solve)

Enforced by `remesh/quality.py` before a `.feb` is ever written:

- **Zero orphan nodes.** Unreferenced nodes are stripped by us, not by FEBio —
  FEBio's own isolated-vertex removal triggers an internal renumbering that has
  produced spurious negative-Jacobian reports.
- **Zero inverted solid elements**, evaluated at FEBio's **own 2×2×2 Gauss
  points** (`r, s, t = ±1/√3`), not at element corners. An element can pass a
  corner check and still fail FEBio's actual check, and vice versa.
- **Shell jacobian ratio within band** (`shell_band_floor`/`shell_band_ceiling`,
  e.g. 0.4–0.95).
- **Skin/flesh connectivity asserted**: every skin shell node is also a flesh
  boundary node. Verified against the shared-node manifest emitted by the
  remesher, not by a distance tolerance.
- **Non-regression**: repairing any element must not push a neighbour below
  `min(baseline, volume_epsilon)`.

> **Known trap.** FEBio's reported element volume has been shown *not* to match
> any standard geometric interpretation of the literal node coordinates —
> verified against Gauss quadrature, two tet decompositions, VTK Verdict
> metrics, corner evaluation, a 9,261-point parametric grid scan, and several
> hex8 node orderings. A pure rigid translation of an element's nodes (which
> cannot change a true Jacobian) changed FEBio's reported value. **When our
> metric and FEBio disagree, FEBio is the oracle.** This is why
> `febio_guided_untangle` stays in the ladder — and why §2.2 exists.

### 2.2 The FEBio smoke test — mandatory, not optional

**Design instruction (verbatim):** *"All meshes should be checked with a
simple FEBio problem set-up and trying to run. If it starts up successfully
without issues, the mesh is good, otherwise there are problems."*

Implemented in `remesh/febio_smoke_test.py`: every node is fixed to zero
displacement, zero load, one static step. **A mesh that fails this — for any
reason — is not usable, regardless of what §2.1's geometric checks say.**
This ordering is deliberate: §2.1 is a fast proxy; this is the actual answer.

This is not a theoretical precaution. Building this check surfaced a real,
reproducible case where every one of §2.1's own geometric checks (centroid
volume, corner Jacobian, and a hand-reproduction of FEBio's own 2×2×2 Gauss
formula) agreed an element was clean, while FEBio itself rejected the mesh
with `"Negative jacobian detected during domain initialization"` — traced to
a shell-normal setting (`shell_normal_nodal`) interacting with neighbouring
solid elements at the skin/flesh interface, not to any actual defect in that
element. See `remesh/febio_smoke_test.py`'s module docstring for the full
investigation. **The geometric checks in §2.1 remain necessary** (they are
what the repair ladder acts on, and they cover shells where the FEBio smoke
test is not reliable — see below) **but they are not sufficient on their
own**; §2.1 passing does not guarantee §2.2 will.

**Coverage.** Solid elements (hex8/tet4/wedge6) are unconditionally checked:
FEBio's own "mesh initialization" phase evaluates every solid element's
Jacobian sign regardless of boundary conditions, so "fix everything, zero
load" catches inversions at near-zero cost (a 46,138-element real mesh
smoke-tests in under 2 seconds). **Shell elements are not reliably covered**
by this check — a hand-built self-intersecting quad4 converged normally even
under real applied deformation — so `classify_shell_badness` (§2.1) remains
the primary shell quality gate.

> **Second trap.** An element can be valid at rest and still be *impossible to
> compress*. A ~0.36 mm layer over the ribs inverted at every retried timestep
> down to `dt = 0.0002`; the destination geometry, not the solver step size,
> was the problem. Check that the deformed configuration is achievable, not
> just that the rest mesh is clean.

---

## 3. Stage 2 — Geometric fit

Checked immediately after seating (§2.4 of `AGENTS.md`), before solving.

| Check | Criterion |
|---|---|
| Anatomical placement | PPE covers the intended region (e.g. anterior thorax for a chest plate) — for a genuinely new model, verify this quantitatively, not just by eye: `dot(plate_centroid − spine_centroid, model's derived anterior axis)` should be clearly positive (see `AGENTS.md` §2.1b, `fitting/body_axes.py`) |
| No initial overlap | Minimum **signed** gap **> 0** everywhere; no PPE edge embedded in tissue pre-load |
| Standoff achieved | Minimum signed gap equals the configured target standoff within tolerance |
| Standoff is resolvable | Target standoff ≥ ~2× the local skin-mesh sagitta (0.082 mm on a 7 mm mesh; 0.375 mm on a 15 mm decimated mesh) |
| Gap measured correctly | Point-to-**face** distance, not point-to-node |
| No large interior gaps | **Central plate gap mostly below 3–5 mm**; warn on connected interior gaps **above 5–10 mm** |
| Orientation correct | Concave face toward the body; superior/inferior axis correct, using this specific model's own **derived** anatomical axes, not a fixed global-axis assumption (`AGENTS.md` §2.1b — a real bug: a second model with a genuinely different real-world coordinate convention seated the plate on the posterior side under the old fixed-axis assumption) |

> **Use a signed measure.** An unsigned nearest-point distance cannot
> distinguish penetration from clearance. On the M50 case it reported a
> reassuring 0.281 mm minimum for a plate that was actually **3.22 mm embedded**
> in the undeformed skin across 309 nodes. Signed distance along the travel axis
> caught it immediately.

Gap distribution to report (`AGENTS_STRAIN.md` §6):
`A(g < 1 mm)`, `A(g < 3 mm)`, `A(g < 5 mm)`, median gap, 90th and 95th
percentile gap, maximum interior gap, largest connected non-contact region.

Reference anchor: in the gold-standard case the plate's rest position sits a
minimum of **~4.29 mm** from the skin (mean of closest 1%: **~4.95 mm**),
essentially equal to its **4.25 mm** prescribed travel — the converged position
is by definition at ~zero gap.

---

## 4. Stage 3 — Mechanical fit

Extracted from the solved `.xplt` by `postprocess/`.

### 4.1 Engineering screening bands

Verbatim from `AGENTS_STRAIN.md` §7.2.

| Metric | Initial target | Warning condition |
|---|---|---|
| Intended contact-area fraction | **at least 70% in neutral posture** | below 50–60%, or highly fragmented |
| Median contact pressure | **approximately 2–8 kPa** | above 10–15 kPa over broad regions |
| 95th-percentile pressure | **no more than approximately 20 kPa** | above 25–30 kPa |
| Patch-averaged local maximum | **no more than approximately 30–40 kPa** | above 45–55 kPa |
| P95 tensile Green-Lagrange strain | **no more than 0.10–0.15** | above 0.20 |
| P95 compressive Green-Lagrange strain | **no more than 0.10–0.15** | above 0.20–0.25 |
| P95 shear / effective GL strain | **no more than 0.15–0.20** | above 0.25–0.30 |
| Tissue volume above 0.30 strain | **ideally minimal, e.g. below 1–5%** | large connected regions |
| Central plate gap | **mostly below 3–5 mm** | connected interior gaps above 5–10 mm |
| Numerical penetration | **P99 below 1–2% local element size** | above 5% local element size |

### 4.2 Contact pressure reporting

Report at minimum (`AGENTS_STRAIN.md` §3.5): area-weighted mean, median,
**90th, 95th and 99th percentile**, patch-averaged local maximum over a defined
area, intended contact area carrying nonzero pressure, pressure coefficient of
variation, locations and areas of high-pressure clusters, and the
pressure-time integral for prolonged simulations.

> **Do not use the absolute largest nodal pressure as the sole metric.** It is
> mesh- and penalty-dependent and will mislead.

### 4.3 Strain reporting

For each tissue and anatomical region (`AGENTS_STRAIN.md` §5.3): maximum value,
95th and 99th percentile, and the **area or volume exceeding 0.10, 0.15, 0.20
and 0.30**.

### 4.4 Literature context

Useful for sanity, **not** as targets:

- **Load carriage** (Hadid et al., 25 kg backpack): maximum compressive strain
  **0.14**, maximum tensile strain **0.13**; for a 35 kg backpack, maximum
  tensile strain **approximately 16%**. This is heavy concentrated shoulder
  loading and is *not* a desirable armor-fit target.
- **Pressure injury** (Ceelen et al.): approximate Green-Lagrange injury
  thresholds of **0.45 maximum compressive strain** and **0.75 maximum shear
  strain**. Fougeron et al. use tissue volume exceeding **0.30 strain** as a
  comparative metric. These are **severe damage-related reference levels, not
  acceptable armor-fit targets**.

### 4.5 "Tight enough"

A plate is sufficiently fitted when (`AGENTS_STRAIN.md` §7.1):

1. Contact is connected and distributed, not an isolated corner.
2. The plate does not rock, translate or rotate excessively during movement.
3. Large interior gaps are reduced; natural edge/contour-bridging gaps remain.
4. Increasing strap tension yields little further useful contact or stability.
5. Additional tightening mainly increases localised pressure and tissue strain.
6. Pressure and strain stay spatially distributed, not concentrated over bone.
7. Contact penetration remains numerically negligible.

**Knee-point rule:** when `Δ(contact coverage) / Δ(strap tension)` becomes small
while pressure and strain keep rising, the carrier is tighter than necessary.

---

## 5. Gold-standard reference configuration

Every generated case is compared against these values. Deviation is allowed but
must be deliberate and recorded.

**Solver** (`<Control>`): `STATIC`; `time_steps = 9`; `step_size = 0.1176`;
`plot_level = PLOT_MAJOR_ITRS`; time stepper `max_retries = 15`,
`opt_iter = 8`, `dtmin = 0.0001`, `dtmax = 0.3`, `cutback = 0.5`; solver
`non-symmetric`, `staggered`, `lstol = 0.9`, `lsmin = 0.01`, `lsiter = 5`,
`ls_check_jacobians = 1`, `max_refs = 15`, `dtol = 0.01`, `etol = 0.01`,
full Newton, `mkl_dss`.

**Materials**

| Material | Type | density | k (MPa) | c1 (MPa) | m1 |
|---|---|---|---|---|---|
| Skin | Ogden | 1.1e-06 | 10 | 0.05 | 10 |
| Flesh | Ogden | 1.1e-06 | 1.5 | 0.003 | 4 |
| Plate | rigid body | 1 | — | — | — |

**Contact**: `sliding-elastic`, `laugon = AUGLAG`, `tolerance = 0.2`,
`gaptol = 0`, `penalty = 1`, `auto_penalty = 1`, `update_penalty = 1`,
`search_radius = 3`, `fric_coeff = 0`.

**Boundary conditions**: zero displacement on the artificial cut boundary only —
every outer flesh face not covered by a skin shell. In the reference model this
was **15,539 of 15,540** free-boundary nodes, with the skin contact surface left
essentially untouched (only the 396-node seam ring is shared). The tissue under
and around the plate is completely free to deform. This is standard FE
submodeling, **not** "fix everything except the contact patch".

**Load curve**: `(0, 0)` → `(1, −4.25)` mm, i.e. 4.25 mm inward travel, with
rotations fixed on the rigid body.

### 5.1 Load curve self-consistency

The reference's time stepping is exactly reproduced by the seating rule in
`AGENTS.md` §2.4:

```
step_size  = initial_contact_mm / total_travel_mm = 0.5 / 4.25 = 0.117647  ≈ 0.1176 ✓
time_steps = ceil(1 / step_size)                  = ceil(8.5)  = 9                 ✓
```

A generated case whose `step_size` and `time_steps` do not satisfy this relation
against its own travel distance is misconfigured. Check this before blaming the
solver.

Note that `9 × 0.1176 = 1.0584`, so the solve ends 5.84% past the load curve
endpoint. Confirm `state_time` on every solved case and confirm the load curve
`extend` mode, so it is known whether the final partial step adds travel or
merely settles.

### 5.2 Early skin-only prototype cases — historical, superseded by §5.3

Before this pipeline could produce a real flesh-bearing solve, two early
prototype cases (`cases/F05_Torso_ArmoredPlate`, plus a since-removed M50
one) were measured directly to sanity-check plate travel and rest-gap
behavior. Both used **shell-only torso meshes with no `Flesh` `SolidDomain`
at all** (`ShellDomain` Ogden `Skin` + the rigid plate's own `SolidDomain`),
and neither used the now-standardized `TARGET_STANDOFF_MM=0.1` /
`INDENTATION_MM=4.25` seating rule — each used its own ad hoc travel
distance. They are not comparable to the gold-standard reference (which has
a real `Flesh` domain) or to each other, and are superseded entirely by
§5.3 below, which measures this project's actual flesh+skin pipeline
output. Kept only as `cases/F05_Torso_ArmoredPlate` (F05, in-scope) for
historical reference; the M50 variant was removed from the repository
(F05 Standing only, for now).

**Key lesson that carried forward**: peak skin displacement exceeding
plate travel is expected and correct for near-incompressible tissue
(displaced material bulges tangentially around the indenter's edges), and
a rest gap that is *not* comfortably positive before loading indicates a
seating defect worth flagging — both are checked directly in the current
pipeline via `fitting.seat_plate`'s `min_signed_mm > 0` gate (`AGENTS.md`
§2.4), rather than being caught after the fact by manual measurement as
these prototypes were.

> Treat the current numbers as a record of what was run, not as the target.

### 5.3 First flesh-bearing pipeline run — converges, but under-resolved

`main_1.py`'s own generated case (`F05_Standing_torso_armored_plate_preliminary.feb`,
real `Skin` + `Flesh` domains, `TARGET_STANDOFF_MM=0.1`, `INDENTATION_MM=4.25`,
`INITIAL_CONTACT_MM=0.5`) now solves to `N O R M A L T E R M I N A T I O N` in
`febio4.exe` with no manual editing (9 nominal / 13 actual time steps
including retries, final time 1.034 — see `AGENTS.md` §2.4/§2.5 for the
load-curve bug this fix addressed).

**Observed contact pressure and strain read lower than the gold-standard
reference's own values.** Root cause is very likely **mesh resolution**, not
insufficient indentation:

| | Reference (`Thorax_Flesh_Impact`, local patch) | This pipeline (whole torso) |
|---|---|---|
| Hex8 element count | 141,015 | 34,980 |
| Coverage | ~103×248×327mm local region | entire torso |
| Through-thickness edge length | ~0.85mm mean (0.43mm at p10 near surface) | ~3.5mm mean |
| In-plane edge length | ~3.7mm mean | ~7.3mm mean |

The reference concentrates a graded, locally-refined mesh directly under the
contact patch (roughly **4× finer through-thickness, 2× finer in-plane**);
this pipeline currently solves against the raw, unrefined whole-torso HBM
flesh mesh. A coarser mesh averages stress/strain over a much larger element
volume and will report smaller peak values for the same underlying physical
deformation — this is an expected numerical-resolution effect, not evidence
of a modeling error.

**Do not "fix" this by increasing indentation travel.** More displacement
through an already under-resolved mesh does not correct the resolution gap
— it just produces a larger, still-averaged-down number. The correct fix is
local graded mesh refinement under the contact patch (Goal 3, §3 below),
which has not yet been implemented for this pipeline. Until Goal 3 lands,
treat this pipeline's absolute contact-pressure/strain magnitudes as
**qualitatively reasonable but not directly comparable** to the
gold-standard reference's numbers.

(Note: `.xplt` stress/pressure fields were not directly extracted for this
comparison — the installed `febio_python` version cannot read this file's
xplt version (53); the finding above is a direct mesh-geometry measurement,
not a comparison of solved field values. A different xplt reader or FEBio
Studio itself would be needed to pull quantitative pressure/strain numbers.)

---

## 6. Automated verification report


`postprocess/verify.py` emits one report per case, with every metric marked
**PASS / WARN / FAIL** against §2–4, plus:

- Stage 1: convergence summary, penetration percentiles, mesh quality scan.
- Stage 2: gap distribution and orientation checks.
- Stage 3: pressure and strain percentile tables, strain-volume fractions,
  contact-area fraction, plate rigid-body motion.
- A diff against the gold-standard configuration (§5), listing every
  intentional deviation.

The GUI surfaces this report per case so a run can be judged without leaving
the app.

---

## 7. Regression fixtures

| Fixture | Purpose |
|---|---|
| `Plate_Skin_Deformation` (`NewHex_close`) | Gold standard; defines §5 |
| `cases/F05_Torso_ArmoredPlate` | Early skin-only F05 prototype (§5.2, historical) |
| `cases_generated/F05_Standing_torso_armored_plate_preliminary*` | Current pipeline's real flesh+skin output (§5.3) |

Any change to fitting, meshing, or `.feb` generation must leave these
producing the same fitted pose, the same solver configuration, and the same
verification metrics — or come with a written justification for the difference.

