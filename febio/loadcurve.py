"""FEBio load-curve / time-stepping derivation from the seated PPE's own
standoff and desired indentation (AGENTS.md section 2.4).

The reference case (``NewHex_close.feb``) is self-consistent:

    total_travel     = 4.25 mm          (<pt>1,-4.25</pt>)
    initial_contact   = 0.5 mm          (first-contact increment)
    step_size         = 0.5 / 4.25 = 0.117647  ~= 0.1176   (the reference's <step_size>)
    time_steps        = ceil(1 / step_size) = 9            (the reference's <time_steps>)

Generalizing:

    total_travel_mm = target_standoff_mm + indentation_mm
    step_size       = initial_contact_mm / total_travel_mm
    time_steps      = ceil(1.0 / step_size)

**A real bug, found and fixed here.** An earlier version of this module used
``target_standoff_mm`` directly in place of ``initial_contact_mm`` in the
step_size formula above -- i.e. it assumed "how precisely the plate is
seated at rest" and "how far it should travel on the first solved step to
establish meaningful contact" are the same number. They are NOT: the
reference case's own construction only makes them look the same because its
plate happened to sit off by exactly 0.5mm at rest, and 0.5mm was *also* a
reasonable first-step size. Once this project started seating far more
precisely (``target_standoff_mm = 0.1``, for geometric accuracy -- see
``fitting.standoff``), reusing it as the first-step size shrank the plate's
actual first-step travel to ~0.1mm on a mesh with ~3.5-7mm element edges --
far too small relative to the mesh's own resolution to register as engaged
contact. Confirmed directly: a real solve with the conflated formula never
converged (residuals stuck at ~1e-20, "No force acting on the system"); the
same case with ``step_size`` corrected to move ~0.5mm on the first step
solved to completion in 9 steps, matching the reference exactly.

``initial_contact_mm`` is now a genuinely separate parameter (default 0.5mm,
matching the reference) -- decoupled from ``target_standoff_mm``, which
remains purely a *seating* parameter (fitting.seat_plate's own target gap).

Verified: plugging the reference's own numbers back in
(target_standoff_mm=0.5, initial_contact_mm=0.5, indentation_mm=3.75, so
total_travel_mm=4.25) reproduces step_size=0.1176 and time_steps=9 exactly.
With this project's own defaults (target_standoff_mm=0.1,
initial_contact_mm=0.5, indentation_mm=4.25, so total_travel_mm=4.35):
step_size=0.114943, time_steps=9 -- still 9 steps, but now driven by a
genuine 0.5mm first-step target instead of an accidental 0.1mm one.

**Load-factor overshoot.** ``ceil`` means the solve runs past the load
curve's own endpoint: ``time_steps * step_size`` is >= 1.0, not exactly 1.0
(the reference's own 9 * 0.1176 = 1.0584, a 5.84% overshoot -- confirmed by
both existing solved cases recording ``state_time = 1.0584``). With
``extend=CONSTANT`` (set explicitly here, never left to FEBio's default) the
final partial step is a settling/relaxation step past the last load point,
not extra travel. This is accepted deliberately, matching the reference,
rather than forcing an exact landing at 1.0 (which `derive_load_curve`
still exposes via `exact_step_size` for a caller that wants it instead).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class LoadCurveSpec:
    target_standoff_mm: float
    initial_contact_mm: float
    indentation_mm: float
    total_travel_mm: float
    step_size: float
    time_steps: int
    extend: str = "CONSTANT"

    @property
    def exact_step_size(self) -> float:
        """``1 / time_steps`` -- lands the solve exactly on load factor 1.0,
        with no overshoot, at the cost of not matching the reference case's
        own (overshooting) step size.
        """
        return 1.0 / self.time_steps

    @property
    def overshoot_fraction(self) -> float:
        """How far past load factor 1.0 the solve runs with ``step_size``
        (not ``exact_step_size``). Reference case: 0.0584 (5.84%).
        """
        return self.time_steps * self.step_size - 1.0

    @property
    def points(self) -> List[Tuple[float, float]]:
        """``<LoadData>`` points: displacement is negative (into the body,
        matching the reference's own sign convention and the shipped rigid
        body's prescribed-displacement DOF).
        """
        return [(0.0, 0.0), (1.0, -self.total_travel_mm)]


def derive_load_curve(
    target_standoff_mm: float,
    indentation_mm: float = 4.25,
    initial_contact_mm: float = 0.5,
) -> LoadCurveSpec:
    """Derive total travel, step size, and time steps from a seated PPE's
    standoff and the desired post-contact indentation.

    ``target_standoff_mm`` (the plate's seating gap -- typically very
    small, e.g. 0.1mm) and ``initial_contact_mm`` (how far the plate should
    move on the first solved step to register as engaged contact -- default
    0.5mm, matching the reference case) are deliberately separate
    parameters. Do not pass the same value for both unless the mesh is fine
    enough to resolve a first step that small (see this module's docstring
    for why conflating them broke convergence on the current whole-torso
    mesh).

    All three inputs must be positive: a non-positive value makes the
    step-size formula (a fraction of total travel) meaningless.
    """
    if target_standoff_mm <= 0:
        raise ValueError(f"target_standoff_mm must be positive, got {target_standoff_mm}")
    if indentation_mm <= 0:
        raise ValueError(f"indentation_mm must be positive, got {indentation_mm}")
    if initial_contact_mm <= 0:
        raise ValueError(f"initial_contact_mm must be positive, got {initial_contact_mm}")

    total_travel_mm = target_standoff_mm + indentation_mm
    step_size = initial_contact_mm / total_travel_mm
    time_steps = math.ceil(1.0 / step_size)

    return LoadCurveSpec(
        target_standoff_mm=target_standoff_mm,
        initial_contact_mm=initial_contact_mm,
        indentation_mm=indentation_mm,
        total_travel_mm=total_travel_mm,
        step_size=step_size,
        time_steps=time_steps,
    )
