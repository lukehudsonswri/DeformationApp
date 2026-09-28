"""Tests for febio.loadcurve -- deriving FEBio time-stepping from a seated
PPE's standoff, first-contact travel, and desired indentation (AGENTS.md
section 2.4).
"""
import math

import pytest

from febio.loadcurve import derive_load_curve


def test_reproduces_the_gold_standard_reference_case_exactly():
    """NewHex_close.feb: total_travel=4.25mm, step_size=0.1176,
    time_steps=9. The reference's own accounting is 0.5mm to first contact
    + 3.75mm of indentation = 4.25mm total -- and, in that specific case,
    the seating standoff and the first-contact travel happen to be the same
    0.5mm value.
    """
    spec = derive_load_curve(target_standoff_mm=0.5, indentation_mm=3.75, initial_contact_mm=0.5)
    assert spec.total_travel_mm == pytest.approx(4.25)
    assert spec.step_size == pytest.approx(0.117647, abs=1e-5)
    assert spec.time_steps == 9


def test_reference_case_overshoot_matches_recorded_state_time():
    """Both existing solved cases recorded state_time=1.0584 at the end of
    the run -- i.e. 9 * 0.1176 overshoots load factor 1.0 by 5.84%.
    """
    spec = derive_load_curve(target_standoff_mm=0.5, indentation_mm=3.75, initial_contact_mm=0.5)
    assert spec.time_steps * spec.step_size == pytest.approx(1.0584, abs=1e-3)
    assert spec.overshoot_fraction == pytest.approx(0.0584, abs=1e-3)


def test_project_default_standoff_and_indentation_with_default_initial_contact():
    """Our project's own defaults: TARGET_STANDOFF_MM=0.1 (tight seating),
    INDENTATION_MM=4.25, and the default INITIAL_CONTACT_MM=0.5 (a real,
    physically meaningful first-step size, decoupled from the tiny seating
    gap -- see module docstring for the bug this fixes).
    """
    spec = derive_load_curve(target_standoff_mm=0.1, indentation_mm=4.25)
    assert spec.total_travel_mm == pytest.approx(4.35)
    assert spec.step_size == pytest.approx(0.5 / 4.35)
    assert spec.time_steps == math.ceil(4.35 / 0.5)
    assert spec.time_steps == 9


def test_first_step_travel_is_close_to_initial_contact_mm():
    """The actual physical distance travelled on the first solved step is
    step_size * total_travel_mm -- this should land close to
    initial_contact_mm (small discrepancy only from the ceil() rounding of
    time_steps), not collapse to the (much smaller) seating standoff. This
    is the exact regression this module's docstring describes: an earlier
    version used target_standoff_mm here, producing a ~0.1mm first step
    that never engaged contact on the real (coarser) whole-torso mesh.
    """
    spec = derive_load_curve(target_standoff_mm=0.1, indentation_mm=4.25, initial_contact_mm=0.5)
    first_step_travel_mm = spec.step_size * spec.total_travel_mm
    assert first_step_travel_mm == pytest.approx(0.5, abs=0.01)


def test_points_have_correct_sign_and_shape():
    spec = derive_load_curve(target_standoff_mm=0.1, indentation_mm=4.25)
    assert spec.points == [(0.0, 0.0), (1.0, pytest.approx(-4.35))]


def test_exact_step_size_lands_on_one_with_no_overshoot():
    spec = derive_load_curve(target_standoff_mm=0.5, indentation_mm=3.75, initial_contact_mm=0.5)
    assert spec.time_steps * spec.exact_step_size == pytest.approx(1.0)


def test_smaller_initial_contact_relative_to_indentation_needs_more_steps():
    """A finer first-contact resolution request (smaller initial_contact_mm)
    should require more, not fewer, time steps for the same total travel.
    """
    loose = derive_load_curve(target_standoff_mm=0.1, indentation_mm=4.25, initial_contact_mm=1.0)
    tight = derive_load_curve(target_standoff_mm=0.1, indentation_mm=4.25, initial_contact_mm=0.2)
    assert tight.time_steps > loose.time_steps


def test_standoff_alone_no_longer_affects_step_size():
    """The bug this module's docstring describes: standoff and first-step
    size must be independent. Changing target_standoff_mm alone (same
    initial_contact_mm, same indentation_mm) should barely move the actual
    first-step travel -- only through total_travel_mm's denominator, not by
    controlling the numerator as the old (buggy) formula did.
    """
    tight_seating = derive_load_curve(target_standoff_mm=0.1, indentation_mm=4.25, initial_contact_mm=0.5)
    loose_seating = derive_load_curve(target_standoff_mm=0.5, indentation_mm=4.25, initial_contact_mm=0.5)
    first_step_tight = tight_seating.step_size * tight_seating.total_travel_mm
    first_step_loose = loose_seating.step_size * loose_seating.total_travel_mm
    assert first_step_tight == pytest.approx(first_step_loose, rel=0.05)


def test_nonpositive_inputs_raise():
    with pytest.raises(ValueError):
        derive_load_curve(target_standoff_mm=0.0, indentation_mm=4.25)
    with pytest.raises(ValueError):
        derive_load_curve(target_standoff_mm=0.1, indentation_mm=0.0)
    with pytest.raises(ValueError):
        derive_load_curve(target_standoff_mm=-0.1, indentation_mm=4.25)
    with pytest.raises(ValueError):
        derive_load_curve(target_standoff_mm=0.1, indentation_mm=4.25, initial_contact_mm=0.0)
