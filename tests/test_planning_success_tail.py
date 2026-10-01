import pytest

from vd.motion import Motion
from vd.planning import Planner
from vd.vision import Arc


def test_visible_trailing_good_uses_current_occurrence_not_next_rotation():
    p=Planner(lead_seconds=.035,lead_uncertainty=.020)
    p.begin(Arc(20,10),40,Arc(31,42))
    m=Motion(1,45,280,0,0,1,6)

    plan=p.update(m,frame_at=1,now=1)

    assert plan is not None
    assert plan.target_grade=='GOOD'
    assert p.occurrence.great_start==20
    assert plan.target_window_start==pytest.approx(54.8)
    assert plan.target_window_width==pytest.approx(18.2)
    assert plan.target_phase==pytest.approx(63.9)
    assert 1<plan.press_at<plan.latest_press_at


def test_remaining_good_aims_at_center_of_reachable_tail():
    p=Planner(lead_seconds=0,lead_uncertainty=.020)
    p.begin(Arc(20,10),40,Arc(31,42))
    m=Motion(1,60,280,0,0,1,6)

    plan=p.update(m,frame_at=1,now=1)

    assert plan is not None
    assert plan.target_grade=='GOOD'
    assert plan.target_window_start==60
    assert plan.target_window_width==13
    assert plan.target_phase==66.5


def test_latched_success_occurrence_never_rearms_next_rotation():
    p=Planner(lead_seconds=.035,lead_uncertainty=.020)
    p.begin(Arc(20,10),40,Arc(31,42))
    original=p.occurrence
    assert p.update(Motion(1,45,280,0,0,1,6),frame_at=1,now=1) is not None

    plan=p.update(Motion(1.05,80,280,0,0,1.05,6),frame_at=1.05,now=1.05)

    assert plan is None
    assert p.reason=='TARGET_PASSED'
    assert p.occurrence==original


def test_normal_wrapped_target_still_waits_for_first_sweep_target():
    p=Planner(lead_seconds=.035,lead_uncertainty=.020)
    p.begin(Arc(170,10),270,Arc(181,42))
    m=Motion(1,280,280,0,0,1,6)

    plan=p.update(m,frame_at=1,now=1)

    assert plan is not None
    assert p.occurrence.great_start==530
    assert p.occurrence.great_center==535
    assert plan.target_grade in ('GREAT','GOOD')
    assert plan.press_at>1.7


def test_chained_generation_uses_current_good_when_effect_can_still_land_inside_it():
    p=Planner(lead_seconds=.035,lead_uncertainty=.020)
    p.begin(Arc(20,10),32,Arc(31,42),occurrence_start=20)
    m=Motion(1,32,700,0,0,1,6)

    plan=p.update(m,frame_at=1,now=1)

    assert plan is not None
    assert plan.target_grade=='GOOD'
    assert p.occurrence.great_start==20
    assert plan.target_window_start==pytest.approx(56.5)
    assert plan.target_window_width==pytest.approx(16.5)
    assert plan.target_phase==pytest.approx(64.75)
    assert plan.press_at>1
    assert plan.latest_press_at>=plan.press_at


def test_chained_generation_that_is_already_passed_does_not_rearm():
    p=Planner(lead_seconds=.035,lead_uncertainty=.020)
    p.begin(Arc(20,10),40,Arc(31,42),occurrence_start=20)
    original=p.occurrence
    m=Motion(1,45,1000,0,0,1,6)

    plan=p.update(m,frame_at=1,now=1)

    assert plan is None
    assert p.reason=='TARGET_PASSED'
    assert p.occurrence==original

    later=p.update(Motion(1.36,405,1000,0,0,1.36,6),frame_at=1.36,now=1.36)
    assert later is None
    assert p.reason=='TARGET_PASSED'
    assert p.occurrence==original
