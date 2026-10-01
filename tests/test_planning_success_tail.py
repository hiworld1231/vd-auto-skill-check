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
    assert plan.press_at==1
    assert 45 <= plan.target_phase <= 73
    assert plan.latest_press_at>1


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
    assert p.update(Motion(1,45,280,0,0,1,6),frame_at=1,now=1) is not None

    plan=p.update(Motion(1.05,80,280,0,0,1.05,6),frame_at=1.05,now=1.05)

    assert plan is None
    assert p.reason=='TARGET_PASSED'


def test_normal_wrapped_target_still_waits_for_first_sweep_target():
    p=Planner(lead_seconds=.035,lead_uncertainty=.020)
    p.begin(Arc(170,10),270,Arc(181,42))
    m=Motion(1,280,280,0,0,1,6)

    plan=p.update(m,frame_at=1,now=1)

    assert plan is not None
    assert plan.target_grade=='GREAT'
    assert plan.target_phase==535
    assert plan.press_at>1.7


def test_chained_generation_uses_current_good_when_effect_can_still_land_inside_it():
    p=Planner(lead_seconds=.035,lead_uncertainty=.020)
    p.begin(Arc(20,10),32,Arc(31,42),allow_trailing_good=False)
    m=Motion(1,32,700,0,0,1,6)

    plan=p.update(m,frame_at=1,now=1)

    assert plan is not None
    assert plan.target_grade=='GOOD'
    assert 31<=plan.target_phase<=73
    assert plan.press_at==1
    assert plan.latest_press_at>=1


def test_chained_generation_that_is_already_passed_does_not_rearm():
    p=Planner(lead_seconds=.035,lead_uncertainty=.020)
    p.begin(Arc(20,10),40,Arc(31,42),allow_trailing_good=False)
    m=Motion(1,45,1000,0,0,1,6)

    plan=p.update(m,frame_at=1,now=1)

    assert plan is None
    assert p.reason=='TARGET_PASSED'

    later=p.update(Motion(1.36,405,1000,0,0,1.36,6),frame_at=1.36,now=1.36)
    assert later is None
    assert p.reason=='TARGET_PASSED'
