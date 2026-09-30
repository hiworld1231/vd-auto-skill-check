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
    assert plan.target_phase==52
    assert plan.latest_press_at>1


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
