from vd.motion import Motion
from vd.planning import Planner
from vd.vision import Arc


def setup():
    p=Planner(lead_seconds=.1,lead_uncertainty=.002)
    p.begin(Arc(96,8),60)
    m=Motion(1,70,300,0,0,1,6)
    return p,m


def test_replaced_plan_and_dead_source_cannot_dispatch():
    p,m=setup()
    old=p.update(m,frame_at=1,now=1)
    current=p.update(m,frame_at=1,now=1)
    assert not p.claim(old,now=1,held=True,capture_alive=True)
    assert not p.claim(current,now=1,held=True,capture_alive=False)
    assert p.claim(current,now=1,held=True,capture_alive=True)
    assert not p.claim(current,now=1,held=True,capture_alive=True)


def test_future_deadline_cannot_outlive_its_observation():
    p,m=setup()
    p.begin(Arc(126,8),60)
    plan=p.update(m,frame_at=1,now=1)
    assert plan.press_at>plan.valid_until
    assert not p.claim(plan,now=plan.press_at,held=True,capture_alive=True)


def test_passed_target_does_not_become_next_rotation():
    p,_=setup()
    m=Motion(1,110,300,0,0,1,6)
    assert p.update(m,frame_at=1,now=1) is None
    assert p.reason=='TARGET_PASSED'


def test_frozen_motion_does_not_get_freshness_from_new_delivery():
    p,m=setup()
    assert p.update(m,frame_at=1.1,now=1.1) is None
    assert p.reason=='STALE_OBSERVATION'


def test_late_wake_and_release_do_not_press():
    p,m=setup()
    plan=p.update(m,frame_at=1,now=1)
    assert not p.claim(plan,now=1,held=False,capture_alive=True)
    assert not p.claim(plan,now=1.01,held=True,capture_alive=True)


def test_small_scheduler_delay_cannot_exceed_remaining_target_margin():
    p=Planner(lead_seconds=.1,lead_uncertainty=.002)
    p.begin(Arc(96,8),60)
    # 3*residual + speed*lead_uncertainty = 3.6°, leaving only 0.4°.
    m=Motion(1,70,300,1,0,1,6)
    plan=p.update(m,frame_at=1,now=1)
    assert plan is not None
    # 1.9ms is below the generic 2ms wake limit, but moves the needle 0.57°.
    assert not p.claim(plan,now=1.0019,held=True,capture_alive=True)


def test_good_fallback_is_explicit_when_great_is_too_narrow():
    p=Planner(lead_seconds=.060,lead_uncertainty=.015)
    p.begin(Arc(120.87349400366959,10.728950934823914),66,
            Arc(132.41010851560065,42.14130537921591))
    m=Motion(12403.950864845,66.51839764396223,251.15077255039893,
             1.7121848764034837,44.415726752027744,12403.950864845,5)
    plan=p.update(m,frame_at=m.at,now=m.at)
    assert plan is not None
    assert plan.target_grade=='GOOD_FALLBACK'
    assert plan.target_window_width>50
    assert plan.uncertainty_degrees<plan.target_window_width/2


def test_great_remains_preferred_when_its_envelope_is_safe():
    p,m=setup()
    p.good=Arc(105,30)
    plan=p.update(m,frame_at=1,now=1)
    assert plan is not None
    assert plan.target_grade=='GREAT'
    assert plan.target_window_width==8


def test_remote_good_arc_cannot_expand_target_permission():
    p=Planner(lead_seconds=.060,lead_uncertainty=.015)
    p.begin(Arc(96,8),60,Arc(180,40))
    m=Motion(1,70,300,2,20,1,6)
    assert p.update(m,frame_at=1,now=1) is None
    assert p.reason=='TARGET_UNCERTAIN'
