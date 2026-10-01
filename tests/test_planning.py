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
    original=p.occurrence
    m=Motion(1,110,300,0,0,1,6)

    assert p.update(m,frame_at=1,now=1) is None
    assert p.reason=='TARGET_PASSED'
    assert p.occurrence==original

    later=Motion(2.2,470,300,0,0,2.2,6)
    assert p.update(later,frame_at=2.2,now=2.2) is None
    assert p.reason=='TARGET_PASSED'
    assert p.occurrence==original


def test_frozen_motion_does_not_get_freshness_from_new_delivery():
    p,m=setup()
    assert p.update(m,frame_at=1.1,now=1.1) is None
    assert p.reason=='STALE_OBSERVATION'


def test_recent_motion_survives_one_delayed_capture_delivery():
    p=Planner(lead_seconds=.035,lead_uncertainty=.020)
    p.begin(Arc(106.926,10.7),353.729)
    m=Motion(.761846,353.729,226.676,.972,32.946,.761846,4)

    plan=p.update(m,frame_at=.811844,now=.819908)

    assert plan is not None
    assert plan.timing_mode=='PREDICTED'
    assert plan.target_grade=='GREAT'
    assert plan.press_at>plan.valid_until


def test_release_blocks_but_late_wake_attempts():
    p,m=setup()
    plan=p.update(m,frame_at=1,now=1)
    assert not p.claim(plan,now=1,held=False,capture_alive=True)
    assert p.claim(plan,now=1.01,held=True,capture_alive=True)


def test_small_scheduler_delay_is_recorded_but_does_not_block_attempt():
    p=Planner(lead_seconds=.1,lead_uncertainty=.002)
    p.begin(Arc(96,8),60)
    m=Motion(1,70,300,1,0,1,6)
    plan=p.update(m,frame_at=1,now=1)
    assert plan is not None
    assert p.claim(plan,now=1.0019,held=True,capture_alive=True)


def test_uncertain_adjacent_great_falls_back_to_wider_good_window():
    p=Planner(lead_seconds=.005,lead_uncertainty=.003)
    p.begin(Arc(120,10),66,Arc(131,42))
    m=Motion(1,70,1000,1.5,200,1,2)

    plan=p.update(m,frame_at=1,now=1)

    assert plan is not None
    assert plan.target_grade=='GOOD'
    assert plan.target_phase==152
    assert plan.target_window_start==131
    assert plan.target_window_width==42
    assert plan.uncertainty_degrees > 5


def test_precise_high_speed_motion_keeps_great_target():
    p=Planner(lead_seconds=.005,lead_uncertainty=.001)
    p.begin(Arc(120,10),66,Arc(131,42))
    m=Motion(1,70,1000,.1,2,1,8)

    plan=p.update(m,frame_at=1,now=1)

    assert plan is not None
    assert plan.target_grade=='GREAT'
    assert plan.target_phase==125
    assert plan.target_window_width==10
    assert plan.uncertainty_degrees < plan.target_window_width/2


def test_great_remains_preferred_when_its_envelope_is_safe():
    p=Planner(lead_seconds=.1,lead_uncertainty=.002)
    p.begin(Arc(96,8),60,Arc(105,30))
    m=Motion(1,70,300,0,0,1,6)
    plan=p.update(m,frame_at=1,now=1)
    assert plan is not None
    assert plan.target_grade=='GREAT'
    assert plan.target_window_width==8


def test_remote_good_arc_cannot_expand_target_permission():
    p=Planner(lead_seconds=.060,lead_uncertainty=.015)
    p.begin(Arc(96,8),60,Arc(180,40))
    m=Motion(1,70,300,2,20,1,6)
    plan=p.update(m,frame_at=1,now=1)
    assert plan.target_grade=='GREAT'
    assert plan.target_window_width<=8


def test_passed_white_window_is_not_silently_rearmed():
    p=Planner(lead_seconds=.060,lead_uncertainty=.015)
    p.begin(Arc(96,8),60)
    m=Motion(1,110,300,0,0,1,6)
    assert p.update(m,frame_at=1,now=1) is None
    assert p.reason=='TARGET_PASSED'
    assert p.occurrence.great_start==96


def test_uncertain_plan_can_claim_after_scheduler_wakes_late():
    p=Planner(lead_seconds=.06,lead_uncertainty=.015)
    p.begin(Arc(96,8),60)
    m=Motion(1,70,300,2,20,1,6)
    plan=p.update(m,frame_at=1,now=1)
    assert plan is not None
    assert p.claim(plan,now=plan.press_at+.004,held=True,capture_alive=True)


def test_passed_target_has_no_future_deadline():
    p=Planner(lead_seconds=.06,lead_uncertainty=.015)
    p.begin(Arc(96,8),60)
    m=Motion(1,110,300,0,0,1,6)
    assert p.update(m,frame_at=1,now=1) is None
    assert p.reason=='TARGET_PASSED'


def test_white_exit_deadline_uses_arc_width_not_model_uncertainty():
    p=Planner(lead_seconds=.06,lead_uncertainty=.015)
    p.begin(Arc(96,10),60)
    m=Motion(1,70,300,2,20,1,6)
    plan=p.update(m,frame_at=1,now=1)
    assert plan.uncertainty_degrees>5
    assert 0.015<plan.latest_press_at-plan.intended_press_at<0.018
