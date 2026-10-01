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


def test_good_aim_does_not_run_away_on_each_fresh_observation():
    p=Planner(lead_seconds=0,lead_uncertainty=.020)
    p.begin(Arc(20,10),40,Arc(31,42))

    first=p.update(Motion(1.00,40,280,0,0,1.00,6),frame_at=1.00,now=1.00)
    second=p.update(Motion(1.02,45.6,280,0,0,1.02,7),frame_at=1.02,now=1.02)
    third=p.update(Motion(1.04,51.2,280,0,0,1.04,8),frame_at=1.04,now=1.04)

    assert first is not None and first.target_grade=='GOOD'
    assert second is not None and second.target_grade=='GOOD'
    assert third is not None and third.target_grade=='GOOD'
    assert second.target_phase==pytest.approx(first.target_phase)
    assert third.target_phase==pytest.approx(first.target_phase)
    assert second.press_at==pytest.approx(first.press_at)
    assert third.press_at==pytest.approx(first.press_at)


def test_good_aim_crossed_by_fresh_frame_presses_now_instead_of_moving_later():
    p=Planner(lead_seconds=0,lead_uncertainty=.020)
    p.begin(Arc(20,10),40,Arc(31,42))

    first=p.update(Motion(1.00,40,280,0,0,1.00,6),frame_at=1.00,now=1.00)
    assert first is not None and first.target_grade=='GOOD'

    crossed=p.update(Motion(1.07,61,280,0,0,1.07,8),frame_at=1.07,now=1.07)

    assert crossed is not None
    assert crossed.target_grade=='GOOD'
    assert crossed.press_at==pytest.approx(1.07)
    assert crossed.target_phase<=61.25
    assert crossed.latest_press_at>crossed.press_at


def test_latched_good_upgrades_to_great_when_motion_becomes_safe_before_great():
    p=Planner(lead_seconds=0,lead_uncertainty=.003)
    p.begin(Arc(20,10),300,Arc(31,42))

    noisy=p.update(Motion(1.00,300,280,0,20,1.00,6),frame_at=1.00,now=1.00)
    fallback_aim=p.good_aim
    improved=p.update(Motion(1.20,356,280,0,2,1.20,12),frame_at=1.20,now=1.20)

    assert noisy is not None
    assert noisy.target_grade=='GOOD'
    assert fallback_aim is not None
    assert improved is not None
    assert improved.target_grade=='GREAT'
    assert improved.target_phase==pytest.approx(p.occurrence.great_center)
    assert p.good_aim==fallback_aim


def test_committed_great_survives_small_uncertainty_excursion():
    p=Planner(lead_seconds=0,lead_uncertainty=.003)
    p.begin(Arc(20,10),300,Arc(31,42))

    fallback=p.update(Motion(1.00,300,280,0,20,1.00,6),frame_at=1.00,now=1.00)
    safe=p.update(Motion(1.20,356,280,0,2,1.20,12),frame_at=1.20,now=1.20)
    slightly_noisy=p.update(Motion(1.22,361.6,280,1.45,2,1.22,13),
                            frame_at=1.22,now=1.22)

    assert fallback is not None and fallback.target_grade=='GOOD'
    assert safe is not None and safe.target_grade=='GREAT'
    assert slightly_noisy is not None
    assert slightly_noisy.uncertainty_degrees>p.target.width/2
    assert slightly_noisy.uncertainty_degrees<p.target.width/2+1
    assert slightly_noisy.target_grade=='GREAT'
    assert slightly_noisy.target_phase==pytest.approx(p.occurrence.great_center)


def test_committed_great_falls_back_when_uncertainty_really_worsens():
    p=Planner(lead_seconds=0,lead_uncertainty=.003)
    p.begin(Arc(20,10),300,Arc(31,42))

    p.update(Motion(1.00,300,280,0,20,1.00,6),frame_at=1.00,now=1.00)
    safe=p.update(Motion(1.20,356,280,0,2,1.20,12),frame_at=1.20,now=1.20)
    degraded=p.update(Motion(1.24,367.2,280,2.0,2,1.24,14),
                      frame_at=1.24,now=1.24)

    assert safe is not None and safe.target_grade=='GREAT'
    assert degraded is not None
    assert degraded.uncertainty_degrees>p.target.width/2+1
    assert degraded.target_grade=='GOOD'


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
