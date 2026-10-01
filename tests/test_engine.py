from vd.engine import Engine
from vd.vision import Arc, Measurement, Needle


def measurement(at, *, prompt=.95, angle=70.0, great=Arc(96, 10)):
    return Measurement(at, (160.0, 162.5), prompt, great, Arc(107, 42),
                       (Needle(angle, 75, .8, 30, 2),), 'OK')


def test_background_arcs_and_needle_do_not_start_a_check_without_prompt():
    engine = Engine()
    engine.observe(measurement(1.0, prompt=.67), now=1.0)
    engine.observe(measurement(1.02, prompt=.70, angle=75), now=1.02)
    assert not engine.active
    assert not any(event['kind'] == 'BEGIN' for event in engine.take_events())


def test_ambiguous_red_candidates_still_get_a_success_attempt():
    engine=Engine(lead_seconds=.06,lead_uncertainty=.015)

    def ambiguous(at, angle):
        return Measurement(at,(160,162.5),.99,Arc(96,10),Arc(107,42),(
            Needle(angle,80,.9,50,2),Needle(angle+110,78,.9,46,2)), 'OK')

    engine.observe(ambiguous(1.0,70),now=1.0)
    engine.observe(ambiguous(1.02,76),now=1.02)
    engine.observe(ambiguous(1.04,82),now=1.04)
    engine.observe(ambiguous(1.06,88),now=1.06)

    plan=engine.planner.current
    assert engine.active
    assert plan is not None
    assert plan.target_grade in ('GREAT','GOOD')
    assert engine.poll(plan.press_at+.001) is plan


def test_visible_check_without_needle_gets_one_blind_attempt_after_startup_grace():
    engine=Engine(lead_seconds=.06,lead_uncertainty=.015)

    def no_needle(at):
        return Measurement(at,(160,162.5),.99,Arc(96,10),Arc(107,42),(),
                           'NO_LINE_CANDIDATE')

    engine.observe(no_needle(1.0),now=1.0)
    engine.observe(no_needle(1.02),now=1.02)
    for at in (1.08,1.14,1.20,1.26,1.30):
        engine.observe(no_needle(at),now=at)

    assert engine.active
    assert engine.planner.current is None
    engine.observe(no_needle(1.321),now=1.321)
    plan=engine.planner.current
    assert plan is not None
    assert plan.timing_mode=='BLIND_NO_NEEDLE'
    assert plan.target_grade=='GREAT'
    assert engine.poll(1.321) is plan
    assert engine.poll(1.321) is None


def test_small_startup_angle_jitter_does_not_trigger_blind_attempt_before_motion():
    engine=Engine(lead_seconds=.035,lead_uncertainty=.020)
    great=Arc(96,10)

    engine.observe(measurement(1.0,angle=70.0,great=great),now=1.0)
    engine.observe(measurement(1.02,angle=70.7,great=great),now=1.02)
    engine.observe(measurement(1.08,angle=71.5,great=great),now=1.08)
    engine.observe(measurement(1.12,angle=71.5,great=great),now=1.12)
    engine.observe(measurement(1.18,angle=71.5,great=great),now=1.18)
    engine.observe(measurement(1.24,angle=71.5,great=great),now=1.24)

    assert engine.planner.current is None

    engine.observe(measurement(1.25,angle=75.5,great=great),now=1.25)
    assert engine.planner.current is not None
    assert engine.planner.current.timing_mode=='PREDICTED'


def test_visible_check_with_stalled_needle_gets_blind_attempt_after_motion_expires():
    engine=Engine(lead_seconds=.06,lead_uncertainty=.015)

    engine.observe(measurement(1.0,angle=70),now=1.0)
    engine.observe(measurement(1.02,angle=72),now=1.02)
    engine.observe(measurement(1.04,angle=74),now=1.04)
    engine.observe(measurement(1.06,angle=76),now=1.06)
    for at in (1.10,1.14,1.18,1.22):
        engine.observe(measurement(at,angle=76),now=at)
        if at<1.21:
            plan=engine.planner.current
            assert plan is None or plan.timing_mode!='BLIND_NO_MOTION'

    plan=engine.planner.current
    assert plan is not None
    assert plan.timing_mode=='BLIND_NO_MOTION'
    assert engine.poll(1.22) is plan


def test_one_second_capture_gap_restarts_motion_grace_and_reacquires_before_fallback():
    engine=Engine(lead_seconds=.035,lead_uncertainty=.020)
    great=Arc(96,10)

    for at,angle in ((1.0,70),(1.02,72),(1.04,74),(1.06,76)):
        engine.observe(measurement(at,angle=angle,great=great),now=at)

    engine.observe(measurement(2.10,angle=76,great=great),now=2.10)
    assert engine.planner.current is None
    engine.observe(measurement(2.12,angle=82,great=great),now=2.12)

    plan=engine.planner.current
    assert plan is not None
    assert plan.timing_mode=='PREDICTED'
    assert plan.target_grade in ('GREAT','GOOD')


def test_delayed_static_frame_discards_stale_prediction_then_reacquires_motion():
    engine=Engine(lead_seconds=.035,lead_uncertainty=.020)
    great=Arc(106.926,10.7)

    def moving(at,angle):
        return measurement(at,angle=angle,great=great)

    for at,angle in ((1.0,353.729),(1.02,358.262),(1.036,1.888),(1.052,5.515)):
        engine.observe(moving(at,angle),now=at)
    engine.observe(moving(1.102,5.515),now=1.110)

    assert engine.planner.current is None

    for at,angle in ((1.122,10.048),(1.142,14.581)):
        engine.observe(moving(at,angle),now=at)
    plan=engine.planner.current
    assert plan is not None
    assert plan.timing_mode=='PREDICTED'
    assert plan.target_grade=='GREAT'
    assert plan.press_at>1.142


def test_slow_observed_needle_does_not_trigger_start_age_blind_press():
    engine=Engine(lead_seconds=.035,lead_uncertainty=.020)
    great=Arc(96,10)

    for at,angle in ((1.0,70),(1.02,70.48),(1.06,71.44),(1.10,72.4),
                     (1.14,73.36),(1.18,74.32)):
        engine.observe(measurement(at,angle=angle,great=great),now=at)

    assert engine.active
    assert engine.planner.current is None
    engine.observe(measurement(1.22,angle=75.28,great=great),now=1.22)

    assert engine.planner.current is not None
    assert engine.planner.current.timing_mode=='PREDICTED'


def test_confirmed_prompt_starts_check_and_brief_prompt_loss_uses_absence_grace():
    engine = Engine()
    engine.observe(measurement(1.0), now=1.0)
    engine.observe(measurement(1.02, angle=75), now=1.02)
    assert engine.active
    assert sum(event['kind'] == 'BEGIN' for event in engine.take_events()) == 1

    engine.observe(measurement(1.05, prompt=.66, angle=82, great=Arc(120, 10)), now=1.05)
    assert engine.active
    assert engine.target == Arc(96, 10)

    engine.observe(measurement(1.15, prompt=.66, angle=105), now=1.15)
    assert not engine.active
    assert [event['reason'] for event in engine.take_events() if event['kind'] == 'END'] == ['RING_ENDED']


def test_frenzy_target_change_keeps_measured_motion():
    engine=Engine(lead_seconds=.06,lead_uncertainty=.015)
    for i,angle in enumerate((30,36,42,48,54,60)):
        engine.observe(measurement(1+i*.02,angle=angle,great=Arc(96,10)),now=1+i*.02)
    assert engine.motion.estimate is not None
    assert engine.active
    engine.planner.fired=True
    engine.observe(measurement(1.12,angle=66,great=Arc(150,10)),now=1.12)
    assert engine.generation==2
    assert engine.planner.current is not None
    engine.observe(measurement(1.14,angle=72,great=Arc(150,10)),now=1.14)
    assert engine.generation==2
    assert engine.motion.estimate is not None
    assert engine.planner.current is not None


def test_frenzy_speed_jump_cannot_plan_with_the_previous_check_speed():
    engine=Engine(lead_seconds=.035,lead_uncertainty=.020)
    for i,angle in enumerate((30,36,42,48,54,60)):
        engine.observe(measurement(1+i*.02,angle=angle,great=Arc(96,10)),
                       now=1+i*.02)
    assert engine.motion.estimate is not None
    assert abs(engine.motion.estimate.speed-300)<1
    engine.planner.fired=True

    engine.observe(measurement(1.12,angle=86,great=Arc(220,10)),now=1.12)
    assert engine.planner.current is None

    engine.observe(measurement(1.14,angle=112,great=Arc(220,10)),now=1.14)
    plan=engine.planner.current
    assert plan is not None
    assert plan.timing_mode=='PREDICTED'
    assert abs(engine.motion.estimate.speed-1300)<1
    assert plan.press_at<1.25


def test_small_frenzy_arc_change_still_requires_confirmation():
    engine=Engine(lead_seconds=.035,lead_uncertainty=.020)
    for i,angle in enumerate((30,36,42,48,54,60)):
        engine.observe(measurement(1+i*.02,angle=angle,great=Arc(96,10)),now=1+i*.02)
    engine.planner.fired=True

    engine.observe(measurement(1.12,angle=66,great=Arc(101,10)),now=1.12)
    assert engine.generation==1
    assert engine.planner.current is None

    engine.observe(measurement(1.14,angle=72,great=Arc(101,10)),now=1.14)
    assert engine.generation==2
    assert engine.planner.current is not None


def test_frenzy_after_full_turn_aims_at_current_revolution():
    engine=Engine(lead_seconds=.06,lead_uncertainty=.015)
    for i,angle in enumerate((340,350,0,10,20,30)):
        engine.observe(measurement(1+i*.02,angle=angle,great=Arc(96,10)),now=1+i*.02)
    assert engine.motion.estimate.phase>360
    engine.planner.fired=True
    engine.observe(measurement(1.12,angle=40,great=Arc(90,10)),now=1.12)
    engine.observe(measurement(1.14,angle=50,great=Arc(90,10)),now=1.14)
    assert engine.generation==2
    assert engine.planner.target_phase>360
    assert engine.planner.current.press_at>1.14


def test_frenzy_reacquire_keeps_revolution_after_fit_reset():
    engine=Engine(lead_seconds=.035,lead_uncertainty=.020)
    for i,angle in enumerate((340,350,0,10,20,30)):
        engine.observe(measurement(1+i*.02,angle=angle,great=Arc(96,10)),now=1+i*.02)
    assert engine.motion.estimate.phase>360
    engine.planner.fired=True

    engine.motion.reset_fit()
    assert engine.motion.last_unwrapped is None
    assert engine.motion.unwrap_floor>360

    engine.observe(measurement(1.12,angle=40,great=Arc(220,10)),now=1.12)
    engine.observe(measurement(1.14,angle=75,great=Arc(220,10)),now=1.14)

    plan=engine.planner.current
    assert plan is not None
    assert plan.target_phase==585
    assert plan.press_at>1.14
