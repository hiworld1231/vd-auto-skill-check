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


def test_ambiguous_red_candidates_still_get_a_great_attempt():
    engine=Engine(lead_seconds=.06,lead_uncertainty=.015)

    def ambiguous(at, angle):
        return Measurement(at,(160,162.5),.99,Arc(96,10),Arc(107,42),(
            Needle(angle,80,.9,50,2),Needle(angle+110,78,.9,46,2)), 'OK')

    engine.observe(ambiguous(1.0,70),now=1.0)
    engine.observe(ambiguous(1.02,76),now=1.02)
    engine.observe(ambiguous(1.04,82),now=1.04)
    engine.observe(ambiguous(1.06,94),now=1.06)

    plan=engine.planner.current
    assert engine.active
    assert plan is not None
    assert plan.target_grade=='GREAT'
    assert engine.poll(plan.press_at+.001) is plan


def test_visible_check_without_needle_gets_one_blind_attempt():
    engine=Engine(lead_seconds=.06,lead_uncertainty=.015)

    def no_needle(at):
        return Measurement(at,(160,162.5),.99,Arc(96,10),Arc(107,42),(),
                           'NO_LINE_CANDIDATE')

    engine.observe(no_needle(1.0),now=1.0)
    engine.observe(no_needle(1.02),now=1.02)

    plan=engine.planner.current
    assert engine.active
    assert plan is not None
    assert plan.timing_mode=='BLIND_NO_NEEDLE'
    assert plan.target_grade=='GREAT'
    assert engine.poll(1.02) is plan
    assert engine.poll(1.02) is None


def test_visible_check_with_stalled_needle_gets_blind_attempt_after_motion_expires():
    engine=Engine(lead_seconds=.06,lead_uncertainty=.015)

    engine.observe(measurement(1.0,angle=70),now=1.0)
    engine.observe(measurement(1.02,angle=72),now=1.02)
    engine.observe(measurement(1.08,angle=72.1),now=1.08)

    plan=engine.planner.current
    assert plan is not None
    assert plan.timing_mode=='BLIND_NO_MOTION'
    assert engine.poll(1.08) is plan


def test_confirmed_prompt_starts_check_and_brief_prompt_loss_uses_absence_grace():
    engine = Engine()
    engine.observe(measurement(1.0), now=1.0)
    engine.observe(measurement(1.02, angle=75), now=1.02)
    assert engine.active
    assert sum(event['kind'] == 'BEGIN' for event in engine.take_events()) == 1

    # A background-shaped target with low prompt confidence must not refresh
    # the active generation or replace its locked target.
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
