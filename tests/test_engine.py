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
    engine.observe(measurement(1.14,angle=72,great=Arc(150,10)),now=1.14)
    assert engine.generation==2
    assert engine.motion.estimate is not None
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
