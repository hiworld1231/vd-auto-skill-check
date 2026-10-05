from vd.engine import Engine
from vd.vision import Arc, Measurement, Needle


def measurement(at, angle):
    return Measurement(at, (160.0, 162.5), .99, Arc(96, 10), Arc(107, 42),
                       (Needle(angle, 75, .8, 30, 2),), 'OK')


def test_unpressed_ring_end_reports_state_before_invalidation():
    engine = Engine(lead_seconds=.035, lead_uncertainty=.020)
    engine.observe(measurement(1.0, 70), now=1.0)
    engine.observe(measurement(1.02, 75), now=1.02)
    engine.take_events()

    assert engine.active
    assert not engine.planner.fired

    engine.cancel('RING_ENDED', 1.25)
    events = engine.take_events()

    diagnostic = next(event for event in events
                      if event['kind'] == 'MISSED_END_DIAGNOSTIC')
    assert diagnostic['end_reason'] == 'RING_ENDED'
    assert diagnostic['planner_reason_before_end'] != 'RING_ENDED'
    assert diagnostic['motion_reason_before_end'] == engine.motion.reason or diagnostic['motion_reason_before_end']
    assert 'plan_before_end' in diagnostic
    assert 'motion_before_end' in diagnostic
    assert diagnostic['started_at'] is not None

    end = next(event for event in events if event['kind'] == 'END')
    assert end['reason'] == 'RING_ENDED'
    assert end['pressed'] is False
