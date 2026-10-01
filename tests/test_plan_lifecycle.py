from vd.engine import Engine
from vd.motion import Motion
from vd.vision import Arc, Measurement


class _NoMotionTracker:
    max_gap = .080

    def __init__(self):
        self.last_frame = 1.0
        self.last_unwrapped = 80.0
        self.unwrap_floor = 80.0
        self.estimate = None
        self.reason = 'NO_NEEDLE'

    def update(self, timestamp, candidates):
        self.last_frame = timestamp
        self.estimate = None
        self.reason = 'NO_NEEDLE'
        return None


class _GreatTailMotionTracker(_NoMotionTracker):
    def update(self, timestamp, candidates):
        self.last_frame = timestamp
        self.estimate = Motion(timestamp, 102.0, 400.0, 0.0, 0.0, timestamp, 9)
        self.reason = 'MEASURED'
        return self.estimate


def _armed_engine_with_pending_great():
    engine = Engine(lead_seconds=0, lead_uncertainty=.003)
    great = Arc(96, 10)
    good = Arc(107, 42)
    center = (160.0, 162.5)

    engine.active = True
    engine.center = center
    engine.target = great
    engine.good = good
    engine.started_at = 1.0
    engine.last_visible = 1.0
    engine.last_timestamp = 1.0
    engine.last_reliable_motion_at = 1.0
    engine.chain = 1

    engine.planner.begin(great, 70.0, good)
    plan = engine.planner.update(
        Motion(1.0, 80.0, 400.0, 0.0, 0.0, 1.0, 8),
        frame_at=1.0,
        now=1.0,
    )
    assert plan is not None
    assert plan.target_grade == 'GREAT'
    assert 1.0 < plan.press_at < plan.valid_until

    engine.motion = _NoMotionTracker()
    return engine, plan, great, good, center


def test_fresh_observation_without_new_motion_preserves_pending_great_until_dispatch():
    engine, plan, great, good, center = _armed_engine_with_pending_great()

    engine.observe(
        Measurement(1.02, center, .99, great, good, (), 'NO_LINE_CANDIDATE'),
        now=1.02,
    )

    assert engine.planner.current is plan
    assert engine.poll(plan.press_at + .0001, held=True, capture_alive=True) is plan


def test_committed_great_survives_fresh_frame_after_center_while_window_is_reachable():
    engine, plan, great, good, center = _armed_engine_with_pending_great()
    engine.motion = _GreatTailMotionTracker()

    engine.observe(
        Measurement(1.04, center, .99, great, good, (), 'OK'),
        now=1.04,
    )

    assert engine.planner.current is plan
    assert engine.planner.current.target_grade == 'GREAT'
    assert engine.poll(plan.press_at + .0001, held=True, capture_alive=True) is plan


def test_capture_loss_still_invalidates_preserved_pending_plan():
    engine, plan, *_ = _armed_engine_with_pending_great()

    assert engine.poll(1.02, held=True, capture_alive=False) is None
    assert engine.planner.current is None
    assert engine.reason == 'CAPTURE_LOST'
    assert not engine.planner.fired
