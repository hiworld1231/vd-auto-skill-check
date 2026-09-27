from vd.motion import MotionTracker
from vd.vision import Needle


def candidate(angle):
    return (Needle(angle,80,.9,30,2),)


def test_two_frames_produce_early_speed_for_fast_check():
    tracker=MotionTracker()
    assert tracker.update(1.0,candidate(30)) is None
    estimate=tracker.update(1.017,candidate(48))
    assert estimate is not None
    assert 1000<estimate.speed<1100
    assert estimate.samples==2
    assert estimate.speed_scatter>0
