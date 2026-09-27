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


def test_two_frames_of_small_pixel_jitter_do_not_create_a_speed_estimate():
    tracker=MotionTracker()
    assert tracker.update(1.0,candidate(131.36)) is None
    assert tracker.update(1.033,candidate(133.20)) is None

    estimate=tracker.update(1.067,candidate(139.0))

    assert estimate is not None
    assert estimate.speed>100


def test_fitted_motion_requires_more_than_four_degrees_of_total_travel():
    tracker=MotionTracker()
    estimates=[tracker.update(1+i*.02,candidate(angle)) for i,angle in enumerate(
        (131.0,132.0,133.0,133.5))]

    assert estimates[-1] is None
    assert tracker.reason=='MEASURING_MOTION'
