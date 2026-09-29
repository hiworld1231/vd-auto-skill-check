from vd.calibration import FreezeObserver, Landing, LeadEstimator
from vd.motion import Motion
from vd.vision import Arc, Measurement, Needle


def test_freeze_can_be_confirmed_before_the_ring_disappears():
    observer = FreezeObserver(
        motion=Motion(at=1.0, phase=20, speed=280, residual=.2,
                     speed_scatter=1, last_motion_at=1.0, samples=9),
        press_at=1.0, frame_at=1.0, great=Arc(35, 12),
    )
    for timestamp, angle, target_visible in ((1.02, 25, True), (1.04, 31, True),
                                              (1.06, 37, True), (1.08, 37.2, False),
                                              (1.11, 37.1, False)):
        observer.feed(Measurement(timestamp, (160, 160), .99,
                                  Arc(35, 12) if target_visible else None,
                                  Arc(48, 40) if target_visible else None,
                                  (Needle(angle, 100, 1, 50, 3),), 'OK'))
    observer.feed(Measurement(1.13, None, 0, None, None, (), 'NO_PROMPT'))

    landing = observer.finish('RING_ENDED')

    assert landing.label == 'CV_GREAT'
    assert landing.eligible
    assert abs(landing.angle - 37.1) < .001


def test_needle_loss_before_a_stable_suffix_stays_unconfirmed():
    observer = FreezeObserver(
        motion=Motion(at=1.0, phase=20, speed=280, residual=.2,
                     speed_scatter=1, last_motion_at=1.0, samples=9),
        press_at=1.0, frame_at=1.0, great=Arc(35, 12),
    )
    observer.feed(Measurement(1.02, (160, 160), .99, Arc(35, 12), Arc(48, 40),
                              (Needle(25, 100, 1, 50, 3),), 'OK'))
    observer.feed(Measurement(1.04, (160, 160), .99, Arc(35, 12), Arc(48, 40), (), 'NO_NEEDLE'))

    landing = observer.finish('RING_ENDED')

    assert landing.label == 'UNCONFIRMED'
    assert not landing.eligible


def test_lead_estimator_applies_two_consistent_physical_measurements():
    estimator = LeadEstimator(.060, .015)
    first = Landing('CV_MISS', 104.04, .033633, .020419, True,
                    'CV_FREEZE_ESTIMATE')
    second = Landing('CV_MISS', 34.71, .036445, .021414, True,
                     'CV_FREEZE_ESTIMATE')

    assert not estimator.observe(first, at=1.0, physical=True)
    assert estimator.lead == .060
    assert estimator.observe(second, at=2.0, physical=True)
    assert abs(estimator.lead - .035039) < .000001
    assert abs(estimator.uncertainty - .0209165) < .000001
    assert estimator.reason == 'FRESH_CV_GROUP'


def test_lead_estimator_rejects_two_inconsistent_measurements():
    estimator = LeadEstimator(.060, .015)
    first = Landing('CV_GREAT', 100, .035, .010, True, 'CV_FREEZE_ESTIMATE')
    second = Landing('CV_MISS', 130, .065, .010, True, 'CV_FREEZE_ESTIMATE')

    assert not estimator.observe(first, at=1.0, physical=True)
    assert not estimator.observe(second, at=2.0, physical=True)
    assert estimator.lead == .060
    assert estimator.reason == 'INCONSISTENT_RESPONSE'
