import math

import cv2
import numpy as np
import pytest

from vd.vision import Arc, Detector
from vd.engine import Engine


def bench_frame(center, needle_angle=210, great_start=40):
    frame = np.full((240, 320, 3), (18, 21, 25), np.uint8)
    template = cv2.imread('assets/space_template.png', cv2.IMREAD_UNCHANGED)
    if template.ndim == 2:
        template = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    elif template.shape[2] == 4:
        alpha = template[:, :, 3:4].astype(np.float32) / 255
        foreground = template[:, :, :3].astype(np.float32)
        background = np.full_like(foreground, (25, 23, 20), dtype=np.float32)
        template = (foreground * alpha + background * (1 - alpha)).astype(np.uint8)

    cx, cy = center
    left, top = round(cx - 35), round(cy - 17.5)
    frame[top:top + 35, left:left + 70] = template[:, :, :3]
    origin = (round(cx), round(cy))
    cv2.circle(frame, origin, 67, (220, 225, 229), 2, cv2.LINE_AA)

    def arc(start, width, color, thickness):
        points = []
        for angle in np.linspace(start, start + width, max(3, int(width * 2))):
            radians = math.radians(float(angle))
            points.append((round(cx + 66.5 * math.cos(radians)),
                           round(cy + 66.5 * math.sin(radians))))
        cv2.polylines(frame, [np.asarray(points, np.int32)], False,
                      color, thickness, cv2.LINE_AA)

    arc(great_start, 10, (255, 255, 255), 10)
    arc(great_start+11, 42, (8, 9, 10), 7)
    tip = (round(cx + 74 * math.cos(math.radians(needle_angle))),
           round(cy + 74 * math.sin(math.radians(needle_angle))))
    cv2.line(frame, origin, tip, (67, 61, 232), 2, cv2.LINE_AA)
    return frame


def test_solver_detector_recognizes_synthetic_bench_at_both_observed_centers():
    detector = Detector()
    for center in ((160, 162.5), (170, 82.5)):
        result = detector.measure(bench_frame(center), 1.0)
        assert result.reason == 'OK'
        assert result.center == center
        assert result.prompt_score >= .9
        assert result.great is not None and 5 <= result.great.width <= 16
        assert result.good is not None and 18 <= result.good.width <= 65
        assert len(result.candidates) == 1
        assert abs((result.candidates[0].angle - 210 + 180) % 360 - 180) < 3


def test_synthetic_bench_frames_drive_detector_motion_and_one_planner_claim():
    detector = Detector()
    engine = Engine(lead_seconds=.06, lead_uncertainty=.015)
    events = []
    for index in range(70):
        at = index / 60
        angle = (270 + 278 * at) % 360
        measured = detector.measure(bench_frame((160, 162.5), angle), at)
        engine.observe(measured, now=at, held=True)
        engine.poll(at, held=True)
        events.extend(engine.take_events())

    kinds = [event['kind'] for event in events]
    assert kinds.count('BEGIN') == 1
    assert kinds.count('PRESS_CLAIM') == 1
    claim = next(event for event in events if event['kind'] == 'PRESS_CLAIM')
    assert claim['plan']['target_grade'] == 'GREAT'


def test_continuous_frenzy_ring_gets_two_great_claims():
    detector=Detector()
    engine=Engine(lead_seconds=.06,lead_uncertainty=.015)
    events=[]
    target=40
    for index in range(100):
        at=index/60
        angle=(270+550*at)%360
        measured=detector.measure(bench_frame((160,162.5),angle,target),at)
        engine.observe(measured,now=at,held=True)
        engine.poll(at,held=True)
        new=engine.take_events()
        events.extend(new)
        if target==40 and any(e['kind']=='PRESS_CLAIM' for e in new):
            target=160
    claims=[e for e in events if e['kind']=='PRESS_CLAIM']
    assert len(claims)==2
    assert all(e['plan']['target_grade']=='GREAT' for e in claims)


@pytest.mark.parametrize('speed', [278, 550, 700, 1000, 1300])
def test_twenty_continuous_frenzy_checks_each_get_one_attempt(speed):
    detector=Detector()
    engine=Engine(lead_seconds=.035,lead_uncertainty=.020)
    target=40
    next_target_at=None
    claims=[]
    outcomes=[]
    for index in range(3600):
        at=index/60
        if next_target_at is not None and at>=next_target_at:
            target=(target+150)%360
            next_target_at=None
        angle=(270+speed*at)%360
        measured=detector.measure(bench_frame((160,162.5),angle,target),at)
        engine.observe(measured,now=at,held=True)
        plan=engine.planner.current
        # The physical runtime wakes at the planner deadline even between frames.
        wake=(plan.press_at if plan is not None and at<=plan.press_at<at+1/60
              else at)
        engine.poll(wake,held=True)
        for event in engine.take_events():
            if event['kind']=='PRESS_CLAIM':
                claims.append(event)
                plan=event['plan']
                impact=(270+speed*(event['at']+.036))%360
                outcomes.append(Arc(plan['target_window_start'],
                                    plan['target_window_width']).contains(impact))
                next_target_at=event['at']+.15
        if len(claims)>=20:
            break
    assert len(claims)==20
    assert len({event['generation'] for event in claims})==20
    assert all(event['plan']['target_grade']=='GREAT' for event in claims)
    assert all(event['plan']['timing_mode']=='PREDICTED' for event in claims)
    # At 1000°/s, some bench target transitions arrive too late for this
    # measured response; every visible check must still receive one attempt.
    if speed<=700:
        assert outcomes==[True]*20
