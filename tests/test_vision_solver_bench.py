import math

import cv2
import numpy as np

from vd.vision import Detector
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
