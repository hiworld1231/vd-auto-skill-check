import math

import cv2
import numpy as np
import pytest

from vd.vision import Arc, Detector, retained_target
from vd.engine import Engine


def bench_frame(center, needle_angle=210, great_start=40, scale=1.0,
                size=(320, 240)):
    frame = np.full((size[1], size[0], 3), (18, 21, 25), np.uint8)
    template = cv2.imread('assets/space_template.png', cv2.IMREAD_UNCHANGED)
    if template.ndim == 2:
        template = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    elif template.shape[2] == 4:
        alpha = template[:, :, 3:4].astype(np.float32) / 255
        foreground = template[:, :, :3].astype(np.float32)
        background = np.full_like(foreground, (25, 23, 20), dtype=np.float32)
        template = (foreground * alpha + background * (1 - alpha)).astype(np.uint8)
    if scale != 1:
        template = cv2.resize(template, (round(70*scale), round(35*scale)),
                              interpolation=cv2.INTER_LINEAR)

    cx, cy = center
    th, tw = template.shape[:2]
    left, top = round(cx - tw/2), round(cy - th/2)
    frame[top:top + th, left:left + tw] = template[:, :, :3]
    origin = (round(cx), round(cy))
    cv2.circle(frame, origin, round(67*scale), (220, 225, 229),
               max(1, round(2*scale)), cv2.LINE_AA)

    def arc(start, width, color, thickness):
        outer = []
        inner = []
        for angle in np.linspace(start, start + width, max(3, int(width * 2))):
            radians = math.radians(float(angle))
            cosine, sine = math.cos(radians), math.sin(radians)
            radius = 66.5*scale
            outer.append((round(cx + (radius + thickness*scale/2) * cosine),
                          round(cy + (radius + thickness*scale/2) * sine)))
            inner.append((round(cx + (radius - thickness*scale/2) * cosine),
                          round(cy + (radius - thickness*scale/2) * sine)))
        sector = np.asarray(outer + inner[::-1], np.int32)
        cv2.fillPoly(frame, [sector], color, lineType=cv2.LINE_AA)

    arc(great_start, 10, (255, 255, 255), 10)
    arc(great_start+11, 42, (8, 9, 10), 7)
    tip = (round(cx + 74*scale * math.cos(math.radians(needle_angle))),
           round(cy + 74*scale * math.sin(math.radians(needle_angle))))
    cv2.line(frame, origin, tip, (67, 61, 232), max(1, round(2*scale)), cv2.LINE_AA)
    return frame


def test_solver_detector_recognizes_synthetic_bench_at_both_observed_centers():
    detector = Detector()
    for center in ((160, 162.5), (170, 82.5)):
        result = detector.measure(bench_frame(center), 1.0)
        assert result.reason == 'OK'
        assert result.center == center
        assert result.prompt_score >= .9
        assert result.great is not None and 5 <= result.great.width <= 16
        assert abs((result.great.start - 40 + 180) % 360 - 180) < 2
        assert result.good is not None and 18 <= result.good.width <= 65
        assert len(result.candidates) == 1
        assert abs((result.candidates[0].angle - 210 + 180) % 360 - 180) < 3


def test_compact_region_preserves_detection_at_both_observed_centers():
    detector = Detector(roi_offset=(80, 0))
    for center in ((160, 162.5), (170, 82.5)):
        compact = bench_frame(center)[:, 80:260]
        result = detector.measure(compact, 1.0)

        assert result.reason == 'OK'
        assert result.center == (center[0] - 80, center[1])
        assert result.great is not None
        assert abs((result.great.start - 40 + 180) % 360 - 180) < 2
        assert len(result.candidates) == 1
        assert abs((result.candidates[0].angle - 210 + 180) % 360 - 180) < 3


def test_compact_region_finds_prompt_shifted_from_1080p_positions():
    detector = Detector(roi_offset=(80, 0))
    compact = bench_frame((160, 110))[:, 80:260]

    result = detector.measure(compact, 1.0)

    assert result.reason == 'OK'
    assert result.center == pytest.approx((80, 110), abs=.5)
    assert result.great is not None


def test_compact_region_does_not_find_a_prompt_in_a_blank_frame():
    blank = np.full((240, 180, 3), (18, 21, 25), np.uint8)

    result = Detector(roi_offset=(80, 0)).measure(blank, 1.0)

    assert result.reason == 'NO_PROMPT'


def test_blank_frame_full_search_runs_on_half_size_image(monkeypatch):
    calls = []
    match_template = cv2.matchTemplate

    def track_search(image, template, method):
        calls.append(image.shape[:2])
        return match_template(image, template, method)

    monkeypatch.setattr(cv2, 'matchTemplate', track_search)
    blank = np.full((240, 180, 3), (18, 21, 25), np.uint8)
    Detector(roi_offset=(80, 0)).measure(blank, 1.0)

    assert (120, 90) in calls
    assert (240, 180) not in calls


def test_region_detector_works_at_both_positions_in_current_roi():
    detector = Detector(roi_offset=(75, -2))
    for source_center, expected_center in (
            ((160, 82.5), (85, 84.5)), ((170, 162.5), (95, 164.5))):
        frame = bench_frame(source_center, size=(320, 260))
        padded = cv2.copyMakeBorder(frame, 2, 0, 0, 0, cv2.BORDER_CONSTANT)
        compact = padded[:250, 75:255]

        result = detector.measure(compact, 1.0)

        assert result.reason == 'OK'
        assert result.center == pytest.approx(expected_center, abs=.5)
        assert result.great is not None
        assert len(result.candidates) == 1


def test_region_detector_scales_prompt_and_ring_to_1600x900_capture_geometry():
    detector = Detector(roi_offset=(75, -2), ui_scale=1.2)
    for source_center, expected_center in (
            ((160, 82.5), (85, 84.5)), ((170, 162.5), (95, 164.5))):
        frame = bench_frame(source_center, scale=1.2, size=(320, 260))
        padded = cv2.copyMakeBorder(frame, 2, 0, 0, 0, cv2.BORDER_CONSTANT)
        compact = padded[:250, 75:255]

        result = detector.measure(compact, 1.0)

        assert result.reason == 'OK'
        assert result.center == pytest.approx(expected_center, abs=.5)
        assert result.great is not None
        assert abs((result.great.start - 40 + 180) % 360 - 180) < 2
        assert len(result.candidates) == 1
        assert abs((result.candidates[0].angle - 210 + 180) % 360 - 180) < 3


@pytest.mark.parametrize('scale', (2 / 3, .5))
def test_solver_detector_recognizes_white_arc_after_window_scale_normalization(scale):
    frame = bench_frame((160, 162.5))
    reduced = cv2.resize(frame, (round(320 * scale), round(240 * scale)),
                         interpolation=cv2.INTER_AREA)
    normalized = cv2.resize(reduced, (320, 240), interpolation=cv2.INTER_LINEAR)

    result = Detector().measure(normalized, 1.0)

    assert result.reason == 'OK'
    assert result.great is not None
    assert abs((result.great.start - 40 + 180) % 360 - 180) < 2
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
        measured=detector.measure(bench_frame((160,162.5),angle,target),at,
                                  center_hint=engine.center)
        measured=retained_target(measured,great=engine.target,good=engine.good,
                                 center=engine.center)
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
@pytest.mark.parametrize('fps', [60, 30])
def test_twenty_continuous_frenzy_checks_each_get_one_attempt(speed, fps):
    detector=Detector()
    engine=Engine(lead_seconds=.035,lead_uncertainty=.020)
    target=40
    next_target_at=None
    claims=[]
    outcomes=[]
    missed=False
    for index in range(fps*60):
        at=index/fps
        if next_target_at is not None and at>=next_target_at:
            target=(target+150)%360
            next_target_at=None
        angle=(270+speed*at)%360
        measured=detector.measure(bench_frame((160,162.5),angle,target),at,
                                  center_hint=engine.center)
        measured=retained_target(measured,great=engine.target,good=engine.good,
                                 center=engine.center)
        engine.observe(measured,now=at,held=True)
        plan=engine.planner.current
        # The physical runtime wakes at the planner deadline even between frames.
        wake=(plan.press_at if plan is not None and at<=plan.press_at<at+1/fps
              else at)
        engine.poll(wake,held=True)
        for event in engine.take_events():
            if event['kind']=='PRESS_CLAIM':
                claims.append(event)
                impact=(270+speed*(event['at']+.035))%360
                great=Arc(target,10).contains(impact)
                good=Arc(target+11,42).contains(impact)
                outcomes.append(great)
                if great or good:
                    # The bench shows the next check 150 ms after the Space
                    # response has been applied.
                    next_target_at=event['at']+.035+.15
                else:
                    missed=True
        if missed or len(claims)>=20:
            break
    assert len(claims)==20
    assert len({event['generation'] for event in claims})==20
    assert all(event['plan']['target_grade']=='GREAT' for event in claims)
    assert all(event['plan']['timing_mode']=='PREDICTED' for event in claims)
    assert outcomes==[True]*20
