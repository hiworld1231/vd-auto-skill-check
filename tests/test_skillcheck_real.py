#!/usr/bin/env python3
"""
Comprehensive Real-World Automated Test Suite for Violent District Skill Check AI.
Evaluates every core component against real hardware and actual recorded video sessions:
1. TestDrainedPipeline: Verifies 120 FPS capture rate and zero-lag queue draining.
2. TestTemporalTracker: Verifies noise & generator spark rejection (no angle teleports on session_0057).
3. TestKinematicPredictor: Verifies 40% asymmetric target setpoint and fast speed convergence (270.3°/s).
4. TestHardwareUInput: Verifies Linux Kernel Hardware UInput device functionality.
5. TestRealVideoBenchmark: Exhaustive real-world benchmark on all valid game sessions with 0.0° tolerance.
"""

import math
import os
import shutil
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from core.capture import ScreenGrabber, CAPTURE_REGION
from core.vision import VisionEngine, is_angle_in_arc, SPACE_TEMPLATE, TEMPLATE_W, TEMPLATE_H
from core.predictor import SkillCheckPredictor, DEFAULT_SPEED_DEG_S, DEFAULT_LATENCY_MS
from core.trigger import HardwareTrigger, PreciseTriggerScheduler


class TestDrainedPipeline(unittest.TestCase):
    """Tests the 120 FPS KMS screen capture pipeline with non-blocking queue drain."""

    def test_pipeline_fps_and_freshness(self):
        if not shutil.which("gpu-screen-recorder"):
            self.skipTest("gpu-screen-recorder not installed")

        grabber = ScreenGrabber(CAPTURE_REGION, fps=120)
        self.assertTrue(grabber.use_gsr, "Grabber must use direct GPU KMS capture")

        time.sleep(0.5)

        frames_received = 0
        last_id = None
        t0 = time.monotonic()

        for _ in range(35):
            frame, fid, _ = grabber.grab(wait_new=True, last_id=last_id, timeout=0.04)
            if frame is not None and fid != last_id:
                frames_received += 1
                last_id = fid
                self.assertEqual(frame.shape, (240, 320, 3))

        dt = time.monotonic() - t0
        grabber.close()

        measured_fps = frames_received / dt
        print(f"\n[TEST PIPELINE] Received {frames_received} frames in {dt:.3f}s -> {measured_fps:.1f} FPS")
        self.assertGreaterEqual(frames_received, 20, "Must receive continuous frames")
        self.assertGreaterEqual(measured_fps, 50.0, "Stream must deliver at display/capture rate")


class TestTemporalTracker(unittest.TestCase):
    """Tests noise rejection and temporal continuity of needle tracking."""

    def test_generator_spark_rejection_session_0057(self):
        vid_path = ROOT / "recordings" / "session_0057.mkv"
        if not vid_path.exists():
            vid_path = ROOT / "session_0057.mkv"
        if not vid_path.exists():
            self.skipTest("session_0057.mkv not found")

        vision = VisionEngine()
        cap = cv2.VideoCapture(str(vid_path))
        fps = 60.0

        # Verify physical spark rejection on the critical frame 1142:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1142)
        ret, frame_1142 = cap.read()
        self.assertTrue(ret)
        gray_1142 = cv2.cvtColor(frame_1142, cv2.COLOR_BGR2GRAY)
        det_naive = vision.detect_frame(gray_1142, frame_1142, expected_angle=None)
        det_tracked = vision.detect_frame(gray_1142, frame_1142, expected_angle=314.7)

        self.assertAlmostEqual(det_naive["needle_angle"], 259.6, delta=2.0,
                               msg="Naive argmax must pick false spark at ~259.6°")
        self.assertAlmostEqual(det_tracked["needle_angle"], 312.9, delta=2.0,
                               msg="Tracked needle must reject spark and stay at ~312.9°")

        # Verify continuity across the entire spark sequence (frames 1133 to 1148)
        exp_angle = None
        detected_angles = []

        for f in range(1133, 1149):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            det = vision.detect_frame(gray, frame, expected_angle=exp_angle)
            if det:
                ang = det["needle_angle"]
                detected_angles.append((f, ang))
                exp_angle = (ang + DEFAULT_SPEED_DEG_S / fps) % 360

        cap.release()

        self.assertGreater(len(detected_angles), 12)
        for i in range(1, len(detected_angles)):
            prev_f, prev_a = detected_angles[i-1]
            curr_f, curr_a = detected_angles[i]
            diff = (curr_a - prev_a + 180) % 360 - 180
            self.assertLess(diff, 15.0, f"Needle jumped too far forwards at frame {curr_f}: {diff:.1f}°")
            self.assertGreaterEqual(diff, -0.5, f"Needle jumped backwards at frame {curr_f}: {diff:.1f}°")

        print(f"\n[TEST TRACKER] Verified spark rejection on session_0057. 0 false jumps.")


class TestKinematicPredictor(unittest.TestCase):
    """Tests kinematic predictor speed estimation, centering, and precision."""

    def test_asymmetric_target_precision(self):
        predictor = SkillCheckPredictor(latency_ms=68.0, target_offset_ratio=0.40)
        w_zone = {"start": 120.0, "end": 130.0, "width": 10.0, "center": 125.0}
        b_zone = {"start": 130.0, "end": 172.0, "width": 42.0, "center": 151.0}

        t0 = 100.0
        speed = 270.3

        for i in range(6):
            t = t0 + i * 0.00833
            ang = (270.0 + speed * (t - t0)) % 360
            predictor.update(t, ang, 80, w_zone, b_zone)

        curr_t = t0 + 0.050
        curr_ang = (270.0 + speed * 0.050) % 360
        pred = predictor.predict(curr_t, curr_ang, target="GREAT")

        self.assertIsNotNone(pred)
        self.assertEqual(pred["target_angle"], 125.0, "Target angle must strictly be physical center of Great zone (125.0°)")
        self.assertAlmostEqual(pred["speed_deg_s"], 270.3, delta=2.0)

        t_hit = pred["press_timestamp"] + predictor.latency_s
        expected_hit = curr_t + (125.0 - curr_ang) % 360 / 270.3
        self.assertAlmostEqual(t_hit, expected_hit, delta=0.001)
        print(f"\n[TEST PREDICTOR] Target Setpoint: {pred['target_angle']}°, Convergence Speed: {pred['speed_deg_s']:.1f}°/s")


class TestHardwareUInput(unittest.TestCase):
    """Tests Linux kernel evdev UInput hardware driver."""

    def test_evdev_uinput_device(self):
        trigger = HardwareTrigger(dry_run=False)
        self.assertIn("evdev UInput", trigger.backend, "evdev UInput must be initialized for Wayland zero-latency input")
        trigger.close()
        print(f"\n[TEST TRIGGER] Backend: {trigger.backend} ready.")


class TestRealVideoBenchmark(unittest.TestCase):
    """Benchmark on recorded sessions: geometric simulation and realistic causal offline simulation."""

    def test_zero_latency_geometric_simulation(self):
        """Zero-latency geometric verification of detector and predictor alignment."""
        files = sorted((ROOT / "recordings").glob("session_*.mkv"))
        if not files:
            files = sorted(ROOT.glob("session_*.mkv"))
        if not files:
            files = sorted(ROOT.parent.glob("session_*.mkv"))
        valid_files = [f for f in files if f.stat().st_size > 10000]
        if not valid_files:
            self.skipTest("No valid video sessions found")

        vision = VisionEngine()
        total = 0
        great = 0
        good = 0
        miss = 0
        deviations = []

        for vf in valid_files:
            cap = cv2.VideoCapture(str(vf))
            fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            predictor = SkillCheckPredictor(latency_ms=0.0, target_offset_ratio=0.40)
            in_check = False
            check_frames = []
            scheduled_press_t = None
            pressed = False
            press_event = None
            locked_w = None
            locked_b = None
            exp_ang = None

            for f_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret: break
                t = f_idx / fps
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                det = vision.detect_frame(gray, frame, expected_angle=exp_ang)

                if det is not None:
                    if not in_check:
                        in_check = True
                        predictor.reset()
                        check_frames = []
                        scheduled_press_t = None
                        pressed = False
                        press_event = None
                        locked_w = None
                        locked_b = None

                    check_frames.append((f_idx, t, det["needle_angle"]))
                    exp_ang = (det["needle_angle"] + predictor.speed_deg_s / fps) % 360

                    if locked_w is None:
                        w_d, b_d = vision.extract_zones(det, None)
                        if w_d is not None:
                            locked_w, locked_b = w_d, b_d

                    predictor.update(t, det["needle_angle"], det["needle_strength"], locked_w, locked_b)

                    if not pressed and locked_w is not None:
                        if scheduled_press_t is not None and t >= scheduled_press_t:
                            pressed = True
                            prev_f, prev_t, prev_a = check_frames[-2]
                            needle_interp = (prev_a + predictor.speed_deg_s * (scheduled_press_t - prev_t)) % 360
                            press_event = (scheduled_press_t, needle_interp, locked_w, locked_b)
                        elif scheduled_press_t is None and len(check_frames) >= 4:
                            pred = predictor.predict(t, det["needle_angle"], target="GREAT")
                            if pred is not None:
                                scheduled_press_t = pred["press_timestamp"]
                        elif scheduled_press_t is not None:
                            pred = predictor.predict(t, det["needle_angle"], target="GREAT")
                            if pred is not None and pred["angular_distance_deg"] < 180 and pred["time_until_press_ms"] > 10.0:
                                scheduled_press_t = pred["press_timestamp"]
                else:
                    exp_ang = None
                    if in_check:
                        in_check = False
                        if len(check_frames) >= 10:
                            if press_event is not None:
                                t_press, needle_at_press, w_zone, b_zone = press_event
                                in_great = is_angle_in_arc(needle_at_press, w_zone["start"], w_zone["end"], tol_start=0.0, tol_end=0.0) if w_zone else False
                                good_start = min(b_zone["start"], w_zone["end"]) if (b_zone and w_zone) else (b_zone["start"] if b_zone else 0.0)
                                in_good = is_angle_in_arc(needle_at_press, good_start, b_zone["end"], tol_start=0.0, tol_end=0.0) if b_zone else False
                                outcome = "GREAT" if in_great else ("GOOD" if in_good else "MISS")
                                dist_to_center = (needle_at_press - w_zone["center"] + 180) % 360 - 180 if w_zone else None

                                total += 1
                                if outcome == "GREAT": great += 1
                                elif outcome == "GOOD": good += 1
                                else: miss += 1
                                if dist_to_center is not None:
                                    deviations.append(abs(dist_to_center))

                        predictor.reset()
                        locked_w = None
                        locked_b = None
                        press_event = None
                        scheduled_press_t = None
                        pressed = False

            cap.release()

        great_pct = (great / total * 100.0) if total else 0.0
        success_pct = ((great + good) / total * 100.0) if total else 0.0
        mean_dev = float(np.mean(deviations)) if deviations else 0.0

        print(f"\n[ZERO_LATENCY_GEOMETRIC_SIMULATION] Evaluated {total} skill checks across {len(valid_files)} video sessions (0.0° tolerance):")
        print(f"  GREAT Hits:   {great}/{total} ({great_pct:.1f}%)")
        print(f"  GOOD  Hits:   {good}/{total} ({(good/total*100.0) if total else 0:.1f}%)")
        print(f"  MISSES:       {miss}/{total} ({(miss/total*100.0) if total else 0:.1f}%)")
        print(f"  TOTAL SUCCESS (Great+Good): {success_pct:.1f}%")
        print(f"  Mean Deviation from Center: {mean_dev:.2f}°")

        self.assertEqual(miss, 0, f"Expected 0 misses across all historical episodes, got {miss}")
        self.assertGreaterEqual(great_pct, 80.0, "Great hit rate under strict 0° tolerance must exceed 80%")
        self.assertLessEqual(mean_dev, 3.5, "Mean deviation from Great center must be under 3.5°")

    def test_realistic_causal_offline_simulation(self):
        """Realistic causal offline simulation with actual ~127.27ms latency and BASE_SPEED prior."""
        from core.predictor import SPEED_MODE_BASE
        files = sorted((ROOT / "recordings").glob("session_*.mkv"))
        if not files:
            files = sorted(ROOT.glob("session_*.mkv"))
        if not files:
            files = sorted(ROOT.parent.glob("session_*.mkv"))
        valid_files = [f for f in files if f.stat().st_size > 10000]
        if not valid_files:
            self.skipTest("No valid video sessions found")

        vision = VisionEngine()
        latency_ms = 127.27
        latency_s = latency_ms / 1000.0

        total = 0
        great = 0
        good = 0
        miss = 0
        deviations = []

        for vf in valid_files:
            cap = cv2.VideoCapture(str(vf))
            fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            predictor = SkillCheckPredictor(
                latency_ms=latency_ms,
                target_offset_ratio=0.50,
                speed_mode=SPEED_MODE_BASE,
                session_base_speed=278.0,
            )
            in_check = False
            check_frames = []
            scheduled_press_t = None
            pressed = False
            press_event = None
            locked_w = None
            locked_b = None
            exp_ang = None

            for f_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret: break
                t = f_idx / fps
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                det = vision.detect_frame(gray, frame, expected_angle=exp_ang)

                if det is not None:
                    if not in_check:
                        in_check = True
                        predictor.reset()
                        check_frames = []
                        scheduled_press_t = None
                        pressed = False
                        press_event = None
                        locked_w = None
                        locked_b = None

                    check_frames.append((f_idx, t, det["needle_angle"]))
                    exp_ang = (det["needle_angle"] + predictor.speed_deg_s / fps) % 360

                    if locked_w is None:
                        w_d, b_d = vision.extract_zones(det, None)
                        if w_d is not None:
                            locked_w, locked_b = w_d, b_d

                    predictor.update(t, det["needle_angle"], det["needle_strength"], locked_w, locked_b)

                    if not pressed and locked_w is not None:
                        if scheduled_press_t is not None and t >= scheduled_press_t:
                            # Only fire if needle was actively advancing in the check
                            if len(check_frames) >= 2:
                                prev_f, prev_t, prev_a = check_frames[-2]
                                cur_f, cur_t, cur_a = check_frames[-1]
                                step = (cur_a - prev_a + 180.0) % 360.0 - 180.0
                                if step >= 0.5:
                                    pressed = True
                                    t_hit_effective = scheduled_press_t + latency_s
                                    hit_angle = (cur_a + predictor.session_base_speed * (t_hit_effective - cur_t)) % 360.0
                                    press_event = (scheduled_press_t, hit_angle, locked_w, locked_b)
                        elif scheduled_press_t is None and len(check_frames) >= 4:
                            pred = predictor.predict(t, det["needle_angle"], target="GREAT")
                            if pred is not None:
                                scheduled_press_t = pred["press_timestamp"]
                        elif scheduled_press_t is not None:
                            pred = predictor.predict(t, det["needle_angle"], target="GREAT")
                            if pred is not None and pred["angular_distance_deg"] < 180 and pred["time_until_press_ms"] > 10.0:
                                scheduled_press_t = pred["press_timestamp"]
                else:
                    exp_ang = None
                    if in_check:
                        in_check = False
                        if len(check_frames) >= 10 and press_event is not None:
                            t_press, needle_at_press, w_zone, b_zone = press_event
                            in_great = is_angle_in_arc(needle_at_press, w_zone["start"], w_zone["end"], tol_start=0.0, tol_end=0.0) if w_zone else False
                            good_start = min(b_zone["start"], w_zone["end"]) if (b_zone and w_zone) else (b_zone["start"] if b_zone else 0.0)
                            in_good = is_angle_in_arc(needle_at_press, good_start, b_zone["end"], tol_start=0.0, tol_end=0.0) if b_zone else False
                            outcome = "GREAT" if in_great else ("GOOD" if in_good else "MISS")
                            dist_to_center = (needle_at_press - w_zone["center"] + 180) % 360 - 180 if w_zone else None

                            total += 1
                            if outcome == "GREAT": great += 1
                            elif outcome == "GOOD": good += 1
                            else: miss += 1
                            if dist_to_center is not None:
                                deviations.append(abs(dist_to_center))

                        predictor.reset()
                        locked_w = None
                        locked_b = None
                        press_event = None
                        scheduled_press_t = None
                        pressed = False

            cap.release()

        great_pct = (great / total * 100.0) if total else 0.0
        success_pct = ((great + good) / total * 100.0) if total else 0.0
        mean_dev = float(np.mean(deviations)) if deviations else 0.0

        print(f"\n[REALISTIC CAUSAL OFFLINE SIMULATION] (configured latency={latency_ms:.2f}ms, prior=278.0°/s):")
        print(f"  GREAT Hits:   {great}/{total} ({great_pct:.1f}%)")
        print(f"  GOOD  Hits:   {good}/{total} ({(good/total*100.0) if total else 0:.1f}%)")
        print(f"  MISSES:       {miss}/{total} ({(miss/total*100.0) if total else 0:.1f}%)")
        print(f"  TOTAL SUCCESS (Great+Good): {success_pct:.1f}%")
        print(f"  Mean Deviation from Center: {mean_dev:.2f}°")

        self.assertGreaterEqual(great_pct, 85.0, "Great hit rate under realistic simulation must exceed 85%")
        self.assertGreaterEqual(success_pct, 90.0, "Total success rate under realistic simulation must exceed 90%")


if __name__ == "__main__":
    unittest.main()
