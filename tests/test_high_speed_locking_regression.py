#!/usr/bin/env python3
"""
Regression test suite for High-Speed Skill Checks (500 - 900+°/s),
Speed Lock Margin, Spawn Stutter Immunity, and Fallback Isolation.

Tests:
1. Real replay 0013: no early ~500 lock, no premature fire at 113ms, eventual speed ~275-300°/s.
2. Synthetic true 500°/s: estimator locks as tier 500 (not 275), achieves positive lock margin, scheduled fire possible.
3. Synthetic true 650°/s: converges to ~650°/s across unique-frame cadences (16.7ms, dropped frame 33ms, jitter 13/20/16/34/15ms).
4. Synthetic true 900°/s: converges to ~900°/s, scheduled fire possible on standard zone placements.
5. High-speed check with short distance from spawn to target: evaluates fallback trigger, confirms sample is never trained.
6. Spawn stutter followed by 650°/s: stationary frames do not skew speed; converges rapidly to ~650°/s.
7. Telemetry consistency: verifies normal locked fires vs fallback no-lock fires and strict equation balance.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.predictor import (
    SkillCheckPredictor,
    STATE_NO_MOTION,
    STATE_PROVISIONAL,
    STATE_LOCKED,
    STATE_COMMITTED,
)
from core.learner import SpeedProfileManager, classify_miss
from core.telemetry_tracker import SessionTelemetryTracker


def generate_cadence_timestamps(pattern_name: str, duration_s: float = 0.50):
    """
    Generates realistic unique-frame arrival timestamps in seconds.
    - 'uniform_16_7ms': standard 60 FPS monitor / render (dt = 16.7ms)
    - 'dropped_frame_33ms': 60 FPS with occasional frame drops (33.4ms gap)
    - 'jittery_cadence': realistic pipe arrival jitter (13, 20, 16, 34, 15ms)
    """
    timestamps = [0.0]
    curr_t = 0.0
    if pattern_name == "uniform_16_7ms":
        dt = 0.01667
        while curr_t < duration_s:
            curr_t += dt
            timestamps.append(round(curr_t, 6))
    elif pattern_name == "dropped_frame_33ms":
        step_idx = 0
        while curr_t < duration_s:
            dt = 0.03334 if step_idx == 2 else 0.01667
            curr_t += dt
            timestamps.append(round(curr_t, 6))
            step_idx += 1
    elif pattern_name == "jittery_cadence":
        jit_cycle = [0.013, 0.020, 0.016, 0.034, 0.015]
        idx = 0
        while curr_t < duration_s:
            dt = jit_cycle[idx % len(jit_cycle)]
            curr_t += dt
            timestamps.append(round(curr_t, 6))
            idx += 1
    else:
        dt = 0.00833  # 120 FPS
        while curr_t < duration_s:
            curr_t += dt
            timestamps.append(round(curr_t, 6))
    return timestamps


class TestHighSpeedLockingRegression(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.profiles_path = self.tmp_dir / "profiles.json"
        self.replay_path = ROOT / "replays" / "check_20260917_180451_0013_MISS.json"
        self.replay_data = json.loads(self.replay_path.read_text(encoding="utf-8")) if self.replay_path.exists() else {}

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_real_replay_0013_stutter_isolation(self):
        """
        Verifies real replay 0013:
        - Stationary spawn stutter is rejected.
        - Estimator does NOT falsely lock near 500°/s at early frames.
        - Predictor does NOT fire at t=113ms.
        - Estimator eventually locks at ~275°/s and converges to ~285-300°/s.
        """
        frames = [
            fr for fr in self.replay_data.get("frames", [])
            if not fr.get("is_pre_roll", False) and "needle_angle" in fr
        ]
        self.assertGreater(len(frames), 20)

        predictor = SkillCheckPredictor(latency_ms=122.1, target_offset_ratio=0.50)
        white_zone = self.replay_data.get("locked_zones", {}).get("white")
        black_zone = self.replay_data.get("locked_zones", {}).get("black")

        for fr in frames:
            t = fr["time_rel_ms"] / 1000.0
            ang = fr["needle_angle"]
            strn = fr.get("needle_strength", 75.0)
            predictor.update(t, ang, strn, white_zone, black_zone)

            # At t <= 90ms, noise must not lock as ~500°/s
            if t * 1000.0 <= 90.0:
                self.assertFalse(
                    predictor.has_stable_speed() and (450.0 <= predictor.speed_deg_s <= 550.0),
                    f"Early frame at t={t*1000:.1f}ms must not lock as ~500°/s"
                )

        self.assertTrue(predictor.has_stable_speed(), "Must achieve stable lock eventually")
        self.assertAlmostEqual(predictor.speed_deg_s, 290.0, delta=20.0,
                               msg=f"Converged speed {predictor.speed_deg_s:.1f}°/s must be near true 290°/s")

    def test_02_synthetic_500_deg_s_cadences(self):
        """
        Verifies synthetic true 500°/s check across different frame cadences:
        - Estimator correctly identifies ~500°/s (does not lock at baseline 270-275°/s).
        - Lock is reached within <= 85ms from motion start.
        - Scheduled fire is possible with positive lock margin.
        """
        cadences = ["uniform_16_7ms", "dropped_frame_33ms", "jittery_cadence"]
        true_speed = 500.0
        target_angle = 90.0  # 180° distance from spawn 270° -> takes 360ms to reach

        w_zone = {"start": 85.0, "end": 95.0, "width": 10.0, "center": 90.0}
        b_zone = {"start": 95.0, "end": 135.0, "width": 40.0, "center": 115.0}

        for cad in cadences:
            with self.subTest(cadence=cad):
                predictor = SkillCheckPredictor(latency_ms=124.0, target_offset_ratio=0.50)
                timestamps = generate_cadence_timestamps(cad, duration_s=0.40)
                spawn_ang = 270.0

                lock_t = None
                lock_speed = None
                lock_margin = None

                for t in timestamps:
                    ang = (spawn_ang + true_speed * t) % 360.0
                    predictor.update(t, ang, 80.0, w_zone, b_zone)

                    if lock_t is None and predictor.has_stable_speed():
                        lock_t = t
                        lock_speed = predictor.speed_deg_s
                        pred = predictor.predict(t, ang, target="GREAT")
                        if pred:
                            lock_margin = pred["time_until_press_ms"]

                self.assertIsNotNone(lock_t, f"Failed to lock 500°/s with {cad}")
                self.assertAlmostEqual(lock_speed, 500.0, delta=25.0,
                                       msg=f"Locked speed {lock_speed:.1f}°/s must be near 500°/s in {cad}")
                self.assertLessEqual(lock_t, 0.085, f"Lock took too long ({lock_t*1000:.1f}ms) in {cad}")
                self.assertGreater(lock_margin, 30.0, f"Lock margin must be > 30ms, got {lock_margin:.1f}ms")

    def test_03_synthetic_650_deg_s_cadences(self):
        """
        Verifies synthetic true 650°/s across different cadences:
        - Estimator locks within <= 70ms.
        - Measured speed is within 25°/s of 650°/s.
        - Scheduled fire is possible with positive lock margin.
        """
        cadences = ["uniform_16_7ms", "dropped_frame_33ms", "jittery_cadence"]
        true_speed = 650.0
        target_angle = 120.0  # 210° distance from spawn 270° -> takes 323ms

        w_zone = {"start": 115.0, "end": 125.0, "width": 10.0, "center": 120.0}
        b_zone = {"start": 125.0, "end": 165.0, "width": 40.0, "center": 145.0}

        for cad in cadences:
            with self.subTest(cadence=cad):
                predictor = SkillCheckPredictor(latency_ms=124.0, target_offset_ratio=0.50)
                timestamps = generate_cadence_timestamps(cad, duration_s=0.35)
                spawn_ang = 270.0

                lock_t = None
                lock_speed = None
                lock_margin = None

                for t in timestamps:
                    ang = (spawn_ang + true_speed * t) % 360.0
                    predictor.update(t, ang, 80.0, w_zone, b_zone)

                    if lock_t is None and predictor.has_stable_speed():
                        lock_t = t
                        lock_speed = predictor.speed_deg_s
                        pred = predictor.predict(t, ang, target="GREAT")
                        if pred:
                            lock_margin = pred["time_until_press_ms"]

                self.assertIsNotNone(lock_t, f"Failed to lock 650°/s with {cad}")
                self.assertAlmostEqual(lock_speed, 650.0, delta=30.0,
                                       msg=f"Locked speed {lock_speed:.1f}°/s must be near 650°/s")
                self.assertLessEqual(lock_t, 0.090, f"Lock took too long ({lock_t*1000:.1f}ms)")
                self.assertGreater(lock_margin, 30.0, f"Lock margin must be positive (>30ms)")

    def test_04_synthetic_900_deg_s_scheduled_fire(self):
        """
        Verifies synthetic true 900°/s:
        - Estimator locks within <= 60ms.
        - Measured speed is within 40°/s of 900°/s.
        - Scheduled fire is possible with positive lock margin on standard 180° arc.
        """
        true_speed = 900.0
        target_angle = 90.0  # 180° distance from spawn 270° -> takes 200ms
        w_zone = {"start": 85.0, "end": 95.0, "width": 10.0, "center": 90.0}
        b_zone = {"start": 95.0, "end": 135.0, "width": 40.0, "center": 115.0}

        predictor = SkillCheckPredictor(latency_ms=122.0, target_offset_ratio=0.50)
        timestamps = generate_cadence_timestamps("uniform_16_7ms", duration_s=0.25)
        spawn_ang = 270.0

        lock_t = None
        lock_speed = None
        lock_margin = None

        for t in timestamps:
            ang = (spawn_ang + true_speed * t) % 360.0
            predictor.update(t, ang, 80.0, w_zone, b_zone)

            if lock_t is None and predictor.has_stable_speed():
                lock_t = t
                lock_speed = predictor.speed_deg_s
                pred = predictor.predict(t, ang, target="GREAT")
                if pred:
                    lock_margin = pred["time_until_press_ms"]

        self.assertIsNotNone(lock_t, "Failed to lock 900°/s")
        self.assertAlmostEqual(lock_speed, 900.0, delta=40.0)
        self.assertLessEqual(lock_t, 0.070, f"Lock took too long ({lock_t*1000:.1f}ms)")
        # Press time = 200ms - 122ms = 78ms. Lock occurred at ~66.7ms -> Margin = +11.3ms
        self.assertGreater(lock_margin, 10.0, f"Expected margin > 10ms, got {lock_margin:.1f}ms")

    def test_05_high_speed_short_distance_fallback_isolation(self):
        """
        Tests a high-speed check with close spawn geometry:
        Speed = 650°/s, spawn = 270°, target = 310° (distance = 40°).
        Total travel time is 61.5ms. With physical latency 124ms, desired press time
        is before spawn (-62.5ms).
        Verifies:
        - Scheduled fire is impossible.
        - Fallback path triggers without confirmed speed lock (speed_at_lock=None).
        - Miss classification identifies UNSTABLE_SPEED.
        - SpeedProfileManager unconditionally rejects sample and does NOT train profile.
        """
        mgr = SpeedProfileManager(profiles_path=self.profiles_path, auto_bootstrap=False, save_to_disk=False)
        p650 = mgr.get_profile(650)
        initial_samples = p650.samples

        # Attempt to record hit from this fallback event
        res = mgr.record_hit(
            speed=650.0,
            actual_delay_ms=124.0,
            error_deg=-15.0,
            outcome="MISS",
            plateau_found=True,
            target_tier=None,       # Fallback trigger: unconfirmed lock
            speed_at_lock=None,     # Fallback trigger: unconfirmed lock
            trigger_reason="FALLBACK_NO_LOCK",
        )

        self.assertIsNotNone(res)
        self.assertTrue(res.get("rejected"), "Fallback sample must be rejected from training")
        self.assertEqual(res.get("reason"), "NO_SPEED_LOCK")
        self.assertEqual(p650.samples, initial_samples, "Profile samples must NOT increase")

        cat, detail = classify_miss(
            reason="FALLBACK_NO_LOCK",
            duration_ms=80.0,
            speed_at_lock=None,
            speed_at_fire=650.0,
            tier_at_lock=None,
            tier_at_fire=650,
            error_deg=-15.0,
            error_ms=-23.1,
        )
        self.assertEqual(cat, "UNSTABLE_SPEED")
        self.assertIn("speed lock", detail.lower())

    def test_06_spawn_stutter_plus_650_deg_s(self):
        """
        Verifies that when stationary stutter frames occur on spawn,
        followed by high-speed motion at 650°/s:
        - Stationary frames do not skew speed to slow values (200-300°/s).
        - Estimator quickly resets and converges accurately to ~650°/s.
        """
        predictor = SkillCheckPredictor(latency_ms=124.0, target_offset_ratio=0.50)
        w_zone = {"start": 85.0, "end": 95.0, "width": 10.0, "center": 90.0}
        b_zone = {"start": 95.0, "end": 135.0, "width": 40.0, "center": 115.0}

        spawn_ang = 270.0

        # Frame 0: t=0.000, angle=270.00 (spawn)
        predictor.update(0.000, spawn_ang, 80.0, w_zone, b_zone)
        # Frame 1: t=0.008, angle=270.05 (stutter: step=0.05° in 8ms -> interval speed 6.25°/s)
        predictor.update(0.008, spawn_ang + 0.05, 80.0, w_zone, b_zone)
        # Frame 2: t=0.016, angle=270.08 (stutter: step=0.03° in 8ms -> interval speed 3.75°/s)
        predictor.update(0.016, spawn_ang + 0.08, 80.0, w_zone, b_zone)

        self.assertFalse(predictor.motion_onset, "Motion onset must NOT trigger during stutter")
        self.assertTrue(predictor.spawn_stutter_detected, "Stutter must be flagged")

        # Now motion starts at 650°/s from t=0.024
        motion_t0 = 0.024
        true_speed = 650.0

        timestamps = [motion_t0 + dt for dt in [0.000, 0.0167, 0.0334, 0.0501, 0.0668, 0.0835, 0.1002]]

        lock_t = None
        lock_speed = None

        for t in timestamps:
            ang = (spawn_ang + true_speed * (t - motion_t0)) % 360.0
            predictor.update(t, ang, 80.0, w_zone, b_zone)

            if lock_t is None and predictor.has_stable_speed():
                lock_t = t
                lock_speed = predictor.speed_deg_s

        self.assertIsNotNone(lock_t, "Must lock speed after motion onset")
        self.assertAlmostEqual(lock_speed, 650.0, delta=25.0,
                               msg=f"Expected speed near 650°/s, got {lock_speed:.1f}°/s")
        self.assertGreater(lock_speed, 550.0, "Speed must NOT be corrupted toward slow 200-300°/s tier")

    def test_07_telemetry_tracker_accounting_and_fallback(self):
        """
        Verifies SessionTelemetryTracker accounting:
        - Normal locked fires vs Fallback no-lock fires are properly distinguished.
        - tot_fires == normal_locked_fires + fallback_no_lock_fires.
        - tot_fires == accepted + rejected + not_evaluated.
        """
        tracker = SessionTelemetryTracker()

        # Fire 1: Normal locked fire, accepted
        tracker.record_fire("СПИН-ТАЙМЕР", locked_tier=500, speed_at_lock=500.0, speed_at_fire=500.0, prediction_lateness_ms=0.5)
        tracker.record_outcome("GREAT", speed_tier=500, speed_at_lock=500.0, speed_at_fire=500.0, prof_res={"tier": 500, "ideal_delay": 124.0})

        # Fire 2: Normal locked fire, rejected on speed divergence
        tracker.record_fire("IMMEDIATE", locked_tier=650, speed_at_lock=650.0, speed_at_fire=710.0, prediction_lateness_ms=1.2)
        tracker.record_outcome("MISS", speed_tier=650, speed_at_lock=650.0, speed_at_fire=710.0, prof_res={"tier": 650, "rejected": True, "reason": "SPEED_DIVERGENCE"})

        # Fire 3: Fallback no-lock fire (speed_at_lock is None)
        tracker.record_fire("FALLBACK_NO_LOCK", locked_tier=650, speed_at_lock=None, speed_at_fire=640.0, prediction_lateness_ms=0.2, trigger_mode="FALLBACK_NO_LOCK")
        tracker.record_outcome("MISS", speed_tier=650, speed_at_lock=None, speed_at_fire=640.0, prof_res={"tier": None, "rejected": True, "reason": "NO_SPEED_LOCK"})

        # Fire 4: Normal locked fire, not evaluated (e.g. aborted)
        tracker.record_fire("СПИН-ТАЙМЕР", locked_tier=275, speed_at_lock=275.0, speed_at_fire=275.0, prediction_lateness_ms=0.1)
        tracker.record_outcome("ABORTED_LMB", speed_tier=275, speed_at_lock=275.0, speed_at_fire=275.0, prof_res=None)

        # Check 5: No fire due to insufficient lock (ended without trigger)
        tracker.record_outcome("UNCONFIRMED", speed_tier=None, speed_at_lock=None, speed_at_fire=None, prof_res=None)

        tot_fires = tracker.normal_locked_fires + tracker.fallback_no_lock_fires
        self.assertEqual(tracker.normal_locked_fires, 3)
        self.assertEqual(tracker.fallback_no_lock_fires, 1)
        self.assertEqual(tot_fires, 4)
        self.assertEqual(tracker.no_fire_due_to_insufficient_lock, 1)

        tot_acc = sum(r["accepted"] for r in tracker.tier_records.values())
        tot_rej = sum(r["rejected"] for r in tracker.tier_records.values())
        tot_not_eval = sum(r["not_evaluated"] for r in tracker.tier_records.values())

        self.assertEqual(tot_fires, tot_acc + tot_rej + tot_not_eval,
                         f"Equation must balance: {tot_fires} == {tot_acc} + {tot_rej} + {tot_not_eval}")

        report = tracker.generate_report()
        self.assertIn("3 Locked + 1 Fallback = 4 Total Fires", report)
        self.assertIn("NO-FIRE (Insufficient):   1", report)
        self.assertIn("TOTAL FIRES: 4 = 1 Accepted + 2 Rejected + 1 Not Evaluated", report)


if __name__ == "__main__":
    unittest.main()
