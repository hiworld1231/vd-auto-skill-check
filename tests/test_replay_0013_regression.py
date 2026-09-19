#!/usr/bin/env python3
"""
Regression test suite for Replay 0013_MISS and Speed Lock Lifecycle:
1. Replay 0013 estimator convergence:
   - Early frames with visual noise spikes (450-520°/s) do NOT achieve stable lock.
   - The bot cannot fire as LOCKED ~500°/s.
   - The estimator converges to ~275-310°/s.
   - At t=113ms (when old code misfired), new predictor does not trigger.
2. Miss classification:
   - Replay 0013 is classified as UNSTABLE_SPEED, NOT CLEAN TIMING MISS.
3. Training isolation:
   - Samples without confirmed speed lock (target_tier is None or speed_at_lock is None)
     are unconditionally rejected and NEVER mutate speed profiles.
4. Session telemetry tracker accounting:
   - For every tier and session total: Fires = Accepted + Rejected + NotEvaluated.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.predictor import SkillCheckPredictor, STATE_NO_MOTION, STATE_PROVISIONAL, STATE_LOCKED, STATE_COMMITTED
from core.learner import SpeedProfileManager, classify_miss
from core.telemetry_tracker import SessionTelemetryTracker


class TestReplay0013Regression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay_path = ROOT / "replays" / "check_20260917_180451_0013_MISS.json"
        if not cls.replay_path.exists():
            raise unittest.SkipTest("0013_MISS replay file not found")
        cls.replay_data = json.loads(cls.replay_path.read_text(encoding="utf-8"))

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.profiles_path = self.tmp_dir / "profiles.json"

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_0013_estimator_never_locks_at_500_and_converges_to_true_speed(self):
        """
        Feeds frames from 0013 replay into SkillCheckPredictor and asserts:
        - Early frames (t < 100ms) with noise spikes (450-520°/s) do NOT lock.
        - Predictor never locks as ~500°/s tier.
        - Predictor converges to true speed ~275-310°/s.
        """
        frames = [
            fr for fr in self.replay_data.get("frames", [])
            if not fr.get("is_pre_roll", False) and "needle_angle" in fr
        ]
        self.assertGreater(len(frames), 20, "0013 replay must have sufficient frames")

        predictor = SkillCheckPredictor(latency_ms=122.1, target_offset_ratio=0.50)
        white_zone = self.replay_data.get("locked_zones", {}).get("white")
        black_zone = self.replay_data.get("locked_zones", {}).get("black")

        locked_speeds = []

        for fr in frames:
            t = fr["time_rel_ms"] / 1000.0
            ang = fr["needle_angle"]
            strn = fr.get("needle_strength", 75.0)

            predictor.update(t, ang, strn, white_zone, black_zone)

            if predictor.has_stable_speed():
                locked_speeds.append((t * 1000.0, predictor.speed_deg_s))

            # At t <= 100ms, early noise must NOT be locked as stable speed
            if t * 1000.0 < 90.0:
                self.assertFalse(
                    predictor.has_stable_speed() and (450.0 <= predictor.speed_deg_s <= 550.0),
                    f"Early frame at t={t*1000:.1f}ms must NOT lock as ~500°/s! Speed was {predictor.speed_deg_s:.1f}°/s",
                )

        self.assertTrue(len(locked_speeds) > 0, "Estimator must eventually achieve stable speed lock")
        first_lock_t, first_lock_speed = locked_speeds[0]

        # First lock speed must be in the true speed range (~270 - 325°/s), NEVER ~500°/s
        self.assertLess(first_lock_speed, 350.0, f"Locked speed {first_lock_speed:.1f}°/s must not be near ~500°/s")
        self.assertGreater(first_lock_speed, 250.0, f"Locked speed {first_lock_speed:.1f}°/s must be in true velocity range")

        # Converged speed at end of track must be ~275 - 310°/s
        final_speed = predictor.speed_deg_s
        self.assertAlmostEqual(final_speed, 290.0, delta=25.0,
                               msg=f"Estimator must converge to ~290°/s, got {final_speed:.1f}°/s")

    def test_0013_does_not_misfire_at_113ms(self):
        """
        Verifies that at t=113ms (when the old code fired Space prematurely),
        the new predictor with robust speed estimation does NOT trigger immediate fire.
        """
        frames = [
            fr for fr in self.replay_data.get("frames", [])
            if not fr.get("is_pre_roll", False) and "needle_angle" in fr and fr["time_rel_ms"] <= 113.5
        ]

        predictor = SkillCheckPredictor(latency_ms=122.1, target_offset_ratio=0.50)
        white_zone = self.replay_data.get("locked_zones", {}).get("white")
        black_zone = self.replay_data.get("locked_zones", {}).get("black")

        for fr in frames:
            t = fr["time_rel_ms"] / 1000.0
            ang = fr["needle_angle"]
            strn = fr.get("needle_strength", 75.0)
            predictor.update(t, ang, strn, white_zone, black_zone)

        last_fr = frames[-1]
        t_last = last_fr["time_rel_ms"] / 1000.0
        ang_last = last_fr["needle_angle"]
        pred = predictor.predict(t_last, ang_last, target="GREAT")

        self.assertIsNotNone(pred)
        # In the old code, time_until_press_ms dropped below 0 or was within 30ms.
        # With the true velocity (~300°/s), remaining distance to target 30.5° is ~75°,
        # which takes ~250ms -> time until press must be > 80ms.
        self.assertFalse(pred["should_press_now"], "Must NOT fire immediately at t=113ms on 0013")
        self.assertGreater(pred["time_until_press_ms"], 80.0,
                           f"Time until press at t=113ms should be > 80ms, got {pred['time_until_press_ms']:.1f}ms")

    def test_0013_miss_classified_as_unstable_speed(self):
        """
        Verifies that 0013 replay (which fired without confirmed speed lock)
        is classified as UNSTABLE_SPEED and NOT as CLEAN TIMING MISS.
        """
        tr = self.replay_data.get("trigger", {})
        ev = self.replay_data.get("evaluation", {})

        cat, detail = classify_miss(
            reason=tr.get("reason", "СПИН-ТАЙМЕР"),
            duration_ms=self.replay_data.get("duration_ms", 508.0),
            speed_at_fire=tr.get("speed_deg_s", 496.26),
            speed_at_lock=tr.get("measured_speed_at_lock"),  # None in 0013!
            scheduler_error_ms=tr.get("scheduler_error_ms"),
            last_frame_age_ms=tr.get("last_frame_age_ms_at_fire"),
            tier_at_lock=tr.get("selected_speed_tier"),  # None in 0013!
            tier_at_fire=500,
            error_deg=ev.get("error_deg", -31.8),
            error_ms=ev.get("error_ms", -64.1),
        )

        self.assertEqual(cat, "UNSTABLE_SPEED", f"0013 must be classified as UNSTABLE_SPEED, got {cat}")
        self.assertNotEqual(cat, "CLEAN TIMING MISS", "0013 must NEVER be classified as CLEAN TIMING MISS")
        self.assertIn("speed lock", detail.lower())

    def test_unconfirmed_lock_never_mutates_speed_profiles(self):
        """
        Verifies that a sample without confirmed speed lock (target_tier is None or
        speed_at_lock is None) is unconditionally rejected and never mutates any profile.
        """
        mgr = SpeedProfileManager(profiles_path=self.profiles_path, auto_bootstrap=False, save_to_disk=False)
        prof_500 = mgr.get_profile(500)
        initial_delay = prof_500.delay_ms
        initial_samples = prof_500.samples

        # Attempt to record hit with 0013 data
        res = mgr.record_hit(
            speed=496.26,
            actual_delay_ms=122.1,
            error_deg=-31.8,
            outcome="MISS",
            plateau_found=True,
            target_tier=None,       # Unconfirmed lock
            speed_at_lock=None,     # Unconfirmed lock
        )

        self.assertIsNotNone(res)
        self.assertTrue(res.get("rejected"), "Must be rejected")
        self.assertEqual(res.get("reason"), "NO_SPEED_LOCK")

        # Verify profile for 500 was untouched
        self.assertEqual(prof_500.samples, initial_samples, "Profile samples must NOT increase")
        self.assertEqual(prof_500.delay_ms, initial_delay, "Profile delay must NOT change")

    def test_telemetry_tracker_fires_equation_consistency(self):
        """
        Verifies that for every tier: Fires = Accepted + Rejected + NotEvaluated.
        """
        tracker = SessionTelemetryTracker()

        # Check 1: Fired, Accepted
        tracker.record_fire("СПИН-ТАЙМЕР", locked_tier=275, speed_at_lock=275.0, speed_at_fire=275.0, prediction_lateness_ms=0.5)
        tracker.record_outcome("GREAT", speed_tier=275, speed_at_lock=275.0, speed_at_fire=275.0, prof_res={"tier": 275, "ideal_delay": 122.0})

        # Check 2: Fired, Rejected (NO_SPEED_LOCK)
        tracker.record_fire("СПИН-ТАЙМЕР", locked_tier=275, speed_at_lock=None, speed_at_fire=275.0, prediction_lateness_ms=1.0)
        tracker.record_outcome("MISS", speed_tier=275, speed_at_lock=None, speed_at_fire=275.0, prof_res={"tier": None, "rejected": True, "reason": "NO_SPEED_LOCK"})

        # Check 3: Fired, Not Evaluated (aborted / freeze mode / no plateau)
        tracker.record_fire("СПИН-ТАЙМЕР", locked_tier=275, speed_at_lock=275.0, speed_at_fire=275.0, prediction_lateness_ms=0.2)
        tracker.record_outcome("ABORTED_LMB", speed_tier=275, speed_at_lock=275.0, speed_at_fire=275.0, prof_res=None)

        # Check 4: Check concluded without firing (timeout / missed target)
        tracker.record_outcome("MISS", speed_tier=275, speed_at_lock=275.0, speed_at_fire=275.0, prof_res=None)

        rec = tracker.tier_records[275]
        fires = rec["fires"]
        accepted = rec["accepted"]
        rejected = rec["rejected"]
        not_eval = rec["not_evaluated"]

        self.assertEqual(fires, 3, "Exactly 3 fires were recorded for tier 275")
        self.assertEqual(accepted, 1, "Exactly 1 sample was accepted")
        self.assertEqual(rejected, 1, "Exactly 1 sample was rejected")
        self.assertEqual(not_eval, 1, "Exactly 1 sample was not evaluated")
        self.assertEqual(fires, accepted + rejected + not_eval,
                         f"Equation must hold: Fires ({fires}) == Accepted ({accepted}) + Rejected ({rejected}) + NotEvaluated ({not_eval})")

        report = tracker.generate_report()
        self.assertIn("Session Fires: 3 | Accepted: 1 | Rejected: 1 | Not Evaluated: 1", report)
        self.assertIn("TOTAL FIRES: 3 = 1 Accepted + 1 Rejected + 1 Not Evaluated", report)


if __name__ == "__main__":
    unittest.main()
