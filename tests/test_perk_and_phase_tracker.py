#!/usr/bin/env python3
"""
Unit tests for Violent District QTE Speed Perk, Target Geometry, and Response Phase Tracking.
Verifies mathematical and lifecycle invariants strictly without simulated hardware delays.
"""

import math
import unittest
from core.predictor import SkillCheckPredictor, SPEED_MODE_BASE, SPEED_MODE_VARIABLE, SPEED_MODE_GEN_RUSH
from core.response_phase_tracker import ResponsePhaseTracker


class TestTargetGeometryInvariant(unittest.TestCase):
    def test_target_great_strictly_equals_white_center(self):
        predictor = SkillCheckPredictor(
            latency_ms=122.1,
            target_offset_ratio=0.5,
            speed_mode=SPEED_MODE_GEN_RUSH,
            session_base_speed=278.0,
        )

        white_zone = {"start": 30.0, "end": 42.0, "center": 36.0, "width": 12.0}
        black_zone = {"start": 42.0, "end": 80.0, "center": 61.0, "width": 38.0}

        now = 100.0
        # Feed trajectory
        predictor.update(now + 0.000, 0.0, 50.0, white_zone, black_zone)
        predictor.update(now + 0.010, 3.0, 50.0, white_zone, black_zone)
        predictor.update(now + 0.020, 6.0, 50.0, white_zone, black_zone)
        predictor.update(now + 0.030, 9.0, 50.0, white_zone, black_zone)

        pred = predictor.predict(now + 0.030, 9.0, target="GREAT")
        self.assertIsNotNone(pred)
        self.assertAlmostEqual(pred["target_angle"], white_zone["center"] % 360.0, delta=1e-3)

    def test_target_great_independent_of_ratio_crutches(self):
        # Even if someone accidentally set target_offset_ratio to 0.40 or 0.60
        predictor = SkillCheckPredictor(
            latency_ms=122.1,
            target_offset_ratio=0.30,  # Biased ratio
            speed_mode=SPEED_MODE_GEN_RUSH,
            session_base_speed=278.0,
        )

        white_zone = {"start": 120.0, "end": 134.0, "center": 127.0, "width": 14.0}
        now = 10.0
        predictor.update(now, 50.0, 50.0, white_zone, None)
        predictor.update(now + 0.010, 53.0, 50.0, white_zone, None)

        pred = predictor.predict(now + 0.010, 53.0, target="GREAT")
        self.assertIsNotNone(pred)
        # Target GREAT must strictly be white_center, ignoring ratio crutch
        self.assertAlmostEqual(pred["target_angle"], 127.0, delta=1e-3)


class TestResponsePhaseTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = ResponsePhaseTracker(
            base_latency_ms=122.1,
            hard_cap_ms=20.0,
            max_step_ms=4.0,
            window_size=10,
            idle_threshold_s=45.0,
        )

    def test_filter_rejects_unqualified_samples(self):
        # 1. Non-scheduled trigger
        qual, reason = self.tracker.is_clean_sample(
            outcome="GREAT",
            plateau_found=True,
            trigger_reason="IMMEDIATE",
            scheduler_error_ms=0.1,
            speed_at_lock=280.0,
            speed_at_fire=280.0,
            last_frame_age_ms=5.0,
            actual_used_delay_ms=122.1,
            is_detector_fallback=False,
        )
        self.assertFalse(qual)
        self.assertIn("Non-scheduled", reason)

        # 2. Scheduler jitter > 2.0 ms
        qual, reason = self.tracker.is_clean_sample(
            outcome="GREAT",
            plateau_found=True,
            trigger_reason="СПИН-ТАЙМЕР",
            scheduler_error_ms=2.5,
            speed_at_lock=280.0,
            speed_at_fire=280.0,
            last_frame_age_ms=5.0,
            actual_used_delay_ms=122.1,
            is_detector_fallback=False,
        )
        self.assertFalse(qual)
        self.assertIn("jitter too high", reason)

        # 3. Plateau not confirmed
        qual, reason = self.tracker.is_clean_sample(
            outcome="GREAT",
            plateau_found=False,
            trigger_reason="СПИН-ТАЙМЕР",
            scheduler_error_ms=0.2,
            speed_at_lock=280.0,
            speed_at_fire=280.0,
            last_frame_age_ms=5.0,
            actual_used_delay_ms=122.1,
            is_detector_fallback=False,
        )
        self.assertFalse(qual)
        self.assertIn("Plateau", reason)

        # 4. Detector fallback
        qual, reason = self.tracker.is_clean_sample(
            outcome="GREAT",
            plateau_found=True,
            trigger_reason="СПИН-ТАЙМЕР",
            scheduler_error_ms=0.2,
            speed_at_lock=280.0,
            speed_at_fire=280.0,
            last_frame_age_ms=5.0,
            actual_used_delay_ms=122.1,
            is_detector_fallback=True,
        )
        self.assertFalse(qual)
        self.assertIn("Detector used fallback", reason)

        # 5. Clean sample
        qual, reason = self.tracker.is_clean_sample(
            outcome="GREAT",
            plateau_found=True,
            trigger_reason="СПИН-ТАЙМЕР",
            scheduler_error_ms=0.05,
            speed_at_lock=280.0,
            speed_at_fire=280.0,
            last_frame_age_ms=5.0,
            actual_used_delay_ms=122.1,
            is_detector_fallback=False,
        )
        self.assertTrue(qual)
        self.assertEqual(reason, "CLEAN")

    def test_damped_correction_and_rate_limiting(self):
        t = 100.0
        self.tracker.last_activity_mono = t
        # Feed clean samples indicating needle is hitting early by 15 ms (negative error)
        # 1st sample: insufficient history for rolling median (need >= 3)
        res1 = self.tracker.record_outcome(
            center_error_ms=-15.0,
            outcome="GREAT",
            plateau_found=True,
            trigger_reason="СПИН-ТАЙМЕР",
            scheduler_error_ms=0.01,
            speed_at_lock=350.0,
            speed_at_fire=350.0,
            last_frame_age_ms=4.0,
            actual_used_delay_ms=122.1,
            now=t,
        )
        self.assertTrue(res1["accepted"])
        self.assertEqual(self.tracker.phase_correction_ms, 0.0)

        # 2nd sample
        res2 = self.tracker.record_outcome(
            center_error_ms=-14.0,
            outcome="GREAT",
            plateau_found=True,
            trigger_reason="СПИН-ТАЙМЕР",
            scheduler_error_ms=0.02,
            speed_at_lock=350.0,
            speed_at_fire=350.0,
            last_frame_age_ms=4.0,
            actual_used_delay_ms=122.1,
            now=t + 5.0,
        )
        self.assertEqual(self.tracker.phase_correction_ms, 0.0)

        # 3rd sample: median = -14.0 ms. Max step is 4.0 ms.
        # Should adjust phase_correction_ms towards -14.0 ms by at most 4.0 ms!
        res3 = self.tracker.record_outcome(
            center_error_ms=-15.0,
            outcome="GREAT",
            plateau_found=True,
            trigger_reason="СПИН-ТАЙМЕР",
            scheduler_error_ms=0.01,
            speed_at_lock=350.0,
            speed_at_fire=350.0,
            last_frame_age_ms=4.0,
            actual_used_delay_ms=122.1,
            now=t + 10.0,
        )
        self.assertAlmostEqual(self.tracker.phase_correction_ms, -4.0, delta=1e-2)
        # Effective latency = 122.1 - 4.0 = 118.1 ms
        self.assertAlmostEqual(self.tracker.get_effective_latency(chain_count=1), 118.1, delta=1e-2)

    def test_hard_cap_enforcement(self):
        t = 100.0
        self.tracker.last_activity_mono = t
        # Force feed multiple large negative error samples
        for i in range(10):
            self.tracker.record_outcome(
                center_error_ms=-30.0,
                outcome="GREAT",
                plateau_found=True,
                trigger_reason="СПИН-ТАЙМЕР",
                scheduler_error_ms=0.01,
                speed_at_lock=350.0,
                speed_at_fire=350.0,
                last_frame_age_ms=4.0,
                actual_used_delay_ms=122.1,
                now=t + i * 2.0,
            )
        # Must not exceed hard cap of -20.0 ms
        self.assertGreaterEqual(self.tracker.phase_correction_ms, -20.0)
        self.assertAlmostEqual(self.tracker.phase_correction_ms, -20.0, delta=1e-2)

    def test_idle_gap_decay(self):
        t = 100.0
        # Set an initial correction
        self.tracker.phase_correction_ms = -10.0
        self.tracker.last_activity_mono = t
        self.tracker.clean_errors_ms.append(-10.0)

        # 50 seconds idle (> 45s timeout)
        gap = self.tracker.check_idle(now=t + 50.0)
        self.assertIsNotNone(gap)
        # Correction should decay by 50% towards 0 -> -5.0 ms
        self.assertAlmostEqual(self.tracker.phase_correction_ms, -5.0, delta=1e-2)
        # Clean sample window should be cleared
        self.assertEqual(len(self.tracker.clean_errors_ms), 0)


class TestGenRushPredictorInvariants(unittest.TestCase):
    def test_no_early_lock_in_gen_rush_mode(self):
        predictor = SkillCheckPredictor(
            latency_ms=122.1,
            target_offset_ratio=0.5,
            speed_mode=SPEED_MODE_GEN_RUSH,
            session_base_speed=278.0,
        )

        now = 100.0
        white_zone = {"start": 40.0, "end": 50.0, "center": 45.0, "width": 10.0}

        # 3 frames (25 ms)
        predictor.update(now + 0.000, 0.0, 50.0, white_zone, None)
        predictor.update(now + 0.012, 5.0, 50.0, white_zone, None)
        predictor.update(now + 0.024, 10.0, 50.0, white_zone, None)

        # In GEN_RUSH mode, has_stable_speed() MUST return False at frame 3 (25 ms)!
        self.assertFalse(predictor.has_stable_speed())

    def test_clean_independent_check_reset(self):
        predictor = SkillCheckPredictor(
            latency_ms=122.1,
            target_offset_ratio=0.5,
            speed_mode=SPEED_MODE_GEN_RUSH,
            session_base_speed=278.0,
        )

        # Simulate a high perk speed lock
        predictor.speed_deg_s = 550.0
        predictor.active_speed_mode = SPEED_MODE_VARIABLE
        predictor.mode_switched = True

        # When an independent check concludes (is_chain=False):
        predictor.reset(keep_speed=False, default_speed=None, is_chain=False)

        # Speed must cleanly revert to base speed, mode_switched must be False
        self.assertAlmostEqual(predictor.speed_deg_s, 278.0, delta=1e-3)
        self.assertEqual(predictor.active_speed_mode, SPEED_MODE_GEN_RUSH)
        self.assertFalse(predictor.mode_switched)


if __name__ == "__main__":
    unittest.main()
