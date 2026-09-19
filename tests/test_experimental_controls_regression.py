#!/usr/bin/env python3
"""
Regression test suite for Experimental Controls & Latency Isolation:
1. BASE -> VARIABLE state leak isolation across independent checks.
2. Memory immutability of active profiles and latency learner under --freeze.
3. Strict BASE_LATENCY_TEST mode enforcement (no mode switch, no profile lookup).
4. Telemetry tracker audit reporting (requested latency, min/med/max delay, overrides count).
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.predictor import (
    SkillCheckPredictor,
    SPEED_MODE_BASE,
    SPEED_MODE_VARIABLE,
    SPEED_MODE_BASE_LATENCY_TEST,
)
from core.learner import SpeedProfileManager, AdaptiveLatencyLearner, SpeedProfile
from core.telemetry_tracker import SessionTelemetryTracker


class TestExperimentalControlsRegression(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.profiles_path = self.tmp_dir / "speed_profiles.json"
        self.config_path = self.tmp_dir / "config.json"

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_base_to_variable_leak_isolation(self):
        """A new independent check (is_chain=False) must ALWAYS start in configured BASE mode."""
        p = SkillCheckPredictor(
            latency_ms=135.0,
            speed_mode=SPEED_MODE_BASE,
            session_base_speed=278.0,
        )
        self.assertEqual(p.configured_speed_mode, SPEED_MODE_BASE)
        self.assertEqual(p.active_speed_mode, SPEED_MODE_BASE)
        self.assertEqual(p.speed_mode, SPEED_MODE_BASE)

        # Feed frames at fast speed (500 deg/s) across 150ms and 75 deg to trigger mode switch
        w_zone = {"start": 200.0, "end": 210.0, "width": 10.0, "center": 205.0}
        b_zone = {"start": 211.0, "end": 250.0, "width": 39.0, "center": 230.5}
        for i in range(10):
            p.update(0.015 * i, 7.5 * i, 25.0, w_zone, b_zone)

        # Confirm mode switched during update
        self.assertTrue(p.mode_switched, "High speed must switch active mode to VARIABLE during perk check")
        self.assertEqual(p.active_speed_mode, SPEED_MODE_VARIABLE)
        self.assertEqual(p.speed_mode, SPEED_MODE_VARIABLE)

        # End of check: new independent check arrives (is_chain=False)
        p.reset(keep_speed=True, default_speed=278.0, is_chain=False, session_base_speed=278.0)

        # Assert active_speed_mode is strictly restored to configured BASE_SPEED
        self.assertEqual(p.active_speed_mode, SPEED_MODE_BASE, "Independent check MUST reset active_speed_mode to BASE")
        self.assertEqual(p.speed_mode, SPEED_MODE_BASE)
        self.assertFalse(p.mode_switched, "Independent check must clear mode_switched flag")
        self.assertAlmostEqual(p.speed_deg_s, 278.0, places=1)

    def test_freeze_immutability_active_profiles(self):
        """When freeze=True, active runtime profiles must be strictly immutable in memory."""
        # Create a manager with freeze=True and seed tier 275
        pm = SpeedProfileManager(profiles_path=self.profiles_path, freeze=True, auto_bootstrap=False)
        initial_delay = 127.27
        initial_samples = 15
        prof = SpeedProfile(
            tier_speed=275,
            delay_ms=initial_delay,
            samples=initial_samples,
            trusted=True,
            accepted_samples=15,
            rejected_samples=0,
        )
        pm.profiles[275] = prof
        pm._init_shadow_profiles()

        # Record 5 hits that would trigger a batch step on shadow profile (from 127.27 towards ~140ms)
        for _ in range(5):
            res = pm.record_hit(
                speed=278.0,
                actual_delay_ms=127.27,
                error_deg=4.0,
                outcome="GOOD",
                plateau_found=True,
                scheduler_error_ms=0.05,
                trigger_reason="СПИН-ТАЙМЕР",
            )
            self.assertIsNotNone(res)
            self.assertFalse(res.get("rejected", False), "Valid hit should not be rejected")

        # Verify active runtime profile is STRICTLY UNCHANGED
        active_prof = pm.profiles[prof.tier_speed]
        self.assertEqual(active_prof.delay_ms, initial_delay, "Active delay_ms must be strictly immutable under freeze")
        self.assertEqual(active_prof.samples, initial_samples, "Active samples count must be immutable under freeze")
        self.assertEqual(active_prof.accepted_samples, 15, "Active accepted_samples must be immutable under freeze")
        self.assertEqual(len(active_prof.batch_buffer), 0, "Active batch buffer must remain empty under freeze")
        self.assertEqual(sum(len(v) for v in pm.sector_errors.values()), 0, "Active sector errors must not accumulate under freeze")

        # Verify shadow profile DID receive the hypothetical adjustment
        shadow_prof = pm.shadow_profiles[prof.tier_speed]
        self.assertNotEqual(shadow_prof.delay_ms, initial_delay, "Shadow profile must record hypothetical learning after batch")
        self.assertEqual(shadow_prof.samples, initial_samples + 5, "Shadow profile samples must increment by 5")
        self.assertIn("shadow_delay", res, "Result must report shadow delay")

    def test_adaptive_learner_freeze_immutability(self):
        """AdaptiveLatencyLearner under freeze=True must keep latency_ms immutable."""
        learner = AdaptiveLatencyLearner(
            initial_latency_ms=140.0,
            use_speed_profiles=False,
            freeze=True,
            profiles_path=self.profiles_path,
            config_path=self.config_path,
        )
        self.assertEqual(learner.latency_ms, 140.0)

        # Simulate a trigger and check conclusion with error that would adjust latency
        learner.on_trigger(
            press_time=100.0,
            target_angle=180.0,
            speed_deg_s=278.0,
            white_zone={"start": 175.0, "end": 185.0, "width": 10.0, "center": 180.0},
            black_zone={"start": 185.0, "end": 220.0, "width": 35.0, "center": 202.5},
            used_latency_ms=140.0,
            scheduler_error_ms=0.02,
        )
        # Provide 3 stationary frames representing confirmed Roblox freeze plateau
        learner.observe_sample(100.050, 186.0, 30.0)
        learner.observe_sample(100.060, 186.0, 30.0)
        learner.observe_sample(100.070, 186.0, 30.0)
        res = learner.conclude_check()

        self.assertIsNotNone(res)
        self.assertEqual(learner.latency_ms, 140.0, "learner.latency_ms must NOT mutate when freeze=True")
        self.assertEqual(res.get("new_latency_ms"), 140.0)
        self.assertEqual(res.get("adjustment_ms"), 0.0)
        self.assertIn("hypothetical_adjustment_ms", res)

    def test_base_latency_test_mode_strict_isolation(self):
        """SPEED_MODE_BASE_LATENCY_TEST must forbid mode switching and bypass speed profiles."""
        p = SkillCheckPredictor(
            latency_ms=145.0,
            speed_mode=SPEED_MODE_BASE_LATENCY_TEST,
            session_base_speed=278.0,
        )
        self.assertEqual(p.speed_mode, SPEED_MODE_BASE_LATENCY_TEST)
        self.assertEqual(p.configured_speed_mode, SPEED_MODE_BASE_LATENCY_TEST)

        # Feed 10 frames at 600 deg/s
        for i in range(10):
            p.update(0.010 * i, 10.0 * i, 25.0, None, None)

        switched = p._evaluate_mode_switch(600.0)
        self.assertFalse(switched, "BASE_LATENCY_TEST must strictly forbid mode switching")
        self.assertEqual(p.active_speed_mode, SPEED_MODE_BASE_LATENCY_TEST)
        self.assertFalse(p.mode_switched)

        telem = p.get_shadow_telemetry()
        self.assertEqual(telem["speed_mode"], SPEED_MODE_BASE_LATENCY_TEST)
        self.assertEqual(telem["configured_speed_mode"], SPEED_MODE_BASE_LATENCY_TEST)

    def test_telemetry_tracker_latency_audit(self):
        """SessionTelemetryTracker correctly records requested latency, actual delay, and overrides."""
        tracker = SessionTelemetryTracker()

        # Fire 1: Requested 140.0ms, used 140.0ms (no override)
        tracker.record_fire(
            reason="СПИН-ТАЙМЕР",
            locked_tier=278,
            speed_at_lock=278.0,
            speed_at_fire=278.0,
            prediction_lateness_ms=0.01,
            requested_latency_ms=140.0,
            actual_delay_ms=140.0,
            is_profile_override=False,
            is_experiment_valid=True,
        )

        # Fire 2: Requested 140.0ms, used 127.3ms (profile override)
        tracker.record_fire(
            reason="СПИН-ТАЙМЕР",
            locked_tier=275,
            speed_at_lock=275.0,
            speed_at_fire=275.0,
            prediction_lateness_ms=0.02,
            requested_latency_ms=140.0,
            actual_delay_ms=127.3,
            is_profile_override=True,
            is_experiment_valid=False,
        )

        self.assertEqual(tracker.requested_base_latency, 140.0)
        self.assertEqual(tracker.profile_overrides_count, 1)
        self.assertEqual(tracker.experiment_discrepancies, 1)
        self.assertEqual(tracker.actual_used_delays, [140.0, 127.3])

        report = tracker.generate_report()
        self.assertIn("LATENCY CONTROL & EXPERIMENT AUDIT", report)
        self.assertIn("Requested Base Latency:     140.0ms", report)
        self.assertIn("min=127.3ms | median=133.7ms | max=140.0ms", report)
        self.assertIn("Profile Overrides Count:    1", report)
        self.assertIn("EXPERIMENT DISCREPANCY:  1 checks", report)


if __name__ == "__main__":
    unittest.main()
