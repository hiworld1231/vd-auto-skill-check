#!/usr/bin/env python3
"""
Unit test suite for SpeedProfileManager, multi-speed adaptive calibration,
shadow learning mode, position bias tracking, and selective media recording.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.learner import SpeedProfile, SpeedProfileManager, AdaptiveLatencyLearner
from core.flight_recorder import FlightRecorder


class TestSpeedProfiles(unittest.TestCase):
    """Tests discrete speed tier profiles and batch median updates."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.profiles_path = self.tmp_dir / "test_profiles.json"

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_speed_tier_bucketing(self):
        """Verifies speeds map to standard 25°/s game tiers."""
        mgr = SpeedProfileManager(profiles_path=self.profiles_path, auto_bootstrap=False)
        self.assertEqual(mgr.get_tier(271.4), 275)
        self.assertEqual(mgr.get_tier(286.0), 275)
        self.assertEqual(mgr.get_tier(318.0), 325)
        self.assertEqual(mgr.get_tier(374.0), 375)
        self.assertEqual(mgr.get_tier(645.0), 650)

    def test_batch_median_update_prevents_single_spike_jitter(self):
        """Verifies that a single noisy sample does not immediately move tier delay."""
        profile = SpeedProfile(tier_speed=275, delay_ms=120.0, samples=0, trusted=True)
        # Add a single noisy outlier
        step = profile.add_ideal_delay(150.0, batch_size=5)
        self.assertIsNone(step, "Must NOT adjust delay on 1 single sample (no jitter)")
        self.assertEqual(profile.delay_ms, 120.0)

        # Feed 4 more samples around 130ms (median = 130.0)
        profile.add_ideal_delay(128.0, batch_size=5)
        profile.add_ideal_delay(131.0, batch_size=5)
        profile.add_ideal_delay(129.0, batch_size=5)
        step = profile.add_ideal_delay(132.0, batch_size=5)
        self.assertIsNotNone(step, "Must commit batch update after 5 clean samples")
        # Step is clamped to max 2.5ms
        self.assertAlmostEqual(step, 2.5, delta=0.01)
        self.assertAlmostEqual(profile.delay_ms, 122.5, delta=0.01)

    def test_shadow_mode_new_speed_perk(self):
        """
        Verifies that a new perk speed (e.g. 650°/s) starts in shadow mode,
        uses the closest trusted tier delay, and becomes trusted after sufficient hits.
        """
        mgr = SpeedProfileManager(profiles_path=self.profiles_path, auto_bootstrap=False)
        # Seed an established trusted tier at 275°/s
        p275 = mgr.get_profile(275.0)
        p275.delay_ms = 124.0
        p275.samples = 25
        p275.trusted = True

        # Brand new perk speed 650°/s
        initial_lat = mgr.get_latency_for_speed(650.0)
        self.assertEqual(initial_lat, 124.0, "Must borrow delay from closest trusted tier")

        p650 = mgr.get_profile(650.0)
        self.assertFalse(p650.trusted, "New tier must start untrusted (shadow mode)")
        self.assertEqual(p650.confidence, "low")

        # Simulate 10 clean hits at 650°/s
        for _ in range(10):
            res = mgr.record_hit(
                speed=650.0,
                actual_delay_ms=124.0,
                error_deg=0.5,
                outcome="GREAT",
                plateau_found=True,
                target_angle=90.0,
            )
            self.assertIsNotNone(res)

        self.assertTrue(p650.trusted, "Profile must become trusted after 8+ clean samples")
        self.assertIn(p650.confidence, ("medium", "high"))

    def test_noise_and_abort_rejection(self):
        """Verifies UNCONFIRMED, ABORTED, and unverified checks do not corrupt training."""
        mgr = SpeedProfileManager(profiles_path=self.profiles_path, auto_bootstrap=False)
        p = mgr.get_profile(275.0)
        p.delay_ms = 120.0

        # UNCONFIRMED
        res = mgr.record_hit(275.0, 120.0, 15.0, "UNCONFIRMED", plateau_found=False)
        self.assertIsNone(res)
        self.assertEqual(p.samples, 0)

        # ABORTED
        res = mgr.record_hit(275.0, 120.0, 0.0, "ABORTED", plateau_found=True, is_aborted=True)
        self.assertIsNone(res)
        self.assertEqual(p.samples, 0)

        # Extreme non-physical anomaly (e.g. 500ms delay) -> rejected from training
        res = mgr.record_hit(275.0, 120.0, 200.0, "MISS", plateau_found=True)
        self.assertIsNotNone(res)
        self.assertTrue(res.get("rejected"))
        self.assertEqual(p.samples, 0)
        self.assertEqual(p.rejected_samples, 1)

    def test_bootstrap_from_real_replays(self):
        """Verifies bootstrapping loads profiles from the historical replays/ directory as epoch 1 shadow priors."""
        mgr = SpeedProfileManager(profiles_path=self.profiles_path, auto_bootstrap=True)
        self.assertGreater(len(mgr.profiles), 3, "Must bootstrap multiple speed tiers from replays")
        self.assertIn(275, mgr.profiles, "Standard 275°/s tier must exist")
        self.assertFalse(mgr.profiles[275].trusted, "Historical replay tiers start as shadow epoch 1 (untrusted)")
        self.assertEqual(mgr.profiles[275].epoch, 1)
        self.assertGreater(mgr.profiles[275].samples, 50, "Must have 50+ samples for standard tier")

        summary = mgr.get_summary()
        self.assertIn("SKILL CHECK SPEED PROFILES", summary)
        self.assertIn("POSITION BIAS", summary)

    def test_adaptive_learner_speed_integration(self):
        """Verifies AdaptiveLatencyLearner dynamically routes lookups to SpeedProfileManager."""
        learner = AdaptiveLatencyLearner(auto_save=False, profiles_path=self.profiles_path, use_speed_profiles=True)
        lat_280 = learner.get_latency_for_speed(280.0)
        self.assertGreater(lat_280, 80.0)
        self.assertLess(lat_280, 160.0)

        summary = learner.get_training_summary()
        self.assertIn("speed ~275", summary)


class TestSelectiveFlightRecorder(unittest.TestCase):
    """Tests that FlightRecorder only renders heavy media on MISS/unconfirmed/anomalies."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_selective_recording_rules(self):
        rec = FlightRecorder(output_dir=self.tmp_dir, record_all=False, save_video=True, save_diagnostic_strip=True)

        # Normal clean GREAT hit (low error, normal speed) -> Should NOT render MP4
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frames = [(10.0 + i * 0.008, dummy_frame, {}, {}, False) for i in range(15)]

        episode_great = {
            "check_id": "test_great",
            "start_monotonic": 10.0,
            "frames": frames,
            "trigger_event": {"speed_deg_s": 275.0, "latency_ms": 120.0},
            "outcome_info": {"outcome": "GREAT", "error_deg": 0.5},
        }
        rec._process_episode(episode_great)

        self.assertTrue((self.tmp_dir / "test_great_GREAT.json").exists(), "JSON must always be saved")
        self.assertFalse((self.tmp_dir / "test_great_GREAT.mp4").exists(), "Clean GREAT should NOT render MP4")

        # MISS -> MUST render MP4
        episode_miss = {
            "check_id": "test_miss",
            "start_monotonic": 20.0,
            "frames": frames,
            "trigger_event": {"speed_deg_s": 275.0, "latency_ms": 120.0},
            "outcome_info": {"outcome": "MISS", "error_deg": -15.0},
        }
        rec._process_episode(episode_miss)
        self.assertTrue((self.tmp_dir / "test_miss_MISS.json").exists())
        self.assertTrue((self.tmp_dir / "test_miss_MISS.mp4").exists(), "MISS must render MP4")

        # UNCONFIRMED -> MUST render MP4
        episode_unconf = {
            "check_id": "test_unconf",
            "start_monotonic": 30.0,
            "frames": frames,
            "trigger_event": {"speed_deg_s": 275.0, "latency_ms": 120.0},
            "outcome_info": {"outcome": "UNCONFIRMED", "error_deg": 0.0},
        }
        rec._process_episode(episode_unconf)
        self.assertTrue((self.tmp_dir / "test_unconf_UNCONFIRMED.json").exists())
        self.assertTrue((self.tmp_dir / "test_unconf_UNCONFIRMED.mp4").exists(), "UNCONFIRMED must render MP4")

        # New speed record (e.g. 600°/s) -> MUST render MP4 even if GREAT
        episode_record = {
            "check_id": "test_speed_record",
            "start_monotonic": 40.0,
            "frames": frames,
            "trigger_event": {"speed_deg_s": 600.0, "latency_ms": 120.0},
            "outcome_info": {"outcome": "GREAT", "error_deg": 1.0},
        }
        rec._process_episode(episode_record)
        self.assertTrue((self.tmp_dir / "test_speed_record_GREAT.mp4").exists(), "Speed record must render MP4")

        rec.close()


if __name__ == "__main__":
    unittest.main()
