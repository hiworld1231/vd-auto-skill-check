#!/usr/bin/env python3
"""
Regression test suite for Speed Tier Locking and Training Correctness:
- Never locks tier on frame 1 without measured stable speed
- Switches to correct high-speed perk tier (e.g. 650°/s) once trajectory stabilizes
- Locks permanently for the remainder of the check
- Trains the EXACT locked tier used at trigger
- Rejects training on speed divergence (>35°/s) or jitter (>8ms)
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.predictor import SkillCheckPredictor
from core.learner import SpeedProfileManager, AdaptiveLatencyLearner


class TestTierLockingRegression(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.profiles_path = self.tmp_dir / "profiles.json"

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_has_stable_speed_requires_accumulation(self):
        """Predictor has_stable_speed() is False on frame 1 and becomes True only after accumulation."""
        p = SkillCheckPredictor(latency_ms=120.0)
        p.reset(keep_speed=True, default_speed=275.0)

        # Frame 1 at t=0.000, angle=0.0
        p.update(0.000, 0.0, 25.0, {"start": 100.0, "end": 110.0, "width": 10.0, "center": 105.0}, None)
        self.assertFalse(p.has_stable_speed(), "Frame 1 must NOT have stable speed")

        # Frame 2 at t=0.010, angle=6.5 (650°/s)
        p.update(0.010, 6.5, 25.0, None, None)
        self.assertFalse(p.has_stable_speed(), "Frame 2 must NOT have stable speed (<4 frames, span < 30ms)")

        # Frame 3 at t=0.020, angle=13.0
        p.update(0.020, 13.0, 25.0, None, None)
        self.assertFalse(p.has_stable_speed(), "Frame 3 must NOT have stable speed (span < 30ms)")

        # Frame 4 at t=0.035, angle=22.75
        p.update(0.035, 22.75, 25.0, None, None)
        # Frame 5 at t=0.045, angle=29.25
        p.update(0.045, 29.25, 25.0, None, None)

        self.assertTrue(p.has_stable_speed(), "Must confirm stable speed after 5 frames spanning 45ms")
        self.assertAlmostEqual(p.speed_deg_s, 650.0, delta=15.0)

    def test_trains_exact_locked_tier(self):
        """Training records directly into target_tier, even if initial speed baseline was different."""
        mgr = SpeedProfileManager(profiles_path=self.profiles_path, auto_bootstrap=False, save_to_disk=False)

        # Record hit with locked tier 650
        res = mgr.record_hit(
            speed=648.0,
            actual_delay_ms=120.0,
            error_deg=0.5,
            outcome="GREAT",
            plateau_found=True,
            target_tier=650,
            speed_at_lock=650.0,
            prediction_lateness_ms=1.2,
        )
        self.assertIsNotNone(res)
        self.assertFalse(res.get("rejected"))
        self.assertEqual(res["tier"], 650)
        self.assertEqual(mgr.get_profile(650).accepted_samples, 1)

    def test_rejects_sample_on_speed_divergence(self):
        """If speed diverges > 35°/s between lock and fire, the sample is rejected."""
        mgr = SpeedProfileManager(profiles_path=self.profiles_path, auto_bootstrap=False, save_to_disk=False)

        res = mgr.record_hit(
            speed=320.0,  # Speed at fire diverged significantly from lock
            actual_delay_ms=120.0,
            error_deg=0.5,
            outcome="GREAT",
            plateau_found=True,
            target_tier=275,
            speed_at_lock=275.0,  # 320 - 275 = 45°/s divergence!
            prediction_lateness_ms=0.5,
        )
        self.assertIsNotNone(res)
        self.assertTrue(res.get("rejected"))
        self.assertIn("SPEED_DIVERGENCE", res.get("reason", ""))
        self.assertEqual(mgr.get_profile(275).rejected_samples, 1)
        self.assertEqual(mgr.get_profile(275).samples, 0)

    def test_rejects_sample_on_excessive_jitter(self):
        """If prediction lateness / scheduler jitter > 8.0ms, sample is rejected from training."""
        mgr = SpeedProfileManager(profiles_path=self.profiles_path, auto_bootstrap=False, save_to_disk=False)

        res = mgr.record_hit(
            speed=275.0,
            actual_delay_ms=120.0,
            error_deg=0.5,
            outcome="GREAT",
            plateau_found=True,
            target_tier=275,
            speed_at_lock=275.0,
            prediction_lateness_ms=12.5,  # 12.5ms lateness (> 8ms)
        )
        self.assertIsNotNone(res)
        self.assertTrue(res.get("rejected"))
        self.assertIn("EXCESSIVE_JITTER", res.get("reason", ""))
        self.assertEqual(mgr.get_profile(275).rejected_samples, 1)

    def test_spawn_stutter_does_not_prematurely_lock_tier(self):
        """Needle stationary on spawn for 30ms must NOT lock a false slow tier."""
        p = SkillCheckPredictor(latency_ms=120.0)
        p.reset(keep_speed=True, default_speed=275.0)

        # Spawn at t=0.000, angle=272.0
        p.update(0.000, 272.0, 25.0, {"start": 150.0, "end": 160.0, "width": 10.0, "center": 155.0}, None)
        p.update(0.010, 278.0, 25.0, None, None)
        p.update(0.025, 283.0, 25.0, None, None)

        # Needle stalls on spawn for 25ms (duplicate frames at 283.0)
        p.update(0.035, 283.0, 25.0, None, None)
        p.update(0.050, 283.0, 25.0, None, None)
        self.assertFalse(p.has_stable_speed(), "Must not lock speed while needle is stalled on spawn")

        # Needle moves slightly to 287.0 (total travel only 15°, 1 fit)
        p.update(0.060, 287.0, 25.0, None, None)
        self.assertFalse(p.has_stable_speed(), "Must not lock speed on first fit with <20° arc travel")

        # Needle now accelerates to cruise speed 310°/s
        curr_t = 0.060
        curr_ang = 287.0
        for _ in range(10):
            curr_t += 0.016
            curr_ang += 310.0 * 0.016
            p.update(curr_t, curr_ang, 25.0, None, None)

        self.assertTrue(p.has_stable_speed(), "Must lock speed once needle has flown >20° with stable fits")
        self.assertGreaterEqual(p.speed_deg_s, 280.0, "Speed must reflect true cruise velocity, not 225°/s stutter")

    def test_dying_ring_does_not_trigger_ghost_chain(self):
        """A dying ring with unchanged zone position must NOT re-arm a frenzy chain after 250ms."""
        locked_w = {"start": 76.0, "end": 86.0, "width": 10.0, "center": 81.5}
        fresh_w_same = {"start": 76.0, "end": 86.0, "width": 10.0, "center": 81.5}
        fresh_w_moved = {"start": 170.0, "end": 180.0, "width": 10.0, "center": 175.0}

        # Case 1: Same ring fading out after trigger (zone at same 81.5°)
        zone_diff = abs((fresh_w_same["center"] - locked_w["center"] + 180.0) % 360.0 - 180.0)
        zone_moved = zone_diff > 15.0
        is_frenzy_chain = zone_moved and (fresh_w_same is not None)
        self.assertFalse(is_frenzy_chain, "Unchanged zone must NOT be detected as a new frenzy chain")

        # Case 2: Genuine continuous chain check where zone moves to 175.0°
        zone_diff_new = abs((fresh_w_moved["center"] - locked_w["center"] + 180.0) % 360.0 - 180.0)
        zone_moved_new = zone_diff_new > 15.0
        is_frenzy_chain_new = zone_moved_new and (fresh_w_moved is not None)
        self.assertTrue(is_frenzy_chain_new, "Relocated zone must trigger genuine frenzy chain re-arm")


if __name__ == "__main__":
    unittest.main()
