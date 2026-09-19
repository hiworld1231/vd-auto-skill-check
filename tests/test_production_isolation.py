#!/usr/bin/env python3
"""
Regression test: Production File Immutability and Test Isolation.
Guarantees that test suites and simulated checks NEVER mutate production
speed_profiles.json, config.json, or any replay files by even 1 byte.
"""

import hashlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.learner import AdaptiveLatencyLearner, SpeedProfileManager, PROFILES_PATH, CONFIG_PATH, REPLAYS_DIR


def get_file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


class TestProductionIsolation(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.tmp_profiles = self.tmp_dir / "speed_profiles.json"
        self.tmp_config = self.tmp_dir / "config.json"

        # Snapshot production files sha256 and mtimes
        self.prod_profiles_sha = get_file_sha256(PROFILES_PATH)
        self.prod_config_sha = get_file_sha256(CONFIG_PATH)
        self.prod_replays_sha = {
            p.name: get_file_sha256(p) for p in REPLAYS_DIR.glob("check_*.json")
        }

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        # Ensure production files were not mutated
        curr_profiles_sha = get_file_sha256(PROFILES_PATH)
        curr_config_sha = get_file_sha256(CONFIG_PATH)
        self.assertEqual(
            self.prod_profiles_sha,
            curr_profiles_sha,
            "CRITICAL VIOLATION: Production speed_profiles.json was mutated by a test!",
        )
        self.assertEqual(
            self.prod_config_sha,
            curr_config_sha,
            "CRITICAL VIOLATION: Production config.json was mutated by a test!",
        )
        for p in REPLAYS_DIR.glob("check_*.json"):
            self.assertEqual(
                self.prod_replays_sha.get(p.name),
                get_file_sha256(p),
                f"CRITICAL VIOLATION: Production replay {p.name} was mutated by a test!",
            )

    def test_default_learner_never_touches_production_profiles(self):
        """AdaptiveLatencyLearner without use_speed_profiles must never write to disk."""
        learner = AdaptiveLatencyLearner(
            initial_latency_ms=100.0,
            auto_save=False,
            use_speed_profiles=False,
        )
        for i in range(25):
            learner.on_trigger(
                press_time=100.0 + i,
                target_angle=120.0,
                speed_deg_s=275.0,
                white_zone={"start": 115.0, "end": 125.0, "width": 10.0, "center": 120.0},
                black_zone={"start": 125.0, "end": 165.0, "width": 40.0, "center": 145.0},
            )
            # Add freeze plateau observations
            for k in range(5):
                learner.observe_sample(100.1 + i + k * 0.008, 120.0, 25.0)
            res = learner.conclude_check()
            self.assertIsNotNone(res)

    def test_speed_profiles_manager_with_save_disabled(self):
        """SpeedProfileManager with save_to_disk=False must never write to path."""
        fake_path = self.tmp_dir / "must_not_exist.json"
        mgr = SpeedProfileManager(profiles_path=fake_path, auto_bootstrap=False, save_to_disk=False)
        res = mgr.record_hit(
            speed=275.0,
            actual_delay_ms=120.0,
            error_deg=0.0,
            outcome="GREAT",
            plateau_found=True,
            target_angle=90.0,
        )
        self.assertIsNotNone(res)
        self.assertFalse(fake_path.exists(), "File must NOT be written when save_to_disk=False")

    def test_learner_isolated_with_temp_paths(self):
        """Learner configured with temp paths only writes to those temp paths."""
        self.tmp_config.write_text('{"latency_ms": 110.0}', encoding="utf-8")
        learner = AdaptiveLatencyLearner(
            initial_latency_ms=110.0,
            auto_save=True,
            use_speed_profiles=True,
            profiles_path=self.tmp_profiles,
            config_path=self.tmp_config,
        )
        for i in range(10):
            learner.on_trigger(
                press_time=200.0 + i,
                target_angle=150.0,
                speed_deg_s=275.0,
                white_zone={"start": 145.0, "end": 155.0, "width": 10.0, "center": 150.0},
                black_zone={"start": 155.0, "end": 195.0, "width": 40.0, "center": 175.0},
                selected_speed_tier=275,
                speed_at_lock=275.0,
            )
            for k in range(5):
                learner.observe_sample(200.1 + i + k * 0.008, 150.0, 25.0)
            learner.conclude_check()

        self.assertTrue(self.tmp_profiles.exists(), "Temp profiles file should be written")
        self.assertIn("tiers", self.tmp_profiles.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
