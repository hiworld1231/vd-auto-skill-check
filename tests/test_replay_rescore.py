import json
import tempfile
import unittest
from pathlib import Path

from tools.rescore_replays import analyze


class ReplayRescoreTests(unittest.TestCase):
    def test_detects_false_frenzy_plateau_and_impossible_jump(self):
        doc = {
            "check_id": "synthetic",
            "chain_count": 1,
            "trigger_event": {"trigger_time": 1.0, "speed_deg_s": 300.0},
            "frames": [
                {"t": 1.01, "needle_angle": 3.0},
                {"t": 1.03, "needle_angle": 8.0},
                {"t": 1.04, "needle_angle": 292.0},
                {"t": 1.06, "needle_angle": 292.1},
                {"t": 1.08, "needle_angle": 291.9},
                {"t": 1.10, "needle_angle": 292.0},
            ],
            "outcome_info": {
                "outcome": "FRENZY_TRANSITION",
                "speed_at_fire": 300.0,
            },
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "check_synthetic.json"
            p.write_text(json.dumps(doc), encoding="utf-8")
            result = analyze([p])
        self.assertEqual(result["impossible_post_fire_jump_count"], 1)
        self.assertEqual(result["false_frenzy_plateau_count"], 1)

    def test_trusted_ideal_lead_summary(self):
        doc = {
            "check_id": "clean",
            "chain_count": 1,
            "trigger_event": {"trigger_time": 1.0, "speed_deg_s": 300.0},
            "frames": [],
            "outcome_info": {
                "outcome": "GREAT",
                "plateau_found": True,
                "detector_fallback": False,
                "white_source": "MEASURED",
                "scheduler_jitter_ms": 0.1,
                "effective_dispatch_lead_ms": 90.0,
                "center_error_ms": 5.0,
                "landing_uncertainty_width_deg": 3.0,
                "delivery_uncertainty_ms": 2.5,
                "fire_policy_reason": "GREAT_INTERVAL_SAFE",
                "fire_policy_best_effort": False,
                "white_source_at_fire": "MEASURED",
                "fit_telemetry": {
                    "fit_sample_count": 8,
                    "fit_residual_mad_deg": 0.5,
                    "live_fit_spread": 10.0,
                    "short_vs_long_delta": 2.0,
                },
            },
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "check_clean.json"
            p.write_text(json.dumps(doc), encoding="utf-8")
            result = analyze([p])
        self.assertEqual(result["trusted_ideal_lead_samples"], 1)
        self.assertAlmostEqual(result["trusted_ideal_lead_median_ms"], 95.0)
        val = result["validation"]
        self.assertEqual(val["confirmed_fired_checks"], 1)
        self.assertAlmostEqual(val["great_rate_confirmed"], 1.0)
        self.assertAlmostEqual(val["center_error_ms"]["median"], 5.0)
        self.assertAlmostEqual(
            val["physical_keydown_residual_ms"]["median"], 0.1
        )
        self.assertAlmostEqual(
            val["landing_uncertainty_width_deg"]["median"], 3.0
        )
        self.assertAlmostEqual(
            val["delivery_uncertainty_ms"]["median"], 2.5
        )
        self.assertEqual(
            val["fire_policy_reasons"], {"GREAT_INTERVAL_SAFE": 1}
        )
        self.assertEqual(val["geometry_sources"], {"MEASURED": 1})


    def test_validation_counts_best_effort_and_no_fire_reasons(self):
        docs = [
            {
                "check_id": "best_effort",
                "chain_count": 1,
                "frames": [],
                "outcome_info": {
                    "outcome": "GOOD",
                    "center_error_ms": 8.0,
                    "scheduler_jitter_ms": -0.4,
                    "landing_uncertainty_width_deg": 7.0,
                    "delivery_uncertainty_ms": 3.5,
                    "fire_policy_reason": "RECONSTRUCTED_GREAT_STABLE_FIT",
                    "fire_policy_best_effort": True,
                    "white_source_at_fire": "RECONSTRUCTED_FROM_BLACK",
                },
            },
            {
                "check_id": "no_fire",
                "chain_count": 1,
                "frames": [],
                "outcome_info": {
                    "outcome": "NO_FIRE",
                    "no_fire_reason": "GREAT_GEOMETRY_UNTRUSTED",
                },
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for i, doc in enumerate(docs):
                p = Path(td) / f"check_{i}.json"
                p.write_text(json.dumps(doc), encoding="utf-8")
                paths.append(p)
            result = analyze(paths)

        val = result["validation"]
        self.assertEqual(val["confirmed_fired_checks"], 1)
        self.assertEqual(val["best_effort_fires"], 1)
        self.assertEqual(val["best_effort_outcomes"], {"GOOD": 1})
        self.assertEqual(
            val["geometry_sources"], {"RECONSTRUCTED_FROM_BLACK": 1}
        )
        self.assertEqual(
            val["no_fire_reasons"], {"GREAT_GEOMETRY_UNTRUSTED": 1}
        )


if __name__ == "__main__":
    unittest.main()
