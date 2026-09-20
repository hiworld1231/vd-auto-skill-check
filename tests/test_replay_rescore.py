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


if __name__ == "__main__":
    unittest.main()
