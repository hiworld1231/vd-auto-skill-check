import json
import tempfile
import unittest
from pathlib import Path

from core.continuous_predictor import ContinuousAngularPredictor
from core.outcome_observer import OutcomeObserver
from core.post_fire_lifecycle import PostFireLifecycle
from tools.rescore_replays import analyze


FIXTURE = Path(__file__).parent / "fixtures" / "golden_replays.json"


def _circ(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


class GoldenReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_outcome_golden_cases(self):
        for case in self.cases["outcome_cases"]:
            with self.subTest(case=case["name"]):
                o = OutcomeObserver(case["lead_ms"] if "lead_ms" in case else 60.0)
                o.on_trigger(
                    case["trigger_time"],
                    case["target"],
                    case["speed"],
                    case.get("white"),
                    case.get("black"),
                    used_latency_ms=case.get("lead_ms", 60.0),
                )
                for t, angle in case["samples"]:
                    o.observe_sample(t, angle, 30.0)
                result = o.conclude_check()
                self.assertEqual(result["outcome"], case["expected"])
                self.assertTrue(result["plateau_found"])

    def test_lifecycle_golden_cases(self):
        for case in self.cases["lifecycle_cases"]:
            with self.subTest(case=case["name"]):
                sm = PostFireLifecycle()
                sm.begin(case["fire_time"])
                for step in case["steps"]:
                    decision = sm.update(
                        step["t"],
                        ring_present=step["ring_present"],
                        plateau_found=step["plateau_found"],
                        zone_moved=step.get("zone_moved", False),
                        rollback=step.get("rollback", False),
                        fresh_motion=step.get("fresh_motion", False),
                    )
                    self.assertEqual(decision.state, step["expected"])

    def test_predictor_center_golden_cases(self):
        for case in self.cases["predictor_cases"]:
            with self.subTest(case=case["name"]):
                p = ContinuousAngularPredictor(
                    case["lead_ms"],
                    fit_window=10,
                    session_base_speed=278.0,
                )
                p.set_delivery_lead(case["lead_ms"], 0.0)
                white = case["white"]
                dt = case["dt"]
                last_t = 0.0
                last_angle = case["start_angle"]
                for i in range(case["samples"]):
                    last_t = i * dt
                    last_angle = (
                        case["start_angle"] + case["speed"] * last_t
                    ) % 360.0
                    p.update(last_t, last_angle, 30.0, white, None)

                self.assertTrue(p.has_usable_speed())
                pred = p.predict(last_t, last_angle, "GREAT")
                self.assertIsNotNone(pred)
                self.assertAlmostEqual(
                    pred["target_angle"], case["expected_target"], delta=0.01
                )
                self.assertAlmostEqual(
                    pred["speed_deg_s"], case["speed"], delta=max(2.0, case["speed"] * 0.01)
                )

                # Exact synthetic trajectory + nominal delivery lead must land
                # on the selected white-zone center, including wrap-around.
                landing_t = pred["press_timestamp"] + case["lead_ms"] / 1000.0
                landing_angle = (
                    case["start_angle"] + case["speed"] * landing_t
                ) % 360.0
                self.assertLessEqual(
                    _circ(landing_angle, case["expected_target"]), 0.6
                )

    def test_rescore_golden_cases(self):
        for case in self.cases["rescore_cases"]:
            with self.subTest(case=case["name"]):
                with tempfile.TemporaryDirectory() as td:
                    path = Path(td) / "check_fixture.json"
                    path.write_text(
                        json.dumps(case["document"]), encoding="utf-8"
                    )
                    result = analyze([path])
                self.assertEqual(
                    result["impossible_post_fire_jump_count"],
                    1 if case["expected_impossible_jump"] else 0,
                )


if __name__ == "__main__":
    unittest.main()
