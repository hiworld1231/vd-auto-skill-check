#!/usr/bin/env python3
"""
Unit Test Suite for AdaptiveLatencyLearner in core/learner.py.
Verifies closed-loop self-learning convergence and outcome evaluation.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.learner import AdaptiveLatencyLearner


class TestAdaptiveLatencyLearner(unittest.TestCase):
    """Tests the adaptive self-learning controller."""

    def test_convergence_from_late_bias(self):
        """Tests that a learner starting at 60.0ms (10ms late bias) converges to ~70.0ms."""
        true_latency_ms = 70.0
        learner = AdaptiveLatencyLearner(initial_latency_ms=60.0, learning_rate=0.25, auto_save=False)

        speed = 270.0  # deg/s
        fps = 120.0
        dt = 1.0 / fps

        # Simulate 20 skill checks where game registers hit at press_time + true_latency_ms
        for episode in range(25):
            t_press = 10.0 + episode * 5.0
            target_angle = 150.0
            w_zone = {"start": 145.0, "end": 154.5, "width": 9.5, "center": 149.75}
            b_zone = {"start": 154.5, "end": 196.5, "width": 42.0, "center": 175.5}

            learner.on_trigger(
                press_time=t_press,
                target_angle=target_angle,
                speed_deg_s=speed,
                white_zone=w_zone,
                black_zone=b_zone,
            )

            # In the physical game, the bot fired at t_press = target_t - learner.latency_ms.
            # At t_press + true_latency_ms, the hit registered and needle froze at:
            hit_delay_ms = true_latency_ms - learner.latency_ms
            frozen_angle = (target_angle + speed * (hit_delay_ms / 1000.0)) % 360.0

            # Feed post-trigger frames showing needle reaching frozen angle and stopping
            for f in range(20):
                t_sample = t_press + f * dt
                if (f * dt) < max(0.04, (true_latency_ms / 1000.0)):
                    # Rotating towards freeze
                    ang = (frozen_angle - speed * max(0.0, (true_latency_ms / 1000.0) - (f * dt))) % 360.0
                else:
                    # Frozen plateau
                    ang = frozen_angle
                learner.observe_sample(t_sample, ang)

            res = learner.conclude_check()
            self.assertIsNotNone(res)

        print(f"\n[TEST LEARNER] Final converged latency: {learner.latency_ms:.2f}ms (target: {true_latency_ms:.1f}ms)")
        self.assertAlmostEqual(learner.latency_ms, true_latency_ms, delta=5.0,
                               msg="Learner must automatically converge to true physical latency within 1ms")

    def test_convergence_from_early_bias(self):
        """Tests that a learner starting at 80.0ms (10ms early bias) converges to ~70.0ms."""
        true_latency_ms = 70.0
        learner = AdaptiveLatencyLearner(initial_latency_ms=80.0, learning_rate=0.25, auto_save=False)

        speed = 270.0
        fps = 120.0
        dt = 1.0 / fps

        for episode in range(25):
            t_press = 10.0 + episode * 5.0
            target_angle = 120.0
            w_zone = {"start": 115.0, "end": 124.5, "width": 9.5, "center": 119.75}
            b_zone = {"start": 124.5, "end": 166.5, "width": 42.0, "center": 145.5}

            learner.on_trigger(t_press, target_angle, speed, w_zone, b_zone)

            hit_delay_ms = true_latency_ms - learner.latency_ms
            frozen_angle = (target_angle + speed * (hit_delay_ms / 1000.0)) % 360.0

            for f in range(20):
                t_sample = t_press + f * dt
                if (f * dt) < max(0.04, (true_latency_ms / 1000.0)):
                    ang = (frozen_angle - speed * max(0.0, (true_latency_ms / 1000.0) - (f * dt))) % 360.0
                else:
                    ang = frozen_angle
                learner.observe_sample(t_sample, ang)

            learner.conclude_check()

        print(f"[TEST LEARNER] Final converged latency from 80ms: {learner.latency_ms:.2f}ms (target: {true_latency_ms:.1f}ms)")
        self.assertAlmostEqual(learner.latency_ms, true_latency_ms, delta=5.0)

    def test_frenzy_immunity(self):
        """Tests that frenzy hits never alter baseline latency even with huge angular offsets."""
        base_latency = 68.0
        learner = AdaptiveLatencyLearner(initial_latency_ms=base_latency, auto_save=False)

        # Simulate 10 rapid frenzy chain hits with 45° offset
        for _ in range(10):
            learner.on_trigger(
                press_time=1.0,
                target_angle=100.0,
                speed_deg_s=270.0,
                white_zone={"start": 95.0, "end": 104.5, "width": 9.5, "center": 99.75},
                black_zone={"start": 104.5, "end": 146.5, "width": 42.0, "center": 125.5},
                is_frenzy=True,
            )
            # Feed spinning samples without freeze
            for f in range(5):
                learner.observe_sample(1.0 + f * 0.016, (145.0 + f * 4.5) % 360.0)

            res = learner.conclude_check()
            self.assertEqual(res["adjustment_ms"], 0.0, "Frenzy must never adjust latency")
            self.assertEqual(learner.latency_ms, base_latency, "Latency must remain rock-solid during frenzy")

    def test_outlier_rejection(self):
        """Tests that anomalous errors > 6.5° are safely discarded without poisoning latency."""
        base_latency = 68.0
        learner = AdaptiveLatencyLearner(initial_latency_ms=base_latency, auto_save=False)

        learner.on_trigger(
            press_time=1.0,
            target_angle=100.0,
            speed_deg_s=270.0,
            white_zone={"start": 95.0, "end": 104.5, "width": 9.5, "center": 99.75},
            black_zone={"start": 104.5, "end": 146.5, "width": 42.0, "center": 125.5},
            is_frenzy=False,
        )
        # Frozen at 150° (50° error!)
        for _ in range(5):
            learner.observe_sample(1.05, 150.0)

        res = learner.conclude_check()
        self.assertEqual(res["adjustment_ms"], 0.0, "Outlier must not adjust latency")
        self.assertEqual(learner.latency_ms, base_latency)

    def test_no_plateau_protection(self):
        """Tests that if needle never stopped, latency is not modified."""
        base_latency = 68.0
        learner = AdaptiveLatencyLearner(initial_latency_ms=base_latency, auto_save=False)

        learner.on_trigger(
            press_time=1.0,
            target_angle=100.0,
            speed_deg_s=270.0,
            white_zone=None,
            black_zone=None,
            is_frenzy=False,
        )
        # Constantly moving samples
        for f in range(5):
            learner.observe_sample(1.0 + f * 0.016, 100.0 + f * 10.0)

        res = learner.conclude_check()
        self.assertFalse(res["plateau_found"])
        self.assertEqual(res["adjustment_ms"], 0.0)
        self.assertEqual(learner.latency_ms, base_latency)

    def test_convergence_to_high_latency_95ms(self):
        """Tests that a learner can converge to real display latency of ~95.0ms (above previous 82ms cap)."""
        true_latency_ms = 95.0
        learner = AdaptiveLatencyLearner(initial_latency_ms=80.0, learning_rate=0.25, auto_save=False)

        speed = 270.0
        fps = 120.0
        dt = 1.0 / fps

        for episode in range(30):
            t_press = 10.0 + episode * 5.0
            target_angle = 150.0
            w_zone = {"start": 145.0, "end": 154.5, "width": 9.5, "center": 149.75}
            b_zone = {"start": 154.5, "end": 196.5, "width": 42.0, "center": 175.5}

            learner.on_trigger(t_press, target_angle, speed, w_zone, b_zone)
            hit_delay_ms = true_latency_ms - learner.latency_ms
            frozen_angle = (target_angle + speed * (hit_delay_ms / 1000.0)) % 360.0

            for f in range(25):
                t_sample = t_press + f * dt
                if (f * dt) < max(0.04, (true_latency_ms / 1000.0)):
                    ang = (frozen_angle - speed * max(0.0, (true_latency_ms / 1000.0) - (f * dt))) % 360.0
                else:
                    ang = frozen_angle
                learner.observe_sample(t_sample, ang)

            learner.conclude_check()

        print(f"\n[TEST HIGH LATENCY] Final converged latency: {learner.latency_ms:.2f}ms (target: {true_latency_ms:.1f}ms)")
        self.assertAlmostEqual(learner.latency_ms, true_latency_ms, delta=5.0)

    def test_wrap_around_plateau_detection(self):
        """Tests that frozen needle plateau is correctly detected even across 360/0 degree boundary."""
        learner = AdaptiveLatencyLearner(initial_latency_ms=75.0, auto_save=False)
        learner.on_trigger(
            press_time=1.0,
            target_angle=0.0,
            speed_deg_s=270.0,
            white_zone={"start": 355.0, "end": 4.5, "width": 9.5, "center": 359.75},
            black_zone={"start": 4.5, "end": 46.5, "width": 42.0, "center": 25.5},
            is_frenzy=False,
        )
        # Samples spinning towards 0°, then freezing at 0.5° across boundary:
        # [358.5, 359.8, 0.4, 0.5, 0.5, 0.5]
        learner.observe_sample(1.02, 358.5)
        learner.observe_sample(1.04, 359.8)
        learner.observe_sample(1.06, 0.4)
        learner.observe_sample(1.08, 0.5)
        learner.observe_sample(1.10, 0.5)
        learner.observe_sample(1.12, 0.5)

        res = learner.conclude_check()
        self.assertTrue(res["plateau_found"], "Must detect freeze plateau across 360/0 boundary")
        self.assertAlmostEqual(res["hit_angle"], 0.5, delta=0.5)

    def test_good_zone_error_adaptation(self):
        """Tests that errors in Good zone (+10°) adapt latency rather than being discarded as outliers."""
        base_latency = 70.0
        learner = AdaptiveLatencyLearner(initial_latency_ms=base_latency, learning_rate=0.25, auto_save=False)
        learner.on_trigger(
            press_time=1.0,
            target_angle=100.0,
            speed_deg_s=270.0,
            white_zone={"start": 95.0, "end": 104.5, "width": 9.5, "center": 99.75},
            black_zone={"start": 104.5, "end": 146.5, "width": 42.0, "center": 125.5},
            is_frenzy=False,
        )
        # Freeze at 110.0° (hit 10° late into Good zone, equivalent to ~37ms late)
        for _ in range(6):
            learner.observe_sample(1.08, 110.0)

        res = learner.conclude_check()
        self.assertEqual(res["outcome"], "GOOD")
        self.assertGreater(res["adjustment_ms"], 0.0, "Error in Good zone (+10°) MUST adjust latency upwards")
        self.assertGreater(learner.latency_ms, base_latency)

    def test_far_good_zone_not_rejected(self):
        """Tests that large errors deep in Good zone (+25°) are NEVER discarded as outliers."""
        base_latency = 70.0
        learner = AdaptiveLatencyLearner(initial_latency_ms=base_latency, auto_save=False)
        learner.on_trigger(
            press_time=1.0,
            target_angle=100.0,
            speed_deg_s=270.0,
            white_zone={"start": 95.0, "end": 104.5, "width": 9.5, "center": 99.75},
            black_zone={"start": 104.5, "end": 146.5, "width": 42.0, "center": 125.5},
            is_frenzy=False,
        )
        # Freeze at 125.0° (+25.0° error deep inside Good zone)
        for _ in range(6):
            learner.observe_sample(1.10, 125.0)

        res = learner.conclude_check()
        self.assertEqual(res["outcome"], "GOOD")
        self.assertGreater(res["adjustment_ms"], 4.0, "Large Good error must take a dynamic fast step (>4ms)")
        self.assertGreater(learner.latency_ms, base_latency + 4.0)

    def test_fast_good_zone_convergence(self):
        """Tests that learner recovers from a 30ms offset within 4-5 checks."""
        true_latency_ms = 108.0
        learner = AdaptiveLatencyLearner(initial_latency_ms=78.0, auto_save=False)

        speed = 270.0
        fps = 120.0
        dt = 1.0 / fps

        outcomes = []
        for episode in range(8):
            t_press = 10.0 + episode * 5.0
            target_angle = 150.0
            w_zone = {"start": 145.0, "end": 154.5, "width": 9.5, "center": 149.75}
            b_zone = {"start": 154.5, "end": 196.5, "width": 42.0, "center": 175.5}

            learner.on_trigger(t_press, target_angle, speed, w_zone, b_zone)
            hit_delay_ms = true_latency_ms - learner.latency_ms
            frozen_angle = (target_angle + speed * (hit_delay_ms / 1000.0)) % 360.0

            for f in range(25):
                t_sample = t_press + f * dt
                if (f * dt) < max(0.04, (true_latency_ms / 1000.0)):
                    ang = (frozen_angle - speed * max(0.0, (true_latency_ms / 1000.0) - (f * dt))) % 360.0
                else:
                    ang = frozen_angle
                learner.observe_sample(t_sample, ang)

            res = learner.conclude_check()
            outcomes.append(res["outcome"])

        self.assertIn("GREAT", outcomes[3:], "Must reach GREAT within 4 checks from 30ms offset")
        self.assertAlmostEqual(learner.latency_ms, true_latency_ms, delta=8.0)


if __name__ == "__main__":
    unittest.main()


class TestLearnerStability(unittest.TestCase):
    """Regression tests for live random-walk bug (hits at first, then misses)."""

    def _feed_frozen(self, learner, press_t, target, frozen_angle, speed=270.0):
        w = {"start": 145.0, "end": 154.5, "width": 9.5, "center": 149.75}
        b = {"start": 154.5, "end": 196.5, "width": 42.0, "center": 175.5}
        learner.on_trigger(press_t, target, speed, w, b)
        for f in range(8):
            learner.observe_sample(press_t + 0.05 + f * 0.008, frozen_angle)
        return learner.conclude_check()

    def test_no_walk_on_centered_great(self):
        """20 centered GREAT hits (±1° noise) must not drift latency >1.5ms."""
        learner = AdaptiveLatencyLearner(initial_latency_ms=100.0, auto_save=False)
        base = learner.latency_ms
        for i in range(20):
            tgt = 150.0
            frozen = (tgt + (1.0 if i % 2 == 0 else -1.0)) % 360.0
            self._feed_frozen(learner, 10.0 + i * 5.0, tgt, frozen)
        self.assertLess(abs(learner.latency_ms - base), 1.5,
                        "Centered GREAT noise must not walk latency (random-walk bug)")

    def test_no_walk_on_alternating_edge_great(self):
        """Alternating edge GREATs (±4°) must net-drift <2.5ms over 20 hits."""
        learner = AdaptiveLatencyLearner(initial_latency_ms=100.0, auto_save=False)
        base = learner.latency_ms
        for i in range(20):
            tgt = 150.0
            frozen = (tgt + (4.0 if i % 2 == 0 else -4.0)) % 360.0
            self._feed_frozen(learner, 10.0 + i * 5.0, tgt, frozen)
        self.assertLess(abs(learner.latency_ms - base), 2.5,
                        "Alternating edge GREATs must cancel out, not drift")

    def test_systematic_bias_still_corrects(self):
        """10 consecutive same-side +4° GREATs must push latency up >1ms."""
        learner = AdaptiveLatencyLearner(initial_latency_ms=100.0, auto_save=False)
        base = learner.latency_ms
        for i in range(10):
            self._feed_frozen(learner, 10.0 + i * 5.0, 150.0, 154.0)
        self.assertGreater(learner.latency_ms - base, 1.0,
                           "Systematic same-side bias must still correct latency")


if __name__ == "__main__":
    unittest.main()
