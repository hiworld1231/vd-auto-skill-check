#!/usr/bin/env python3
import unittest

from core.continuous_predictor import ContinuousAngularPredictor
from core.continuous_predictor import STATE_LOCKED


class TestContinuousGenRush(unittest.TestCase):
    def _run_speed(self, speed: float):
        p = ContinuousAngularPredictor(
            latency_ms=122.1,
            target_offset_ratio=0.5,
            session_base_speed=278.0,
            fit_window=10,
        )
        w = {"start": 85.0, "end": 95.0, "center": 90.0, "width": 10.0}
        b = {"start": 95.0, "end": 135.0, "center": 115.0, "width": 40.0}
        spawn = 270.0
        # Irregular but plausible unique-frame cadence.
        ts = [0.000, 0.016, 0.033, 0.049, 0.066, 0.083, 0.100, 0.117, 0.134, 0.151]
        lock = None
        for t in ts:
            a = (spawn + speed * t) % 360.0
            p.update(t, a, 80.0, w, b)
            if lock is None and p.has_stable_speed():
                lock = (t, p.speed_deg_s)
        return p, lock

    def test_measures_base_without_base_mode(self):
        p, lock = self._run_speed(278.0)
        self.assertIsNotNone(lock)
        self.assertAlmostEqual(lock[1], 278.0, delta=8.0)
        self.assertFalse(p.mode_switched)

    def test_measures_perk_speeds_directly(self):
        for speed in (350.0, 450.0, 550.0, 900.0):
            with self.subTest(speed=speed):
                p, lock = self._run_speed(speed)
                self.assertIsNotNone(lock)
                self.assertAlmostEqual(lock[1], speed, delta=max(10.0, speed * 0.03))
                self.assertFalse(p.mode_switched)

    def test_prediction_uses_measured_speed(self):
        p, lock = self._run_speed(550.0)
        self.assertIsNotNone(lock)
        t = 0.151
        a = (270.0 + 550.0 * t) % 360.0
        pred = p.predict(t, a, target="GREAT")
        self.assertIsNotNone(pred)
        self.assertAlmostEqual(pred["speed_deg_s"], 550.0, delta=15.0)
        expected_dist = (90.0 - a) % 360.0
        self.assertAlmostEqual(pred["time_to_hit_ms"], expected_dist / 550.0 * 1000.0, delta=4.0)

    def test_wraparound(self):
        p = ContinuousAngularPredictor(100.0, session_base_speed=278.0)
        w = {"start": 15.0, "end": 25.0, "center": 20.0, "width": 10.0}
        samples = [(0.000, 350.0), (0.016, 354.8), (0.032, 359.6), (0.048, 4.4), (0.064, 9.2)]
        for t, a in samples:
            p.update(t, a, 80.0, w, None)
        self.assertTrue(p.has_stable_speed())
        self.assertAlmostEqual(p.speed_deg_s, 300.0, delta=12.0)


    def test_no_mode_switch_for_fast_check(self):
        p, lock = self._run_speed(550.0)
        self.assertIsNotNone(lock)
        self.assertFalse(p.mode_switched)
        self.assertAlmostEqual(p.speed_deg_s, 550.0, delta=15.0)
        telem = p.get_shadow_telemetry()
        self.assertGreaterEqual(telem["fit_sample_count"], 5)

    def test_target_more_than_180_ahead_is_not_marked_passed(self):
        p = ContinuousAngularPredictor(60.0, session_base_speed=278.0)
        w = {"start": 75.0, "end": 85.0, "center": 80.0, "width": 10.0}
        b = {"start": 85.0, "end": 130.0, "center": 107.5, "width": 45.0}
        # Start at 230°.  Target 80° is 210° ahead clockwise, not "behind".
        for t, a in [(0.000,230.0),(0.020,236.0),(0.040,242.0),(0.060,248.0),(0.080,254.0)]:
            p.update(t,a,80.0,w,b)
        self.assertTrue(p.has_stable_speed())
        pred=p.predict(0.080,254.0,target="GREAT")
        self.assertIsNotNone(pred)
        self.assertFalse(pred["reactive_safe_fallback"])
        self.assertGreater(pred["angular_distance_deg"],180.0)
        self.assertFalse(pred["should_press_now"])

    def test_passed_center_never_schedules_full_revolution(self):
        p = ContinuousAngularPredictor(60.0, session_base_speed=278.0)
        w = {"start": 75.0, "end": 85.0, "center": 80.0, "width": 10.0}
        b = {"start": 85.0, "end": 130.0, "center": 107.5, "width": 45.0}
        # Cross 80° after wrapping from a 350° start.
        for t,a in [(0.000,350.0),(0.020,358.0),(0.040,6.0),(0.060,14.0),(0.080,22.0),
                    (0.100,30.0),(0.120,38.0),(0.140,46.0),(0.160,54.0),(0.180,62.0),
                    (0.200,70.0),(0.220,78.0),(0.240,86.0)]:
            p.update(t,a,80.0,w,b)
        pred=p.predict(0.240,86.0,target="GREAT")
        self.assertIsNotNone(pred)
        self.assertEqual(pred["angular_distance_deg"],0.0)
        self.assertLessEqual(pred["time_to_hit_ms"],0.001)

    def test_replay5_unstable_1k_track_uses_conservative_local_speed(self):
        # Live replay(5), 2026-09-19 15:20:07.  The old urgent path used the
        # unstable 1030.7 deg/s all-pairs fit and landed ~13 deg before GREAT.
        p = ContinuousAngularPredictor(81.92233377980533, session_base_speed=278.0)
        w = {"start": 91.0, "end": 101.0, "center": 96.0, "width": 10.0}
        b = {"start": 101.0, "end": 141.0, "center": 121.0, "width": 40.0}
        samples = [
            (0.0000, 276.6385542314031),
            (0.0091, 276.77),
            (0.0166, 291.50),
            (0.0334, 306.69),
            (0.0506, 321.23),
            (0.0667, 336.47),
            (0.0831, 358.01),
            (0.0995, 14.54),
        ]
        for t, a in samples:
            p.update(t, a, 79.0, w, b)

        self.assertTrue(p.has_usable_speed())
        raw = p.speed_deg_s
        used = p.get_actuation_speed()
        telem = p.get_shadow_telemetry()
        self.assertGreater(raw, 1000.0)
        self.assertLess(used, raw)
        self.assertGreater(used, 880.0)
        self.assertLess(used, 980.0)
        self.assertEqual(telem["actuation_speed_reason"], "HIGH_SPEED_CONSERVATIVE_LOCAL")
        pred = p.predict(samples[-1][0], samples[-1][1], target="GREAT")
        self.assertIsNotNone(pred)
        self.assertGreater(pred["time_to_hit_ms"], 84.0)
        self.assertLess(pred["time_to_hit_ms"], 90.0)

    def test_replay6_three_point_1k_track_is_provisionally_slowed(self):
        p = ContinuousAngularPredictor(80.255536, session_base_speed=278.0)
        w = {"start": 78.0, "end": 88.0, "center": 83.0, "width": 10.0}
        b = {"start": 89.0, "end": 131.0, "center": 110.0, "width": 42.0}
        for t, a in [(0.0000, 324.4), (0.0120, 337.2), (0.0242, 350.17)]:
            p.update(t, a, 80.0, w, b)
        self.assertTrue(p.has_usable_speed())
        self.assertGreater(p.speed_deg_s, 1000.0)
        used = p.get_actuation_speed()
        self.assertAlmostEqual(used, p.speed_deg_s * 0.85, delta=1.0)
        self.assertEqual(
            p.get_shadow_telemetry()["actuation_speed_reason"],
            "HIGH_SPEED_PROVISIONAL_85PCT",
        )

    def test_uncertainty_shadow_is_tight_on_clean_linear_track(self):
        p, lock = self._run_speed(550.0)
        self.assertIsNotNone(lock)
        telem = p.get_shadow_telemetry()
        self.assertTrue(telem["speed_uncertainty_reliable"])
        self.assertLess(telem["speed_uncertainty_low"], 550.0)
        self.assertGreater(telem["speed_uncertainty_high"], 550.0)
        self.assertLess(telem["speed_uncertainty_high"] - telem["speed_uncertainty_low"], 50.0)

        t = 0.151
        a = (270.0 + 550.0 * t) % 360.0
        pred = p.predict(t, a, target="GREAT")
        self.assertIsNotNone(pred)
        self.assertGreaterEqual(pred["crossing_uncertainty_ms"], 0.0)
        self.assertGreater(pred["white_window_ms"], 0.0)

    def test_uncertainty_shadow_expands_for_disagreeing_track(self):
        p = ContinuousAngularPredictor(80.0, session_base_speed=278.0)
        w = {"start": 90.0, "end": 100.0, "center": 95.0, "width": 10.0}
        for t, a in [
            (0.000, 270.0),
            (0.016, 278.0),
            (0.033, 291.0),
            (0.049, 299.0),
            (0.066, 315.0),
            (0.083, 324.0),
        ]:
            p.update(t, a, 80.0, w, None)
        telem = p.get_shadow_telemetry()
        self.assertIsNotNone(telem["slope_mad_deg_s"])
        self.assertGreater(
            telem["speed_uncertainty_high"] - telem["speed_uncertainty_low"],
            0.06 * telem["raw_fit_speed"],
        )

    def test_committed_does_not_bypass_new_fit_instability(self):
        p, lock = self._run_speed(450.0)
        self.assertIsNotNone(lock)
        self.assertTrue(p.has_stable_speed())
        p.mark_committed()
        self.assertEqual(p.fire_state, "ARMED")

        # Simulate newer fits disagreeing after a deadline was already armed.
        # Old CLEAN V5 returned True solely because state==COMMITTED.
        p.speed_fits.clear()
        p.speed_fits.extend([(1.0, 430.0), (1.1, 520.0), (1.2, 610.0)])
        self.assertFalse(p.has_stable_speed())

    def test_short_check_never_substitutes_base_prior(self):
        p = ContinuousAngularPredictor(100.0, session_base_speed=278.0)
        w = {"start": 300.0, "end": 310.0, "center": 305.0, "width": 10.0}
        p.update(0.000, 270.0, 80.0, w, None)
        p.update(0.012, 276.6, 80.0, w, None)
        p.update(0.024, 283.2, 80.0, w, None)
        self.assertTrue(p.has_usable_speed())
        self.assertAlmostEqual(p.speed_deg_s, 550.0, delta=15.0)
        self.assertNotAlmostEqual(p.speed_deg_s, 278.0, delta=30.0)


if __name__ == "__main__":
    unittest.main()
