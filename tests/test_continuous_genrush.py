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

    def test_replay5_unstable_1k_track_has_no_magic_speed_slowdown(self):
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
        self.assertAlmostEqual(used, raw, delta=1.0)
        self.assertEqual(telem["actuation_speed_reason"], "UNCERTAINTY_MIDPOINT")
        pred = p.predict(samples[-1][0], samples[-1][1], target="GREAT")
        self.assertIsNotNone(pred)
        self.assertGreater(pred["landing_uncertainty_width_deg"], 0.0)
        self.assertIn("great_interval_safe", pred)

    def test_replay6_three_point_1k_track_keeps_unbiased_speed(self):
        p = ContinuousAngularPredictor(80.255536, session_base_speed=278.0)
        w = {"start": 78.0, "end": 88.0, "center": 83.0, "width": 10.0}
        b = {"start": 89.0, "end": 131.0, "center": 110.0, "width": 42.0}
        for t, a in [(0.0000, 324.4), (0.0120, 337.2), (0.0242, 350.17)]:
            p.update(t, a, 80.0, w, b)
        self.assertTrue(p.has_usable_speed())
        self.assertGreater(p.speed_deg_s, 1000.0)
        used = p.get_actuation_speed()
        self.assertAlmostEqual(used, p.speed_deg_s, delta=1.0)
        self.assertEqual(
            p.get_shadow_telemetry()["actuation_speed_reason"],
            "UNCERTAINTY_MIDPOINT",
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
        self.assertTrue(pred["great_interval_safe"])
        self.assertLess(pred["landing_uncertainty_width_deg"], pred["great_width_deg"])

    def test_lead_uncertainty_expands_great_landing_envelope(self):
        p, lock = self._run_speed(550.0)
        self.assertIsNotNone(lock)
        t = 0.151
        a = (270.0 + 550.0 * t) % 360.0

        p.set_delivery_lead(122.1, 0.0)
        tight = p.predict(t, a, target="GREAT")
        self.assertIsNotNone(tight)

        p.set_delivery_lead(122.1, 4.0)
        wide = p.predict(t, a, target="GREAT")
        self.assertIsNotNone(wide)
        self.assertAlmostEqual(wide["lead_uncertainty_ms"], 4.0)
        self.assertGreater(
            wide["landing_uncertainty_width_deg"],
            tight["landing_uncertainty_width_deg"],
        )

    def test_dispatch_uncertainty_expands_total_delivery_envelope(self):
        p, lock = self._run_speed(550.0)
        self.assertIsNotNone(lock)
        t = 0.151
        a = (270.0 + 550.0 * t) % 360.0

        p.set_delivery_lead(122.1, 2.0, 0.0)
        lead_only = p.predict(t, a, target="GREAT")
        self.assertIsNotNone(lead_only)

        p.set_delivery_lead(122.1, 2.0, 1.5)
        combined = p.predict(t, a, target="GREAT")
        self.assertIsNotNone(combined)
        self.assertAlmostEqual(combined["lead_uncertainty_ms"], 2.0)
        self.assertAlmostEqual(combined["dispatch_uncertainty_ms"], 1.5)
        self.assertAlmostEqual(combined["delivery_uncertainty_ms"], 3.5)
        self.assertGreater(
            combined["landing_uncertainty_width_deg"],
            lead_only["landing_uncertainty_width_deg"],
        )

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

    def test_live_chain8_handoff_does_not_wait_a_full_revolution(self):
        # Current-build replay check_20260921_165550_0024:
        # the first chain8 handoff frame appears 10.4° after GREAT centre, but
        # the trailing GOOD sector is still reachable with the 348°/s prior.
        p = ContinuousAngularPredictor(
            80.0,
            session_base_speed=278.0,
        )
        w = {
            "start": 95.53,
            "end": 106.30,
            "center": 100.91,
            "width": 10.77,
            "source": "MEASURED",
        }
        b = {
            "start": 112.98,
            "end": 149.18,
            "center": 131.08,
            "width": 36.20,
            "source": "MEASURED",
        }
        p.reset(
            keep_speed=True,
            default_speed=348.0,
            is_chain=True,
            session_base_speed=278.0,
        )
        p.update(0.0, 111.31, 80.0, w, b)
        pred = p.predict(0.0, 111.31, target="GREAT")
        self.assertIsNotNone(pred)
        self.assertTrue(pred["frenzy_handoff_success_tail"])
        self.assertTrue(pred["target_passed"])
        self.assertLessEqual(pred["time_to_hit_ms"], 0.001)
        self.assertTrue(pred["reactive_safe_fallback"])
        self.assertEqual(pred["speed_source"], "FRENZY_PRIOR")

    def test_frenzy_passed_handoff_never_retargets_next_revolution(self):
        # Regression for the chain4 failure seen after the handoff-tail patch:
        # frame 1 proves this generation's GREAT centre is already behind us,
        # but 80ms delivery is too long to salvage the remaining GOOD tail.
        # Frame 2 must remain target_passed; it must not reinterpret GREAT as
        # target+360 and schedule a bogus one-lap GREAT_CENTER_BEST_EFFORT.
        p = ContinuousAngularPredictor(
            80.0,
            session_base_speed=278.0,
        )
        w = {
            "start": 95.0,
            "end": 105.0,
            "center": 100.0,
            "width": 10.0,
            "source": "MEASURED",
        }
        b = {
            "start": 105.0,
            "end": 150.0,
            "center": 127.5,
            "width": 45.0,
            "source": "MEASURED",
        }
        p.reset(
            keep_speed=True,
            default_speed=900.0,
            is_chain=True,
            session_base_speed=278.0,
        )

        p.update(0.000, 111.0, 80.0, w, b)
        first = p.predict(0.000, 111.0, target="GREAT")
        self.assertIsNotNone(first)
        self.assertTrue(first["frenzy_handoff_success_tail"])
        self.assertTrue(first["frenzy_target_occurrence_latched"])
        self.assertTrue(first["target_passed"])
        self.assertFalse(first["reactive_safe_fallback"])
        self.assertFalse(first["should_press_now"])

        p.update(0.016, 125.0, 80.0, w, b)
        second = p.predict(0.016, 125.0, target="GREAT")
        self.assertIsNotNone(second)
        self.assertTrue(second["frenzy_target_occurrence_latched"])
        self.assertTrue(second["target_passed"])
        self.assertEqual(second["angular_distance_deg"], 0.0)
        self.assertLessEqual(second["time_until_press_ms"], 0.001)
        self.assertFalse(second["should_press_now"])

    def test_deep_frenzy_sample6_stays_blended_with_prior(self):
        # Current-build chain15 jumped from a 441.8°/s prior to an 820°/s raw
        # fit on sample 6. That boundary must remain blended instead of becoming
        # fully trusted in one frame.
        p = ContinuousAngularPredictor(
            80.0,
            session_base_speed=278.0,
        )
        w = {
            "start": 355.6,
            "end": 6.4,
            "center": 1.0,
            "width": 10.8,
            "source": "MEASURED",
        }
        b = {
            "start": 7.0,
            "end": 50.2,
            "center": 28.6,
            "width": 43.2,
            "source": "MEASURED",
        }
        p.reset(
            keep_speed=True,
            default_speed=441.8,
            is_chain=True,
            session_base_speed=278.0,
        )
        for i in range(6):
            t = i * 0.016
            p.update(t, (250.0 + 820.0 * t) % 360.0, 80.0, w, b)

        self.assertEqual(p._fit_sample_count, 6)
        self.assertTrue(p.has_usable_speed())
        pred = p.predict(0.080, (250.0 + 820.0 * 0.080) % 360.0, target="GREAT")
        self.assertIsNotNone(pred)
        self.assertEqual(pred["speed_source"], "FRENZY_BLEND")
        self.assertLess(pred["speed_deg_s"], 720.0)
        self.assertGreater(pred["speed_deg_s"], 600.0)
        self.assertLessEqual(
            pred["landing_uncertainty_width_deg"],
            pred["success_width_deg"],
        )

    def test_frenzy_80ms_lead_keeps_1000dps_success_window_open(self):
        w = {
            "start": 95.0, "end": 105.0, "center": 100.0,
            "width": 10.0, "source": "MEASURED",
        }
        b = {
            "start": 105.0, "end": 145.0, "center": 125.0,
            "width": 40.0, "source": "MEASURED",
        }

        late = ContinuousAngularPredictor(126.9, session_base_speed=278.0)
        late.reset(
            keep_speed=True,
            default_speed=1000.0,
            is_chain=True,
            session_base_speed=278.0,
        )
        late.update(0.0, 20.0, 80.0, w, b)
        late_pred = late.predict(0.0, 20.0, target="GREAT")
        self.assertIsNotNone(late_pred)
        self.assertFalse(late_pred["should_press_now"])

        frenzy = ContinuousAngularPredictor(80.0, session_base_speed=278.0)
        frenzy.reset(
            keep_speed=True,
            default_speed=1000.0,
            is_chain=True,
            session_base_speed=278.0,
        )
        frenzy.update(0.0, 20.0, 80.0, w, b)
        frenzy_pred = frenzy.predict(0.0, 20.0, target="GREAT")
        self.assertIsNotNone(frenzy_pred)
        self.assertTrue(frenzy_pred["should_press_now"])

    def test_frenzy_ignores_first_doubled_segment_and_holds_prior(self):
        p = ContinuousAngularPredictor(126.9, session_base_speed=278.0)
        w = {
            "start": 210.0, "end": 220.0, "center": 215.0,
            "width": 10.0, "source": "MEASURED",
        }
        p.reset(
            keep_speed=True,
            default_speed=324.9,
            is_chain=True,
            session_base_speed=278.0,
        )
        p.update(0.0000, 10.0, 80.0, w, None)
        p.update(0.0085, 22.4, 80.0, w, None)  # ~1459 deg/s first segment
        pred = p.predict(0.0085, 22.4, target="GREAT")
        self.assertIsNotNone(pred)
        self.assertEqual(pred["speed_source"], "FRENZY_PRIOR")
        self.assertAlmostEqual(pred["speed_deg_s"], 324.9, delta=0.1)
        self.assertFalse(p.has_usable_speed())

    def test_real_frenzy_chain2_three_point_fit_blends_to_true_speed(self):
        # Real replay 2026-09-19 17:46:04 chain=2.
        # First adjacent segment looked ~548 deg/s while the robust fire speed
        # was ~309 deg/s. Previous generation was ~264.4 deg/s.
        p = ContinuousAngularPredictor(126.9, session_base_speed=278.0)
        w = {
            "start": 250.0, "end": 260.0, "center": 255.0,
            "width": 10.0, "source": "MEASURED",
        }
        p.reset(
            keep_speed=True,
            default_speed=264.4338,
            is_chain=True,
            session_base_speed=278.0,
        )
        for t, a in [
            (0.0000, 161.83),
            (0.0066, 165.46),
            (0.0233, 170.86),
        ]:
            p.update(t, a, 80.0, w, None)

        self.assertTrue(p.has_usable_speed())
        pred = p.predict(0.0233, 170.86, target="GREAT")
        self.assertIsNotNone(pred)
        self.assertEqual(pred["speed_source"], "FRENZY_BLEND")
        self.assertEqual(pred["fit_sample_count"], 3)
        self.assertAlmostEqual(pred["speed_deg_s"], 308.0, delta=12.0)

    def test_real_frenzy_chain5_short_fit_stays_near_generation_speed(self):
        # Real replay chain=5: previous generation ~455.1, true chain speed
        # ~533.9, but the first adjacent segment looked >1000 deg/s.
        p = ContinuousAngularPredictor(126.9, session_base_speed=278.0)
        w = {
            "start": 300.0, "end": 310.0, "center": 305.0,
            "width": 10.0, "source": "MEASURED",
        }
        p.reset(
            keep_speed=True,
            default_speed=455.0789,
            is_chain=True,
            session_base_speed=278.0,
        )
        for t, a in [
            (0.0000, 191.43),
            (0.0085, 200.50),
            (0.0247, 209.64),
        ]:
            p.update(t, a, 80.0, w, None)

        self.assertTrue(p.has_usable_speed())
        pred = p.predict(0.0247, 209.64, target="GREAT")
        self.assertIsNotNone(pred)
        self.assertEqual(pred["speed_source"], "FRENZY_BLEND")
        self.assertGreater(pred["speed_deg_s"], 520.0)
        self.assertLess(pred["speed_deg_s"], 575.0)

    def test_first_frame_can_predict_from_session_prior(self):
        p = ContinuousAngularPredictor(
            138.9,
            session_base_speed=278.0,
        )
        w = {
            "start": 85.0,
            "end": 95.0,
            "center": 90.0,
            "width": 10.0,
            "source": "MEASURED",
        }
        b = {
            "start": 95.0,
            "end": 135.0,
            "center": 115.0,
            "width": 40.0,
            "source": "MEASURED",
        }
        p.update(0.0, 0.0, 80.0, w, b)
        pred = p.predict(0.0, 0.0, target="GREAT")
        self.assertIsNotNone(pred)
        self.assertEqual(pred["speed_source"], "SESSION_PRIOR")
        self.assertAlmostEqual(pred["speed_deg_s"], 278.0, delta=0.1)
        self.assertGreater(pred["time_until_press_ms"], 150.0)

    def test_second_unique_frame_replaces_prior_with_segment_speed(self):
        p = ContinuousAngularPredictor(
            138.9,
            session_base_speed=278.0,
        )
        w = {
            "start": 85.0,
            "end": 95.0,
            "center": 90.0,
            "width": 10.0,
            "source": "MEASURED",
        }
        b = {
            "start": 95.0,
            "end": 135.0,
            "center": 115.0,
            "width": 40.0,
            "source": "MEASURED",
        }
        p.update(0.000, 0.0, 80.0, w, b)
        p.update(0.016, 5.6, 80.0, w, b)
        pred = p.predict(0.016, 5.6, target="GREAT")
        self.assertIsNotNone(pred)
        self.assertEqual(pred["speed_source"], "SEGMENT_PROVISIONAL")
        self.assertAlmostEqual(pred["speed_deg_s"], 350.0, delta=3.0)

    def test_prior_can_immediately_salvage_success_when_center_deadline_passed(self):
        p = ContinuousAngularPredictor(
            138.9,
            session_base_speed=278.0,
        )
        w = {
            "start": 85.0,
            "end": 95.0,
            "center": 90.0,
            "width": 10.0,
            "source": "MEASURED",
        }
        b = {
            "start": 95.0,
            "end": 135.0,
            "center": 115.0,
            "width": 40.0,
            "source": "MEASURED",
        }
        p.update(0.0, 60.0, 80.0, w, b)
        pred = p.predict(0.0, 60.0, target="GREAT")
        self.assertIsNotNone(pred)
        self.assertLessEqual(pred["time_until_press_ms"], 0.0)
        self.assertTrue(pred["should_press_now"])

    def test_short_check_never_substitutes_base_prior(self):
        p = ContinuousAngularPredictor(100.0, session_base_speed=278.0)
        w = {"start": 300.0, "end": 310.0, "center": 305.0, "width": 10.0}
        p.update(0.000, 270.0, 80.0, w, None)
        p.update(0.012, 276.6, 80.0, w, None)
        p.update(0.024, 283.2, 80.0, w, None)
        self.assertTrue(p.has_usable_speed())
        self.assertAlmostEqual(p.speed_deg_s, 550.0, delta=15.0)
        self.assertNotAlmostEqual(p.speed_deg_s, 278.0, delta=30.0)


    def test_robust_phase_ignores_single_forward_angle_jump(self):
        p = ContinuousAngularPredictor(80.0, session_base_speed=278.0)
        w = {
            "start": 95.0,
            "end": 105.0,
            "center": 100.0,
            "width": 10.0,
            "source": "MEASURED",
        }
        b = {
            "start": 105.0,
            "end": 145.0,
            "center": 125.0,
            "width": 40.0,
            "source": "MEASURED",
        }

        # Clean motion is 500 deg/s and should be at 60 deg on the final frame.
        # The detector instead reports one accepted +10 deg forward jump.  The
        # robust slope is still exactly 500 deg/s; prediction phase must come
        # from that same fitted trajectory, not from the raw 70 deg sample.
        for t, a in [
            (0.000, 20.0),
            (0.016, 28.0),
            (0.032, 36.0),
            (0.048, 44.0),
            (0.064, 52.0),
            (0.080, 70.0),
        ]:
            p.update(t, a, 80.0, w, b)

        self.assertTrue(p.has_usable_speed())
        self.assertAlmostEqual(p.speed_deg_s, 500.0, delta=1.0)

        pred = p.predict(0.080, 70.0, target="GREAT")
        self.assertIsNotNone(pred)
        self.assertEqual(pred["phase_source"], "ROBUST_FIT")
        self.assertAlmostEqual(pred["raw_phase_angle_deg"], 70.0, delta=0.1)
        self.assertAlmostEqual(pred["phase_angle_deg"], 60.0, delta=0.5)
        self.assertAlmostEqual(pred["phase_correction_deg"], -10.0, delta=0.5)
        self.assertAlmostEqual(pred["time_to_hit_ms"], 80.0, delta=2.0)



if __name__ == "__main__":
    unittest.main()
