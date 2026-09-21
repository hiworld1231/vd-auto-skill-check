import unittest

from core.fire_policy import decide_great_fire


class GreatFirePolicyTests(unittest.TestCase):
    def test_stable_safe_interval_is_allowed(self):
        d = decide_great_fire(
            {
                "time_until_press_ms": 42.0,
                "great_interval_safe": True,
                "great_interval_intersects": True,
                "landing_uncertainty_width_deg": 4.0,
                "great_width_deg": 10.0,
                "white_source": "MEASURED",
            },
            fit_stable=True,
            speed_usable=True,
        )
        self.assertTrue(d.allow)
        self.assertEqual(d.reason, "GREAT_INTERVAL_SAFE")
        self.assertFalse(d.best_effort)

    def test_center_deadline_stays_armed_when_interval_is_uncertain(self):
        d = decide_great_fire(
            {
                "time_until_press_ms": 18.0,
                "great_interval_safe": False,
                "great_interval_intersects": True,
                "landing_uncertainty_width_deg": 18.0,
                "great_width_deg": 10.0,
                "white_source": "MEASURED",
                "speed_source": "MEASURED",
                "target_passed": False,
            },
            fit_stable=False,
            speed_usable=True,
        )
        self.assertTrue(d.allow)
        self.assertTrue(d.best_effort)
        self.assertEqual(d.reason, "GREAT_CENTER_BEST_EFFORT")

    def test_short_track_can_fire_when_full_interval_still_fits_great(self):
        d = decide_great_fire(
            {
                "time_until_press_ms": 20.0,
                "great_interval_safe": True,
                "great_interval_intersects": True,
                "landing_uncertainty_width_deg": 5.0,
                "great_width_deg": 10.0,
                "white_source": "MEASURED",
            },
            fit_stable=False,
            speed_usable=True,
        )
        self.assertTrue(d.allow)

    def test_last_chance_intersection_is_bounded(self):
        d = decide_great_fire(
            {
                "time_until_press_ms": 3.0,
                "great_interval_safe": False,
                "great_interval_intersects": True,
                "landing_uncertainty_width_deg": 10.5,
                "great_width_deg": 10.0,
                "white_source": "MEASURED",
            },
            fit_stable=False,
            speed_usable=True,
        )
        self.assertTrue(d.allow)
        self.assertTrue(d.best_effort)
        self.assertEqual(d.reason, "GREAT_LAST_CHANCE_INTERSECTION")

    def test_last_chance_refuses_broad_uncertainty(self):
        d = decide_great_fire(
            {
                "time_until_press_ms": 0.0,
                "great_interval_safe": False,
                "great_interval_intersects": True,
                "landing_uncertainty_width_deg": 16.0,
                "great_width_deg": 10.0,
                "white_source": "MEASURED",
            },
            fit_stable=True,
            speed_usable=True,
        )
        self.assertFalse(d.allow)
        self.assertEqual(d.reason, "GREAT_INTERVAL_UNSAFE")

    def test_reconstructed_great_safe_envelope_can_fire_on_usable_speed(self):
        pred = {
            "time_until_press_ms": 20.0,
            "great_interval_safe": True,
            "great_interval_intersects": True,
            "landing_uncertainty_width_deg": 4.0,
            "great_width_deg": 9.5,
            "white_source": "RECONSTRUCTED_FROM_BLACK",
        }
        d = decide_great_fire(
            pred,
            fit_stable=False,
            speed_usable=True,
        )
        self.assertTrue(d.allow)
        self.assertTrue(d.best_effort)
        self.assertEqual(d.reason, "RECONSTRUCTED_GREAT_SAFE_ENVELOPE")

    def test_reconstructed_great_keeps_center_best_effort_before_deadline(self):
        d = decide_great_fire(
            {
                "time_until_press_ms": 2.0,
                "great_interval_safe": False,
                "great_interval_intersects": True,
                "landing_uncertainty_width_deg": 9.0,
                "great_width_deg": 9.5,
                "white_source": "RECONSTRUCTED_FROM_BLACK",
                "speed_source": "MEASURED",
                "target_passed": False,
            },
            fit_stable=True,
            speed_usable=True,
        )
        self.assertTrue(d.allow)
        self.assertEqual(d.reason, "GREAT_CENTER_BEST_EFFORT")

    def test_frenzy_prior_never_fires_by_itself(self):
        d = decide_great_fire(
            {
                "time_until_press_ms": -5.0,
                "great_interval_safe": False,
                "great_interval_intersects": True,
                "landing_uncertainty_width_deg": 20.0,
                "great_width_deg": 10.0,
                "white_source": "MEASURED",
                "speed_source": "FRENZY_PRIOR",
                "is_chain": True,
                "fit_sample_count": 0,
                "target_passed": False,
                "should_press_now": True,
                "reactive_safe_fallback": True,
            },
            fit_stable=False,
            speed_usable=False,
        )
        self.assertFalse(d.allow)
        self.assertEqual(d.reason, "FRENZY_WAIT_MEASURED_SPEED")

    def test_frenzy_prior_can_fire_only_explicit_safe_handoff_tail(self):
        d = decide_great_fire(
            {
                "time_until_press_ms": 0.0,
                "great_interval_safe": False,
                "great_interval_intersects": False,
                "landing_uncertainty_width_deg": 20.0,
                "great_width_deg": 10.8,
                "success_width_deg": 53.6,
                "white_source": "MEASURED",
                "speed_source": "FRENZY_PRIOR",
                "is_chain": True,
                "fit_sample_count": 0,
                "target_passed": True,
                "should_press_now": True,
                "reactive_safe_fallback": True,
                "frenzy_handoff_success_tail": True,
            },
            fit_stable=False,
            speed_usable=False,
        )
        self.assertTrue(d.allow)
        self.assertTrue(d.best_effort)
        self.assertEqual(d.reason, "FRENZY_HANDOFF_REACTIVE_SUCCESS")

    def test_deep_frenzy_immediate_refuses_near_full_success_uncertainty(self):
        # Live chain15: 53.55° landing uncertainty for a ~54.55° success arc.
        d = decide_great_fire(
            {
                "time_until_press_ms": -33.0,
                "great_interval_safe": False,
                "great_interval_intersects": True,
                "landing_uncertainty_width_deg": 53.55,
                "great_width_deg": 10.83,
                "success_width_deg": 54.55,
                "white_source": "MEASURED",
                "speed_source": "FRENZY_BLEND",
                "is_chain": True,
                "fit_sample_count": 6,
                "target_passed": False,
                "should_press_now": True,
            },
            fit_stable=False,
            speed_usable=True,
        )
        self.assertFalse(d.allow)
        self.assertEqual(d.reason, "FRENZY_IMMEDIATE_UNCERTAINTY_TOO_WIDE")

    def test_successful_frenzy_immediate_with_moderate_uncertainty_stays_allowed(self):
        # Current successful chain6 had ~34.5° uncertainty in a ~54° success arc.
        d = decide_great_fire(
            {
                "time_until_press_ms": -5.0,
                "great_interval_safe": False,
                "great_interval_intersects": True,
                "landing_uncertainty_width_deg": 34.5,
                "great_width_deg": 10.5,
                "success_width_deg": 54.0,
                "white_source": "MEASURED",
                "speed_source": "FRENZY_BLEND",
                "is_chain": True,
                "fit_sample_count": 5,
                "target_passed": False,
                "should_press_now": True,
            },
            fit_stable=False,
            speed_usable=True,
        )
        self.assertTrue(d.allow)
        self.assertEqual(d.reason, "IMMEDIATE_SUCCESS_FALLBACK")

    def test_frenzy_blended_measured_speed_can_schedule(self):
        d = decide_great_fire(
            {
                "time_until_press_ms": 35.0,
                "great_interval_safe": True,
                "great_interval_intersects": True,
                "landing_uncertainty_width_deg": 5.0,
                "great_width_deg": 10.0,
                "white_source": "MEASURED",
                "speed_source": "FRENZY_BLEND",
                "is_chain": True,
                "fit_sample_count": 3,
                "target_passed": False,
                "should_press_now": False,
            },
            fit_stable=False,
            speed_usable=True,
        )
        self.assertTrue(d.allow)
        self.assertEqual(d.reason, "GREAT_INTERVAL_SAFE")

    def test_first_frame_session_prior_prearms_future_deadline(self):
        d = decide_great_fire(
            {
                "time_until_press_ms": 120.0,
                "great_interval_safe": False,
                "great_interval_intersects": True,
                "landing_uncertainty_width_deg": 12.0,
                "great_width_deg": 10.0,
                "white_source": "MEASURED",
                "speed_source": "SESSION_PRIOR",
                "target_passed": False,
                "should_press_now": False,
            },
            fit_stable=False,
            speed_usable=False,
        )
        self.assertTrue(d.allow)
        self.assertTrue(d.best_effort)
        self.assertEqual(d.reason, "SESSION_PRIOR_PREARM")

    def test_second_frame_segment_speed_prearms_without_full_fit(self):
        d = decide_great_fire(
            {
                "time_until_press_ms": 60.0,
                "great_interval_safe": False,
                "great_interval_intersects": True,
                "landing_uncertainty_width_deg": 14.0,
                "great_width_deg": 10.0,
                "white_source": "MEASURED",
                "speed_source": "SEGMENT_PROVISIONAL",
                "target_passed": False,
                "should_press_now": False,
            },
            fit_stable=False,
            speed_usable=False,
        )
        self.assertTrue(d.allow)
        self.assertEqual(d.reason, "SEGMENT_PROVISIONAL_PREARM")

    def test_unknown_geometry_fails_closed(self):
        d = decide_great_fire(
            {
                "time_until_press_ms": 20.0,
                "great_interval_safe": True,
                "great_interval_intersects": True,
                "landing_uncertainty_width_deg": 3.0,
                "great_width_deg": 10.0,
            },
            fit_stable=True,
            speed_usable=True,
        )
        self.assertFalse(d.allow)
        self.assertEqual(d.reason, "GREAT_GEOMETRY_UNTRUSTED")

    def test_safe_short_fit_does_not_wait_for_five_samples(self):
        d = decide_great_fire(
            {
                "time_until_press_ms": 80.0,
                "great_interval_safe": True,
                "great_interval_intersects": True,
                "landing_uncertainty_width_deg": 4.0,
                "great_width_deg": 10.0,
                "white_source": "MEASURED",
                "speed_source": "MEASURED",
            },
            fit_stable=False,
            speed_usable=True,
        )
        self.assertTrue(d.allow)
        self.assertEqual(d.reason, "GREAT_INTERVAL_SAFE")


if __name__ == "__main__":
    unittest.main()
