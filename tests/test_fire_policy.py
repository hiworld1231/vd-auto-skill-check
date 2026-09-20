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
            },
            fit_stable=True,
            speed_usable=True,
        )
        self.assertTrue(d.allow)
        self.assertEqual(d.reason, "GREAT_INTERVAL_SAFE")
        self.assertFalse(d.best_effort)

    def test_old_armed_deadline_must_be_rejected_when_interval_becomes_unsafe(self):
        d = decide_great_fire(
            {
                "time_until_press_ms": 18.0,
                "great_interval_safe": False,
                "great_interval_intersects": True,
                "landing_uncertainty_width_deg": 18.0,
                "great_width_deg": 10.0,
            },
            fit_stable=False,
            speed_usable=True,
        )
        self.assertFalse(d.allow)
        self.assertEqual(d.reason, "GREAT_INTERVAL_UNSAFE")

    def test_short_track_can_fire_when_full_interval_still_fits_great(self):
        d = decide_great_fire(
            {
                "time_until_press_ms": 20.0,
                "great_interval_safe": True,
                "great_interval_intersects": True,
                "landing_uncertainty_width_deg": 5.0,
                "great_width_deg": 10.0,
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
            },
            fit_stable=True,
            speed_usable=True,
        )
        self.assertFalse(d.allow)
        self.assertEqual(d.reason, "GREAT_INTERVAL_UNSAFE")

    def test_far_short_fit_waits_for_more_evidence(self):
        d = decide_great_fire(
            {
                "time_until_press_ms": 80.0,
                "great_interval_safe": True,
                "great_interval_intersects": True,
                "landing_uncertainty_width_deg": 4.0,
                "great_width_deg": 10.0,
            },
            fit_stable=False,
            speed_usable=True,
        )
        self.assertFalse(d.allow)
        self.assertEqual(d.reason, "FIT_NOT_READY")


if __name__ == "__main__":
    unittest.main()
