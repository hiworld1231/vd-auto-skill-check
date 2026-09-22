import unittest

from core.fire_policy import decide_great_fire
from core.outcome_observer import OutcomeObserver
from core.post_fire_lifecycle import FRENZY, LANDED, WAIT, PostFireLifecycle


class LiveReplayRegressionTests(unittest.TestCase):
    def test_20260922_180429_mid_motion_pause_is_not_final_landing(self):
        """A 30 ms render pause must be invalidated when motion resumes.

        This is the shape of live replay check_20260922_180429_0003. The old
        observer permanently selected the early ~54.86° pause and reported a
        MISS. The needle then continued and finally froze around 94.4°, inside
        the measured GOOD sector.
        """
        white = {
            "start": 70.98866550673331,
            "end": 81.3512754042117,
            "center": 76.16997045547251,
            "width": 10.362609897478393,
            "source": "MEASURED",
        }
        black = {
            "start": 82.44347826086957,
            "end": 124.91420907876017,
            "center": 103.67884366981487,
            "width": 42.4707308178906,
            "source": "MEASURED",
        }
        o = OutcomeObserver(125.8)
        o.on_trigger(0.0, white["center"], 300.386826409114, white, black)

        for t, angle in [
            (0.0383, 54.861111111111114),
            (0.0448, 54.861111111111114),
            (0.0596, 54.861111111111114),
            (0.0686, 54.861111111111114),
        ]:
            o.observe_sample(t, angle, 30.0)
        self.assertTrue(o.has_plateau())
        self.assertEqual(o.conclude_check()["outcome"], "MISS")

        # Fresh motion invalidates the historical pause immediately.
        for t, angle in [
            (0.0735, 57.08163265306123),
            (0.1080, 62.567669172932334),
            (0.1399, 71.07738095238095),
            (0.1535, 79.92857142857143),
        ]:
            o.observe_sample(t, angle, 30.0)
        self.assertFalse(o.has_plateau())
        self.assertEqual(o.conclude_check()["outcome"], "UNCONFIRMED")

        # The later stable tail is the real landing and lies in GOOD.
        for t, angle in [
            (0.2196, 94.41025641025641),
            (0.2262, 94.41025641025641),
            (0.2361, 94.41025641025641),
            (0.2430, 94.41025641025641),
            (0.2666, 94.36440677966101),
        ]:
            o.observe_sample(t, angle, 30.0)
        self.assertTrue(o.has_plateau())
        result = o.conclude_check()
        self.assertEqual(result["outcome"], "GOOD")
        self.assertAlmostEqual(result["hit_angle"], 94.41025641025641, delta=0.1)

    def test_lifecycle_can_leave_temporary_landed_state_when_motion_resumes(self):
        sm = PostFireLifecycle()
        sm.begin(1.0)
        landed = sm.update(
            1.070,
            ring_present=True,
            plateau_found=True,
            zone_moved=False,
            fresh_motion=False,
        )
        self.assertEqual(landed.state, LANDED)

        resumed = sm.update(
            1.090,
            ring_present=True,
            plateau_found=False,
            zone_moved=True,
            fresh_motion=True,
        )
        self.assertEqual(resumed.state, WAIT)

        sm.update(
            1.130,
            ring_present=True,
            plateau_found=False,
            zone_moved=True,
            fresh_motion=True,
        )
        frenzy = sm.update(
            1.150,
            ring_present=True,
            plateau_found=False,
            zone_moved=True,
            fresh_motion=True,
        )
        self.assertEqual(frenzy.state, FRENZY)

    def test_20260922_180327_false_segment_speed_cannot_immediate_fire(self):
        """One adjacent segment cannot physically fire before robust fitting."""
        d = decide_great_fire(
            {
                "time_until_press_ms": -12.0,
                "great_interval_safe": False,
                "great_interval_intersects": True,
                "landing_uncertainty_width_deg": 0.0,
                "great_width_deg": 10.0,
                "white_source": "MEASURED",
                "speed_source": "SEGMENT_PROVISIONAL",
                "target_passed": False,
                "should_press_now": True,
            },
            fit_stable=False,
            speed_usable=False,
        )
        self.assertFalse(d.allow)
        self.assertEqual(d.reason, "SEGMENT_PROVISIONAL_WAIT_MEASURED_SPEED")


if __name__ == "__main__":
    unittest.main()
