import unittest

from core.post_fire_lifecycle import FRENZY, LANDED, RING_END, WAIT, PostFireLifecycle


class PostFireLifecycleTests(unittest.TestCase):
    def test_false_zone_relocation_is_vetoed_by_landing_plateau(self):
        sm = PostFireLifecycle()
        sm.begin(1.0)
        for t in (1.080, 1.100, 1.120):
            d = sm.update(
                t,
                ring_present=True,
                plateau_found=False,
                zone_moved=True,
                fresh_motion=True,
            )
            self.assertEqual(d.state, WAIT)
        d = sm.update(
            1.130,
            ring_present=True,
            plateau_found=True,
            zone_moved=True,
            fresh_motion=False,
        )
        self.assertEqual(d.state, LANDED)

    def test_absence_then_reappearance_confirms_frenzy(self):
        sm = PostFireLifecycle()
        sm.begin(1.0)
        self.assertEqual(
            sm.update(1.050, ring_present=False, plateau_found=False).state,
            WAIT,
        )
        self.assertEqual(
            sm.update(1.085, ring_present=False, plateau_found=False).state,
            WAIT,
        )
        d = sm.update(1.090, ring_present=True, plateau_found=False)
        self.assertEqual(d.state, FRENZY)
        self.assertEqual(d.reason, "ABSENCE_REAPPEAR")
        self.assertGreaterEqual(d.absence_ms, 30.0)

    def test_sustained_absence_ends_check_instead_of_fake_frenzy(self):
        sm = PostFireLifecycle(ring_end_absence_s=0.180)
        sm.begin(1.0)
        sm.update(1.050, ring_present=False, plateau_found=False)
        d = sm.update(1.231, ring_present=False, plateau_found=False)
        self.assertEqual(d.state, RING_END)

    def test_persistent_relocation_needs_motion_and_three_frames(self):
        sm = PostFireLifecycle(relocation_not_before_s=0.125, relocation_frames=3)
        sm.begin(1.0)
        for t in (1.100, 1.120):
            d = sm.update(
                t,
                ring_present=True,
                plateau_found=False,
                zone_moved=True,
                fresh_motion=True,
            )
            self.assertEqual(d.state, WAIT)
        d = sm.update(
            1.140,
            ring_present=True,
            plateau_found=False,
            zone_moved=True,
            fresh_motion=True,
        )
        self.assertEqual(d.state, FRENZY)
        self.assertEqual(d.reason, "PERSISTENT_RELOCATION_WITH_MOTION")

    def test_rollback_alone_never_confirms_frenzy(self):
        sm = PostFireLifecycle()
        sm.begin(1.0)
        for t in (1.130, 1.150, 1.170):
            d = sm.update(
                t,
                ring_present=True,
                plateau_found=False,
                zone_moved=False,
                rollback=True,
                fresh_motion=True,
            )
            self.assertEqual(d.state, WAIT)


if __name__ == "__main__":
    unittest.main()
