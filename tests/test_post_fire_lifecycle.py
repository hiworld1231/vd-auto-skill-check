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

    def test_absence_then_reappearance_alone_does_not_confirm_frenzy(self):
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
        d = sm.update(
            1.090,
            ring_present=True,
            plateau_found=False,
            zone_moved=False,
            fresh_motion=False,
        )
        self.assertEqual(d.state, WAIT)

    def test_absence_reappearance_needs_stable_relocation_and_new_motion(self):
        sm = PostFireLifecycle(
            relocation_not_before_s=0.500,
            relocation_frames=3,
        )
        sm.begin(1.0)
        sm.update(1.050, ring_present=False, plateau_found=False)
        sm.update(1.085, ring_present=False, plateau_found=False)

        first = sm.update(
            1.090,
            ring_present=True,
            plateau_found=False,
            zone_moved=True,
            zone_center=242.0,
            reappearance_proof=True,
            fresh_motion=False,
        )
        self.assertEqual(first.state, WAIT)

        second = sm.update(
            1.110,
            ring_present=True,
            plateau_found=False,
            zone_moved=True,
            zone_center=243.0,
            reappearance_proof=True,
            fresh_motion=True,
        )
        self.assertEqual(second.state, FRENZY)
        self.assertEqual(
            second.reason, "ABSENCE_REAPPEAR_WITH_STABLE_RELOCATION"
        )

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
                zone_center=242.0,
                fresh_motion=True,
            )
            self.assertEqual(d.state, WAIT)
        d = sm.update(
            1.140,
            ring_present=True,
            plateau_found=False,
            zone_moved=True,
            zone_center=243.0,
            fresh_motion=True,
        )
        self.assertEqual(d.state, FRENZY)
        self.assertEqual(
            d.reason, "PERSISTENT_STABLE_RELOCATION_WITH_NEW_MOTION"
        )

    def test_jittering_relocation_candidates_never_build_streak(self):
        sm = PostFireLifecycle(
            relocation_not_before_s=0.100,
            relocation_frames=3,
            relocation_center_tolerance_deg=6.0,
        )
        sm.begin(1.0)
        centers = (220.0, 245.0, 270.0, 225.0)
        for i, center in enumerate(centers):
            d = sm.update(
                1.120 + i * 0.020,
                ring_present=True,
                plateau_found=False,
                zone_moved=True,
                zone_center=center,
                fresh_motion=True,
            )
            self.assertEqual(d.state, WAIT)
            self.assertEqual(d.relocated_streak, 1)

    def test_old_needle_rollback_cannot_substitute_for_new_generation_motion(self):
        sm = PostFireLifecycle(
            relocation_not_before_s=0.100,
            relocation_frames=3,
        )
        sm.begin(1.0)
        for t in (1.120, 1.140, 1.160, 1.180):
            d = sm.update(
                t,
                ring_present=True,
                plateau_found=False,
                zone_moved=True,
                zone_center=245.0,
                rollback=True,
                fresh_motion=False,
            )
            self.assertEqual(d.state, WAIT)

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
