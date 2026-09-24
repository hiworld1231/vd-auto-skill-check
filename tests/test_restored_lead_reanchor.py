import unittest

from core.lead_level_controller import LeadLevelController


class RestoredLeadReanchorTests(unittest.TestCase):
    @staticmethod
    def clean(c: LeadLevelController, ideal: float):
        used = c.current_lead_ms
        return c.record_outcome(
            center_error_ms=ideal - used,
            actual_used_delay_ms=used,
            outcome="GOOD",
            plateau_found=True,
            trigger_mode="IMMEDIATE",
            scheduler_jitter_ms=0.2,
            frame_age_ms=1.0,
            detector_fallback=False,
            compensation_regime="CONTINUOUS_MEASURED_SPEED",
            fit_sample_count=8,
            fit_residual_mad_deg=1.0,
            fit_spread_deg_s=8.0,
            short_vs_long_delta_deg_s=0.0,
            target_passed=False,
            observed_response_ms=ideal,
            white_source="MEASURED",
            black_source="MEASURED",
        )

    def test_live_119_to_151_cluster_reanchors_on_third_trusted_sample(self):
        c = LeadLevelController(60.0)
        c.restore_calibration(119.8, 6.9)

        for ideal in (151.2, 155.4):
            r = self.clean(c, ideal)
            self.assertTrue(r["accepted"])
            self.assertFalse(r["updated"])
            self.assertAlmostEqual(c.current_lead_ms, 119.8, delta=0.01)

        r = self.clean(c, 151.2)
        self.assertTrue(r["updated"])
        self.assertEqual(r["update_reason"], "RESTORED_CALIBRATION_REANCHOR")
        self.assertAlmostEqual(c.current_lead_ms, 151.2, delta=0.01)
        self.assertFalse(c.restored_from_disk)

    def test_live_126_with_saved_uncertainty_reanchors(self):
        c = LeadLevelController(60.0)
        c.restore_calibration(126.1, 7.8)

        for ideal in (106.3, 98.6):
            r = self.clean(c, ideal)
            self.assertTrue(r["accepted"])
            self.assertFalse(r["updated"])

        r = self.clean(c, 113.5)
        self.assertTrue(r["updated"])
        self.assertEqual(r["update_reason"], "RESTORED_CALIBRATION_REANCHOR")
        self.assertAlmostEqual(c.current_lead_ms, 106.3, delta=0.01)
        self.assertFalse(c.restored_from_disk)

    def test_restored_reanchor_requires_all_samples_on_same_side(self):
        c = LeadLevelController(60.0)
        c.restore_calibration(126.1, 10.0)

        for ideal in (104.0, 106.0, 127.0):
            r = self.clean(c, ideal)

        self.assertFalse(r["updated"])
        self.assertAlmostEqual(c.current_lead_ms, 126.1, delta=0.01)
        self.assertTrue(c.restored_from_disk)

    def test_small_live_difference_does_not_fast_reanchor(self):
        c = LeadLevelController(60.0)
        c.restore_calibration(119.8, 6.9)

        for ideal in (126.0, 125.5, 126.5):
            r = self.clean(c, ideal)

        self.assertFalse(r["updated"])
        self.assertAlmostEqual(c.current_lead_ms, 119.8, delta=0.01)
        self.assertTrue(c.restored_from_disk)

    def test_non_restored_large_shift_stays_on_bounded_slew(self):
        c = LeadLevelController(119.8)
        c.initialized = True

        for ideal in (151.2, 155.4, 151.2):
            r = self.clean(c, ideal)
            self.assertFalse(r["updated"])

        r = self.clean(c, 152.0)
        self.assertTrue(r["updated"])
        self.assertEqual(r["update_reason"], "ROBUST_LEVEL_SHIFT")
        self.assertLessEqual(r["step_ms"], 6.0 + 1e-9)
        self.assertAlmostEqual(c.current_lead_ms, 125.8, delta=0.01)


if __name__ == "__main__":
    unittest.main()
