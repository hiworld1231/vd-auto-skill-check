import unittest

from core.lead_level_controller import LeadLevelController


class TestLeadLevelControllerV5(unittest.TestCase):
    def clean(self, c, used, err, *, mode="SCHEDULED", short_delta=0.0):
        return c.record_outcome(
            center_error_ms=err,
            actual_used_delay_ms=used,
            outcome="GREAT",
            plateau_found=True,
            trigger_mode=mode,
            scheduler_jitter_ms=0.2,
            frame_age_ms=5.0,
            detector_fallback=False,
            compensation_regime="CONTINUOUS_MEASURED_SPEED",
            fit_sample_count=8,
            fit_residual_mad_deg=1.0,
            fit_spread_deg_s=8.0,
            short_vs_long_delta_deg_s=short_delta,
            target_passed=False,
        )

    def test_ideal_lead_is_controller_invariant(self):
        c = LeadLevelController(60)
        vals = [self.clean(c, u, e)["ideal_lead_ms"] for u, e in [(60, 45), (90, 15), (105, 0)]]
        self.assertEqual(vals, [105.0, 105.0, 105.0])

    def test_cold_start_uses_four_sample_robust_cluster(self):
        c = LeadLevelController(60)
        for e in (45, 44, 46):
            self.clean(c, 60, e)
        self.assertEqual(c.current_lead_ms, 60)
        r = self.clean(c, 60, 45)
        self.assertTrue(r["updated"])
        self.assertEqual(r["update_reason"], "COLD_MEDIAN")
        self.assertAlmostEqual(c.current_lead_ms, 105.0)

    def test_safe_immediate_can_train_with_effective_dispatch_lead(self):
        c = LeadLevelController(60)
        r = self.clean(c, 54.6, 40.8, mode="IMMEDIATE")
        self.assertTrue(r["accepted"])
        self.assertAlmostEqual(r["ideal_lead_ms"], 95.4, delta=0.01)

    def test_target_passed_immediate_cannot_train(self):
        c = LeadLevelController(60)
        r = c.record_outcome(
            center_error_ms=40,
            actual_used_delay_ms=0,
            outcome="GOOD",
            plateau_found=True,
            trigger_mode="IMMEDIATE",
            scheduler_jitter_ms=.1,
            frame_age_ms=2,
            detector_fallback=False,
            compensation_regime="CONTINUOUS_MEASURED_SPEED",
            fit_sample_count=8,
            fit_residual_mad_deg=1,
            fit_spread_deg_s=5,
            target_passed=True,
        )
        self.assertFalse(r["accepted"])
        self.assertEqual(r["reject_reason"], "TARGET_ALREADY_PASSED")

    def test_recent_speed_disagreement_rejected(self):
        c = LeadLevelController(60)
        r = self.clean(c, 60, 45, short_delta=80)
        self.assertFalse(r["accepted"])
        self.assertEqual(r["reject_reason"], "RECENT_SPEED_DISAGREEMENT")

    def test_replays4_cold_start_breaks_out_of_60ms(self):
        # Live 2026-09-19 replay(4): these are effective dispatch leads
        # reconstructed from target-angle, fire-angle and speed, plus measured
        # landing center errors.  Old V5 accepted only one of nine and stayed at
        # 60ms forever.  The fixed controller must converge without a hardcoded
        # perk/base tier.
        c = LeadLevelController(60)
        samples = [
            (54.6264, 40.8307, "IMMEDIATE", 27.90),
            (62.6575, 51.0059, "SCHEDULED", 66.02),  # rejected recency disagreement
            (62.5183, 55.6235, "SCHEDULED", 40.41),
            (70.0800, 34.4384, "SCHEDULED", 25.56),
            (58.8446, 27.3543, "IMMEDIATE", 24.61),
        ]
        results = [
            self.clean(c, used, err, mode=mode, short_delta=short)
            for used, err, mode, short in samples
        ]
        self.assertFalse(results[1]["accepted"])
        self.assertTrue(c.initialized)
        self.assertGreater(c.current_lead_ms, 95.0)
        self.assertLess(c.current_lead_ms, 106.0)

    def test_frenzy_rejected_before_missing_plateau_noise(self):
        c = LeadLevelController(60)
        r = c.record_outcome(
            center_error_ms=None,
            actual_used_delay_ms=None,
            outcome="FRENZY_TRANSITION",
            plateau_found=False,
            trigger_mode="SCHEDULED",
            scheduler_jitter_ms=.1,
            frame_age_ms=2,
            detector_fallback=False,
            compensation_regime="CONTINUOUS_MEASURED_SPEED",
            fit_sample_count=8,
            fit_residual_mad_deg=1,
            fit_spread_deg_s=5,
            chain_count=2,
            frenzy_transition=True,
        )
        self.assertFalse(r["accepted"])
        self.assertEqual(r["reject_reason"], "FRENZY_UNVALIDATED")

    def test_clear_level_shift_moves_after_recent_cluster(self):
        c = LeadLevelController(60)
        for e in [45, 44, 46, 45]:
            self.clean(c, 60, e)
        self.assertAlmostEqual(c.current_lead_ms, 105)
        for ideal in [80, 81, 79, 80]:
            self.clean(c, c.current_lead_ms, ideal - c.current_lead_ms)
        self.assertLess(c.current_lead_ms, 100)


if __name__ == "__main__":
    unittest.main()
