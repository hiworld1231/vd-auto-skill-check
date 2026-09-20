import unittest

from core.lead_level_controller import LeadLevelController


class TestRobustLeadLevelController(unittest.TestCase):
    def clean(
        self,
        c,
        used,
        err,
        *,
        observed=None,
        mode="SCHEDULED",
        short_delta=0.0,
        spread=8.0,
        jitter=0.2,
        outcome="GREAT",
    ):
        return c.record_outcome(
            center_error_ms=err,
            actual_used_delay_ms=used,
            outcome=outcome,
            plateau_found=True,
            trigger_mode=mode,
            scheduler_jitter_ms=jitter,
            frame_age_ms=5.0,
            detector_fallback=False,
            compensation_regime="CONTINUOUS_MEASURED_SPEED",
            fit_sample_count=8,
            fit_residual_mad_deg=1.0,
            fit_spread_deg_s=spread,
            short_vs_long_delta_deg_s=short_delta,
            target_passed=False,
            observed_response_ms=observed,
            white_source="MEASURED",
            black_source="MEASURED",
        )

    def test_geometric_dispatch_invariant_is_primary(self):
        c = LeadLevelController(60)
        vals = [
            self.clean(c, u, e, observed=135)["ideal_lead_ms"]
            for u, e in [(60, 45), (90, 15), (105, 0)]
        ]
        self.assertEqual(vals, [105.0, 105.0, 105.0])

    def test_replay8_cold_cluster_sets_robust_median(self):
        c = LeadLevelController(60)
        samples = [
            (60.0, 45.8),
            (57.9, 59.9),
            (59.4, 29.4),
            (58.9, 33.7),
        ]
        for used, err in samples[:-1]:
            r = self.clean(c, used, err)
            self.assertFalse(r.get("updated", False))
        r = self.clean(c, *samples[-1])
        self.assertTrue(r["updated"])
        self.assertEqual(r["update_reason"], "COLD_ROBUST_MEDIAN")
        self.assertAlmostEqual(c.current_lead_ms, 99.2, delta=0.2)

    def test_freeze_delay_is_diagnostic_only(self):
        c = LeadLevelController(60)
        r = self.clean(c, 60, 45, observed=155)
        self.assertTrue(r["accepted"])
        self.assertEqual(r["observation_source"], "GEOMETRIC_DISPATCH_INVARIANT")
        self.assertAlmostEqual(r["ideal_lead_ms"], 105.0)
        self.assertAlmostEqual(r["response_disagreement_ms"], 50.0)

    def test_noisy_cold_start_recovers_when_recent_cluster_becomes_coherent(self):
        c = LeadLevelController(60)
        samples = [
            (54.4, 49.8),   # ideal 104.2
            (60.0, 98.3),   # 158.3
            (32.3, 72.5),   # 104.8
            (48.4, 105.1),  # 153.5 -> first four too noisy
            (46.3, 52.2),   # 98.5
            (58.4, 43.8),   # 102.2 -> last four coherent enough
        ]
        results = [self.clean(c, u, e, outcome="GOOD") for u, e in samples]
        self.assertFalse(results[3].get("updated", False))
        self.assertTrue(c.initialized)
        self.assertEqual(results[-1]["update_reason"], "COLD_ROBUST_MEDIAN")
        self.assertAlmostEqual(c.current_lead_ms, 103.5, delta=0.3)

    def test_positive_level_shift_is_symmetric_and_bounded(self):
        c = LeadLevelController(60)
        for ideal in [75.1, 103.6, 78.3, 91.6]:
            self.clean(c, 60, ideal - 60)
        self.assertAlmostEqual(c.current_lead_ms, 84.95, delta=0.2)

        late = [95.1, 107.7, 98.9, 96.2]
        for ideal in late[:-1]:
            r = self.clean(c, c.current_lead_ms, ideal - c.current_lead_ms)
            self.assertFalse(r.get("updated", False))
        r = self.clean(c, c.current_lead_ms, late[-1] - c.current_lead_ms)
        self.assertTrue(r["updated"])
        self.assertEqual(r["update_reason"], "ROBUST_LEVEL_SHIFT")
        self.assertGreater(r["step_ms"], 0.0)
        self.assertLessEqual(r["step_ms"], 12.0 + 1e-9)

    def test_negative_level_shift_uses_same_rule(self):
        c = LeadLevelController(60)
        for ideal in [99, 100, 101, 100]:
            self.clean(c, 60, ideal - 60)
        self.assertAlmostEqual(c.current_lead_ms, 100.0, delta=0.1)
        before = c.current_lead_ms

        for ideal in [79, 80, 78, 79]:
            r = self.clean(c, before, ideal - before)
        self.assertTrue(r["updated"])
        self.assertEqual(r["update_reason"], "ROBUST_LEVEL_SHIFT")
        self.assertLess(r["step_ms"], 0.0)
        self.assertGreaterEqual(r["step_ms"], -12.0 - 1e-9)

    def test_lead_uncertainty_has_floor_and_shrinks_on_tight_cluster(self):
        c = LeadLevelController(60)
        self.assertAlmostEqual(c.get_uncertainty_ms(), 12.0)
        for ideal in [100, 100, 100, 100]:
            self.clean(c, 60, ideal - 60)
        self.assertTrue(c.initialized)
        self.assertAlmostEqual(c.get_uncertainty_ms(), 2.0)
        self.assertAlmostEqual(c.telemetry()["lead_uncertainty_ms"], 2.0)

    def test_unstable_fit_cannot_train(self):
        c = LeadLevelController(60)
        r = self.clean(c, 60, 45, spread=80.0)
        self.assertFalse(r["accepted"])
        self.assertEqual(r["reject_reason"], "UNSTABLE_SPEED_FIT")

    def test_target_passed_cannot_train(self):
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
            observed_response_ms=90,
        )
        self.assertFalse(r["accepted"])
        self.assertEqual(r["reject_reason"], "TARGET_ALREADY_PASSED")

    def test_frenzy_never_trains_session_lead(self):
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


if __name__ == "__main__":
    unittest.main()
