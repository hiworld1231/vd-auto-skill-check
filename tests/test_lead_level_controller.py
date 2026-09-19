import unittest

from core.lead_level_controller import LeadLevelController


class TestLeadLevelControllerV5(unittest.TestCase):
    def clean(
        self,
        c,
        used=60.0,
        err=0.0,
        *,
        mode="SCHEDULED",
        short_delta=0.0,
        observed=None,
        jitter=0.2,
    ):
        return c.record_outcome(
            center_error_ms=err,
            actual_used_delay_ms=used,
            outcome="GREAT",
            plateau_found=True,
            trigger_mode=mode,
            scheduler_jitter_ms=jitter,
            frame_age_ms=5.0,
            detector_fallback=False,
            compensation_regime="CONTINUOUS_MEASURED_SPEED",
            fit_sample_count=8,
            fit_residual_mad_deg=1.0,
            fit_spread_deg_s=8.0,
            short_vs_long_delta_deg_s=short_delta,
            target_passed=False,
            observed_response_ms=observed,
        )

    def test_geometric_fallback_invariant(self):
        c = LeadLevelController(60)
        vals = [
            self.clean(c, u, e)["ideal_lead_ms"]
            for u, e in [(60, 45), (90, 15), (105, 0)]
        ]
        self.assertEqual(vals, [105.0, 105.0, 105.0])

    def test_observed_freeze_delay_is_primary_signal(self):
        c = LeadLevelController(60)
        r = self.clean(c, used=20, err=70, observed=84.5, jitter=12.0)
        self.assertTrue(r["accepted"])
        self.assertAlmostEqual(r["ideal_lead_ms"], 84.5)
        self.assertEqual(r["observation_source"], "OBSERVED_FREEZE_DELAY")

    def test_replay6_cold_start_uses_direct_response_cluster(self):
        c = LeadLevelController(60)
        vals = [74.96, 74.79, 95.71, 99.06]
        for x in vals[:-1]:
            r = self.clean(c, observed=x)
            self.assertFalse(r.get("updated", False))
        r = self.clean(c, observed=vals[-1])
        self.assertTrue(r["updated"])
        self.assertEqual(r["update_reason"], "COLD_MEDIAN")
        self.assertAlmostEqual(c.current_lead_ms, 85.335, delta=0.1)

    def test_replay6_noise_does_not_cause_100_to_80_oscillation(self):
        c = LeadLevelController(60)
        # First four normal response observations from replay(6).
        for x in [74.96, 74.79, 95.71, 99.06]:
            self.clean(c, observed=x)
        base = c.current_lead_ms
        self.assertAlmostEqual(base, 85.335, delta=0.1)

        # Remaining normal checks are noisy/bimodal.  The old four-sample
        # changepoint detector jumped down by 20 ms.  Seven-sample confirmation
        # must keep the session lead stable here.
        for x in [133.07, 102.54, 82.23, 126.54, 65.89, 65.95]:
            self.clean(c, observed=x)
        self.assertAlmostEqual(c.current_lead_ms, base, delta=0.1)

    def test_confirmed_seven_sample_level_shift_is_bounded(self):
        c = LeadLevelController(60)
        for x in [99, 100, 101, 100]:
            self.clean(c, observed=x)
        self.assertAlmostEqual(c.current_lead_ms, 100.0, delta=1.0)
        for x in [80, 81, 79, 80]:
            self.clean(c, observed=x)
        self.assertAlmostEqual(c.current_lead_ms, 88.0, delta=1.0)

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

    def test_frenzy_rejected_before_response_learning(self):
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
            observed_response_ms=None,
        )
        self.assertFalse(r["accepted"])
        self.assertEqual(r["reject_reason"], "FRENZY_UNVALIDATED")


if __name__ == "__main__":
    unittest.main()
