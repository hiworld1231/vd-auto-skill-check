import unittest

from core.lead_level_controller import LeadLevelController


class TestLeadLevelControllerV5(unittest.TestCase):
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

    def test_corrected_geometric_invariant_is_primary(self):
        c = LeadLevelController(60)
        vals = [
            self.clean(c, u, e, observed=135)["ideal_lead_ms"]
            for u, e in [(60, 45), (90, 15), (105, 0)]
        ]
        self.assertEqual(vals, [105.0, 105.0, 105.0])

    def test_replay8_cold_start_is_about_99ms_not_113ms(self):
        # First four live checks from replays(8), using real dispatch eff + landing residual.
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
        self.assertEqual(r["update_reason"], "COLD_MEDIAN")
        self.assertAlmostEqual(c.current_lead_ms, 99.2, delta=0.2)

    def test_freeze_delay_is_diagnostic_not_controller_target(self):
        c = LeadLevelController(60)
        # Geometric target says 105 ms; response observer is noisy but still plausible.
        r = self.clean(c, 60, 45, observed=130)
        self.assertTrue(r["accepted"])
        self.assertEqual(r["observation_source"], "GEOMETRIC_DISPATCH_INVARIANT")
        self.assertAlmostEqual(r["ideal_lead_ms"], 105.0)
        self.assertAlmostEqual(r["response_disagreement_ms"], 25.0)

    def test_large_response_geometry_disagreement_is_rejected(self):
        c = LeadLevelController(60)
        r = self.clean(c, 60, 45, observed=155)
        self.assertFalse(r["accepted"])
        self.assertEqual(r["reject_reason"], "RESPONSE_GEOMETRY_DISAGREE")

    def test_warm_shift_uses_seven_samples_and_bounded_step(self):
        c = LeadLevelController(60)
        for ideal in [98, 100, 99, 101]:
            self.clean(c, 60, ideal - 60)
        self.assertTrue(c.initialized)
        self.assertAlmostEqual(c.current_lead_ms, 99.5, delta=0.6)

        # Coherent new level near 132 ms.  Controller needs seven recent samples
        # and can move by at most 8 ms per update.
        before = c.current_lead_ms
        for ideal in [131, 132, 130, 133, 132, 131]:
            self.clean(c, before, ideal - before)
        self.assertAlmostEqual(c.current_lead_ms, before, delta=0.01)
        r = self.clean(c, before, 132 - before)
        self.assertTrue(r["updated"])
        self.assertLessEqual(abs(r["step_ms"]), 8.0 + 1e-9)

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

    def test_frenzy_never_trains(self):
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
