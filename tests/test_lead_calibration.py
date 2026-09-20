import json
import tempfile
import unittest
from pathlib import Path

from core.lead_calibration import (
    CALIBRATION_MODEL,
    LeadCalibrationStore,
    make_calibration_fingerprint,
)
from core.lead_level_controller import LeadLevelController


class PersistentLeadCalibrationTests(unittest.TestCase):
    def fingerprint(self):
        return make_calibration_fingerprint(
            fps=120,
            region={"left": 800, "top": 420, "width": 320, "height": 240},
            detector="hybrid",
            capture_backend="GSR_VFR",
            input_backend="EVDEV_UINPUT",
        )

    def test_round_trip_with_matching_environment(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "calibration.json"
            store = LeadCalibrationStore(
                path, self.fingerprint(), max_age_s=100.0
            )
            store.save(
                lead_ms=98.0,
                uncertainty_ms=2.5,
                trusted_sample_count=17,
                now_epoch=1000.0,
            )
            result = store.load(now_epoch=1050.0)
            self.assertTrue(result.accepted)
            self.assertEqual(result.reason, "RESTORED")
            self.assertAlmostEqual(result.lead_ms, 98.0)
            self.assertAlmostEqual(result.uncertainty_ms, 2.5)
            self.assertEqual(result.trusted_sample_count, 17)

    def test_fingerprint_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "calibration.json"
            store = LeadCalibrationStore(path, self.fingerprint())
            store.save(
                lead_ms=98.0,
                uncertainty_ms=2.5,
                trusted_sample_count=8,
                now_epoch=1000.0,
            )
            changed = self.fingerprint()
            changed["fps"] = 60
            result = LeadCalibrationStore(path, changed).load(now_epoch=1001.0)
            self.assertFalse(result.accepted)
            self.assertEqual(result.reason, "FINGERPRINT_MISMATCH")

    def test_expired_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "calibration.json"
            store = LeadCalibrationStore(
                path, self.fingerprint(), max_age_s=10.0
            )
            store.save(
                lead_ms=98.0,
                uncertainty_ms=2.5,
                trusted_sample_count=8,
                now_epoch=1000.0,
            )
            result = store.load(now_epoch=1011.0)
            self.assertFalse(result.accepted)
            self.assertEqual(result.reason, "EXPIRED")

    def test_old_cross_session_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "calibration.json"
            payload = {
                "schema_version": 1,
                "calibration_model": "v6-keydown-syn-great-center-predictive-unc-v2",
                "saved_at_epoch": 1000.0,
                "fingerprint": self.fingerprint(),
                "lead_ms": 126.0,
                "uncertainty_ms": 8.0,
                "trusted_sample_count": 12,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = LeadCalibrationStore(
                path, self.fingerprint(), max_age_s=100.0
            ).load(now_epoch=1001.0)
            self.assertFalse(result.accepted)
            self.assertEqual(result.reason, "MODEL_MISMATCH")

    def test_calibration_store_range_matches_controller_180ms_ceiling(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "calibration.json"
            store = LeadCalibrationStore(path, self.fingerprint())
            store.save(
                lead_ms=173.2,
                uncertainty_ms=4.0,
                trusted_sample_count=8,
                now_epoch=1000.0,
            )
            result = store.load(now_epoch=1001.0)
            self.assertTrue(result.accepted)
            self.assertAlmostEqual(result.lead_ms, 173.2)

    def test_corrupt_or_out_of_range_data_never_restores(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "calibration.json"
            path.write_text("{bad", encoding="utf-8")
            store = LeadCalibrationStore(path, self.fingerprint())
            self.assertEqual(store.load().reason, "UNREADABLE")

            payload = {
                "schema_version": 1,
                "calibration_model": CALIBRATION_MODEL,
                "saved_at_epoch": 1000.0,
                "fingerprint": self.fingerprint(),
                "lead_ms": 999.0,
                "uncertainty_ms": 2.0,
                "trusted_sample_count": 5,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                store.load(now_epoch=1001.0).reason, "INVALID_LEAD"
            )

    def test_controller_restores_without_fake_samples_then_revalidates(self):
        c = LeadLevelController(60)
        c.restore_calibration(100.0, 3.0)
        self.assertTrue(c.initialized)
        self.assertTrue(c.restored_from_disk)
        self.assertEqual(len(c.samples), 0)
        self.assertAlmostEqual(c.get_lead_ms(), 100.0)
        self.assertAlmostEqual(c.get_uncertainty_ms(), 3.0)

        def clean(ideal):
            return c.record_outcome(
                center_error_ms=ideal - c.current_lead_ms,
                actual_used_delay_ms=c.current_lead_ms,
                outcome="GREAT",
                plateau_found=True,
                trigger_mode="SCHEDULED",
                scheduler_jitter_ms=0.1,
                frame_age_ms=2.0,
                detector_fallback=False,
                compensation_regime="CONTINUOUS_MEASURED_SPEED",
                fit_sample_count=8,
                fit_residual_mad_deg=0.5,
                fit_spread_deg_s=5.0,
                white_source="MEASURED",
                short_vs_long_delta_deg_s=0.0,
            )

        for ideal in (101.0, 100.0, 101.0):
            clean(ideal)
            self.assertTrue(c.restored_from_disk)
        clean(100.0)
        self.assertFalse(c.restored_from_disk)
        self.assertAlmostEqual(c.current_lead_ms, 100.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
