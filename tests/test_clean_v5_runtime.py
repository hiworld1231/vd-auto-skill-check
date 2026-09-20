import tempfile
import time
import unittest
from types import SimpleNamespace
from pathlib import Path

from core.continuous_predictor import ContinuousAngularPredictor
from core.flight_recorder import FlightRecorder
from core.lead_level_controller import LeadLevelController
from core.outcome_observer import OutcomeObserver
from core.trigger import HardwareTrigger, PreciseTriggerScheduler
from core.detectors.base import extract_zones_from_masks
from core.detectors.hybrid import HybridDetector


class CleanV5Tests(unittest.TestCase):
    def test_predictor_450(self):
        p = ContinuousAngularPredictor(60.0, fit_window=10)
        w = {"start": 300.0, "end": 310.0, "center": 305.0, "width": 10.0, "source": "MEASURED"}
        for i in range(8):
            t = i * .016
            p.update(t, (30 + 450 * t) % 360, 30, w, None)
        self.assertTrue(p.has_stable_speed())
        self.assertAlmostEqual(p.speed_deg_s, 450, delta=12)

    def test_scheduler_typeerror_not_retried(self):
        calls = []
        def cb(reason, **kw):
            calls.append(reason)
            raise TypeError("inside")
        s = PreciseTriggerScheduler(cb)
        try:
            with self.assertRaises(TypeError):
                s.trigger_now("IMMEDIATE")
            self.assertEqual(calls, ["IMMEDIATE"])
        finally:
            s.close()

    def test_dry_trigger_explicit(self):
        h = HardwareTrigger(dry_run=True)
        try:
            self.assertTrue(h.trigger().success)
        finally:
            h.close()

    def test_evdev_keydown_returns_before_hold_and_timestamps_syn(self):
        class FakeUI:
            def __init__(self):
                self.events = []
            def write(self, etype, code, value):
                self.events.append(("write", etype, code, value))
            def syn(self):
                self.events.append(("syn",))
            def close(self):
                self.events.append(("close",))

        h = HardwareTrigger(dry_run=True, hold_seconds=0.080)
        h.dry_run = False
        h.backend = "EVDEV_UINPUT"
        fake = FakeUI()
        h._ui = fake
        h._ecodes = SimpleNamespace(EV_KEY=1, KEY_SPACE=57)
        try:
            t0 = time.monotonic()
            result = h.trigger()
            elapsed = time.monotonic() - t0
            self.assertTrue(result.success)
            self.assertLess(elapsed, 0.050)
            self.assertEqual(result.keydown_syn_at, result.finished_at)
            self.assertEqual(fake.events[0], ("write", 1, 57, 1))
            self.assertEqual(fake.events[1], ("syn",))
        finally:
            h.close()
        self.assertIn(("write", 1, 57, 0), fake.events)
        self.assertIsNotNone(h.last_release_at())

    def test_outcome_plateau(self):
        o = OutcomeObserver(60)
        w = {"start": 95.0, "end": 105.0, "center": 100.0, "width": 10.0, "source": "MEASURED"}
        b = {"start": 105.0, "end": 145.0, "center": 125.0, "width": 40.0, "source": "MEASURED"}
        o.on_trigger(1.0, 100.0, 300.0, w, b, used_latency_ms=60)
        for t, a in [(1.05, 101.0), (1.07, 101.2), (1.09, 100.9), (1.11, 101.1)]:
            o.observe_sample(t, a, 30)
        result = o.conclude_check()
        self.assertTrue(result["plateau_found"])
        self.assertEqual(result["outcome"], "GREAT")
        self.assertAlmostEqual(result["observed_response_ms"], 50.0, delta=0.01)

    def test_decoded_duplicates_do_not_fake_a_landing_plateau(self):
        o = OutcomeObserver(60)
        w = {"start": 95.0, "end": 105.0, "center": 100.0, "width": 10.0, "source": "MEASURED"}
        o.on_trigger(1.0, 100.0, 900.0, w, None, used_latency_ms=60)
        # Four 120-FPS decoded samples span only ~25 ms and can represent just
        # two unique 60-Hz render frames.
        for t in (1.050, 1.0583, 1.0666, 1.0749):
            o.observe_sample(t, 100.0, 30)
        self.assertFalse(o.has_plateau())

    def test_time_supported_freeze_confirms_landing(self):
        o = OutcomeObserver(60)
        w = {"start": 95.0, "end": 105.0, "center": 100.0, "width": 10.0, "source": "MEASURED"}
        o.on_trigger(1.0, 100.0, 900.0, w, None, used_latency_ms=60)
        for t, a in [
            (1.050, 100.2),
            (1.058, 100.0),
            (1.067, 100.1),
            (1.075, 100.0),
            (1.084, 100.1),
            (1.092, 100.0),
        ]:
            o.observe_sample(t, a, 30)
        self.assertTrue(o.has_plateau())
        result = o.conclude_check()
        self.assertEqual(result["outcome"], "GREAT")
        self.assertGreaterEqual(result["plateau_span_ms"], 28.0)
        self.assertGreaterEqual(result["plateau_sample_count"], 3)

    def test_three_samples_over_render_time_confirm_landing(self):
        o = OutcomeObserver(60)
        w = {"start": 95.0, "end": 105.0, "center": 100.0, "width": 10.0, "source": "MEASURED"}
        o.on_trigger(1.0, 100.0, 300.0, w, None, used_latency_ms=60)
        for t, a in [(1.050, 100.2), (1.066, 100.0), (1.082, 100.1)]:
            o.observe_sample(t, a, 30)
        self.assertTrue(o.has_plateau())
        self.assertEqual(o.conclude_check()["outcome"], "GREAT")

    def test_capture_gap_does_not_bridge_into_fake_plateau(self):
        o = OutcomeObserver(60)
        w = {"start": 95.0, "end": 105.0, "center": 100.0, "width": 10.0, "source": "MEASURED"}
        o.on_trigger(1.0, 100.0, 300.0, w, None, used_latency_ms=60)
        for t in (1.050, 1.058, 1.066, 1.120, 1.128, 1.136):
            o.observe_sample(t, 100.0, 30)
        self.assertFalse(o.has_plateau())

    def test_mask_gap_after_great_is_good_not_miss(self):
        o = OutcomeObserver(60)
        w = {"start": 95.0, "end": 105.0, "center": 100.0, "width": 10.0, "source": "MEASURED"}
        # Deliberate 2 degree CV segmentation gap.
        b = {"start": 107.0, "end": 149.0, "center": 128.0, "width": 42.0, "source": "MEASURED"}
        o.on_trigger(1.0, 100.0, 300.0, w, b, used_latency_ms=60)
        for t, a in [(1.05, 106.0), (1.07, 106.1), (1.09, 105.9), (1.11, 106.0)]:
            o.observe_sample(t, a, 30)
        result = o.conclude_check()
        self.assertTrue(result["plateau_found"])
        self.assertEqual(result["outcome"], "GOOD")

    def test_frenzy_transition_is_not_reported_as_unconfirmed(self):
        o = OutcomeObserver(60)
        w = {"start": 20.0, "end": 30.0, "center": 25.0, "width": 10.0, "source": "MEASURED"}
        b = {"start": 30.0, "end": 70.0, "center": 50.0, "width": 40.0, "source": "MEASURED"}
        o.on_trigger(1.0, 25.0, 300.0, w, b, used_latency_ms=60)
        for t, a in [(1.02, 18.0), (1.04, 24.0), (1.06, 30.0), (1.08, 36.0)]:
            o.observe_sample(t, a, 30)
        result = o.conclude_check(frenzy_transition=True)
        self.assertEqual(result["outcome"], "FRENZY_TRANSITION")
        self.assertFalse(result["plateau_found"])

    def test_recorder_drains(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as td:
            r = FlightRecorder(Path(td), save_diagnostic_strip=False, record_all=True)
            t = time.monotonic()
            r.start_check(t, latency_ms=60)
            r.on_frame(t, np.zeros((20, 20, 3), dtype=np.uint8), None)
            r.end_check(t + .1, {"outcome": "GREAT"})
            r.close()
            self.assertEqual(len(list(Path(td).glob("check_*.json"))), 1)


    def test_recorder_persists_session_metadata(self):
        import json
        with tempfile.TemporaryDirectory() as td:
            r = FlightRecorder(
                Path(td),
                save_diagnostic_strip=False,
                session_meta={"session_id": "abc", "build_git_sha": "deadbeef"},
            )
            t = time.monotonic()
            r.start_check(t, latency_ms=60)
            r.end_check(t + .1, {"outcome": "GREAT"})
            r.close()
            payload = json.loads(next(Path(td).glob("check_*.json")).read_text())
            self.assertEqual(payload["session"]["session_id"], "abc")
            self.assertEqual(payload["session"]["build_git_sha"], "deadbeef")

    def test_recorder_keeps_great_json_without_record_all(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as td:
            r = FlightRecorder(Path(td), save_diagnostic_strip=False, record_all=False)
            t = time.monotonic()
            r.start_check(t, latency_ms=60)
            r.on_frame(t, np.zeros((20, 20, 3), dtype=np.uint8), None)
            r.end_check(t + .1, {"outcome": "GREAT"})
            r.close()
            self.assertEqual(len(list(Path(td).glob("check_*.json"))), 1)

    def test_reconstructed_zone_provenance_is_preserved(self):
        import numpy as np
        white = np.zeros(360, dtype=bool)
        white[40:50] = True
        black = np.zeros(360, dtype=bool)
        w, b = extract_zones_from_masks(white, black)
        self.assertEqual(w["source"], "MEASURED")
        self.assertEqual(b["source"], "RECONSTRUCTED_FROM_WHITE")

    def test_hybrid_geometry_rebuilds_ray_tables(self):
        h = HybridDetector()
        before = h.needle_indices_1d.copy()
        h.set_geometry(163.0, 160.0)
        self.assertEqual(h.cx, 163.0)
        self.assertEqual(h.cy, 160.0)
        self.assertFalse((before == h.needle_indices_1d).all())

    def test_plateau_query_is_non_destructive(self):
        o = OutcomeObserver(60)
        w = {"start": 95.0, "end": 105.0, "center": 100.0, "width": 10.0, "source": "MEASURED"}
        o.on_trigger(1.0, 100.0, 300.0, w, None, used_latency_ms=60)
        for t, a in [(1.05, 101.0), (1.07, 101.1), (1.09, 100.9), (1.11, 101.0)]:
            o.observe_sample(t, a, 30)
        self.assertTrue(o.has_plateau())
        self.assertEqual(o.conclude_check()["outcome"], "GREAT")

    def test_frenzy_does_not_train_lead(self):
        c = LeadLevelController(60)
        result = c.record_outcome(
            center_error_ms=5, actual_used_delay_ms=60, outcome="GREAT", plateau_found=True,
            trigger_mode="SCHEDULED", scheduler_jitter_ms=.1, frame_age_ms=2,
            detector_fallback=False, compensation_regime="CONTINUOUS_MEASURED_SPEED",
            fit_sample_count=8, fit_residual_mad_deg=.3, fit_spread_deg_s=3,
            speed_at_lock=330, speed_at_fire=331, chain_count=2, white_source="MEASURED",
        )
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reject_reason"], "FRENZY_UNVALIDATED")


if __name__ == "__main__":
    unittest.main()
