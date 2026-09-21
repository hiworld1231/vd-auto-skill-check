import tempfile
import time
import unittest
from types import SimpleNamespace
from pathlib import Path

from core.continuous_predictor import ContinuousAngularPredictor
from core.flight_recorder import FlightRecorder
from core.lead_level_controller import LeadLevelController
from core.genrush_runtime import (
    _frenzy_relocation_evidence,
    _generation_lead,
    _generation_motion_update,
    _presence_absence_update,
    _trusted_postfire_landing_sample,
)
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

    def test_prefire_absence_requires_continuous_gap(self):
        since, elapsed = _presence_absence_update(None, now=10.000, present=False)
        self.assertAlmostEqual(since, 10.000)
        self.assertAlmostEqual(elapsed, 0.0)

        since, elapsed = _presence_absence_update(since, now=10.060, present=False)
        self.assertAlmostEqual(elapsed, 0.060, delta=1e-6)

        since, elapsed = _presence_absence_update(since, now=10.070, present=True)
        self.assertIsNone(since)
        self.assertEqual(elapsed, 0.0)

        # A later single dropped frame starts a new absence window; it does
        # not inherit the age of the skillcheck or the previous loss.
        since, elapsed = _presence_absence_update(since, now=10.500, present=False)
        self.assertAlmostEqual(since, 10.500)
        self.assertAlmostEqual(elapsed, 0.0)

    def test_prefire_continuous_absence_reaches_timeout(self):
        since, _ = _presence_absence_update(None, now=20.000, present=False)
        since, elapsed = _presence_absence_update(since, now=20.101, present=False)
        self.assertGreaterEqual(elapsed, 0.100)

    def test_postfire_red_peak_without_ring_is_not_landing_evidence(self):
        self.assertFalse(
            _trusted_postfire_landing_sample(
                {
                    "ring_present": False,
                    "needle_valid": True,
                    "needle_angle": 200.0,
                    "needle_strength": 90.0,
                }
            )
        )
        self.assertTrue(
            _trusted_postfire_landing_sample(
                {
                    "ring_present": True,
                    "needle_valid": True,
                    "needle_angle": 100.0,
                    "needle_strength": 90.0,
                }
            )
        )

    def test_next_generation_motion_is_independent_from_old_landing_needle(self):
        samples = []
        self.assertFalse(
            _generation_motion_update(
                samples, t=1.000, angle=10.0, valid=True
            )
        )
        self.assertFalse(
            _generation_motion_update(
                samples, t=1.016, angle=10.1, valid=True
            )
        )
        self.assertTrue(
            _generation_motion_update(
                samples, t=1.033, angle=18.0, valid=True
            )
        )

    def test_invalid_generation_needle_cannot_fake_motion(self):
        samples = []
        for i, angle in enumerate((10.0, 30.0, 60.0)):
            moving = _generation_motion_update(
                samples,
                t=2.0 + i * 0.016,
                angle=angle,
                valid=False,
            )
            self.assertFalse(moving)
        self.assertEqual(samples, [])

    def test_small_post_hit_zone_shift_cannot_fake_frenzy(self):
        old_w = {
            "start": 80.0, "end": 90.0, "center": 85.0,
            "width": 10.0, "source": "MEASURED",
        }
        new_w = {
            "start": 102.0, "end": 112.0, "center": 107.0,
            "width": 10.0, "source": "MEASURED_FIXED_CENTER",
        }
        new_b = {
            "start": 113.0, "end": 153.0, "center": 133.0,
            "width": 40.0, "source": "MEASURED_FIXED_CENTER",
        }
        ev = _frenzy_relocation_evidence(
            new_w, new_b, old_w,
            generation_needle_valid=True,
            min_move_deg=90.0,
            max_white_black_gap_deg=10.0,
        )
        self.assertFalse(ev["qualifies"])
        self.assertAlmostEqual(ev["relocation_delta_deg"], 22.0)

    def test_large_but_unpaired_visual_artifact_cannot_fake_frenzy(self):
        old_w = {
            "start": 80.0, "end": 90.0, "center": 85.0,
            "width": 10.0, "source": "MEASURED",
        }
        new_w = {
            "start": 220.0, "end": 230.0, "center": 225.0,
            "width": 10.0, "source": "MEASURED_FIXED_CENTER",
        }
        # Geometrically impossible for the normal white->black success sector:
        new_b = {
            "start": 270.0, "end": 310.0, "center": 290.0,
            "width": 40.0, "source": "MEASURED_FIXED_CENTER",
        }
        ev = _frenzy_relocation_evidence(
            new_w, new_b, old_w,
            generation_needle_valid=True,
            min_move_deg=90.0,
            max_white_black_gap_deg=10.0,
        )
        self.assertFalse(ev["qualifies"])
        self.assertGreater(ev["relocation_delta_deg"], 90.0)
        self.assertGreater(ev["white_black_gap_deg"], 10.0)

    def test_realistic_frenzy_relocation_geometry_qualifies(self):
        old_w = {
            "start": 80.0, "end": 90.0, "center": 85.0,
            "width": 10.0, "source": "MEASURED",
        }
        new_w = {
            "start": 238.0, "end": 248.0, "center": 243.0,
            "width": 10.0, "source": "MEASURED_FIXED_CENTER",
        }
        new_b = {
            "start": 250.0, "end": 290.0, "center": 270.0,
            "width": 40.0, "source": "MEASURED_FIXED_CENTER",
        }
        ev = _frenzy_relocation_evidence(
            new_w, new_b, old_w,
            generation_needle_valid=True,
            min_move_deg=90.0,
            max_white_black_gap_deg=10.0,
        )
        self.assertTrue(ev["qualifies"])
        self.assertAlmostEqual(ev["relocation_delta_deg"], 158.0)
        self.assertAlmostEqual(ev["white_black_gap_deg"], 2.0)

    def test_archived_minimum_true_frenzy_relocation_still_qualifies(self):
        # Archived real chain6->7 transition:
        old_w = {
            "start": 221.0, "end": 231.0, "center": 226.48939900947659,
            "width": 10.0, "source": "MEASURED",
        }
        new_w = {
            "start": 98.4534192667837,
            "end": 109.29413683471543,
            "center": 103.87377805074956,
            "width": 10.840717567931733,
            "source": "MEASURED_FIXED_CENTER",
        }
        new_b = {
            "start": 109.97915310662923,
            "end": 152.11025444230802,
            "center": 131.04470377446862,
            "width": 42.13110133567879,
            "source": "MEASURED_FIXED_CENTER",
        }
        ev = _frenzy_relocation_evidence(
            new_w, new_b, old_w,
            generation_needle_valid=True,
            min_move_deg=90.0,
            max_white_black_gap_deg=10.0,
        )
        self.assertTrue(ev["qualifies"])
        self.assertAlmostEqual(
            ev["relocation_delta_deg"], 122.61562095872702, places=3
        )
        self.assertAlmostEqual(
            ev["white_black_gap_deg"], 0.6850162719138, places=3
        )

    def test_frenzy_generation_uses_separate_lead(self):
        lead_ms, unc_ms = _generation_lead(
            2,
            normal_lead_ms=126.9,
            normal_uncertainty_ms=8.1,
            frenzy_lead_ms=80.0,
            frenzy_uncertainty_ms=0.0,
        )
        self.assertAlmostEqual(lead_ms, 80.0)
        self.assertAlmostEqual(unc_ms, 0.0)

        normal_ms, normal_unc = _generation_lead(
            1,
            normal_lead_ms=126.9,
            normal_uncertainty_ms=8.1,
            frenzy_lead_ms=80.0,
            frenzy_uncertainty_ms=0.0,
        )
        self.assertAlmostEqual(normal_ms, 126.9)
        self.assertAlmostEqual(normal_unc, 8.1)

    def test_scheduler_propagates_dispatch_token(self):
        calls = []
        def cb(reason, **kw):
            calls.append((reason, kw.get("scheduler_token")))
        sched = PreciseTriggerScheduler(cb)
        try:
            sched.trigger_now("IMMEDIATE", dispatch_token=17)
            self.assertEqual(calls, [("IMMEDIATE", 17)])
        finally:
            sched.close()

    def test_old_scheduler_token_can_be_rejected_after_generation_change(self):
        physical = []
        active = {"token": 1}
        def cb(reason, **kw):
            token = kw.get("scheduler_token")
            if token != active["token"]:
                return
            physical.append(reason)

        sched = PreciseTriggerScheduler(cb, spin_window_s=0.0005)
        try:
            sched.schedule(
                time.monotonic() + 0.030,
                reason="SCHEDULED_TEST",
                dispatch_token=1,
            )
            time.sleep(0.010)
            active["token"] = 2
            time.sleep(0.050)
            self.assertEqual(physical, [])
        finally:
            sched.close()

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

    def test_extreme_phase_plateau_is_rejected_before_lifecycle_completion(self):
        o = OutcomeObserver(120)
        w = {
            "start": 95.0, "end": 105.0, "center": 100.0,
            "width": 10.0, "source": "MEASURED",
        }
        b = {
            "start": 105.0, "end": 145.0, "center": 125.0,
            "width": 40.0, "source": "MEASURED",
        }
        o.on_trigger(1.0, 100.0, 280.0, w, b, used_latency_ms=120.0)
        for t, a in [
            (1.060, 220.0),
            (1.076, 220.2),
            (1.092, 219.9),
        ]:
            o.observe_sample(t, a, 40.0)
        self.assertFalse(o.has_plateau())
        result = o.conclude_check()
        self.assertEqual(result["outcome"], "UNCONFIRMED")
        self.assertEqual(result["unconfirmed_reason"], "PHASE_OUTLIER")
        self.assertFalse(result["plateau_found"])
        self.assertFalse(result["plateau_trusted"])
        self.assertGreater(result["implied_delivery_ms"], 180.0)

    def test_impossible_early_plateau_is_skipped_for_later_valid_freeze(self):
        o = OutcomeObserver(80)
        w = {
            "start": 95.0, "end": 105.0, "center": 100.0,
            "width": 10.0, "source": "MEASURED",
        }
        b = {
            "start": 105.0, "end": 145.0, "center": 125.0,
            "width": 40.0, "source": "MEASURED",
        }
        o.on_trigger(1.0, 100.0, 300.0, w, b, used_latency_ms=65.0)

        # First stable red object implies 265ms keydown->landing and must not
        # stop lifecycle. A later freeze at 102° implies ~72ms and is valid.
        for t, a in [
            (1.030, 160.0),
            (1.046, 160.2),
            (1.062, 159.9),
            (1.082, 102.0),
            (1.098, 102.1),
            (1.114, 101.9),
        ]:
            o.observe_sample(t, a, 40.0)

        self.assertTrue(o.has_plateau())
        result = o.conclude_check()
        self.assertEqual(result["outcome"], "GREAT")
        self.assertAlmostEqual(result["hit_angle"], 102.0, delta=0.2)
        self.assertGreaterEqual(result["implied_delivery_ms"], 35.0)
        self.assertLessEqual(result["implied_delivery_ms"], 180.0)

    def test_live_bad_sample_245ms_is_not_a_plateau(self):
        o = OutcomeObserver(80)
        w = {
            "start": 46.0, "end": 56.0, "center": 51.0,
            "width": 10.0, "source": "MEASURED",
        }
        o.on_trigger(
            1.0,
            51.1,
            306.1,
            w,
            None,
            used_latency_ms=65.6,
        )
        for t, a in [
            (1.060, 106.0),
            (1.076, 106.1),
            (1.092, 105.9),
        ]:
            o.observe_sample(t, a, 40.0)

        self.assertFalse(o.has_plateau())
        result = o.conclude_check()
        self.assertEqual(result["unconfirmed_reason"], "PHASE_OUTLIER")
        self.assertAlmostEqual(result["implied_delivery_ms"], 245.0, delta=1.5)

    def test_plausible_near_sector_miss_remains_miss(self):
        o = OutcomeObserver(120)
        w = {
            "start": 95.0, "end": 105.0, "center": 100.0,
            "width": 10.0, "source": "MEASURED",
        }
        b = {
            "start": 105.0, "end": 145.0, "center": 125.0,
            "width": 40.0, "source": "MEASURED",
        }
        o.on_trigger(1.0, 100.0, 280.0, w, b, used_latency_ms=120.0)
        for t, a in [
            (1.060, 82.0),
            (1.076, 82.2),
            (1.092, 81.9),
        ]:
            o.observe_sample(t, a, 40.0)
        result = o.conclude_check()
        self.assertEqual(result["outcome"], "MISS")
        self.assertTrue(result["plateau_trusted"])

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
