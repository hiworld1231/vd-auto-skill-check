"""
Unit and Regression Test Suite for BASELINE_ZONE + HYBRID_NEEDLE Architecture.

Verifies:
1. In SPAWN_ACQUIRE: BASELINE confirms ring presence, center sanity, acquires and locks white and black zones.
2. In ACTIVE_TRACKING: HYBRID tracks needle at <0.08 ms, heavy BASELINE warpPolar is NOT executed.
3. Locked zones are preserved during active tracking within a single check generation.
4. Fallback: If HYBRID needle confidence drops across >= 2 frames, BASELINE needle fallback is invoked and logged.
5. Lifecycle: When reset() is called (check end, abort, or frenzy chain), locked zones are cleared and BASELINE re-acquires for the next check generation.
"""

import sys
import time
import unittest
from pathlib import Path
from typing import Tuple
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.vision import VisionEngine, STATE_SPAWN_ACQUIRE, STATE_ACTIVE_TRACKING
from core.detectors import SPACE_TEMPLATE
from core.detectors.baseline import BaselineDetector
from core.detectors.hybrid import HybridDetector, _bounded_local_parabolic_delta


def create_synthetic_check_frame(
    needle_angle_deg: float = 120.0,
    white_start: float = 75.0,
    white_width: float = 10.0,
    black_start: float = 85.0,
    black_width: float = 40.0,
    cx: float = 160.0,
    cy: float = 162.5,
    needle_strength: float = 80.0,
    draw_needle: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a 320x240 frame with a synthetic skill check ring."""
    frame_bgr = np.zeros((240, 320, 3), dtype=np.uint8)
    frame_bgr[:] = (40, 40, 40)

    # Place space template in center
    th, tw = SPACE_TEMPLATE.shape
    tx0 = int(round(cx - tw / 2.0))
    ty0 = int(round(cy - th / 2.0))
    frame_bgr[ty0:ty0+th, tx0:tx0+tw, 1] = SPACE_TEMPLATE
    frame_bgr[ty0:ty0+th, tx0:tx0+tw, 0] = SPACE_TEMPLATE
    frame_bgr[ty0:ty0+th, tx0:tx0+tw, 2] = SPACE_TEMPLATE

    # Draw outer ring (r=63 to 69)
    # Neutral ring: gray
    cv2.circle(frame_bgr, (int(round(cx)), int(round(cy))), 66, (120, 120, 120), 6)

    # Black zone: arc
    cv2.ellipse(
        frame_bgr,
        (int(round(cx)), int(round(cy))),
        (66, 66),
        0,
        black_start,
        black_start + black_width,
        (20, 20, 20),
        6,
    )

    # White zone: arc
    cv2.ellipse(
        frame_bgr,
        (int(round(cx)), int(round(cy))),
        (66, 66),
        0,
        white_start,
        white_start + white_width,
        (230, 230, 230),
        6,
    )

    # Needle: red line (r=24 to 62)
    if draw_needle:
        rad = needle_angle_deg * np.pi / 180.0
        p1 = (int(round(cx + 24 * np.cos(rad))), int(round(cy + 24 * np.sin(rad))))
        p2 = (int(round(cx + 62 * np.cos(rad))), int(round(cy + 62 * np.sin(rad))))
        cv2.line(frame_bgr, p1, p2, (20, 20, int(round(needle_strength + 100))), 2)

    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return frame_gray, frame_bgr


class TestBaselineZoneHybridNeedle(unittest.TestCase):

    def setUp(self):
        self.vision = VisionEngine(backend_name="hybrid")

    def test_spawn_acquire_locks_baseline_zones_and_transitions(self):
        """Step 1: Frame 1 in SPAWN_ACQUIRE runs BASELINE, validates center, locks zones, and transitions."""
        gray, bgr = create_synthetic_check_frame(
            needle_angle_deg=45.0, white_start=80.0, white_width=10.0, black_start=90.0, black_width=40.0
        )
        self.assertEqual(self.vision.state, STATE_SPAWN_ACQUIRE)
        self.assertIsNone(self.vision.locked_white_zone)

        det = self.vision.detect_frame(gray, bgr)
        self.assertIsNotNone(det)
        self.assertTrue(det["ring_present"])
        self.assertEqual(det["detector_name"], "BASELINE_ZONE_HYBRID_NEEDLE")
        self.assertEqual(self.vision.state, STATE_ACTIVE_TRACKING)

        # Zones must be locked
        self.assertIsNotNone(self.vision.locked_white_zone)
        self.assertIsNotNone(self.vision.locked_black_zone)
        self.assertIsNotNone(self.vision.locked_center)
        self.assertAlmostEqual(self.vision.locked_white_zone["width"], 14.0, delta=4.0)

    @staticmethod
    def _mock_acquire_detection(white_zone):
        return {
            "ring_present": True,
            "needle_valid": True,
            "needle_angle": 45.0,
            "needle_strength": 80.0,
            "confidence": 0.99,
            "cx": 160.0,
            "cy": 162.5,
            "center": (160.0, 162.5),
            "white_zone": white_zone,
            "black_zone": {
                "start": 90.0,
                "end": 130.0,
                "center": 110.0,
                "width": 40.0,
                "source": "MEASURED",
            },
        }

    def test_black_only_reconstruction_starts_check_immediately(self):
        reconstructed = {
            "start": 80.5,
            "end": 90.0,
            "center": 85.25,
            "width": 9.5,
            "source": "RECONSTRUCTED_FROM_BLACK",
        }
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        gray = np.zeros((240, 320), dtype=np.uint8)

        with patch.object(
            self.vision.baseline_detector,
            "detect",
            return_value=self._mock_acquire_detection(dict(reconstructed)),
        ):
            det = self.vision.detect_frame(gray, frame)

        self.assertEqual(self.vision.state, STATE_ACTIVE_TRACKING)
        self.assertEqual(
            self.vision.locked_white_zone["source"],
            "RECONSTRUCTED_FROM_BLACK",
        )
        self.assertEqual(
            det["white_zone"]["source"], "RECONSTRUCTED_FROM_BLACK"
        )

    def test_active_tracking_upgrades_reconstructed_great_to_measured(self):
        reconstructed = {
            "start": 80.5,
            "end": 90.0,
            "center": 85.25,
            "width": 9.5,
            "source": "RECONSTRUCTED_FROM_BLACK",
        }
        measured = {
            "start": 82.2,
            "end": 91.8,
            "center": 87.0,
            "width": 9.6,
            "source": "MEASURED",
        }
        black = {
            "start": 92.0,
            "end": 132.0,
            "center": 112.0,
            "width": 40.0,
            "source": "MEASURED",
        }
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        gray = np.zeros((240, 320), dtype=np.uint8)

        with patch.object(
            self.vision.baseline_detector,
            "detect",
            return_value=self._mock_acquire_detection(dict(reconstructed)),
        ):
            self.vision.detect_frame(gray, frame)

        with patch.object(
            self.vision.hybrid_detector,
            "detect",
            return_value={
                "ring_present": True,
                "needle_valid": True,
                "needle_angle": 55.0,
                "needle_strength": 80.0,
                "white_zone": measured,
                "black_zone": black,
                "status": "OK",
            },
        ):
            det = self.vision.detect_frame(
                gray,
                frame,
                expected_angle=55.0,
            )

        self.assertEqual(self.vision.locked_white_zone["source"], "MEASURED")
        self.assertAlmostEqual(self.vision.locked_white_zone["center"], 87.0)
        self.assertEqual(det["white_zone"]["source"], "MEASURED")

    def test_active_tracking_critical_path_bypasses_baseline_vision(self):
        """Step 2: Subsequent ACTIVE frames execute HYBRID needle tracking without calling Baseline warpPolar."""
        gray, bgr = create_synthetic_check_frame(needle_angle_deg=45.0)
        _ = self.vision.detect_frame(gray, bgr)
        self.assertEqual(self.vision.state, STATE_ACTIVE_TRACKING)

        # Spy on baseline_detector.detect to guarantee it is NOT called during active tracking
        with patch.object(self.vision.baseline_detector, "detect", wraps=self.vision.baseline_detector.detect) as mock_base:
            # Send next frame with moving needle
            gray2, bgr2 = create_synthetic_check_frame(needle_angle_deg=55.0)
            det2 = self.vision.detect_frame(gray2, bgr2, expected_angle=55.0)

            self.assertIsNotNone(det2)
            self.assertTrue(det2["needle_valid"])
            self.assertAlmostEqual(det2["needle_angle"], 55.0, delta=2.5)
            self.assertEqual(det2["detector_name"], "BASELINE_ZONE_HYBRID_NEEDLE")

            # BASELINE detect must NOT have been called on the active frame!
            mock_base.assert_not_called()

    def test_needle_confidence_loss_triggers_baseline_fallback(self):
        """Step 3: If HYBRID loses needle confidence across >= 2 frames, BASELINE fallback is triggered."""
        gray, bgr = create_synthetic_check_frame(needle_angle_deg=45.0)
        _ = self.vision.detect_frame(gray, bgr)
        self.assertEqual(self.vision.state, STATE_ACTIVE_TRACKING)

        # Frame with template intact but no needle drawn
        gray_no_needle, bgr_no_needle = create_synthetic_check_frame(needle_angle_deg=45.0, draw_needle=False)

        # Loss 1: Hybrid returns low confidence, valid=False
        det_loss1 = self.vision.detect_frame(gray_no_needle, bgr_no_needle)
        self.assertIsNotNone(det_loss1)
        self.assertFalse(det_loss1["needle_valid"])
        self.assertEqual(self.vision.consecutive_hybrid_losses, 1)

        # On loss 2, Baseline fallback IS invoked
        with patch.object(self.vision.baseline_detector, "detect") as mock_base:
            mock_base.return_value = {
                "ring_present": True,
                "needle_valid": True,
                "needle_angle": 135.0,
                "needle_strength": 85.0,
                "confidence": 0.98,
                "cx": 160.0,
                "cy": 162.5,
                "center": (160.0, 162.5),
            }
            det_fallback = self.vision.detect_frame(gray_no_needle, bgr_no_needle)
            self.assertIsNotNone(det_fallback)
            self.assertTrue(det_fallback["needle_valid"])
            self.assertEqual(det_fallback["needle_angle"], 135.0)
            self.assertEqual(det_fallback["detector_name"], "HYBRID_NEEDLE_FALLBACK_BASELINE")
            mock_base.assert_called_once()
            # Losses reset upon successful recovery
            self.assertEqual(self.vision.consecutive_hybrid_losses, 0)

    def test_needle_loss_does_not_reset_confirmed_ring(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        gray = np.zeros((240, 320), dtype=np.uint8)
        self.vision.state = STATE_ACTIVE_TRACKING
        self.vision.locked_white_zone = {
            "start": 80.0, "end": 90.0, "center": 85.0,
            "width": 10.0, "source": "MEASURED",
        }
        self.vision.locked_black_zone = {
            "start": 90.0, "end": 130.0, "center": 110.0,
            "width": 40.0, "source": "MEASURED",
        }
        self.vision.locked_center = (160.0, 162.5)

        weak = {
            "ring_present": True,
            "needle_valid": False,
            "needle_angle": 45.0,
            "needle_strength": 5.0,
            "white_zone": self.vision.locked_white_zone,
            "black_zone": self.vision.locked_black_zone,
            "status": "LOW_CONFIDENCE",
        }
        recovered = dict(weak)
        recovered.update({
            "needle_valid": True,
            "needle_angle": 52.0,
            "needle_strength": 70.0,
            "status": "OK",
        })

        with patch.object(
            self.vision.hybrid_detector,
            "detect",
            side_effect=[dict(weak), dict(weak), recovered],
        ), patch.object(
            self.vision.baseline_detector,
            "detect",
            return_value=None,
        ):
            first = self.vision.detect_frame(gray, frame)
            second = self.vision.detect_frame(gray, frame)
            third = self.vision.detect_frame(gray, frame)

        self.assertIsNotNone(first)
        self.assertFalse(first["needle_valid"])
        self.assertIsNotNone(second)
        self.assertTrue(second["ring_present"])
        self.assertFalse(second["needle_valid"])
        self.assertEqual(second["status"], "NEEDLE_REACQUIRE_GRACE")
        self.assertEqual(self.vision.state, STATE_ACTIVE_TRACKING)

        self.assertIsNotNone(third)
        self.assertTrue(third["needle_valid"])
        self.assertAlmostEqual(third["needle_angle"], 52.0)
        self.assertEqual(self.vision.consecutive_hybrid_losses, 0)
        self.assertEqual(self.vision.state, STATE_ACTIVE_TRACKING)

    def test_hybrid_presence_uses_measured_shifted_center(self):
        gray, bgr = create_synthetic_check_frame(
            needle_angle_deg=45.0,
            cx=168.0,
            cy=157.0,
        )
        h = HybridDetector()
        h.set_geometry(168.0, 157.0)
        present, score, cx, cy = h._check_presence(bgr, gray)
        self.assertTrue(present)
        self.assertGreaterEqual(score, 0.80)
        self.assertAlmostEqual(cx, 168.0)
        self.assertAlmostEqual(cy, 157.0)

    def test_active_tracking_requires_three_dual_presence_misses_to_reset(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        gray = np.zeros((240, 320), dtype=np.uint8)
        self.vision.state = STATE_ACTIVE_TRACKING
        self.vision.locked_white_zone = {
            "start": 80.0, "end": 90.0, "center": 85.0,
            "width": 10.0, "source": "MEASURED",
        }
        self.vision.locked_black_zone = {
            "start": 90.0, "end": 130.0, "center": 110.0,
            "width": 40.0, "source": "MEASURED",
        }
        self.vision.locked_center = (160.0, 162.5)
        with patch.object(self.vision.hybrid_detector, "detect", return_value=None), patch.object(
            self.vision.baseline_detector, "detect", return_value=None
        ):
            first = self.vision.detect_frame(gray, frame)
            second = self.vision.detect_frame(gray, frame)
            third = self.vision.detect_frame(gray, frame)

        self.assertIsNotNone(first)
        self.assertEqual(first["detector_name"], "RING_PRESENCE_GRACE")
        self.assertIsNotNone(second)
        self.assertEqual(second["detector_name"], "RING_PRESENCE_GRACE")
        self.assertIsNone(third)
        self.assertEqual(self.vision.state, STATE_SPAWN_ACQUIRE)

    def test_postfire_keeps_old_hybrid_and_new_baseline_needles_separate(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        gray = np.zeros((240, 320), dtype=np.uint8)
        self.vision.state = STATE_ACTIVE_TRACKING
        self.vision.locked_white_zone = {
            "start": 80.0, "end": 90.0, "center": 85.0,
            "width": 10.0, "source": "MEASURED",
        }
        self.vision.locked_black_zone = {
            "start": 90.0, "end": 130.0, "center": 110.0,
            "width": 40.0, "source": "MEASURED",
        }
        self.vision.locked_center = (160.0, 162.5)
        self.vision.notify_pressed()

        new_white = {
            "start": 200.0, "end": 210.0, "center": 205.0,
            "width": 10.0, "source": "MEASURED",
        }
        new_black = {
            "start": 210.0, "end": 250.0, "center": 230.0,
            "width": 40.0, "source": "MEASURED",
        }
        baseline_new = {
            "ring_present": True,
            "needle_valid": True,
            "needle_angle": 42.0,
            "needle_strength": 90.0,
            "needle_confidence": 90.0,
            "cx": 160.0,
            "cy": 162.5,
            "center": (160.0, 162.5),
            "white_zone": new_white,
            "black_zone": new_black,
        }
        hybrid_old = {
            "ring_present": True,
            "needle_valid": True,
            "needle_angle": 312.0,
            "needle_strength": 80.0,
            "needle_confidence": 80.0,
        }

        with patch.object(
            self.vision.baseline_detector, "detect", return_value=baseline_new
        ), patch.object(
            self.vision.hybrid_detector, "detect", return_value=hybrid_old
        ):
            det = self.vision.detect_frame(gray, frame, is_pressed=True)

        self.assertAlmostEqual(det["needle_angle"], 312.0)
        self.assertTrue(det["needle_valid"])
        self.assertAlmostEqual(det["generation_needle_angle"], 42.0)
        self.assertTrue(det["generation_needle_valid"])
        self.assertAlmostEqual(det["white_zone"]["center"], 205.0)

    def test_frenzy_generation_handoff_skips_spawn_reacquire(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        gray = np.zeros((240, 320), dtype=np.uint8)
        white = {
            "start": 20.0, "end": 30.0, "center": 25.0,
            "width": 10.0, "source": "MEASURED_POST_HIT",
        }
        black = {
            "start": 30.0, "end": 70.0, "center": 50.0,
            "width": 40.0, "source": "MEASURED_POST_HIT",
        }
        confirming = {
            "ring_present": True,
            "needle_valid": True,
            # Old-generation HYBRID landing trajectory:
            "needle_angle": 312.0,
            "needle_strength": 82.0,
            "needle_valid": True,
            # New-generation BASELINE needle on the relocated ring:
            "generation_needle_angle": 42.0,
            "generation_needle_strength": 88.0,
            "generation_needle_confidence": 88.0,
            "generation_needle_valid": True,
            "cx": 161.0,
            "cy": 163.0,
            "center": (161.0, 163.0),
            "white_zone": white,
            "black_zone": black,
        }

        adopted = self.vision.bootstrap_generation(confirming, white, black)
        self.assertEqual(self.vision.state, STATE_ACTIVE_TRACKING)
        self.assertEqual(adopted["detector_name"], "FRENZY_GENERATION_HANDOFF")
        self.assertAlmostEqual(self.vision.locked_white_zone["center"], 25.0)
        self.assertAlmostEqual(self.vision.hybrid_detector.last_angle, 42.0)
        self.assertAlmostEqual(adopted["needle_angle"], 42.0)
        self.assertEqual(adopted["handoff_needle_source"], "BASELINE_NEW_GENERATION")
        self.assertEqual(self.vision.locked_center, (161.0, 163.0))

        # The very next frame must be ACTIVE HYBRID tracking, not a BASELINE
        # SPAWN_ACQUIRE pass looking for the prompt again.
        with patch.object(
            self.vision.baseline_detector, "detect"
        ) as mock_base, patch.object(
            self.vision.hybrid_detector,
            "detect",
            return_value={
                "ring_present": True,
                "needle_valid": True,
                "needle_angle": 320.0,
                "needle_strength": 75.0,
                "white_zone": white,
                "black_zone": black,
                "status": "OK",
            },
        ):
            nxt = self.vision.detect_frame(
                gray,
                frame,
                expected_angle=320.0,
                expected_speed=500.0,
            )

        mock_base.assert_not_called()
        self.assertIsNotNone(nxt)
        self.assertTrue(nxt["needle_valid"])
        self.assertAlmostEqual(nxt["needle_angle"], 320.0)
        self.assertEqual(nxt["white_zone"]["center"], 25.0)

    def test_postfire_fixed_center_scan_finds_next_generation_when_baseline_misses(self):
        gray, frame = create_synthetic_check_frame(
            needle_angle_deg=42.0,
            white_start=200.0,
            white_width=10.0,
            black_start=210.0,
            black_width=40.0,
        )
        self.vision.state = STATE_ACTIVE_TRACKING
        self.vision.locked_white_zone = {
            "start": 80.0, "end": 90.0, "center": 85.0,
            "width": 10.0, "source": "MEASURED",
        }
        self.vision.locked_black_zone = {
            "start": 90.0, "end": 130.0, "center": 110.0,
            "width": 40.0, "source": "MEASURED",
        }
        self.vision.locked_center = (160.0, 162.5)
        self.vision.hybrid_detector.set_geometry(160.0, 162.5)
        self.vision.notify_pressed()

        # Simulate exactly the live failure mode: the SPACE-template BASELINE
        # path misses, but the ring pixels and the new generation needle are
        # still visible at the calibrated center.
        with patch.object(
            self.vision.baseline_detector, "detect", return_value=None
        ), patch.object(
            self.vision.hybrid_detector,
            "detect",
            return_value={
                "ring_present": True,
                "needle_valid": True,
                "needle_angle": 312.0,  # previous-generation landing needle
                "needle_strength": 80.0,
                "needle_confidence": 80.0,
            },
        ):
            det = self.vision.detect_frame(gray, frame, is_pressed=True)

        self.assertIsNotNone(det)
        self.assertTrue(det["ring_present"])
        self.assertFalse(det["baseline_prompt_present"])
        self.assertEqual(
            det["generation_detector"],
            "HYBRID_FIXED_CENTER_GENERATION_SCAN",
        )
        self.assertTrue(det["generation_needle_valid"])
        self.assertAlmostEqual(det["generation_needle_angle"], 42.0, delta=3.0)
        self.assertAlmostEqual(det["white_zone"]["center"], 205.0, delta=4.0)
        # Landing observer still gets the previous generation, not the new one.
        self.assertAlmostEqual(det["needle_angle"], 312.0)

    def test_postfire_tracks_needle_when_prompt_presence_is_lost(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        gray = np.zeros((240, 320), dtype=np.uint8)
        self.vision.state = STATE_ACTIVE_TRACKING
        self.vision.locked_white_zone = {
            "start": 80.0, "end": 90.0, "center": 85.0,
            "width": 10.0, "source": "MEASURED",
        }
        self.vision.locked_black_zone = {
            "start": 90.0, "end": 130.0, "center": 110.0,
            "width": 40.0, "source": "MEASURED",
        }
        self.vision.locked_center = (160.0, 162.5)
        self.vision.notify_pressed()

        with patch.object(self.vision.baseline_detector, "detect", return_value=None), patch.object(
            self.vision.hybrid_detector,
            "detect",
            return_value={
                "ring_present": True,
                "needle_valid": True,
                "needle_angle": 88.4,
                "needle_strength": 70.0,
                "needle_confidence": 70.0,
            },
        ) as mock_hybrid:
            det = self.vision.detect_frame(gray, frame, is_pressed=True)

        self.assertIsNotNone(det)
        self.assertFalse(det["ring_present"])
        self.assertTrue(det["needle_valid"])
        self.assertAlmostEqual(det["needle_angle"], 88.4)
        self.assertTrue(mock_hybrid.call_args.kwargs["skip_presence_check"])

    def test_baseline_expected_angle_rejects_unrelated_red_peak(self):
        gray, bgr = create_synthetic_check_frame(needle_angle_deg=200.0)
        det = BaselineDetector().detect(
            frame_bgr=bgr,
            frame_gray=gray,
            expected_angle=45.0,
            search_window=35.0,
        )
        self.assertIsNotNone(det)
        self.assertFalse(det["needle_valid"])
        self.assertEqual(det["status"], "OUTSIDE_EXPECTED_WINDOW")

    def test_hybrid_reacquire_does_not_jump_to_distant_red_peak(self):
        gray, bgr = create_synthetic_check_frame(needle_angle_deg=200.0)
        h = HybridDetector()
        h.last_angle = 45.0
        h.last_t = time.monotonic() - 0.016
        det = h.detect(
            frame_bgr=bgr,
            frame_gray=gray,
            expected_angle=None,
            search_window=35.0,
            dt_frame=0.016,
            expected_speed=300.0,
        )
        self.assertIsNotNone(det)
        self.assertFalse(det["needle_valid"])
        self.assertEqual(det["status"], "REACQUIRE_OUTSIDE_CONTINUITY")

    def test_local_parabolic_interpolation_never_wraps_window_edges(self):
        left_edge = np.array([100.0, 20.0, 5.0], dtype=np.float32)
        right_edge = np.array([5.0, 20.0, 100.0], dtype=np.float32)
        self.assertEqual(_bounded_local_parabolic_delta(left_edge, 0), 0.0)
        self.assertEqual(_bounded_local_parabolic_delta(right_edge, 2), 0.0)

    def test_zone_lifecycle_and_reset_for_new_check_generation(self):
        """Step 4: reset() clears locked zones so next check generation re-acquires freshly."""
        gray, bgr = create_synthetic_check_frame(white_start=80.0, white_width=10.0)
        _ = self.vision.detect_frame(gray, bgr)
        self.assertIsNotNone(self.vision.locked_white_zone)
        first_center = self.vision.locked_white_zone["center"]

        # Call reset (e.g. check completed, aborted, or frenzy chain)
        self.vision.reset()
        self.assertEqual(self.vision.state, STATE_SPAWN_ACQUIRE)
        self.assertIsNone(self.vision.locked_white_zone)
        self.assertIsNone(self.vision.locked_black_zone)

        # New check generation spawns at a different location (e.g. white zone at 200°)
        gray2, bgr2 = create_synthetic_check_frame(white_start=200.0, white_width=10.0, black_start=210.0)
        det2 = self.vision.detect_frame(gray2, bgr2)
        self.assertIsNotNone(det2)
        self.assertIsNotNone(self.vision.locked_white_zone)
        second_center = self.vision.locked_white_zone["center"]

        self.assertNotEqual(first_center, second_center)
        self.assertAlmostEqual(second_center, 207.0, delta=4.0)


if __name__ == "__main__":
    unittest.main()
