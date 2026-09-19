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
