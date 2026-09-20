"""
Contour / WarpPolar Baseline Detector.
Wraps the existing OpenCV template-matching and cv2.warpPolar pipeline for A/B testing and fallback.
"""

import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from core.detectors.base import (
    BaseDetector,
    SPACE_TEMPLATE,
    TEMPLATE_H,
    TEMPLATE_W,
    parabolic_peak,
    extract_zones_from_masks,
)


class BaselineDetector(BaseDetector):
    name: str = "CONTOUR_BASELINE"

    def __init__(self):
        self.template = SPACE_TEMPLATE
        self.tpl_h, self.tpl_w = TEMPLATE_H, TEMPLATE_W

    def detect(
        self,
        frame_bgr: np.ndarray,
        frame_gray: Optional[np.ndarray] = None,
        expected_angle: Optional[float] = None,
        search_window: float = 35.0,
        dt_frame: float = 1.0 / 120.0,
        expected_speed: float = 278.0,
        locked_zones: Optional[Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]] = None,
    ) -> Optional[Dict[str, Any]]:
        t0 = time.perf_counter()

        if frame_gray is None:
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Fast Center ROI matching (0.08 ms vs 1.2 ms full frame)
        roi_y0, roi_y1 = 110, 215
        roi_x0, roi_x1 = 100, 220
        crop = frame_gray[roi_y0:roi_y1, roi_x0:roi_x1]
        res = cv2.matchTemplate(crop, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_l = cv2.minMaxLoc(res)

        if max_v >= 0.80:
            cx = float(roi_x0 + max_l[0] + self.tpl_w / 2.0)
            cy = float(roi_y0 + max_l[1] + self.tpl_h / 2.0)
        else:
            # Full-frame fallback
            res_full = cv2.matchTemplate(frame_gray, self.template, cv2.TM_CCOEFF_NORMED)
            _, max_v, _, max_l = cv2.minMaxLoc(res_full)
            if max_v < 0.80:
                return None
            cx = float(max_l[0] + self.tpl_w / 2.0)
            cy = float(max_l[1] + self.tpl_h / 2.0)

        # 2. Polar unroll: cols=radius (0 to 78), rows=angle (0 to 360)
        max_r = 78
        polar = cv2.warpPolar(frame_bgr, (max_r, 360), (cx, cy), max_r, cv2.WARP_POLAR_LINEAR)

        # 3. Needle detection (radial ray from r=22 to r=64)
        needle_crop = polar[:, 22:64]
        r_ch = needle_crop[:, :, 2].astype(np.float32)
        g_ch = needle_crop[:, :, 1].astype(np.float32)
        b_ch = needle_crop[:, :, 0].astype(np.float32)
        redness = r_ch - np.maximum(g_ch, b_ch)
        red_profile = np.mean(np.maximum(redness, 0), axis=1)

        # Local peak search for spark rejection
        peaks = []
        for i in range(360):
            prev_v = red_profile[(i - 1) % 360]
            curr_v = red_profile[i]
            next_v = red_profile[(i + 1) % 360]
            if curr_v >= prev_v and curr_v > next_v and curr_v > 15.0:
                peaks.append((i, curr_v))

        outside_expected_window = False
        if expected_angle is not None and peaks:
            cand = [
                p for p in peaks
                if abs((p[0] - expected_angle + 180) % 360 - 180) <= search_window
            ]
            if cand:
                peak_idx = max(cand, key=lambda x: x[1])[0]
            else:
                peak_idx = int(np.argmax(red_profile))
                outside_expected_window = True
        else:
            peak_idx = int(np.argmax(red_profile))

        needle_strength = float(red_profile[peak_idx])
        needle_angle = parabolic_peak(red_profile, peak_idx)

        # 4. Success zone detection on outer ring (r=63 to r=69 for maximum SNR)
        ring_crop = polar[:, 63:69].astype(np.float32)
        r66_b = np.mean(ring_crop[:, :, 0], axis=1)
        r66_g = ring_crop[:, :, 1].mean(axis=1)
        r66_r = ring_crop[:, :, 2].mean(axis=1)
        r66_val = (r66_b + r66_g + r66_r) / 3.0

        ring_median = float(np.median(r66_val))
        th_white = min(185.0, max(160.0, ring_median + 40.0))
        th_black = max(42.0, min(55.0, ring_median - 40.0))

        white_mask = ((r66_val > th_white) & (r66_r > 150) & (r66_g > 150) & (r66_b > 150)) | (r66_val > 185)
        black_mask = (r66_val < th_black) | (r66_val < 42)

        is_needle_valid = needle_strength >= 15.0 and not outside_expected_window
        w_d, b_d = extract_zones_from_masks(white_mask, black_mask)

        t1 = time.perf_counter()
        det_time_ms = (t1 - t0) * 1000.0

        return {
            "confidence": float(max_v),
            "cx": cx,
            "cy": cy,
            "center": (cx, cy),
            "needle_angle": needle_angle,
            "needle_strength": needle_strength,
            "needle_confidence": needle_strength,
            "needle_valid": is_needle_valid,
            "white_mask": white_mask,
            "black_mask": black_mask,
            "r66_val": r66_val,
            "white_zone": w_d,
            "black_zone": b_d,
            "ring_present": True,
            "detector_name": self.name,
            "detector_time_ms": det_time_ms,
            "status": (
                "OK"
                if is_needle_valid
                else (
                    "OUTSIDE_EXPECTED_WINDOW"
                    if outside_expected_window
                    else "LOW_CONFIDENCE"
                )
            ),
        }
