"""
Detector E: Minimal Color Index.
Evaluates minimal direct BGR channel thresholding without grayscale/HSV conversion.
Tests whether cv2.cvtColor() can be completely bypassed.
"""

import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from core.detectors.base import (
    BaseDetector,
    GeometryConfig,
    DEFAULT_GEOMETRY,
    SPACE_TEMPLATE,
    parabolic_peak,
    extract_zones_from_masks,
)


class MinimalColorDetector(BaseDetector):
    name: str = "MINIMAL_COLOR"

    def __init__(self, geometry: GeometryConfig = DEFAULT_GEOMETRY):
        self.geo = geometry
        self.cx = self.geo.center_x
        self.cy = self.geo.center_y

        tpl_f = SPACE_TEMPLATE.astype(np.float32)
        self.tpl_norm = tpl_f - np.mean(tpl_f)
        self.tpl_std = float(np.linalg.norm(self.tpl_norm))

        angles = np.arange(360, dtype=np.float32) * (np.pi / 180.0)
        cos_a = np.cos(angles)[:, None]
        sin_a = np.sin(angles)[:, None]

        needle_radii = np.linspace(24.0, 62.0, 8, dtype=np.float32)[None, :]
        x_ndl = np.clip(np.round(self.cx + needle_radii * cos_a).astype(np.int32), 0, self.geo.roi_width - 1)
        y_ndl = np.clip(np.round(self.cy + needle_radii * sin_a).astype(np.int32), 0, self.geo.roi_height - 1)
        self.needle_indices_1d = (y_ndl * self.geo.roi_width + x_ndl).astype(np.int32)

        ring_radii = np.linspace(63.0, 68.0, 6, dtype=np.float32)[None, :]
        x_ring = np.clip(np.round(self.cx + ring_radii * cos_a).astype(np.int32), 0, self.geo.roi_width - 1)
        y_ring = np.clip(np.round(self.cy + ring_radii * sin_a).astype(np.int32), 0, self.geo.roi_height - 1)
        self.ring_indices_1d = (y_ring * self.geo.roi_width + x_ring).astype(np.int32)

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

        # Pure BGR slice for template presence (green channel)
        patch_g = frame_bgr[self.geo.tpl_y0:self.geo.tpl_y1, self.geo.tpl_x0:self.geo.tpl_x1, 1].astype(np.float32)
        p_norm = patch_g - np.mean(patch_g)
        p_std = float(np.linalg.norm(p_norm))
        score = float(np.sum(p_norm * self.tpl_norm) / (p_std * self.tpl_std)) if p_std > 1e-5 else 0.0

        if score < 0.80:
            return None

        frame_flat = frame_bgr.reshape(-1, 3)

        # Needle ray sampling: integer arithmetic
        ndl_samples = frame_flat[self.needle_indices_1d]  # (360, 8, 3) uint8
        r_ch = ndl_samples[:, :, 2].astype(np.int16)
        g_ch = ndl_samples[:, :, 1].astype(np.int16)
        b_ch = ndl_samples[:, :, 0].astype(np.int16)
        max_gb = np.maximum(g_ch, b_ch)
        redness = np.maximum(0, r_ch - max_gb)
        red_profile = np.mean(redness, axis=1)

        peak_idx = int(np.argmax(red_profile))
        needle_strength = float(red_profile[peak_idx])
        needle_angle = parabolic_peak(red_profile, peak_idx)

        # Zone sampling
        ring_samples = frame_flat[self.ring_indices_1d].astype(np.int16)
        r66_val = (ring_samples[:, :, 0] + ring_samples[:, :, 1] + ring_samples[:, :, 2]) / 3.0
        r66_val = np.mean(r66_val, axis=1)

        ring_median = float(np.median(r66_val))
        th_white = min(185.0, max(160.0, ring_median + 40.0))
        th_black = max(42.0, min(55.0, ring_median - 40.0))

        r_mean = np.mean(ring_samples[:, :, 2], axis=1)
        g_mean = np.mean(ring_samples[:, :, 1], axis=1)
        b_mean = np.mean(ring_samples[:, :, 0], axis=1)

        white_mask = ((r66_val > th_white) & (r_mean > 150) & (g_mean > 150) & (b_mean > 150)) | (r66_val > 185)
        black_mask = (r66_val < th_black) | (r66_val < 42)

        is_needle_valid = needle_strength >= 15.0
        w_d, b_d = extract_zones_from_masks(white_mask, black_mask)

        t1 = time.perf_counter()
        det_time_ms = (t1 - t0) * 1000.0

        return {
            "confidence": score,
            "cx": self.cx,
            "cy": self.cy,
            "center": (self.cx, self.cy),
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
            "status": "OK" if is_needle_valid else "LOW_CONFIDENCE",
        }
