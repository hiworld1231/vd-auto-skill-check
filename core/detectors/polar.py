"""
Detector A: Fixed Polar Ring.
Precomputes angular lookup tables for needle and ring pixels based on calibrated center geometry.
Eliminates cv2.warpPolar and full-frame searching.
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


class FixedPolarDetector(BaseDetector):
    name: str = "FIXED_POLAR"

    def __init__(self, geometry: GeometryConfig = DEFAULT_GEOMETRY):
        self.geo = geometry
        self.cx = self.geo.center_x
        self.cy = self.geo.center_y

        # Precompute template normalized array for ultra-fast 1-position NCC
        tpl_f = SPACE_TEMPLATE.astype(np.float32)
        self.tpl_norm = tpl_f - np.mean(tpl_f)
        self.tpl_std = float(np.linalg.norm(self.tpl_norm))

        # Precompute coordinate grids
        angles = np.arange(360, dtype=np.float32) * (np.pi / 180.0)
        cos_a = np.cos(angles)[:, None]  # (360, 1)
        sin_a = np.sin(angles)[:, None]  # (360, 1)

        # Needle radii: 16 radial samples from 22px to 63px
        needle_radii = np.linspace(self.geo.needle_r_min, self.geo.needle_r_max, 16, dtype=np.float32)[None, :]
        x_ndl = np.clip(np.round(self.cx + needle_radii * cos_a).astype(np.int32), 0, self.geo.roi_width - 1)
        y_ndl = np.clip(np.round(self.cy + needle_radii * sin_a).astype(np.int32), 0, self.geo.roi_height - 1)
        self.needle_indices_1d = (y_ndl * self.geo.roi_width + x_ndl).astype(np.int32)

        # Ring radii: 6 samples from 63px to 68px
        ring_radii = np.linspace(self.geo.ring_r_min, self.geo.ring_r_max - 1.0, 6, dtype=np.float32)[None, :]
        x_ring = np.clip(np.round(self.cx + ring_radii * cos_a).astype(np.int32), 0, self.geo.roi_width - 1)
        y_ring = np.clip(np.round(self.cy + ring_radii * sin_a).astype(np.int32), 0, self.geo.roi_height - 1)
        self.ring_indices_1d = (y_ring * self.geo.roi_width + x_ring).astype(np.int32)

    def _check_ring_presence(self, frame_bgr: np.ndarray, frame_gray: Optional[np.ndarray]) -> Tuple[bool, float, float, float]:
        """
        Fast template check. Returns (present, confidence, cx, cy).
        Checks fixed position first (< 0.03ms). If slightly off, falls back to tight crop.
        """
        if frame_gray is None:
            # Direct green channel as proxy for grayscale (avoids full cvtColor)
            patch = frame_bgr[self.geo.tpl_y0:self.geo.tpl_y1, self.geo.tpl_x0:self.geo.tpl_x1, 1].astype(np.float32)
        else:
            patch = frame_gray[self.geo.tpl_y0:self.geo.tpl_y1, self.geo.tpl_x0:self.geo.tpl_x1].astype(np.float32)

        patch_norm = patch - np.mean(patch)
        p_std = float(np.linalg.norm(patch_norm))
        score = float(np.sum(patch_norm * self.tpl_norm) / (p_std * self.tpl_std)) if p_std > 1e-5 else 0.0

        if score >= 0.80:
            return True, score, self.cx, self.cy

        if score >= 0.65:
            # Tight 9x9 crop around expected location
            if frame_gray is None:
                crop = frame_bgr[141:184, 121:199, 1]
            else:
                crop = frame_gray[141:184, 121:199]
            res = cv2.matchTemplate(crop, SPACE_TEMPLATE, cv2.TM_CCOEFF_NORMED)
            _, max_v, _, max_l = cv2.minMaxLoc(res)
            if max_v >= 0.80:
                cx = float(121 + max_l[0] + self.geo.tpl_x1 - self.geo.tpl_x0) / 2.0
                cy = float(141 + max_l[1] + self.geo.tpl_y1 - self.geo.tpl_y0) / 2.0
                return True, float(max_v), self.cx, self.cy

        return False, score, self.cx, self.cy

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

        # 1. Ring presence check
        present, conf, cx, cy = self._check_ring_presence(frame_bgr, frame_gray)
        if not present:
            return None

        frame_flat = frame_bgr.reshape(-1, 3)

        # 2. Needle detection via precomputed 1D radial indexing
        ndl_samples = frame_flat[self.needle_indices_1d].astype(np.float32)  # (360, 16, 3)
        r_ch = ndl_samples[:, :, 2]
        g_ch = ndl_samples[:, :, 1]
        b_ch = ndl_samples[:, :, 0]
        redness = np.maximum(0.0, r_ch - np.maximum(g_ch, b_ch))
        red_profile = np.mean(redness, axis=1)

        # Spark rejection
        peaks = []
        for i in range(360):
            prev_v = red_profile[(i - 1) % 360]
            curr_v = red_profile[i]
            next_v = red_profile[(i + 1) % 360]
            if curr_v >= prev_v and curr_v > next_v and curr_v > 15.0:
                peaks.append((i, curr_v))

        if expected_angle is not None and peaks:
            cand = [p for p in peaks if abs((p[0] - expected_angle + 180.0) % 360.0 - 180.0) <= search_window]
            if cand:
                peak_idx = max(cand, key=lambda x: x[1])[0]
            else:
                peak_idx = int(np.argmax(red_profile))
        else:
            peak_idx = int(np.argmax(red_profile))

        needle_strength = float(red_profile[peak_idx])
        needle_angle = parabolic_peak(red_profile, peak_idx)

        # 3. Zone detection via precomputed ring indexing
        ring_samples = frame_flat[self.ring_indices_1d].astype(np.float32)  # (360, 6, 3)
        b_ring = ring_samples[:, :, 0].mean(axis=1)
        g_ring = ring_samples[:, :, 1].mean(axis=1)
        r_ring = ring_samples[:, :, 2].mean(axis=1)
        r66_val = (b_ring + g_ring + r_ring) / 3.0

        ring_median = float(np.median(r66_val))
        th_white = min(185.0, max(160.0, ring_median + 40.0))
        th_black = max(42.0, min(55.0, ring_median - 40.0))

        white_mask = ((r66_val > th_white) & (r_ring > 150) & (g_ring > 150) & (b_ring > 150)) | (r66_val > 185)
        black_mask = (r66_val < th_black) | (r66_val < 42)

        is_needle_valid = needle_strength >= 15.0
        w_d, b_d = extract_zones_from_masks(white_mask, black_mask)

        t1 = time.perf_counter()
        det_time_ms = (t1 - t0) * 1000.0

        return {
            "confidence": conf,
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
            "status": "OK" if is_needle_valid else "LOW_CONFIDENCE",
        }
