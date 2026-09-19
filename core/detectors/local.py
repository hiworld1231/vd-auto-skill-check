"""
Detector C: Local Angle Tracker.
Restricts needle search to a dynamic local window after initial acquire.
Adapts window size based on frame dt and angular velocity, expanding and reacquiring on signal loss.
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


class LocalTrackerDetector(BaseDetector):
    name: str = "LOCAL_TRACKER"

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
        self.needle_indices = (y_ndl * self.geo.roi_width + x_ndl).astype(np.int32)

        ring_radii = np.linspace(64.0, 68.0, 4, dtype=np.float32)[None, :]
        x_ring = np.clip(np.round(self.cx + ring_radii * cos_a).astype(np.int32), 0, self.geo.roi_width - 1)
        y_ring = np.clip(np.round(self.cy + ring_radii * sin_a).astype(np.int32), 0, self.geo.roi_height - 1)
        self.ring_indices = (y_ring * self.geo.roi_width + x_ring).astype(np.int32)

        # Tracker state
        self.last_angle: Optional[float] = None
        self.last_t: Optional[float] = None
        self.consecutive_losses: int = 0
        self.reacquire_count: int = 0
        self.cached_white_zone: Optional[Dict[str, float]] = None
        self.cached_black_zone: Optional[Dict[str, float]] = None

    def reset(self):
        self.last_angle = None
        self.last_t = None
        self.consecutive_losses = 0
        self.cached_white_zone = None
        self.cached_black_zone = None

    def _check_presence(self, frame_bgr: np.ndarray) -> Tuple[bool, float]:
        patch = frame_bgr[self.geo.tpl_y0:self.geo.tpl_y1, self.geo.tpl_x0:self.geo.tpl_x1, 1].astype(np.float32)
        patch_norm = patch - np.mean(patch)
        p_std = float(np.linalg.norm(patch_norm))
        score = float(np.sum(patch_norm * self.tpl_norm) / (p_std * self.tpl_std)) if p_std > 1e-5 else 0.0
        return (score >= 0.80), score

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

        present, conf = self._check_presence(frame_bgr)
        if not present:
            self.reset()
            return None

        frame_flat = frame_bgr.reshape(-1, 3)
        now = time.monotonic()
        dt = (now - self.last_t) if (self.last_t is not None and 0.001 <= now - self.last_t <= 0.150) else dt_frame

        # Decide candidate angles: local window vs full 360
        is_local = False
        cand_indices = None
        if self.last_angle is not None and self.consecutive_losses < 2:
            pred_ang = (self.last_angle + expected_speed * dt) % 360.0
            win_radius = max(20.0, expected_speed * dt + 15.0 + self.consecutive_losses * 15.0)
            low_a = int(np.floor((pred_ang - win_radius) % 360.0))
            high_a = int(np.ceil((pred_ang + win_radius) % 360.0))
            if low_a <= high_a:
                cand_indices = np.arange(low_a, high_a + 1) % 360
            else:
                cand_indices = np.concatenate([np.arange(low_a, 360), np.arange(0, high_a + 1)])
            is_local = True

        status = "OK"
        if is_local and cand_indices is not None and len(cand_indices) > 0:
            sub_indices = self.needle_indices[cand_indices]  # shape (K, 8)
            sub_samples = frame_flat[sub_indices].astype(np.float32)
            sub_red = np.maximum(0.0, sub_samples[:, :, 2] - np.maximum(sub_samples[:, :, 1], sub_samples[:, :, 0]))
            sub_prof = np.mean(sub_red, axis=1)
            k_peak = int(np.argmax(sub_prof))
            peak_str = float(sub_prof[k_peak])

            if peak_str >= 15.0:
                peak_angle_int = int(cand_indices[k_peak])
                # Sub-degree parabolic fit on candidate neighbor indices
                idx_prev = (k_peak - 1) % len(sub_prof)
                idx_next = (k_peak + 1) % len(sub_prof)
                p_prev = float(sub_prof[idx_prev])
                p_curr = peak_str
                p_next = float(sub_prof[idx_next])
                denom = 2.0 * (2.0 * p_curr - p_prev - p_next)
                delta = ((p_next - p_prev) / denom) if denom > 1e-5 else 0.0
                needle_angle = float((peak_angle_int + delta) % 360.0)
                needle_strength = peak_str
                self.last_angle = needle_angle
                self.last_t = now
                self.consecutive_losses = 0
            else:
                # Local search failed: step 1 expand/reacquire
                self.consecutive_losses += 1
                status = "REACQUIRE"
                self.reacquire_count += 1
                # Trigger full scan
                cand_indices = None

        if cand_indices is None:
            # Full 360 ray scan
            samples = frame_flat[self.needle_indices].astype(np.float32)
            redness = np.maximum(0.0, samples[:, :, 2] - np.maximum(samples[:, :, 1], samples[:, :, 0]))
            red_prof = np.mean(redness, axis=1)
            peak_idx = int(np.argmax(red_prof))
            needle_strength = float(red_prof[peak_idx])
            needle_angle = parabolic_peak(red_prof, peak_idx)

            if needle_strength >= 15.0:
                self.last_angle = needle_angle
                self.last_t = now
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
                status = "LOW_CONFIDENCE"

        # Zone detection: perform on initial acquire or refresh cached zones
        w_d = locked_zones[0] if locked_zones else self.cached_white_zone
        b_d = locked_zones[1] if locked_zones else self.cached_black_zone
        white_mask = None
        black_mask = None
        r66_val = None

        if w_d is None or b_d is None:
            ring_s = frame_flat[self.ring_indices].astype(np.float32)
            b_ring = ring_s[:, :, 0].mean(axis=1)
            g_ring = ring_s[:, :, 1].mean(axis=1)
            r_ring = ring_s[:, :, 2].mean(axis=1)
            r66_val = (b_ring + g_ring + r_ring) / 3.0

            ring_median = float(np.median(r66_val))
            th_white = min(185.0, max(160.0, ring_median + 40.0))
            th_black = max(42.0, min(55.0, ring_median - 40.0))

            white_mask = ((r66_val > th_white) & (r_ring > 150) & (g_ring > 150) & (b_ring > 150)) | (r66_val > 185)
            black_mask = (r66_val < th_black) | (r66_val < 42)
            fresh_w, fresh_b = extract_zones_from_masks(white_mask, black_mask)
            if fresh_w is not None:
                self.cached_white_zone = fresh_w
                w_d = fresh_w
            if fresh_b is not None:
                self.cached_black_zone = fresh_b
                b_d = fresh_b

        t1 = time.perf_counter()
        det_time_ms = (t1 - t0) * 1000.0

        return {
            "confidence": conf,
            "cx": self.cx,
            "cy": self.cy,
            "center": (self.cx, self.cy),
            "needle_angle": needle_angle,
            "needle_strength": needle_strength,
            "needle_confidence": needle_strength,
            "needle_valid": (needle_strength >= 15.0),
            "white_mask": white_mask,
            "black_mask": black_mask,
            "r66_val": r66_val,
            "white_zone": w_d,
            "black_zone": b_d,
            "ring_present": True,
            "detector_name": self.name,
            "detector_time_ms": det_time_ms,
            "status": status,
        }
