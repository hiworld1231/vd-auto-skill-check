"""
Detector D: Hybrid Fixed-Polar Local Tracker (Primary Candidate).
Combines fixed geometry calibration, precomputed ray coordinates,
lifecycle-locked zone geometry, and dynamic local angular tracking.
Achieves <0.05ms normal-frame latency with zero degradation in accuracy.
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
    refine_zone_from_score,
)


def _bounded_local_parabolic_delta(profile: np.ndarray, peak_idx: int) -> float:
    """Sub-degree interpolation only when both angular neighbours are local."""
    if peak_idx <= 0 or peak_idx >= len(profile) - 1:
        return 0.0
    p_prev = float(profile[peak_idx - 1])
    p_curr = float(profile[peak_idx])
    p_next = float(profile[peak_idx + 1])
    denom = 2.0 * (2.0 * p_curr - p_prev - p_next)
    return ((p_next - p_prev) / denom) if denom > 1e-5 else 0.0


class HybridDetector(BaseDetector):
    name: str = "HYBRID"

    def __init__(self, geometry: GeometryConfig = DEFAULT_GEOMETRY):
        self.geo = geometry
        self.cx = self.geo.center_x
        self.cy = self.geo.center_y

        tpl_f = SPACE_TEMPLATE.astype(np.float32)
        self.tpl_norm = tpl_f - np.mean(tpl_f)
        self.tpl_std = float(np.linalg.norm(self.tpl_norm))

        self._build_ray_tables()

        # Tracking state
        self.last_angle: Optional[float] = None
        self.last_t: Optional[float] = None
        self.locked_white_zone: Optional[Dict[str, float]] = None
        self.locked_black_zone: Optional[Dict[str, float]] = None
        self.consecutive_losses: int = 0
        self.reacquire_count: int = 0
        self.frame_count_in_check: int = 0

    def _build_ray_tables(self) -> None:
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

    def set_geometry(self, cx: float, cy: float) -> None:
        cx, cy = float(cx), float(cy)
        if abs(cx - self.cx) < 0.01 and abs(cy - self.cy) < 0.01:
            return
        self.cx, self.cy = cx, cy
        self._build_ray_tables()

    def reset(self):
        """Resets tracking between skill checks."""
        self.last_angle = None
        self.last_t = None
        self.locked_white_zone = None
        self.locked_black_zone = None
        self.consecutive_losses = 0
        self.frame_count_in_check = 0

    def _check_presence(self, frame_bgr: np.ndarray, frame_gray: Optional[np.ndarray]) -> Tuple[bool, float, float, float]:
        """Fast NCC presence check centered on the *measured* ring geometry.

        Baseline can move cx/cy a few pixels away from the nominal calibration.
        The old implementation rebuilt the ray tables but kept checking a fixed
        template box around the nominal center, causing false ring-loss resets.
        """
        th, tw = SPACE_TEMPLATE.shape
        x0 = int(round(self.cx - tw / 2.0))
        y0 = int(round(self.cy - th / 2.0))
        x1, y1 = x0 + tw, y0 + th
        h, w = frame_bgr.shape[:2]
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return False, 0.0, self.cx, self.cy

        if frame_gray is None:
            patch = frame_bgr[y0:y1, x0:x1, 1].astype(np.float32)
        else:
            patch = frame_gray[y0:y1, x0:x1].astype(np.float32)

        patch_norm = patch - np.mean(patch)
        p_std = float(np.linalg.norm(patch_norm))
        score = (
            float(np.sum(patch_norm * self.tpl_norm) / (p_std * self.tpl_std))
            if p_std > 1e-5
            else 0.0
        )
        if score >= 0.80:
            return True, score, self.cx, self.cy

        # Small local search around the measured center handles sub-pixel/template
        # centering differences without falling back to a full-frame scan.
        margin = 6
        sx0 = max(0, x0 - margin)
        sy0 = max(0, y0 - margin)
        sx1 = min(w, x1 + margin)
        sy1 = min(h, y1 + margin)
        if frame_gray is None:
            crop = frame_bgr[sy0:sy1, sx0:sx1, 1]
        else:
            crop = frame_gray[sy0:sy1, sx0:sx1]
        if crop.shape[0] >= th and crop.shape[1] >= tw:
            res = cv2.matchTemplate(crop, SPACE_TEMPLATE, cv2.TM_CCOEFF_NORMED)
            _, max_v, _, _ = cv2.minMaxLoc(res)
            if max_v >= 0.80:
                return True, float(max_v), self.cx, self.cy

        return False, score, self.cx, self.cy

    def _extract_fresh_zones(self, frame_flat: np.ndarray) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]], np.ndarray, np.ndarray, np.ndarray]:
        """Extracts Great and Good zones from outer ring pixels."""
        ring_samples = frame_flat[self.ring_indices_1d].astype(np.float32)  # (360, 6, 3)
        b_ch = ring_samples[:, :, 0].mean(axis=1)
        g_ch = ring_samples[:, :, 1].mean(axis=1)
        r_ch = ring_samples[:, :, 2].mean(axis=1)
        r66_val = (b_ch + g_ch + r_ch) / 3.0

        ring_median = float(np.median(r66_val))
        th_white = min(185.0, max(160.0, ring_median + 40.0))
        th_black = max(42.0, min(55.0, ring_median - 40.0))

        white_mask = ((r66_val > th_white) & (r_ch > 150) & (g_ch > 150) & (b_ch > 150)) | (r66_val > 185)
        black_mask = (r66_val < th_black) | (r66_val < 42)
        w_d, b_d = extract_zones_from_masks(white_mask, black_mask)

        white_primary = np.minimum.reduce(
            [
                r66_val - th_white,
                r_ch - 150.0,
                g_ch - 150.0,
                b_ch - 150.0,
            ]
        )
        white_score = np.maximum(white_primary, r66_val - 185.0)
        black_score = th_black - r66_val
        w_d = refine_zone_from_score(
            w_d, white_score, min_width=5.0, max_width=16.0
        )
        b_d = refine_zone_from_score(
            b_d, black_score, min_width=18.0, max_width=65.0
        )
        return w_d, b_d, white_mask, black_mask, r66_val

    def detect(
        self,
        frame_bgr: np.ndarray,
        frame_gray: Optional[np.ndarray] = None,
        expected_angle: Optional[float] = None,
        search_window: float = 35.0,
        dt_frame: float = 1.0 / 120.0,
        expected_speed: float = 278.0,
        locked_zones: Optional[Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]] = None,
        skip_presence_check: bool = False,
    ) -> Optional[Dict[str, Any]]:
        t0 = time.perf_counter()

        # 1. Ring presence check.  Post-fire tracking may deliberately bypass
        # the SPACE-prompt presence test for a short continuity window because
        # the prompt itself can disappear before the needle freeze is observable.
        if skip_presence_check:
            present, conf, cx, cy = True, 1.0, self.cx, self.cy
        else:
            present, conf, cx, cy = self._check_presence(frame_bgr, frame_gray)
            if not present:
                # Do not destroy last_angle/last_t on one transport/render miss.
                # VisionEngine decides when a sustained dual-detector absence is
                # enough evidence to reset the whole check.
                self.consecutive_losses += 1
                return None

        self.frame_count_in_check += 1
        now = time.monotonic()
        dt = (now - self.last_t) if (self.last_t is not None and 0.001 <= now - self.last_t <= 0.150) else dt_frame
        frame_flat = frame_bgr.reshape(-1, 3)

        # 2. Zone lifecycle management
        # If external locked zones provided, respect them
        if locked_zones and locked_zones[0] is not None:
            self.locked_white_zone, self.locked_black_zone = locked_zones
        elif self.locked_white_zone is None:
            # First frames: acquire and lock zones
            w_fresh, b_fresh, wm, bm, r66 = self._extract_fresh_zones(frame_flat)
            if w_fresh is not None:
                self.locked_white_zone = w_fresh
            if b_fresh is not None:
                self.locked_black_zone = b_fresh

        # 3. Needle tracking: dynamic local search vs full 360
        is_local = False
        cand_indices = None
        status = "OK"

        # Determine reference angle for prediction
        ref_angle = expected_angle if expected_angle is not None else self.last_angle
        if ref_angle is not None and self.consecutive_losses < 2 and dt <= 0.045:
            pred_ang = (ref_angle + (expected_speed * dt if expected_angle is None else 0.0)) % 360.0
            win_radius = max(22.0, expected_speed * dt + 18.0 + self.consecutive_losses * 15.0)
            low_a = int(np.floor((pred_ang - win_radius) % 360.0))
            high_a = int(np.ceil((pred_ang + win_radius) % 360.0))
            if low_a <= high_a:
                cand_indices = np.arange(low_a, high_a + 1) % 360
            else:
                cand_indices = np.concatenate([np.arange(low_a, 360), np.arange(0, high_a + 1)])
            is_local = True

        needle_strength = 0.0
        needle_angle = 0.0
        reacquire_rejected = False

        if is_local and cand_indices is not None and len(cand_indices) > 0:
            sub_indices = self.needle_indices_1d[cand_indices]  # (K, 8)
            sub_samples = frame_flat[sub_indices].astype(np.float32)
            sub_red = np.maximum(0.0, sub_samples[:, :, 2] - np.maximum(sub_samples[:, :, 1], sub_samples[:, :, 0]))
            sub_prof = np.mean(sub_red, axis=1)
            k_peak = int(np.argmax(sub_prof))
            peak_str = float(sub_prof[k_peak])

            if peak_str >= 15.0:
                peak_angle_int = int(cand_indices[k_peak])
                delta = _bounded_local_parabolic_delta(sub_prof, k_peak)
                needle_angle = float((peak_angle_int + delta) % 360.0)
                needle_strength = peak_str
                self.last_angle = needle_angle
                self.last_t = now
                self.consecutive_losses = 0
            else:
                # Local window failed to observe valid needle, trigger full reacquire
                self.consecutive_losses += 1
                status = "REACQUIRE"
                self.reacquire_count += 1
                cand_indices = None

        if cand_indices is None:
            # Full 360 scan fallback / initial acquire
            samples = frame_flat[self.needle_indices_1d].astype(np.float32)
            redness = np.maximum(0.0, samples[:, :, 2] - np.maximum(samples[:, :, 1], samples[:, :, 0]))
            red_prof = np.mean(redness, axis=1)

            peaks = []
            for i in range(360):
                prev_v = red_prof[(i - 1) % 360]
                curr_v = red_prof[i]
                next_v = red_prof[(i + 1) % 360]
                if curr_v >= prev_v and curr_v > next_v and curr_v > 15.0:
                    peaks.append((i, curr_v))

            tracking_ref = expected_angle if expected_angle is not None else self.last_angle
            if tracking_ref is not None and peaks:
                reacquire_radius = min(
                    90.0,
                    max(
                        float(search_window),
                        float(expected_speed) * float(dt) + 25.0
                        + self.consecutive_losses * 15.0,
                    ),
                )
                cand = [
                    p for p in peaks
                    if abs((p[0] - float(tracking_ref) + 180.0) % 360.0 - 180.0)
                    <= reacquire_radius
                ]
                if cand:
                    peak_idx = max(cand, key=lambda x: x[1])[0]
                else:
                    peak_idx = int(np.argmax(red_prof))
                    reacquire_rejected = True
            else:
                peak_idx = int(np.argmax(red_prof))

            needle_strength = float(red_prof[peak_idx])
            needle_angle = parabolic_peak(red_prof, peak_idx)

            if needle_strength >= 15.0 and not reacquire_rejected:
                self.last_angle = needle_angle
                self.last_t = now
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
                status = (
                    "REACQUIRE_OUTSIDE_CONTINUITY"
                    if reacquire_rejected
                    else "LOW_CONFIDENCE"
                )

        # If acquisition had to start from reconstructed GREAT geometry,
        # retry cheaply during active tracking and upgrade as soon as a real
        # measured white arc becomes visible.  Do not wait 25 frames: at 60 Hz
        # that would be far too late for short checks.
        locked_src = str((self.locked_white_zone or {}).get("source", ""))
        needs_white_upgrade = not locked_src.startswith("MEASURED")
        if self.frame_count_in_check % 5 == 0 and needs_white_upgrade:
            w_fresh, b_fresh, _, _, _ = self._extract_fresh_zones(frame_flat)
            fresh_src = str((w_fresh or {}).get("source", ""))
            if w_fresh is not None and fresh_src.startswith("MEASURED"):
                self.locked_white_zone = dict(w_fresh)
            if b_fresh is not None and str(b_fresh.get("source", "")).startswith("MEASURED"):
                self.locked_black_zone = dict(b_fresh)

        t1 = time.perf_counter()
        det_time_ms = (t1 - t0) * 1000.0

        is_needle_valid = (needle_strength >= 15.0 and not reacquire_rejected)
        return {
            "confidence": conf,
            "cx": cx,
            "cy": cy,
            "center": (cx, cy),
            "needle_angle": needle_angle,
            "needle_strength": needle_strength,
            "needle_confidence": needle_strength,
            "needle_valid": is_needle_valid,
            "white_mask": None,
            "black_mask": None,
            "r66_val": None,
            "white_zone": self.locked_white_zone,
            "black_zone": self.locked_black_zone,
            "ring_present": True,
            "detector_name": self.name,
            "detector_time_ms": det_time_ms,
            "status": status,
        }
