"""
Base definitions, geometry configuration, and shared geometric helpers for skill check detectors.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
TEMPLATE_PATH = ROOT / "space_template.png"


def load_space_template() -> np.ndarray:
    if not TEMPLATE_PATH.is_file():
        raise FileNotFoundError(
            f"Required detector template is missing: {TEMPLATE_PATH}. "
            "Restore space_template.png next to skillcheck_bot.py."
        )
    image = cv2.imread(str(TEMPLATE_PATH), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        raise RuntimeError(f"Detector template is unreadable/corrupt: {TEMPLATE_PATH}")
    return image


SPACE_TEMPLATE = load_space_template()
TEMPLATE_H, TEMPLATE_W = SPACE_TEMPLATE.shape


@dataclass(frozen=True)
class GeometryConfig:
    """
    Fixed calibrated geometry of the skill check on screen.
    Derived from screen resolution 320x240 with ROI +800+420.
    """
    center_x: float = 160.0
    center_y: float = 162.5
    roi_width: int = 320
    roi_height: int = 240
    needle_r_min: float = 22.0
    needle_r_max: float = 63.0
    ring_r_min: float = 63.0
    ring_r_max: float = 69.0
    ring_thickness: float = 6.0
    # Space prompt template box: (y0, y1, x0, x1) in 240x320 ROI
    tpl_y0: int = 145
    tpl_y1: int = 180
    tpl_x0: int = 125
    tpl_x1: int = 195


DEFAULT_GEOMETRY = GeometryConfig()


def is_angle_in_arc(angle: float, start: float, end: float, tol_start: float = 2.0, tol_end: float = 2.0) -> bool:
    """Checks if an angle is within [start - tol_start, end + tol_end], handling 360/0 wrap-around."""
    s = (start - tol_start) % 360.0
    e = (end + tol_end) % 360.0
    a = angle % 360.0
    if s <= e:
        return s <= a <= e
    return a >= s or a <= e


def parabolic_peak(profile: np.ndarray, peak_idx: int) -> float:
    """
    Computes sub-degree needle angle via parabolic interpolation around the peak.
    """
    n = len(profile)
    p_prev = float(profile[(peak_idx - 1) % n])
    p_curr = float(profile[peak_idx])
    p_next = float(profile[(peak_idx + 1) % n])
    denom = 2.0 * (2.0 * p_curr - p_prev - p_next)
    if denom > 1e-5:
        delta = (p_next - p_prev) / denom
        return float((peak_idx + delta) % float(n))
    return float(peak_idx)


def extract_runs(mask: np.ndarray, min_len: int, max_len: int) -> List[Tuple[float, float, float]]:
    """
    Extracts contiguous true runs from a 360-element circular boolean array.
    Returns list of (start_deg, end_deg, length_deg).
    """
    doubled = np.concatenate([mask, mask])
    runs = []
    start = None
    for i in range(len(doubled)):
        if doubled[i] and start is None:
            start = i
        elif not doubled[i] and start is not None:
            length = i - start
            if start < 360 and min_len <= length <= max_len:
                s_deg = start % 360
                e_deg = (start + length - 1) % 360
                runs.append((float(s_deg), float(e_deg), float(length)))
            start = None
    return runs


def extract_zones_from_masks(
    white_mask: Optional[np.ndarray],
    black_mask: Optional[np.ndarray],
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
    """
    High-precision extraction of Great (white) and Good (black) zone boundaries.
    Fully compatible with baseline VisionEngine logic.
    """
    w_dict = None
    b_dict = None

    if white_mask is not None:
        runs = extract_runs(white_mask, min_len=5, max_len=15)
        if runs:
            runs.sort(key=lambda x: x[2], reverse=True)
            w_start, w_end, w_len = runs[0]
            w_center = (w_start + w_len / 2.0) % 360.0
            w_dict = {
                "start": float(w_start),
                "end": float(w_end),
                "width": float(w_len),
                "center": float(w_center),
                "source": "MEASURED",
            }

    if black_mask is not None:
        b_runs = extract_runs(black_mask, min_len=18, max_len=65)
        if w_dict is not None and b_runs:
            valid_b = [r for r in b_runs if abs((r[0] - w_dict["end"]) % 360) <= 8]
            if valid_b:
                b_s, b_e, b_l = valid_b[0]
                b_dict = {
                    "start": float(b_s),
                    "end": float(b_e),
                    "width": float(b_l),
                    "center": float((b_s + b_l / 2.0) % 360),
                    "source": "MEASURED",
                }

        # Physical fallback if white is clear but black is shadowed
        if w_dict is not None and b_dict is None:
            b_s = (w_dict["end"] + 1.0) % 360
            b_dict = {
                "start": float(b_s),
                "end": float((b_s + 42.0) % 360),
                "width": 42.0,
                "center": float((b_s + 21.0) % 360),
                "source": "RECONSTRUCTED_FROM_WHITE",
            }

        # Occlusion fallback: if white was occluded, deduce Great zone from Good zone
        if w_dict is None and b_runs:
            b_runs.sort(key=lambda x: x[2], reverse=True)
            b_s, b_e, b_l = b_runs[0]
            w_s = (b_s - 9.5) % 360.0
            w_dict = {
                "start": float(w_s),
                "end": float(b_s),
                "width": 9.5,
                "center": float((w_s + 4.75) % 360),
                "source": "RECONSTRUCTED_FROM_BLACK",
            }
            b_dict = {
                "start": float(b_s),
                "end": float(b_e),
                "width": float(b_l),
                "center": float((b_s + b_l / 2.0) % 360),
                "source": "MEASURED",
            }

    return w_dict, b_dict


class BaseDetector:
    """
    Unified Abstract Interface for Skill Check Detection Backends.
    """
    name: str = "BASE"

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
        """
        Executes detection on a frame.
        Returns a dict conforming to the unified detection schema, or None if no check is present.
        """
        raise NotImplementedError

    def reset(self):
        """Resets tracking state between checks."""
        pass

    def extract_zones(
        self,
        det: Optional[Dict[str, Any]],
        locked_zones: Optional[Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]] = None,
    ) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        """Extracts Great and Good zones from detection or returns locked zones."""
        if det is None:
            return locked_zones or (None, None)
        if locked_zones and locked_zones[0] is not None:
            return locked_zones
        w_d = det.get("white_zone")
        b_d = det.get("black_zone")
        if w_d is not None or b_d is not None:
            return w_d, b_d
        return extract_zones_from_masks(det.get("white_mask"), det.get("black_mask"))
