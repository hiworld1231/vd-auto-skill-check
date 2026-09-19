"""
High-Performance Computer Vision Engine for Violent District Skill Checks.

Production Architecture: BASELINE_ZONE + HYBRID_NEEDLE
- BASELINE is used exclusively during SPAWN_ACQUIRE for:
  * Ring presence verification
  * Center sanity checks
  * Lifecycle/spawn confirmation
  * White and black zone acquisition and locking
- Once zones are confidently locked, critical production path switches to HYBRID:
  * High-speed needle tracking (<0.08 ms)
  * Dynamic local angular search and sub-degree interpolation
  * Lifecycle-locked zone geometry caching (no warpPolar on active frames)
- Robust fallback:
  * If HYBRID loses needle confidence across consecutive frames, invokes BASELINE fallback
  * Explicitly logs HYBRID_NEEDLE_FALLBACK_BASELINE
  * Never emits random or unconfirmed needle angles
  * Automatically resets and reacquires on frenzy chains or ring disappearance
"""

from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from core.detectors import (
    BaseDetector,
    DETECTOR_REGISTRY,
    get_detector,
    SPACE_TEMPLATE,
    TEMPLATE_H,
    TEMPLATE_W,
    is_angle_in_arc,
)
from core.detectors.baseline import BaselineDetector
from core.detectors.hybrid import HybridDetector

ROOT = Path(__file__).resolve().parent.parent

STATE_SPAWN_ACQUIRE = "SPAWN_ACQUIRE"
STATE_ACTIVE_TRACKING = "ACTIVE_TRACKING"


class VisionEngine:
    """
    Production Computer Vision Engine: BASELINE_ZONE + HYBRID_NEEDLE.
    Orchestrates high-precision baseline zone acquisition with ultra-low-latency
    hybrid needle tracking on the critical path.
    """

    def __init__(self, backend_name: str = "hybrid"):
        self.backend_name = (backend_name or "hybrid").lower().strip().replace("-", "_")
        self.template = SPACE_TEMPLATE
        self.tpl_h, self.tpl_w = TEMPLATE_H, TEMPLATE_W

        self.baseline_detector: Optional[BaselineDetector] = None
        self.hybrid_detector: Optional[HybridDetector] = None
        self.backend: Optional[BaseDetector] = None

        self.state: str = STATE_SPAWN_ACQUIRE
        self.locked_white_zone: Optional[Dict[str, float]] = None
        self.locked_black_zone: Optional[Dict[str, float]] = None
        self.locked_center: Optional[Tuple[float, float]] = None
        self.consecutive_hybrid_losses: int = 0
        self.is_pressed: bool = False

        self._configure_backends()

    def _configure_backends(self):
        if self.backend_name in ("hybrid", "baseline_zone_hybrid_needle", "default"):
            self.is_orchestrated = True
            self.baseline_detector = BaselineDetector()
            self.hybrid_detector = HybridDetector()
            self.backend = None
        else:
            self.is_orchestrated = False
            self.baseline_detector = None
            self.hybrid_detector = None
            self.backend = get_detector(self.backend_name)

    def set_backend(self, name: str):
        """Switches the active detection backend."""
        self.backend_name = (name or "hybrid").lower().strip().replace("-", "_")
        self._configure_backends()
        self.reset()

    def reset(self):
        """Resets tracker state between skill checks and for new check generations."""
        self.state = STATE_SPAWN_ACQUIRE
        self.locked_white_zone = None
        self.locked_black_zone = None
        self.locked_center = None
        self.consecutive_hybrid_losses = 0
        self.is_pressed = False
        if self.baseline_detector is not None:
            self.baseline_detector.reset()
        if self.hybrid_detector is not None:
            self.hybrid_detector.reset()
        if self.backend is not None:
            self.backend.reset()

    def reset_generation(self, preserve_center: bool = True):
        """Reset a Frenzy/new-generation trajectory without pretending the whole ring vanished.

        The new generation must reacquire its zone geometry, but the already validated
        ring centre can be retained as a diagnostic hint.
        """
        center = self.locked_center if preserve_center else None
        self.state = STATE_SPAWN_ACQUIRE
        self.locked_white_zone = None
        self.locked_black_zone = None
        self.locked_center = center
        self.consecutive_hybrid_losses = 0
        self.is_pressed = False
        if self.baseline_detector is not None:
            self.baseline_detector.reset()
        if self.hybrid_detector is not None:
            self.hybrid_detector.reset()
        if self.backend is not None:
            self.backend.reset()

    def notify_pressed(self):
        """Signals that the current skill check has been fired."""
        self.is_pressed = True

    def detect_frame(
        self,
        frame_gray: Optional[np.ndarray],
        frame_bgr: np.ndarray,
        expected_angle: Optional[float] = None,
        search_window: float = 35.0,
        dt_frame: float = 1.0 / 120.0,
        expected_speed: float = 278.0,
        locked_zones: Optional[Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]] = None,
        is_pressed: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Processes a single video frame.
        In BASELINE_ZONE + HYBRID_NEEDLE mode:
        - Executes BASELINE during SPAWN_ACQUIRE until zones and center are locked.
        - Executes HYBRID during ACTIVE_TRACKING on the critical decision path.
        - Falls back to BASELINE if HYBRID needle confidence drops across >= 2 frames.
        """
        if is_pressed is not None:
            self.is_pressed = is_pressed

        if not self.is_orchestrated:
            return self.backend.detect(
                frame_bgr=frame_bgr,
                frame_gray=frame_gray,
                expected_angle=expected_angle,
                search_window=search_window,
                dt_frame=dt_frame,
                expected_speed=expected_speed,
                locked_zones=locked_zones,
            )

        # -------------------------------------------------------------
        # 1. Post-Fire / Pressed state (monitoring for frenzy or ring end)
        # -------------------------------------------------------------
        if self.is_pressed:
            det_base = self.baseline_detector.detect(
                frame_bgr=frame_bgr,
                frame_gray=frame_gray,
                expected_angle=expected_angle,
                search_window=search_window,
                dt_frame=dt_frame,
                expected_speed=expected_speed,
            )
            if det_base is None or not det_base.get("ring_present"):
                self.reset()
                return None
            if det_base.get("white_zone") is not None:
                det_base["white_zone"] = dict(det_base["white_zone"])
                det_base["white_zone"].setdefault("source", "MEASURED_POST_HIT")
            if det_base.get("black_zone") is not None:
                det_base["black_zone"] = dict(det_base["black_zone"])
                det_base["black_zone"].setdefault("source", "MEASURED_POST_HIT")
            det_base["detector_name"] = "BASELINE_POST_HIT"
            return det_base

        # -------------------------------------------------------------
        # 2. SPAWN_ACQUIRE state (BASE presence, center sanity, zone lock)
        # -------------------------------------------------------------
        if self.state == STATE_SPAWN_ACQUIRE or (self.locked_white_zone is None and self.locked_black_zone is None):
            det_base = self.baseline_detector.detect(
                frame_bgr=frame_bgr,
                frame_gray=frame_gray,
                expected_angle=expected_angle,
                search_window=search_window,
                dt_frame=dt_frame,
                expected_speed=expected_speed,
            )
            if det_base is None or not det_base.get("ring_present"):
                return None

            # Center sanity check
            cx, cy = det_base["