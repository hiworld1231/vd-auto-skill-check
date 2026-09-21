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
        self.consecutive_ring_losses: int = 0
        self.is_pressed: bool = False
        # A black-only frame can reconstruct a plausible GREAT arc, but locking
        # it immediately throws away the chance to measure the real white arc
        # on the next source frame.  Wait briefly before falling back.
        self.black_only_acquire_count: int = 0
        self.black_only_fallback_frames: int = 3

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
        self.consecutive_ring_losses = 0
        self.black_only_acquire_count = 0
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
        self.consecutive_ring_losses = 0
        self.black_only_acquire_count = 0
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

    def reseed_postfire_tracking(self, angle: Optional[float]) -> None:
        """Restore HYBRID to the last trusted old-generation phase.

        A rejected post-fire red candidate must not become the detector's new
        reference on the next frame.
        """
        if (
            angle is None
            or not self.is_orchestrated
            or self.hybrid_detector is None
        ):
            return
        self.hybrid_detector.last_angle = float(angle) % 360.0
        self.hybrid_detector.last_t = time.monotonic()
        self.hybrid_detector.consecutive_losses = 0

    def bootstrap_generation(
        self,
        det: Dict[str, Any],
        white_zone: Dict[str, float],
        black_zone: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Adopt the already-observed next Frenzy generation without reacquiring.

        The post-fire frame that confirms Frenzy already contains the relocated
        ring geometry. Throwing that frame away and returning to SPAWN_ACQUIRE
        can lose very fast generations before the SPACE template is visible
        again. This method turns the confirming frame directly into ACTIVE
        tracking state.
        """
        if not self.is_orchestrated:
            raise RuntimeError("generation bootstrap requires orchestrated vision")
        if not isinstance(det, dict):
            raise ValueError("bootstrap detection is required")
        if white_zone is None:
            raise ValueError("bootstrap white zone is required")

        center = det.get("center")
        cx = det.get("cx")
        cy = det.get("cy")
        if center is not None and (cx is None or cy is None):
            cx, cy = center
        if cx is None or cy is None:
            if self.locked_center is None:
                raise ValueError("bootstrap center is required")
            cx, cy = self.locked_center

        cx = float(cx)
        cy = float(cy)
        self.state = STATE_ACTIVE_TRACKING
        self.locked_white_zone = dict(white_zone)
        self.locked_black_zone = dict(black_zone) if black_zone is not None else None
        self.locked_center = (cx, cy)
        self.consecutive_hybrid_losses = 0
        self.consecutive_ring_losses = 0
        self.black_only_acquire_count = 0
        self.is_pressed = False

        if self.baseline_detector is not None:
            self.baseline_detector.reset()
        if self.hybrid_detector is not None:
            self.hybrid_detector.reset()
            self.hybrid_detector.set_geometry(cx, cy)
            self.hybrid_detector.locked_white_zone = self.locked_white_zone
            self.hybrid_detector.locked_black_zone = self.locked_black_zone

            generation_valid = bool(det.get("generation_needle_valid"))
            if generation_valid:
                bootstrap_angle = det.get("generation_needle_angle")
                bootstrap_strength = det.get("generation_needle_strength")
                bootstrap_confidence = det.get("generation_needle_confidence")
            else:
                bootstrap_angle = det.get("needle_angle")
                bootstrap_strength = det.get("needle_strength")
                bootstrap_confidence = det.get("needle_confidence")

            if bootstrap_angle is not None and (
                generation_valid or det.get("needle_valid", True)
            ):
                self.hybrid_detector.last_angle = float(bootstrap_angle)
                self.hybrid_detector.last_t = time.monotonic()
                self.hybrid_detector.consecutive_losses = 0

        out = dict(det)
        if bool(det.get("generation_needle_valid")):
            out["needle_angle"] = float(det["generation_needle_angle"])
            out["needle_strength"] = float(det.get("generation_needle_strength") or 0.0)
            out["needle_confidence"] = float(
                det.get("generation_needle_confidence")
                or det.get("generation_needle_strength")
                or 0.0
            )
            out["needle_valid"] = True
            out["handoff_needle_source"] = "BASELINE_NEW_GENERATION"
        out["ring_present"] = True
        out["white_zone"] = self.locked_white_zone
        out["black_zone"] = self.locked_black_zone
        out["cx"] = cx
        out["cy"] = cy
        out["center"] = (cx, cy)
        out["detector_name"] = "FRENZY_GENERATION_HANDOFF"
        return out

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
            # The SPACE prompt/lifecycle template can disappear immediately
            # after keydown, before the needle freeze is visible.  Keep landing
            # motion independent from prompt presence: BASELINE supplies ring/
            # zone lifecycle, HYBRID continues the old needle trajectory without
            # requiring the prompt to remain on-screen.
            det_base = self.baseline_detector.detect(
                frame_bgr=frame_bgr,
                frame_gray=frame_gray,
                expected_angle=expected_angle,
                search_window=search_window,
                dt_frame=dt_frame,
                expected_speed=expected_speed,
            )
            det_generation = self.hybrid_detector.scan_generation_candidate(
                frame_bgr
            )
            det_hyb = self.hybrid_detector.detect(
                frame_bgr=frame_bgr,
                frame_gray=frame_gray,
                expected_angle=expected_angle,
                search_window=search_window,
                dt_frame=dt_frame,
                expected_speed=expected_speed,
                locked_zones=(self.locked_white_zone, self.locked_black_zone),
                skip_presence_check=True,
                strict_continuity=True,
            )

            # Lifecycle/new-generation evidence must not depend only on
            # the SPACE prompt template. Prefer a prompt-independent scan at
            # the already calibrated center; fall back to BASELINE when that
            # scan cannot recover sane ring geometry.
            lifecycle_det = det_generation if det_generation is not None else det_base
            if lifecycle_det is not None:
                out = dict(lifecycle_det)
                out["ring_present"] = bool(lifecycle_det.get("ring_present", True))
                out["generation_needle_angle"] = lifecycle_det.get("needle_angle")
                out["generation_needle_strength"] = lifecycle_det.get("needle_strength")
                out["generation_needle_confidence"] = lifecycle_det.get("needle_confidence")
                out["generation_needle_valid"] = bool(
                    lifecycle_det.get("needle_valid")
                    and float(lifecycle_det.get("needle_strength", 0.0) or 0.0) >= 15.0
                )
                out["generation_detector"] = lifecycle_det.get("detector_name")
                out["baseline_prompt_present"] = bool(det_base is not None)
            else:
                out = {
                    "ring_present": False,
                    "confidence": 0.0,
                    "white_zone": None,
                    "black_zone": None,
                    "cx": self.locked_center[0] if self.locked_center else self.hybrid_detector.cx,
                    "cy": self.locked_center[1] if self.locked_center else self.hybrid_detector.cy,
                    "center": self.locked_center,
                    "generation_needle_angle": None,
                    "generation_needle_strength": 0.0,
                    "generation_needle_confidence": 0.0,
                    "generation_needle_valid": False,
                    "generation_detector": None,
                    "baseline_prompt_present": False,
                }

            if det_hyb is not None and det_hyb.get("needle_valid") and float(det_hyb.get("needle_strength", 0.0) or 0.0) >= 15.0:
                out["needle_angle"] = det_hyb["needle_angle"]
                out["needle_strength"] = det_hyb.get("needle_strength")
                out["needle_confidence"] = det_hyb.get("needle_confidence")
                out["needle_valid"] = True
                out["detector_name"] = "POST_HIT_HYBRID_NEEDLE_GENERATION_LIFECYCLE"
            else:
                # Never substitute a fresh BASELINE full-scan red peak as the
                # landing angle.
                out["needle_valid"] = False
                out["needle_strength"] = 0.0
                out["detector_name"] = "POST_HIT_LIFECYCLE_ONLY"

            if out.get("white_zone") is not None:
                out["white_zone"] = dict(out["white_zone"])
                out["white_zone"].setdefault("source", "MEASURED_POST_HIT")
            if out.get("black_zone") is not None:
                out["black_zone"] = dict(out["black_zone"])
                out["black_zone"].setdefault("source", "MEASURED_POST_HIT")
            return out

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
            cx, cy = det_base["cx"], det_base["cy"]
            if not (140.0 <= cx <= 180.0 and 142.5 <= cy <= 182.5):
                return None

            # Confident zone acquisition check.  "Valid width" is not the
            # same as "measured": extract_zones_from_masks can intentionally
            # reconstruct a 9.5° white arc from a measured black arc.
            w_d = det_base.get("white_zone")
            b_d = det_base.get("black_zone")
            w_valid = (w_d is not None and 5.0 <= w_d.get("width", 0.0) <= 16.0)
            b_valid = (b_d is not None and 18.0 <= b_d.get("width", 0.0) <= 65.0)

            if w_valid:
                w_d = dict(w_d)
                w_d.setdefault("source", "MEASURED")
            if b_valid:
                b_d = dict(b_d)
                b_d.setdefault("source", "MEASURED")

            # Do not burn 2-3 render frames waiting for ideal white geometry:
            # short checks can pass GREAT before the predictor even gets enough
            # motion samples.  A reconstructed white arc is immediately playable
            # and remains explicitly marked as reconstructed downstream.
            should_lock = bool(w_valid or b_valid)
            if not w_valid and b_valid:
                w_s = (b_d["start"] - 9.5) % 360.0
                w_d = {
                    "start": float(w_s),
                    "end": float(b_d["start"]),
                    "width": 9.5,
                    "center": float((w_s + 4.75) % 360.0),
                    "source": "RECONSTRUCTED_FROM_BLACK",
                }
            self.black_only_acquire_count = 0

            if should_lock:
                self.locked_white_zone = w_d
                self.locked_black_zone = b_d
                self.locked_center = (cx, cy)
                self.state = STATE_ACTIVE_TRACKING
                self.consecutive_hybrid_losses = 0
                self.black_only_acquire_count = 0

                # Initialize Hybrid detector state with baseline locked parameters
                self.hybrid_detector.reset()
                self.hybrid_detector.set_geometry(cx, cy)
                self.hybrid_detector.last_angle = det_base["needle_angle"]
                self.hybrid_detector.last_t = time.monotonic()
                self.hybrid_detector.consecutive_losses = 0
                self.hybrid_detector.locked_white_zone = self.locked_white_zone
                self.hybrid_detector.locked_black_zone = self.locked_black_zone

            det_base["detector_name"] = "BASELINE_ZONE_HYBRID_NEEDLE"
            if self.locked_white_zone is not None:
                det_base["white_zone"] = self.locked_white_zone
                det_base["black_zone"] = self.locked_black_zone
            return det_base

        # -------------------------------------------------------------
        # 3. ACTIVE_TRACKING state (Critical Path: HYBRID needle tracking)
        # -------------------------------------------------------------
        det_hyb = self.hybrid_detector.detect(
            frame_bgr=frame_bgr,
            frame_gray=frame_gray,
            expected_angle=expected_angle,
            search_window=search_window,
            dt_frame=dt_frame,
            expected_speed=expected_speed,
            locked_zones=(self.locked_white_zone, self.locked_black_zone),
        )

        if det_hyb is None or not det_hyb.get("ring_present"):
            # Do not kill an active check on one strict HYBRID presence miss.
            # Confirm with the slower BASELINE detector first; only sustained
            # dual-detector absence ends the generation.
            fallback_expected = (
                expected_angle
                if expected_angle is not None
                else self.hybrid_detector.last_angle
            )
            det_base = self.baseline_detector.detect(
                frame_bgr=frame_bgr,
                frame_gray=frame_gray,
                expected_angle=fallback_expected,
                search_window=max(float(search_window), 60.0),
                dt_frame=dt_frame,
                expected_speed=expected_speed,
            )
            if det_base is not None and det_base.get("ring_present"):
                self.consecutive_ring_losses = 0
                det_base["white_zone"] = self.locked_white_zone
                det_base["black_zone"] = self.locked_black_zone
                det_base["detector_name"] = "HYBRID_PRESENCE_FALLBACK_BASELINE"
                if det_base.get("needle_valid"):
                    self.hybrid_detector.last_angle = det_base.get("needle_angle")
                    self.hybrid_detector.last_t = time.monotonic()
                    self.hybrid_detector.consecutive_losses = 0
                return det_base

            self.consecutive_ring_losses += 1
            if self.consecutive_ring_losses < 3:
                return {
                    "ring_present": True,
                    "needle_valid": False,
                    "needle_strength": 0.0,
                    "white_zone": self.locked_white_zone,
                    "black_zone": self.locked_black_zone,
                    "cx": self.locked_center[0],
                    "cy": self.locked_center[1],
                    "center": self.locked_center,
                    "detector_name": "RING_PRESENCE_GRACE",
                    "status": "PRESENCE_RECHECK",
                    "confidence": 0.0,
                }

            self.reset()
            return None

        self.consecutive_ring_losses = 0

        is_needle_valid = det_hyb.get("needle_valid", False) and (det_hyb.get("needle_strength", 0.0) >= 15.0)

        if is_needle_valid:
            self.consecutive_hybrid_losses = 0

            fresh_w = det_hyb.get("white_zone")
            fresh_b = det_hyb.get("black_zone")
            locked_src = str((self.locked_white_zone or {}).get("source", ""))
            fresh_src = str((fresh_w or {}).get("source", ""))
            if (
                not locked_src.startswith("MEASURED")
                and fresh_w is not None
                and fresh_src.startswith("MEASURED")
            ):
                self.locked_white_zone = dict(fresh_w)
                if fresh_b is not None:
                    self.locked_black_zone = dict(fresh_b)

            det_hyb["white_zone"] = self.locked_white_zone
            det_hyb["black_zone"] = self.locked_black_zone
            det_hyb["cx"] = self.locked_center[0]
            det_hyb["cy"] = self.locked_center[1]
            det_hyb["center"] = self.locked_center
            det_hyb["detector_name"] = "BASELINE_ZONE_HYBRID_NEEDLE"
            return det_hyb

        # Needle confidence lost
        self.consecutive_hybrid_losses += 1

        # Fallback handling:
        # Hybrid already attempted local window search + full 360 scan internally.
        # If recovery does not occur for >= 2 consecutive frames:
        if self.consecutive_hybrid_losses >= 2:
            fallback_expected = (
                expected_angle
                if expected_angle is not None
                else self.hybrid_detector.last_angle
            )
            det_base = self.baseline_detector.detect(
                frame_bgr=frame_bgr,
                frame_gray=frame_gray,
                expected_angle=fallback_expected,
                search_window=max(float(search_window), 60.0),
                dt_frame=dt_frame,
                expected_speed=expected_speed,
            )
            if det_base is not None and det_base.get("ring_present"):
                if det_base.get("needle_valid", False):
                    print(
                        f"[VISION] HYBRID_NEEDLE_FALLBACK_BASELINE: recovered needle={det_base['needle_angle']:.1f}°, "
                        f"strength={det_base['needle_strength']:.1f} (losses={self.consecutive_hybrid_losses})"
                    )
                    self.hybrid_detector.last_angle = det_base["needle_angle"]
                    self.hybrid_detector.last_t = time.monotonic()
                    self.hybrid_detector.consecutive_losses = 0
                    self.consecutive_hybrid_losses = 0

                    det_base["white_zone"] = self.locked_white_zone
                    det_base["black_zone"] = self.locked_black_zone
                    det_base["detector_name"] = "HYBRID_NEEDLE_FALLBACK_BASELINE"
                    return det_base
                else:
                    det_base["white_zone"] = self.locked_white_zone
                    det_base["black_zone"] = self.locked_black_zone
                    det_base["needle_valid"] = False
                    det_base["status"] = str(det_base.get("status") or "LOW_CONFIDENCE")
                    det_base["detector_name"] = "HYBRID_NEEDLE_FALLBACK_BASELINE"
                    return det_base
            else:
                # HYBRID presence succeeded on this frame; only the needle
                # tracker is weak.  A failed BASELINE recovery is therefore
                # NOT evidence that the whole skillcheck vanished.  Keep the
                # generation/zones alive and let HYBRID reacquire next frame.
                det_hyb["white_zone"] = self.locked_white_zone
                det_hyb["black_zone"] = self.locked_black_zone
                det_hyb["cx"] = self.locked_center[0]
                det_hyb["cy"] = self.locked_center[1]
                det_hyb["center"] = self.locked_center
                det_hyb["ring_present"] = True
                det_hyb["needle_valid"] = False
                det_hyb["needle_strength"] = 0.0
                det_hyb["status"] = "NEEDLE_REACQUIRE_GRACE"
                det_hyb["detector_name"] = "HYBRID_NEEDLE_REACQUIRE_GRACE"
                return det_hyb

        # Frame 1 loss: return hybrid marked invalid (never emit random or guess angle)
        det_hyb["white_zone"] = self.locked_white_zone
        det_hyb["black_zone"] = self.locked_black_zone
        det_hyb["cx"] = self.locked_center[0]
        det_hyb["cy"] = self.locked_center[1]
        det_hyb["center"] = self.locked_center
        det_hyb["needle_valid"] = False
        det_hyb["status"] = "LOW_CONFIDENCE"
        det_hyb["detector_name"] = "BASELINE_ZONE_HYBRID_NEEDLE"
        return det_hyb

    def extract_zones(
        self, det: Optional[Dict[str, Any]], locked_zones: Optional[Tuple[Any, Any]] = None
    ) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        """
        Extracts high-precision Great (white) and Good (black) zones.
        Preserves locked coordinates within a single check generation.
        """
        if locked_zones and locked_zones[0] is not None:
            return locked_zones

        if self.is_orchestrated:
            if self.is_pressed and det is not None:
                w_d = det.get("white_zone")
                b_d = det.get("black_zone")
                if w_d is not None or b_d is not None:
                    return w_d, b_d

            if self.locked_white_zone is not None:
                return self.locked_white_zone, self.locked_black_zone

            if det is not None:
                w_d = det.get("white_zone")
                b_d = det.get("black_zone")
                if w_d is not None or b_d is not None:
                    return w_d, b_d

            return None, None

        if self.backend is not None:
            return self.backend.extract_zones(det, locked_zones)
        return None, None
