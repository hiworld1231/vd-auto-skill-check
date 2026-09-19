"""
Continuous angular predictor for Violent District GEN_RUSH.

Design goal:
    fresh unique frames -> recent angle/time samples -> measured angular velocity
    -> time-to-center -> one delivery lead -> schedule Space.

This class intentionally has no BASE/VARIABLE switch, no speed tiers, and no
speed-profile latency lookup.  It keeps a small recent window so speed changes
are reflected quickly.  The interface mirrors the subset of SkillCheckPredictor
used by skillcheck_bot.py so GEN_RUSH can use it without touching other modes.
"""
from __future__ import annotations

import collections
import math
import statistics
from typing import Any, Deque, Dict, List, Optional, Tuple

# Local runtime states: CLEAN V5 deliberately does not import the legacy
# predictor/state machine.
SPEED_MODE_GEN_RUSH = "GEN_RUSH_CONTINUOUS"
STATE_NO_MOTION = "NO_MOTION"
STATE_PROVISIONAL = "PROVISIONAL"
STATE_LOCKED = "LOCKED"
STATE_COMMITTED = "COMMITTED"


class ContinuousAngularPredictor:
    """Small-window, continuous-speed predictor used only by GEN_RUSH."""

    def __init__(
        self,
        latency_ms: float,
        target_offset_ratio: float = 0.50,
        session_base_speed: float = 278.0,
        *,
        fit_window: int = 10,
    ):
        self.base_latency_ms = float(latency_ms)
        self.latency_s = float(latency_ms) / 1000.0
        self.target_offset_ratio = float(target_offset_ratio)
        self.session_base_speed = float(session_base_speed)
        self.configured_speed_mode = SPEED_MODE_GEN_RUSH
        self.active_speed_mode = SPEED_MODE_GEN_RUSH
        self.mode_switched = False  # compatibility only; never drives timing
        self.switch_info = None
        self.fit_window = max(5, min(14, int(fit_window)))
        self.reset()

    @property
    def speed_mode(self) -> str:
        return SPEED_MODE_GEN_RUSH

    @speed_mode.setter
    def speed_mode(self, _value: str) -> None:
        # GEN_RUSH continuous mode has no state-machine speed mode.
        self.active_speed_mode = SPEED_MODE_GEN_RUSH

    def reset(
        self,
        keep_speed: bool = False,
        default_speed: Optional[float] = None,
        is_chain: bool = False,
        session_base_speed: Optional[float] = None,
    ) -> None:
        if session_base_speed is not None:
            self.session_base_speed = float(session_base_speed)

        prior_speed = getattr(self, "speed_deg_s", self.session_base_speed)
        self.history: List[Tuple[float, float]] = []
        self._unwrapped: Deque[Tuple[float, float]] = collections.deque(maxlen=self.fit_window)
        self.speed_fits: Deque[Tuple[float, float]] = collections.deque(maxlen=8)
        self.locked_zones: Optional[Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]] = None
        self.spawn_angle: Optional[float] = None
        self.spawn_t: Optional[float] = None
        self.spawn_stutter_detected = False
        self.motion_onset = bool(is_chain and keep_speed)
        self.state = STATE_PROVISIONAL if self.motion_onset else STATE_NO_MOTION
        self.speed_deg_s = float(default_speed if default_speed is not None else (prior_speed if keep_speed else self.session_base_speed))
        self.shadow_live_speed = self.speed_deg_s
        self.locked_speed: Optional[float] = None
        self.has_adapted = False
        self.is_chain = bool(is_chain)
        self.mode_switched = False
        self.switch_info = None
        self._fit_residual_mad_deg: Optional[float] = None
        self._fit_span_s = 0.0
        self._fit_sample_count = 0
        self._last_fit_speed_spread = 0.0
        self._short_speed_shadow: Optional[float] = None

    @staticmethod
    def _signed_step(current: float, previous: float) -> float:
        return (float(current) - float(previous) + 180.0) % 360.0 - 180.0

    def _fit_recent_speed(self) -> Optional[float]:
        pts = list(self._unwrapped)
        if len(pts) < 3:
            return None

        slopes: List[float] = []
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dt = pts[j][0] - pts[i][0]
                if dt >= 0.006:
                    slope = (pts[j][1] - pts[i][1]) / dt
                    if 20.0 <= slope <= 1500.0:
                        slopes.append(slope)
        if len(slopes) < 2:
            return None

        speed = float(statistics.median(slopes))
        intercepts = [ang - speed * t for t, ang in pts]
        intercept = float(statistics.median(intercepts))
        residuals = [abs(ang - (intercept + speed * t)) for t, ang in pts]
        residual_mad = float(statistics.median(residuals)) if residuals else 0.0

        self._fit_residual_mad_deg = residual_mad
        self._fit_span_s = pts[-1][0] - pts[0][0]
        self._fit_sample_count = len(pts)

        # Shadow-only recency diagnostic.  If Roblox ever changes speed inside
        # one check, this exposes long-fit lag without silently switching the
        # production estimator.
        short_pts = pts[-4:]
        self._short_speed_shadow = None
        if len(short_pts) >= 3:
            short_slopes = []
            for i in range(len(short_pts)):
                for j in range(i + 1, len(short_pts)):
                    dt = short_pts[j][0] - short_pts[i][0]
                    if dt >= 0.006:
                        s = (short_pts[j][1] - short_pts[i][1]) / dt
                        if 20.0 <= s <= 1500.0:
                            short_slopes.append(s)
            if short_slopes:
                self._short_speed_shadow = float(statistics.median(short_slopes))
        return speed

    def update(
        self,
        t: float,
        needle_angle: float,
        needle_strength: float,
        white_zone: Optional[Dict[str, float]],
        black_zone: Optional[Dict[str, float]],
    ) -> bool:
        if white_zone and white_zone.get("start") is not None:
            if self.locked_zones is None or self.locked_zones[0] is None:
                self.locked_zones = (dict(white_zone), dict(black_zone) if black_zone else None)
        elif black_zone and self.locked_zones is None:
            self.locked_zones = (None, dict(black_zone))

        if needle_strength <= 15.0:
            return False

        t = float(t)
        angle = float(needle_angle) % 360.0
        if not self.history:
            self.spawn_angle = angle
            self.spawn_t = t
            self.history.append((t, angle))
            self._unwrapped.append((t, angle))
            return True

        last_t, last_angle = self.history[-1]
        dt = t - last_t
        if dt <= 0.001:
            return False

        step = self._signed_step(angle, last_angle)
        if abs(step) < 0.10:
            # Duplicate / effectively duplicate frame.
            return False

        interval_speed = abs(step) / dt
        if not self.motion_onset:
            if interval_speed < 15.0:
                self.spawn_stutter_detected = True
                self.spawn_angle = angle
                self.spawn_t = t
                self.history = [(t, angle)]
                self._unwrapped.clear()
                self._unwrapped.append((t, angle))
                return False
            self.motion_onset = True
            self.state = STATE_PROVISIONAL

        # Needle should move clockwise.  Allow tiny negative detector noise, reject
        # genuine backwards jumps rather than poisoning the short fit window.
        if step < -1.0:
            return False

        # Hard physical/outlier guard.  A little angle slack is allowed because the
        # detector itself has quantization/jitter.
        max_forward_step = 1500.0 * dt + 8.0
        if step > max_forward_step:
            return False

        prev_unwrapped = self._unwrapped[-1][1]
        unwrapped_angle = prev_unwrapped + step
        self.history.append((t, angle))
        # Bound compatibility history too; the scheduler only needs recent samples.
        if len(self.history) > 1