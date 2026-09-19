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
        if len(self.history) > 12:
            del self.history[:-12]
        self._unwrapped.append((t, unwrapped_angle))

        measured = self._fit_recent_speed()
        if measured is not None:
            self.speed_deg_s = measured
            self.shadow_live_speed = measured
            self.speed_fits.append((t, measured))
            self.has_adapted = True

            recent = [x[1] for x in list(self.speed_fits)[-3:]]
            self._last_fit_speed_spread = (max(recent) - min(recent)) if len(recent) >= 2 else 0.0

        return True

    def has_usable_speed(self) -> bool:
        """Early-but-measured speed for short checks; never falls back to base prior."""
        return bool(
            self.has_adapted
            and self._fit_sample_count >= 3
            and self._fit_span_s >= 0.020
            and 20.0 <= self.speed_deg_s <= 1500.0
            and (self._fit_residual_mad_deg is None or self._fit_residual_mad_deg <= 4.0)
        )

    def has_stable_speed(self) -> bool:
        if self.state in (STATE_LOCKED, STATE_COMMITTED):
            return True
        if not self.has_usable_speed():
            return False
        # Moonlight's real-match data shows short tracks are the dominant miss
        # source.  Normal commits therefore wait for a longer fit; the explicit
        # short-check path can still use >=3 measured samples when there is no
        # time left to wait.
        if self._fit_sample_count < 5 or self._fit_span_s < 0.045:
            return False

        recent = [x[1] for x in list(self.speed_fits)[-3:]]
        if len(recent) < 2:
            return False
        med = float(statistics.median(recent))
        spread = max(recent) - min(recent)
        allowed_spread = max(30.0, 0.08 * abs(med))
        if spread > allowed_spread:
            return False
        if self._fit_residual_mad_deg is not None and self._fit_residual_mad_deg > 2.5:
            return False

        self.state = STATE_LOCKED
        self.locked_speed = self.speed_deg_s
        return True

    def get_shadow_telemetry(self) -> Dict[str, Any]:
        return {
            "speed_mode": SPEED_MODE_GEN_RUSH,
            "configured_speed_mode": SPEED_MODE_GEN_RUSH,
            "continuous_tracking": True,
            "session_base_speed": self.session_base_speed,
            "live_speed": self.shadow_live_speed,
            "live_vs_prior_delta": self.shadow_live_speed - self.session_base_speed,
            "prediction_speed_used": self.speed_deg_s,
            "live_fit_spread": self._last_fit_speed_spread,
            "fit_residual_mad_deg": self._fit_residual_mad_deg,
            "fit_span_ms": self._fit_span_s * 1000.0,
            "fit_sample_count": self._fit_sample_count,
            "short_speed_shadow": self._short_speed_shadow,
            "short_vs_long_delta": (
                None if self._short_speed_shadow is None
                else self._short_speed_shadow - self.speed_deg_s
            ),
            # Compatibility: this architecture never switches modes.
            "mode_switched": False,
            "switch_info": None,
        }

    def predict(self, current_t: float, current_angle: float, target: str = "GREAT") -> Optional[Dict[str, Any]]:
        if not self.locked_zones:
            return None
        white_zone, black_zone = self.locked_zones

        if target == "GREAT" and white_zone:
            if white_zone.get("center") is not None:
                target_angle = float(white_zone["center"]) % 360.0
            elif white_zone.get("start") is not None and white_zone.get("width") is not None:
                target_angle = (float(white_zone["start"]) + 0.5 * float(white_zone["width"])) % 360.0
            else:
                return None
        elif target == "GOOD" and black_zone and black_zone.get("center") is not None:
            target_angle = float(black_zone["center"]) % 360.0
        else:
            return None

        speed = float(self.speed_deg_s)
        if not math.isfinite(speed) or speed < 20.0:
            return None

        current_t = float(current_t)
        current_angle = float(current_angle) % 360.0

        # Work in the same unwrapped phase as the fit.  A signed circular diff is
        # NOT sufficient here: when the target is >180° ahead clockwise it looks
        # "negative" even though it has not been passed.  Pick the first target
        # occurrence at/after this check's first measured phase instead.
        if self._unwrapped:
            current_u = float(self._unwrapped[-1][1])
            start_u = float(self._unwrapped[0][1])
        else:
            current_u = current_angle
            start_u = current_angle
        target_u = float(target_angle)
        while target_u + 1e-9 < start_u:
            target_u += 360.0
        remaining_to_target = target_u - current_u
        passed_target = remaining_to_target < -0.25

        # If the Great centre is already behind us, never pretend the next full
        # revolution is the same check.  A reactive fallback is allowed only when
        # the measured delivery travel still fits inside the trailing GOOD zone.
        reactive_safe = False
        if passed_target and black_zone and black_zone.get("end") is not None:
            good_end_u = float(black_zone["end"])
            while good_end_u + 1e-9 < target_u:
                good_end_u += 360.0
            remaining_success_deg = good_end_u - current_u
            expected_delivery_deg = speed * self.latency_s
            reactive_safe = (
                remaining_success_deg > 0.0
                and expected_delivery_deg + 2.0 <= remaining_success_deg
                and len(self.history) >= 3
            )

        expected_landing_u = current_u + speed * self.latency_s
        success_end_u = None
        if black_zone and black_zone.get("end") is not None:
            success_end_u = float(black_zone["end"])
            while success_end_u + 1e-9 < target_u:
                success_end_u += 360.0
        elif white_zone and white_zone.get("end") is not None:
            success_end_u = float(white_zone["end"])
            while success_end_u + 1e-9 < target_u:
                success_end_u += 360.0

        if passed_target:
            angular_dist = 0.0
            time_to_hit_s = 0.0
            press_timestamp = current_t
            should_press_now = reactive_safe
        else:
            angular_dist = max(0.0, remaining_to_target)
            time_to_hit_s = angular_dist / speed
            press_timestamp = current_t + time_to_hit_s - self.latency_s
            if press_timestamp <= current_t:
                # Deadline already arrived.  Direct firing is allowed only when the
                # predicted delivered position still lies inside the known success
                # arc.  This prevents an unsafe late IMMEDIATE from becoming a MISS.
                should_press_now = (
                    success_end_u is not None
                    and expected_landing_u <= (success_end_u - 1.0)
                )
            else:
                should_press_now = False

        return {
            "target": target,
            "target_angle": target_angle,
            "speed_deg_s": speed,
            "angular_distance_deg": angular_dist,
            "time_to_hit_ms": time_to_hit_s * 1000.0,
            "press_timestamp": press_timestamp,
            "time_until_press_ms": (press_timestamp - current_t) * 1000.0,
            "should_press_now": should_press_now,
            "state": self.state,
            "is_locked": self.state in (STATE_LOCKED, STATE_COMMITTED),
            "continuous_tracking": True,
            "reactive_safe_fallback": reactive_safe,
            "target_passed": passed_target,
            "expected_landing_angle": expected_landing_u % 360.0,
            "success_end_angle": (success_end_u % 360.0) if success_end_u is not None else None,
        }
