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
STATE_COMMITTED = "COMMITTED"  # compatibility label; not a fit-quality state
FIRE_TRACKING = "TRACKING"
FIRE_ARMED = "ARMED"
FIRE_FIRED = "FIRED"


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
        self._slope_mad_deg_s: Optional[float] = None
        self._speed_uncertainty_low: Optional[float] = None
        self._speed_uncertainty_high: Optional[float] = None
        self._segment_speeds: Deque[float] = collections.deque(maxlen=7)
        self._actuation_speed: Optional[float] = None
        self._actuation_speed_reason = "ROBUST_LONG_FIT"
        self.fire_state = FIRE_TRACKING

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
        slope_mad = float(statistics.median(abs(x - speed) for x in slopes))
        # Theil-Sen pairwise slopes are robust but correlated.  Treat the
        # MAD-derived standard error as a shadow confidence envelope instead
        # of pretending the median is exact.  A small 3% floor protects
        # perfectly quantized synthetic tracks from false zero uncertainty.
        robust_se = 1.4826 * slope_mad / math.sqrt(max(1.0, float(len(pts))))
        half_width = max(3.0 * robust_se, 0.03 * abs(speed))
        self._slope_mad_deg_s = slope_mad
        self._speed_uncertainty_low = max(20.0, speed - half_width)
        self._speed_uncertainty_high = min(1500.0, speed + half_width)

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

        # Keep a second, deliberately local estimator made only from adjacent
        # accepted unique-frame segments.  It is not used on the normal path.
        # At very high speed, occasional detector jumps/skipped source frames can
        # make the all-pairs fit temporarily optimistic even though the fit's own
        # spread telemetry says it is unstable.
        segment_speed = step / dt
        if 20.0 <= segment_speed <= 1500.0:
            self._segment_speeds.append(float(segment_speed))

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

    def _mark_fit_uncertain(self) -> bool:
        if self.motion_onset:
            self.state = STATE_PROVISIONAL
        return False

    def has_stable_speed(self) -> bool:
        # Fit quality must be evaluated from the current samples on every frame.
        # Being ARMED is a scheduler state, not proof that a newer fit is still
        # trustworthy.
        if not self.has_usable_speed():
            return self._mark_fit_uncertain()
        # Normal commits wait for a longer fit; the explicit urgent path can
        # still inspect >=3 measured samples when there is no time left to wait.
        if self._fit_sample_count < 5 or self._fit_span_s < 0.045:
            return self._mark_fit_uncertain()

        recent = [x[1] for x in list(self.speed_fits)[-3:]]
        if len(recent) < 2:
            return self._mark_fit_uncertain()
        med = float(statistics.median(recent))
        spread = max(recent) - min(recent)
        allowed_spread = max(30.0, 0.08 * abs(med))
        if spread > allowed_spread:
            return self._mark_fit_uncertain()
        if self._fit_residual_mad_deg is not None and self._fit_residual_mad_deg > 2.5:
            return self._mark_fit_uncertain()

        self.state = STATE_LOCKED
        self.locked_speed = self.speed_deg_s
        return True

    def mark_committed(self) -> None:
        self.fire_state = FIRE_ARMED

    def mark_fired(self) -> None:
        self.fire_state = FIRE_FIRED

    def get_actuation_speed(self) -> float:
        """Speed used for target timing.

        The robust all-pairs fit remains authoritative while stable.  When a
        very fast track is explicitly unstable, bias toward the slower local
        segment median.  Violence District's success arc trails the GREAT zone,
        so under uncertainty a small late bias is safer than an early landing.
        The correction is bounded to 20% and never affects normal stable checks.
        """
        raw = float(self.speed_deg_s)
        local = (
            float(statistics.median(list(self._segment_speeds)[-5:]))
            if len(self._segment_speeds) >= 3
            else None
        )
        short_delta = (
            None if self._short_speed_shadow is None
            else float(self._short_speed_shadow) - raw
        )
        short_track = self._fit_sample_count < 5 or self._fit_span_s < 0.045
        unstable = (
            short_track
            or self._last_fit_speed_spread > max(60.0, 0.08 * abs(raw))
            or (short_delta is not None and abs(short_delta) > 60.0)
        )
        chosen = raw
        reason = "ROBUST_LONG_FIT"
        if raw >= 750.0 and unstable:
            # Never speed the prediction up on an uncertain track.  If a local
            # adjacent-segment median exists, prefer it.  Otherwise a very short
            # 3-4 point fit gets a bounded 15% slowdown.  That buys another
            # capture frame or two before the deadline and avoids committing on
            # the first optimistic >1k deg/s estimate.
            if local is not None and math.isfinite(local):
                chosen = max(0.80 * raw, min(raw, local))
                reason = "HIGH_SPEED_CONSERVATIVE_LOCAL"
            elif short_track:
                chosen = 0.85 * raw
                reason = "HIGH_SPEED_PROVISIONAL_85PCT"

        self._actuation_speed = float(chosen)
        self._actuation_speed_reason = reason
        return float(chosen)

    def get_shadow_telemetry(self) -> Dict[str, Any]:
        actuation_speed = self.get_actuation_speed()
        segment_median = (
            float(statistics.median(list(self._segment_speeds)[-5:]))
            if len(self._segment_speeds) >= 3
            else None
        )
        return {
            "speed_mode": SPEED_MODE_GEN_RUSH,
            "configured_speed_mode": SPEED_MODE_GEN_RUSH,
            "continuous_tracking": True,
            "session_base_speed": self.session_base_speed,
            "live_speed": self.shadow_live_speed,
            "live_vs_prior_delta": self.shadow_live_speed - self.session_base_speed,
            "prediction_speed_used": actuation_speed,
            "raw_fit_speed": self.speed_deg_s,
            "segment_speed_median": segment_median,
            "actuation_speed_reason": self._actuation_speed_reason,
            "fit_quality_state": self.state,
            "fire_state": self.fire_state,
            "live_fit_spread": self._last_fit_speed_spread,
            "fit_residual_mad_deg": self._fit_residual_mad_deg,
            "slope_mad_deg_s": self._slope_mad_deg_s,
            "speed_uncertainty_low": self._speed_uncertainty_low,
            "speed_uncertainty_high": self._speed_uncertainty_high,
            "speed_uncertainty_reliable": self._fit_sample_count >= 5,
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

        speed = float(self.get_actuation_speed())
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

        speed_low = float(self._speed_uncertainty_low or speed)
        speed_high = float(self._speed_uncertainty_high or speed)
        crossing_earliest_ms = (
            angular_dist / max(speed_high, 20.0) * 1000.0
            if not passed_target else 0.0
        )
        crossing_latest_ms = (
            angular_dist / max(speed_low, 20.0) * 1000.0
            if not passed_target else 0.0
        )
        white_window_ms = None
        if white_zone and white_zone.get("width") is not None:
            white_window_ms = float(white_zone["width"]) / max(speed, 20.0) * 1000.0

        return {
            "target": target,
            "target_angle": target_angle,
            "speed_deg_s": speed,
            "raw_fit_speed_deg_s": float(self.speed_deg_s),
            "actuation_speed_reason": self._actuation_speed_reason,
            "angular_distance_deg": angular_dist,
            "time_to_hit_ms": time_to_hit_s * 1000.0,
            "crossing_earliest_ms": crossing_earliest_ms,
            "crossing_latest_ms": crossing_latest_ms,
            "crossing_uncertainty_ms": max(0.0, crossing_latest_ms - crossing_earliest_ms),
            "white_window_ms": white_window_ms,
            "press_timestamp": press_timestamp,
            "time_until_press_ms": (press_timestamp - current_t) * 1000.0,
            "should_press_now": should_press_now,
            "state": self.state,
            "fit_quality_state": self.state,
            "fire_state": self.fire_state,
            "is_locked": self.state == STATE_LOCKED,
            "continuous_tracking": True,
            "reactive_safe_fallback": reactive_safe,
            "target_passed": passed_target,
            "expected_landing_angle": expected_landing_u % 360.0,
            "success_end_angle": (success_end_u % 360.0) if success_end_u is not None else None,
        }
