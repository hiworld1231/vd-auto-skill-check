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
        self.lead_uncertainty_s = 0.0
        self.dispatch_uncertainty_s = 0.0
        self.latency_uncertainty_s = 0.0
        self.target_offset_ratio = float(target_offset_ratio)
        self.session_base_speed = float(session_base_speed)
        self.configured_speed_mode = SPEED_MODE_GEN_RUSH
        self.active_speed_mode = SPEED_MODE_GEN_RUSH
        self.mode_switched = False  # compatibility only; never drives timing
        self.switch_info = None
        self.fit_window = max(5, min(14, int(fit_window)))
        self.reset()

    def set_delivery_lead(
        self,
        lead_ms: float,
        uncertainty_ms: float = 0.0,
        dispatch_uncertainty_ms: float = 0.0,
    ) -> None:
        self.latency_s = max(0.0, float(lead_ms)) / 1000.0
        self.lead_uncertainty_s = max(0.0, float(uncertainty_ms)) / 1000.0
        self.dispatch_uncertainty_s = (
            max(0.0, float(dispatch_uncertainty_ms)) / 1000.0
        )
        # Both are predictive one-event error scales.  Sum them conservatively
        # for the hard GREAT gate instead of letting two independent timing
        # uncertainties cancel each other on paper.
        self.latency_uncertainty_s = (
            self.lead_uncertainty_s + self.dispatch_uncertainty_s
        )

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
        self.generation_prior_speed: Optional[float] = (
            float(self.speed_deg_s) if is_chain and keep_speed else None
        )
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
        # Frenzy handoff can first become visible just after this generation's
        # GREAT centre. Once that is proven from the trailing success sector,
        # the target occurrence is permanently the one behind the spawn phase.
        # Never silently switch to target+360 on the next frame.
        self._frenzy_target_occurrence_passed = False
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
            current_w = self.locked_zones[0] if self.locked_zones else None
            current_src = str((current_w or {}).get("source", ""))
            new_src = str(white_zone.get("source", ""))
            should_upgrade = (
                self.locked_zones is None
                or current_w is None
                or (
                    not current_src.startswith("MEASURED")
                    and new_src.startswith("MEASURED")
                )
            )
            if should_upgrade:
                self.locked_zones = (
                    dict(white_zone),
                    dict(black_zone) if black_zone else (
                        dict(self.locked_zones[1])
                        if self.locked_zones and self.locked_zones[1]
                        else None
                    ),
                )
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
        """Early measured speed, with a Frenzy transition sanity gate."""
        if not (
            self.has_adapted
            and self._fit_sample_count >= 3
            and self._fit_span_s >= 0.020
            and 20.0 <= self.speed_deg_s <= 1500.0
            and (self._fit_residual_mad_deg is None or self._fit_residual_mad_deg <= 4.0)
        ):
            return False

        if (
            self.is_chain
            and self.generation_prior_speed is not None
            and self._fit_sample_count < 8
        ):
            prior = max(20.0, float(self.generation_prior_speed))
            # Keep the previous generation as a sanity reference through the
            # first seven fit samples. Deep Frenzy replays show that the raw fit
            # can jump sharply on sample 6 before settling; do not suddenly
            # remove the prior exactly at that boundary.
            if not (0.60 * prior <= self.speed_deg_s <= 2.00 * prior):
                return False

        return True

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

    def mark_tracking(self) -> None:
        self.fire_state = FIRE_TRACKING

    def mark_fired(self) -> None:
        self.fire_state = FIRE_FIRED

    def get_actuation_speed(self) -> float:
        """Best currently available speed for scheduling.

        Scheduling must start before a full 3-5 frame fit exists.  Use the
        session prior on the first frame, the adjacent-frame segment estimate
        as soon as a second unique frame arrives, and the robust fit once it is
        available.  Every later frame may reschedule the same pending keydown.
        """
        if not self.has_adapted:
            if self.is_chain and self.generation_prior_speed is not None:
                # Do not use the first adjacent Frenzy segment.  At 120 FPS
                # capture / ~60 Hz source updates that segment frequently looks
                # 2x too fast.  Hold the previous generation speed until a
                # robust 3-point fit exists.
                chosen = float(self.generation_prior_speed)
                reason = "FRENZY_PRIOR_HOLD"
            elif self._segment_speeds:
                chosen = float(statistics.median(list(self._segment_speeds)[-2:]))
                reason = "SEGMENT_PROVISIONAL"
            else:
                chosen = float(self.session_base_speed)
                reason = "SESSION_PRIOR_PREARM"
            self._actuation_speed = chosen
            self._actuation_speed_reason = reason
            return chosen

        raw = float(self.speed_deg_s)

        if (
            self.is_chain
            and self.generation_prior_speed is not None
            and self._fit_sample_count < 8
        ):
            prior = max(20.0, float(self.generation_prior_speed))
            # Short Frenzy fits are noisy but the immediately previous
            # generation is a strong physical prior. Keep a diminishing amount
            # of shrinkage through samples 6-7 instead of jumping to the raw fit
            # in one frame.
            clamped = max(0.65 * prior, min(1.65 * prior, raw))
            alpha_by_n = {
                3: 0.35,
                4: 0.55,
                5: 0.75,
                6: 0.85,
                7: 0.93,
            }
            alpha = alpha_by_n.get(int(self._fit_sample_count), 1.0)
            chosen = prior + alpha * (clamped - prior)
            self._actuation_speed = float(chosen)
            self._actuation_speed_reason = "FRENZY_PRIOR_BLEND"
            return float(chosen)

        low = self._speed_uncertainty_low
        high = self._speed_uncertainty_high
        if (
            low is not None
            and high is not None
            and math.isfinite(float(low))
            and math.isfinite(float(high))
        ):
            chosen = 0.5 * (float(low) + float(high))
            reason = "UNCERTAINTY_MIDPOINT"
        else:
            chosen = raw
            reason = "ROBUST_MEDIAN"

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
            "generation_prior_speed": self.generation_prior_speed,
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

        frenzy_handoff_success_tail = False
        if (
            self.is_chain
            and len(self._unwrapped) <= 1
            and black_zone
            and black_zone.get("end") is not None
        ):
            target_to_current = (current_angle - target_angle) % 360.0
            target_to_success_end = (
                float(black_zone["end"]) - target_angle
            ) % 360.0
            # A relocated Frenzy generation can first become observable only
            # after the GREAT centre has just passed. If the first handoff frame
            # is still inside the trailing success sector, latch that occurrence
            # for the entire generation. The previous implementation fixed only
            # this one predict() call; the second frame reverted to target+360.
            if (
                0.25 < target_to_current < target_to_success_end
                and target_to_success_end < 120.0
            ):
                self._frenzy_target_occurrence_passed = True
                frenzy_handoff_success_tail = True

        if self.is_chain and self._frenzy_target_occurrence_passed:
            # Choose the last occurrence at/before the generation's first phase.
            # This target can never become "next revolution" later in the same
            # one-sweep skillcheck, even if GREAT geometry refines slightly.
            target_u = float(target_angle)
            while target_u > start_u + 1e-9:
                target_u -= 360.0
            while target_u + 360.0 <= start_u + 1e-9:
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
            frenzy_prior_tail_safe = (
                frenzy_handoff_success_tail
                and not self.has_adapted
                and self.generation_prior_speed is not None
                and expected_delivery_deg * 1.25 + 2.0
                <= remaining_success_deg
            )
            reactive_safe = (
                remaining_success_deg > 0.0
                and expected_delivery_deg + 2.0 <= remaining_success_deg
                and (
                    len(self.history) >= 3
                    or (not self.has_adapted and not self.is_chain)
                    or frenzy_prior_tail_safe
                )
            )

        expected_landing_u = current_u + speed * self.latency_s
        success_end_u = None
        success_start_u = None
        if white_zone and white_zone.get("start") is not None:
            success_start_u = float(white_zone["start"])
            while success_start_u > target_u + 1e-9:
                success_start_u -= 360.0
            while success_start_u + 360.0 <= target_u + 1e-9:
                success_start_u += 360.0
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

        if self.is_chain and self._actuation_speed_reason == "FRENZY_PRIOR_HOLD":
            speed_low = max(20.0, speed * 0.70)
            speed_high = min(1500.0, speed * 1.45)
            speed_source = "FRENZY_PRIOR"
        elif self.is_chain and self._actuation_speed_reason == "FRENZY_PRIOR_BLEND":
            raw = float(self.speed_deg_s)
            raw_low = float(self._speed_uncertainty_low or raw)
            raw_high = float(self._speed_uncertainty_high or raw)
            shift = speed - raw
            prior = max(20.0, float(self.generation_prior_speed or speed))
            prior_low = 0.65 * prior
            prior_high = 1.65 * prior
            shifted_low = raw_low + shift
            shifted_high = raw_high + shift
            speed_low = max(20.0, prior_low, min(speed, shifted_low))
            speed_high = min(
                1500.0,
                prior_high,
                max(speed, shifted_high),
            )
            if speed_low > speed_high:
                speed_low = speed_high = speed
            speed_source = "FRENZY_BLEND"
        elif self.has_adapted:
            speed_low = float(self._speed_uncertainty_low or speed)
            speed_high = float(self._speed_uncertainty_high or speed)
            speed_source = "MEASURED"
        elif self._segment_speeds:
            # One adjacent-frame segment is enough to move a tentative
            # deadline on normal checks, but not in Frenzy.
            speed_low = max(20.0, speed * 0.85)
            speed_high = min(1500.0, speed * 1.15)
            speed_source = "SEGMENT_PROVISIONAL"
        else:
            speed_low = max(20.0, speed * 0.88)
            speed_high = min(1500.0, speed * 1.12)
            speed_source = "SESSION_PRIOR"
        crossing_earliest_ms = (
            angular_dist / max(speed_high, 20.0) * 1000.0
            if not passed_target else 0.0
        )
        crossing_latest_ms = (
            angular_dist / max(speed_low, 20.0) * 1000.0
            if not passed_target else 0.0
        )
        white_window_ms = None
        great_interval_safe = False
        great_interval_intersects = False
        landing_low_u = expected_landing_u
        landing_high_u = expected_landing_u
        great_start_u = None
        great_end_u = None

        if white_zone and white_zone.get("width") is not None:
            white_width = float(white_zone["width"])
            white_window_ms = white_width / max(speed, 20.0) * 1000.0

            # Use the measured center/width pair as one continuous unwrapped
            # interval.  It is more robust around 359°->0° than comparing the
            # raw circular endpoints independently.
            great_start_u = target_u - 0.5 * white_width
            great_end_u = target_u + 0.5 * white_width

            # If the key is scheduled, landing occurs time_to_hit_s from this
            # frame.  If the deadline is already due, the earliest possible
            # landing is one delivery-lead later.
            landing_horizon_s = (
                self.latency_s
                if passed_target or press_timestamp <= current_t
                else max(self.latency_s, time_to_hit_s)
            )
            horizon_low_s = max(
                0.0, landing_horizon_s - self.latency_uncertainty_s
            )
            horizon_high_s = (
                landing_horizon_s + self.latency_uncertainty_s
            )
            landing_low_u = current_u + speed_low * horizon_low_s
            landing_high_u = current_u + speed_high * horizon_high_s
            if landing_low_u > landing_high_u:
                landing_low_u, landing_high_u = landing_high_u, landing_low_u

            # Reserve a sub-degree segmentation margin so a mathematically
            # boundary-touching prediction is not called "safe".
            margin = min(1.0, max(0.35, 0.08 * white_width))
            great_interval_safe = (
                landing_low_u >= great_start_u + margin
                and landing_high_u <= great_end_u - margin
            )
            great_interval_intersects = (
                landing_high_u >= great_start_u
                and landing_low_u <= great_end_u
            )

        return {
            "target": target,
            "target_angle": target_angle,
            "speed_deg_s": speed,
            "raw_fit_speed_deg_s": float(self.speed_deg_s),
            "actuation_speed_reason": self._actuation_speed_reason,
            "speed_source": speed_source,
            "is_chain": bool(self.is_chain),
            "fit_sample_count": int(self._fit_sample_count),
            "generation_prior_speed": self.generation_prior_speed,
            "angular_distance_deg": angular_dist,
            "time_to_hit_ms": time_to_hit_s * 1000.0,
            "crossing_earliest_ms": crossing_earliest_ms,
            "crossing_latest_ms": crossing_latest_ms,
            "crossing_uncertainty_ms": max(0.0, crossing_latest_ms - crossing_earliest_ms),
            "white_window_ms": white_window_ms,
            "great_width_deg": float(white_zone["width"]) if white_zone and white_zone.get("width") is not None else None,
            "white_source": white_zone.get("source") if white_zone else None,
            "great_geometry_refined": bool(white_zone.get("geometry_refined", False)) if white_zone else False,
            "great_boundary_method": white_zone.get("boundary_method") if white_zone else None,
            "landing_uncertainty_low_angle": landing_low_u % 360.0,
            "landing_uncertainty_high_angle": landing_high_u % 360.0,
            "landing_uncertainty_width_deg": max(0.0, landing_high_u - landing_low_u),
            "great_interval_safe": bool(great_interval_safe),
            "great_interval_intersects": bool(great_interval_intersects),
            "speed_uncertainty_reliable": self._fit_sample_count >= 5,
            "lead_uncertainty_ms": self.lead_uncertainty_s * 1000.0,
            "dispatch_uncertainty_ms": self.dispatch_uncertainty_s * 1000.0,
            "delivery_uncertainty_ms": self.latency_uncertainty_s * 1000.0,
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
            "success_width_deg": (
                max(0.0, success_end_u - success_start_u)
                if success_end_u is not None and success_start_u is not None
                else None
            ),
            "frenzy_handoff_success_tail": bool(frenzy_handoff_success_tail),
            "frenzy_target_occurrence_latched": bool(
                self._frenzy_target_occurrence_passed
            ),
        }
