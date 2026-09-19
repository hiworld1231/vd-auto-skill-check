"""
Kinematic Trajectory Predictor for Violent District Skill Checks.
Performs weighted linear regression on circular trajectory points, accurately
estimates needle velocity, and calculates exact microsecond press times with
asymmetric 40% target setpoint and calibrated physical latency compensation.
"""

import collections
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

DEFAULT_SPEED_DEG_S = 270.3
DEFAULT_LATENCY_MS = 56.0  # Calibrated hardware pipeline latency at 240 FPS

SPEED_MODE_BASE = "BASE_SPEED"
SPEED_MODE_VARIABLE = "VARIABLE_SPEED"
SPEED_MODE_BASE_LATENCY_TEST = "BASE_LATENCY_TEST"
SPEED_MODE_GEN_RUSH = "GEN_RUSH"

STATE_NO_MOTION = "NO_MOTION"
STATE_PROVISIONAL = "PROVISIONAL_SPEED"
STATE_LOCKED = "LOCKED_SPEED"
STATE_COMMITTED = "COMMITTED"


class SkillCheckPredictor:
    """
    Real-time kinematic predictor.
    Given needle trajectory points and success zone boundaries, computes the optimal
    press timestamp to hit the Great zone with sub-millisecond precision.
    Uses robust Theil-Sen pairwise median slope estimation and a 4-state lifecycle:
    NO_MOTION -> PROVISIONAL_SPEED -> LOCKED_SPEED -> COMMITTED.
    Supports NO_PERK / BASE_SPEED mode with persistent variable-speed shadow switching.
    """

    def __init__(
        self,
        latency_ms: float = DEFAULT_LATENCY_MS,
        target_offset_ratio: float = 0.50,
        speed_mode: str = SPEED_MODE_VARIABLE,
        session_base_speed: float = 278.0,
    ):
        self.base_latency_ms = float(latency_ms)
        self.latency_s = latency_ms / 1000.0
        self.target_offset_ratio = target_offset_ratio
        self.configured_speed_mode = str(speed_mode)
        self.active_speed_mode = str(speed_mode)
        self.session_base_speed = float(session_base_speed)
        self.shadow_live_speed = float(session_base_speed)
        self.mode_switched = False
        self.switch_info: Optional[Dict[str, Any]] = None
        self.speed_history: collections.deque = collections.deque(maxlen=8)
        self.speed_fits: collections.deque = collections.deque(maxlen=10)
        self.reset()

    @property
    def speed_mode(self) -> str:
        return self.active_speed_mode

    @speed_mode.setter
    def speed_mode(self, val: str):
        self.active_speed_mode = str(val)

    def reset(
        self,
        keep_speed: bool = False,
        default_speed: Optional[float] = None,
        is_chain: bool = False,
        session_base_speed: Optional[float] = None,
    ):
        self.history: List[Tuple[float, float]] = []
        self.has_adapted: bool = False
        self.speed_history.clear()
        self.speed_fits.clear()
        self.is_chain: bool = bool(is_chain)
        if session_base_speed is not None:
            self.session_base_speed = float(session_base_speed)

        # Mode lifecycle isolation:
        # Every NEW independent check (not is_chain) MUST restore configured_speed_mode.
        # VARIABLE mode can only live inside an active confirmed accelerated/frenzy chain context.
        # Once the chain ends, the next check starts strictly in configured_speed_mode.
        if not is_chain:
            self.active_speed_mode = self.configured_speed_mode
            self.mode_switched = False
            self.switch_info = None
            self.speed_deg_s = self.session_base_speed
            self.latency_s = self.base_latency_ms / 1000.0
        else:
            if not self.mode_switched:
                self.active_speed_mode = self.configured_speed_mode

        if self.active_speed_mode in (SPEED_MODE_BASE, SPEED_MODE_BASE_LATENCY_TEST, SPEED_MODE_GEN_RUSH) and not self.mode_switched:
            if not is_chain or default_speed is None:
                self.speed_deg_s = self.session_base_speed
                self.latency_s = self.base_latency_ms / 1000.0
            else:
                self.speed_deg_s = float(default_speed)
        elif default_speed is not None:
            self.speed_deg_s = float(default_speed)
        elif not keep_speed:
            self.speed_deg_s = self.session_base_speed

        self.locked_zones: Optional[Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]] = None
        self.was_approaching: bool = False
        self.spawn_angle: Optional[float] = None
        self.spawn_t: Optional[float] = None
        self.spawn_stutter_detected: bool = False
        self.motion_onset: bool = bool(is_chain and keep_speed)
        self.state: str = STATE_PROVISIONAL if self.motion_onset else STATE_NO_MOTION
        self.locked_speed: Optional[float] = None

    def has_stable_speed(self) -> bool:
        """
        Returns True when needle velocity has been reliably measured:
        - In BASE_SPEED mode (unswitched): confirms motion onset and initial trajectory span;
        - In GEN_RUSH mode (unswitched): requires sustained motion evidence confirming BASE speed
          (>= 4 fits, >= 40ms, spread <= 25°/s, |delta| <= 35°/s) or confirmed switch to VARIABLE;
        - In VARIABLE_SPEED mode: requires at least 3 speed fits with spread <= 25°/s,
          span >= 40ms, and arc >= 20°.
        """
        if self.state in (STATE_LOCKED, STATE_COMMITTED):
            return True

        if self.active_speed_mode == SPEED_MODE_GEN_RUSH and not self.mode_switched:
            # In GEN_RUSH mode, never prematurely lock at 25ms before motion evidence!
            # Must either switch to VARIABLE_SPEED via _evaluate_mode_switch,
            # or collect >= 4 fits over >= 40ms confirming BASE speed without perk acceleration.
            if len(self.speed_fits) < 4:
                return False
            last_4 = list(self.speed_fits)[-4:]
            dt_span = last_4[-1][0] - last_4[0][0]
            if dt_span < 0.040:
                return False
            speeds = [f[1] for f in last_4]
            spread = max(speeds) - min(speeds)
            med_4 = float(np.median(speeds))
            delta = med_4 - self.session_base_speed
            if spread <= 25.0 and abs(delta) <= 35.0:
                self.state = STATE_LOCKED
                self.locked_speed = self.session_base_speed
                return True
            return False

        if self.active_speed_mode in (SPEED_MODE_BASE, SPEED_MODE_BASE_LATENCY_TEST) and not self.mode_switched:
            min_hist = 2 if self.is_chain else 3
            if not self.motion_onset or len(self.history) < min_hist:
                return False
            span = self.history[-1][0] - self.history[0][0]
            if span < (0.015 if self.is_chain else 0.025):
                return False
            self.state = STATE_LOCKED
            self.locked_speed = self.session_base_speed
            return True

        min_hist = 3 if self.is_chain else 5
        if not self.has_adapted or len(self.history) < min_hist:
            return False
        span = self.history[-1][0] - self.history[0][0]
        min_span = 0.025 if self.is_chain else 0.040
        if span < min_span:
            return False
        start_angle = self.history[0][1]
        curr_angle = self.history[-1][1]
        arc_travel = (curr_angle - start_angle + 360.0) % 360.0
        min_arc = 15.0 if self.is_chain else 20.0
        if arc_travel < min_arc:
            return False
        min_fits = 2 if self.is_chain else 3
        if len(self.speed_fits) < min_fits:
            return False
        last_fits = [f[1] for f in list(self.speed_fits)[-min_fits:]]
        spread = max(last_fits) - min(last_fits)
        if spread > 25.0:
            return False
        self.state = STATE_LOCKED
        self.locked_speed = self.speed_deg_s
        return True

    def update(
        self,
        t: float,
        needle_angle: float,
        needle_strength: float,
        white_zone: Optional[Dict[str, float]],
        black_zone: Optional[Dict[str, float]],
    ) -> bool:
        """
        Updates the kinematic tracker with a new visual sample.
        t: monotonic timestamp in seconds
        needle_angle: needle angle in degrees (0..359.9)
        needle_strength: peak redness signal strength
        white_zone: Great zone boundaries dict
        black_zone: Good zone boundaries dict
        Returns True if a valid non-duplicate frame was processed, False otherwise.
        """
        # Lock zone boundaries once confirmed
        if white_zone and white_zone.get("start") is not None:
            if self.locked_zones is None or self.locked_zones[0] is None:
                self.locked_zones = (white_zone, black_zone)

        # Discard false spark spikes
        if needle_strength <= 15.0:
            return False

        # First sample initialization
        if not self.history:
            self.spawn_angle = needle_angle
            self.spawn_t = t
            if self.is_chain:
                self.motion_onset = True
                self.state = STATE_PROVISIONAL
            self.history.append((t, needle_angle))
            return True

        last_t, last_ang = self.history[-1]
        dt = t - last_t
        step = (needle_angle - last_ang + 180.0) % 360.0 - 180.0
        interval_speed = (abs(step) / dt) if dt > 0.001 else 0.0

        if not self.motion_onset:
            if interval_speed < 15.0 or abs(step) < 0.10:
                self.spawn_stutter_detected = True
                self.spawn_angle = needle_angle
                self.spawn_t = t
                self.history = [(t, needle_angle)]
                return False
            else:
                self.motion_onset = True
                self.state = STATE_PROVISIONAL
                if self.spawn_stutter_detected:
                    self.history = [(t, needle_angle)]
                else:
                    self.history.append((t, needle_angle))
        else:
            # Reject duplicate delivery frames (delta < 0.10°)
            if abs(step) < 0.10:
                return False

            # Backward step handling: check if previous point was an outlier spike
            if step < -1.0:
                if len(self.history) >= 2:
                    prev_t, prev_ang = self.history[-2]
                    step_from_prev = (needle_angle - prev_ang + 180.0) % 360.0 - 180.0
                    dt_from_prev = t - prev_t
                    max_step_from_prev = max(25.0, 1500.0 * dt_from_prev + 10.0)
                    if -0.5 <= step_from_prev <= max_step_from_prev:
                        # Outlier spike popped from history, accept current valid point
                        self.history.pop()
                        last_t, last_ang = self.history[-1]
                        step = (needle_angle - last_ang + 180.0) % 360.0 - 180.0
                    else:
                        return False
                else:
                    return False

            # Forward outlier rejection
            max_allowed = max(25.0, 1500.0 * dt + 10.0)
            if step > max_allowed:
                return False

            self.history.append((t, needle_angle))

        # Robust Theil-Sen pairwise median slope estimator
        # Pairwise time threshold dt >= 10ms reduces sensitivity to decode-ready timestamp delivery jitter
        if len(self.history) >= 3:
            recent = self.history[-14:]
            slopes = []
            for i in range(len(recent)):
                for j in range(i + 1, len(recent)):
                    p_dt = recent[j][0] - recent[i][0]
                    if p_dt >= 0.010:
                        p_dtheta = (recent[j][1] - recent[i][1] + 360.0) % 360.0
                        slopes.append(p_dtheta / p_dt)

            if len(slopes) >= 2:
                measured_speed = float(np.median(slopes))
                if 15.0 <= measured_speed <= 1500.0:
                    self.speed_fits.append((t, measured_speed))
                    self.speed_history.append(measured_speed)
                    self.shadow_live_speed = measured_speed
                    self.has_adapted = True

                    if self.speed_mode in (SPEED_MODE_BASE, SPEED_MODE_BASE_LATENCY_TEST, SPEED_MODE_GEN_RUSH) and not self.mode_switched:
                        if self.configured_speed_mode != SPEED_MODE_BASE_LATENCY_TEST:
                            self._evaluate_mode_switch(t)
                        if not self.mode_switched:
                            self.speed_deg_s = self.session_base_speed
                            self.latency_s = self.base_latency_ms / 1000.0
                        else:
                            self.speed_deg_s = self.locked_speed
                    else:
                        self.speed_deg_s = measured_speed
                        self.has_stable_speed()

        return True

    def _evaluate_mode_switch(self, current_t: float):
        """
        Evaluates criteria to switch from BASE_SPEED mode to VARIABLE_SPEED mode.
        Switches only upon confirmed, persistent divergent signal:
        - At least 4 consecutive speed fits;
        - Movement duration across the 4 fits >= 40ms;
        - Arc traversed across the 4 fits >= 20°;
        - Fits all deviate in the same direction by > 35°/s from session_base_speed;
        - Spread among fits <= 25°/s (or up to 15% for extreme high-speed perks > 150°/s deviation);
        - Once switched to VARIABLE_SPEED, never switches back to BASE inside current check.
        In BASE_LATENCY_TEST mode: strictly disabled (never switches).
        """
        if self.configured_speed_mode == SPEED_MODE_BASE_LATENCY_TEST:
            return

        if len(self.speed_fits) < 4:
            return

        last_4 = list(self.speed_fits)[-4:]
        t_start = last_4[0][0]
        t_end = last_4[-1][0]
        dt_span = t_end - t_start
        if dt_span < 0.040:
            return

        h_times = [h[0] for h in self.history]
        idx_start = min(range(len(h_times)), key=lambda i: abs(h_times[i] - t_start))
        idx_end = min(range(len(h_times)), key=lambda i: abs(h_times[i] - t_end))
        if abs(h_times[idx_start] - t_start) > 0.005 or abs(h_times[idx_end] - t_end) > 0.005:
            return

        ang_start = self.history[idx_start][1]
        ang_end = self.history[idx_end][1]
        arc_travel = (ang_end - ang_start + 360.0) % 360.0
        if arc_travel < 20.0:
            return

        speeds = [f[1] for f in last_4]
        spread = max(speeds) - min(speeds)
        med_4 = float(np.median(speeds))
        delta = med_4 - self.session_base_speed

        max_allowed_spread = max(25.0, 0.15 * med_4) if abs(delta) > 150.0 else 25.0
        if spread > max_allowed_spread:
            return

        if abs(delta) > 35.0:
            same_direction = all((s - self.session_base_speed) * delta > 0 for s in speeds)
            all_exceed = all(abs(s - self.session_base_speed) > 35.0 for s in speeds)
            if same_direction and all_exceed:
                self.active_speed_mode = SPEED_MODE_VARIABLE
                self.mode_switched = True
                self.speed_deg_s = med_4
                self.locked_speed = med_4
                self.state = STATE_LOCKED
                self.switch_info = {
                    "t_ms": current_t * 1000.0,
                    "prior_speed": self.session_base_speed,
                    "live_speed": med_4,
                    "delta": delta,
                    "fits": [round(f, 1) for f in speeds],
                    "spread": round(spread, 1),
                    "reason": f"Confirmed persistent speed shift ({delta:+.1f}°/s across 4 fits, spread={spread:.1f}°/s)",
                }

    def get_shadow_telemetry(self) -> Dict[str, Any]:
        """
        Returns full diagnostic telemetry for shadow tracking and logging.
        """
        last_spread = 0.0
        if len(self.speed_fits) >= 2:
            fits = [f[1] for f in list(self.speed_fits)[-3:]]
            last_spread = float(max(fits) - min(fits))
        return {
            "speed_mode": self.active_speed_mode,
            "configured_speed_mode": self.configured_speed_mode,
            "session_base_speed": self.session_base_speed,
            "live_speed": self.shadow_live_speed,
            "live_vs_prior_delta": self.shadow_live_speed - self.session_base_speed,
            "live_fit_spread": last_spread,
            "prediction_speed_used": self.speed_deg_s,
            "mode_switched": self.mode_switched,
            "switch_info": self.switch_info,
        }

    def predict(
        self, current_t: float, current_angle: float, target: str = "GREAT"
    ) -> Optional[Dict[str, Any]]:
        """
        Calculates the exact scheduled timestamp for the Space key press.
        """
        if not self.locked_zones:
            return None

        white_zone, black_zone = self.locked_zones
        if target == "GREAT" and white_zone:
            if white_zone.get("center") is not None:
                target_angle = white_zone["center"] % 360.0
            elif white_zone.get("start") is not None and white_zone.get("width") is not None:
                target_angle = (white_zone["start"] + 0.50 * white_zone["width"]) % 360.0
            elif white_zone.get("start") is not None:
                target_angle = (white_zone["start"] + 4.75) % 360.0
            else:
                return None

            # Assertion: target_angle must match white_center (0.50 center setpoint)
            if white_zone.get("center") is not None:
                center_diff = abs((target_angle - white_zone["center"] + 180.0) % 360.0 - 180.0)
                assert center_diff < 1e-3, (
                    f"GREAT target_angle ({target_angle:.3f}°) != white_center ({white_zone['center']:.3f}°)"
                )
        elif target == "GOOD" and black_zone and black_zone.get("center") is not None:
            target_angle = black_zone["center"] % 360.0
        else:
            return None

        # Calculate shortest signed angular difference to target
        # positive = target is ahead (clockwise), negative = needle has passed target
        direct_diff = (target_angle - current_angle + 180.0) % 360.0 - 180.0

        # Overdue recovery:
        # Needle has arrived at or stepped past target within the overdue recovery window.
        # Guard against false trigger on spawn (frame 1) when needle spawns at standard
        # 270° spawn point with target located just behind spawn (240°-268°), which
        # requires a full 350° clockwise rotation.
        overdue_window = -max(25.0, min(45.0, self.speed_deg_s * 0.10))
        is_overdue = False
        if overdue_window <= direct_diff <= 0.0:
            is_spawn_edge = (
                len(self.history) <= 1
                and 260.0 <= current_angle <= 285.0
                and 230.0 <= target_angle <= 265.0
            )
            if not is_spawn_edge:
                is_overdue = True

        if is_overdue:
            angular_dist = 0.0
            time_to_hit_s = 0.0
            press_timestamp = current_t
            time_until_press_ms = 0.0
            should_press_now = True
        else:
            angular_dist = (target_angle - current_angle) % 360.0
            time_to_hit_s = angular_dist / self.speed_deg_s
            press_timestamp = current_t + time_to_hit_s - self.latency_s
            time_until_press_ms = (press_timestamp - current_t) * 1000.0
            should_press_now = time_until_press_ms <= 0.0

        return {
            "target": target,
            "target_angle": target_angle,
            "speed_deg_s": self.speed_deg_s,
            "angular_distance_deg": angular_dist,
            "time_to_hit_ms": time_to_hit_s * 1000.0,
            "press_timestamp": press_timestamp,
            "time_until_press_ms": time_until_press_ms,
            "should_press_now": should_press_now,
            "state": self.state,
            "is_locked": (self.state in (STATE_LOCKED, STATE_COMMITTED)),
        }
