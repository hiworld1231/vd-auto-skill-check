"""Runtime response-phase correction for GEN_RUSH.

This module intentionally does NOT learn persistent speed profiles.  It only
tracks a short-lived time-domain response bias from clean LIVE outcomes.

Sign convention:
    center_error_ms < 0  -> observed stop was before target center ->
                            reduce compensation (press later)
    center_error_ms > 0  -> observed stop was after target center ->
                            increase compensation (press earlier)
"""
from __future__ import annotations

import math
import statistics
import time
from collections import deque
from typing import Any, Deque, Dict, Optional


class ResponsePhaseTracker:
    def __init__(
        self,
        base_latency_ms: float,
        *,
        window: int = 7,
        min_samples: int = 4,
        deadband_ms: float = 2.0,
        hard_cap_ms: float = 20.0,
        max_step_warmup_ms: float = 3.0,
        max_step_stable_ms: float = 2.0,
        alpha: float = 0.50,
        stale_after_s: float = 45.0,
        reacquire_after_s: float = 120.0,
    ):
        self.base_latency_ms = float(base_latency_ms)
        self.window = max(3, int(window))
        self.min_samples = max(3, int(min_samples))
        self.deadband_ms = float(deadband_ms)
        self.hard_cap_ms = abs(float(hard_cap_ms))
        self.max_step_warmup_ms = abs(float(max_step_warmup_ms))
        self.max_step_stable_ms = abs(float(max_step_stable_ms))
        self.alpha = min(1.0, max(0.05, float(alpha)))
        self.stale_after_s = float(stale_after_s)
        self.reacquire_after_s = float(reacquire_after_s)

        self.phase_correction_ms = 0.0
        self._errors: Deque[float] = deque(maxlen=self.window)
        self._accepted_ids: Deque[str] = deque(maxlen=self.window)
        self._total_clean_samples = 0
        self._last_clean_t: Optional[float] = None
        self._last_outcome_t: Optional[float] = None
        self._state = "WARMUP"
        self._last_update: Dict[str, Any] = {
            "accepted": False,
            "reject_reason": "NO_SAMPLES",
        }

    @staticmethod
    def _finite(value: Any) -> bool:
        return isinstance(value, (int, float)) and math.isfinite(float(value))

    @staticmethod
    def _trigger_mode(trigger_mode: Optional[str], trigger_reason: Optional[str]) -> str:
        if trigger_mode:
            return str(trigger_mode).upper()
        reason = str(trigger_reason or "").upper()
        if "FALLBACK" in reason:
            return "FALLBACK_NO_LOCK"
        if "BACKUP" in reason:
            return "BACKUP"
        if "СПИН" in reason or "SCHEDULED" in reason:
            return "SCHEDULED"
        return "IMMEDIATE"

    def set_base_latency(self, value_ms: float, *, reset_history: bool = True, reason: str = "MANUAL") -> None:
        self.base_latency_ms = float(value_ms)
        if reset_history:
            self._errors.clear()
            self._accepted_ids.clear()
            self.phase_correction_ms = 0.0
            self._state = "WARMUP"
            self._last_clean_t = None
            self._last_update = {
                "accepted": False,
                "reject_reason": f"BASE_RESET_{reason}",
            }

    def get_effective_latency(self, *, chain_count: int = 1, chain_offset_ms: float = 0.0) -> float:
        # Chain offset is an explicit independent term.  No speed-profile delay is
        # applied here by design.
        extra = float(chain_offset_ms) * max(0, int(chain_count) - 1)
        return self.base_latency_ms + self.phase_correction_ms + extra

    def _apply_idle_policy(self, now: float) -> None:
        if self._last_clean_t is None:
            return
        idle = now - self._last_clean_t
        if idle >= self.reacquire_after_s:
            self._errors.clear()
            self._accepted_ids.clear()
            self.phase_correction_ms *= 0.50
            if abs(self.phase_correction_ms) < 0.25:
                self.phase_correction_ms = 0.0
            self._state = "REACQUIRE"
            self._last_clean_t = now
        elif idle >= self.stale_after_s:
            self._state = "STALE"

    def record_outcome(
        self,
        *,
        center_error_ms: Optional[float],
        outcome: str,
        plateau_found: Any,
        trigger_reason: Optional[str] = None,
        trigger_mode: Optional[str] = None,
        scheduler_error_ms: Optional[float] = None,
        scheduler_jitter_ms: Optional[float] = None,
        speed_at_lock: Optional[float] = None,
        speed_at_fire: Optional[float] = None,
        last_frame_age_ms: Optional[float] = None,
        actual_used_delay_ms: Optional[float] = None,
        is_detector_fallback: bool = False,
        detector_source: Optional[str] = None,
        mode_switched_this_check: bool = False,
        live_fit_spread_deg_s: Optional[float] = None,
        sample_id: Optional[str] = None,
        now: Optional[float] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        now = float(now if now is not None else time.monotonic())
        self._apply_idle_policy(now)
        self._last_outcome_t = now

        mode = self._trigger_mode(trigger_mode, trigger_reason)
        jitter = scheduler_jitter_ms if scheduler_jitter_ms is not None else scheduler_error_ms

        reject: Optional[str] = None
        if not self._finite(center_error_ms):
            reject = "MISSING_CENTER_ERROR"
        elif plateau_found is not True:
            reject = "NO_CONFIRMED_PLATEAU"
        elif str(outcome).upper() not in {"GREAT", "GOOD", "MISS"}:
            reject = f"OUTCOME_{str(outcome).upper()}"
        elif mode != "SCHEDULED":
            reject = f"TRIGGER_{mode}"
        elif not self._finite(jitter) or abs(float(jitter)) > 2.0:
            reject = "SCHEDULER_JITTER"
        elif not self._finite(last_frame_age_ms) or float(last_frame_age_ms) > 15.0:
            reject = "STALE_FRAME"
        elif not self._finite(speed_at_lock):
            reject = "NO_SPEED_LOCK"
        elif not self._finite(speed_at_fire):
            reject = "NO_SPEED_AT_FIRE"
        elif bool(is_detector_fallback):
            reject = "DETECTOR_FALLBACK"
        elif mode_switched_this_check:
            # A speed transition is exactly where prediction/model phase is most
            # ambiguous.  Log it, but do not let it steer global response phase.
            reject = "MODE_SWITCHED_THIS_CHECK"
        elif self._finite(live_fit_spread_deg_s) and float(live_fit_spread_deg_s) > 25.0:
            reject = "UNSTABLE_SPEED_FIT"
        else:
            lock = float(speed_at_lock)
            fire = float(speed_at_fire)
            max_drift = max(25.0, 0.08 * abs(lock))
            if abs(fire - lock) > max_drift:
                reject = "SPEED_DRIFT"
            elif abs(float(center_error_ms)) > 60.0:
                reject = "PHASE_OUTLIER"

        before_corr = self.phase_correction_ms
        before_med = statistics.median(self._errors) if self._errors else None

        if reject is not None:
            result = {
                "accepted": False,
                "reject_reason": reject,
                "trigger_mode": mode,
                "input_error_ms": center_error_ms,
                "rolling_median_before_ms": before_med,
                "rolling_median_after_ms": before_med,
                "correction_before_ms": before_corr,
                "correction_after_ms": before_corr,
                "update_delta_ms": 0.0,
                "sample_id": sample_id,
                "detector_source": detector_source,
            }
            self._last_update = result
            return result

        err = float(center_error_ms)
        self._errors.append(err)
        self._accepted_ids.append(sample_id or "")
        self._total_clean_samples += 1
        self._last_clean_t = now

        median = float(statistics.median(self._errors))
        delta = 0.0

        if len(self._errors) >= self.min_samples:
            target = max(-self.hard_cap_ms, min(self.hard_cap_ms, median))
            if abs(target) <= self.deadband_ms:
                target = 0.0

            max_step = self.max_step_warmup_ms if self._state in {"WARMUP", "STALE", "REACQUIRE"} else self.max_step_stable_ms
            raw_delta = self.alpha * (target - self.phase_correction_ms)
            delta = max(-max_step, min(max_step, raw_delta))
            self.phase_correction_ms = max(
                -self.hard_cap_ms,
                min(self.hard_cap_ms, self.phase_correction_ms + delta),
            )
            self._state = "STABLE"
        else:
            self._state = "WARMUP"

        result = {
            "accepted": True,
            "reject_reason": None,
            "trigger_mode": mode,
            "input_error_ms": err,
            "rolling_median_before_ms": before_med,
            "rolling_median_after_ms": median,
            "correction_before_ms": before_corr,
            "correction_after_ms": self.phase_correction_ms,
            "update_delta_ms": delta,
            "sample_id": sample_id,
            "detector_source": detector_source,
        }
        self._last_update = result
        return result

    def get_telemetry(self) -> Dict[str, Any]:
        rolling = float(statistics.median(self._errors)) if self._errors else None
        return {
            "base_compensation_ms": round(self.base_latency_ms, 3),
            "response_phase_correction_ms": round(self.phase_correction_ms, 3),
            "effective_compensation_ms": round(self.base_latency_ms + self.phase_correction_ms, 3),
            "rolling_error_median_ms": None if rolling is None else round(rolling, 3),
            "phase_sample_count": len(self._errors),
            "total_clean_samples": self._total_clean_samples,
            "recent_clean_errors_ms": [round(x, 3) for x in self._errors],
            "accepted_sample_ids": [x for x in self._accepted_ids if x],
            "phase_state": self._state,
            "last_update": dict(self._last_update),
        }
