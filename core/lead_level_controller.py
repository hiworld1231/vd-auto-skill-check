from __future__ import annotations

import math
import statistics
from collections import deque
from typing import Any, Deque, Dict, Optional


class LeadLevelController:
    """Robust session-level delivery lead policy for clean GEN_RUSH.

    Observation (controller invariant):
        ideal_lead_ms = actual_used_delay_ms + center_error_ms

    The controller does not chase individual landings.  It cold-starts from a
    small consistent cluster and later follows only a confirmed recent level
    shift.  Speed-specific tiers are deliberately absent.
    """

    def __init__(
        self,
        seed_lead_ms: float,
        *,
        min_lead_ms: float = 35.0,
        max_lead_ms: float = 160.0,
        window_size: int = 9,
        cold_min_samples: int = 3,
        deadband_ms: float = 15.0,
        cold_max_mad_ms: float = 12.0,
        recent_shift_samples: int = 4,
        recent_shift_max_mad_ms: float = 8.0,
        max_shift_step_ms: float = 20.0,
    ):
        self.seed_lead_ms = float(seed_lead_ms)
        self.current_lead_ms = float(seed_lead_ms)
        self.min_lead_ms = float(min_lead_ms)
        self.max_lead_ms = float(max_lead_ms)
        self.window_size = max(5, int(window_size))
        self.cold_min_samples = max(3, int(cold_min_samples))
        self.deadband_ms = float(deadband_ms)
        self.cold_max_mad_ms = float(cold_max_mad_ms)
        self.recent_shift_samples = max(3, int(recent_shift_samples))
        self.recent_shift_max_mad_ms = float(recent_shift_max_mad_ms)
        self.max_shift_step_ms = float(max_shift_step_ms)
        self.samples: Deque[float] = deque(maxlen=self.window_size)
        self.accepted_total = 0
        self.rejected_total = 0
        self.updates_total = 0
        self.initialized = False
        self.last_result: Dict[str, Any] = {}

    @staticmethod
    def _finite(v: Optional[float]) -> bool:
        try:
            return v is not None and math.isfinite(float(v))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _mad(vals, med: float) -> float:
        return float(statistics.median(abs(float(v) - med) for v in vals)) if vals else 0.0

    def get_lead_ms(self, *, chain_count: int = 1, chain_offset_ms: float = 0.0) -> float:
        # No unvalidated Frenzy offset in the clean runtime.  Kept in the signature
        # for config compatibility, but production should leave it at 0.
        extra = float(chain_offset_ms) * max(0, int(chain_count) - 1)
        return self.current_lead_ms + extra

    def _reject(self, reason: str, **extra: Any) -> Dict[str, Any]:
        self.rejected_total += 1
        self.last_result = {
            "accepted": False,
            "reject_reason": reason,
            "current_lead_ms": self.current_lead_ms,
            "sample_count": len(self.samples),
            "initialized": self.initialized,
            **extra,
        }
        return dict(self.last_result)

    def record_outcome(
        self,
        *,
        center_error_ms: Optional[float],
        actual_used_delay_ms: Optional[float],
        outcome: Optional[str],
        plateau_found: Optional[bool],
        trigger_mode: Optional[str],
        scheduler_jitter_ms: Optional[float],
        frame_age_ms: Optional[float],
        detector_fallback: bool,
        compensation_regime: Optional[str],
        fit_sample_count: Optional[int],
        fit_residual_mad_deg: Optional[float],
        fit_spread_deg_s: Optional[float],
        speed_at_lock: Optional[float] = None,
        speed_at_fire: Optional[float] = None,
        chain_count: int = 1,
        white_source: Optional[str] = None,
        black_source: Optional[str] = None,
        frenzy_transition: bool = False,
    ) -> Dict[str, Any]:
        if not self._finite(center_error_ms):
            return self._reject("NO_CENTER_ERROR")
        if not self._finite(actual_used_delay_ms):
            return self._reject("NO_USED_DELAY")
        if plateau_found is not True:
            return self._reject("NO_CONFIRMED_PLATEAU")
        if outcome not in {"GREAT", "GOOD", "MISS"}:
            return self._reject(f"OUTCOME_{outcome or 'UNKNOWN'}")
        if trigger_mode != "SCHEDULED":
            return self._reject(f"TRIGGER_{trigger_mode or 'UNKNOWN'}")
        if not self._finite(scheduler_jitter_ms) or abs(float(scheduler_jitter_ms)) > 4.5:
            return self._reject("SCHEDULER_JITTER")
        if not self._finite(frame_age_ms) or float(frame_age_ms) > 20.0:
            return self._reject("STALE_FIRE_FRAME")
        if detector_fallback:
            return self._reject("DETECTOR_FALLBACK")
        if compensation_regime != "CONTINUOUS_MEASURED_SPEED":
            return self._reject(f"REGIME_{compensation_regime or 'UNKNOWN'}")
        if fit_sample_count is None or int(fit_sample_count) < 5:
            return self._reject("SHORT_TRACK")
        if self._finite(fit_residual_mad_deg) and float(fit_residual_mad_deg) > 3.0:
            return self._reject("POOR_FIT_RESIDUAL")
        if self._finite(fit_spread_deg_s) and float(fit_spread_deg_s) > 45.0:
            return self._reject("UNSTABLE_SPEED_FIT")
        if abs(float(center_error_ms)) > 90.0:
            return self._reject("PHASE_OUTLIER")
        if bool(frenzy_transition) or int(chain_count) > 1:
            return self._reject("FRENZY_UNVALIDATED")
        if white_source and not str(white_source).startswith("MEASURED"):
            return self._reject("RECONSTRUCTED_GREAT_GEOMETRY")

        if self._finite(speed_at_lock) and self._finite(speed_at_fire):
            lock = abs(float(speed_at_lock))
            drift = abs(float(speed_at_fire) - float(speed_at_lock))
            if drift > max(25.0, 0.08 * lock):
                return self._reject("SPEED_DRIFT", speed_drift_deg_s=drift)

        ideal = float(actual_used_delay_ms) + float(center_error_ms)
        if not (self.min_lead_ms <= ideal <= self.max_lead_ms):
            return self._reject("IDEAL_LEAD_OUT_OF_RANGE", ideal_lead_ms=ideal)

        self.samples.append(ideal)
        self.accepted_total += 1
        vals = list(self.samples)
        med = float(statistics.median(vals))
        mad = self._mad(vals, med)
        before = self.current_lead_ms
        updated = False
        update_reason = None
        recent_med = None
        recent_mad = None

        if not self.initialized and len(vals) >= self.cold_min_samples:
            cold = vals[-self.cold_min_samples:]
            cold_med = float(statistics.median(cold))
            cold_mad = self._mad(cold, cold_med)
            if cold_mad <= self.cold_max_mad_ms:
                self.current_lead_ms = max(self.min_lead_ms, min(self.max_lead_ms, cold_med))
                self.initialized = True
                updated = abs(self.current_lead_ms - before) > 1e-9
                update_reason = "COLD_MEDIAN"
        elif self.initialized:
            # Explicit change-point detector.  A mixed old/new window naturally has
            # large MAD, so long-window MAD must not veto a coherent new cluster.
            if len(vals) >= self.recent_shift_samples:
                recent = vals[-self.recent_shift_samples:]
                recent_med = float(statistics.median(recent))
                recent_mad = self._mad(recent, recent_med)
                delta = recent_med - self.current_lead_ms
                if abs(delta) > self.deadband_ms and recent_mad <= self.recent_shift_max_mad_ms:
                    step = max(-self.max_shift_step_ms, min(self.max_shift_step_ms, delta))
                    self.current_lead_ms = max(
                        self.min_lead_ms,
                        min(self.max_lead_ms, self.current_lead_ms + step),
                    )
                    updated = abs(self.current_lead_ms - before) > 1e-9
                    update_reason = "CONFIRMED_RECENT_LEVEL_SHIFT"

        if updated:
            self.updates_total += 1

        self.last_result = {
            "accepted": True,
            "ideal_lead_ms": ideal,
            "rolling_median_ms": med,
            "rolling_mad_ms": mad,
            "recent_median_ms": recent_med,
            "recent_mad_ms": recent_mad,
            "sample_count": len(vals),
            "updated": updated,
            "update_reason": update_reason,
            "step_ms": self.current_lead_ms - before,
            "lead_before_ms": before,
            "current_lead_ms": self.current_lead_ms,
            "initialized": self.initialized,
        }
        return dict(self.last_result)

    def telemetry(self) -> Dict[str, Any]:
        vals = list(self.samples)
        med = float(statistics.median(vals)) if vals else None
        mad = self._mad(vals, med) if med is not None else None
        recent = vals[-self.recent_shift_samples:]
        recent_med = float(statistics.median(recent)) if recent else None
        recent_mad = self._mad(recent, recent_med) if recent_med is not None else None
        return {
            "seed_lead_ms": self.seed_lead_ms,
            "current_lead_ms": self.current_lead_ms,
            "rolling_median_ms": med,
            "rolling_mad_ms": mad,
            "recent_median_ms": recent_med,
            "recent_mad_ms": recent_mad,
            "sample_count": len(vals),
            "accepted_total": self.accepted_total,
            "rejected_total": self.rejected_total,
            "updates_total": self.updates_total,
            "initialized": self.initialized,
            "deadband_ms": self.deadband_ms,
        }
