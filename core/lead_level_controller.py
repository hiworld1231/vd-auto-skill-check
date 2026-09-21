from __future__ import annotations

import math
import statistics
from collections import deque
from typing import Any, Deque, Dict, Optional


class LeadLevelController:
    """Robust session-level delivery lead estimator.

    Trusted observation invariant:
        ideal_lead_ms = effective_dispatch_lead_ms + center_error_ms

    Lead is a session-level delivery/capture/game delay, not a speed tier.
    Speed is handled by the trajectory predictor.  This controller therefore
    keeps one robust level and moves it only when a fresh coherent cluster says
    the level itself changed.
    """

    def __init__(
        self,
        seed_lead_ms: float,
        *,
        min_lead_ms: float = 35.0,
        max_lead_ms: float = 180.0,
        window_size: int = 15,
        cold_min_samples: int = 4,
        cold_max_mad_ms: float = 15.0,
        recent_shift_samples: int = 4,
        recent_shift_max_mad_ms: float = 10.0,
        deadband_ms: float = 5.0,
        max_shift_step_ms: float = 6.0,
        uncertainty_floor_ms: float = 2.0,
        uncertainty_default_ms: float = 12.0,
    ):
        self.seed_lead_ms = float(seed_lead_ms)
        self.current_lead_ms = float(seed_lead_ms)
        self.min_lead_ms = float(min_lead_ms)
        self.max_lead_ms = float(max_lead_ms)
        self.window_size = max(5, int(window_size))
        self.cold_min_samples = max(3, int(cold_min_samples))
        self.cold_max_mad_ms = float(cold_max_mad_ms)
        self.recent_shift_samples = max(3, int(recent_shift_samples))
        self.recent_shift_max_mad_ms = float(recent_shift_max_mad_ms)
        self.deadband_ms = float(deadband_ms)
        self.max_shift_step_ms = float(max_shift_step_ms)
        self.uncertainty_floor_ms = max(0.5, float(uncertainty_floor_ms))
        self.uncertainty_default_ms = max(
            self.uncertainty_floor_ms, float(uncertainty_default_ms)
        )

        self.samples: Deque[float] = deque(maxlen=self.window_size)
        self.accepted_total = 0
        self.rejected_total = 0
        self.updates_total = 0
        self.accepted_at_last_update = 0
        self.initialized = False
        self.restored_from_disk = False
        self._restored_uncertainty_ms: Optional[float] = None
        self.last_result: Dict[str, Any] = {}

    @staticmethod
    def _finite(v: Optional[float]) -> bool:
        try:
            return v is not None and math.isfinite(float(v))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _mad(vals, med: float) -> float:
        return (
            float(statistics.median(abs(float(v) - med) for v in vals))
            if vals
            else 0.0
        )

    def _clip(self, value: float) -> float:
        return max(self.min_lead_ms, min(self.max_lead_ms, float(value)))

    def _current_cluster(self):
        vals = list(self.samples)
        if not vals:
            return []
        n = self.recent_shift_samples if self.initialized else self.cold_min_samples
        return vals[-min(len(vals), n):]

    def restore_calibration(
        self, lead_ms: float, uncertainty_ms: float
    ) -> None:
        """Warm-start from a validated persistent level without fake samples."""
        if not self._finite(lead_ms) or not (
            self.min_lead_ms <= float(lead_ms) <= self.max_lead_ms
        ):
            raise ValueError("restored lead outside controller range")
        if not self._finite(uncertainty_ms):
            raise ValueError("restored uncertainty must be finite")
        self.current_lead_ms = self._clip(float(lead_ms))
        self.initialized = True
        self.restored_from_disk = True
        self._restored_uncertainty_ms = max(
            self.uncertainty_floor_ms, min(25.0, float(uncertainty_ms))
        )
        self.samples.clear()
        self.accepted_total = 0
        self.accepted_at_last_update = 0

    def get_uncertainty_ms(self) -> float:
        """Predictive one-check delivery uncertainty for fire-envelope math.

        The landing envelope needs the expected scatter of the next individual
        delivery, not the standard error of the estimated session median.
        Therefore the robust MAD scale is intentionally *not* divided by
        sqrt(n).  More samples make the center estimate more trustworthy, but
        they do not make per-check delivery variability disappear.
        """
        cluster = self._current_cluster()
        if (
            self.restored_from_disk
            and self._restored_uncertainty_ms is not None
            and self.accepted_total < self.recent_shift_samples
        ):
            return self._restored_uncertainty_ms
        if len(cluster) < 2:
            return self.uncertainty_default_ms
        med = float(statistics.median(cluster))
        mad = self._mad(cluster, med)
        estimate = 1.4826 * mad
        return max(self.uncertainty_floor_ms, min(25.0, estimate))

    def get_lead_ms(
        self, *, chain_count: int = 1, chain_offset_ms: float = 0.0
    ) -> float:
        # No unvalidated Frenzy offset.  Signature remains compatible with old
        # callers, but production passes zero.
        extra = float(chain_offset_ms) * max(0, int(chain_count) - 1)
        return self.current_lead_ms + extra

    def _reject(self, reason: str, **extra: Any) -> Dict[str, Any]:
        self.rejected_total += 1
        self.last_result = {
            "accepted": False,
            "reject_reason": reason,
            "current_lead_ms": self.current_lead_ms,
            "lead_uncertainty_ms": self.get_uncertainty_ms(),
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
        short_vs_long_delta_deg_s: Optional[float] = None,
        target_passed: bool = False,
        observed_response_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        # Frenzy does not expose the same clean freeze semantics.  Freeze the
        # learned session level across chain generations.
        if bool(frenzy_transition) or int(chain_count) > 1:
            return self._reject("FRENZY_UNVALIDATED")
        if bool(target_passed):
            return self._reject("TARGET_ALREADY_PASSED")
        if plateau_found is not True:
            return self._reject("NO_CONFIRMED_PLATEAU")
        if outcome not in {"GREAT", "GOOD", "MISS"}:
            return self._reject(f"OUTCOME_{outcome or 'UNKNOWN'}")
        if trigger_mode not in {"SCHEDULED", "IMMEDIATE"}:
            return self._reject(f"TRIGGER_{trigger_mode or 'UNKNOWN'}")
        if not self._finite(frame_age_ms) or float(frame_age_ms) > 20.0:
            return self._reject("STALE_DECODE_DELIVERY")
        if detector_fallback:
            return self._reject("DETECTOR_FALLBACK")
        if compensation_regime != "CONTINUOUS_MEASURED_SPEED":
            return self._reject(f"REGIME_{compensation_regime or 'UNKNOWN'}")

        if not self._finite(center_error_ms):
            return self._reject("NO_CENTER_ERROR")
        if not self._finite(actual_used_delay_ms):
            return self._reject("NO_USED_DELAY")
        # Only scheduled dispatches have meaningful scheduler jitter.
        # For IMMEDIATE fallback the desired center deadline may already be in
        # the past; dispatch_done-desired is planning lateness, not scheduler
        # execution error. Rejecting it here prevented exactly the late GOOD
        # landings that should teach the controller to raise normal lead.
        if trigger_mode == "SCHEDULED":
            if (
                not self._finite(scheduler_jitter_ms)
                or abs(float(scheduler_jitter_ms)) > 4.5
            ):
                return self._reject("SCHEDULER_JITTER")
        if fit_sample_count is None or int(fit_sample_count) < 5:
            return self._reject("SHORT_TRACK")
        if (
            self._finite(fit_residual_mad_deg)
            and float(fit_residual_mad_deg) > 3.0
        ):
            return self._reject("POOR_FIT_RESIDUAL")
        if (
            self._finite(fit_spread_deg_s)
            and float(fit_spread_deg_s) > 60.0
        ):
            return self._reject("UNSTABLE_SPEED_FIT")
        if abs(float(center_error_ms)) > 130.0:
            return self._reject("PHASE_OUTLIER")
        if white_source and not str(white_source).startswith("MEASURED"):
            return self._reject("RECONSTRUCTED_GREAT_GEOMETRY")
        if (
            self._finite(short_vs_long_delta_deg_s)
            and abs(float(short_vs_long_delta_deg_s)) > 75.0
        ):
            return self._reject(
                "RECENT_SPEED_DISAGREEMENT",
                short_vs_long_delta_deg_s=float(short_vs_long_delta_deg_s),
            )

        ideal = float(actual_used_delay_ms) + float(center_error_ms)
        if not (self.min_lead_ms <= ideal <= self.max_lead_ms):
            return self._reject("IDEAL_LEAD_OUT_OF_RANGE", ideal_lead_ms=ideal)

        response_disagreement_ms = None
        if self._finite(observed_response_ms):
            response_disagreement_ms = float(observed_response_ms) - ideal

        self.samples.append(ideal)
        self.accepted_total += 1
        vals = list(self.samples)
        rolling_med = float(statistics.median(vals))
        rolling_mad = self._mad(vals, rolling_med)

        before = self.current_lead_ms
        updated = False
        update_reason = None
        recent_med = None
        recent_mad = None

        if not self.initialized and len(vals) >= self.cold_min_samples:
            cold = vals[-self.cold_min_samples:]
            recent_med = float(statistics.median(cold))
            recent_mad = self._mad(cold, recent_med)
            if recent_mad <= self.cold_max_mad_ms:
                self.current_lead_ms = self._clip(recent_med)
                self.initialized = True
                updated = abs(self.current_lead_ms - before) > 1e-9
                self.accepted_at_last_update = self.accepted_total
                update_reason = "COLD_ROBUST_MEDIAN"

        elif self.initialized:
            fresh = self.accepted_total - self.accepted_at_last_update
            if fresh >= self.recent_shift_samples:
                recent = vals[-self.recent_shift_samples:]
                recent_med = float(statistics.median(recent))
                recent_mad = self._mad(recent, recent_med)
                delta = recent_med - self.current_lead_ms
                coherent = recent_mad <= self.recent_shift_max_mad_ms
                if coherent and self.restored_from_disk:
                    # Four fresh trusted samples validate the persisted prior.
                    # From here uncertainty is derived only from live samples.
                    self.restored_from_disk = False
                    self._restored_uncertainty_ms = None
                if coherent and abs(delta) >= self.deadband_ms:
                    step = max(
                        -self.max_shift_step_ms,
                        min(self.max_shift_step_ms, delta),
                    )
                    self.current_lead_ms = self._clip(self.current_lead_ms + step)
                    updated = abs(self.current_lead_ms - before) > 1e-9
                    self.accepted_at_last_update = self.accepted_total
                    update_reason = "ROBUST_LEVEL_SHIFT"

        if updated:
            self.updates_total += 1

        self.last_result = {
            "accepted": True,
            "ideal_lead_ms": ideal,
            "observation_source": "GEOMETRIC_DISPATCH_INVARIANT",
            "observed_response_ms": (
                float(observed_response_ms)
                if self._finite(observed_response_ms)
                else None
            ),
            "response_disagreement_ms": response_disagreement_ms,
            "rolling_median_ms": rolling_med,
            "rolling_mad_ms": rolling_mad,
            "recent_median_ms": recent_med,
            "recent_mad_ms": recent_mad,
            "lead_uncertainty_ms": self.get_uncertainty_ms(),
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
        cluster = self._current_cluster()
        cluster_med = float(statistics.median(cluster)) if cluster else None
        cluster_mad = (
            self._mad(cluster, cluster_med) if cluster_med is not None else None
        )
        return {
            "seed_lead_ms": self.seed_lead_ms,
            "current_lead_ms": self.current_lead_ms,
            "lead_uncertainty_ms": self.get_uncertainty_ms(),
            "lead_uncertainty_kind": "PREDICTIVE_ROBUST_SIGMA",
            "rolling_median_ms": med,
            "rolling_mad_ms": mad,
            "current_cluster_median_ms": cluster_med,
            "current_cluster_mad_ms": cluster_mad,
            "sample_count": len(vals),
            "accepted_total": self.accepted_total,
            "rejected_total": self.rejected_total,
            "updates_total": self.updates_total,
            "accepted_at_last_update": self.accepted_at_last_update,
            "accepted_since_update": (
                self.accepted_total - self.accepted_at_last_update
            ),
            "initialized": self.initialized,
            "restored_from_disk": self.restored_from_disk,
            "deadband_ms": self.deadband_ms,
            "recent_shift_samples": self.recent_shift_samples,
            "recent_shift_max_mad_ms": self.recent_shift_max_mad_ms,
        }
