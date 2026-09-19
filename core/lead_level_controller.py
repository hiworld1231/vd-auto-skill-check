from __future__ import annotations

import math
import statistics
from collections import deque
from typing import Any, Deque, Dict, Optional


class LeadLevelController:
    """Robust session-level delivery lead policy for clean GEN_RUSH.

    Observation (controller invariant):
        ideal_lead_ms = effective_dispatch_lead_ms + center_error_ms

    For SCHEDULED and IMMEDIATE_SAFE fires alike, the value passed as
    actual_used_delay_ms must be the predictor's latest time-to-target at the
    physical dispatch frame. It is not the configured compensation value.

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
        window_size: int = 15,
        cold_min_samples: int = 4,
        deadband_ms: float = 12.0,
        cold_max_mad_ms: float = 15.0,
        recent_shift_samples: int = 7,
        recent_shift_max_mad_ms: float = 15.0,
        max_shift_step_ms: float = 8.0,
        fast_late_samples: int = 4,
        fast_late_error_ms: float = 8.0,
        fast_late_max_mad_ms: float = 12.0,
        max_late_step_ms: float = 16.0,
        cold_late_nudge_samples: int = 3,
        cold_late_nudge_error_ms: float = 20.0,
        cold_late_nudge_step_ms: float = 12.0,
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
        self.fast_late_samples = max(3, int(fast_late_samples))
        self.fast_late_error_ms = float(fast_late_error_ms)
        self.fast_late_max_mad_ms = float(fast_late_max_mad_ms)
        self.max_late_step_ms = float(max_late_step_ms)
        self.cold_late_nudge_samples = max(3, int(cold_late_nudge_samples))
        self.cold_late_nudge_error_ms = float(cold_late_nudge_error_ms)
        self.cold_late_nudge_step_ms = float(cold_late_nudge_step_ms)
        self.samples: Deque[float] = deque(maxlen=self.window_size)
        self.center_errors: Deque[float] = deque(maxlen=self.window_size)
        self.accepted_total = 0
        self.rejected_total = 0
        self.updates_total = 0
        self.accepted_at_last_update = 0
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
        short_vs_long_delta_deg_s: Optional[float] = None,
        target_passed: bool = False,
        observed_response_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        # Frenzy does not expose the normal post-hit freeze and must never
        # train the session lead.
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
            return self._reject("STALE_FIRE_FRAME")
        if detector_fallback:
            return self._reject("DETECTOR_FALLBACK")
        if compensation_regime != "CONTINUOUS_MEASURED_SPEED":
            return self._reject(f"REGIME_{compensation_regime or 'UNKNOWN'}")

        # Primary controller invariant.  This directly answers the question
        # "what lead would have landed this exact check at GREAT centre?"
        if not self._finite(center_error_ms):
            return self._reject("NO_CENTER_ERROR")
        if not self._finite(actual_used_delay_ms):
            return self._reject("NO_USED_DELAY")
        if not self._finite(scheduler_jitter_ms) or abs(float(scheduler_jitter_ms)) > 4.5:
            return self._reject("SCHEDULER_JITTER")
        if fit_sample_count is None or int(fit_sample_count) < 5:
            return self._reject("SHORT_TRACK")
        if self._finite(fit_residual_mad_deg) and float(fit_residual_mad_deg) > 3.0:
            return self._reject("POOR_FIT_RESIDUAL")
        if self._finite(fit_spread_deg_s) and float(fit_spread_deg_s) > 60.0:
            return self._reject("UNSTABLE_SPEED_FIT")
        if abs(float(center_error_ms)) > 90.0:
            return self._reject("PHASE_OUTLIER")
        if white_source and not str(white_source).startswith("MEASURED"):
            return self._reject("RECONSTRUCTED_GREAT_GEOMETRY")
        if self._finite(short_vs_long_delta_deg_s) and abs(float(short_vs_long_delta_deg_s)) > 75.0:
            return self._reject(
                "RECENT_SPEED_DISAGREEMENT",
                short_vs_long_delta_deg_s=float(short_vs_long_delta_deg_s),
            )

        ideal = float(actual_used_delay_ms) + float(center_error_ms)
        observation_source = "GEOMETRIC_DISPATCH_INVARIANT"

        # Freeze onset stays as an independent diagnostic.  When both physical
        # measurements disagree wildly, reject the sample rather than letting
        # either noisy observation move the session lead.
        response_disagreement_ms = None
        if self._finite(observed_response_ms):
            # Freeze onset is useful telemetry, but replay evidence shows it is
            # too noisy to veto an otherwise clean geometric landing sample.
            response_disagreement_ms = float(observed_response_ms) - ideal

        if not (self.min_lead_ms <= ideal <= self.max_lead_ms):
            return self._reject("IDEAL_LEAD_OUT_OF_RANGE", ideal_lead_ms=ideal)

        self.samples.append(ideal)
        self.center_errors.append(float(center_error_ms))
        self.accepted_total += 1
        vals = list(self.samples)
        med = float(statistics.median(vals))
        mad = self._mad(vals, med)
        before = self.current_lead_ms
        updated = False
        update_reason = None
        recent_med = None
        recent_mad = None

        if not self.initialized:
            # Before a stable cold cluster exists, do not sit at the seed while
            # several clean checks all land far on the late side.  A nudge
            # consumes its samples, so the same late cluster cannot staircase.
            cold_fresh = self.accepted_total - self.accepted_at_last_update
            if (
                len(vals) >= self.cold_late_nudge_samples
                and cold_fresh >= self.cold_late_nudge_samples
            ):
                cold_errors = list(self.center_errors)[-self.cold_late_nudge_samples:]
                late_votes = sum(
                    1 for e in cold_errors if e >= self.cold_late_nudge_error_ms
                )
                if late_votes >= self.cold_late_nudge_samples:
                    self.current_lead_ms = min(
                        self.max_lead_ms,
                        self.current_lead_ms + self.cold_late_nudge_step_ms,
                    )
                    updated = abs(self.current_lead_ms - before) > 1e-9
                    if updated:
                        self.accepted_at_last_update = self.accepted_total
                    update_reason = "COLD_LATE_NUDGE"

            if len(vals) >= self.cold_min_samples:
                cold = vals[-self.cold_min_samples:]
                cold_med = float(statistics.median(cold))
                cold_mad = self._mad(cold, cold_med)
                if cold_mad <= self.cold_max_mad_ms:
                    # Cold median is authoritative; it may replace a provisional
                    # nudge from the same outcome.
                    self.current_lead_ms = max(
                        self.min_lead_ms, min(self.max_lead_ms, cold_med)
                    )
                    self.initialized = True
                    updated = abs(self.current_lead_ms - before) > 1e-9
                    if updated:
                        self.accepted_at_last_update = self.accepted_total
                    update_reason = "COLD_MEDIAN"
        elif self.initialized:
            # Do not reuse the same observations for repeated moves.
            fresh_since_update = self.accepted_total - self.accepted_at_last_update

            # Asymmetric correction: a run of late landings is relatively safe
            # (it normally lands in the trailing GOOD sector), so converge faster.
            # Early landings are dangerous and are corrected only by the slower
            # seven-sample path below.
            if fresh_since_update >= self.fast_late_samples:
                late_ideals = vals[-self.fast_late_samples:]
                late_errors = list(self.center_errors)[-self.fast_late_samples:]
                late_med = float(statistics.median(late_ideals))
                late_mad = self._mad(late_ideals, late_med)
                late_votes = sum(
                    1 for e in late_errors if e >= self.fast_late_error_ms
                )
                delta = late_med - self.current_lead_ms
                if (
                    late_votes >= self.fast_late_samples - 1
                    and late_mad <= self.fast_late_max_mad_ms
                    and delta > 2.0
                ):
                    step = min(self.max_late_step_ms, delta)
                    self.current_lead_ms = max(
                        self.min_lead_ms,
                        min(self.max_lead_ms, self.current_lead_ms + step),
                    )
                    updated = abs(self.current_lead_ms - before) > 1e-9
                    if updated:
                        self.accepted_at_last_update = self.accepted_total
                    update_reason = "FAST_LATE_CORRECTION"
                    recent_med = late_med
                    recent_mad = late_mad

            # Slow symmetric/early change-point path.  It only runs if the fast
            # late path did not already consume this fresh cluster.
            if (
                not updated
                and fresh_since_update >= self.recent_shift_samples
                and len(vals) >= self.recent_shift_samples
            ):
                recent = vals[-self.recent_shift_samples:]
                recent_errors = list(self.center_errors)[-self.recent_shift_samples:]
                recent_med = float(statistics.median(recent))
                recent_mad = self._mad(recent, recent_med)
                delta = recent_med - self.current_lead_ms
                early_votes = sum(
                    1 for e in recent_errors if e <= -self.fast_late_error_ms
                )
                if (
                    delta < -self.deadband_ms
                    and early_votes >= self.recent_shift_samples - 2
                    and recent_mad <= self.recent_shift_max_mad_ms
                ):
                    step = max(-self.max_shift_step_ms, delta)
                    self.current_lead_ms = max(
                        self.min_lead_ms,
                        min(self.max_lead_ms, self.current_lead_ms + step),
                    )
                    updated = abs(self.current_lead_ms - before) > 1e-9
                    if updated:
                        self.accepted_at_last_update = self.accepted_total
                    update_reason = "CONFIRMED_EARLY_LEVEL_SHIFT"

        if updated:
            self.updates_total += 1

        self.last_result = {
            "accepted": True,
            "ideal_lead_ms": ideal,
            "observation_source": observation_source,
            "observed_response_ms": float(observed_response_ms) if self._finite(observed_response_ms) else None,
            "response_disagreement_ms": response_disagreement_ms,
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
            "accepted_at_last_update": self.accepted_at_last_update,
            "accepted_since_update": self.accepted_total - self.accepted_at_last_update,
            "initialized": self.initialized,
            "deadband_ms": self.deadband_ms,
            "fast_late_samples": self.fast_late_samples,
            "fast_late_error_ms": self.fast_late_error_ms,
            "cold_late_nudge_samples": self.cold_late_nudge_samples,
        }
