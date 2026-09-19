from __future__ import annotations

import math
import statistics
from collections import deque
from typing import Any, Deque, Dict, Optional


class LeadLevelController:
    """Slow, robust session-level delivery-lead policy for GEN_RUSH.

    We observe the controller-independent quantity:

        ideal_lead_ms = actual_used_delay_ms + center_error_ms

    The current check's lead is frozen before FIRE; any update affects only later
    checks.  The policy intentionally does *not* chase every landing.  It uses a
    cold-start median and then a median-of-9 level-shift rule with a wide deadband.
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
        max_mad_ms: float = 12.0,
    ):
        self.seed_lead_ms=float(seed_lead_ms)
        self.current_lead_ms=float(seed_lead_ms)
        self.min_lead_ms=float(min_lead_ms)
        self.max_lead_ms=float(max_lead_ms)
        self.window_size=max(5,int(window_size))
        self.cold_min_samples=max(3,int(cold_min_samples))
        self.deadband_ms=float(deadband_ms)
        self.max_mad_ms=float(max_mad_ms)
        self.samples:Deque[float]=deque(maxlen=self.window_size)
        self.accepted_total=0
        self.rejected_total=0
        self.updates_total=0
        self.initialized=False
        self.last_result:Dict[str,Any]={}

    @staticmethod
    def _finite(v:Optional[float])->bool:
        try: return v is not None and math.isfinite(float(v))
        except (TypeError,ValueError): return False

    @staticmethod
    def _mad(vals,med:float)->float:
        return float(statistics.median(abs(float(v)-med) for v in vals)) if vals else 0.0

    def get_lead_ms(self, *, chain_count:int=1, chain_offset_ms:float=0.0)->float:
        extra=float(chain_offset_ms)*max(0,int(chain_count)-1)
        return self.current_lead_ms+extra

    def record_outcome(
        self, *, center_error_ms:Optional[float], actual_used_delay_ms:Optional[float],
        outcome:Optional[str], plateau_found:Optional[bool], trigger_mode:Optional[str],
        scheduler_jitter_ms:Optional[float], frame_age_ms:Optional[float],
        detector_fallback:bool, compensation_regime:Optional[str],
        fit_sample_count:Optional[int], fit_residual_mad_deg:Optional[float],
        fit_spread_deg_s:Optional[float],
    )->Dict[str,Any]:
        reason=None
        if not self._finite(center_error_ms): reason='NO_CENTER_ERROR'
        elif not self._finite(actual_used_delay_ms): reason='NO_USED_DELAY'
        elif plateau_found is not True: reason='NO_CONFIRMED_PLATEAU'
        elif outcome not in {'GREAT','GOOD','MISS'}: reason=f'OUTCOME_{outcome or "UNKNOWN"}'
        elif trigger_mode!='SCHEDULED': reason=f'TRIGGER_{trigger_mode or "UNKNOWN"}'
        elif not self._finite(scheduler_jitter_ms) or abs(float(scheduler_jitter_ms))>3.0: reason='SCHEDULER_JITTER'
        elif not self._finite(frame_age_ms) or float(frame_age_ms)>20.0: reason='STALE_FIRE_FRAME'
        elif detector_fallback: reason='DETECTOR_FALLBACK'
        elif compensation_regime!='CONTINUOUS_MEASURED_SPEED': reason=f'REGIME_{compensation_regime or "UNKNOWN"}'
        elif fit_sample_count is None or int(fit_sample_count)<5: reason='SHORT_TRACK'
        elif self._finite(fit_residual_mad_deg) and float(fit_residual_mad_deg)>3.0: reason='POOR_FIT_RESIDUAL'
        elif self._finite(fit_spread_deg_s) and float(fit_spread_deg_s)>45.0: reason='UNSTABLE_SPEED_FIT'
        elif abs(float(center_error_ms))>90.0: reason='PHASE_OUTLIER'

        if reason:
            self.rejected_total+=1
            self.last_result={'accepted':False,'reject_reason':reason,'current_lead_ms':self.current_lead_ms,'sample_count':len(self.samples),'initialized':self.initialized}
            return dict(self.last_result)

        ideal=float(actual_used_delay_ms)+float(center_error_ms)
        if not (self.min_lead_ms<=ideal<=self.max_lead_ms):
            self.rejected_total+=1
            self.last_result={'accepted':False,'reject_reason':'IDEAL_LEAD_OUT_OF_RANGE','ideal_lead_ms':ideal,'current_lead_ms':self.current_lead_ms,'sample_count':len(self.samples),'initialized':self.initialized}
            return dict(self.last_result)

        self.samples.append(ideal); self.accepted_total+=1
        vals=list(self.samples); med=float(statistics.median(vals)); mad=self._mad(vals,med)
        before=self.current_lead_ms; updated=False; update_reason=None

        # Cold start: three internally-consistent landings are enough to leave the
        # deliberately late/safe seed.  We snap to their median once, not step by step.
        if not self.initialized and len(vals)>=self.cold_min_samples and mad<=self.max_mad_ms:
            self.current_lead_ms=max(self.min_lead_ms,min(self.max_lead_ms,med))
            self.initialized=True
            updated=abs(self.current_lead_ms-before)>1e-9
            update_reason='COLD_MEDIAN'
        # Warm state: only a clear level shift gets followed.  Small landing noise
        # inside the deadband is intentionally ignored.
        elif self.initialized and len(vals)>=5 and mad<=self.max_mad_ms:
            if abs(med-self.current_lead_ms)>self.deadband_ms:
                self.current_lead_ms=max(self.min_lead_ms,min(self.max_lead_ms,med))
                updated=abs(self.current_lead_ms-before)>1e-9
                update_reason='LEVEL_SHIFT_MEDIAN9'

        if updated: self.updates_total+=1
        self.last_result={
            'accepted':True,'ideal_lead_ms':ideal,'rolling_median_ms':med,'rolling_mad_ms':mad,
            'sample_count':len(vals),'updated':updated,'update_reason':update_reason,
            'step_ms':self.current_lead_ms-before,'lead_before_ms':before,
            'current_lead_ms':self.current_lead_ms,'initialized':self.initialized,
        }
        return dict(self.last_result)

    def telemetry(self)->Dict[str,Any]:
        vals=list(self.samples); med=float(statistics.median(vals)) if vals else None
        mad=self._mad(vals,med) if med is not None else None
        return {'seed_lead_ms':self.seed_lead_ms,'current_lead_ms':self.current_lead_ms,
                'rolling_median_ms':med,'rolling_mad_ms':mad,'sample_count':len(vals),
                'accepted_total':self.accepted_total,'rejected_total':self.rejected_total,
                'updates_total':self.updates_total,'initialized':self.initialized,
                'deadband_ms':self.deadband_ms}
