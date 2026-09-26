"""Pure scheduling contracts. This module never opens an input device.

A caller owns generations and must invalidate on capture loss, lifecycle change
or LMB release. The deadline alone is never permission to dispatch.
"""
from dataclasses import dataclass
import math

from vd.motion import Motion
from vd.vision import Arc


@dataclass(frozen=True)
class Plan:
    generation: int
    version: int
    press_at: float
    valid_until: float
    target_phase: float
    uncertainty_degrees: float
    latest_press_at: float
    target_grade: str
    target_window_start: float
    target_window_width: float


class Planner:
    def __init__(self, *, lead_seconds, lead_uncertainty=.015, max_age=.050):
        if not all(math.isfinite(v) for v in (lead_seconds,lead_uncertainty,max_age)):
            raise ValueError('Non-finite timing configuration')
        if lead_seconds<0 or lead_uncertainty<0 or max_age<=0:
            raise ValueError('Invalid timing configuration')
        self.lead=lead_seconds
        self.lead_uncertainty=lead_uncertainty
        self.max_age=max_age
        self.generation=0
        self.version=0
        self.target_phase=None
        self.target=None
        self.good=None
        self.fired=False
        self.current=None
        self.reason='NO_GENERATION'

    def begin(self, target: Arc, initial_phase: float, good: Arc | None = None):
        if not (math.isfinite(initial_phase) and math.isfinite(target.start)
                and math.isfinite(target.width) and 0<target.width<180):
            raise ValueError('Invalid generation geometry')
        if good is not None and not (math.isfinite(good.start)
                and math.isfinite(good.width) and 0<good.width<180):
            raise ValueError('Invalid GOOD geometry')
        self.invalidate('NEW_GENERATION')
        self.generation+=1
        self.fired=False
        self.target=target
        self.good=good
        # Fix this occurrence once. Passing it must not schedule a full turn.
        self.target_phase=initial_phase+(target.center-initial_phase)%360

    def invalidate(self, reason):
        self.version+=1
        self.current=None
        self.reason=reason

    def _uncertainty(self, motion: Motion, phase: float):
        horizon=max(0,(phase-motion.phase)/motion.speed)
        return (max(.75,3*motion.residual)+motion.speed_scatter*horizon
                +motion.speed*self.lead_uncertainty)

    def _good_fallback_window(self):
        """Return the current unwrapped success window, or None.

        Detector pairs GOOD immediately after GREAT (allowing a small measured
        boundary gap). Both outcomes are accepted by the game, so when GREAT's
        narrow envelope cannot contain the timing uncertainty we may aim at the
        wider combined success interval instead of refusing to press entirely.
        """
        if self.good is None:
            return None
        great_start=self.target_phase-self.target.width/2
        offset=(self.good.start-self.target.start)%360
        # Keep this fallback tied to the GOOD arc paired with this GREAT. A
        # remote arc must never turn into permission to target another region.
        if offset>self.target.width+8:
            return None
        good_start=great_start+offset
        end=good_start+self.good.width
        if end<=great_start or end-great_start>=180:
            return None
        return great_start,end

    def update(self, motion: Motion | None, *, frame_at: float, now: float):
        self.invalidate('NO_MOTION')
        if self.target is None or self.fired or motion is None:
            return None
        if not all(math.isfinite(v) for v in (frame_at,now,motion.at,motion.phase,
                motion.speed,motion.residual,motion.speed_scatter,motion.last_motion_at)):
            self.reason='INVALID_TIMING'
            return None
        valid_until=min(frame_at,motion.last_motion_at)+self.max_age
        if frame_at>now+.002 or motion.at>now+.002 or now>valid_until:
            self.reason='STALE_OBSERVATION'
            return None
        if motion.speed<=0 or motion.residual<0 or motion.speed_scatter<0:
            self.reason='INVALID_MOTION'
            return None

        # GREAT is always the first choice. Only widen to the paired acceptable
        # GOOD interval when the predicted error no longer fits inside GREAT.
        aim=self.target_phase
        window_start=self.target_phase-self.target.width/2
        window_width=self.target.width
        grade='GREAT'
        uncertainty=self._uncertainty(motion,aim)
        margin=window_width/2-uncertainty
        if margin<0:
            fallback=self._good_fallback_window()
            if fallback is None:
                self.reason='TARGET_UNCERTAIN'
                return None
            window_start,window_end=fallback
            window_width=window_end-window_start
            aim=window_start+window_width/2
            uncertainty=self._uncertainty(motion,aim)
            margin=window_width/2-uncertainty
            if margin<0:
                self.reason='TARGET_UNCERTAIN'
                return None
            grade='GOOD_FALLBACK'

        press_at=motion.at+(aim-motion.phase)/motion.speed-self.lead
        landing=motion.phase_at(now+self.lead)
        # Speed uncertainty also grows while a timer wakes late.
        latest_press_at=press_at+margin/(motion.speed+motion.speed_scatter)
        if press_at<now:
            if abs(landing-aim)>margin or now>latest_press_at:
                self.reason='TARGET_PASSED'
                return None
            press_at=now
        self.current=Plan(self.generation,self.version,press_at,valid_until,
                          aim,uncertainty,latest_press_at,grade,
                          window_start,window_width)
        self.reason='PLANNED'
        return self.current

    def claim(self, plan: Plan, *, now: float, held: bool, capture_alive: bool):
        """Consume once immediately before dispatch, under the runtime lock."""
        if (not math.isfinite(now) or not held or not capture_alive or self.fired
                or self.current is not plan or plan.generation!=self.generation
                or plan.version!=self.version or now<plan.press_at
                or now>plan.valid_until):
            return False
        # A late wake can miss the planned success envelope even while the
        # observation is fresh. Runtime must update before claiming.
        if now-plan.press_at>.002:
            self.invalidate('MISSED_DEADLINE')
            return False
        if now>plan.latest_press_at:
            self.invalidate('TARGET_WINDOW_PASSED')
            return False
        self.fired=True
        self.invalidate('CLAIMED')
        return True
