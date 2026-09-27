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
    intended_press_at: float
    valid_until: float
    target_phase: float
    uncertainty_degrees: float
    latest_press_at: float
    target_grade: str
    target_window_start: float
    target_window_width: float
    timing_mode: str = 'PREDICTED'


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

        # Every visible check gets a GREAT attempt. Uncertainty is diagnostic,
        # never permission to redirect the press to GOOD or abandon the check.
        aim=self.target_phase
        window_start=self.target_phase-self.target.width/2
        window_width=self.target.width
        grade='GREAT'
        uncertainty=self._uncertainty(motion,aim)
        intended_press_at=motion.at+(aim-motion.phase)/motion.speed-self.lead
        # This is the nominal white-sector exit time for diagnostics. Model
        # uncertainty is reported separately and does not erase the window.
        latest_press_at=intended_press_at+window_width/(2*motion.speed)
        press_at=max(now,intended_press_at)
        self.current=Plan(self.generation,self.version,press_at,intended_press_at,valid_until,
                          aim,uncertainty,latest_press_at,grade,
                          window_start,window_width)
        self.reason='PLANNED'
        return self.current

    def attempt_now(self, *, frame_at: float, now: float, timing_mode: str):
        """Queue one immediate GREAT attempt when visible motion cannot be timed."""
        self.invalidate(timing_mode)
        if self.target is None or self.fired:
            return None
        if not all(math.isfinite(v) for v in (frame_at,now)):
            self.reason='INVALID_TIMING'
            return None
        valid_until=min(frame_at,now)+self.max_age
        if frame_at>now+.002 or now>valid_until:
            self.reason='STALE_OBSERVATION'
            return None
        aim=self.target_phase if self.target_phase is not None else self.target.center
        self.current=Plan(self.generation,self.version,now,now,valid_until,aim,180.,now,
                          'GREAT',self.target.start,self.target.width,timing_mode)
        self.reason=timing_mode
        return self.current

    def claim(self, plan: Plan, *, now: float, held: bool, capture_alive: bool):
        """Consume once immediately before dispatch, under the runtime lock."""
        if (not math.isfinite(now) or not held or not capture_alive or self.fired
                or self.current is not plan or plan.generation!=self.generation
                or plan.version!=self.version or now<plan.press_at
                or now>plan.valid_until):
            return False
        # The deadline is a target, not a veto. Dispatch records lateness so
        # a visible check still receives its single attempt.
        self.fired=True
        self.invalidate('CLAIMED')
        return True
