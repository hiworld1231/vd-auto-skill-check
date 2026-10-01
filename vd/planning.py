"""Pure scheduling contracts. This module never opens an input device.

A caller owns generations and must invalidate on capture loss, lifecycle change
or LMB release. Each generation owns one immutable unwrapped success occurrence;
update() may choose GREAT or GOOD inside it, but never another revolution.
"""
from dataclasses import dataclass
import math

from vd.motion import Motion
from vd.vision import Arc


@dataclass(frozen=True)
class SuccessOccurrence:
    great_start: float
    great_end: float
    great_center: float
    good_start: float | None
    good_end: float | None

    @property
    def success_end(self):
        return self.good_end if self.good_end is not None else self.great_end


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
    EPS=.25

    def __init__(self, *, lead_seconds, lead_uncertainty=.015, max_age=.080):
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
        self.occurrence=None
        self.good_aim=None
        self.fired=False
        self.current=None
        self.reason='NO_GENERATION'

    def _occurrence_from_start(self, target, good, start):
        great_start=float(start)
        great_end=great_start+target.width
        great_center=great_start+target.width/2
        good_start=good_end=None
        if good is not None:
            gap=(good.start-(target.start+target.width))%360
            if gap<=8:
                good_start=great_end+gap
                good_end=good_start+good.width
        return SuccessOccurrence(great_start,great_end,great_center,
                                 good_start,good_end)

    def begin(self, target: Arc, initial_phase: float, good: Arc | None = None, *,
              occurrence_start: float | None = None, allow_trailing_good=None):
        if not (math.isfinite(initial_phase) and math.isfinite(target.start)
                and math.isfinite(target.width) and 0<target.width<180):
            raise ValueError('Invalid generation geometry')
        if good is not None and not (math.isfinite(good.start)
                and math.isfinite(good.width) and 0<good.width<180):
            raise ValueError('Invalid GOOD geometry')
        if occurrence_start is not None and not math.isfinite(occurrence_start):
            raise ValueError('Invalid occurrence anchor')
        self.invalidate('NEW_GENERATION')
        self.generation+=1
        self.fired=False
        self.target=target
        self.good=good
        self.good_aim=None

        if occurrence_start is None:
            next_start=initial_phase+(target.start-initial_phase)%360
            previous=self._occurrence_from_start(target,good,next_start-360)
            if previous.great_start-self.EPS<=initial_phase<=previous.success_end+self.EPS:
                self.occurrence=previous
            else:
                self.occurrence=self._occurrence_from_start(target,good,next_start)
        else:
            self.occurrence=self._occurrence_from_start(target,good,occurrence_start)
        self.target_phase=self.occurrence.great_center

    def invalidate(self, reason):
        self.version+=1
        self.current=None
        self.reason=reason

    def _uncertainty(self, motion: Motion, phase: float):
        horizon=max(0,(phase-motion.phase)/motion.speed)
        return (max(.75,3*motion.residual)+motion.speed_scatter*horizon
                +motion.speed*self.lead_uncertainty)

    def _make_plan(self, motion, *, frame_at, now, aim, window_start,
                   window_end, grade, uncertainty):
        valid_until=min(frame_at,motion.last_motion_at)+self.max_age
        intended_press_at=motion.at+(aim-motion.phase)/motion.speed-self.lead
        latest_press_at=motion.at+(window_end-motion.phase)/motion.speed-self.lead
        press_at=max(now,intended_press_at)
        self.current=Plan(self.generation,self.version,press_at,intended_press_at,valid_until,
                          aim,uncertainty,latest_press_at,grade,
                          window_start,window_end-window_start)
        self.reason='PLANNED'
        return self.current

    def update(self, motion: Motion | None, *, frame_at: float, now: float):
        self.invalidate('NO_MOTION')
        if self.target is None or self.occurrence is None or self.fired or motion is None:
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

        occ=self.occurrence
        effect_phase=motion.phase_at(now+self.lead)
        if effect_phase>occ.success_end+self.EPS:
            self.reason='TARGET_PASSED'
            return None

        # Once this generation falls back to GOOD, keep one fixed aim. Without
        # this latch each fresh frame would choose the midpoint of a shorter
        # remaining tail and push the deadline forward indefinitely.
        if self.good_aim is not None:
            if occ.good_start is None or effect_phase>=occ.good_end-self.EPS:
                self.reason='TARGET_PASSED'
                return None
            remaining_start=max(occ.good_start,effect_phase)
            if effect_phase>self.good_aim+self.EPS:
                # The fixed aim was crossed between observations. The current
                # effect phase is still inside GOOD, so dispatch immediately
                # instead of moving the target farther into the tail.
                aim=effect_phase
            else:
                aim=self.good_aim
            return self._make_plan(
                motion,frame_at=frame_at,now=now,aim=aim,
                window_start=remaining_start,window_end=occ.good_end,
                grade='GOOD',uncertainty=self._uncertainty(motion,aim))

        great_uncertainty=self._uncertainty(motion,occ.great_center)
        if (effect_phase<=occ.great_center+self.EPS
                and great_uncertainty<=self.target.width/2):
            return self._make_plan(
                motion,frame_at=frame_at,now=now,aim=occ.great_center,
                window_start=occ.great_start,window_end=occ.great_end,
                grade='GREAT',uncertainty=great_uncertainty)

        if occ.good_start is not None and effect_phase<occ.good_end-self.EPS:
            remaining_start=max(occ.good_start,effect_phase)
            self.good_aim=(remaining_start+occ.good_end)/2
            return self._make_plan(
                motion,frame_at=frame_at,now=now,aim=self.good_aim,
                window_start=remaining_start,window_end=occ.good_end,
                grade='GOOD',uncertainty=self._uncertainty(motion,self.good_aim))

        # If no adjacent GOOD was detected, GREAT is the only known success
        # sector. Keep one attempt while the occurrence is still physically
        # reachable; never convert it into a retry one revolution later.
        if occ.good_start is None and effect_phase<=occ.great_end+self.EPS:
            aim=max(occ.great_center,effect_phase)
            return self._make_plan(
                motion,frame_at=frame_at,now=now,aim=aim,
                window_start=max(occ.great_start,effect_phase),window_end=occ.great_end,
                grade='GREAT',uncertainty=self._uncertainty(motion,aim))

        self.reason='TARGET_PASSED'
        return None

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
        aim=(self.occurrence.great_center if self.occurrence is not None
             else self.target.center)
        start=(self.occurrence.great_start if self.occurrence is not None
               else self.target.start)
        self.current=Plan(self.generation,self.version,now,now,valid_until,aim,180.,now,
                          'GREAT',start,self.target.width,timing_mode)
        self.reason=timing_mode
        return self.current

    def claim(self, plan: Plan, *, now: float, held: bool, capture_alive: bool):
        """Consume once immediately before dispatch, under the runtime lock."""
        if (not math.isfinite(now) or not held or not capture_alive or self.fired
                or self.current is not plan or plan.generation!=self.generation
                or plan.version!=self.version or now<plan.press_at
                or now>plan.valid_until):
            return False
        self.fired=True
        self.invalidate('CLAIMED')
        return True
