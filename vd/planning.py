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
        self.fired=False
        self.current=None
        self.reason='NO_GENERATION'
        self.allow_trailing_good=True
        self.target_occurrence_latched=False
        self.latched_good_start=None

    def begin(self, target: Arc, initial_phase: float, good: Arc | None = None, *,
              allow_trailing_good=True):
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
        self.allow_trailing_good=bool(allow_trailing_good)
        self.target_occurrence_latched=False
        self.latched_good_start=None
        self.target_phase=initial_phase+(target.center-initial_phase)%360

    def invalidate(self, reason):
        self.version+=1
        self.current=None
        self.reason=reason

    def _uncertainty(self, motion: Motion, phase: float):
        horizon=max(0,(phase-motion.phase)/motion.speed)
        return (max(.75,3*motion.residual)+motion.speed_scatter*horizon
                +motion.speed*self.lead_uncertainty)

    def _adjacent_good_occurrence(self):
        """Return GOOD beside the selected GREAT occurrence, unwrapped."""
        if self.good is None or self.target is None or self.target_phase is None:
            return None
        gap=(self.good.start-(self.target.start+self.target.width))%360
        if gap>8:
            return None
        target_start=self.target_phase-self.target.width/2
        good_start=target_start+self.target.width+gap
        return good_start,good_start+self.good.width

    def _visible_success_occurrence(self, phase):
        """Return the current GREAT/adjacent-GOOD occurrence containing phase."""
        if self.good is None:
            return None
        gap=(self.good.start-(self.target.start+self.target.width))%360
        if gap>8:
            return None
        target_start=float(self.target.start)
        while target_start>phase:
            target_start-=360
        while target_start+360<=phase:
            target_start+=360
        target_end=target_start+self.target.width
        good_start=target_end+gap
        good_end=good_start+self.good.width
        if (target_start<=phase<=target_end
                or good_start<=phase<=good_end):
            return target_start,good_start,good_end
        return None

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

        # Compare windows against where the needle is expected to be when an
        # input sent *now* can actually take effect. This matters when capture
        # cadence is low: a chained target may first be observed inside GOOD,
        # yet that same GOOD can still be reachable after the timing lead.
        effect_phase_now=motion.phase_at(now)+motion.speed*self.lead

        if self.target_occurrence_latched:
            good_end=self.latched_good_start+self.good.width
            if effect_phase_now>good_end+.25:
                self.reason='TARGET_PASSED'
                return None
        else:
            occurrence=self._visible_success_occurrence(motion.phase)
            if occurrence is not None:
                target_start,good_start,good_end=occurrence
                # A normal one-shot check must never re-arm the next rotation.
                # A chained check may use the visible occurrence only when an
                # input sent now can still land inside its success tail.
                if self.allow_trailing_good or effect_phase_now<=good_end+.25:
                    self.target_phase=target_start+self.target.width/2
                    self.target_occurrence_latched=True
                    self.latched_good_start=good_start
                    if effect_phase_now>good_end+.25:
                        self.reason='TARGET_PASSED'
                        return None

        target_start=self.target_phase-self.target.width/2
        target_end=target_start+self.target.width
        great_uncertainty=self._uncertainty(motion,self.target_phase)
        great_reachable=effect_phase_now<=target_end+.25

        if great_reachable and great_uncertainty<=self.target.width/2:
            aim=self.target_phase
            window_start=target_start
            window_width=self.target.width
            grade='GREAT'
            uncertainty=great_uncertainty
        else:
            adjacent=self._adjacent_good_occurrence()
            if adjacent is not None and effect_phase_now<=adjacent[1]+.25:
                good_start,good_end=adjacent
                aim=(good_start+good_end)/2
                window_start=good_start
                window_width=self.good.width
                grade='GOOD'
                uncertainty=self._uncertainty(motion,aim)
            elif self.target_occurrence_latched:
                self.reason='TARGET_PASSED'
                return None
            else:
                # No adjacent rescue sector exists. Preserve the historical
                # one-attempt contract rather than silently dropping the check.
                aim=self.target_phase
                window_start=target_start
                window_width=self.target.width
                grade='GREAT'
                uncertainty=great_uncertainty

        intended_press_at=motion.at+(aim-motion.phase)/motion.speed-self.lead
        window_end=window_start+window_width
        latest_press_at=motion.at+(window_end-motion.phase)/motion.speed-self.lead
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
        self.fired=True
        self.invalidate('CLAIMED')
        return True
