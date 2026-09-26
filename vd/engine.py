"""Single-owner check lifecycle used by live dry-run and recorded replay.

Only observations create generations. Timeouts can end a generation, never
create one. There is no physical input here; callers must dispatch a claimed
plan under the same lock as all engine mutations.
"""
from dataclasses import asdict
import math

from vd.motion import MotionTracker, delta
from vd.planning import Planner


def same_target(a, b):
    return abs(delta(a.center,b.center))<=3 and abs(a.width-b.width)<=3


class Engine:
    def __init__(self, *, lead_seconds=.060, lead_uncertainty=.015,
                 absence_seconds=.12):
        self.planner=Planner(lead_seconds=lead_seconds,lead_uncertainty=lead_uncertainty)
        self.motion=MotionTracker()
        self.absence_seconds=absence_seconds
        self.active=False
        self.center=None
        self.target=None
        self.good=None
        self.pending=None
        self.pending_at=None
        self.last_visible=None
        self.last_timestamp=None
        self.started_at=None
        self.chain=0
        self.reason='WAITING_FOR_RING'
        self.events=[]

    @property
    def generation(self):
        return self.planner.generation

    def emit(self, kind, at, **values):
        self.events.append(dict(kind=kind,at=at,generation=self.generation,
                                chain=self.chain,**values))

    def take_events(self):
        events,self.events=self.events,[]
        return events

    def cancel(self, reason, now):
        self.planner.invalidate(reason)
        self.reason=reason
        if self.active:
            self.emit('END',now,reason=reason,pressed=self.planner.fired)
        self.active=False
        self.center=None
        self.target=None
        self.good=None
        self.pending=None
        self.pending_at=None
        self.last_visible=None
        self.motion.reset()
        self.chain=0

    def observe(self, m, *, now, held=True):
        if not math.isfinite(now):
            raise ValueError('Non-finite decision time')
        if self.last_timestamp is not None and m.timestamp<=self.last_timestamp:
            self.planner.invalidate('NONINCREASING_TIME')
            self.reason='NONINCREASING_TIME'
            return
        self.last_timestamp=m.timestamp
        self.planner.invalidate('NEW_OBSERVATION')
        if not held:
            self.cancel('LMB_RELEASED',now)
            return
        if m.timestamp>now+.002 or now-m.timestamp>self.planner.max_age:
            self.reason='STALE_FRAME'
            return
        visible=m.center is not None and m.great is not None
        if not visible:
            self.pending=None
            self.pending_at=None
            self.motion.update(m.timestamp,())
            self.reason='RING_NOT_VISIBLE'
            if self.last_visible is not None and m.timestamp-self.last_visible>=self.absence_seconds:
                self.cancel('RING_ENDED',now)
            return
        self.last_visible=m.timestamp
        shifted=(not self.active or not same_target(self.target,m.great)
                 or math.dist(self.center,m.center)>3)
        if shifted:
            self.reason='CONFIRMING_GEOMETRY'
            self.motion.reset_fit()
            if (self.pending is None or not same_target(self.pending[0],m.great)
                    or math.dist(self.pending[1],m.center)>3
                    or m.timestamp-self.pending_at>self.motion.max_gap):
                first=max(m.candidates,key=lambda c:c.contrast,default=None)
                self.pending=(m.great,m.center,first.angle if first else None)
                self.pending_at=m.timestamp
                return
            if m.timestamp-self.pending_at<.012 or not m.candidates:
                return
            # Strong competing red lines need motion acquisition, not a guess
            # used to select a possibly wrong target occurrence.
            ranked=sorted(m.candidates,key=lambda c:c.contrast,reverse=True)
            if len(ranked)>1 and ranked[1].contrast>.8*ranked[0].contrast:
                self.reason='AMBIGUOUS_NEEDLE'
                return
            prior_fired=self.active and self.planner.fired
            if self.active:
                self.emit('END',now,reason='GEOMETRY_CHANGED',pressed=self.planner.fired)
            self.chain=self.chain+1 if prior_fired else 1
            self.target=m.great
            self.good=m.good
            self.center=m.center
            initial=self.pending[2] if self.pending[2] is not None else ranked[0].angle
            self.planner.begin(m.great,initial,m.good)
            self.motion.reset()
            self.motion.unwrap_floor=initial
            self.active=True
            self.started_at=m.timestamp
            self.pending=None
            self.pending_at=None
            self.emit('BEGIN',now,target=asdict(m.great))
        else:
            self.pending=None
            self.pending_at=None
        estimate=self.motion.update(m.timestamp,m.candidates)
        if self.planner.fired:
            self.reason='OBSERVING_AFTER_PRESS'
            return
        plan=self.planner.update(estimate,frame_at=m.timestamp,now=now)
        self.reason=self.planner.reason if estimate is not None else self.motion.reason
        if plan:
            self.reason='PLANNED'

    def poll(self, now, *, held=True, capture_alive=True):
        if not held or not capture_alive:
            self.cancel('LMB_RELEASED' if not held else 'CAPTURE_LOST',now)
            return None
        plan=self.planner.current
        if plan is not None and now>plan.valid_until:
            self.planner.invalidate('OBSERVATION_EXPIRED')
            self.reason='OBSERVATION_EXPIRED'
            return None
        if plan is None:
            return None
        if not self.planner.claim(plan,now=now,held=held,capture_alive=capture_alive):
            if self.planner.current is None:
                self.reason=self.planner.reason
            return None
        self.emit('PRESS_CLAIM',now,plan=asdict(plan))
        self.reason='OBSERVING_AFTER_PRESS'
        return plan

    def snapshot(self):
        return dict(generation=self.generation,chain=self.chain,active=self.active,
                    lead_seconds=self.planner.lead,lead_uncertainty=self.planner.lead_uncertainty,
                    reason=self.reason,motion_reason=self.motion.reason,
                    motion=asdict(self.motion.estimate) if self.motion.estimate else None,
                    plan=asdict(self.planner.current) if self.planner.current else None)
