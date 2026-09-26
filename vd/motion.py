"""Robust phase and speed from timestamped independent measurements.

No session speed prior and no key dispatch. A frame discontinuity invalidates
motion. Latest phase comes from a robust fitted line, not the newest pixel.
"""
from collections import deque
from dataclasses import dataclass
import math
import statistics


def delta(a,b):
    return (a-b+180)%360-180


@dataclass(frozen=True)
class Motion:
    at: float
    phase: float
    speed: float
    residual: float
    speed_scatter: float
    last_motion_at: float
    samples: int

    def phase_at(self, now):
        return self.phase+self.speed*(now-self.at)


class MotionTracker:
    def __init__(self, window=.22, max_gap=.055):
        self.window=window
        self.max_gap=max_gap
        self.points=deque(maxlen=64)
        self.last_frame=None
        self.last_angle=None
        self.last_unwrapped=None
        self.last_accepted_at=None
        self.unwrap_floor=None
        self.estimate=None
        self.reason='EMPTY'

    def reset(self):
        self.points.clear()
        self.last_frame=None
        self.last_angle=None
        self.last_unwrapped=None
        self.last_accepted_at=None
        self.unwrap_floor=None
        self.estimate=None
        self.reason='EMPTY'

    def reset_fit(self):
        """Discard velocity samples while retaining the observed revolution."""
        floor=self.last_unwrapped if self.last_unwrapped is not None else self.unwrap_floor
        self.reset()
        self.unwrap_floor=floor

    def update(self, timestamp, candidates):
        if not math.isfinite(timestamp):
            raise ValueError('Non-finite timestamp')
        if self.last_frame is not None and timestamp<=self.last_frame:
            self.reason='NONINCREASING_TIME'
            return None
        if self.last_frame is not None and timestamp-self.last_frame>self.max_gap:
            self.reset_fit()
            self.reason='FRAME_GAP'
        if self.last_accepted_at is not None and timestamp-self.last_accepted_at>self.max_gap:
            # Delivery can continue while the needle disappears or is frozen.
            # Reacquire from scratch instead of joining disconnected tracks.
            self.reset_fit()
            self.reason='TRACK_GAP'
        self.last_frame=timestamp
        while self.points and timestamp-self.points[0][0]>self.window:
            self.points.popleft()
        if not candidates:
            self.estimate=None
            self.reason='NO_NEEDLE'
            return None
        if self.estimate:
            expected=self.estimate.phase_at(timestamp)%360
            chosen=min(candidates,key=lambda c:abs(delta(c.angle,expected)))
        elif self.last_angle is not None:
            chosen=min(candidates,key=lambda c:abs(delta(c.angle,self.last_angle)))
        else:
            chosen=max(candidates,key=lambda c:c.contrast)
        angle=chosen.angle
        if self.estimate is not None:
            innovation=abs(delta(angle,self.estimate.phase_at(timestamp)%360))
            if innovation>max(5.0,3*self.estimate.residual):
                # Keep the preceding fit only for its original freshness lease.
                # A rejected pixel must not move the unwrap anchor or renew it.
                self.reason='PHASE_OUTLIER'
                return self.estimate
        if self.last_angle is None:
            # Reacquisition drops the fit, not the turn already observed.
            # Otherwise a post-wrap frame could turn a passed target into an
            # apparent target one rotation in the future.
            if self.unwrap_floor is None:
                unwrapped=angle
            else:
                step=delta(angle,self.unwrap_floor)
                # The same <=2° backward pixel noise is tolerated during
                # reacquisition as during an established track.
                unwrapped=self.unwrap_floor+(step if step>=-2 else step+360)
        else:
            step=delta(angle,self.last_angle)
            if abs(step)<.5:
                # Repeated images do not add independent fit samples or renew
                # last_motion_at. Leave freeze detection to lifecycle.
                if self.estimate and timestamp-self.estimate.last_motion_at>self.max_gap:
                    self.estimate=None
                self.reason='REPEATED_ANGLE'
                return self.estimate
            if step<-2 or step>90:
                self.estimate=None
                self.reason='ANGLE_DISCONTINUITY'
                return None
            unwrapped=self.last_unwrapped+step
        self.last_angle=angle
        self.last_unwrapped=unwrapped
        self.last_accepted_at=timestamp
        self.points.append((timestamp,unwrapped))
        if len(self.points)<4 or self.points[-1][0]-self.points[0][0]<.04:
            self.estimate=None
            self.reason='MEASURING_MOTION'
            return None
        points=list(self.points)
        slopes=[(b[1]-a[1])/(b[0]-a[0]) for i,a in enumerate(points)
                for b in points[i+1:] if b[0]-a[0]>=.025]
        if not slopes:
            return None
        speed=statistics.median(slopes)
        if not 20<=speed<=1800:
            self.estimate=None
            self.reason='INVALID_SPEED'
            return None
        # Center time avoids precision loss from large monotonic timestamps.
        offsets=[a-speed*(t-timestamp) for t,a in points]
        phase=statistics.median(offsets)
        residual=statistics.median(abs(x-phase) for x in offsets)
        scatter=1.4826*statistics.median(abs(x-speed) for x in slopes)
        if residual>3 or scatter>max(45,.20*speed):
            self.estimate=None
            self.reason='INCOHERENT_MOTION'
            return None
        self.estimate=Motion(timestamp,phase,speed,residual,scatter,timestamp,len(points))
        self.reason='MEASURED'
        return self.estimate
