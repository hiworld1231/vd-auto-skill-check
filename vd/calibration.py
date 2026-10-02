"""CV freeze evidence and session-local latency estimates.

Media-time phase extrapolation yields an observed response delay, not a
hardware constant or server-confirmed game result. A provisional plateau can
be revoked by later motion; evidence is finalized only when the ring ends.
"""
from dataclasses import dataclass
import math
import statistics

from vd.motion import delta


@dataclass(frozen=True)
class Landing:
    label: str
    angle: float | None
    latency: float | None
    uncertainty: float | None
    eligible: bool
    reason: str
    dispatch_lag: float = 0.0


class FreezeObserver:
    def __init__(self, *, motion, press_at, frame_at, great, good=None, chain=1,
                 late_dispatch=False):
        self.motion=motion
        self.press_at=press_at
        self.great=great
        self.good=good
        self.chain=chain
        self.last_time=frame_at
        self.intervals=[]
        self.plateau=[]
        self.plateau_start=None
        self.plateau_origin=None
        self.offset_low=0.
        self.offset_high=0.
        self.tainted='LATE_DISPATCH' if late_dispatch else None

    def _clear_plateau(self):
        self.plateau=[]
        self.plateau_start=None
        self.plateau_origin=None
        self.offset_low=self.offset_high=0.

    def _has_stable_suffix(self):
        return len(self.plateau)>=2 and self.plateau[-1][0]-self.plateau_start>=.015

    def feed(self, measurement):
        t=measurement.timestamp
        if t<=self.last_time:
            self.tainted='NONINCREASING_TIME'
            return
        gap=t-self.last_time
        self.last_time=t
        if gap>.055:
            self.tainted='FRAME_GAP'
            self._clear_plateau()
        if t<self.press_at:
            return
        self.intervals.append(gap)
        self.intervals=self.intervals[-32:]
        if measurement.great is not None and (
                abs(delta(measurement.great.center,self.great.center))>3
                or abs(measurement.great.width-self.great.width)>3):
            self._clear_plateau()
            return
        if not measurement.candidates:
            if measurement.great is not None and not self._has_stable_suffix():
                self.tainted='NEEDLE_LOST'
                self._clear_plateau()
            return
        if len(measurement.candidates)!=1:
            self.tainted='AMBIGUOUS_NEEDLE'
            self._clear_plateau()
            return
        angle=measurement.candidates[0].angle
        if self.plateau:
            value=delta(angle,self.plateau_origin)
            low=min(self.offset_low,value)
            high=max(self.offset_high,value)
            if high-low>1.5:
                self._clear_plateau()
            else:
                self.offset_low,self.offset_high=low,high
        if not self.plateau:
            self.plateau_start=t
            self.plateau_origin=angle
        self.plateau.append((t,angle))
        self.plateau=self.plateau[-32:]

    def finish(self, reason):
        if not self._has_stable_suffix():
            return Landing('UNCONFIRMED',None,None,None,False,'NO_STABLE_SUFFIX')
        origin=self.plateau_origin
        offsets=[delta(a,origin) for _,a in self.plateau]
        angle=(origin+statistics.median(offsets))%360
        label=('CV_GREAT' if self.great.contains(angle) else
               'CV_GOOD' if self.good is not None and self.good.contains(angle) else 'CV_MISS')
        rejection=self.tainted
        if reason!='RING_ENDED':rejection='END_'+reason
        if self.chain!=1:rejection='CHAIN_TRANSITION'
        m=self.motion
        if m.samples<6 or m.residual>1.5 or m.speed_scatter>.08*m.speed:
            rejection='UNRELIABLE_PREFIRE_FIT'
        if self.plateau_start-self.press_at>.35:
            rejection='LATE_PLATEAU'
        phase_at_press=m.phase_at(self.press_at)
        angular_delay=delta(angle,phase_at_press%360)
        latency=angular_delay/m.speed
        if not 0<=latency<=.250 or m.speed*.250>=180:
            rejection='AMBIGUOUS_RESPONSE_DELAY'
        period=statistics.median(self.intervals) if self.intervals else .050
        spread=self.offset_high-self.offset_low
        uncertainty=(max(.75,3*m.residual)+spread/2)/m.speed
        uncertainty+=abs(latency)*m.speed_scatter/m.speed+period/2
        return Landing(label,float(angle),float(latency),float(uncertainty),
                       rejection is None,rejection or 'CV_FREEZE_ESTIMATE')


class LeadEstimator:
    """Learn scheduler/output delay only; visual freeze never calibrates lead."""
    def __init__(self, lead, uncertainty):
        self.lead=lead
        self.uncertainty=uncertainty
        self.samples=[]
        self.reason='INITIAL_MODEL'

    def observe_dispatch(self, *, at, dispatch_lag, physical, eligible=True):
        if not physical:
            self.reason='NONPHYSICAL'
            return False
        if not eligible:
            self.reason='INELIGIBLE_DISPATCH'
            return False
        if (not all(math.isfinite(v) for v in (at,dispatch_lag))
                or not 0<=dispatch_lag<=.250):
            self.reason='INVALID_DISPATCH'
            return False
        if self.samples and (at<=self.samples[-1][0] or at-self.samples[-1][0]>120):
            self.samples=[]
        self.samples=[s for s in self.samples if at-s[0]<=120]
        self.samples.append((at,dispatch_lag))
        self.samples=self.samples[-4:]
        if len(self.samples)<2:
            self.reason='COLLECTING'
            return False
        values=[s[1] for s in self.samples]
        low=min(values)
        high=max(values)
        if high-low>.025:
            self.reason='INCONSISTENT_DISPATCH'
            return False

        # Dispatch latency on a normal desktop is not Gaussian. Most keydowns
        # are cheap, but scheduler/uinput stalls occasionally create a long
        # positive tail. Median/MAD alone therefore understates the exact error
        # that matters for a narrow GREAT window. Use the Chebyshev center of
        # the recent accepted interval: it minimizes the worst observed timing
        # error, while the half-range explicitly carries that tail into planner
        # uncertainty for the next checks.
        center=(low+high)/2
        median=statistics.median(values)
        scatter=1.4826*statistics.median(abs(v-median) for v in values)
        envelope=(high-low)/2
        self.lead=center
        self.uncertainty=max(.003,scatter,envelope)
        self.reason='FRESH_DISPATCH_GROUP'
        return True

    def observe(self, landing, *, at, physical):
        """Compatibility wrapper for old callers and recordings."""
        return self.observe_dispatch(at=at,dispatch_lag=landing.dispatch_lag,
                                     physical=physical,eligible=True)
