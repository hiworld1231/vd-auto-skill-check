"""Independent per-image measurements; no wall clock and no hidden tracking.

Geometry is calibrated for the user's 1080p game UI. Red candidates require
radial support and local angular contrast, not just a high red-channel mean.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class Arc:
    start: float
    width: float

    @property
    def center(self):
        return (self.start + self.width / 2) % 360

    def contains(self, angle):
        return (angle - self.start) % 360 <= self.width


@dataclass(frozen=True)
class Needle:
    angle: float
    strength: float
    radial_support: float
    contrast: float
    angular_width: float


@dataclass(frozen=True)
class Measurement:
    timestamp: float
    center: tuple[float, float] | None
    prompt_score: float
    great: Arc | None
    good: Arc | None
    candidates: tuple[Needle, ...]
    reason: str
    great_source: str = 'MEASURED'


def same_arc(a, b):
    return abs((a.center-b.center+180)%360-180)<=3 and abs(a.width-b.width)<=3


def retained_target(m, *, great, good, center):
    """Retain an observed arc only when a thin needle occludes its boundary.

    This is explicitly tracked geometry, not a new per-image measurement.
    Missing white also requires the unchanged black arc as independent evidence.
    """
    if (great is None or center is None or m.center is None or m.prompt_score<.80
            or math.dist(center,m.center)>3 or len(m.candidates)!=1):
        return m
    needle=m.candidates[0]
    if not great.contains(needle.angle):
        return m
    if m.great is None:
        if good is None or m.good is None or not same_arc(good,m.good):
            return m
    else:
        if same_arc(great,m.great):
            return m
        offset=(m.great.start-great.start+180)%360-180
        end=offset+m.great.width
        if not (-1.5<=offset and end<=great.width+1.5 and m.great.width<great.width):
            return m
        start_visible=abs(offset)<=1.5
        end_visible=abs(end-great.width)<=1.5
        boundary=(m.great.start+m.great.width) if start_visible else m.great.start
        near_needle=abs((boundary-needle.angle+180)%360-180)<=needle.angular_width/2+2.5
        if not (start_visible or end_visible) or not near_needle:
            return m
    return replace(m,great=great,great_source='TRACKED_NEEDLE_OCCLUSION',reason='OK')


def arcs_from_score(score, minimum, maximum):
    """Circular positive runs with interpolated zero crossings, in degrees."""
    score = np.asarray(score, dtype=float)
    n = len(score)
    mask = score > 0
    starts = np.flatnonzero(mask & ~np.roll(mask, 1))
    result = []
    for start in starts:
        length = 1
        while length < n and mask[(start + length) % n]:
            length += 1
        end = (start + length - 1) % n
        before = score[(start - 1) % n]
        first = score[start]
        last = score[end]
        after = score[(end + 1) % n]
        a = start - 1 + (-before / (first - before))
        b = start + length - 1 + last / (last - after)
        width = (b - a) * 360 / n
        if minimum <= width <= maximum:
            result.append(Arc(float(a * 360 / n % 360), float(width)))
    return result


class Detector:
    def __init__(self, template=None, ring_radius=66.0):
        path = Path(template) if template else Path(__file__).resolve().parents[1] / 'assets/space_template.png'
        self.template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if self.template is None:
            raise ValueError(f'Cannot load prompt template: {path}')
        self.ring_radius = float(ring_radius)
        self.angles = np.arange(720, dtype=np.float32) * (math.pi / 360)

    def _sample(self, frame, center, radii):
        r = np.asarray(radii, np.float32)[None, :]
        x = center[0] + np.cos(self.angles)[:, None] * r
        y = center[1] + np.sin(self.angles)[:, None] * r
        return cv2.remap(frame, x, y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT).astype(np.float32)

    def measure(self, frame, timestamp, *, center_hint=None):
        if not math.isfinite(timestamp):
            raise ValueError('Frame timestamp must be finite')
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError('Expected uint8 BGR image')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        th, tw = self.template.shape
        h, w = gray.shape
        if h < th or w < tw:
            raise ValueError('ROI is smaller than prompt template')
        result = cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, pos = cv2.minMaxLoc(result)
        center = (pos[0] + tw/2, pos[1] + th/2) if score >= .80 else center_hint
        if center is None:
            return Measurement(timestamp, None, float(score), None, None, (), 'NO_PROMPT')
        cx, cy = center
        radius = self.ring_radius + 4
        if not (radius <= cx < w-radius and radius <= cy < h-radius):
            return Measurement(timestamp, center, float(score), None, None, (), 'RING_OUTSIDE_ROI')
        ring = self._sample(frame, center, np.linspace(self.ring_radius-3, self.ring_radius+3, 9))
        levels = ring.mean(axis=1)
        intensity = levels.mean(axis=1)
        white_threshold = min(185, max(160, float(np.median(intensity))+40))
        white_score = np.minimum(intensity-white_threshold, levels.min(axis=1)-150)
        black_threshold = max(42, min(55, float(np.median(intensity))-40))
        whites = arcs_from_score(white_score, 5, 16)
        blacks = arcs_from_score(black_threshold-intensity, 18, 65)
        pairs = [(a,b) for a in whites for b in blacks
                 if (b.start-a.start-a.width) % 360 <= 8]
        great, good = max(pairs, key=lambda ab: ab[0].width, default=(None,None))
        if great is None and len(whites)==1:
            great = whites[0]
        if good is None and len(blacks)==1:
            good=blacks[0]
        # A local angular ridge must persist along the radial segment. Broad
        # colored clothes/background can be red but do not form that ridge.
        samples = self._sample(frame, center, np.linspace(24, 61, 32))
        red = np.maximum(0, samples[:,:,2]-np.maximum(samples[:,:,0],samples[:,:,1]))
        side = np.maximum(np.roll(red,12,axis=0),np.roll(red,-12,axis=0))
        contrast = red-side
        support = ((red>12)&(contrast>7)).mean(axis=1)
        profile = np.maximum(contrast,0).mean(axis=1)
        peaks = np.flatnonzero((profile>=np.roll(profile,1)) & (profile>np.roll(profile,-1)) &
                              (profile>10) & (support>=.55))
        candidates=[]
        for k in sorted(peaks, key=lambda i:profile[i],reverse=True):
            if any(abs((k*.5-c.angle+180)%360-180)<5 for c in candidates):
                continue
            width=1
            for direction in (-1,1):
                j=1
                while j<30 and profile[(k+direction*j)%720]>.5*profile[k]:
                    width+=1;j+=1
            width*=.5
            if width>10:
                continue
            prev,mid,nxt=profile[(k-1)%720],profile[k],profile[(k+1)%720]
            denom=2*(2*mid-prev-nxt)
            delta=float((nxt-prev)/denom) if denom>1e-6 else 0
            candidates.append(Needle(float((k+delta)*.5%360),float(red[k].mean()),
                float(support[k]),float(profile[k]),width))
            if len(candidates)==4:break
        reason='OK' if great is not None and candidates else ('NO_TARGET' if great is None else 'NO_LINE_CANDIDATE')
        return Measurement(timestamp,center,float(score),great,good,tuple(candidates),reason)
