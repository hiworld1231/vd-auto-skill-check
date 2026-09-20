from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple


NONE = "NONE"
FRENZY_ABSENCE_REAPPEAR = "FRENZY_ABSENCE_REAPPEAR"
FRENZY_RELOCATED_MOTION = "FRENZY_RELOCATED_MOTION"
RING_END = "RING_END"


def _signed_delta(a: float, b: float) -> float:
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class LifecycleDecision:
    action: str = NONE
    reason: Optional[str] = None
    motion_spread_deg: float = 0.0
    unique_motion_samples: int = 0


class FrenzyLifecycle:
    """Post-fire lifecycle classifier.

    Landing detection belongs to OutcomeObserver.  This class only decides
    whether the current generation ended, or whether there is enough
    independent evidence for a Frenzy generation transition.
    """

    def __init__(
        self,
        *,
        min_transition_age_s: float = 0.070,
        min_absence_s: float = 0.040,
        ring_end_timeout_s: float = 0.120,
        zone_move_frames: int = 2,
    ) -> None:
        self.min_transition_age_s = float(min_transition_age_s)
        self.min_absence_s = float(min_absence_s)
        self.ring_end_timeout_s = float(ring_end_timeout_s)
        self.zone_move_frames = max(1, int(zone_move_frames))
        self.reset()

    def reset(self) -> None:
        self.fire_t: Optional[float] = None
        self.absent_since: Optional[float] = None
        self.zone_move_streak = 0
        self.landed = False
        self._motion: Deque[Tuple[float, float]] = deque(maxlen=10)

    def on_fire(self, t: float) -> None:
        self.reset()
        self.fire_t = float(t)

    def _add_motion(self, t: float, angle: Optional[float]) -> None:
        if angle is None:
            return
        t = float(t)
        angle = float(angle) % 360.0
        if self._motion:
            last_t, last_angle = self._motion[-1]
            if t <= last_t:
                return
            # Decoded 120 Hz often contains duplicate Roblox 60 Hz content.
            # Keep only angle-changing observations as unique motion evidence.
            if abs(_signed_delta(angle, last_angle)) < 0.20:
                return
        self._motion.append((t, angle))

    def _continuing_motion(self, expected_speed: float) -> Tuple[bool, float, int]:
        if len(self._motion) < 3:
            return False, 0.0, len(self._motion)

        pts = list(self._motion)[-4:]
        ref = pts[0][1]
        unwrapped = [ref]
        positive_steps = 0
        for _, angle in pts[1:]:
            step = _signed_delta(angle, unwrapped[-1] % 360.0)
            if step < -1.5:
                return False, 0.0, len(pts)
            if step > 0.20:
                positive_steps += 1
            unwrapped.append(unwrapped[-1] + step)

        spread = max(unwrapped) - min(unwrapped)
        span = max(0.0, pts[-1][0] - pts[0][0])
        # Replay-backed floor: real Frenzy transitions retained >=4.2 degrees
        # of post-fire motion while the known false transition froze at 0.
        # Also scale gently with measured speed/span so late fast chains need
        # proportionate evidence without waiting for another full render frame.
        required = max(3.0, abs(float(expected_speed or 0.0)) * span * 0.15)
        return positive_steps >= 2 and spread >= required, spread, len(pts)

    def observe(
        self,
        *,
        now: float,
        ring_present: bool,
        needle_angle: Optional[float],
        zone_moved: bool,
        plateau_found: bool,
        expected_speed: float,
    ) -> LifecycleDecision:
        now = float(now)
        if self.fire_t is None:
            return LifecycleDecision()

        if plateau_found:
            self.landed = True

        if not ring_present:
            if self.absent_since is None:
                self.absent_since = now
            absent_for = now - self.absent_since
            if absent_for >= self.ring_end_timeout_s:
                return LifecycleDecision(RING_END, "RING_ABSENT_TIMEOUT")
            return LifecycleDecision()

        # Reappearance after a meaningful gap is the strongest Frenzy signal,
        # but a confirmed freeze always wins: normal landing animations must
        # not be promoted into a new generation.
        if self.absent_since is not None:
            absent_for = now - self.absent_since
            self.absent_since = None
            if not self.landed and absent_for >= self.min_absence_s:
                return LifecycleDecision(
                    FRENZY_ABSENCE_REAPPEAR,
                    f"ABSENCE_REAPPEAR_{absent_for * 1000.0:.1f}MS",
                )

        self._add_motion(now, needle_angle)

        if zone_moved:
            self.zone_move_streak += 1
        else:
            self.zone_move_streak = 0

        age = now - self.fire_t
        if self.landed or age < self.min_transition_age_s:
            return LifecycleDecision()

        moving, spread, n = self._continuing_motion(expected_speed)
        if self.zone_move_streak >= self.zone_move_frames and moving:
            return LifecycleDecision(
                FRENZY_RELOCATED_MOTION,
                "RELOCATED_ZONE_PLUS_CONTINUING_MOTION",
                spread,
                n,
            )

        return LifecycleDecision(
            NONE,
            None,
            spread,
            n,
        )
