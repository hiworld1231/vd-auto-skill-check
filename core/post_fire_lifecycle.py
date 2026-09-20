from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _circ(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


WAIT = "WAIT"
LANDED = "LANDED_FREEZE"
FRENZY = "FRENZY_CONFIRMED"
RING_END = "RING_END"


@dataclass(frozen=True)
class LifecycleDecision:
    state: str
    reason: Optional[str] = None
    absence_ms: Optional[float] = None
    relocated_streak: int = 0


class PostFireLifecycle:
    """Evidence-based post-fire lifecycle for normal checks and Frenzy.

    Strong signal:
      sustained ring absence followed by reappearance.

    Fallback signal for transitions without a visible disappearance:
      no landing plateau + the same relocated zone on several unique frames +
      motion measured by the fresh-generation scan, after enough time has
      elapsed for a normal landing plateau to become visible.

    A confirmed freeze always vetoes visual relocation/rollback artifacts.
    """

    def __init__(
        self,
        *,
        min_reappear_absence_s: float = 0.030,
        ring_end_absence_s: float = 0.180,
        relocation_not_before_s: float = 0.125,
        relocation_frames: int = 3,
        relocation_center_tolerance_deg: float = 6.0,
    ) -> None:
        self.min_reappear_absence_s = float(min_reappear_absence_s)
        self.ring_end_absence_s = float(ring_end_absence_s)
        self.relocation_not_before_s = float(relocation_not_before_s)
        self.relocation_frames = max(2, int(relocation_frames))
        self.relocation_center_tolerance_deg = max(
            0.5, float(relocation_center_tolerance_deg)
        )
        self.reset()

    def reset(self) -> None:
        self.fire_time: Optional[float] = None
        self.absent_since: Optional[float] = None
        self.relocated_streak = 0
        self.relocated_center: Optional[float] = None
        self.rollback_seen = False

    def begin(self, fire_time: float) -> None:
        self.reset()
        self.fire_time = float(fire_time)

    def update(
        self,
        now: float,
        *,
        ring_present: bool,
        plateau_found: bool,
        zone_moved: bool = False,
        zone_center: Optional[float] = None,
        rollback: bool = False,
        fresh_motion: bool = False,
    ) -> LifecycleDecision:
        now = float(now)
        fire_time = self.fire_time if self.fire_time is not None else now
        elapsed = max(0.0, now - fire_time)

        if not ring_present:
            if self.absent_since is None:
                self.absent_since = now
            self.relocated_streak = 0
            self.relocated_center = None
            absent_s = max(0.0, now - self.absent_since)
            if absent_s >= self.ring_end_absence_s:
                return LifecycleDecision(
                    RING_END,
                    "SUSTAINED_ABSENCE",
                    absence_ms=absent_s * 1000.0,
                )
            return LifecycleDecision(
                WAIT,
                "RING_ABSENT_PENDING",
                absence_ms=absent_s * 1000.0,
            )

        # Ring is visible now.  A real disappearance/reappearance is the
        # strongest new-generation signal and does not depend on zone color.
        if self.absent_since is not None:
            absent_s = max(0.0, now - self.absent_since)
            self.absent_since = None
            if not plateau_found and absent_s >= self.min_reappear_absence_s:
                return LifecycleDecision(
                    FRENZY,
                    "ABSENCE_REAPPEAR",
                    absence_ms=absent_s * 1000.0,
                )

        if plateau_found:
            self.relocated_streak = 0
            self.relocated_center = None
            self.rollback_seen = False
            return LifecycleDecision(LANDED, "FREEZE_PLATEAU")

        if rollback:
            self.rollback_seen = True

        if zone_moved:
            if zone_center is None:
                # Compatibility for generic callers/tests. Runtime supplies the
                # actual center so real Frenzy requires spatial consistency.
                self.relocated_streak += 1
            else:
                center = float(zone_center) % 360.0
                if (
                    self.relocated_center is None
                    or _circ(center, self.relocated_center)
                    > self.relocation_center_tolerance_deg
                ):
                    self.relocated_center = center
                    self.relocated_streak = 1
                else:
                    self.relocated_streak += 1
        else:
            self.relocated_streak = 0
            self.relocated_center = None

        # Rollback/old-needle motion is telemetry only. Persistent-relocation
        # fallback requires motion from the freshly scanned generation itself.
        if (
            elapsed >= self.relocation_not_before_s
            and self.relocated_streak >= self.relocation_frames
            and fresh_motion
        ):
            return LifecycleDecision(
                FRENZY,
                "PERSISTENT_STABLE_RELOCATION_WITH_NEW_MOTION",
                relocated_streak=self.relocated_streak,
            )

        return LifecycleDecision(
            WAIT,
            "POST_FIRE_TRACK",
            relocated_streak=self.relocated_streak,
        )
