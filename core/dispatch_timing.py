from __future__ import annotations

import math
import statistics
from collections import deque
from typing import Any, Deque, Dict, Optional


class DispatchTimingModel:
    """Robust model from scheduler deadline to physical keydown visibility.

    The scheduler wakes at a userspace deadline, then the callback and input
    backend still need a small amount of time before UInput.syn()/press()
    completes.  That lag is measurable on every scheduled fire and should be
    compensated separately from the game's delivery lead.
    """

    def __init__(
        self,
        *,
        window_size: int = 21,
        min_samples: int = 3,
        max_sample_ms: float = 50.0,
        uncertainty_floor_ms: float = 0.20,
    ) -> None:
        self.samples_ms: Deque[float] = deque(maxlen=max(5, int(window_size)))
        self.min_samples = max(2, int(min_samples))
        self.max_sample_ms = max(5.0, float(max_sample_ms))
        self.uncertainty_floor_ms = max(0.05, float(uncertainty_floor_ms))
        self.rejected_total = 0

    @property
    def initialized(self) -> bool:
        return len(self.samples_ms) >= self.min_samples

    @staticmethod
    def _finite(value: Optional[float]) -> bool:
        try:
            return value is not None and math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def record(self, deadline: float, physical_keydown: float) -> Optional[float]:
        if not self._finite(deadline) or not self._finite(physical_keydown):
            self.rejected_total += 1
            return None
        lag_ms = (float(physical_keydown) - float(deadline)) * 1000.0
        # Small negative values can appear from timer granularity; they cannot
        # represent a real physical keydown before the callback deadline.
        if lag_ms < -0.5 or lag_ms > self.max_sample_ms:
            self.rejected_total += 1
            return None
        lag_ms = max(0.0, lag_ms)
        self.samples_ms.append(lag_ms)
        return lag_ms

    def compensation_ms(self) -> float:
        if not self.initialized:
            return 0.0
        return float(statistics.median(self.samples_ms))

    def uncertainty_ms(self) -> float:
        if not self.initialized:
            return 0.0
        vals = list(self.samples_ms)
        med = float(statistics.median(vals))
        mad = float(statistics.median(abs(x - med) for x in vals))
        return max(self.uncertainty_floor_ms, min(10.0, 1.4826 * mad))

    def deadline_for_physical_press(self, desired_press_time: float) -> float:
        return float(desired_press_time) - self.compensation_ms() / 1000.0

    def telemetry(self) -> Dict[str, Any]:
        vals = list(self.samples_ms)
        return {
            "initialized": self.initialized,
            "sample_count": len(vals),
            "compensation_ms": self.compensation_ms(),
            "uncertainty_ms": self.uncertainty_ms(),
            "rejected_total": self.rejected_total,
            "recent_samples_ms": vals[-7:],
        }
