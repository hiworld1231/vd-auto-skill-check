from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Tuple

from core.detectors.base import is_angle_in_arc


def _signed_delta(a: float, b: float) -> float:
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


class OutcomeObserver:
    """Post-fire landing observer independent from the old adaptive learner."""
    def __init__(self, initial_latency_ms: float, session_base_speed: float = 278.0):
        self.initial_latency_ms = float(initial_latency_ms)
        self.session_base_speed = float(session_base_speed)
        self.reset()

    def reset(self) -> None:
        self.trigger_t: Optional[float] = None
        self.target_angle: Optional[float] = None
        self.speed_deg_s: Optional[float] = None
        self.white_zone: Optional[Dict[str, Any]] = None
        self.black_zone: Optional[Dict[str, Any]] = None
        self.used_latency_ms: Optional[float] = None
        self.samples: List[Tuple[float, float, float]] = []

    def on_trigger(self, press_time: float, target_angle: float, speed_deg_s: float,
                   white_zone: Optional[Dict[str, Any]], black_zone: Optional[Dict[str, Any]],
                   used_latency_ms: Optional[float] = None, **_: Any) -> None:
        self.trigger_t = float(press_time)
        self.target_angle = float(target_angle) % 360.0
        self.speed_deg_s = abs(float(speed_deg_s)) if speed_deg_s else self.session_base_speed
        self.white_zone = dict(white_zone) if white_zone else None
        self.black_zone = dict(black_zone) if black_zone else None
        self.used_latency_ms = float(used_latency_ms) if used_latency_ms is not None else self.initial_latency_ms
        self.samples.clear()

    def observe_sample(self, t: float, angle: float, strength: float = 30.0) -> None:
        if self.trigger_t is None or t < self.trigger_t - 0.005 or strength < 10.0:
            return
        self.samples.append((float(t), float(angle) % 360.0, float(strength)))
        if len(self.samples) > 40:
            del self.samples[:-40]

    def _find_plateau(self, allow_before_new_motion: bool) -> Optional[float]:
        if len(self.samples) < 3:
            return None
        for i in range(len(self.samples) - 2):
            chunk = self.samples[i:i+3]
            if chunk[-1][0] - chunk[0][0] < 0.018:
                continue
            ref = chunk[0][1]
            vals = [ref + _signed_delta(x[1], ref) for x in chunk]
            if max(vals) - min(vals) <= 1.4:
                return statistics.median(vals) % 360.0
        if not allow_before_new_motion and len(self.samples) >= 4:
            chunk = self.samples[-4:]
            ref = chunk[0][1]
            vals = [ref + _signed_delta(x[1], ref) for x in chunk]
            if max(vals) - min(vals) <= 1.0:
                return statistics.median(vals) % 360.0
        return None

    def conclude_check(self, frenzy_transition: bool = False, no_fire_reason: Optional[str] = None) -> Dict[str, Any]:
        if no_fire_reason:
            return {"outcome": "NO_FIRE", "no_fire_reason": no_fire_reason, "plateau_found": False,
                    "hit_angle": None, "target_angle": self.target_angle}
        hit = self._find_plateau(bool(frenzy_transition))
        if hit is None:
            return {"outcome": "UNCONFIRMED", "plateau_found": False, "hit_angle": None,
                    "target_angle": self.target_angle}
        white = self.white_zone
        black = self.black_zone
        if white and is_angle_in_arc(hit, float(white["start"]), float(white["end"]), 0.0, 0.0):
            outcome = "GREAT"
        elif black and is_angle_in_arc(hit, float(black["start"]), float(black["end"]), 0.0, 0.0):
            outcome = "GOOD"
        else:
            outcome = "MISS"
        target = self.target_angle if self.target_angle is not None else (float(white.get("center")) if white else hit)
        err_deg = _signed_delta(hit, target)
        speed = max(20.0, float(self.speed_deg_s or self.session_base_speed))
        err_ms = err_deg / speed * 1000.0
        return {
            "outcome": outcome, "plateau_found": True, "hit_angle": hit, "target_angle": target,
            "error_deg": err_deg, "error_ms": err_ms, "center_error_deg": err_deg, "center_error_ms": err_ms,
            "white_source": white.get("source") if white else None,
            "black_source": black.get("source") if black else None,
            "frenzy_transition": bool(frenzy_transition),
        }
