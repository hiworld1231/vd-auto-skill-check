from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class GreatFireDecision:
    allow: bool
    reason: str
    best_effort: bool = False


def decide_great_fire(
    pred: Dict[str, Any],
    *,
    fit_stable: bool,
    speed_usable: bool,
    urgent_window_ms: float = 35.0,
    last_chance_ms: float = 5.0,
) -> GreatFireDecision:
    """Gate scheduling by whether the predicted landing envelope fits GREAT.

    Normal path: require a current stable fit and the full uncertainty envelope
    inside the measured white zone.

    Short-track path: inside the final urgent window we allow a usable measured
    speed, but still require the full GREAT-safe envelope.

    Last-chance path: only in the final few milliseconds, permit an envelope
    that intersects GREAT when it is at most slightly wider than the white arc.
    This avoids turning harmless estimator uncertainty into guaranteed NO_FIRE,
    while still refusing broad/poorly constrained shots.
    """
    raw_time_until = pred.get("time_until_press_ms", 9999.0)
    time_until = float(9999.0 if raw_time_until is None else raw_time_until)
    urgent = bool(speed_usable and time_until <= float(urgent_window_ms))
    fit_ready = bool(fit_stable or urgent)
    if not fit_ready:
        return GreatFireDecision(False, "FIT_NOT_READY")

    if bool(pred.get("great_interval_safe", False)):
        return GreatFireDecision(True, "GREAT_INTERVAL_SAFE")

    width = pred.get("landing_uncertainty_width_deg")
    great_width = pred.get("great_width_deg")
    intersects = bool(pred.get("great_interval_intersects", False))
    if (
        speed_usable
        and time_until <= float(last_chance_ms)
        and intersects
        and width is not None
        and great_width is not None
        and float(width) <= 1.15 * float(great_width)
    ):
        return GreatFireDecision(True, "GREAT_LAST_CHANCE_INTERSECTION", best_effort=True)

    return GreatFireDecision(False, "GREAT_INTERVAL_UNSAFE")
