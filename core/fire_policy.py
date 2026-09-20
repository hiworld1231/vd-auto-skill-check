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

    First frame pre-arms from the normal-speed session prior; the second unique
    frame may replace that deadline from its adjacent-segment speed; robust
    measured fits then keep refining it.  Full GREAT-envelope containment is
    preferred, but an uncertain center estimate remains scheduled rather than
    being cancelled into a guaranteed NO_FIRE.
    """
    raw_time_until = pred.get("time_until_press_ms", 9999.0)
    time_until = float(9999.0 if raw_time_until is None else raw_time_until)
    white_source = str(pred.get("white_source") or "UNKNOWN")
    measured_geometry = white_source.startswith("MEASURED")
    reconstructed_geometry = white_source.startswith("RECONSTRUCTED")
    known_geometry = measured_geometry or reconstructed_geometry
    speed_source = str(pred.get("speed_source") or "MEASURED")
    provisional_speed = speed_source in {"SESSION_PRIOR", "SEGMENT_PROVISIONAL"}

    if not known_geometry:
        return GreatFireDecision(False, "GREAT_GEOMETRY_UNTRUSTED")

    # First-frame/second-frame pre-arm.  This is intentionally tentative:
    # subsequent unique frames continuously replace the pending deadline with
    # better measured-speed estimates.  Waiting for a 3-5 frame fit can make a
    # high-lead calibration physically impossible on short checks.
    if provisional_speed:
        if bool(pred.get("target_passed", False)):
            if bool(pred.get("reactive_safe_fallback", False)):
                return GreatFireDecision(
                    True, "PROVISIONAL_REACTIVE_SUCCESS", best_effort=True
                )
            return GreatFireDecision(False, "TOO_LATE_PROVISIONAL")
        if bool(pred.get("should_press_now", False)):
            return GreatFireDecision(
                True, "PROVISIONAL_IMMEDIATE_SUCCESS", best_effort=True
            )
        if time_until > 0.0:
            return GreatFireDecision(
                True, f"{speed_source}_PREARM", best_effort=True
            )
        return GreatFireDecision(False, "TOO_LATE_PROVISIONAL")

    # A full uncertainty interval inside GREAT is already a stronger condition
    # than the old separate 5-sample "stable" gate.  Do not wait twice.
    if speed_usable and bool(pred.get("great_interval_safe", False)):
        return GreatFireDecision(
            True,
            "GREAT_INTERVAL_SAFE" if measured_geometry
            else "RECONSTRUCTED_GREAT_SAFE_ENVELOPE",
            best_effort=not measured_geometry,
        )

    width = pred.get("landing_uncertainty_width_deg")
    great_width = pred.get("great_width_deg")
    intersects = bool(pred.get("great_interval_intersects", False))
    if (
        measured_geometry
        and speed_usable
        and time_until <= float(last_chance_ms)
        and intersects
        and width is not None
        and great_width is not None
        and float(width) <= 1.15 * float(great_width)
    ):
        return GreatFireDecision(
            True, "GREAT_LAST_CHANCE_INTERSECTION", best_effort=True
        )

    # If the ideal center deadline has already arrived, an immediate keydown is
    # still worthwhile when delivery remains inside the known success sector.
    if speed_usable and bool(pred.get("should_press_now", False)):
        return GreatFireDecision(
            True, "IMMEDIATE_SUCCESS_FALLBACK", best_effort=True
        )

    # Before the deadline, keep a center-targeted best-effort schedule alive.
    # Later frames continuously replace it with better estimates.
    if speed_usable and not bool(pred.get("target_passed", False)) and time_until > 0.0:
        return GreatFireDecision(
            True, "GREAT_CENTER_BEST_EFFORT", best_effort=True
        )

    return GreatFireDecision(False, "GREAT_INTERVAL_UNSAFE")
