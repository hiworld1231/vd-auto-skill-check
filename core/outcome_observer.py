from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Tuple

from core.detectors.base import is_angle_in_arc


def _signed_delta(a: float, b: float) -> float:
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


Plateau = Tuple[float, float, float, int]


class OutcomeObserver:
    """Post-fire landing observer independent from the old adaptive learner."""

    # A 28 ms stable tail is useful immediate evidence for lifecycle decisions,
    # but live replays contain render pauses around 28-30 ms that later resume
    # motion. A plateau that remains stable for 60 ms is strong enough to keep
    # as the terminal landing candidate even if ring loss later makes the red
    # needle detector jump onto unrelated pixels.
    CANDIDATE_SPAN_S = 0.028
    CONFIRMED_SPAN_S = 0.060
    MAX_ADJACENT_GAP_S = 0.040
    MAX_SPREAD_DEG = 1.6

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
        self._confirmed_plateau: Optional[Plateau] = None

    def on_trigger(
        self,
        press_time: float,
        target_angle: float,
        speed_deg_s: float,
        white_zone: Optional[Dict[str, Any]],
        black_zone: Optional[Dict[str, Any]],
        used_latency_ms: Optional[float] = None,
        **_: Any,
    ) -> None:
        self.trigger_t = float(press_time)
        self.target_angle = float(target_angle) % 360.0
        self.speed_deg_s = abs(float(speed_deg_s)) if speed_deg_s else self.session_base_speed
        self.white_zone = dict(white_zone) if white_zone else None
        self.black_zone = dict(black_zone) if black_zone else None
        self.used_latency_ms = (
            float(used_latency_ms) if used_latency_ms is not None else self.initial_latency_ms
        )
        self.samples.clear()
        self._confirmed_plateau = None

    def observe_sample(self, t: float, angle: float, strength: float = 30.0) -> None:
        if self.trigger_t is None or t < self.trigger_t - 0.005 or strength < 10.0:
            return
        self.samples.append((float(t), float(angle) % 360.0, float(strength)))
        if len(self.samples) > 40:
            del self.samples[:-40]

    def _find_tail_plateau(self, min_span_s: float) -> Optional[Plateau]:
        """Return a stable suffix of the observations available right now.

        The old implementation scanned from the beginning and returned the
        first stable chunk it ever saw. That made a short mid-motion render
        pause remain a permanent landing even after the needle moved again.
        Only the current tail can represent the current freeze state.
        """
        min_samples = 3
        if len(self.samples) < min_samples or self.trigger_t is None:
            return None

        end = len(self.samples) - 1
        end_t, end_angle, _ = self.samples[end]
        ref = end_angle
        vals = [ref]
        start = end

        for i in range(end - 1, -1, -1):
            t, angle, _strength = self.samples[i]
            next_t = self.samples[i + 1][0]
            if next_t - t > self.MAX_ADJACENT_GAP_S:
                break
            value = ref + _signed_delta(angle, ref)
            candidate = vals + [value]
            if max(candidate) - min(candidate) > self.MAX_SPREAD_DEG:
                break
            vals = candidate
            start = i

        count = end - start + 1
        span_s = end_t - self.samples[start][0]
        if count < min_samples or span_s < float(min_span_s):
            return None

        hit = float(statistics.median(vals) % 360.0)
        response_ms = max(
            0.0, (self.samples[start][0] - self.trigger_t) * 1000.0
        )
        return hit, response_ms, span_s * 1000.0, count

    def _refresh_confirmed_plateau(self) -> None:
        if self._confirmed_plateau is not None:
            return
        confirmed = self._find_tail_plateau(self.CONFIRMED_SPAN_S)
        if confirmed is not None:
            self._confirmed_plateau = confirmed

    def has_recent_motion(self, sample_count: int = 3, min_span_deg: float = 2.0) -> bool:
        """Return whether recent trusted post-fire samples still show motion."""
        n = max(2, int(sample_count))
        if len(self.samples) < n:
            return False
        chunk = self.samples[-n:]
        ref = chunk[0][1]
        vals = [ref + _signed_delta(x[1], ref) for x in chunk]
        return (max(vals) - min(vals)) >= float(min_span_deg)

    def has_plateau(self) -> bool:
        """Return whether a trustworthy post-fire freeze is visible *now*."""
        self._refresh_confirmed_plateau()
        return self._find_tail_plateau(self.CANDIDATE_SPAN_S) is not None

    def conclude_check(
        self,
        frenzy_transition: bool = False,
        no_fire_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        if no_fire_reason:
            return {
                "outcome": "NO_FIRE",
                "no_fire_reason": no_fire_reason,
                "plateau_found": False,
                "hit_angle": None,
                "target_angle": self.target_angle,
                "observed_response_ms": None,
            }

        if frenzy_transition:
            # Frenzy transitions do not provide a normal terminal freeze.
            return {
                "outcome": "FRENZY_TRANSITION",
                "plateau_found": False,
                "hit_angle": None,
                "target_angle": self.target_angle,
                "frenzy_transition": True,
                "observed_response_ms": None,
            }

        self._refresh_confirmed_plateau()
        plateau = self._confirmed_plateau
        if plateau is None:
            # At terminal ring end, a still-current 28 ms candidate is useful
            # fallback evidence even if it did not live long enough to become a
            # cached 60 ms plateau. Historical candidates are never reused.
            plateau = self._find_tail_plateau(self.CANDIDATE_SPAN_S)
        if plateau is None:
            return {
                "outcome": "UNCONFIRMED",
                "plateau_found": False,
                "hit_angle": None,
                "target_angle": self.target_angle,
                "observed_response_ms": None,
            }

        hit, observed_response_ms, plateau_span_ms, plateau_sample_count = plateau
        white = self.white_zone
        black = self.black_zone
        if white and is_angle_in_arc(
            hit, float(white["start"]), float(white["end"]), 0.0, 0.0
        ):
            outcome = "GREAT"
        elif white and black and is_angle_in_arc(
            hit, float(white["end"]), float(black["end"]), 0.0, 0.0
        ):
            # CV masks leave a 1-7 degree segmentation gap between the visible
            # white GREAT arc and the following black GOOD arc. Physically this
            # is one continuous success sector, so do not manufacture MISSes in
            # the mask gap.
            outcome = "GOOD"
        elif black and is_angle_in_arc(
            hit, float(black["start"]), float(black["end"]), 0.0, 0.0
        ):
            outcome = "GOOD"
        else:
            outcome = "MISS"

        target = (
            self.target_angle
            if self.target_angle is not None
            else (float(white.get("center")) if white else hit)
        )
        err_deg = _signed_delta(hit, target)
        speed = max(20.0, float(self.speed_deg_s or self.session_base_speed))
        err_ms = err_deg / speed * 1000.0
        return {
            "outcome": outcome,
            "plateau_found": True,
            "hit_angle": hit,
            "target_angle": target,
            "error_deg": err_deg,
            "error_ms": err_ms,
            "center_error_deg": err_deg,
            "center_error_ms": err_ms,
            "observed_response_ms": observed_response_ms,
            "plateau_span_ms": plateau_span_ms,
            "plateau_sample_count": plateau_sample_count,
            "white_source": white.get("source") if white else None,
            "black_source": black.get("source") if black else None,
            "frenzy_transition": bool(frenzy_transition),
        }
