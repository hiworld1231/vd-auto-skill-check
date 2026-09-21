from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Tuple

from core.detectors.base import is_angle_in_arc


def _signed_delta(a: float, b: float) -> float:
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


class OutcomeObserver:
    """Post-fire landing observer independent from the old adaptive learner."""

    def __init__(
        self,
        initial_latency_ms: float,
        session_base_speed: float = 278.0,
    ):
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
        self.chain_count: int = 1
        self.min_plateau_samples: int = 5
        self.min_plateau_span_s: float = 0.045
        self.continuity_rejections: int = 0
        self.last_continuity_rejection: Optional[Dict[str, float]] = None

    def on_trigger(
        self,
        press_time: float,
        target_angle: float,
        speed_deg_s: float,
        white_zone: Optional[Dict[str, Any]],
        black_zone: Optional[Dict[str, Any]],
        used_latency_ms: Optional[float] = None,
        chain_count: int = 1,
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
        self.chain_count = max(1, int(chain_count))
        # Normal checks need a longer freeze proof because 120-FPS capture can
        # duplicate a ~60-Hz rendered needle long enough to fake a 28-36ms
        # plateau. Archived recent normal GREAT/GOOD checks all retain at least
        # 45ms / 5 trusted samples, while Frenzy end-of-chain checks can be much
        # shorter and keep the old 28ms / 3-sample rule.
        if self.chain_count == 1:
            self.min_plateau_samples = 5
            self.min_plateau_span_s = 0.045
        else:
            self.min_plateau_samples = 3
            self.min_plateau_span_s = 0.028
        self.samples.clear()
        self.continuity_rejections = 0
        self.last_continuity_rejection = None

    def observe_sample(self, t: float, angle: float, strength: float = 30.0) -> None:
        if self.trigger_t is None or t < self.trigger_t - 0.005 or strength < 10.0:
            return
        self.samples.append((float(t), float(angle) % 360.0, float(strength)))
        if len(self.samples) > 40:
            del self.samples[:-40]

    def _implied_delivery_ms(self, hit_angle: float) -> Optional[float]:
        if (
            self.target_angle is None
            or self.speed_deg_s is None
            or self.used_latency_ms is None
        ):
            return None
        speed = max(20.0, float(self.speed_deg_s))
        center_error_ms = _signed_delta(hit_angle, self.target_angle) / speed * 1000.0
        return float(self.used_latency_ms) + center_error_ms

    def note_continuity_rejection(
        self,
        *,
        t: float,
        angle: float,
        delta_deg: float,
        max_forward_deg: float,
    ) -> None:
        self.continuity_rejections += 1
        self.last_continuity_rejection = {
            "t": float(t),
            "angle": float(angle) % 360.0,
            "delta_deg": float(delta_deg),
            "max_forward_deg": float(max_forward_deg),
        }

    def _find_plateau(self) -> Optional[Tuple[float, float, float, int]]:
        """Return earliest time-supported stable freeze.

        Capture can decode near 120 FPS while Roblox only changes the rendered
        needle near 60 Hz.  Counting four equal decoded frames is therefore not
        enough evidence: two duplicated source frames can look like a freeze.
        Require both angular stability and a real monotonic time span.
        """
        # Live checks can remove the ring quickly after a hit.  Four samples
        # over 35 ms was too slow and produced UNCONFIRMED even on real hits.
        # Keep the time-supported guard so duplicate decoded frames alone are
        # insufficient, but allow confirmation from three stable samples over
        # at least 28 ms. This is still faster than the old 35 ms gate, while
        # four 120-FPS duplicates spanning only ~25 ms cannot fake a landing.
        min_samples = int(self.min_plateau_samples)
        min_span_s = float(self.min_plateau_span_s)
        max_adjacent_gap_s = 0.040
        max_spread_deg = 1.6

        if len(self.samples) < min_samples or self.trigger_t is None:
            return None

        for i in range(len(self.samples) - min_samples + 1):
            ref = self.samples[i][1]
            vals = []
            previous_t = None
            for j in range(i, len(self.samples)):
                t, angle, _strength = self.samples[j]
                if previous_t is not None and t - previous_t > max_adjacent_gap_s:
                    break
                previous_t = t
                vals.append(ref + _signed_delta(angle, ref))
                if max(vals) - min(vals) > max_spread_deg:
                    break

                span_s = t - self.samples[i][0]
                count = j - i + 1
                if count >= min_samples and span_s >= min_span_s:
                    hit = float(statistics.median(vals) % 360.0)
                    response_ms = max(
                        0.0, (self.samples[i][0] - self.trigger_t) * 1000.0
                    )
                    return hit, response_ms, span_s * 1000.0, count
        return None

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
        """Return whether a trustworthy post-fire freeze is already visible."""
        return self._find_plateau() is not None

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
            # Frenzy transitions do not provide a normal freeze plateau.
            return {
                "outcome": "FRENZY_TRANSITION",
                "plateau_found": False,
                "hit_angle": None,
                "target_angle": self.target_angle,
                "frenzy_transition": True,
                "observed_response_ms": None,
            }

        plateau = self._find_plateau()
        if plateau is None:
            rejected = self.last_continuity_rejection
            return {
                "outcome": "UNCONFIRMED",
                "unconfirmed_reason": (
                    "POSTFIRE_CONTINUITY_LOST"
                    if self.continuity_rejections > 0
                    else "NO_TRUSTED_PLATEAU"
                ),
                "plateau_found": False,
                "plateau_trusted": False,
                "hit_angle": (
                    rejected.get("angle") if rejected is not None else None
                ),
                "target_angle": self.target_angle,
                "continuity_rejections": int(self.continuity_rejections),
                "last_continuity_rejection": rejected,
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
            # white GREAT arc and the following black GOOD arc.  Physically this
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
            "plateau_trusted": True,
            "hit_angle": hit,
            "target_angle": target,
            "error_deg": err_deg,
            "error_ms": err_ms,
            "center_error_deg": err_deg,
            "center_error_ms": err_ms,
            "observed_response_ms": observed_response_ms,
            "implied_delivery_ms": self._implied_delivery_ms(hit),
            "continuity_rejections": int(self.continuity_rejections),
            "plateau_span_ms": plateau_span_ms,
            "plateau_sample_count": plateau_sample_count,
            "white_source": white.get("source") if white else None,
            "black_source": black.get("source") if black else None,
            "frenzy_transition": bool(frenzy_transition),
        }
