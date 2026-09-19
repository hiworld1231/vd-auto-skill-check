"""Clean session telemetry for the CLEAN V5 runtime.

This module is descriptive only.  It never changes predictor, lead, capture, or
input behaviour.
"""
from __future__ import annotations

import collections
import math
from typing import Any, Dict, List, Optional

import numpy as np


class CleanSessionTelemetry:
    def __init__(self) -> None:
        self.outcomes = collections.Counter()
        self.trigger_counts = collections.Counter()
        self.prediction_lateness_ms: List[float] = []
        self.scheduler_jitter_ms: List[float] = []
        self.grab_wait_ms: List[float] = []
        self.frame_age_ms: List[float] = []
        self.decoded_intervals_ms: List[float] = []
        self.unique_intervals_ms: List[float] = []
        self.vision_times_ms: List[float] = []
        self.predict_times_ms: List[float] = []
        self.actual_used_leads_ms: List[float] = []
        self.total_decoded_frames = 0
        self.unique_frames_count = 0
        self.last_decode_t: Optional[float] = None
        self.last_unique_t: Optional[float] = None
        self.prev_frame_thumb = None
        self.no_fire_count = 0

    @staticmethod
    def _finite(v: Any) -> bool:
        try:
            return v is not None and math.isfinite(float(v))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _pct(vals: List[float], p: float) -> Optional[float]:
        if not vals:
            return None
        return float(np.percentile(np.asarray(vals, dtype=float), p))

    @staticmethod
    def _fmt(v: Optional[float], spec: str = ".2f") -> str:
        return "N/A" if v is None or not math.isfinite(float(v)) else format(float(v), spec)

    def record_frame(
        self,
        *,
        t_grab_done: float,
        decode_ready_ts: Optional[float],
        grab_wait_ms: float,
        python_handoff_age_ms: Optional[float],
        frame_bgr,
    ) -> None:
        self.total_decoded_frames += 1
        if self._finite(grab_wait_ms):
            self.grab_wait_ms.append(float(grab_wait_ms))
        if self._finite(python_handoff_age_ms):
            self.frame_age_ms.append(float(python_handoff_age_ms))

        decode_t = float(decode_ready_ts) if self._finite(decode_ready_ts) else float(t_grab_done)
        if self.last_decode_t is not None and decode_t > self.last_decode_t:
            self.decoded_intervals_ms.append((decode_t - self.last_decode_t) * 1000.0)
        self.last_decode_t = decode_t

        # A very small thumbnail is enough to distinguish a new visual frame for
        # diagnostics.  This never affects the runtime's transport frame-id gate.
        try:
            thumb = frame_bgr[::24, ::32, :].copy()
            is_unique = self.prev_frame_thumb is None or not np.array_equal(thumb, self.prev_frame_thumb)
            self.prev_frame_thumb = thumb
        except Exception:
            is_unique = True
        if is_unique:
            self.unique_frames_count += 1
            if self.last_unique_t is not None and decode_t > self.last_unique_t:
                self.unique_intervals_ms.append((decode_t - self.last_unique_t) * 1000.0)
            self.last_unique_t = decode_t

    def record_diagnostics(self, *, vision_ms: Optional[float], predict_ms: Optional[float]) -> None:
        if self._finite(vision_ms):
            self.vision_times_ms.append(float(vision_ms))
        if self._finite(predict_ms):
            self.predict_times_ms.append(float(predict_ms))

    def record_fire(
        self,
        *,
        reason: str,
        prediction_lateness_ms: Optional[float],
        scheduler_jitter_ms: Optional[float],
        trigger_mode: Optional[str] = None,
        actual_delay_ms: Optional[float] = None,
        **_: Any,
    ) -> None:
        mode = str(trigger_mode or reason or "UNKNOWN").upper()
        if "EARLY_TRACK" in mode:
            mode = "SCHEDULED_EARLY_TRACK"
        elif "SCHEDULED" in mode or "СПИН" in mode:
            mode = "SCHEDULED"
        elif "IMMEDIATE" in mode or "DIRECT" in mode:
            mode = "IMMEDIATE"
        elif "BACKUP" in mode:
            mode = "BACKUP"
        self.trigger_counts[mode] += 1
        if self._finite(prediction_lateness_ms):
            self.prediction_lateness_ms.append(float(prediction_lateness_ms))
        if self._finite(scheduler_jitter_ms):
            self.scheduler_jitter_ms.append(float(scheduler_jitter_ms))
        if self._finite(actual_delay_ms):
            self.actual_used_leads_ms.append(float(actual_delay_ms))

    def record_outcome(self, *, outcome: str, **_: Any) -> None:
        self.outcomes[str(outcome or "UNCONFIRMED").upper()] += 1

    def record_no_fire(self) -> None:
        self.no_fire_count += 1
        self.outcomes["NO_FIRE"] += 1

    def generate_report(self, *, grabber=None, lead_controller=None, recorder=None, no_fire_reasons=None) -> str:
        lines = [
            "=" * 78,
            "VIOLENCE DISTRICT — CLEAN V5 SESSION REPORT",
            "=" * 78,
            "OUTCOMES: " + ", ".join(f"{k}={v}" for k, v in sorted(self.outcomes.items())),
            "FIRES: " + (", ".join(f"{k}={v}" for k, v in sorted(self.trigger_counts.items())) or "none"),
        ]
        if self.actual_used_leads_ms:
            lines.append(
                "USED LEAD: min={}ms med={}ms max={}ms".format(
                    self._fmt(min(self.actual_used_leads_ms), ".1f"),
                    self._fmt(self._pct(self.actual_used_leads_ms, 50), ".1f"),
                    self._fmt(max(self.actual_used_leads_ms), ".1f"),
                )
            )
        lines.append(
            "TIMING: prediction lateness p50={}ms p95={}ms | scheduler jitter p50={}ms p95={}ms".format(
                self._fmt(self._pct(self.prediction_lateness_ms, 50)),
                self._fmt(self._pct(self.prediction_lateness_ms, 95)),
                self._fmt(self._pct(self.scheduler_jitter_ms, 50)),
                self._fmt(self._pct(self.scheduler_jitter_ms, 95)),
            )
        )
        lines.append(
            "PIPELINE: grab p50={}ms p95={}ms | frame-age p50={}ms p95={}ms | vision p95={}ms | predict p95={}ms".format(
                self._fmt(self._pct(self.grab_wait_ms, 50)),
                self._fmt(self._pct(self.grab_wait_ms, 95)),
                self._fmt(self._pct(self.frame_age_ms, 50)),
                self._fmt(self._pct(self.frame_age_ms, 95)),
                self._fmt(self._pct(self.vision_times_ms, 95)),
                self._fmt(self._pct(self.predict_times_ms, 95)),
            )
        )
        lines.append(
            f"FRAMES: decoded={self.total_decoded_frames} visual_unique={self.unique_frames_count} "
            f"unique_interval_p95={self._fmt(self._pct(self.unique_intervals_ms,95))}ms"
        )
        if no_fire_reasons:
            lines.append("NO-FIRE: " + ", ".join(f"{k}={v}" for k, v in sorted(dict(no_fire_reasons).items())))
        if lead_controller is not None:
            lines.append("LEAD: " + str(lead_controller.telemetry()))
        if grabber is not None:
            try:
                lines.append("CAPTURE: " + str({
                    "mode": getattr(grabber, "framerate_mode", None),
                    "discarded_frames": getattr(grabber, "discarded_frames_count", None),
                    "transport_duplicate_waits": getattr(grabber, "transport_duplicate_waits", None),
                    "worker_failed": getattr(grabber, "_worker_failed", None),
                    "last_error": getattr(grabber, "_last_error", None),
                }))
            except Exception:
                pass
        if recorder is not None:
            try:
                diag = recorder.get_diagnostics() if hasattr(recorder, "get_diagnostics") else {}
                lines.append("RECORDER: " + str(diag))
            except Exception:
                pass
        lines.append("=" * 78)
        return "\n".join(lines)
