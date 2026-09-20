from __future__ import annotations

import datetime as dt
import json
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


class FlightRecorder:
    """Async JSON-first replay recorder. No disk I/O occurs in the fire callback."""
    def __init__(self, output_dir: Path, max_queue_size: int = 64, pre_roll_frames: int = 10,
                 save_video: bool = False, save_diagnostic_strip: bool = True, record_all: bool = False,
                 session_meta: Optional[Dict[str, Any]] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_video = bool(save_video)
        self.save_diagnostic_strip = bool(save_diagnostic_strip)
        self.record_all = bool(record_all)
        self.session_meta = dict(session_meta or {})
        self._pre = deque(maxlen=int(pre_roll_frames))
        self._snap_pre = deque(maxlen=6)
        self._episode: Optional[Dict[str, Any]] = None
        self._counter = 0
        self._q: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._stop = object()
        self.dropped = 0
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="FlightRecorder")
        self._worker.start()

    def start_check(self, now: float, chain_count: int = 1, latency_ms: float = 0.0,
                    lead_uncertainty_ms: float = 0.0,
                    target_mode: str = "GREAT", target_ratio: float = 0.5,
                    locked_w=None, locked_b=None, **_: Any) -> None:
        self._counter += 1
        self._snap_pre.clear()
        self._episode = {
            "check_id": f"check_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._counter:04d}",
            "timestamp_iso": dt.datetime.now().isoformat(),
            "session": dict(self.session_meta),
            "start_monotonic": float(now), "chain_count": int(chain_count),
            "configured_latency_ms": float(latency_ms),
            "configured_lead_uncertainty_ms": float(lead_uncertainty_ms),
            "target_mode": target_mode,
            "target_ratio": float(target_ratio), "locked_w": locked_w, "locked_b": locked_b,
            "frames": list(self._pre), "trigger_event": None, "outcome_info": None,
        }

    def on_frame(self, now: float, frame_bgr: np.ndarray, det, pred=None, timing_diag=None) -> None:
        meta = {
            "t": float(now),
            "needle_angle": det.get("needle_angle") if isinstance(det, dict) else None,
            "needle_strength": det.get("needle_strength") if isinstance(det, dict) else None,
            "detector": det.get("detector_name") if isinstance(det, dict) else None,
            "ring_present": det.get("ring_present") if isinstance(det, dict) else None,
            "white_zone": dict(det.get("white_zone")) if isinstance(det, dict) and isinstance(det.get("white_zone"), dict) else None,
            "black_zone": dict(det.get("black_zone")) if isinstance(det, dict) and isinstance(det.get("black_zone"), dict) else None,
            "center": list(det.get("center")) if isinstance(det, dict) and det.get("center") is not None else None,
            "status": det.get("status") if isinstance(det, dict) else None,
            "pred": dict(pred) if isinstance(pred, dict) else None,
            "timing": dict(timing_diag or {}),
        }
        if self._episode is None:
            self._pre.append(meta)
            return
        self._episode["frames"].append(meta)
        if frame_bgr is not None:
            if self._episode.get("trigger_event") is None:
                self._snap_pre.append(frame_bgr.copy())
            else:
                snaps = self._episode.setdefault("snapshots", list(self._snap_pre))
                if len(snaps) < 18:
                    snaps.append(frame_bgr.copy())

    def on_trigger(self, now: float, reason: str, target_angle: float, est_angle: float,
                   last_speed: float, latency_ms: float, **kwargs: Any) -> None:
        if self._episode is not None:
            if self._snap_pre and "snapshots" not in self._episode:
                self._episode["snapshots"] = list(self._snap_pre)
            self._episode["trigger_event"] = {
                "trigger_time": float(now), "reason": reason, "target_angle": target_angle,
                "est_angle": est_angle, "speed_deg_s": last_speed, "latency_ms": latency_ms, **kwargs,
            }

    def end_check(self, now: float, outcome_info: Dict[str, Any]) -> None:
        if self._episode is None:
            return
        ep = self._episode
        self._episode = None
        ep["end_monotonic"] = float(now)
        ep["duration_s"] = float(now) - float(ep["start_monotonic"])
        ep["outcome_info"] = dict(outcome_info or {})
        outcome = str(ep["outcome_info"].get("outcome"))
        # JSON telemetry is cheap and must be complete; dropping GREAT creates
        # survivorship bias and makes replay statistics misleading.  Keep
        # diagnostic images selective unless record_all was explicitly asked.
        if not self.record_all and outcome == "GREAT":
            ep.pop("snapshots", None)
        try:
            self._q.put_nowait(ep)
        except queue.Full:
            self.dropped += 1
            stub = self.output_dir / f"{ep['check_id']}_DROPPED.json"
            stub.write_text(json.dumps({"check_id": ep["check_id"], "reason": "RECORDER_QUEUE_FULL"}, indent=2), encoding="utf-8")

    def _jsonable(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {str(k): self._jsonable(v) for k, v in obj.items() if k != "snapshots"}
        if isinstance(obj, (list, tuple)):
            return [self._jsonable(v) for v in obj]
        return obj

    def _write(self, ep: Dict[str, Any]) -> None:
        base = self.output_dir / ep["check_id"]
        payload = self._jsonable(ep)
        (base.with_suffix(".json")).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        snaps = ep.get("snapshots") or []
        if self.save_diagnostic_strip and snaps:
            resized = [cv2.resize(x, (320, 240)) for x in snaps]
            cv2.imwrite(str(base.with_name(base.name + "_diagnostic.png")), np.hstack(resized))

    def _worker_loop(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is self._stop:
                    return
                self._write(item)
            finally:
                self._q.task_done()

    def close(self) -> None:
        if self._episode is not None:
            self.end_check(time.monotonic(), {"outcome": "ABORTED_SHUTDOWN"})
        self._q.join()
        self._q.put(self._stop)
        self._q.join()
        self._worker.join(timeout=2.0)
