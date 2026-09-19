"""
Asynchronous Non-Blocking Shadow Detector Architecture.
Executes shadow vision detector (e.g. HYBRID) in a dedicated background worker
with a strictly bounded queue (maxsize=2) to guarantee ZERO latency overhead
on the production main thread and critical FIRE dispatch path.
"""

import math
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.detectors import get_detector
from core.predictor import SkillCheckPredictor, SPEED_MODE_BASE


class AsyncShadowDetectorWorker:
    """
    Dedicated background worker for shadow vision auditing.
    Guarantees:
    1. Zero critical-path delay: main thread submit_frame() is purely non-blocking.
    2. Zero buffer backlogs: bounded queue (maxsize=2), drops oldest frame on backlog.
    3. Accurate Near-FIRE telemetry: tracks planned_press_time and actual_press_time.
    4. Diagnostic capture: automatically dumps diagnostic frames on >5° and >10° disagreements.
    """

    def __init__(
        self,
        backend_name: str = "hybrid",
        output_dir: Optional[Path] = None,
        latency_ms: float = 127.27,
        target_offset_ratio: float = 0.50,
        speed_mode: str = SPEED_MODE_BASE,
        session_base_speed: float = 278.0,
    ):
        self.backend_name = backend_name
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent.parent / "replays"
        self.diag_dir = self.output_dir / "diagnostics"
        self.diag_dir.mkdir(parents=True, exist_ok=True)

        self.vision = get_detector(backend_name)
        self.predictor = SkillCheckPredictor(
            latency_ms=latency_ms,
            target_offset_ratio=target_offset_ratio,
            speed_mode=speed_mode,
            session_base_speed=session_base_speed,
        )

        # Bounded queue: maxsize=2 ensures zero multi-frame queue accumulation
        self._queue: queue.Queue = queue.Queue(maxsize=2)
        self._running = True
        self._lock = threading.Lock()

        # Telemetry for active check
        self._in_check = False
        self._check_start_t: float = 0.0
        self._planned_press_t: Optional[float] = None
        self._actual_press_t: Optional[float] = None
        self._fire_speed: Optional[float] = None
        self._fire_reason: Optional[str] = None
        self._target_angle: Optional[float] = None

        self._baseline_lock_t: Optional[float] = None
        self._baseline_speed: Optional[float] = None
        self._shadow_lock_t: Optional[float] = None
        self._shadow_speed: Optional[float] = None

        self._frame_records: List[Dict[str, Any]] = []
        self._shadow_timings_ms: List[float] = []
        self._baseline_timings_ms: List[float] = []

        self.frames_submitted = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.total_checks_evaluated = 0

        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name=f"ShadowDetectorWorker_{backend_name.upper()}"
        )
        self._worker_thread.start()

    def submit_frame(
        self,
        now: float,
        frame_gray: Optional[np.ndarray],
        frame_bgr: np.ndarray,
        in_check: bool,
        baseline_det: Optional[Dict[str, Any]],
        dt_frame: float = 1.0 / 120.0,
        expected_speed: float = 278.0,
        baseline_vision_ms: float = 0.0,
        speed_at_lock: Optional[float] = None,
        target_angle: Optional[float] = None,
    ):
        """
        Main thread entrypoint. Non-blocking.
        If queue is full, drops the oldest frame immediately.
        """
        self.frames_submitted += 1
        payload = (
            now,
            frame_gray,
            frame_bgr.copy(),
            in_check,
            baseline_det,
            dt_frame,
            expected_speed,
            baseline_vision_ms,
            speed_at_lock,
            target_angle,
        )

        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            # Queue full: drop oldest frame to maintain latest real-time frame
            try:
                _ = self._queue.get_nowait()
                self.frames_dropped += 1
                self._queue.put_nowait(payload)
            except (queue.Empty, queue.Full):
                self.frames_dropped += 1

    def on_check_start(self, now: float, target_angle: Optional[float] = None):
        """Signals start of a new skill check to the worker."""
        with self._lock:
            self._in_check = True
            self._check_start_t = now
            self._planned_press_t = None
            self._actual_press_t = None
            self._fire_speed = None
            self._fire_reason = None
            self._target_angle = target_angle
            self._baseline_lock_t = None
            self._baseline_speed = None
            self._shadow_lock_t = None
            self._shadow_speed = None
            self._frame_records.clear()
            self._shadow_timings_ms.clear()
            self._baseline_timings_ms.clear()
            self.vision.reset()
            self.predictor.reset()

    def notify_fire(
        self,
        now: float,
        planned_press_time: Optional[float],
        actual_press_time: float,
        last_speed: float,
        reason: str,
        target_angle: Optional[float] = None,
    ):
        """Records precise FIRE event timestamps from production scheduler."""
        with self._lock:
            self._planned_press_t = planned_press_time
            self._actual_press_t = actual_press_time
            self._fire_speed = last_speed
            self._fire_reason = reason
            if target_angle is not None:
                self._target_angle = target_angle

    def conclude_check(self, now: float, outcome_str: str = "UNKNOWN") -> Optional[Dict[str, Any]]:
        """
        Called when check finishes or is aborted.
        Computes robust frame-by-frame live disagreement telemetry.
        """
        with self._lock:
            if not self._frame_records:
                self._in_check = False
                return None

            records = list(self._frame_records)
            shadow_timings = list(self._shadow_timings_ms)
            baseline_timings = list(self._baseline_timings_ms)
            planned_press = self._planned_press_t
            actual_press = self._actual_press_t
            target_angle = self._target_angle
            b_lock_t = self._baseline_lock_t
            b_spd = self._baseline_speed
            s_lock_t = self._shadow_lock_t
            s_spd = self._shadow_speed
            dropped = self.frames_dropped
            self._in_check = False

        self.total_checks_evaluated += 1

        # 1. Compute frame-by-frame signed deltas
        deltas = [r["diff"] for r in records if r["diff"] is not None]
        abs_deltas = [abs(d) for d in deltas]

        if not abs_deltas:
            return None

        med_signed = float(np.median(deltas))
        mad_delta = float(np.median(np.abs(np.array(deltas) - med_signed)))
        p95_abs = float(np.percentile(abs_deltas, 95))
        p99_abs = float(np.percentile(abs_deltas, 99))
        max_abs = float(np.max(abs_deltas))

        cnt_gt2 = sum(1 for a in abs_deltas if a > 2.0)
        cnt_gt5 = sum(1 for a in abs_deltas if a > 5.0)
        cnt_gt10 = sum(1 for a in abs_deltas if a > 10.0)
        n_frames = len(abs_deltas)

        # 2. Critical window Near-FIRE evaluation
        # Windows based on planned_press_time and actual_press_time
        ref_press = planned_press or actual_press or now
        w150_diffs = [r["diff"] for r in records if r["diff"] is not None and (ref_press - 0.150) <= r["t"] <= ref_press]
        w75_diffs = [r["diff"] for r in records if r["diff"] is not None and (ref_press - 0.075) <= r["t"] <= ref_press]
        w30_diffs = [r["diff"] for r in records if r["diff"] is not None and (ref_press - 0.030) <= r["t"] <= ref_press]

        p95_w150 = float(np.percentile(np.abs(w150_diffs), 95)) if w150_diffs else None
        p95_w75 = float(np.percentile(np.abs(w75_diffs), 95)) if w75_diffs else None
        p95_w30 = float(np.percentile(np.abs(w30_diffs), 95)) if w30_diffs else None

        # Last unique frame before planned press
        frames_before_planned = [r for r in records if r["diff"] is not None and (planned_press is None or r["t"] <= planned_press)]
        last_frame_planned_diff = frames_before_planned[-1]["diff"] if frames_before_planned else None

        # Last unique frame before actual dispatch
        frames_before_actual = [r for r in records if r["diff"] is not None and (actual_press is None or r["t"] <= actual_press)]
        last_frame_actual_diff = frames_before_actual[-1]["diff"] if frames_before_actual else None

        # 3. Zone comparison
        zone_c_diffs = [r["zone_center_diff"] for r in records if r.get("zone_center_diff") is not None]
        zone_w_diffs = [r["zone_width_diff"] for r in records if r.get("zone_width_diff") is not None]
        p95_zone_c = float(np.percentile(zone_c_diffs, 95)) if zone_c_diffs else 0.0
        p95_zone_w = float(np.percentile(zone_w_diffs, 95)) if zone_w_diffs else 0.0

        # 4. Latency / timing comparison
        lock_dt_ms = ((s_lock_t - b_lock_t) * 1000.0) if (s_lock_t and b_lock_t) else None
        speed_delta = (s_spd - b_spd) if (s_spd and b_spd) else None

        # 5. Counterfactual trajectory calculation
        # If Hybrid needle angle had been used with the same baseline latency and prior speed
        counterfactual_hit_angle = None
        counterfactual_outcome = None
        if target_angle is not None and last_frame_actual_diff is not None:
            # Shift by the delta observed at actual fire
            counterfactual_hit_angle = (target_angle + last_frame_actual_diff) % 360.0
            cf_err = abs((counterfactual_hit_angle - target_angle + 180.0) % 360.0 - 180.0)
            if cf_err <= 4.75:
                counterfactual_outcome = "OFFLINE COUNTERFACTUAL: GREAT"
            elif cf_err <= 19.0:
                counterfactual_outcome = "OFFLINE COUNTERFACTUAL: GOOD"
            else:
                counterfactual_outcome = "OFFLINE COUNTERFACTUAL: MISS"

        summary = {
            "backend": self.backend_name,
            "outcome": outcome_str,
            "total_frames": n_frames,
            "dropped_frames": dropped,
            "median_signed_delta": round(med_signed, 3),
            "mad_delta": round(mad_delta, 3),
            "p95_absolute_delta": round(p95_abs, 3),
            "p99_absolute_delta": round(p99_abs, 3),
            "max_absolute_delta": round(max_abs, 3),
            "outliers_gt_2deg": cnt_gt2,
            "outliers_gt_5deg": cnt_gt5,
            "outliers_gt_10deg": cnt_gt10,
            "critical_p95_150ms": round(p95_w150, 3) if p95_w150 is not None else None,
            "critical_p95_75ms": round(p95_w75, 3) if p95_w75 is not None else None,
            "critical_p95_30ms": round(p95_w30, 3) if p95_w30 is not None else None,
            "last_frame_planned_delta": round(last_frame_planned_diff, 3) if last_frame_planned_diff is not None else None,
            "last_frame_actual_delta": round(last_frame_actual_diff, 3) if last_frame_actual_diff is not None else None,
            "zone_center_p95": round(p95_zone_c, 3),
            "zone_width_p95": round(p95_zone_w, 3),
            "lock_dt_ms": round(lock_dt_ms, 2) if lock_dt_ms is not None else None,
            "speed_delta": round(speed_delta, 2) if speed_delta is not None else None,
            "baseline_vision_median_ms": round(float(np.median(baseline_timings)), 3) if baseline_timings else None,
            "hybrid_detector_median_ms": round(float(np.median(shadow_timings)), 3) if shadow_timings else None,
            "counterfactual_hit_angle": round(counterfactual_hit_angle, 2) if counterfactual_hit_angle is not None else None,
            "counterfactual_outcome": counterfactual_outcome,
        }

        # Catch detector disagreements > 5° and save diagnostic frame
        if cnt_gt5 > 0:
            self._save_disagreement_diagnostics(records, cnt_gt10 > 0)

        return summary

    def _save_disagreement_diagnostics(self, records: List[Dict[str, Any]], has_gt10: bool):
        """Saves annotated visual frame on significant disagreement (>5° or >10°)."""
        disagreements = [r for r in records if r["diff"] is not None and abs(r["diff"]) >= 5.0]
        if not disagreements:
            return

        top_err = max(disagreements, key=lambda x: abs(x["diff"]))
        frame = top_err.get("frame_bgr")
        if frame is None:
            return

        diag_img = frame.copy()
        diff = top_err["diff"]
        b_ang = top_err["baseline_angle"]
        s_ang = top_err["shadow_angle"]
        cx, cy = 160, 162

        # Draw baseline needle in red
        b_rad = b_ang * (np.pi / 180.0)
        bx = int(cx + 60.0 * np.cos(b_rad))
        by = int(cy + 60.0 * np.sin(b_rad))
        cv2.line(diag_img, (cx, cy), (bx, by), (0, 0, 255), 2)

        # Draw hybrid needle in green
        s_rad = s_ang * (np.pi / 180.0)
        sx = int(cx + 60.0 * np.cos(s_rad))
        sy = int(cy + 60.0 * np.sin(s_rad))
        cv2.line(diag_img, (cx, cy), (sx, sy), (0, 255, 0), 2)

        # Annotate text
        sev = "GT10" if has_gt10 else "GT5"
        text = f"DIFF:{diff:+.1f} deg (Base:{b_ang:.1f} Hyb:{s_ang:.1f})"
        cv2.putText(diag_img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        fname = f"disagreement_{sev}_{int(time.time()*1000)}_{abs(diff):.1f}deg.png"
        cv2.imwrite(str(self.diag_dir / fname), diag_img)

    def _worker_loop(self):
        """Dedicated shadow processing loop running in background thread."""
        while self._running:
            try:
                payload = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if payload is None:
                break

            (
                now,
                frame_gray,
                frame_bgr,
                in_check,
                baseline_det,
                dt_frame,
                expected_speed,
                baseline_vision_ms,
                speed_at_lock,
                target_angle,
            ) = payload

            t0 = time.perf_counter_ns()
            shadow_det = self.vision.detect(
                frame_bgr=frame_bgr,
                frame_gray=frame_gray,
                dt_frame=dt_frame,
                expected_speed=expected_speed,
            )
            t1 = time.perf_counter_ns()
            shadow_ms = (t1 - t0) / 1e6

            with self._lock:
                self.frames_processed += 1
                if baseline_vision_ms > 0:
                    self._baseline_timings_ms.append(baseline_vision_ms)
                self._shadow_timings_ms.append(shadow_ms)

                if speed_at_lock is not None and self._baseline_lock_t is None:
                    self._baseline_lock_t = now
                    self._baseline_speed = speed_at_lock

                if in_check:
                    diff = None
                    zone_c_diff = None
                    zone_w_diff = None
                    b_ang = baseline_det.get("needle_angle") if baseline_det else None
                    s_ang = shadow_det.get("needle_angle") if shadow_det else None

                    if b_ang is not None and s_ang is not None:
                        # signed delta = hybrid_angle - baseline_angle
                        diff = (s_ang - b_ang + 180.0) % 360.0 - 180.0

                    if baseline_det and shadow_det:
                        bw = baseline_det.get("white_zone")
                        sw = shadow_det.get("white_zone")
                        if bw and sw:
                            zone_c_diff = abs((sw["center"] - bw["center"] + 180.0) % 360.0 - 180.0)
                            zone_w_diff = abs(sw["width"] - bw["width"])

                    # Update shadow predictor
                    if shadow_det is not None and s_ang is not None:
                        sh_w, sh_b = self.vision.extract_zones(shadow_det, None)
                        self.predictor.update(
                            t=now,
                            needle_angle=s_ang,
                            needle_strength=shadow_det.get("needle_strength", 20.0),
                            white_zone=sh_w,
                            black_zone=sh_b,
                        )
                        if self._shadow_lock_t is None and self.predictor.has_stable_speed():
                            self._shadow_lock_t = now
                            self._shadow_speed = self.predictor.speed_deg_s

                    self._frame_records.append(
                        {
                            "t": now,
                            "baseline_angle": b_ang,
                            "shadow_angle": s_ang,
                            "diff": diff,
                            "zone_center_diff": zone_c_diff,
                            "zone_width_diff": zone_w_diff,
                            "baseline_det": baseline_det,
                            "shadow_det": shadow_det,
                            "frame_bgr": frame_bgr if (diff is not None and abs(diff) >= 5.0) else None,
                        }
                    )

    def close(self):
        """Clean shutdown of shadow worker."""
        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
