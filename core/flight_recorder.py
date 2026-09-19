"""
Autonomous Zero-Lag Blackbox Flight Recorder & Telemetry Engine for Violent District Skill Checks.

Buffers screen frames in memory during active skill checks and offloads video rendering,
structured JSON telemetry logging, and multi-panel visual diagnostic strip generation
to a dedicated asynchronous background worker thread.
Zero impact on the 120 FPS vision & trigger loop.
"""

import collections
import datetime
import json
import math
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.vision import is_angle_in_arc


class FlightRecorder:
    """
    Real-time flight recorder and blackbox logger for skill checks.
    - Captures pre-roll frames (frames immediately prior to detection).
    - Records all frames and detections during active check.
    - Captures post-trigger frames to witness the exact freeze/hit result.
    - Dispatches episodes to an async worker thread for disk write & rendering.
    - Outputs:
      1. check_YYYYMMDD_HHMMSS_NNN_[OUTCOME].mp4
      2. check_YYYYMMDD_HHMMSS_NNN.json
      3. check_YYYYMMDD_HHMMSS_NNN_diagnostic.png
      4. manifest.jsonl
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        max_queue_size: int = 30,
        pre_roll_frames: int = 15,
        post_trigger_frames: int = 40,
        save_video: bool = True,
        save_diagnostic_strip: bool = True,
        record_all: bool = True,
    ):
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent.parent / "replays"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "manifest.jsonl"
        self.summary_path = self.output_dir / "session_summary.json"

        self.save_video = save_video
        self.save_diagnostic_strip = save_diagnostic_strip
        self.record_all = record_all
        self.pre_roll_count = pre_roll_frames
        self.post_trigger_count = post_trigger_frames

        # Circular buffer for pre-roll
        self._pre_roll: collections.deque = collections.deque(maxlen=pre_roll_frames)

        # Active episode tracking
        self._active = False
        self._episode: Optional[Dict[str, Any]] = None
        self._frames_after_trigger = 0
        self._check_counter = 0

        # Background processing queue and worker
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="FlightRecorderWorker")
        self._worker_thread.start()

        # Running stats
        self.total_recorded = 0
        self.great_count = 0
        self.good_count = 0
        self.miss_count = 0
        self.max_speed_seen = 350.0

    def start_check(
        self,
        now: float,
        chain_count: int = 1,
        latency_ms: float = 100.0,
        target_mode: str = "GREAT",
        target_ratio: float = 0.50,
        locked_w: Optional[Dict[str, float]] = None,
        locked_b: Optional[Dict[str, float]] = None,
    ):
        """Called when a new skill check is detected."""
        self._check_counter += 1
        now_dt = datetime.datetime.now()
        timestamp_str = now_dt.strftime("%Y%m%d_%H%M%S")
        check_id = f"check_{timestamp_str}_{self._check_counter:04d}"

        # Initialize episode with pre-roll frames (filter to last 150ms on chained checks to avoid stale gaps)
        if chain_count > 1:
            initial_frames = [f for f in self._pre_roll if (now - f[0]) <= 0.150]
        else:
            initial_frames = list(self._pre_roll)
        self._episode = {
            "check_id": check_id,
            "timestamp_iso": now_dt.isoformat(),
            "start_monotonic": now,
            "chain_count": chain_count,
            "configured_latency_ms": latency_ms,
            "target_mode": target_mode,
            "target_ratio": target_ratio,
            "locked_w": locked_w,
            "locked_b": locked_b,
            "frames": initial_frames,  # list of tuples: (t, frame_bgr, det, pred_info, is_pre_roll)
            "trigger_event": None,
            "outcome_info": None,
            "is_frenzy": chain_count > 1,
        }
        self._active = True
        self._frames_after_trigger = 0

    def on_frame(
        self,
        now: float,
        frame_bgr: np.ndarray,
        det: Optional[Dict[str, Any]],
        pred_info: Optional[Dict[str, Any]] = None,
        timing_diag: Optional[Dict[str, float]] = None,
    ):
        """
        Called on every frame from the main CV loop.
        Fast, lightweight, zero disk I/O.
        """
        # Maintain pre-roll
        frame_clone = frame_bgr.copy()
        if not self._active:
            self._pre_roll.append((now, frame_clone, det, pred_info, True, timing_diag))
            return

        # Check is active: append to active episode
        self._episode["frames"].append((now, frame_clone, det, pred_info, False, timing_diag))

        if self._episode.get("trigger_event") is not None:
            self._frames_after_trigger += 1

    def on_trigger(
        self,
        now: float,
        reason: str,
        target_angle: float,
        est_angle: float,
        last_speed: float,
        latency_ms: float,
        selected_speed_tier: Optional[int] = None,
        used_profile_delay: Optional[float] = None,
        measured_speed_at_lock: Optional[float] = None,
        planned_press_time: Optional[float] = None,
        scheduler_error_ms: Optional[float] = None,
        last_frame_age_ms: Optional[float] = None,
        timing_diag: Optional[Dict[str, float]] = None,
        trigger_mode: Optional[str] = None,
        provisional_speed: Optional[float] = None,
        confidence: Optional[str] = None,
        remaining_time_ms: Optional[float] = None,
        phase_telemetry: Optional[Dict[str, Any]] = None,
        shadow_profile_delay: Optional[float] = None,
    ):
        """Called immediately when Space is dispatched."""
        if trigger_mode is None:
            trigger_mode = "FALLBACK_NO_LOCK" if (measured_speed_at_lock is None or "FALLBACK" in reason) else (
                "SCHEDULED" if ("СПИН" in reason or "SCHEDULED" in reason) else (
                    "BACKUP" if "BACKUP" in reason else "IMMEDIATE"
                )
            )

        if self._active and self._episode is not None:
            self._episode["trigger_event"] = {
                "trigger_time": now,
                "reason": reason,
                "trigger_mode": trigger_mode,
                "target_angle": target_angle,
                "est_angle": est_angle,
                "speed_deg_s": last_speed,
                "provisional_speed": provisional_speed,
                "confidence": confidence,
                "remaining_time_ms": remaining_time_ms,
                "latency_ms": latency_ms,
                "selected_speed_tier": selected_speed_tier,
                "used_profile_delay": used_profile_delay,
                "shadow_profile_delay": shadow_profile_delay,
                "phase_telemetry": phase_telemetry,
                "measured_speed_at_lock": measured_speed_at_lock,
                "measured_speed_at_fire": last_speed,
                "planned_press_time": planned_press_time,
                "actual_press_time": now,
                "scheduler_error_ms": scheduler_error_ms,
                "last_frame_age_ms_at_fire": last_frame_age_ms,
                "timing_diag": timing_diag,
                "frame_index_at_trigger": len(self._episode["frames"]) - 1,
            }
            self._frames_after_trigger = 0

    def end_check(
        self,
        now: float,
        outcome_info: Optional[Dict[str, Any]] = None,
        locked_w: Optional[Dict[str, float]] = None,
        locked_b: Optional[Dict[str, float]] = None,
    ):
        """Called when skill check completes/despawns."""
        if not self._active or self._episode is None:
            return

        episode = self._episode
        self._active = False
        self._episode = None

        episode["end_monotonic"] = now
        episode["duration_s"] = now - episode["start_monotonic"]
        episode["outcome_info"] = outcome_info or {}
        if locked_w:
            episode["locked_w"] = locked_w
        if locked_b:
            episode["locked_b"] = locked_b

        # Update stats
        outcome = episode["outcome_info"].get("outcome", "UNKNOWN")
        self.total_recorded += 1
        if outcome == "GREAT":
            self.great_count += 1
        elif outcome == "GOOD":
            self.good_count += 1
        elif outcome == "MISS":
            self.miss_count += 1

        # Enqueue for background rendering & writing
        try:
            self._queue.put_nowait(episode)
        except queue.Full:
            pass  # Drop if worker queue is backlogged to protect game loop

    def close(self):
        """Stops recorder cleanly and drains remaining queue."""
        self._running = False
        self._queue.put(None)
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

    # =========================================================================
    # BACKGROUND WORKER
    # =========================================================================

    def _worker_loop(self):
        while self._running:
            try:
                episode = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if episode is None:
                break

            try:
                self._process_episode(episode)
            except Exception as e:
                try:
                    err_file = self.output_dir / "recorder_errors.log"
                    with open(err_file, "a", encoding="utf-8") as ef:
                        ef.write(f"[{datetime.datetime.now().isoformat()}] Error saving episode {episode.get('check_id')}: {e}\n")
                except Exception:
                    pass
            finally:
                self._queue.task_done()

    def _process_episode(self, episode: Dict[str, Any]):
        check_id = episode["check_id"]
        outcome = episode.get("outcome_info", {}).get("outcome", "UNKNOWN")
        base_name = f"{check_id}_{outcome}"

        # 1. Generate & Save Structured JSON Telemetry (Fast & zero-overhead)
        telemetry_json_path = self.output_dir / f"{base_name}.json"
        self._save_telemetry_json(episode, telemetry_json_path)

        # 2. Render and Save Video Clip & Diagnostic Strip
        # Selective recording to prevent disk bloat and CPU encoding stalls:
        # Always save JSON telemetry.
        # Render heavy media (MP4 & PNG) only for:
        # - MISS
        # - UNCONFIRMED
        # - Large error (|error_deg| >= 8.0)
        # - New speed record (speed > max_speed_seen + 15.0)
        # - Periodic sample (1 in 30 checks)
        # - record_all is explicitly enabled
        speed = float(episode.get("trigger_event", {}).get("speed_deg_s", 0.0) or 0.0)
        error_deg = float(episode.get("outcome_info", {}).get("error_deg", 0.0) or 0.0)

        is_new_speed_record = False
        if speed > self.max_speed_seen + 15.0:
            self.max_speed_seen = speed
            is_new_speed_record = True

        is_large_error = abs(error_deg) >= 8.0
        trigger = episode.get("trigger_event") or {}
        outcome_info = episode.get("outcome_info") or {}
        sched_err = trigger.get("scheduler_error_ms")
        is_sched_late = (sched_err is not None and abs(sched_err) > 4.0)
        is_fallback = ("FALLBACK" in trigger.get("reason", ""))
        is_ghost = ("GHOST" in outcome_info.get("miss_category", ""))
        is_confidence_fail = (outcome_info.get("miss_category") in ("UNSTABLE_SPEED", "VISION/DROPPED FRAMES"))
        is_sample = (self.total_recorded > 0 and self.total_recorded % 10 == 0)

        should_render_media = (
            self.record_all
            or outcome in ("MISS", "GOOD", "UNCONFIRMED")
            or is_large_error
            or is_new_speed_record
            or is_sched_late
            or is_fallback
            or is_ghost
            or is_confidence_fail
            or is_sample
        )

        if self.save_video and should_render_media and episode["frames"]:
            video_path = self.output_dir / f"{base_name}.mp4"
            self._save_video_clip(episode, video_path)

        # 3. Render Multi-Panel Visual Diagnostic Strip
        if self.save_diagnostic_strip and should_render_media and episode["frames"]:
            strip_path = self.output_dir / f"{base_name}_diagnostic.png"
            self._save_diagnostic_strip(episode, strip_path)

        # 4. Append to manifest.jsonl
        self._append_manifest(episode, base_name)

        # 5. Update session summary
        self._update_session_summary()

    def _save_telemetry_json(self, episode: Dict[str, Any], path: Path):
        start_t = episode["start_monotonic"]
        trigger = episode.get("trigger_event") or {}
        outcome = episode.get("outcome_info") or {}

        frame_data = []
        for i, f_item in enumerate(episode["frames"]):
            t, _, det, pred_info, is_pre = f_item[:5]
            timing_diag = f_item[5] if len(f_item) > 5 else None
            f_record = {
                "idx": i,
                "time_rel_ms": round((t - start_t) * 1000.0, 2),
                "is_pre_roll": is_pre,
            }
            if timing_diag is not None:
                f_record["timing"] = timing_diag
            if det is not None:
                f_record.update({
                    "needle_angle": round(det.get("needle_angle", 0.0), 2),
                    "needle_strength": round(det.get("needle_strength", 0.0), 2),
                    "confidence": round(det.get("confidence", 0.0), 3),
                    "cx": round(det.get("cx", 0.0), 1),
                    "cy": round(det.get("cy", 0.0), 1),
                })
            if pred_info is not None:
                f_record.update({
                    "target_angle": round(pred_info.get("target_angle", 0.0), 2),
                    "speed_deg_s": round(pred_info.get("speed_deg_s", 0.0), 1),
                    "time_until_press_ms": round(pred_info.get("time_until_press_ms", 0.0), 1),
                })
            frame_data.append(f_record)

        telemetry = {
            "check_id": episode.get("check_id", "unknown"),
            "timestamp": episode.get("timestamp_iso", datetime.datetime.now().isoformat()),
            "duration_ms": round(episode.get("duration_s", 0.0) * 1000.0, 1),
            "chain_count": episode.get("chain_count", 1),
            "configured_latency_ms": episode.get("configured_latency_ms", 100.0),
            "target_mode": episode.get("target_mode", "GREAT"),
            "target_ratio": episode.get("target_ratio", 0.5),
            "locked_zones": {
                "white": episode.get("locked_w"),
                "black": episode.get("locked_b"),
            },
            "trigger": {
                "fired": trigger.get("trigger_time") is not None,
                "reason": trigger.get("reason"),
                "target_angle": trigger.get("target_angle"),
                "est_angle_at_trigger": trigger.get("est_angle"),
                "speed_deg_s": trigger.get("speed_deg_s"),
                "latency_ms": trigger.get("latency_ms"),
                "selected_speed_tier": trigger.get("selected_speed_tier"),
                "used_profile_delay": trigger.get("used_profile_delay"),
                "shadow_profile_delay": trigger.get("shadow_profile_delay"),
                "phase_telemetry": trigger.get("phase_telemetry"),
                "measured_speed_at_lock": trigger.get("measured_speed_at_lock"),
                "measured_speed_at_fire": trigger.get("measured_speed_at_fire"),
                "planned_press_time": trigger.get("planned_press_time"),
                "actual_press_time": trigger.get("actual_press_time"),
                "scheduler_error_ms": trigger.get("scheduler_error_ms"),
                "last_frame_age_ms_at_fire": trigger.get("last_frame_age_ms_at_fire"),
                "timing": trigger.get("timing_diag"),
            },
            "evaluation": {
                "outcome": outcome.get("outcome", "UNKNOWN"),
                "hit_angle": outcome.get("hit_angle"),
                "target_angle": outcome.get("target_angle"),
                "error_deg": outcome.get("error_deg"),
                "error_ms": outcome.get("error_ms"),
                "white_start": outcome.get("white_start"),
                "white_end": outcome.get("white_end"),
                "white_center": outcome.get("white_center"),
                "white_width": outcome.get("white_width"),
                "center_error_deg": outcome.get("center_error_deg"),
                "center_error_ms": outcome.get("center_error_ms"),
                "entry_edge_error_deg": outcome.get("entry_edge_error_deg"),
                "entry_edge_error_ms": outcome.get("entry_edge_error_ms"),
                "plateau_found": outcome.get("plateau_found", False),
                "prev_latency_ms": outcome.get("prev_latency_ms"),
                "new_latency_ms": outcome.get("new_latency_ms"),
            },
            "frame_count": len(frame_data),
            "frames": frame_data,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(telemetry, f, indent=2, ensure_ascii=False)

    def _save_video_clip(self, episode: Dict[str, Any], path: Path):
        frames = episode["frames"]
        if not frames:
            return

        h, w = frames[0][1].shape[:2]
        fps = 60
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))

        locked_w = episode.get("locked_w")
        locked_b = episode.get("locked_b")
        trig_t = (episode.get("trigger_event") or {}).get("trigger_time")

        for f_item in frames:
            t, img, det, pred, is_pre = f_item[:5]
            disp = img.copy()

            cx = int(det["cx"] if det else 160)
            cy = int(det["cy"] if det else 162)
            radius = 66

            # Draw zones
            if locked_b:
                cv2.ellipse(disp, (cx, cy), (radius, radius), 0, locked_b["start"], locked_b["end"], (220, 160, 0), 3, cv2.LINE_AA)
            if locked_w:
                cv2.ellipse(disp, (cx, cy), (radius, radius), 0, locked_w["start"], locked_w["end"], (0, 255, 0), 4, cv2.LINE_AA)

            # Draw needle
            if det and det.get("needle_angle") is not None:
                ang = det["needle_angle"]
                rad = math.radians(ang)
                nx = int(cx + (radius + 8) * math.cos(rad))
                ny = int(cy + (radius + 8) * math.sin(rad))
                cv2.line(disp, (cx, cy), (nx, ny), (0, 0, 255), 2, cv2.LINE_AA)

            # Trigger flash overlay in clip
            if trig_t and 0.0 <= (t - trig_t) <= 0.15:
                cv2.rectangle(disp, (0, 0), (w - 1, h - 1), (0, 255, 0), 3)
                cv2.putText(disp, "TRIGGER", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            writer.write(disp)

        writer.release()

    def _save_diagnostic_strip(self, episode: Dict[str, Any], path: Path):
        """
        Creates a high-resolution 5-tile horizontal diagnostic contact sheet.
        Tiles:
        1. First lock frame
        2. Approaching midway
        3. 50ms before fire
        4. Fired frame (trigger)
        5. Registered / frozen needle
        """
        frames = episode["frames"]
        if not frames:
            return

        trig_event = episode.get("trigger_event") or {}
        trig_t = trig_event.get("trigger_time")
        outcome_info = episode.get("outcome_info") or {}

        # Select 5 key frames
        active_frames = [f for f in frames if not f[4]]  # filter out pre-roll
        if not active_frames:
            active_frames = frames

        tile_indices = []
        n_active = len(active_frames)

        # Tile 1: Initial lock
        tile_indices.append(0)

        # Find trigger frame index
        trig_idx = None
        if trig_t is not None:
            min_diff = float("inf")
            for i, f in enumerate(active_frames):
                diff = abs(f[0] - trig_t)
                if diff < min_diff:
                    min_diff = diff
                    trig_idx = i

        if trig_idx is not None:
            # Tile 2: ~Midway between start and trigger
            mid_idx = trig_idx // 2
            tile_indices.append(mid_idx)

            # Tile 3: ~3-5 frames before trigger
            pre_trig_idx = max(0, trig_idx - 4)
            tile_indices.append(pre_trig_idx)

            # Tile 4: Trigger frame
            tile_indices.append(trig_idx)

            # Tile 5: Post-trigger (settled/frozen frame matching hit_angle)
            hit_angle = outcome_info.get("hit_angle")
            best_post_idx = min(n_active - 1, trig_idx + 16)
            if hit_angle is not None:
                min_ang_diff = float("inf")
                search_start = min(n_active - 1, trig_idx + 6)
                for cand_idx in range(search_start, n_active):
                    c_det = active_frames[cand_idx][2]
                    if c_det and "needle_angle" in c_det:
                        ang_diff = abs((c_det["needle_angle"] - hit_angle + 180.0) % 360.0 - 180.0)
                        if ang_diff < min_ang_diff:
                            min_ang_diff = ang_diff
                            best_post_idx = cand_idx
            tile_indices.append(best_post_idx)
        else:
            # If no trigger event occurred, sample evenly across active frames
            tile_indices = [
                0,
                n_active // 4,
                n_active // 2,
                (3 * n_active) // 4,
                n_active - 1,
            ]

        # Deduplicate and sort
        tile_indices = sorted(list(dict.fromkeys(tile_indices)))
        while len(tile_indices) < 5 and n_active > len(tile_indices):
            missing = [i for i in range(n_active) if i not in tile_indices]
            if missing:
                tile_indices.append(missing[len(missing) // 2])
                tile_indices.sort()

        tile_labels = [
            "1. SPAWN / LOCK",
            "2. APPROACH",
            "3. FIRE -30ms",
            "4. FIRE TRIGGER",
            "5. FROZEN / RESULT",
        ]

        locked_w = episode.get("locked_w")
        locked_b = episode.get("locked_b")
        tile_w, tile_h = 240, 180

        tiles = []
        for i, idx in enumerate(tile_indices[:5]):
            f_item = active_frames[min(idx, len(active_frames) - 1)]
            t, raw_img, det, pred, is_pre = f_item[:5]

            annotated = cv2.resize(raw_img, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
            sx = tile_w / float(raw_img.shape[1])
            sy = tile_h / float(raw_img.shape[0])
            cx = int((det["cx"] if det else 160) * sx)
            cy = int((det["cy"] if det else 162) * sy)
            rad = int(66.5 * ((sx + sy) / 2.0))

            # Draw zones
            if locked_b:
                cv2.ellipse(annotated, (cx, cy), (rad, rad), 0, locked_b["start"], locked_b["end"], (220, 160, 0), 3, cv2.LINE_AA)
            if locked_w:
                cv2.ellipse(annotated, (cx, cy), (rad, rad), 0, locked_w["start"], locked_w["end"], (0, 255, 0), 4, cv2.LINE_AA)

            # Draw target setpoint dot (Yellow)
            if pred and pred.get("target_angle") is not None:
                trad = math.radians(pred["target_angle"])
                tx = int(cx + rad * math.cos(trad))
                ty = int(cy + rad * math.sin(trad))
                cv2.circle(annotated, (tx, ty), 4, (0, 255, 255), -1, cv2.LINE_AA)

            # Draw needle line (Red)
            ang_val = det.get("needle_angle") if det else None
            if ang_val is not None:
                nrad = math.radians(ang_val)
                nx = int(cx + (rad + 6) * math.cos(nrad))
                ny = int(cy + (rad + 6) * math.sin(nrad))
                cv2.line(annotated, (cx, cy), (nx, ny), (0, 0, 255), 2, cv2.LINE_AA)

            # Header box
            label = tile_labels[i] if i < len(tile_labels) else f"Frame {idx}"
            cv2.rectangle(annotated, (0, 0), (tile_w, 22), (20, 20, 20), -1)
            cv2.putText(annotated, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

            # Sub-info footer
            rel_ms = (t - episode["start_monotonic"]) * 1000.0
            info_str = f"T+{rel_ms:.0f}ms | {ang_val:.0f}°" if ang_val is not None else f"T+{rel_ms:.0f}ms"
            cv2.rectangle(annotated, (0, tile_h - 18), (tile_w, tile_h), (20, 20, 20), -1)
            cv2.putText(annotated, info_str, (6, tile_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

            tiles.append(annotated)

        # Assemble horizontal filmstrip
        strip_tiles = np.hstack(tiles)
        total_w = strip_tiles.shape[1]

        # Top global banner
        outcome = outcome_info.get("outcome", "UNKNOWN")
        banner_h = 44
        banner = np.zeros((banner_h, total_w, 3), dtype=np.uint8)
        banner[:] = (30, 30, 30)

        color_map = {
            "GREAT": (0, 255, 0),
            "GOOD": (255, 200, 0),
            "MISS": (0, 0, 255),
            "UNKNOWN": (200, 200, 200),
        }
        out_color = color_map.get(outcome, (200, 200, 200))

        err_deg = outcome_info.get("error_deg", 0.0)
        err_ms = outcome_info.get("error_ms", 0.0)
        sign_err = "+" if err_deg >= 0 else ""
        title_txt = f"{episode['check_id']}  |  OUTCOME: {outcome}  ({sign_err}{err_deg:.1f}° / {sign_err}{err_ms:.1f}ms)"
        cv2.putText(banner, title_txt, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, out_color, 2, cv2.LINE_AA)

        lat_val = float(episode.get("configured_latency_ms", 100.0))
        ratio_val = float(episode.get("target_ratio", 0.5))
        metrics_txt = (
            f"Lat: {lat_val:.1f}ms | Ratio: {int(ratio_val*100)}% | "
            f"Fact: {outcome_info.get('hit_angle', 0.0):.1f}° | Target: {outcome_info.get('target_angle', 0.0):.1f}°"
        )
        cv2.putText(banner, metrics_txt, (total_w - 480, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

        final_strip = np.vstack([banner, strip_tiles])
        cv2.imwrite(str(path), final_strip)

    def _append_manifest(self, episode: Dict[str, Any], base_name: str):
        outcome = episode.get("outcome_info") or {}
        trig = episode.get("trigger_event") or {}
        record = {
            "check_id": episode.get("check_id", "unknown"),
            "timestamp": episode.get("timestamp_iso", datetime.datetime.now().isoformat()),
            "outcome": outcome.get("outcome", "UNKNOWN"),
            "error_deg": outcome.get("error_deg"),
            "error_ms": outcome.get("error_ms"),
            "configured_latency_ms": episode.get("configured_latency_ms", 100.0),
            "hit_angle": outcome.get("hit_angle"),
            "target_angle": outcome.get("target_angle"),
            "speed_deg_s": trig.get("speed_deg_s"),
            "chain_count": episode.get("chain_count", 1),
            "duration_ms": round(episode.get("duration_s", 0.0) * 1000.0, 1),
            "files": {
                "json": f"{base_name}.json",
                "mp4": f"{base_name}.mp4" if self.save_video else None,
                "diagnostic_png": f"{base_name}_diagnostic.png" if self.save_diagnostic_strip else None,
            },
        }

        try:
            with open(self.manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _update_session_summary(self):
        try:
            pct_great = (self.great_count / self.total_recorded * 100.0) if self.total_recorded else 0.0
            pct_good = (self.good_count / self.total_recorded * 100.0) if self.total_recorded else 0.0
            pct_miss = (self.miss_count / self.total_recorded * 100.0) if self.total_recorded else 0.0

            summary = {
                "total_recorded": self.total_recorded,
                "great": self.great_count,
                "good": self.good_count,
                "miss": self.miss_count,
                "great_percentage": round(pct_great, 2),
                "good_percentage": round(pct_good, 2),
                "miss_percentage": round(pct_miss, 2),
                "last_update": datetime.datetime.now().isoformat(),
            }
            with open(self.summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
