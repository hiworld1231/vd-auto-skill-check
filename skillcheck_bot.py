#!/usr/bin/env python3
"""
Violent District Autonomous Skill Check Bot (DBD Roblox).
Ultra-low-latency computer vision & kinematic auto-presser.
Designed for 60-75 FPS gameplay on Linux Wayland / KMS.
"""

import argparse
import json
import math
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT.parent / ".venv"
VENV_PY = VENV_DIR / "bin" / "python"
if VENV_PY.exists() and sys.prefix != str(VENV_DIR.resolve()):
    os.execv(str(VENV_PY), [str(VENV_PY)] + sys.argv)

import cv2
import numpy as np

from core.capture import ScreenGrabber, CAPTURE_REGION
from core.vision import VisionEngine, is_angle_in_arc
from core.predictor import (
    SkillCheckPredictor,
    DEFAULT_SPEED_DEG_S,
    DEFAULT_LATENCY_MS,
    STATE_LOCKED,
    STATE_COMMITTED,
    SPEED_MODE_BASE,
    SPEED_MODE_VARIABLE,
    SPEED_MODE_BASE_LATENCY_TEST,
    SPEED_MODE_GEN_RUSH,
)
from core.continuous_predictor import ContinuousAngularPredictor
from core.lead_level_controller import LeadLevelController
from core.response_phase_tracker import ResponsePhaseTracker
from core.trigger import HardwareTrigger, PreciseTriggerScheduler, init_keyboard
from core.learner import AdaptiveLatencyLearner
from core.telemetry_tracker import SessionTelemetryTracker
from core.tui import SkillCheckTUI
from core.flight_recorder import FlightRecorder
from core.mouse_tracker import MouseTracker
from core.shadow_detector import AsyncShadowDetectorWorker

CONFIG_PATH = ROOT / "config.json"
LOG_PATH = ROOT / "bot.log"
HUD_WINDOW_NAME = "SkillCheck HUD"
RUNNING = True


def log_flight_record(msg: str):
    """Writes persistent flight recorder logs to bot.log for post-session analysis."""
    try:
        t_str = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{t_str}] {msg}\n")
    except Exception:
        pass


def load_config() -> dict:
    cfg = {
        "fps": 120,
        "latency_ms": 98.0,
        "target": "GREAT",
        "target_offset_ratio": 0.50,
        "region": CAPTURE_REGION,
        "show_hud": False,
        "require_lmb": True,
    }

    if CONFIG_PATH.exists():
        try:
            loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            cfg.update(loaded)
        except Exception:
            pass
    return cfg


def save_config(cfg: dict):
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception:
        pass


def signal_handler(sig, frame):
    global RUNNING
    RUNNING = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def render_hud(
    frame_bgr,
    det=None,
    pred=None,
    current_fps=120.0,
    last_trigger_t=0.0,
    now=0.0,
    dry_run=False,
    target_mode="GREAT",
    locked_w=None,
    locked_b=None,
    latency_ms=70.0,
    target_ratio=0.44,
    chain_count=1,
    out_w=480,
    out_h=360,
    triggered_t=None,
):
    if triggered_t is not None:
        last_trigger_t = triggered_t
    scale_x = out_w / float(frame_bgr.shape[1])
    scale_y = out_h / float(frame_bgr.shape[0])
    scale = (scale_x + scale_y) / 2.0

    hud = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    cx = int((det["cx"] if det else 160.0) * scale_x)
    cy = int((det["cy"] if det else 162.5) * scale_y)
    radius = int(66.5 * scale)

    if det is not None:
        cv2.circle(hud, (cx, cy), radius, (60, 60, 60), 2, cv2.LINE_AA)

        w_zone = locked_w
        b_zone = locked_b

        # Good zone arc
        if b_zone:
            cv2.ellipse(hud, (cx, cy), (radius, radius), 0, b_zone["start"], b_zone["end"], (220, 160, 0), 5, cv2.LINE_AA)

        # Great zone arc in bright Green
        if w_zone:
            cv2.ellipse(hud, (cx, cy), (radius, radius), 0, w_zone["start"], w_zone["end"], (0, 255, 0), 7, cv2.LINE_AA)

        # Needle pointer in Red
        needle_angle = det.get("needle_angle", 0.0)
        rad = math.radians(needle_angle)
        nx = int(cx + (radius + 10) * math.cos(rad))
        ny = int(cy + (radius + 10) * math.sin(rad))
        cv2.line(hud, (cx, cy), (nx, ny), (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(hud, (nx, ny), 4, (0, 0, 255), -1, cv2.LINE_AA)

        if pred is not None:
            t_rad = math.radians(pred["target_angle"])
            tx = int(cx + radius * math.cos(t_rad))
            ty = int(cy + radius * math.sin(t_rad))
            cv2.circle(hud, (tx, ty), 5, (0, 255, 255), -1, cv2.LINE_AA)

            rem_ms = pred.get("time_until_press_ms", 0.0)
            bar_color = (0, 220, 255) if rem_ms > 40 else (0, 255, 0)
            badge = f"LOCK {target_mode}: {max(0.0, rem_ms):.0f}ms ({needle_angle:.0f}° -> {pred['target_angle']:.0f}°)"
            cv2.rectangle(hud, (10, out_h - 36), (out_w - 10, out_h - 10), (20, 20, 20), -1)
            cv2.rectangle(hud, (10, out_h - 36), (out_w - 10, out_h - 10), bar_color, 1)
            cv2.putText(hud, badge, (18, out_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, bar_color, 1, cv2.LINE_AA)
    else:
        cv2.circle(hud, (cx, cy), radius, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(hud, "WAITING FOR SKILL CHECK", (int(out_w * 0.16), cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(hud, "[+/-] Задержка   [[/]] Смещение", (int(out_w * 0.12), out_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 110, 110), 1, cv2.LINE_AA)

    # Top banner
    cv2.rectangle(hud, (0, 0), (out_w, 26), (25, 25, 25), -1)
    cv2.line(hud, (0, 26), (out_w, 26), (50, 50, 50), 1)
    title = f"VD [FRENZY x{chain_count}]" if chain_count > 1 else ("VD [DRY-RUN]" if dry_run else "VD [AUTO-BOT]")
    cv2.putText(hud, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)
    cv2.putText(hud, f"{current_fps:.0f} FPS | {latency_ms:.0f}ms | {int(target_ratio*100)}%", (out_w - 175, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

    # Trigger flash (0.35s for crisp visual pulse on rapid chained hits)
    if 0.0 < (now - last_trigger_t) < 0.35:
        cv2.rectangle(hud, (0, 0), (out_w - 1, out_h - 1), (0, 255, 0), 5)
        flash_text = f"HIT GREAT! x{chain_count}" if chain_count > 1 else "HIT GREAT!"
        cv2.putText(hud, flash_text, (int(out_w * (0.20 if chain_count > 1 else 0.28)), int(out_h * 0.52)), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2, cv2.LINE_AA)

    return hud


def run_bot(
    dry_run: bool = False,
    latency_ms: float = None,
    target_mode: str = None,
    target_offset_ratio: float = None,
    region: dict = None,
    show_hud: bool = False,
    require_lmb: bool = False,
    fps: int = None,
    record_all: bool = False,
    freeze: bool = False,
    speed_mode: str = None,
    session_base_speed: float = None,
    detector_name: str = None,
    shadow_detector_name: str = None,
    base_latency_test: bool = False,
    gen_rush: bool = False,
):
    global RUNNING
    cfg = load_config()

    is_gen_rush = bool(
        gen_rush
        or (speed_mode in (SPEED_MODE_GEN_RUSH, "SPEED_PERK"))
        or (cfg.get("speed_mode") in (SPEED_MODE_GEN_RUSH, "SPEED_PERK"))
        or cfg.get("gen_rush", False)
        or cfg.get("speed_perk", False)
    )

    is_base_latency_test = bool(
        base_latency_test
        or (speed_mode == SPEED_MODE_BASE_LATENCY_TEST)
        or (cfg.get("speed_mode") == SPEED_MODE_BASE_LATENCY_TEST)
        or cfg.get("base_latency_test", False)
    )

    eff_fps = fps or cfg.get("fps", 120)
    eff_latency = latency_ms if latency_ms is not None else cfg.get("latency_ms", 98.0)
    # Keep one explicit delivery lead in GEN_RUSH.  Continuous angular tracking
    # handles speed directly, so there is no BASE/PERK latency split and no
    # speed-profile latency in the FIRE path.  Optional config key lets a later
    # controlled LIVE calibration change only this one scalar.
    configured_base_latency_ms = float(eff_latency)
    # V4 cold start is deliberately late-biased for reliability.  Historical VD
    # sessions showed that an over-large lead creates EARLY misses, while a smaller
    # lead lands into the much wider trailing GOOD zone.  After 3 clean landings the
    # session controller snaps to the measured median ideal lead.
    genrush_delivery_lead_ms = float(cfg.get("genrush_seed_lead_ms", 60.0))
    chain_latency_offset_ms = cfg.get("chain_latency_offset_ms", 0.0)
    eff_target = target_mode or cfg.get("target", "GREAT")
    eff_region = region or cfg.get("region", CAPTURE_REGION)
    eff_ratio = 0.50  # Production GREAT target ratio: strictly 0.50 (physical center)
    if is_gen_rush:
        eff_speed_mode = SPEED_MODE_GEN_RUSH
        freeze = True
    elif is_base_latency_test:
        eff_speed_mode = SPEED_MODE_BASE_LATENCY_TEST
        freeze = True
    else:
        eff_speed_mode = speed_mode or cfg.get("speed_mode", SPEED_MODE_BASE)

    eff_base_speed = session_base_speed if session_base_speed is not None else cfg.get("session_base_speed", 278.0)
    eff_detector = detector_name or cfg.get("detector", "hybrid")
    eff_shadow_detector = shadow_detector_name or cfg.get("shadow_detector", None)

    phase_tracker = ResponsePhaseTracker(base_latency_ms=configured_base_latency_ms)
    lead_controller = LeadLevelController(seed_lead_ms=genrush_delivery_lead_ms) if is_gen_rush else None

    def get_effective_latency(chain: int) -> float:
        if is_base_latency_test:
            return configured_base_latency_ms
        if is_gen_rush:
            return lead_controller.get_lead_ms(
                chain_count=chain,
                chain_offset_ms=chain_latency_offset_ms,
            )
        return phase_tracker.get_effective_latency(chain_count=chain, chain_offset_ms=chain_latency_offset_ms)

    trigger_hw = HardwareTrigger(dry_run=dry_run)
    vision = VisionEngine(backend_name=eff_detector)
    shadow_worker = (
        AsyncShadowDetectorWorker(
            backend_name=eff_shadow_detector,
            output_dir=ROOT / "replays",
            latency_ms=configured_base_latency_ms,
            target_offset_ratio=eff_ratio,
            speed_mode=eff_speed_mode,
            session_base_speed=eff_base_speed,
        )
        if eff_shadow_detector
        else None
    )

    def summarize_shadow(outcome_str: str = "UNKNOWN") -> Optional[Dict[str, Any]]:
        if shadow_worker is None:
            return None
        summary = shadow_worker.conclude_check(now=time.monotonic(), outcome_str=outcome_str)
        if summary is None:
            return None
        med_diff = summary["median_signed_delta"]
        mad_diff = summary["mad_delta"]
        p95_diff = summary["p95_absolute_delta"]
        p95_150 = summary["critical_p95_150ms"]
        p95_150_str = f"{p95_150:.2f}°" if p95_150 is not None else "N/A"
        lock_dt = summary["lock_dt_ms"]
        lock_str = f"{lock_dt:+.1f}ms" if lock_dt is not None else "N/A"
        spd_diff = summary["speed_delta"]
        spd_str = f"{spd_diff:+.1f}°/s" if spd_diff is not None else "N/A"
        dropped = summary["dropped_frames"]

        tui.log(
            f"🔍 [SHADOW {eff_shadow_detector.upper()}] bias={med_diff:+.2f}° (MAD {mad_diff:.2f}°), "
            f"crit_p95={p95_150_str}, p95={p95_diff:.2f}°, lock_dt={lock_str}, "
            f"spd_dt={spd_str}, drop={dropped} ({summary['total_frames']} frames)",
            style="magenta",
        )
        log_flight_record(
            f"[SHADOW {eff_shadow_detector.upper()}] bias={med_diff:+.2f}° MAD={mad_diff:.2f}° "
            f"crit_p95={p95_150_str} p95={p95_diff:.2f}° lock_dt={lock_str} spd_dt={spd_str} drop={dropped}"
        )
        return summary

    if is_gen_rush:
        predictor = ContinuousAngularPredictor(
            latency_ms=genrush_delivery_lead_ms,
            target_offset_ratio=eff_ratio,
            session_base_speed=eff_base_speed,
            fit_window=10,
        )
    else:
        predictor = SkillCheckPredictor(
            latency_ms=configured_base_latency_ms,
            target_offset_ratio=eff_ratio,
            speed_mode=eff_speed_mode,
            session_base_speed=eff_base_speed,
        )
    grabber = ScreenGrabber(
        eff_region,
        fps=eff_fps,
        framerate_mode=("vfr" if is_gen_rush else "cfr"),
    )
    mouse_tracker = MouseTracker(enabled=True)
    flight_recorder = FlightRecorder(output_dir=ROOT / "replays", record_all=record_all)

    def handle_tui_action(action: str):
        nonlocal eff_latency, eff_ratio
        if (is_base_latency_test or is_gen_rush) and action in ("inc_latency", "dec_latency", "inc_offset", "dec_offset"):
            mode_name = "BASE_LATENCY_TEST" if is_base_latency_test else "GEN_RUSH"
            tui.log(f"🔒 {mode_name}: ручное изменение задержки/смещения отключено, чтобы base/phase не рассинхронизировались", style="bold red")
            return
        if action == "inc_latency":
            eff_latency += 4.0
            predictor.base_latency_ms = eff_latency
            predictor.latency_s = eff_latency / 1000.0
            learner.latency_ms = eff_latency
            cfg["latency_ms"] = round(eff_latency, 1)
            if not freeze:
                save_config(cfg)
            tui.latency_ms = eff_latency
            tui.requested_base_latency = eff_latency
            tui.log(f"⚡ Упреждение увеличено (+4мс): {eff_latency:.1f} мс (нажатие РАНЬШЕ)", style="cyan")
            log_flight_record(f"[HOTKEY] Inc latency -> {eff_latency:.1f}ms (earlier press)")
        elif action == "dec_latency":
            eff_latency = max(10.0, eff_latency - 4.0)
            predictor.base_latency_ms = eff_latency
            predictor.latency_s = eff_latency / 1000.0
            learner.latency_ms = eff_latency
            cfg["latency_ms"] = round(eff_latency, 1)
            if not freeze:
                save_config(cfg)
            tui.latency_ms = eff_latency
            tui.requested_base_latency = eff_latency
            tui.log(f"⚡ Упреждение уменьшено (-4мс): {eff_latency:.1f} мс (нажатие ПОЗЖЕ)", style="cyan")
            log_flight_record(f"[HOTKEY] Dec latency -> {eff_latency:.1f}ms (later press)")
        elif action == "inc_offset":
            eff_ratio = min(0.90, eff_ratio + 0.05)
            predictor.target_offset_ratio = eff_ratio
            cfg["target_offset_ratio"] = round(eff_ratio, 2)
            if not freeze:
                save_config(cfg)
            tui.target_ratio = eff_ratio
            tui.log(f"⚡ Смещение увеличено: {eff_ratio*100:.0f}% зоны Great", style="cyan")
        elif action == "dec_offset":
            eff_ratio = max(0.10, eff_ratio - 0.05)
            predictor.target_offset_ratio = eff_ratio
            cfg["target_offset_ratio"] = round(eff_ratio, 2)
            if not freeze:
                save_config(cfg)
            tui.target_ratio = eff_ratio
            tui.log(f"⚡ Смещение уменьшено: {eff_ratio*100:.0f}% зоны Great", style="cyan")
        elif action == "quit":
            global RUNNING
            RUNNING = False

    tui = SkillCheckTUI(
        mode="DRY-RUN" if dry_run else "AUTO-BOT",
        target_mode=eff_target,
        initial_latency=configured_base_latency_ms,
        initial_ratio=eff_ratio,
        backend=trigger_hw.backend,
        fps=eff_fps,
        on_action=handle_tui_action,
    )
    tui.start()
    tui.log(f"Область захвата: {eff_region['width']}x{eff_region['height']}+{eff_region['left']}+{eff_region['top']}", style="dim")
    tui.log(f"Vision Детектор: {eff_detector.upper()}", style="dim")
    if shadow_worker is not None:
        tui.log(f"Shadow Детектор: {eff_shadow_detector.upper()} (параллельный аудит без влияния на выстрел)", style="magenta")

    if show_hud:
        cv2.namedWindow(HUD_WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(HUD_WINDOW_NAME, 480, 360)
        cv2.moveWindow(HUD_WINDOW_NAME, 1420, 20)

    fps_count = 0
    fps_timer = time.monotonic()
    current_fps = 0.0
    last_cfg_check = time.monotonic()
    last_cfg_mtime = CONFIG_PATH.stat().st_mtime if CONFIG_PATH.exists() else 0.0

    in_check = False
    check_start_t = 0.0
    last_det_t = 0.0
    last_check_end_t = 0.0
    armed = False
    pressed = False
    last_trigger_t = 0.0
    target_angle_locked = 0.0
    locked_w = None
    locked_b = None
    last_frame_id = None
    expected_angle = None
    vision_search_win = 40.0
    last_needle_angle = None
    pred = None
    press_t = None
    chain_count = 0
    is_frenzy_chain_active = False
    last_speed = DEFAULT_SPEED_DEG_S
    last_confirmed_speed = DEFAULT_SPEED_DEG_S
    trigger_needle_angle = None
    locked_speed_tier = None
    locked_tier_latency = None
    speed_at_lock = None
    last_timing_diag = {}
    last_hud_t = 0.0
    det = None
    current_needle = 0.0
    dt_frame = 1.0 / eff_fps
    rem_ms = None
    last_fire_delay = None
    last_fire_reason = "SCHEDULED"
    last_fire_sched_jitter = None
    last_fire_frame_age = None
    last_fire_trigger_mode = None
    last_fire_detector_fallback = False
    last_fire_detector_source = None
    last_fire_mode_switched = False
    last_fire_fit_spread = None
    last_fire_fit_residual = None
    last_fire_fit_sample_count = None
    last_fire_fit_span_ms = None
    last_fire_phase_snapshot = None
    last_fire_compensation_regime = "CONTINUOUS" if is_gen_rush else "BASE"
    locked_compensation_regime = "CONTINUOUS" if is_gen_rush else "BASE"
    locked_shadow_profile_delay = None
    last_is_override = False
    last_experiment_valid = True
    learner = AdaptiveLatencyLearner(
        initial_latency_ms=configured_base_latency_ms,
        min_latency_ms=70.0,
        max_latency_ms=135.0,
        deadband_deg=2.00,
        max_step_ms=3.0,
        learning_rate=0.22,
        max_valid_error_deg=30.0,
        auto_save=not freeze,
        save_to_disk=not freeze,
        verbose=False,
        use_speed_profiles=(not is_base_latency_test and not is_gen_rush),
        freeze=freeze,
    )
    if is_gen_rush:
        tui.log(
            f"🚀 GEN_RUSH RESET V4: capture=VFR | cold_lead={genrush_delivery_lead_ms:.1f}ms -> median after 3 clean | "
            f"speed=continuous robust fit | no tiers | no BASE→VARIABLE | profiles=OFF",
            style="bold yellow",
        )
    elif is_base_latency_test:
        tui.log(f"🧪 РЕЖИМ BASE_LATENCY_TEST: forced BASE_SPEED, профили отключены, test latency={eff_latency:.1f}мс", style="bold green")
    elif freeze:
        tui.log("🔒 РЕЖИМ FREEZE: Запись в config.json и speed_profiles.json ОТКЛЮЧЕНА (активные профили заморожены)", style="bold cyan")

    tracker = SessionTelemetryTracker()

    def on_fire(
        reason="SCHEDULED",
        planned_t=None,
        desired_press_time=None,
        scheduler_dispatch_target=None,
        callback_entry_time=None,
        diag_timestamps=None,
    ):
        nonlocal pressed, armed, last_trigger_t, trigger_needle_angle, last_confirmed_speed, last_fire_delay, last_is_override, last_experiment_valid, last_fire_reason, last_fire_sched_jitter, last_fire_frame_age, last_fire_trigger_mode, last_fire_detector_fallback, last_fire_detector_source, last_fire_mode_switched, last_fire_fit_spread, last_fire_fit_residual, last_fire_fit_sample_count, last_fire_fit_span_ms, last_fire_phase_snapshot, last_fire_compensation_regime
        if not pressed:
            pressed = True
            armed = False
            callback_entry_t = callback_entry_time if callback_entry_time is not None else time.monotonic()
            # CRITICAL: Physical input dispatch is executed IMMEDIATELY at function entry
            # before any logging, string formatting, or TUI updates.
            t_dispatch_start, t_dispatch_done = trigger_hw.trigger()
            last_trigger_t = t_dispatch_start
            vision.notify_pressed()

            if diag_timestamps is not None:
                diag_timestamps["uinput_dispatch_start_mono"] = t_dispatch_start
                diag_timestamps["uinput_dispatch_done_mono"] = t_dispatch_done
                diag_timestamps["uinput_duration_us"] = (t_dispatch_done - t_dispatch_start) * 1e6
                if last_timing_diag is not None:
                    last_timing_diag["scheduler_diag"] = diag_timestamps

            target_desired = desired_press_time if desired_press_time is not None else planned_t
            pred_lateness_ms = ((t_dispatch_start - target_desired) * 1000.0) if target_desired is not None else 0.0
            sched_jitter_ms = ((callback_entry_t - scheduler_dispatch_target) * 1000.0) if scheduler_dispatch_target is not None else None
            frame_age_ms = ((t_dispatch_start - last_det_t) * 1000.0) if last_det_t > 0 else 0.0
            trigger_needle_angle = current_needle if (det and "needle_angle" in det) else target_angle_locked
            if 20.0 <= last_speed <= 1500.0:
                last_confirmed_speed = last_speed
            scheduler.cancel()

            is_fallback = (speed_at_lock is None) or ("FALLBACK" in reason)
            trigger_mode = "FALLBACK_NO_LOCK" if is_fallback else (
                "SCHEDULED" if ("СПИН" in reason or "SCHEDULED" in reason) else (
                    "BACKUP" if "BACKUP" in reason else "IMMEDIATE"
                )
            )
            fire_shadow_telem = predictor.get_shadow_telemetry()
            detector_source = det.get("detector_name") if isinstance(det, dict) else None
            detector_fallback = bool(detector_source and "FALLBACK" in detector_source.upper())
            last_fire_trigger_mode = trigger_mode
            last_fire_detector_source = detector_source
            last_fire_detector_fallback = detector_fallback
            last_fire_mode_switched = bool(fire_shadow_telem.get("mode_switched", False))
            last_fire_fit_spread = fire_shadow_telem.get("live_fit_spread")
            last_fire_fit_residual = fire_shadow_telem.get("fit_residual_mad_deg")
            last_fire_fit_sample_count = fire_shadow_telem.get("fit_sample_count")
            last_fire_fit_span_ms = fire_shadow_telem.get("fit_span_ms")

            prefix = f"💥 HIT #{chain_count}" if chain_count > 1 else "💥 HIT"
            cur_est_angle = target_angle_locked
            if last_det_t > 0 and last_needle_angle is not None and last_speed > 0:
                dt_since_det = last_trigger_t - last_det_t
                cur_est_angle = (last_needle_angle + last_speed * dt_since_det) % 360.0

            requested_test_latency = (
                (locked_tier_latency if locked_tier_latency is not None else get_effective_latency(chain_count))
                if is_gen_rush
                else configured_base_latency_ms
            ) if (is_gen_rush or is_base_latency_test) else eff_latency
            if is_base_latency_test:
                cur_lat = requested_test_latency
                is_override = False
                experiment_valid = (abs(cur_lat - requested_test_latency) < 1e-4)
            else:
                cur_lat = locked_tier_latency if locked_tier_latency is not None else get_effective_latency(chain_count)
                # In GEN_RUSH old speed profiles are shadow-only. A BASE/PERK timing
                # regime is not a "profile override".
                is_override = False if is_gen_rush else (abs(cur_lat - requested_test_latency) > 0.05)
                experiment_valid = True

            last_fire_delay = cur_lat
            last_fire_compensation_regime = locked_compensation_regime if is_gen_rush else "LEGACY"
            last_fire_phase_snapshot = phase_tracker.get_telemetry()
            last_fire_phase_snapshot["actuation_mode"] = "SHADOW_ONLY" if is_gen_rush else "ACTIVE"
            last_fire_phase_snapshot["compensation_regime"] = last_fire_compensation_regime
            last_fire_phase_snapshot["genrush_delivery_lead_ms"] = (lead_controller.current_lead_ms if is_gen_rush else None)
            last_fire_phase_snapshot["chain_offset_ms"] = chain_latency_offset_ms * max(0, chain_count - 1)
            last_fire_phase_snapshot["effective_with_chain_ms"] = cur_lat
            last_is_override = is_override
            last_experiment_valid = experiment_valid
            last_fire_reason = reason
            # Keep scheduler callback jitter separate from prediction lateness.
            last_fire_sched_jitter = sched_jitter_ms
            last_fire_frame_age = frame_age_ms

            tracker.record_fire(
                reason=reason,
                locked_tier=locked_speed_tier,
                speed_at_lock=speed_at_lock,
                speed_at_fire=last_speed,
                prediction_lateness_ms=pred_lateness_ms,
                scheduler_jitter_ms=sched_jitter_ms,
                trigger_mode=trigger_mode,
                requested_latency_ms=requested_test_latency,
                actual_delay_ms=cur_lat,
                is_profile_override=is_override,
                is_experiment_valid=experiment_valid,
            )
            tui.record_fire_telemetry(requested_test_latency, cur_lat, is_override=is_override)

            tier_str = f" [~{locked_speed_tier}°/с]" if locked_speed_tier is not None else ""
            tui.set_status("TRIGGERED (SPACE)", style="bold bright_white on red", in_check=True, armed=False, chain=chain_count)
            tui.log(
                f"{prefix} SPACE НАЖАТ! ({reason}{tier_str} -> Цель={target_angle_locked:.1f}°, Оценка={cur_est_angle:.1f}°, "
                f"delay={cur_lat:.1f}мс, lateness={pred_lateness_ms:+.2f}мс)",
                style="bold red",
            )
            if is_base_latency_test:
                val_status = "VALID" if experiment_valid else "INVALID_ASSERTION_FAILED"
                tui.log(
                    f"🧪 [BASE_LATENCY_TEST] requested_test_latency={requested_test_latency:.1f} actual_used_delay={cur_lat:.1f} ({val_status})",
                    style="bold green" if experiment_valid else "bold red",
                )
            learner.on_trigger(
                press_time=last_trigger_t,
                target_angle=target_angle_locked,
                speed_deg_s=last_speed,
                white_zone=locked_w,
                black_zone=locked_b,
                is_frenzy=is_frenzy_chain_active,
                trajectory_samples=len(predictor.history),
                used_latency_ms=cur_lat,
                scheduler_error_ms=sched_jitter_ms if sched_jitter_ms is not None else pred_lateness_ms,
                prediction_lateness_ms=pred_lateness_ms,
                selected_speed_tier=locked_speed_tier,
                speed_at_lock=speed_at_lock,
                last_frame_age_ms=frame_age_ms,
                trigger_reason=reason,
                duration_ms=((last_trigger_t - check_start_t) * 1000.0) if check_start_t else 0.0,
            )
            if last_timing_diag is not None:
                last_timing_diag["prediction_lateness_ms"] = pred_lateness_ms
                last_timing_diag["scheduler_jitter_ms"] = sched_jitter_ms
                last_timing_diag["detector_source_at_fire"] = last_fire_detector_source
                last_timing_diag["detector_fallback_at_fire"] = last_fire_detector_fallback
                last_timing_diag["speed_mode_switched_this_check"] = last_fire_mode_switched
                last_timing_diag["live_fit_spread_deg_s"] = last_fire_fit_spread
                last_timing_diag["fit_residual_mad_deg"] = last_fire_fit_residual
                last_timing_diag["fit_sample_count"] = last_fire_fit_sample_count
                last_timing_diag["fit_span_ms"] = last_fire_fit_span_ms

            if shadow_worker is not None:
                shadow_worker.notify_fire(
                    now=last_trigger_t,
                    planned_press_time=target_desired,
                    actual_press_time=last_trigger_t,
                    last_speed=last_speed,
                    reason=reason,
                    target_angle=target_angle_locked,
                )
            shadow_telem = fire_shadow_telem
            flight_recorder.on_trigger(
                now=last_trigger_t,
                reason=reason,
                target_angle=target_angle_locked,
                est_angle=cur_est_angle,
                last_speed=last_speed,
                latency_ms=cur_lat,
                selected_speed_tier=locked_speed_tier,
                used_profile_delay=cur_lat,
                shadow_profile_delay=locked_shadow_profile_delay,
                phase_telemetry=last_fire_phase_snapshot,
                measured_speed_at_lock=speed_at_lock,
                planned_press_time=target_desired,
                # Legacy recorder field retained for compatibility: this field is
                # prediction lateness. The true scheduler_jitter_ms is stored in
                # timing_diag and used by the phase tracker.
                scheduler_error_ms=pred_lateness_ms,
                last_frame_age_ms=frame_age_ms,
                timing_diag=last_timing_diag,
                trigger_mode=trigger_mode,
                provisional_speed=last_speed if is_fallback else None,
                confidence="none" if is_fallback else ("high" if len(predictor.history) >= 8 else "medium"),
                remaining_time_ms=rem_ms if "rem_ms" in locals() else None,
            )
            log_flight_record(
                f"[FIRE] {prefix} ({reason}) target={target_angle_locked:.1f} est={cur_est_angle:.1f} "
                f"speed={last_speed:.1f} tier={locked_speed_tier} lat={cur_lat:.1f} lateness={pred_lateness_ms:+.2f}ms "
                f"phase_audit_shadow={phase_tracker.phase_correction_ms:+.1f}ms regime={last_fire_compensation_regime} "
                f"requested_test_latency={requested_test_latency:.1f} actual_used_delay={cur_lat:.1f} exp_valid={experiment_valid} | "
                f"mode={shadow_telem['speed_mode']} live={shadow_telem['live_speed']:.1f}°/s spread={shadow_telem['live_fit_spread']:.1f}°/s "
                f"fitN={shadow_telem.get('fit_sample_count')} fitMAD={shadow_telem.get('fit_residual_mad_deg')} "
                f"switched={shadow_telem['mode_switched']}"
            )

    scheduler = PreciseTriggerScheduler(on_fire)

    def apply_geometric_miss_reclassification(learn_res: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Conservatively upgrade UNCONFIRMED -> MISS when post-hit geometry is
        unambiguous.

        Rules:
        - only a fired check can be reclassified;
        - an explicit hit_angle must exist (never use target fallback);
        - both GREAT (white) and GOOD arcs must be known;
        - hit must be outside BOTH success arcs;
        - keep a 1.0° boundary guard so detector noise near an edge stays
          UNCONFIRMED instead of becoming a false MISS;
        - if plateau_found is explicitly False, keep UNCONFIRMED but annotate
          the replay as a geometric MISS candidate.
        """
        if learn_res is None:
            return learn_res
        if learn_res.get("outcome") != "UNCONFIRMED":
            return learn_res
        if not pressed:
            return learn_res

        hit_angle = learn_res.get("hit_angle")
        if hit_angle is None or locked_w is None or locked_b is None:
            return learn_res

        def _arc_contains(angle: float, arc: Dict[str, Any]) -> bool:
            try:
                start = float(arc["start"]) % 360.0
                end = float(arc["end"]) % 360.0
                a = float(angle) % 360.0
            except (KeyError, TypeError, ValueError):
                return False
            width = (end - start) % 360.0
            rel = (a - start) % 360.0
            return rel <= width + 1e-6

        def _distance_to_arc_deg(angle: float, arc: Dict[str, Any]) -> Optional[float]:
            try:
                start = float(arc["start"]) % 360.0
                end = float(arc["end"]) % 360.0
                a = float(angle) % 360.0
            except (KeyError, TypeError, ValueError):
                return None
            if _arc_contains(a, arc):
                return 0.0

            def _abs_signed_delta(x: float, y: float) -> float:
                return abs((x - y + 180.0) % 360.0 - 180.0)

            return min(_abs_signed_delta(a, start), _abs_signed_delta(a, end))

        in_white = _arc_contains(hit_angle, locked_w)
        in_good = _arc_contains(hit_angle, locked_b)
        if in_white or in_good:
            return learn_res

        white_dist = _distance_to_arc_deg(hit_angle, locked_w)
        good_dist = _distance_to_arc_deg(hit_angle, locked_b)
        edge_distances = [d for d in (white_dist, good_dist) if d is not None]
        if not edge_distances:
            return learn_res

        success_edge_margin_deg = min(edge_distances)
        guard_deg = 1.0

        # Preserve the raw classifier result in every geometrically suspicious case.
        learn_res.setdefault("raw_outcome", "UNCONFIRMED")
        learn_res["geometric_outcome"] = "MISS"
        learn_res["geometric_success_edge_margin_deg"] = round(success_edge_margin_deg, 3)

        if success_edge_margin_deg <= guard_deg:
            learn_res["geometric_miss_candidate"] = True
            learn_res["geometric_reclassify_reason"] = "OUTSIDE_SUCCESS_ARC_BUT_WITHIN_BOUNDARY_GUARD"
            log_flight_record(
                f"[GEOMETRIC_MISS_CANDIDATE] hit={float(hit_angle):.2f}° outside success arcs, "
                f"nearest_edge={success_edge_margin_deg:.2f}° <= guard={guard_deg:.1f}°; keeping UNCONFIRMED"
            )
            return learn_res

        # Only an explicitly confirmed plateau may overwrite UNCONFIRMED.
        # Missing/None plateau is uncertainty, not proof of a stable stop angle.
        if learn_res.get("plateau_found") is not True:
            learn_res["geometric_miss_candidate"] = True
            learn_res["geometric_reclassify_reason"] = "NO_CONFIRMED_PLATEAU"
            log_flight_record(
                f"[GEOMETRIC_MISS_CANDIDATE] hit={float(hit_angle):.2f}° outside success arcs by "
                f"{success_edge_margin_deg:.2f}°, but plateau is not confirmed; keeping UNCONFIRMED"
            )
            return learn_res

        learn_res["outcome"] = "MISS"
        learn_res["outcome_reclassified_from"] = "UNCONFIRMED"
        learn_res["outcome_source"] = "GEOMETRIC_RECLASSIFICATION"
        learn_res["miss_category"] = learn_res.get("miss_category") or "GEOMETRIC_MISS"
        learn_res["geometric_miss_candidate"] = False
        learn_res["geometric_reclassify_reason"] = "HIT_OUTSIDE_GREAT_AND_GOOD_ARCS"

        msg = (
            f"🎯 [GEOMETRIC MISS] UNCONFIRMED -> MISS | hit={float(hit_angle):.1f}° "
            f"outside GREAT+GOOD, nearest success edge={success_edge_margin_deg:.1f}°"
        )
        tui.log(msg, style="bold red")
        log_flight_record(msg)
        return learn_res

    def process_check_outcome(learn_res: Optional[Dict[str, Any]], chain_idx: int) -> Tuple[str, str]:
        nonlocal eff_latency
        if learn_res is not None:
            if "session_base_speed" in learn_res:
                predictor.session_base_speed = learn_res["session_base_speed"]
            if not is_base_latency_test and not is_gen_rush:
                eff_latency = learner.latency_ms
                predictor.latency_s = eff_latency / 1000.0
            actual_outcome = learn_res.get("outcome", "УСПЕШНО")
            if is_base_latency_test and not last_experiment_valid:
                actual_outcome = "INVALID_EXPERIMENT"
                learn_res["outcome"] = "INVALID_EXPERIMENT"
                tui.log("⚠️ [BASE_LATENCY_TEST] Check marked INVALID_EXPERIMENT (used delay != requested latency)", style="bold red")
            miss_cat = learn_res.get("miss_category")
            if miss_cat == "CLEAN_RESPONSE_OUTLIER":
                tui.log("⚠️ [CLEAN_RESPONSE_OUTLIER] Scheduler/speed was exact, abnormal engine response delay", style="bold magenta")
            status = f"{actual_outcome} (#{chain_idx})" if chain_idx > 1 else f"{actual_outcome}"
            if not pressed:
                status = "НЕ НАЖАТ"
            fact_ang = learn_res.get("hit_angle")
            tracker.record_outcome(
                outcome=actual_outcome,
                speed_tier=locked_speed_tier,
                speed_at_lock=speed_at_lock,
                speed_at_fire=last_speed,
                prof_res=learn_res.get("speed_profile"),
            )
            tui.record_hit(
                outcome=actual_outcome,
                fact_angle=fact_ang,
                target_angle=learn_res.get("target_angle", target_angle_locked),
                error_deg=learn_res.get("error_deg", 0.0),
                error_ms=learn_res.get("error_ms", 0.0),
                chain=chain_idx,
                latency_ms=(last_fire_delay if last_fire_delay is not None else get_effective_latency(chain_idx)),
            )
            fact_str = f"{fact_ang:.1f}°" if fact_ang is not None else "N/A"
            _tgt_for_log = learn_res.get("target_angle", target_angle_locked)
            tgt_str = f"{_tgt_for_log:.1f}°" if _tgt_for_log is not None else "N/A"
            log_flight_record(
                f"[RESULT] {status} | Fact={fact_str} Target={tgt_str} "
                f"err={learn_res.get('error_deg') if learn_res.get('error_deg') is not None else 'N/A'} "
                f"error_ms={learn_res.get('error_ms') if learn_res.get('error_ms') is not None else 'N/A'} | "
                f"base_lat={configured_base_latency_ms:.1f}ms actual_delay={last_fire_delay if last_fire_delay is not None else 'N/A'}"
            )
            if abs(learn_res.get("adjustment_ms", 0.0)) >= 0.05:
                sign_adj = "+" if learn_res["adjustment_ms"] >= 0 else ""
                tui.log(f"🧠 AI Learner: задержка {learn_res['prev_latency_ms']:.1f}мс -> {learn_res['new_latency_ms']:.1f}мс ({sign_adj}{learn_res['adjustment_ms']:.1f}мс)", style="cyan")

            # Strict Target Geometry Assertion & Logging
            w_start = learn_res.get("white_start")
            w_end = learn_res.get("white_end")
            w_center = learn_res.get("white_center")
            w_width = learn_res.get("white_width")
            tgt_ang_check = learn_res.get("target_angle", target_angle_locked)
            c_err_deg = learn_res.get("center_error_deg")
            if c_err_deg is None:
                c_err_deg = learn_res.get("error_deg")
            c_err_ms = learn_res.get("center_error_ms")
            if c_err_ms is None:
                c_err_ms = learn_res.get("error_ms")
            e_err_deg = learn_res.get("entry_edge_error_deg")
            e_err_ms = learn_res.get("entry_edge_error_ms")

            if eff_target == "GREAT" and w_center is not None and tgt_ang_check is not None:
                ang_diff = abs((tgt_ang_check - w_center + 180.0) % 360.0 - 180.0)
                if ang_diff > 0.1:
                    tui.log(f"❌ [TARGET GEOMETRY ASSERTION FAILED] target_angle={tgt_ang_check:.2f}° != white_center={w_center:.2f}° (diff={ang_diff:.2f}°)", style="bold red")

            w_str = f"[{w_start:.1f}°, {w_end:.1f}°]" if (w_start is not None and w_end is not None) else "N/A"
            w_w_str = f"{w_width:.1f}°" if w_width is not None else "N/A"
            w_c_str = f"{w_center:.1f}°" if w_center is not None else "N/A"
            tgt_str_g = f"{tgt_ang_check:.1f}°" if tgt_ang_check is not None else "N/A"
            fact_str_g = f"{fact_ang:.1f}°" if fact_ang is not None else "N/A"
            e_err_str = f"{e_err_deg:+.1f}° ({e_err_ms:+.1f}ms)" if (e_err_deg is not None and e_err_ms is not None) else "N/A"

            c_deg_str = f"{c_err_deg:+.1f}°" if c_err_deg is not None else "N/A"
            c_ms_str = f"{c_err_ms:+.1f}ms" if c_err_ms is not None else "N/A"
            geom_msg = (
                f"[GEOMETRY] white={w_str} (w={w_w_str}, center={w_c_str}) | target={tgt_str_g} fact={fact_str_g} | "
                f"err_center={c_deg_str} ({c_ms_str}) err_entry={e_err_str}"
            )
            log_flight_record(geom_msg)
            tui.log(geom_msg, style="bright_cyan")

            # Response Phase Tracking
            phase_at_fire = dict(last_fire_phase_snapshot) if isinstance(last_fire_phase_snapshot, dict) else {
                "base_compensation_ms": configured_base_latency_ms,
                "response_phase_correction_ms": None,
                "effective_compensation_ms": last_fire_delay,
                "chain_offset_ms": chain_latency_offset_ms * max(0, chain_idx - 1),
                "effective_with_chain_ms": last_fire_delay,
            }
            phase_res = phase_tracker.record_outcome(
                center_error_ms=c_err_ms,
                outcome=actual_outcome,
                plateau_found=learn_res.get("plateau_found"),
                trigger_reason=last_fire_reason,
                trigger_mode=last_fire_trigger_mode,
                scheduler_jitter_ms=last_fire_sched_jitter,
                speed_at_lock=speed_at_lock,
                speed_at_fire=last_speed,
                last_frame_age_ms=last_fire_frame_age,
                actual_used_delay_ms=last_fire_delay,
                is_detector_fallback=last_fire_detector_fallback,
                detector_source=last_fire_detector_source,
                mode_switched_this_check=last_fire_mode_switched,
                live_fit_spread_deg_s=last_fire_fit_spread,
                now=time.monotonic(),
            )
            p_telem = phase_tracker.get_telemetry()

            lead_res = None
            if is_gen_rush:
                lead_res = lead_controller.record_outcome(
                    center_error_ms=c_err_ms,
                    actual_used_delay_ms=last_fire_delay,
                    outcome=actual_outcome,
                    plateau_found=learn_res.get("plateau_found"),
                    trigger_mode=last_fire_trigger_mode,
                    scheduler_jitter_ms=last_fire_sched_jitter,
                    frame_age_ms=last_fire_frame_age,
                    detector_fallback=last_fire_detector_fallback,
                    compensation_regime=last_fire_compensation_regime,
                    fit_sample_count=last_fire_fit_sample_count,
                    fit_residual_mad_deg=last_fire_fit_residual,
                    fit_spread_deg_s=last_fire_fit_spread,
                )
                lt = lead_controller.telemetry()
                learn_res["lead_level_update"] = lead_res
                learn_res["lead_level_telemetry"] = lt
                if lead_res.get("accepted"):
                    if lead_res.get("updated"):
                        tui.log(
                            f"🧭 [LEAD_LEVEL] ideal={lead_res['ideal_lead_ms']:.1f}ms "
                            f"median={lead_res['rolling_median_ms']:.1f}ms MAD={lead_res['rolling_mad_ms']:.1f}ms "
                            f"lead {lead_res['lead_before_ms']:.1f}->{lead_res['current_lead_ms']:.1f}ms "
                            f"({lead_res.get('update_reason')}, {lead_res['step_ms']:+.1f}ms; next check)",
                            style="bold cyan",
                        )
                    else:
                        tui.log(
                            f"🧭 [LEAD_LEVEL] sample={lead_res['ideal_lead_ms']:.1f}ms "
                            f"median={lead_res['rolling_median_ms']:.1f}ms MAD={lead_res['rolling_mad_ms']:.1f}ms "
                            f"N={lead_res['sample_count']} current={lead_controller.current_lead_ms:.1f}ms",
                            style="dim cyan",
                        )
                else:
                    tui.log(
                        f"🧭 [LEAD_LEVEL] rejected: {lead_res.get('reject_reason')} "
                        f"(current={lead_controller.current_lead_ms:.1f}ms)",
                        style="dim",
                    )

            learn_res["configured_base_latency_ms"] = configured_base_latency_ms
            learn_res["actual_used_delay_ms"] = last_fire_delay
            learn_res["detector_source_at_fire"] = last_fire_detector_source
            learn_res["detector_fallback_at_fire"] = last_fire_detector_fallback
            learn_res["speed_mode_switched_this_check"] = last_fire_mode_switched
            learn_res["live_fit_spread_deg_s"] = last_fire_fit_spread
            learn_res["phase_at_fire"] = phase_at_fire
            learn_res["phase_after_outcome"] = phase_res
            learn_res["phase_telemetry_after_outcome"] = p_telem
            learn_res["compensation_regime"] = last_fire_compensation_regime
            learn_res["genrush_delivery_lead_ms"] = (lead_controller.current_lead_ms if is_gen_rush else None)
            learn_res["configured_perk_latency_ms"] = None
            learn_res["phase_actuation"] = "SHADOW_ONLY" if is_gen_rush else "ACTIVE"
            learn_res["capture_framerate_mode"] = "vfr" if is_gen_rush else "cfr"

            def _fmt_phase_ms(value, *, signed=False):
                if value is None:
                    return "N/A"
                return f"{value:+.1f}ms" if signed else f"{value:.1f}ms"

            phase_accepted = bool(phase_res.get("accepted", False))
            phase_tag = "[LEAD_AUDIT_SHADOW]" if is_gen_rush else "[PHASE]"
            phase_msg = (
                f"{phase_tag} base={_fmt_phase_ms(p_telem.get('base_compensation_ms'))} "
                f"corr={_fmt_phase_ms(p_telem.get('response_phase_correction_ms'), signed=True)} "
                f"effective={_fmt_phase_ms(p_telem.get('effective_compensation_ms'))} "
                f"rolling_med={_fmt_phase_ms(p_telem.get('rolling_error_median_ms'), signed=True)} "
                f"samples={p_telem.get('phase_sample_count', 0)}"
            )
            if not phase_accepted:
                rej = phase_res.get("reject_reason") or phase_res.get("rejection_reason") or "unqualified"
                phase_msg += f" (rejected: {rej})"
            log_flight_record(phase_msg)
            if not is_gen_rush:
                tui.log(phase_msg, style="bold magenta" if phase_accepted else "dim")

            return actual_outcome, status
        else:
            actual_outcome = "MISS" if pressed else "TIMEOUT"
            status = f"{actual_outcome} (#{chain_idx})" if chain_idx > 1 else f"{actual_outcome}"
            if not pressed:
                status = "НЕ НАЖАТ"
            tracker.record_outcome(
                outcome=actual_outcome,
                speed_tier=locked_speed_tier,
                speed_at_lock=speed_at_lock,
                speed_at_fire=last_speed,
                prof_res=None,
            )
            log_flight_record(f"[RESULT] {status} (no learn data)")
            return actual_outcome, status


    try:
        while RUNNING:
            t_loop_start = time.monotonic()
            lmb_held = mouse_tracker.is_held()

            # 1. Grab fresh screen frame
            t_grab_start = time.monotonic()
            frame_bgr, frame_id, decode_ready_ts = grabber.grab(wait_new=True, last_id=last_frame_id, timeout=0.03)
            if frame_bgr is None:
                time.sleep(0.001)
                continue
            t_grab_done = time.monotonic()
            grab_wait_ms = (t_grab_done - t_grab_start) * 1000.0
            now = decode_ready_ts if decode_ready_ts is not None else t_grab_done
            python_handoff_age_ms = (t_grab_done - now) * 1000.0
            capture_wait_ms = grab_wait_ms
            frame_age_ms = python_handoff_age_ms
            last_frame_id = frame_id
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            tracker.record_frame(
                t_grab_done=t_grab_done,
                decode_ready_ts=decode_ready_ts,
                grab_wait_ms=grab_wait_ms,
                python_handoff_age_ms=python_handoff_age_ms,
                frame_bgr=frame_bgr,
            )

            # 2. Vision Detection & LMB Release Tracking
            # If in an active check and LMB was released mid-check: ABORT immediately!
            if in_check and require_lmb and (not lmb_held or mouse_tracker.was_released_since(check_start_t)):
                scheduler.cancel()
                in_check = False
                armed = False
                pressed = True  # Block any subsequent fire dispatch for this check
                dur = now - check_start_t
                last_check_end_t = now
                learn_res = learner.conclude_check(is_aborted=True, abort_reason="LMB_RELEASED")
                flight_recorder.end_check(now, outcome_info=learn_res, locked_w=locked_w, locked_b=locked_b)
                tracker.record_outcome(
                    outcome="ABORTED_LMB",
                    speed_tier=locked_speed_tier,
                    speed_at_lock=speed_at_lock,
                    speed_at_fire=last_speed,
                    prof_res=None,
                )
                tui.record_hit(
                    outcome="ABORTED_LMB",
                    fact_angle=target_angle_locked,
                    target_angle=target_angle_locked,
                    error_deg=0.0,
                    error_ms=0.0,
                    chain=chain_count,
                    latency_ms=eff_latency,
                )
                tui.set_status("ABORTED (LMB UP)", style="bold yellow", in_check=False, armed=False, chain=chain_count)
                tui.log("🛑 ПРОВЕРКА ПРЕРВАНА (отжат ЛКМ) — выстрел отменен, промах не засчитан", style="yellow")
                log_flight_record(f"[ABORT] LMB released during check after {dur:.2f}s (no miss recorded)")
                det = None
                expected_angle = None
                predictor.reset()
                vision.reset()
                summarize_shadow("ABORTED_LMB")
                pred = None
                locked_w = None
                locked_b = None
                chain_count = 0
            elif require_lmb and not lmb_held:
                det = None
                expected_angle = None
            elif require_lmb and mouse_tracker.time_since_release() < 0.15:
                # Suppress ghost trailing check immediately after LMB release
                det = None
                expected_angle = None
            else:
                det = vision.detect_frame(
                    frame_gray,
                    frame_bgr,
                    expected_angle=expected_angle,
                    search_window=vision_search_win,
                    is_pressed=pressed,
                )
            t_vision_done = time.monotonic()
            vision_ms = (t_vision_done - t_grab_done) * 1000.0

            # Asynchronous non-blocking shadow detector submission (zero critical-path wait)
            if shadow_worker is not None:
                shadow_worker.submit_frame(
                    now=now,
                    frame_gray=frame_gray,
                    frame_bgr=frame_bgr,
                    in_check=in_check,
                    baseline_det=det,
                    dt_frame=dt_frame,
                    expected_speed=learner.base_speed_tracker.current_prior,
                    baseline_vision_ms=vision_ms,
                    speed_at_lock=speed_at_lock,
                    target_angle=target_angle_locked,
                )

            last_timing_diag = {
                "capture_wait_ms": round(capture_wait_ms, 2),
                "frame_age_ms": round(frame_age_ms, 2),
                "vision_ms": round(vision_ms, 2),
                "predict_ms": 0.0,
            }

            if det is not None:
                frame_gap_since_last_det = (now - last_det_t) if last_det_t > 0 else 0.0
                dt_frame = (now - last_det_t) if (last_det_t > 0 and 0.001 <= (now - last_det_t) <= 0.1) else (1.0 / eff_fps)
                last_det_t = now
                current_needle = det["needle_angle"]

                if not in_check:
                    in_check = True
                    if shadow_worker is not None:
                        shadow_worker.on_check_start(now, target_angle_locked)
                    # A brand-new ring is always an independent check.  Do NOT
                    # infer Frenzy merely because it appeared within 1.5 s of the
                    # previous ring; real Frenzy is confirmed only below while the
                    # same ring is still active after a press.
                    is_frenzy_chain_active = False
                    chain_count = 1
                    predictor.reset(
                        keep_speed=False,
                        default_speed=None,
                        is_chain=False,
                        session_base_speed=learner.base_speed_tracker.current_prior,
                    )
                    predictor.latency_s = get_effective_latency(chain_count) / 1000.0
                    check_start_t = now
                    armed = False
                    pressed = False
                    pred = None
                    press_t = None
                    locked_w = None
                    locked_b = None
                    expected_angle = None
                    last_needle_angle = current_needle
                    locked_speed_tier = None
                    locked_tier_latency = None
                    locked_compensation_regime = "CONTINUOUS" if is_gen_rush else "BASE"
                    locked_shadow_profile_delay = None
                    speed_at_lock = None
                    scheduler.cancel()
                    flight_recorder.start_check(
                        now=now,
                        chain_count=chain_count,
                        latency_ms=get_effective_latency(chain_count),
                        target_mode=eff_target,
                        target_ratio=eff_ratio,
                        locked_w=locked_w,
                        locked_b=locked_b,
                    )
                    chain_str = f" (СЕРИЯ #{chain_count})" if chain_count > 1 else ""
                    tui.set_status(f"ACQUIRING{chain_str}", style="bold yellow", in_check=True, chain=chain_count)
                    tui.log(f">>> SKILL CHECK ОБНАРУЖЕН!{chain_str} Захват цели...", style="yellow")

                # FRENZY / CONTINUOUS CHAIN DETECTION:
                # If Space was already fired and the ring stayed on screen, check if a NEW check chained in.
                # Hardware roundtrip latency is ~98ms, key released in 30ms, re-arming requires >= 80ms lockout.
                # Re-arming requires physical confirmation:
                # 1. Great/Good zone relocated (>15°)
                # 2. Needle jumped back (< -25°)
                # 3. New orbital pass (90° <= forward_dist <= 300°)
                # 4. Ring reappeared after visual disappearance (gap >= 50ms)
                elif pressed:
                    fresh_w, fresh_b = vision.extract_zones(det, None)
                    is_frenzy_chain = False

                    if (now - last_trigger_t) >= 0.080:
                        zone_relocated = False
                        if fresh_w is not None and locked_w is not None:
                            zone_diff = abs((fresh_w["center"] - locked_w["center"] + 180.0) % 360.0 - 180.0)
                            zone_relocated = zone_diff > 15.0
                        elif fresh_b is not None and locked_b is not None:
                            zone_b_diff = abs((fresh_b["start"] - locked_b["start"] + 180.0) % 360.0 - 180.0)
                            zone_relocated = zone_b_diff > 15.0

                        needle_step = (current_needle - trigger_needle_angle + 180.0) % 360.0 - 180.0 if trigger_needle_angle is not None else 0.0
                        needle_jumped_back = needle_step < -25.0
                        forward_dist = (current_needle - target_angle_locked) % 360.0 if target_angle_locked is not None else 0.0
                        needle_new_pass = (90.0 <= forward_dist <= 300.0) and (fresh_w is not None or fresh_b is not None)
                        ring_reappeared = (frame_gap_since_last_det >= 0.050) and (fresh_w is not None or fresh_b is not None)

                        frenzy_reasons = []
                        if zone_relocated:
                            frenzy_reasons.append("ZONE_RELOCATED")
                        if needle_jumped_back:
                            frenzy_reasons.append("NEEDLE_ROLLBACK")
                        if needle_new_pass:
                            frenzy_reasons.append("NEW_ORBIT_PASS")
                        if ring_reappeared:
                            frenzy_reasons.append("RING_REAPPEARED")
                        is_frenzy_chain = bool(frenzy_reasons)
                        frenzy_reason = "+".join(frenzy_reasons) if frenzy_reasons else None
                    else:
                        frenzy_reason = None

                    if is_frenzy_chain:
                        chain_count += 1
                        is_frenzy_chain_active = True
                        check_start_t = now
                        pressed = False
                        armed = False
                        pred = None
                        press_t = None
                        scheduler.cancel()

                        # Preserve actual measured speed across all game speeds up to 1500°/s
                        chain_speed = last_speed if (20.0 <= last_speed <= 1500.0) else (last_confirmed_speed or DEFAULT_SPEED_DEG_S)
                        predictor.reset(
                            keep_speed=True,
                            default_speed=chain_speed,
                            is_chain=True,
                            session_base_speed=learner.base_speed_tracker.current_prior,
                        )
                        last_speed = chain_speed
                        last_confirmed_speed = chain_speed
                        expected_angle = None
                        vision.reset()
                        summarize_shadow(f"FRENZY_CHAIN_#{chain_count-1}")
                        learn_res = learner.conclude_check()
                        learn_res = apply_geometric_miss_reclassification(learn_res)
                        process_check_outcome(learn_res, chain_idx=chain_count - 1)
                        flight_recorder.end_check(now, outcome_info=learn_res, locked_w=locked_w, locked_b=locked_b)
                        locked_w, locked_b = (fresh_w, fresh_b) if fresh_w is not None else (None, None)
                        locked_speed_tier = None
                        locked_tier_latency = None
                        locked_compensation_regime = "CONTINUOUS" if is_gen_rush else "BASE"
                        locked_shadow_profile_delay = None
                        speed_at_lock = None
                        cur_chain_lat = get_effective_latency(chain_count)
                        predictor.latency_s = cur_chain_lat / 1000.0
                        if not is_gen_rush:
                            eff_latency = cur_chain_lat
                        flight_recorder.start_check(
                            now=now,
                            chain_count=chain_count,
                            latency_ms=(get_effective_latency(chain_count) if is_gen_rush else cur_chain_lat),
                            target_mode=eff_target,
                            target_ratio=eff_ratio,
                            locked_w=locked_w,
                            locked_b=locked_b,
                        )
                        target_str = f"Белая={locked_w['center']:.1f}°" if locked_w else "поиск..."
                        tui.set_status(f"FRENZY #{chain_count} (ПЕРЕЗАХВАТ)", style="bold bright_yellow on red", in_check=True, chain=chain_count)
                        tui.log(
                            f"🔥 [FRENZY #{chain_count}] Непрерывный скилл чек! Перезахват ({target_str}) "
                            f"reason={frenzy_reason or 'UNKNOWN'}",
                            style="bold yellow",
                        )
                        log_flight_record(
                            f"[FRENZY_CONFIRM] chain={chain_count} reason={frenzy_reason or 'UNKNOWN'}"
                        )

                # Lock zones if not locked yet
                if locked_w is None:
                    w_d, b_d = vision.extract_zones(det, None)
                    if w_d is not None:
                        locked_w, locked_b = w_d, b_d
                    elif b_d is not None:
                        # Reconstruct Great zone directly from Good zone geometry
                        w_s = (b_d["start"] - 9.5) % 360.0
                        locked_w = {
                            "start": float(w_s),
                            "end": float(b_d["start"]),
                            "width": 9.5,
                            "center": float((w_s + 4.75) % 360.0),
                        }
                        locked_b = b_d

                # Freeze speed estimate at trigger: post-hit frozen frames would
                # otherwise pollute the regression and corrupt chained checks.
                is_valid_frame = False
                if not pressed:
                    is_valid_frame = predictor.update(now, det["needle_angle"], det["needle_strength"], locked_w, locked_b)
                last_needle_angle = current_needle
                expected_angle = (current_needle + predictor.speed_deg_s * dt_frame) % 360.0
                vision_search_win = max(35.0, predictor.speed_deg_s * dt_frame + 20.0)

                if learner.pending_hit is not None:
                    learner.observe_sample(now, current_needle, det.get("needle_strength", 20.0))

                # Kinematic prediction and trigger scheduling
                if not pressed and (locked_w is not None or locked_b is not None):
                    eff_t = eff_target if locked_w is not None else "GOOD"
                    if is_valid_frame:
                        pred = predictor.predict(now, det["needle_angle"], target=eff_t)
                        # Lock speed tier and delay only when speed is stable or via explicit short check fallback
                        if locked_tier_latency is None and pred is not None:
                            if predictor.has_stable_speed():
                                if is_gen_rush:
                                    speed_at_lock = float(predictor.speed_deg_s)
                                    # Legacy telemetry field only: bucket measured speed for reports.
                                    # This bucket never changes timing or predictor state.
                                    locked_speed_tier = int(round(speed_at_lock / 25.0) * 25)
                                    locked_tier_latency = get_effective_latency(chain_count)
                                    locked_compensation_regime = "CONTINUOUS_MEASURED_SPEED"
                                    locked_shadow_profile_delay = None
                                    predictor.latency_s = locked_tier_latency / 1000.0
                                    ctelem = predictor.get_shadow_telemetry()
                                    tui.log(
                                        f"🔒 [GEN_RUSH CONT] speed={speed_at_lock:.1f}°/s "
                                        f"fitN={ctelem.get('fit_sample_count', 0)} span={ctelem.get('fit_span_ms', 0.0):.1f}ms "
                                        f"resMAD={ctelem.get('fit_residual_mad_deg') if ctelem.get('fit_residual_mad_deg') is not None else 'N/A'}° "
                                        f"-> lead={locked_tier_latency:.1f}ms",
                                        style="bold yellow",
                                    )
                                elif is_base_latency_test or (predictor.speed_mode == SPEED_MODE_BASE and not predictor.mode_switched):
                                    locked_speed_tier = int(round(predictor.session_base_speed))
                                    locked_tier_latency = eff_latency if is_base_latency_test else get_effective_latency(chain_count)
                                    speed_at_lock = predictor.session_base_speed
                                    predictor.latency_s = locked_tier_latency / 1000.0
                                    eff_latency = locked_tier_latency
                                    tui.log(f"🔒 Base Speed Lock: ~{locked_speed_tier}°/с (prior {predictor.session_base_speed:.1f}°/с) -> delay={locked_tier_latency:.1f}мс", style="dim")
                                else:
                                    locked_speed_tier = learner.speed_profiles.get_tier(predictor.speed_deg_s)
                                    locked_tier_latency = learner.get_latency_for_speed(predictor.speed_deg_s)
                                    speed_at_lock = predictor.speed_deg_s
                                    predictor.latency_s = locked_tier_latency / 1000.0
                                    eff_latency = locked_tier_latency
                                    tui.log(f"🔒 Speed Lock: ~{locked_speed_tier}°/с (измерено {speed_at_lock:.1f}°/с, N={len(predictor.history)}) -> delay={locked_tier_latency:.1f}мс", style="dim")
                                pred = predictor.predict(now, det["needle_angle"], target=eff_t)
                            elif pred["angular_distance_deg"] < 35.0 or (pred["time_until_press_ms"] <= 25.0 and len(predictor.history) >= 2):
                                # Explicit logged fallback path for very short checks (< 35°)
                                if is_gen_rush:
                                    # Short check: only commit if we already have a real measured
                                    # speed from >=3 fresh samples.  Never substitute 278°/s.
                                    usable = bool(getattr(predictor, "has_usable_speed", lambda: False)())
                                    if usable:
                                        speed_at_lock = float(predictor.speed_deg_s)
                                        locked_speed_tier = int(round(speed_at_lock / 25.0) * 25)
                                        locked_tier_latency = get_effective_latency(chain_count)
                                        locked_compensation_regime = "CONTINUOUS_EARLY_MEASURED_SPEED"
                                        locked_shadow_profile_delay = None
                                        predictor.state = STATE_LOCKED
                                        predictor.latency_s = locked_tier_latency / 1000.0
                                        tui.log(
                                            f"⚠️ [GEN_RUSH EARLY] short check, using measured speed={speed_at_lock:.1f}°/s "
                                            f"dist={pred['angular_distance_deg']:.1f}° lead={locked_tier_latency:.1f}ms",
                                            style="bold yellow",
                                        )
                                    else:
                                        # No base-speed fallback: wait for a trustworthy measurement.
                                        locked_tier_latency = None
                                        speed_at_lock = None
                                elif is_base_latency_test or (predictor.speed_mode == SPEED_MODE_BASE and not predictor.mode_switched):
                                    fallback_tier = int(round(predictor.session_base_speed))
                                    locked_tier_latency = eff_latency if is_base_latency_test else get_effective_latency(chain_count)
                                    locked_speed_tier = fallback_tier
                                    speed_at_lock = None
                                    predictor.state = STATE_LOCKED
                                    predictor.latency_s = locked_tier_latency / 1000.0
                                    eff_latency = locked_tier_latency
                                    tui.log(
                                        f"⚠️ [FALLBACK_SHORT_CHECK] Close spawn (dist={pred['angular_distance_deg']:.1f}° < 35°), "
                                        f"forcing base speed fallback ~{fallback_tier}°/s (delay={locked_tier_latency:.1f}ms)",
                                        style="bold yellow",
                                    )
                                else:
                                    fallback_tier = learner.speed_profiles.get_tier(predictor.speed_deg_s)
                                    locked_tier_latency = learner.get_latency_for_speed(predictor.speed_deg_s)
                                    locked_speed_tier = fallback_tier
                                    speed_at_lock = None  # None indicates unconfirmed fallback lock (does not train profiles)
                                    predictor.state = STATE_LOCKED
                                    predictor.latency_s = locked_tier_latency / 1000.0
                                    eff_latency = locked_tier_latency
                                    tui.log(
                                        f"⚠️ [FALLBACK_SHORT_CHECK] Close spawn (dist={pred['angular_distance_deg']:.1f}° < 35°), "
                                        f"forcing fallback lock ~{fallback_tier}°/s (delay={locked_tier_latency:.1f}ms)",
                                        style="bold yellow",
                                    )
                                pred = predictor.predict(now, det["needle_angle"], target=eff_t)

                        # LIFECYCLE RULE: Scheduler is strictly forbidden from firing in provisional speed!
                        # Only proceed with commitment and scheduling if speed is LOCKED.
                        if pred is not None and locked_tier_latency is not None:
                            target_angle_locked = pred["target_angle"]
                            rem_ms = pred["time_until_press_ms"]
                            press_t = pred["press_timestamp"]
                            speed = pred["speed_deg_s"]
                            last_speed = speed

                            is_fb = (speed_at_lock is None)
                            imm_reason = "FALLBACK_NO_LOCK" if is_fb else "IMMEDIATE"
                            sched_reason = "FALLBACK_NO_LOCK" if is_fb else "СПИН-ТАЙМЕР"

                            if pred.get("should_press_now", False):
                                predictor.state = STATE_COMMITTED
                                scheduler.trigger_now(imm_reason, desired_press_time=press_t)
                            elif press_t <= now:
                                min_hist = 2 if (chain_count > 1 or is_frenzy_chain_active) else 4
                                if armed or len(predictor.history) >= min_hist:
                                    predictor.state = STATE_COMMITTED
                                    scheduler.trigger_now(imm_reason, desired_press_time=press_t)
                            elif not armed:
                                armed = True
                                predictor.state = STATE_COMMITTED
                                scheduler.schedule(press_t, reason=sched_reason, desired_press_time=press_t)
                                # Clean critical path: TUI updates strictly AFTER scheduling commit
                                chain_str = f" [Серия #{chain_count}]" if chain_count > 1 else ""
                                tui.set_status(f"LOCKED [{target_angle_locked:.0f}° | {rem_ms:.0f}ms]", style="bold bright_green", in_check=True, armed=True, chain=chain_count)
                                cur_rep_lat = locked_tier_latency if locked_tier_latency is not None else get_effective_latency(chain_count)
                                tui.update_metrics(
                                    fps=current_fps or eff_fps,
                                    latency_ms=cur_rep_lat,
                                    target_ratio=eff_ratio,
                                    needle_speed=speed,
                                    target_angle=target_angle_locked,
                                    rem_ms=rem_ms,
                                )
                            elif rem_ms > 1.5 and pred["angular_distance_deg"] <= 330.0:
                                check_age_ms = (now - check_start_t) * 1000.0
                                if rem_ms > 20.0 or check_age_ms < 150.0:
                                    scheduler.schedule(press_t, reason=sched_reason, desired_press_time=press_t)

                    # Safety timer backup: only fire if current time has exceeded target by 8ms and trigger was actually armed
                    if armed and press_t is not None and now >= (press_t + 0.008):
                        predictor.state = STATE_COMMITTED
                        scheduler.trigger_now("ТАЙМЕР-BACKUP", desired_press_time=press_t)

            else:
                # Conclude skill check after 0.120s (120ms) debounce without detection
                if in_check and (now - last_det_t > 0.120):
                    in_check = False
                    scheduler.cancel()
                    last_check_end_t = now
                    dur = now - check_start_t
                    is_aborted = require_lmb and (not lmb_held or mouse_tracker.was_released_since(check_start_t))
                    learn_res = learner.conclude_check(is_aborted=is_aborted, abort_reason="LMB_RELEASED" if is_aborted else "")
                    if not is_aborted:
                        learn_res = apply_geometric_miss_reclassification(learn_res)
                    actual_outcome, status = process_check_outcome(learn_res, chain_idx=chain_count)
                    flight_recorder.end_check(now, outcome_info=learn_res, locked_w=locked_w, locked_b=locked_b)
                    tui.log(f"<<< SKILL CHECK ЗАВЕРШЁН (длительность: {dur:.2f}с, серия: {chain_count}) -> {status}", style="dim")
                    if 20.0 <= last_speed <= 1500.0:
                        last_confirmed_speed = last_speed
                    predictor.reset(
                        keep_speed=False,
                        default_speed=None,
                        is_chain=False,
                        session_base_speed=learner.base_speed_tracker.current_prior,
                    )
                    vision.reset()
                    summarize_shadow(actual_outcome)
                    pred = None
                    locked_w = None
                    locked_b = None
                    expected_angle = None
                    last_needle_angle = None
                    locked_speed_tier = None
                    locked_tier_latency = None
                    locked_compensation_regime = "CONTINUOUS" if is_gen_rush else "BASE"
                    locked_shadow_profile_delay = None
                    speed_at_lock = None
                    trigger_needle_angle = None
                    is_frenzy_chain_active = False
                    pressed = False
                    armed = False
                    last_experiment_valid = True


            # Record stage timing diagnostics and dispatch frame to flight recorder (non-critical path)
            t_decision_done = time.monotonic()
            predict_ms = (t_decision_done - t_vision_done) * 1000.0
            tracker.record_diagnostics(vision_ms=vision_ms, predict_ms=predict_ms)
            timing_diag = {
                "capture_wait_ms": round(capture_wait_ms, 2),
                "frame_age_ms": round(frame_age_ms, 2),
                "vision_ms": round(vision_ms, 2),
                "predict_ms": round(predict_ms, 2),
            }
            last_timing_diag = timing_diag

            flight_recorder.on_frame(
                now=now,
                frame_bgr=frame_bgr,
                det=det,
                pred_info=pred if in_check else None,
                timing_diag=timing_diag,
            )

            # Render HUD if enabled (throttled to 30 FPS to eliminate waitKey jitter)
            if show_hud:
                t_hud_now = time.monotonic()
                if (t_hud_now - last_hud_t) >= 0.033:
                    last_hud_t = t_hud_now
                    hud_frame = render_hud(
                        frame_bgr=frame_bgr,
                        det=det,
                        pred=pred if in_check else None,
                        current_fps=current_fps,
                        last_trigger_t=last_trigger_t,
                        now=now,
                        dry_run=dry_run,
                        target_mode=eff_target,
                        locked_w=locked_w,
                        locked_b=locked_b,
                        latency_ms=eff_latency,
                        target_ratio=eff_ratio,
                        chain_count=max(1, chain_count),
                    )
                    cv2.imshow(HUD_WINDOW_NAME, hud_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        handle_tui_action("quit")
                        break
                    elif key in (ord("+"), ord("=")):
                        handle_tui_action("inc_latency")
                    elif key in (ord("-"), ord("_")):
                        handle_tui_action("dec_latency")
                    elif key == ord("]"):
                        handle_tui_action("inc_offset")
                    elif key == ord("["):
                        handle_tui_action("dec_offset")

            # Hot-reload config if file changed
            if now - last_cfg_check >= 1.0:
                last_cfg_check = now
                if CONFIG_PATH.exists():
                    try:
                        mtime = CONFIG_PATH.stat().st_mtime
                        if mtime != last_cfg_mtime:
                            last_cfg_mtime = mtime
                            fresh_cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                            new_lat = fresh_cfg.get("latency_ms", eff_latency)
                            new_rat = fresh_cfg.get("target_offset_ratio", eff_ratio)
                            if not is_base_latency_test and not is_gen_rush:
                                if abs(new_lat - eff_latency) > 0.05 or abs(new_rat - eff_ratio) > 0.01:
                                    eff_latency = new_lat
                                    eff_ratio = new_rat
                                    predictor.latency_s = eff_latency / 1000.0
                                    predictor.target_offset_ratio = eff_ratio
                                    learner.latency_ms = eff_latency
                                    tui.log(f"🔄 Авто-обновление из config.json: задержка={eff_latency:.1f}мс, смещение={eff_ratio*100:.0f}%", style="cyan")
                    except Exception:
                        pass

            # Telemetry monitor in TUI
            fps_count += 1
            if now - fps_timer >= 1.0:
                current_fps = fps_count / (now - fps_timer)
                fps_count = 0
                fps_timer = now
                tui.update_metrics(
                    fps=current_fps,
                    latency_ms=(lead_controller.current_lead_ms if is_gen_rush else eff_latency),
                    target_ratio=eff_ratio,
                    needle_speed=last_speed if in_check else 0.0,
                    target_angle=target_angle_locked if in_check else 0.0,
                    rem_ms=rem_ms if in_check and pred else 0.0,
                    lmb_held=lmb_held,
                )

    finally:
        flight_recorder.close()
        tui.stop()
        scheduler.cancel()
        mouse_tracker.stop()
        grabber.close()
        trigger_hw.close()
        if show_hud:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        tui.print_summary()
        learner.print_training_summary()
        print(tracker.generate_report(learner=learner, grabber=grabber))


def main():
    ap = argparse.ArgumentParser(description="Violent District Autonomous Skill Check Bot")
    ap.add_argument("--dry-run", action="store_true", help="Тестовый режим без нажатий (эмуляция расчетов)")
    ap.add_argument("--hud", action="store_true", help="Включить графический HUD-оверлей")
    ap.add_argument("--require-lmb", action="store_true", help="Требовать зажатия ЛКМ для активации")
    ap.add_argument("--fps", type=int, default=None, help="FPS захвата экрана (по умолчанию 120)")
    ap.add_argument("--latency", type=float, default=None, help="Компенсация задержки в мс (по умолчанию 68.0)")
    ap.add_argument("--target", choices=["GREAT", "GOOD"], default=None, help="Целевая зона (GREAT / GOOD)")
    ap.add_argument("--region", help="Область захвата WxH+X+Y")
    ap.add_argument("--record-all", action="store_true", help="Записывать MP4 видео для всех проверок (по умолчанию видео только при промахах)")
    ap.add_argument("--freeze", action="store_true", help="Заморозить профили и задержку (без сохранения на диск)")
    ap.add_argument("--base-latency-test", action="store_true", help="Строгий режим тестирования базовой задержки (без смены мода, без профилей, без обучения)")
    ap.add_argument("--speed-mode", choices=["BASE_SPEED", "VARIABLE_SPEED", "BASE_LATENCY_TEST"], default="BASE_SPEED", help="Режим скорости: BASE_SPEED (по умолчанию), VARIABLE_SPEED или BASE_LATENCY_TEST")
    ap.add_argument("--session-base-speed", type=float, default=None, help="Базовая скорость сессии в градусах/сек (по умолчанию 278.0)")
    ap.add_argument(
        "--detector",
        choices=["baseline", "polar", "rays", "local", "hybrid", "color_index"],
        default=None,
        help="Архитектура детектора компьютерного зрения (по умолчанию: baseline)",
    )
    ap.add_argument(
        "--shadow-detector",
        choices=["baseline", "polar", "rays", "local", "hybrid", "color_index"],
        default=None,
        help="Архитектура теневого детектора для параллельного онлайн-аудита (например: hybrid)",
    )
    ap.add_argument(
        "--gen-rush", "--speed-perk",
        dest="gen_rush",
        action="store_true",
        help="Режим GEN RUSH / Perk ускорения стрелки: робастная оценка скорости, отслеживание фазы отклика игры, теневые профили",
    )
    ap.add_argument(
        "--detector-benchmark",
        action="store_true",
        help="Запустить офлайн бенчмарк детекторов на сырых видео и выйти",
    )
    args = ap.parse_args()

    if args.detector_benchmark:
        from tools.benchmark_detectors import main as run_detector_benchmark
        run_detector_benchmark()
        return

    reg = None
    if args.region:
        m = re.match(r"^(\d+)x(\d+)\+(\d+)\+(\d+)$", args.region)
        if m:
            w, h, x, y = map(int, m.groups())
            reg = {"left": x, "top": y, "width": w, "height": h}

    run_bot(
        dry_run=args.dry_run,
        latency_ms=args.latency,
        target_mode=args.target,
        region=reg,
        show_hud=args.hud,
        require_lmb=args.require_lmb,
        fps=args.fps,
        record_all=args.record_all,
        freeze=args.freeze,
        speed_mode=args.speed_mode,
        session_base_speed=args.session_base_speed,
        detector_name=args.detector,
        shadow_detector_name=args.shadow_detector,
        base_latency_test=args.base_latency_test,
        gen_rush=args.gen_rush,
    )


if __name__ == "__main__":
    main()
