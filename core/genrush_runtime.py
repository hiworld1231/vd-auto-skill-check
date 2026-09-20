from __future__ import annotations

import importlib.metadata
import platform
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

from core.capture import CaptureError, ScreenGrabber
from core.continuous_predictor import ContinuousAngularPredictor
from core.flight_recorder import FlightRecorder
from core.lead_level_controller import LeadLevelController
from core.mouse_tracker import MouseTracker
from core.outcome_observer import OutcomeObserver
from core.preflight import run_preflight
from core.post_fire_lifecycle import FRENZY, RING_END, PostFireLifecycle
from core.trigger import HardwareTrigger, PreciseTriggerScheduler
from core.tui import SkillCheckTUI
from core.vision import VisionEngine


def _session_metadata(root: Path, config: Dict[str, Any], *, fps: int, region: Dict[str, int], detector: str) -> Dict[str, Any]:
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=1.0,
        ).stdout.strip()
    except Exception:
        git_sha = "UNKNOWN"

    versions: Dict[str, str] = {
        "python": platform.python_version(),
        "opencv": str(getattr(cv2, "__version__", "UNKNOWN")),
    }
    for package in ("numpy", "rich", "mss", "evdev"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"

    return {
        "session_id": uuid.uuid4().hex,
        "build_git_sha": git_sha,
        "runtime_versions": versions,
        "effective_config": {
            **dict(config),
            "fps": int(fps),
            "region": dict(region),
            "detector": str(detector),
        },
    }


def _circ(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _valid_needle(det: Optional[Dict[str, Any]]) -> bool:
    return bool(
        isinstance(det, dict)
        and det.get("needle_angle") is not None
        and det.get("needle_valid", True) is not False
        and float(det.get("needle_strength", 0.0) or 0.0) >= 15.0
    )


def _sane_zone(z: Optional[Dict[str, Any]], lo: float, hi: float) -> Optional[Dict[str, Any]]:
    if not isinstance(z, dict):
        return None
    try:
        width = float(z.get("width", (float(z["end"]) - float(z["start"])) % 360.0))
        if not lo <= width <= hi:
            return None
        out = dict(z)
        out["width"] = width
        return out
    except Exception:
        return None


@dataclass(frozen=True)
class FireEvent:
    success: bool
    error: Optional[str]
    reason: str
    mode: str
    desired: Optional[float]
    deadline: Optional[float]
    callback_entry: float
    dispatch_start: float
    dispatch_done: float
    context: Dict[str, Any]


def run_genrush_clean(
    *,
    root: Path,
    config: Dict[str, Any],
    dry_run: bool = False,
    fps: Optional[int] = None,
    region: Optional[Dict[str, int]] = None,
    show_hud: Optional[bool] = None,
    require_lmb: Optional[bool] = None,
    record_all: bool = False,
    detector_name: Optional[str] = None,
    should_run=None,
) -> None:
    root = Path(root)
    should_run = should_run or (lambda: True)
    fps = int(fps if fps is not None else config.get("fps", 120))
    region = dict(region or config.get("region") or {"left": 800, "top": 420, "width": 320, "height": 240})
    show_hud = bool(config.get("show_hud", False) if show_hud is None else show_hud)
    require_lmb = bool(config.get("require_lmb", True) if require_lmb is None else require_lmb)
    detector_name = str(detector_name or config.get("detector", "hybrid"))
    seed_lead = float(config.get("genrush_seed_lead_ms", 60.0))
    base_speed = float(config.get("session_base_speed", 278.0))

    pf = run_preflight(dry_run=dry_run, require_lmb=require_lmb)
    if not pf.ok:
        raise RuntimeError("PRE-FLIGHT FAILED: " + pf.summary())

    tui = SkillCheckTUI()
    grabber = ScreenGrabber(
        region,
        fps=fps,
        framerate_mode="vfr",
        allow_mss_fallback=bool(config.get("allow_mss_fallback", False)),
    )
    vision = VisionEngine(backend_name=detector_name)
    predictor = ContinuousAngularPredictor(
        seed_lead, target_offset_ratio=0.5, session_base_speed=base_speed, fit_window=10
    )
    observer = OutcomeObserver(seed_lead, base_speed)
    post_fire = PostFireLifecycle()
    lead = LeadLevelController(seed_lead)
    mouse = MouseTracker(enabled=require_lmb)
    recorder = FlightRecorder(
        root / "replays",
        record_all=record_all,
        session_meta=_session_metadata(root, config, fps=fps, region=region, detector=detector_name),
    )
    keyboard = HardwareTrigger(
        dry_run=dry_run,
        hold_seconds=float(config.get("space_hold_ms", 35.0)) / 1000.0,
        allow_pynput_fallback=bool(config.get("allow_pynput_fallback", False)),
    )
    if not dry_run and keyboard.backend == "UNAVAILABLE":
        raise RuntimeError("No usable keyboard backend; refusing silent dry-run")

    if require_lmb:
        deadline = time.monotonic() + 0.7
        while mouse.backend_name == "DISABLED" and time.monotonic() < deadline:
            time.sleep(0.01)
        if mouse.backend_name == "DISABLED":
            raise RuntimeError("require_lmb=true but mouse backend did not initialize")

    fire_q: queue.SimpleQueue = queue.SimpleQueue()
    ctx_lock = threading.Lock()
    fire_ctx: Dict[str, Any] = {}

    def fire_callback(
        reason: str,
        *,
        desired_press_time=None,
        scheduler_dispatch_target=None,
        callback_entry_time=None,
    ) -> None:
        entry = float(callback_entry_time if callback_entry_time is not None else time.monotonic())
        with ctx_lock:
            ctx = dict(fire_ctx)
        result = keyboard.trigger()
        mode = "SCHEDULED" if reason.startswith("SCHEDULED") else "IMMEDIATE"
        fire_q.put(
            FireEvent(
                success=result.success,
                error=result.error,
                reason=reason,
                mode=mode,
                desired=desired_press_time,
                deadline=scheduler_dispatch_target,
                callback_entry=entry,
                dispatch_start=result.started_at,
                dispatch_done=result.finished_at,
                context=ctx,
            )
        )

    scheduler = PreciseTriggerScheduler(fire_callback)

    in_check = False
    pressed = False
    chain = 0
    check_start = 0.0
    check_lead = seed_lead
    locked_w = None
    locked_b = None
    speed_at_lock = None
    planned_press = None
    no_fire_reason = None
    last_id = None
    last_unique_ts = None
    last_post_angle = None
    last_fire: Optional[FireEvent] = None

    def reset_all() -> None:
        nonlocal in_check, pressed, chain, check_start, locked_w, locked_b, speed_at_lock
        nonlocal planned_press, no_fire_reason, last_fire, last_post_angle
        scheduler.cancel_pending()
        scheduler.rearm()
        vision.reset()
        predictor.reset(keep_speed=False, session_base_speed=base_speed)
        observer.reset()
        post_fire.reset()
        in_check = False
        pressed = False
        chain = 0
        check_start = 0.0
        locked_w = None
        locked_b = None
        speed_at_lock = None
        planned_press = None
        no_fire_reason = None
        last_fire = None
        last_post_angle = None

    def start_generation(now: float, new_chain: int, preserve_center: bool = False) -> None:
        nonlocal in_check, pressed, chain, check_start, check_lead, locked_w, locked_b
        nonlocal speed_at_lock, planned_press, no_fire_reason, last_fire, last_post_angle
        scheduler.cancel_pending()
        scheduler.rearm()
        if preserve_center:
            vision.reset_generation(preserve_center=True)
        else:
            vision.reset()
        predictor.reset(keep_speed=False, is_chain=new_chain > 1, session_base_speed=base_speed)
        observer.reset()
        post_fire.reset()
        in_check = True
        pressed = False
        chain = new_chain
        check_start = now
        check_lead = lead.get_lead_ms(chain_count=1, chain_offset_ms=0.0)
        predictor.latency_s = check_lead / 1000.0
        locked_w = None
        locked_b = None
        speed_at_lock = None
        planned_press = None
        no_fire_reason = None
        last_fire = None
        last_post_angle = None
        recorder.start_check(
            now, chain_count=chain, latency_ms=check_lead, target_mode="GREAT", target_ratio=0.5
        )
        tui.set_status(f"CHECK #{chain}")
        tui.log(f"▶ check chain={chain} lead={check_lead:.1f}ms")

    def finish(
        now: float, *, frenzy_transition: bool = False, reason: Optional[str] = None
    ) -> Dict[str, Any]:
        nonlocal in_check
        info = observer.conclude_check(
            frenzy_transition=frenzy_transition, no_fire_reason=reason
        )
        outcome = str(info.get("outcome", "UNCONFIRMED"))
        info["chain_count"] = chain
        info["frenzy_transition"] = bool(frenzy_transition)
        info["capture_health"] = grabber.get_diagnostics()
        if last_fire:
            ctx = last_fire.context
            sched_jitter = (
                (last_fire.callback_entry - last_fire.deadline) * 1000.0
                if last_fire.deadline is not None
                else None
            )
            info.update(
                {
                    "trigger_mode": last_fire.mode,
                    "requested_lead_ms": check_lead,
                    "actual_used_delay_ms": ctx.get("effective_dispatch_lead_ms", check_lead),
                    "effective_dispatch_lead_ms": ctx.get("effective_dispatch_lead_ms", check_lead),
                    # Legacy name kept for existing replay tooling.  This is
                    # decode-delivery age, not source/render age.
                    "frame_age_ms": ctx.get("frame_age_ms"),
                    "decode_delivery_age_ms": ctx.get("frame_age_ms"),
                    "speed_at_lock": ctx.get("speed_at_lock"),
                    "speed_at_fire": ctx.get("speed_at_fire"),
                    "fit_telemetry": ctx.get("fit", {}),
                    "scheduler_jitter_ms": sched_jitter,
                    "detector_fallback": ctx.get("detector_fallback", False),
                    "compensation_regime": "CONTINUOUS_MEASURED_SPEED",
                }
            )
            fit = ctx.get("fit", {}) or {}
            lr = lead.record_outcome(
                center_error_ms=info.get("center_error_ms"),
                actual_used_delay_ms=ctx.get("effective_dispatch_lead_ms", check_lead),
                outcome=outcome,
                plateau_found=info.get("plateau_found"),
                trigger_mode=last_fire.mode,
                scheduler_jitter_ms=sched_jitter,
                frame_age_ms=ctx.get("frame_age_ms"),
                detector_fallback=bool(ctx.get("detector_fallback", False)),
                compensation_regime="CONTINUOUS_MEASURED_SPEED",
                fit_sample_count=fit.get("fit_sample_count"),
                fit_residual_mad_deg=fit.get("fit_residual_mad_deg"),
                fit_spread_deg_s=fit.get("live_fit_spread"),
                speed_at_lock=ctx.get("speed_at_lock"),
                speed_at_fire=ctx.get("speed_at_fire"),
                chain_count=chain,
                white_source=info.get("white_source"),
                black_source=info.get("black_source"),
                frenzy_transition=frenzy_transition,
                short_vs_long_delta_deg_s=fit.get("short_vs_long_delta"),
                target_passed=bool(ctx.get("target_passed", False)),
                observed_response_ms=info.get("observed_response_ms"),
            )
            info["lead_level_update"] = lr
            info["lead_level_telemetry"] = lead.telemetry()
            if lr.get("accepted"):
                ideal = lr.get("ideal_lead_ms")
                if lr.get("updated"):
                    tui.log(
                        f"🧭 lead {lr['lead_before_ms']:.1f}→{lr['current_lead_ms']:.1f}ms "
                        f"({lr.get('update_reason')})"
                    )
                else:
                    tui.log(
                        f"🧪 LEARN ideal={float(ideal):.1f}ms "
                        f"n={int(lr.get('sample_count') or 0)}"
                    )
            elif not frenzy_transition and int(chain) == 1:
                extra = ""
                if lr.get("ideal_lead_ms") is not None:
                    extra = f" ideal={float(lr['ideal_lead_ms']):.1f}ms"
                tui.log(
                    f"🧪 LEARN reject={lr.get('reject_reason') or 'UNKNOWN'}{extra}"
                )
        recorder.end_check(now, info)
        tui.record_hit(
            outcome,
            info.get("hit_angle"),
            info.get("target_angle"),
            info.get("error_deg"),
            info.get("error_ms"),
            chain=chain,
            latency_ms=check_lead,
        )
        in_check = False
        return info

    try:
        tui.log(
            f"CLEAN V5 | VFR {fps} max FPS | detector={detector_name} | input={keyboard.backend}"
        )
        while should_run():
            try:
                frame, frame_id, frame_ts = grabber.grab(
                    wait_new=True, last_id=last_id, timeout=0.025
                )
            except CaptureError as exc:
                raise RuntimeError(f"capture failed: {exc}") from exc
            now = time.monotonic()
            if frame is None or frame_id is None or frame_ts is None:
                continue
            if frame_id == last_id:
                continue

            dt_frame = (
                frame_ts - last_unique_ts
                if last_unique_ts is not None
                else 1.0 / max(1, fps)
            )
            last_unique_ts = frame_ts
            last_id = frame_id
            frame_age_ms = max(0.0, (now - frame_ts) * 1000.0)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            while True:
                try:
                    ev: FireEvent = fire_q.get_nowait()
                except queue.Empty:
                    break
                if not ev.success:
                    raise RuntimeError(
                        f"keyboard dispatch failed via {keyboard.backend}: {ev.error}"
                    )
                if pressed:
                    continue
                pressed = True
                predictor.mark_fired()
                last_fire = ev
                planned_press = None
                scheduler.cancel_pending()
                vision.notify_pressed()
                c = ev.context
                # Physical predicted time from the actual key dispatch to the
                # target crossing.  This is valid for both scheduled and
                # overdue IMMEDIATE_SAFE fires.  The old code incorrectly used
                # time_to_target from the planning frame.
                if ev.desired is not None:
                    c["effective_dispatch_lead_ms"] = max(
                        0.0,
                        check_lead + (float(ev.desired) - float(ev.dispatch_start)) * 1000.0,
                    )
                else:
                    c["effective_dispatch_lead_ms"] = check_lead
                post_fire.begin(ev.dispatch_start)
                observer.on_trigger(
                    ev.dispatch_start,
                    float(c.get("target_angle") or 0.0),
                    float(c.get("speed_at_fire") or predictor.speed_deg_s),
                    locked_w,
                    locked_b,
                    used_latency_ms=check_lead,
                )
                recorder.on_trigger(
                    ev.dispatch_start,
                    ev.reason,
                    float(c.get("target_angle") or 0.0),
                    float(c.get("estimated_angle") or c.get("target_angle") or 0.0),
                    float(c.get("speed_at_fire") or 0.0),
                    check_lead,
                    trigger_mode=ev.mode,
                    measured_speed_at_lock=c.get("speed_at_lock"),
                    planned_press_time=ev.desired,
                    frame_age_ms=c.get("frame_age_ms"),
                    decode_delivery_age_ms=c.get("frame_age_ms"),
                    fit_telemetry=c.get("fit"),
                )
                tui.log(
                    f"💥 SPACE chain={chain} speed={float(c.get('speed_at_fire') or 0):.1f}°/s "
                    f"lead={check_lead:.1f}ms "
                    f"eff={float(c.get('effective_dispatch_lead_ms') if c.get('effective_dispatch_lead_ms') is not None else check_lead):.1f}ms"
                    + (
                        f" [{c.get('actuation_speed_reason')} raw={float(c.get('raw_fit_speed_at_fire') or 0):.1f}]"
                        if c.get("actuation_speed_reason") == "HIGH_SPEED_CONSERVATIVE_LOCAL"
                        else ""
                    )
                )

            if not in_check:
                if require_lmb and not mouse.is_held():
                    continue
                det = vision.detect_frame(
                    gray,
                    frame,
                    dt_frame=max(0.001, dt_frame),
                    expected_speed=base_speed,
                    is_pressed=False,
                )
                if not det or not det.get("ring_present") or not _valid_needle(det):
                    continue
                w, b = vision.extract_zones(det)
                w = _sane_zone(w, 5.0, 16.0)
                b = _sane_zone(b, 18.0, 65.0)
                if w is None:
                    continue
                in_check = True
                chain = 1
                check_start = now
                pressed = False
                check_lead = lead.get_lead_ms()
                predictor.reset(False, session_base_speed=base_speed)
                predictor.latency_s = check_lead / 1000.0
                observer.reset()
                scheduler.rearm()
                locked_w, locked_b = w, b
                speed_at_lock = None
                planned_press = None
                no_fire_reason = None
                recorder.start_check(
                    now,
                    chain_count=chain,
                    latency_ms=check_lead,
                    locked_w=w,
                    locked_b=b,
                )
                tui.log(f"▶ check lead={check_lead:.1f}ms")
            else:
                det = vision.detect_frame(
                    gray,
                    frame,
                    expected_angle=None,
                    search_window=35.0,
                    dt_frame=max(0.001, dt_frame),
                    expected_speed=predictor.speed_deg_s,
                    locked_zones=(locked_w, locked_b) if locked_w else None,
                    is_pressed=pressed,
                )

            if require_lmb and in_check and not pressed and not mouse.is_held():
                no_fire_reason = "LMB_RELEASED"
                finish(now, reason=no_fire_reason)
                reset_all()
                continue

            if pressed:
                ring_present = bool(det is not None and det.get("ring_present"))

                rollback = False
                zone_move = False
                if ring_present and _valid_needle(det):
                    observer.observe_sample(
                        frame_ts,
                        float(det["needle_angle"]),
                        float(det.get("needle_strength", 30.0)),
                    )
                    cur = float(det["needle_angle"])
                    rollback = (
                        last_post_angle is not None
                        and ((cur - last_post_angle + 180.0) % 360.0 - 180.0) < -25.0
                    )
                    last_post_angle = cur

                if ring_present:
                    nw, nb = vision.extract_zones(det)
                    nw = _sane_zone(nw, 5.0, 16.0)
                    nb = _sane_zone(nb, 18.0, 65.0)
                    zone_move = bool(
                        nw
                        and locked_w
                        and _circ(
                            float(nw.get("center", 0)), float(locked_w.get("center", 0))
                        )
                        > 15.0
                    )

                decision = post_fire.update(
                    now,
                    ring_present=ring_present,
                    plateau_found=observer.has_plateau(),
                    zone_moved=zone_move,
                    rollback=rollback,
                    fresh_motion=observer.has_recent_motion(),
                )

                recorder.on_frame(
                    frame_ts,
                    frame,
                    det,
                    None,
                    {
                        "frame_age_ms": frame_age_ms,
                        "decode_delivery_age_ms": frame_age_ms,
                        "post_fire_state": decision.state,
                        "post_fire_reason": decision.reason,
                        "post_fire_absence_ms": decision.absence_ms,
                        "post_fire_relocated_streak": decision.relocated_streak,
                    },
                )

                if decision.state == FRENZY:
                    tui.log(f"🔥 FRENZY confirm={decision.reason}")
                    finish(now, frenzy_transition=True)
                    start_generation(now, chain + 1, preserve_center=True)
                    continue
                if decision.state == RING_END:
                    finish(now)
                    reset_all()
                    continue
                continue

            if det is None or not det.get("ring_present"):
                if now - check_start > 0.10:
                    finish(now, reason=no_fire_reason or "RING_ENDED_BEFORE_FIRE")
                    reset_all()
                continue

            if frame_age_ms > 25.0:
                recorder.on_frame(
                    frame_ts,
                    frame,
                    det,
                    None,
                    {"frame_age_ms": frame_age_ms, "decode_delivery_age_ms": frame_age_ms, "stale": True},
                )
                continue

            w, b = vision.extract_zones(
                det, (locked_w, locked_b) if locked_w else None
            )
            w = _sane_zone(w, 5.0, 16.0)
            b = _sane_zone(b, 18.0, 65.0)
            if w is not None and locked_w is None:
                locked_w, locked_b = w, b
            if not _valid_needle(det) or locked_w is None:
                continue

            angle = float(det["needle_angle"])
            strength = float(det.get("needle_strength", 0.0))
            predictor.update(frame_ts, angle, strength, locked_w, locked_b)
            pred = (
                predictor.predict(frame_ts, angle, "GREAT")
                if predictor.has_usable_speed()
                else None
            )
            recorder.on_frame(
                frame_ts, frame, det, pred, {"frame_age_ms": frame_age_ms, "decode_delivery_age_ms": frame_age_ms}
            )
            if pred is None:
                continue

            stable = predictor.has_stable_speed()
            urgent = predictor.has_usable_speed() and float(
                pred.get("time_until_press_ms", 9999.0)
            ) <= 35.0
            if not (stable or urgent):
                continue

            if speed_at_lock is None:
                speed_at_lock = float(predictor.speed_deg_s)
            fit = predictor.get_shadow_telemetry()
            press_t = float(pred["press_timestamp"])
            detector_fallback = "FALLBACK" in str(det.get("detector_name", ""))
            with ctx_lock:
                fire_ctx.clear()
                fire_ctx.update(
                    {
                        "target_angle": pred.get("target_angle"),
                        "estimated_angle": angle,
                        "speed_at_lock": speed_at_lock,
                        "speed_at_fire": float(pred.get("speed_deg_s") or predictor.speed_deg_s),
                        "raw_fit_speed_at_fire": float(predictor.speed_deg_s),
                        "actuation_speed_reason": pred.get("actuation_speed_reason"),
                        "frame_age_ms": frame_age_ms,
                        "fit": fit,
                        "detector_fallback": detector_fallback,
                        # Planning-frame value only.  The actual effective
                        # dispatch lead is computed from FireEvent.desired and
                        # the real dispatch timestamp when Space is sent.
                        "planned_time_to_target_ms": max(
                            0.0, float(pred.get("time_to_hit_ms", check_lead))
                        ),
                        "target_passed": bool(pred.get("target_passed", False)),
                    }
                )

            if pred.get("target_passed") and not pred.get("should_press_now"):
                no_fire_reason = "TOO_LATE_UNSAFE"
                scheduler.cancel_pending()
                continue

            if pred.get("should_press_now"):
                scheduler.trigger_now("IMMEDIATE_SAFE", desired_press_time=press_t)
                predictor.mark_committed()
            elif press_t > now:
                if planned_press is None or abs(press_t - planned_press) >= 0.0005:
                    planned_press = press_t
                    scheduler.schedule(
                        press_t,
                        reason="SCHEDULED_CONTINUOUS",
                        desired_press_time=press_t,
                    )
                    predictor.mark_committed()

            if now - check_start > 3.5:
                finish(now, reason=no_fire_reason or "CHECK_TIMEOUT")
                reset_all()

            if show_hud:
                hud = frame.copy()
                cv2.putText(
                    hud,
                    f"speed={predictor.speed_deg_s:.0f} lead={check_lead:.1f} chain={chain}",
                    (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(
                    "VD CLEAN V5",
                    cv2.resize(hud, (640, 480), interpolation=cv2.INTER_NEAREST),
                )
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        try:
            scheduler.close()
        except Exception:
            pass
        try:
            recorder.close()
        except Exception:
            pass
        try:
            mouse.close()
        except Exception:
            pass
        try:
            keyboard.close()
        except Exception:
            pass
        try:
            grabber.close()
        except Exception:
            pass
        try:
            if show_hud:
                cv2.destroyAllWindows()
        except Exception:
            pass
        tui.finish()
