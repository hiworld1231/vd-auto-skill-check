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
from core.dispatch_timing import DispatchTimingModel
from core.fire_policy import decide_great_fire
from core.flight_recorder import FlightRecorder
from core.lead_calibration import LeadCalibrationStore, make_calibration_fingerprint
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


def _is_measured_zone(z: Optional[Dict[str, Any]]) -> bool:
    return bool(
        isinstance(z, dict)
        and str(z.get("source", "")).startswith("MEASURED")
    )


def _presence_absence_update(
    absent_since: Optional[float],
    *,
    now: float,
    present: bool,
) -> tuple[Optional[float], float]:
    """Track continuous pre-fire ring absence.

    A check must never be ended because one late frame was missed after the
    check has existed for a while.  Only continuous absence counts.
    """
    if present:
        return None, 0.0
    if absent_since is None:
        absent_since = float(now)
    return absent_since, max(0.0, float(now) - float(absent_since))


def _generation_lead(
    chain_count: int,
    *,
    normal_lead_ms: float,
    normal_uncertainty_ms: float,
    frenzy_lead_ms: float,
    frenzy_uncertainty_ms: float,
) -> tuple[float, float]:
    if int(chain_count) > 1:
        return float(frenzy_lead_ms), max(0.0, float(frenzy_uncertainty_ms))
    return float(normal_lead_ms), max(0.0, float(normal_uncertainty_ms))


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
    frenzy_lead = float(config.get("frenzy_lead_ms", 80.0))
    frenzy_lead_uncertainty = float(config.get("frenzy_lead_uncertainty_ms", 0.0))
    prefire_ring_end_absence_s = max(
        0.050,
        float(config.get("prefire_ring_end_absence_ms", 100.0)) / 1000.0,
    )
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
    dispatch_timing = DispatchTimingModel()
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

    calibration_store = None
    calibration_load = None
    if bool(config.get("persist_lead_calibration", True)) and not dry_run:
        capture_backend = "GSR_VFR" if grabber.use_gsr else "MSS"
        fingerprint = make_calibration_fingerprint(
            fps=fps,
            region=region,
            detector=detector_name,
            capture_backend=capture_backend,
            input_backend=keyboard.backend,
        )
        calibration_store = LeadCalibrationStore(
            root / ".vd_lead_calibration.json",
            fingerprint,
            max_age_s=float(config.get("lead_calibration_max_age_days", 7.0))
            * 86400.0,
        )
        calibration_load = calibration_store.load()
        recorder.session_meta["lead_calibration_load"] = calibration_load.telemetry()
        if calibration_load.accepted:
            lead.restore_calibration(
                float(calibration_load.lead_ms),
                float(calibration_load.uncertainty_ms),
            )
            tui.log(
                f"🧭 restored lead={lead.current_lead_ms:.1f}ms "
                f"±{lead.get_uncertainty_ms():.1f}ms "
                f"age={float(calibration_load.age_s or 0.0) / 3600.0:.1f}h"
            )
        elif calibration_load.reason != "NOT_FOUND":
            tui.log(f"🧭 calibration ignored: {calibration_load.reason}")

    if require_lmb:
        deadline = time.monotonic() + 0.7
        while mouse.backend_name == "DISABLED" and time.monotonic() < deadline:
            time.sleep(0.01)
        if mouse.backend_name == "DISABLED":
            raise RuntimeError("require_lmb=true but mouse backend did not initialize")

    fire_q: queue.SimpleQueue = queue.SimpleQueue()
    ctx_lock = threading.Lock()
    fire_ctx: Dict[str, Any] = {}
    fire_epoch = 0

    def advance_fire_epoch() -> int:
        nonlocal fire_epoch
        with ctx_lock:
            fire_epoch += 1
            fire_ctx.clear()
            fire_ctx["fire_epoch"] = fire_epoch
            return fire_epoch

    def fire_callback(
        reason: str,
        *,
        desired_press_time=None,
        scheduler_dispatch_target=None,
        callback_entry_time=None,
        scheduler_token=None,
    ) -> None:
        entry = float(callback_entry_time if callback_entry_time is not None else time.monotonic())
        # The token check and physical keydown are intentionally atomic with
        # respect to reset/start_generation.  An old worker callback that has
        # already left the scheduler lock must not press Space in a new chain.
        with ctx_lock:
            if scheduler_token != fire_epoch:
                return
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
    pre_fire_absent_since: Optional[float] = None
    last_fire: Optional[FireEvent] = None

    def reset_all() -> None:
        nonlocal in_check, pressed, chain, check_start, locked_w, locked_b, speed_at_lock
        nonlocal planned_press, no_fire_reason, last_fire, last_post_angle
        nonlocal pre_fire_absent_since
        advance_fire_epoch()
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
        pre_fire_absent_since = None

    def start_generation(
        now: float,
        new_chain: int,
        preserve_center: bool = False,
        *,
        bootstrap_det: Optional[Dict[str, Any]] = None,
        bootstrap_w: Optional[Dict[str, Any]] = None,
        bootstrap_b: Optional[Dict[str, Any]] = None,
        bootstrap_t: Optional[float] = None,
    ) -> None:
        nonlocal in_check, pressed, chain, check_start, check_lead, locked_w, locked_b
        nonlocal speed_at_lock, planned_press, no_fire_reason, last_fire, last_post_angle
        nonlocal pre_fire_absent_since
        previous_fire = last_fire
        advance_fire_epoch()
        scheduler.cancel_pending()
        scheduler.rearm()
        prior_generation_speed = float(
            (
                (previous_fire.context or {}).get("speed_at_fire")
                if previous_fire is not None
                else None
            )
            or predictor.get_actuation_speed()
        )
        if (
            new_chain > 1
            and bootstrap_det is not None
            and bootstrap_w is not None
        ):
            handoff_det = vision.bootstrap_generation(
                bootstrap_det,
                bootstrap_w,
                bootstrap_b,
            )
        else:
            handoff_det = None
            if preserve_center:
                vision.reset_generation(preserve_center=True)
            else:
                vision.reset()
        predictor.reset(
            keep_speed=(new_chain > 1),
            default_speed=(prior_generation_speed if new_chain > 1 else None),
            is_chain=new_chain > 1,
            session_base_speed=base_speed,
        )
        observer.reset()
        post_fire.reset()
        in_check = True
        pressed = False
        chain = new_chain
        check_start = now
        check_lead, generation_lead_uncertainty = _generation_lead(
            new_chain,
            normal_lead_ms=lead.get_lead_ms(
                chain_count=1, chain_offset_ms=0.0
            ),
            normal_uncertainty_ms=(
                lead.get_uncertainty_ms() if lead.initialized else 0.0
            ),
            frenzy_lead_ms=frenzy_lead,
            frenzy_uncertainty_ms=frenzy_lead_uncertainty,
        )
        predictor.set_delivery_lead(
            check_lead,
            generation_lead_uncertainty,
            dispatch_timing.uncertainty_ms(),
        )
        if handoff_det is not None:
            locked_w = dict(bootstrap_w)
            locked_b = dict(bootstrap_b) if bootstrap_b is not None else None
            if _valid_needle(handoff_det):
                predictor.update(
                    float(bootstrap_t if bootstrap_t is not None else now),
                    float(handoff_det["needle_angle"]),
                    float(handoff_det.get("needle_strength", 30.0)),
                    locked_w,
                    locked_b,
                )
        else:
            locked_w = None
            locked_b = None
        speed_at_lock = None
        planned_press = None
        no_fire_reason = None
        last_fire = None
        last_post_angle = None
        pre_fire_absent_since = None
        recorder.start_check(
            now,
            chain_count=chain,
            latency_ms=check_lead,
            lead_uncertainty_ms=generation_lead_uncertainty,
            dispatch_uncertainty_ms=dispatch_timing.uncertainty_ms(),
            delivery_uncertainty_ms=(
                generation_lead_uncertainty
                + dispatch_timing.uncertainty_ms()
            ),
            target_mode="GREAT",
            target_ratio=0.5,
            locked_w=locked_w,
            locked_b=locked_b,
        )
        if handoff_det is not None:
            handoff_frame_t = float(
                bootstrap_t if bootstrap_t is not None else now
            )
            handoff_angle = float(handoff_det.get("needle_angle") or 0.0)
            handoff_pred = (
                predictor.predict(handoff_frame_t, handoff_angle, "GREAT")
                if _valid_needle(handoff_det)
                else None
            )
            recorder.on_frame(
                handoff_frame_t,
                frame,
                handoff_det,
                handoff_pred,
                {
                    "frame_age_ms": frame_age_ms,
                    "decode_delivery_age_ms": frame_age_ms,
                    "frenzy_generation_handoff": True,
                },
            )

            # The frame that confirms a Frenzy relocation is already the first
            # frame of the next generation. On a late handoff it can be the only
            # frame where the trailing GOOD sector is still reachable. Evaluate
            # that exact frame instead of always discarding it.
            if handoff_pred is not None and new_chain > 1:
                handoff_policy = decide_great_fire(
                    handoff_pred,
                    fit_stable=predictor.has_stable_speed(),
                    speed_usable=predictor.has_usable_speed(),
                )
                if (
                    handoff_policy.allow
                    and handoff_policy.reason
                    == "FRENZY_HANDOFF_REACTIVE_SUCCESS"
                ):
                    fit = predictor.get_shadow_telemetry()
                    with ctx_lock:
                        fire_ctx.clear()
                        fire_ctx.update(
                            {
                                "fire_epoch": fire_epoch,
                                "chain_at_plan": chain,
                                "target_angle": handoff_pred.get("target_angle"),
                                "estimated_angle": handoff_angle,
                                "speed_at_lock": None,
                                "speed_at_fire": float(
                                    handoff_pred.get("speed_deg_s")
                                    or predictor.get_actuation_speed()
                                ),
                                "raw_fit_speed_at_fire": float(
                                    predictor.speed_deg_s
                                ),
                                "actuation_speed_reason": handoff_pred.get(
                                    "actuation_speed_reason"
                                ),
                                "speed_source": handoff_pred.get("speed_source"),
                                "fire_policy_reason": handoff_policy.reason,
                                "fire_policy_best_effort": True,
                                "great_interval_safe": bool(
                                    handoff_pred.get("great_interval_safe", False)
                                ),
                                "great_interval_intersects": bool(
                                    handoff_pred.get(
                                        "great_interval_intersects", False
                                    )
                                ),
                                "white_source": handoff_pred.get("white_source"),
                                "great_geometry_refined": bool(
                                    handoff_pred.get(
                                        "great_geometry_refined", False
                                    )
                                ),
                                "great_boundary_method": handoff_pred.get(
                                    "great_boundary_method"
                                ),
                                "landing_uncertainty_width_deg": handoff_pred.get(
                                    "landing_uncertainty_width_deg"
                                ),
                                "crossing_uncertainty_ms": handoff_pred.get(
                                    "crossing_uncertainty_ms"
                                ),
                                "lead_uncertainty_ms": handoff_pred.get(
                                    "lead_uncertainty_ms"
                                ),
                                "dispatch_uncertainty_ms": handoff_pred.get(
                                    "dispatch_uncertainty_ms"
                                ),
                                "delivery_uncertainty_ms": handoff_pred.get(
                                    "delivery_uncertainty_ms"
                                ),
                                "dispatch_lag_compensation_ms": (
                                    dispatch_timing.compensation_ms()
                                ),
                                "dispatch_lag_uncertainty_ms": (
                                    dispatch_timing.uncertainty_ms()
                                ),
                                "dispatch_timing": dispatch_timing.telemetry(),
                                "frame_age_ms": frame_age_ms,
                                "fit": fit,
                                "detector_fallback": False,
                                "planned_time_to_target_ms": 0.0,
                                "target_passed": True,
                                "frenzy_handoff_reactive": True,
                            }
                        )
                    scheduler.trigger_now(
                        "IMMEDIATE_FRENZY_HANDOFF_REACTIVE",
                        desired_press_time=float(
                            handoff_pred.get(
                                "press_timestamp", handoff_frame_t
                            )
                        ),
                        dispatch_token=fire_epoch,
                    )
                    predictor.mark_committed()
        tui.set_status(f"CHECK #{chain}")
        tui.log(
            f"▶ check chain={chain} lead={check_lead:.1f}ms"
            + (
                f" prior_speed={prior_generation_speed:.1f}°/s"
                if chain > 1
                else ""
            )
            + (" [HANDOFF]" if handoff_det is not None else "")
        )

    def finish(
        now: float, *, frenzy_transition: bool = False, reason: Optional[str] = None
    ) -> Dict[str, Any]:
        nonlocal in_check
        # Invalidate any pending callback before outcome/recorder work.  This
        # closes the race where a deadline left the scheduler worker just before
        # cancel_pending() and would otherwise press Space after check end.
        advance_fire_epoch()
        scheduler.cancel_pending()
        info = observer.conclude_check(
            frenzy_transition=frenzy_transition, no_fire_reason=reason
        )
        outcome = str(info.get("outcome", "UNCONFIRMED"))
        info["chain_count"] = chain
        info["frenzy_transition"] = bool(frenzy_transition)
        info["capture_health"] = grabber.get_diagnostics()
        if last_fire:
            ctx = last_fire.context
            callback_jitter = (
                (last_fire.callback_entry - last_fire.deadline) * 1000.0
                if last_fire.deadline is not None
                else None
            )
            raw_dispatch_lag_ms = (
                (last_fire.dispatch_done - last_fire.deadline) * 1000.0
                if last_fire.deadline is not None
                else None
            )
            # Residual physical keydown error relative to the intended press
            # time. Once dispatch-lag compensation is active this, not the raw
            # deadline->SYN lag, is the scheduler error relevant to landing.
            sched_jitter = (
                (last_fire.dispatch_done - last_fire.desired) * 1000.0
                if (
                    last_fire.mode == "SCHEDULED"
                    and last_fire.desired is not None
                )
                else raw_dispatch_lag_ms
                if last_fire.mode == "SCHEDULED"
                else None
            )
            immediate_lateness_ms = (
                max(
                    0.0,
                    (last_fire.dispatch_done - last_fire.desired) * 1000.0,
                )
                if (
                    last_fire.mode == "IMMEDIATE"
                    and last_fire.desired is not None
                )
                else None
            )
            input_dispatch_ms = (
                last_fire.dispatch_done - last_fire.dispatch_start
            ) * 1000.0
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
                    "fire_policy_reason": ctx.get("fire_policy_reason"),
                    "fire_policy_best_effort": bool(ctx.get("fire_policy_best_effort", False)),
                    "great_interval_safe": bool(ctx.get("great_interval_safe", False)),
                    "great_interval_intersects": bool(ctx.get("great_interval_intersects", False)),
                    "white_source_at_fire": ctx.get("white_source"),
                    "great_geometry_refined": bool(ctx.get("great_geometry_refined", False)),
                    "great_boundary_method": ctx.get("great_boundary_method"),
                    "landing_uncertainty_width_deg": ctx.get("landing_uncertainty_width_deg"),
                    "crossing_uncertainty_ms": ctx.get("crossing_uncertainty_ms"),
                    "lead_uncertainty_ms": ctx.get("lead_uncertainty_ms"),
                    "dispatch_uncertainty_ms": ctx.get("dispatch_uncertainty_ms"),
                    "delivery_uncertainty_ms": ctx.get("delivery_uncertainty_ms"),
                    "scheduler_jitter_ms": sched_jitter,
                    "immediate_lateness_ms": immediate_lateness_ms,
                    "dispatch_lag_ms": raw_dispatch_lag_ms,
                    "dispatch_lag_compensation_ms": ctx.get("dispatch_lag_compensation_ms"),
                    "dispatch_lag_uncertainty_ms": ctx.get("dispatch_lag_uncertainty_ms"),
                    "dispatch_timing": ctx.get("dispatch_timing"),
                    "scheduler_callback_jitter_ms": callback_jitter,
                    "input_dispatch_ms": input_dispatch_ms,
                    "keydown_syn_time": last_fire.dispatch_done,
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
            if (
                calibration_store is not None
                and lr.get("accepted")
                and lead.initialized
            ):
                try:
                    calibration_store.save(
                        lead_ms=lead.current_lead_ms,
                        uncertainty_ms=lead.get_uncertainty_ms(),
                        trusted_sample_count=lead.accepted_total,
                    )
                    info["lead_calibration_saved"] = True
                except Exception as exc:
                    info["lead_calibration_saved"] = False
                    info["lead_calibration_save_error"] = str(exc)
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
            reason=(
                info.get("no_fire_reason")
                or info.get("post_fire_reason")
            ),
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
                if int(ev.context.get("fire_epoch", -1)) != fire_epoch or not in_check:
                    # A physical keydown can only be queued for the generation
                    # whose token survived the atomic callback guard.
                    continue
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
                # The physical keydown becomes visible to the Linux input
                # subsystem after UInput.syn()/press() completes.  Use that
                # timestamp, not the pre-write function-entry timestamp.
                physical_press_t = float(ev.dispatch_done)
                observed_dispatch_lag_ms = None
                if ev.deadline is not None and ev.mode == "SCHEDULED":
                    observed_dispatch_lag_ms = dispatch_timing.record(
                        float(ev.deadline), physical_press_t
                    )
                if ev.desired is not None:
                    c["effective_dispatch_lead_ms"] = max(
                        0.0,
                        check_lead + (float(ev.desired) - physical_press_t) * 1000.0,
                    )
                else:
                    c["effective_dispatch_lead_ms"] = check_lead
                c["keydown_syn_time"] = physical_press_t
                c["input_dispatch_ms"] = (
                    float(ev.dispatch_done) - float(ev.dispatch_start)
                ) * 1000.0
                c["dispatch_lag_observed_ms"] = observed_dispatch_lag_ms
                c["dispatch_timing"] = dispatch_timing.telemetry()
                post_fire.begin(physical_press_t)
                observer.on_trigger(
                    physical_press_t,
                    float(c.get("target_angle") or 0.0),
                    float(c.get("speed_at_fire") or predictor.speed_deg_s),
                    locked_w,
                    locked_b,
                    used_latency_ms=check_lead,
                )
                recorder.on_trigger(
                    physical_press_t,
                    ev.reason,
                    float(c.get("target_angle") or 0.0),
                    float(c.get("estimated_angle") or c.get("target_angle") or 0.0),
                    float(c.get("speed_at_fire") or 0.0),
                    check_lead,
                    trigger_mode=ev.mode,
                    measured_speed_at_lock=c.get("speed_at_lock"),
                    planned_press_time=ev.desired,
                    scheduler_deadline=ev.deadline,
                    callback_entry_time=ev.callback_entry,
                    keydown_begin_time=ev.dispatch_start,
                    keydown_syn_time=physical_press_t,
                    input_dispatch_ms=c.get("input_dispatch_ms"),
                    frame_age_ms=c.get("frame_age_ms"),
                    decode_delivery_age_ms=c.get("frame_age_ms"),
                    fit_telemetry=c.get("fit"),
                    fire_policy_reason=c.get("fire_policy_reason"),
                    fire_policy_best_effort=c.get("fire_policy_best_effort"),
                    great_interval_safe=c.get("great_interval_safe"),
                    great_interval_intersects=c.get("great_interval_intersects"),
                    white_source=c.get("white_source"),
                    great_geometry_refined=c.get("great_geometry_refined"),
                    great_boundary_method=c.get("great_boundary_method"),
                    landing_uncertainty_width_deg=c.get("landing_uncertainty_width_deg"),
                    crossing_uncertainty_ms=c.get("crossing_uncertainty_ms"),
                    lead_uncertainty_ms=c.get("lead_uncertainty_ms"),
                    dispatch_uncertainty_ms=c.get("dispatch_uncertainty_ms"),
                    delivery_uncertainty_ms=c.get("delivery_uncertainty_ms"),
                    dispatch_lag_compensation_ms=c.get("dispatch_lag_compensation_ms"),
                    dispatch_lag_uncertainty_ms=c.get("dispatch_lag_uncertainty_ms"),
                    dispatch_lag_observed_ms=c.get("dispatch_lag_observed_ms"),
                    dispatch_timing=c.get("dispatch_timing"),
                )
                tui.log(
                    f"💥 SPACE chain={chain} speed={float(c.get('speed_at_fire') or 0):.1f}°/s "
                    f"lead={check_lead:.1f}ms "
                    f"eff={float(c.get('effective_dispatch_lead_ms') if c.get('effective_dispatch_lead_ms') is not None else check_lead):.1f}ms "
                    f"[{c.get('fire_policy_reason') or c.get('actuation_speed_reason') or 'GREAT'}]"
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
                advance_fire_epoch()
                in_check = True
                chain = 1
                check_start = now
                pressed = False
                check_lead = lead.get_lead_ms()
                predictor.reset(False, session_base_speed=base_speed)
                predictor.set_delivery_lead(
                    check_lead,
                    lead.get_uncertainty_ms() if lead.initialized else 0.0,
                    dispatch_timing.uncertainty_ms(),
                )
                observer.reset()
                scheduler.rearm()
                locked_w, locked_b = w, b
                speed_at_lock = None
                planned_press = None
                no_fire_reason = None
                pre_fire_absent_since = None
                recorder.start_check(
                    now,
                    chain_count=chain,
                    latency_ms=check_lead,
                    lead_uncertainty_ms=(lead.get_uncertainty_ms() if lead.initialized else 0.0),
                    dispatch_uncertainty_ms=dispatch_timing.uncertainty_ms(),
                    delivery_uncertainty_ms=(
                        (lead.get_uncertainty_ms() if lead.initialized else 0.0)
                        + dispatch_timing.uncertainty_ms()
                    ),
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
                # Landing motion can remain trackable after the SPACE prompt
                # disappears.  Do not couple outcome observation to BASELINE's
                # lifecycle presence bit.
                if _valid_needle(det):
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
                    next_chain = chain + 1
                    bootstrap_det = None
                    bootstrap_w = None
                    bootstrap_b = None
                    if ring_present:
                        bootstrap_w = _sane_zone(nw, 5.0, 16.0)
                        bootstrap_b = _sane_zone(nb, 18.0, 65.0)
                        # The post-fire HYBRID needle belongs to the previous
                        # generation. Handoff is safe only if BASELINE measured
                        # a needle on the relocated ring itself.
                        if (
                            bootstrap_w is not None
                            and isinstance(det, dict)
                            and bool(det.get("generation_needle_valid"))
                            and det.get("generation_needle_angle") is not None
                        ):
                            bootstrap_det = dict(det)
                    finish(now, frenzy_transition=True)
                    start_generation(
                        now,
                        next_chain,
                        preserve_center=True,
                        bootstrap_det=bootstrap_det,
                        bootstrap_w=bootstrap_w,
                        bootstrap_b=bootstrap_b,
                        bootstrap_t=frame_ts,
                    )
                    continue
                if decision.state == RING_END:
                    finish(now)
                    reset_all()
                    continue
                continue

            ring_present_now = bool(det is not None and det.get("ring_present"))
            pre_fire_absent_since, pre_fire_absent_s = _presence_absence_update(
                pre_fire_absent_since,
                now=now,
                present=ring_present_now,
            )
            if not ring_present_now:
                recorder.on_frame(
                    frame_ts,
                    frame,
                    det,
                    None,
                    {
                        "frame_age_ms": frame_age_ms,
                        "decode_delivery_age_ms": frame_age_ms,
                        "prefire_ring_absence_ms": pre_fire_absent_s * 1000.0,
                        "prefire_presence_grace": True,
                    },
                )
                if pre_fire_absent_s >= prefire_ring_end_absence_s:
                    no_fire_reason = "RING_ENDED_BEFORE_FIRE"
                    finish(now, reason=no_fire_reason)
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

            det_w = _sane_zone(
                det.get("white_zone") if isinstance(det, dict) else None,
                5.0,
                16.0,
            )
            det_b = _sane_zone(
                det.get("black_zone") if isinstance(det, dict) else None,
                18.0,
                65.0,
            )
            if (
                det_w is not None
                and _is_measured_zone(det_w)
                and not _is_measured_zone(locked_w)
            ):
                locked_w = det_w
                if det_b is not None:
                    locked_b = det_b

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
            # Predict immediately from the chain-1 session prior, then refine
            # the same pending deadline from segment/robust measured speed.
            pred = predictor.predict(frame_ts, angle, "GREAT")
            recorder.on_frame(
                frame_ts, frame, det, pred, {"frame_age_ms": frame_age_ms, "decode_delivery_age_ms": frame_age_ms}
            )
            if pred is None:
                continue

            usable = predictor.has_usable_speed()
            stable = predictor.has_stable_speed()
            fire_policy = decide_great_fire(
                pred,
                fit_stable=stable,
                speed_usable=usable,
            )

            if (
                pred.get("target_passed")
                and not pred.get("should_press_now")
                and not fire_policy.allow
            ):
                no_fire_reason = fire_policy.reason or "TOO_LATE_UNSAFE"
                scheduler.cancel_pending()
                planned_press = None
                predictor.mark_tracking()
                continue

            if not fire_policy.allow:
                # A deadline armed on an older fit is no longer trustworthy if
                # fresh evidence cannot keep the landing envelope inside GREAT.
                # Cancel first; a later clean frame may safely arm it again.
                if planned_press is not None:
                    scheduler.cancel_pending()
                    planned_press = None
                    predictor.mark_tracking()
                if float(pred.get("time_until_press_ms", 9999.0)) <= 0.0:
                    no_fire_reason = fire_policy.reason
                continue

            if speed_at_lock is None and usable:
                speed_at_lock = float(predictor.speed_deg_s)
            fit = predictor.get_shadow_telemetry()
            press_t = float(pred["press_timestamp"])
            detector_fallback = "FALLBACK" in str(det.get("detector_name", ""))
            with ctx_lock:
                fire_ctx.clear()
                fire_ctx.update(
                    {
                        "fire_epoch": fire_epoch,
                        "chain_at_plan": chain,
                        "target_angle": pred.get("target_angle"),
                        "estimated_angle": angle,
                        "speed_at_lock": speed_at_lock,
                        "speed_at_fire": float(pred.get("speed_deg_s") or predictor.speed_deg_s),
                        "raw_fit_speed_at_fire": float(predictor.speed_deg_s),
                        "actuation_speed_reason": pred.get("actuation_speed_reason"),
                        "speed_source": pred.get("speed_source"),
                        "fire_policy_reason": fire_policy.reason,
                        "fire_policy_best_effort": fire_policy.best_effort,
                        "great_interval_safe": bool(pred.get("great_interval_safe", False)),
                        "great_interval_intersects": bool(pred.get("great_interval_intersects", False)),
                        "white_source": pred.get("white_source"),
                        "great_geometry_refined": bool(pred.get("great_geometry_refined", False)),
                        "great_boundary_method": pred.get("great_boundary_method"),
                        "landing_uncertainty_width_deg": pred.get("landing_uncertainty_width_deg"),
                        "crossing_uncertainty_ms": pred.get("crossing_uncertainty_ms"),
                        "lead_uncertainty_ms": pred.get("lead_uncertainty_ms"),
                        "dispatch_uncertainty_ms": pred.get("dispatch_uncertainty_ms"),
                        "delivery_uncertainty_ms": pred.get("delivery_uncertainty_ms"),
                        "dispatch_lag_compensation_ms": dispatch_timing.compensation_ms(),
                        "dispatch_lag_uncertainty_ms": dispatch_timing.uncertainty_ms(),
                        "dispatch_timing": dispatch_timing.telemetry(),
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

            speed_source = str(pred.get("speed_source") or "MEASURED")
            provisional = speed_source in {"SESSION_PRIOR", "SEGMENT_PROVISIONAL"}
            if pred.get("should_press_now"):
                scheduler.trigger_now(
                    "IMMEDIATE_PREARM" if provisional else "IMMEDIATE_GREAT",
                    desired_press_time=press_t,
                    dispatch_token=fire_epoch,
                )
                predictor.mark_committed()
            elif press_t > now:
                if planned_press is None or abs(press_t - planned_press) >= 0.0005:
                    planned_press = press_t
                    dispatch_deadline = dispatch_timing.deadline_for_physical_press(
                        press_t
                    )
                    if dispatch_deadline <= now:
                        scheduler.trigger_now(
                            "IMMEDIATE_DISPATCH_COMPENSATED",
                            desired_press_time=press_t,
                            dispatch_token=fire_epoch,
                        )
                    else:
                        scheduler.schedule(
                            dispatch_deadline,
                            reason=(
                                "SCHEDULED_PREARM"
                                if provisional
                                else "SCHEDULED_GREAT"
                            ),
                            desired_press_time=press_t,
                            dispatch_token=fire_epoch,
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
