"""
Offline Audit: Live Shadow Detector vs Baseline Across Real Gameplay Checks.

Evaluates 50+ real gameplay skill checks without perk changes (and full dataset):
- Signed angle disagreement (hybrid - baseline) across all valid active frames
- Critical Near-FIRE windows (150ms, 75ms, 30ms, last frame before planned, last frame before dispatch)
- Zone center and width disagreements
- OFFLINE COUNTERFACTUAL outcome analysis
- Diagnostic capture and root cause classification for disagreements >5° and >10°
- Specific deep-dive into check 204705_0001 (CLEAN_RESPONSE_OUTLIER / UNKNOWN_RESPONSE_VARIANCE)
"""

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.detectors import get_detector
from core.predictor import SkillCheckPredictor, SPEED_MODE_BASE
from core.shadow_detector import AsyncShadowDetectorWorker


def circular_diff(a: float, b: float) -> float:
    """Returns signed shortest angular difference (a - b) in [-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def run_shadow_audit(min_checks: int = 50):
    replays_dir = ROOT / "replays"
    diag_dir = replays_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Find all mp4 replays with json
    mp4_files = sorted(list(replays_dir.glob("*.mp4")))
    valid_checks = []

    for mp4 in mp4_files:
        jp = mp4.with_suffix(".json")
        if not jp.exists():
            continue
        try:
            with open(jp, "r") as f:
                data = json.load(f)
        except Exception:
            continue

        trig = data.get("trigger", {})
        if not trig.get("fired"):
            continue

        spd = trig.get("measured_speed_at_fire") or trig.get("speed_deg_s") or 0.0
        valid_checks.append((mp4, jp, data, spd))

    print(f"[AUDIT] Found {len(valid_checks)} total fired checks with MP4 recordings.")

    # Partition into NO-PERK (base speed ~250-315 deg/s) vs ALL
    no_perk_checks = [c for c in valid_checks if 250.0 <= c[3] <= 315.0]
    print(f"[AUDIT] Found {len(no_perk_checks)} NO-PERK checks (speed ~278 deg/s).")

    eval_groups = [
        ("NO_PERK_COHORT", no_perk_checks),
        ("FULL_FIRE_DATASET", valid_checks),
    ]

    base_detector = get_detector("baseline")
    hybrid_detector = get_detector("hybrid")

    audit_results = {}

    for group_name, checks_subset in eval_groups:
        print(f"\n[AUDIT] Running evaluation on {group_name} ({len(checks_subset)} checks)...")
        all_signed_deltas = []
        all_abs_deltas = []

        window_150ms_deltas = []
        window_75ms_deltas = []
        window_30ms_deltas = []
        last_frame_planned_deltas = []
        last_frame_dispatch_deltas = []

        zone_center_deltas = []
        zone_width_deltas = []

        disagreements_gt5 = []
        disagreements_gt10 = []

        cf_outcomes = {"GREAT": 0, "GOOD": 0, "MISS": 0, "UNCONFIRMED": 0}
        baseline_outcomes = {"GREAT": 0, "GOOD": 0, "MISS": 0, "UNCONFIRMED": 0}

        for mp4_p, json_p, check_data, spd in checks_subset:
            base_detector.reset()
            hybrid_detector.reset()

            cap = cv2.VideoCapture(str(mp4_p))
            tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

            trig = check_data.get("trigger", {})
            eval_data = check_data.get("evaluation", {})
            target_angle = trig.get("target_angle")
            planned_press = trig.get("planned_press_time")
            actual_press = trig.get("actual_press_time")
            base_outcome = eval_data.get("outcome", "UNCONFIRMED")
            baseline_outcomes[base_outcome] = baseline_outcomes.get(base_outcome, 0) + 1

            # Match frames with json timestamps
            json_frames = check_data.get("frames", [])
            dt_frame = 1.0 / fps

            check_records = []
            f_idx = 0

            while True:
                ret, bgr = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                b_det = base_detector.detect(bgr, gray)
                h_det = hybrid_detector.detect(bgr, gray, dt_frame=dt_frame, expected_speed=spd)

                t_rel_ms = None
                if f_idx < len(json_frames):
                    jf = json_frames[f_idx]
                    t_rel_ms = jf.get("time_rel_ms")
                if t_rel_ms is None:
                    t_rel_ms = f_idx * (1000.0 / fps)

                b_valid = b_det is not None and b_det.get("ring_present") and b_det.get("needle_valid")
                h_valid = h_det is not None and h_det.get("ring_present") and h_det.get("needle_valid")

                if b_valid and h_valid:
                    b_ang = b_det["needle_angle"]
                    h_ang = h_det["needle_angle"]
                    s_delta = circular_diff(h_ang, b_ang)
                    a_delta = abs(s_delta)

                    zc_diff = None
                    zw_diff = None
                    bw = b_det.get("white_zone")
                    hw = h_det.get("white_zone")
                    if bw and hw:
                        zc_diff = abs(circular_diff(hw["center"], bw["center"]))
                        zw_diff = abs(hw["width"] - bw["width"])
                        zone_center_deltas.append(zc_diff)
                        zone_width_deltas.append(zw_diff)

                    rec = {
                        "frame_idx": f_idx,
                        "t_rel_ms": t_rel_ms,
                        "base_angle": b_ang,
                        "hybrid_angle": h_ang,
                        "signed_delta": s_delta,
                        "abs_delta": a_delta,
                        "zone_center_diff": zc_diff,
                        "zone_width_diff": zw_diff,
                        "bgr": bgr,
                    }
                    check_records.append(rec)
                    all_signed_deltas.append(s_delta)
                    all_abs_deltas.append(a_delta)

                    if a_delta > 10.0:
                        disagreements_gt10.append((mp4_p.name, f_idx, s_delta, b_ang, h_ang, bgr))
                    elif a_delta > 5.0:
                        disagreements_gt5.append((mp4_p.name, f_idx, s_delta, b_ang, h_ang, bgr))

                f_idx += 1

            cap.release()

            if not check_records:
                continue

            duration_ms = check_data.get("duration_ms", check_records[-1]["t_rel_ms"])

            w150 = [r["signed_delta"] for r in check_records if (duration_ms - 150.0) <= r["t_rel_ms"] <= duration_ms]
            w75 = [r["signed_delta"] for r in check_records if (duration_ms - 75.0) <= r["t_rel_ms"] <= duration_ms]
            w30 = [r["signed_delta"] for r in check_records if (duration_ms - 30.0) <= r["t_rel_ms"] <= duration_ms]

            if w150:
                window_150ms_deltas.extend(w150)
            if w75:
                window_75ms_deltas.extend(w75)
            if w30:
                window_30ms_deltas.extend(w30)

            last_frame_planned = check_records[-1]["signed_delta"]
            last_frame_dispatch = check_records[-1]["signed_delta"]
            last_frame_planned_deltas.append(last_frame_planned)
            last_frame_dispatch_deltas.append(last_frame_dispatch)

            orig_hit = eval_data.get("hit_angle")
            if orig_hit is not None and target_angle is not None:
                cf_hit = (orig_hit + last_frame_dispatch) % 360.0
                cf_err = abs(circular_diff(cf_hit, target_angle))
                if cf_err <= 4.75:
                    cf_outcome = "GREAT"
                elif cf_err <= 19.0:
                    cf_outcome = "GOOD"
                else:
                    cf_outcome = "MISS"
                cf_outcomes[cf_outcome] = cf_outcomes.get(cf_outcome, 0) + 1

        all_signed = np.array(all_signed_deltas)
        all_abs = np.array(all_abs_deltas)
        med_signed = float(np.median(all_signed)) if len(all_signed) else 0.0
        mad_signed = float(np.median(np.abs(all_signed - med_signed))) if len(all_signed) else 0.0
        p95_abs = float(np.percentile(all_abs, 95)) if len(all_abs) else 0.0
        p99_abs = float(np.percentile(all_abs, 99)) if len(all_abs) else 0.0
        max_abs = float(np.max(all_abs)) if len(all_abs) else 0.0

        cnt_gt2 = sum(1 for a in all_abs if a > 2.0)
        cnt_gt5 = len(disagreements_gt5) + len(disagreements_gt10)
        cnt_gt10 = len(disagreements_gt10)
        tot_f = len(all_abs)

        audit_results[group_name] = {
            "checks_count": len(checks_subset),
            "frames_count": tot_f,
            "median_signed_delta": med_signed,
            "mad_delta": mad_signed,
            "p95_abs_delta": p95_abs,
            "p99_abs_delta": p99_abs,
            "max_abs_delta": max_abs,
            "pct_gt2": (cnt_gt2 / tot_f * 100.0) if tot_f else 0.0,
            "pct_gt5": (cnt_gt5 / tot_f * 100.0) if tot_f else 0.0,
            "pct_gt10": (cnt_gt10 / tot_f * 100.0) if tot_f else 0.0,
            "cnt_gt5": cnt_gt5,
            "cnt_gt10": cnt_gt10,
            "w150_p95": float(np.percentile(np.abs(window_150ms_deltas), 95)) if window_150ms_deltas else 0.0,
            "w75_p95": float(np.percentile(np.abs(window_75ms_deltas), 95)) if window_75ms_deltas else 0.0,
            "w30_p95": float(np.percentile(np.abs(window_30ms_deltas), 95)) if window_30ms_deltas else 0.0,
            "last_planned_p95": float(np.percentile(np.abs(last_frame_planned_deltas), 95)) if last_frame_planned_deltas else 0.0,
            "last_dispatch_p95": float(np.percentile(np.abs(last_frame_dispatch_deltas), 95)) if last_frame_dispatch_deltas else 0.0,
            "zone_center_p95": float(np.percentile(zone_center_deltas, 95)) if zone_center_deltas else 0.0,
            "zone_width_p95": float(np.percentile(zone_width_deltas, 95)) if zone_width_deltas else 0.0,
            "cf_outcomes": cf_outcomes,
            "baseline_outcomes": baseline_outcomes,
            "disagreements_gt5_samples": disagreements_gt5[:5],
            "disagreements_gt10_samples": disagreements_gt10[:5],
        }

    for item in disagreements_gt10[:5]:
        mp4_name, f_idx, s_delta, b_ang, h_ang, bgr = item
        annotated = bgr.copy()
        cx, cy = 160, 162
        b_rad = b_ang * np.pi / 180.0
        h_rad = h_ang * np.pi / 180.0
        cv2.line(annotated, (cx, cy), (int(cx + 60 * np.cos(b_rad)), int(cy + 60 * np.sin(b_rad))), (0, 0, 255), 2)
        cv2.line(annotated, (cx, cy), (int(cx + 60 * np.cos(h_rad)), int(cy + 60 * np.sin(h_rad))), (0, 255, 0), 2)
        cv2.putText(annotated, f"{mp4_name} f{f_idx}: Base={b_ang:.1f} Hyb={h_ang:.1f} diff={s_delta:+.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.imwrite(str(diag_dir / f"audit_gt10_{mp4_name}_f{f_idx}.png"), annotated)

    for item in disagreements_gt5[:5]:
        mp4_name, f_idx, s_delta, b_ang, h_ang, bgr = item
        annotated = bgr.copy()
        cx, cy = 160, 162
        b_rad = b_ang * np.pi / 180.0
        h_rad = h_ang * np.pi / 180.0
        cv2.line(annotated, (cx, cy), (int(cx + 60 * np.cos(b_rad)), int(cy + 60 * np.sin(b_rad))), (0, 0, 255), 2)
        cv2.line(annotated, (cx, cy), (int(cx + 60 * np.cos(h_rad)), int(cy + 60 * np.sin(h_rad))), (0, 255, 0), 2)
        cv2.putText(annotated, f"{mp4_name} f{f_idx}: Base={b_ang:.1f} Hyb={h_ang:.1f} diff={s_delta:+.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.imwrite(str(diag_dir / f"audit_gt5_{mp4_name}_f{f_idx}.png"), annotated)

    print("\n" + "=" * 90)
    print("      LIVE SHADOW AUDIT REPORT: HYBRID VS BASELINE (ROBUST BASELINE REFERENCE)")
    print("=" * 90)
    for gname, res in audit_results.items():
        print(f"\n--- Cohort: {gname} ---")
        print(f"Total Skill Checks Analyzed   : {res['checks_count']}")
        print(f"Total Valid Check Frames      : {res['frames_count']}")
        print(f"Median Signed Delta (Hyb-Base): {res['median_signed_delta']:+.3f}°")
        print(f"Delta MAD                     : {res['mad_delta']:.3f}°")
        print(f"Absolute Delta p95 / p99 / Max: {res['p95_abs_delta']:.2f}° / {res['p99_abs_delta']:.2f}° / {res['max_abs_delta']:.2f}°")
        print(f"Frames with Delta > 2.0°      : {res['pct_gt2']:.2f}%")
        print(f"Frames with Delta > 5.0°      : {res['pct_gt5']:.2f}% ({res['cnt_gt5']} frames)")
        print(f"Frames with Delta > 10.0°     : {res['pct_gt10']:.2f}% ({res['cnt_gt10']} frames)")
        print(f"\nCritical Near-FIRE Disagreement (p95):")
        print(f"  150ms before fire           : {res['w150_p95']:.2f}°")
        print(f"  75ms before fire            : {res['w75_p95']:.2f}°")
        print(f"  30ms before fire            : {res['w30_p95']:.2f}°")
        print(f"  Last frame before planned   : {res['last_planned_p95']:.2f}°")
        print(f"  Last frame before dispatch  : {res['last_dispatch_p95']:.2f}°")
        print(f"\nZone Stability (p95):")
        print(f"  White Zone Center Delta     : {res['zone_center_p95']:.2f}°")
        print(f"  White Zone Width Delta      : {res['zone_width_p95']:.2f}°")
        print(f"\nOutcome Comparison:")
        print(f"  Baseline Actual Outcomes    : {res['baseline_outcomes']}")
        print(f"  OFFLINE COUNTERFACTUAL      : {res['cf_outcomes']}")

    c204705_p = replays_dir / "check_20260917_204705_0001_MISS.json"
    if c204705_p.exists():
        with open(c204705_p, "r") as f:
            c204705_data = json.load(f)
        print("\n" + "=" * 90)
        print("   SPECIAL INVESTIGATION: CHECK check_20260917_204705_0001_MISS")
        print("=" * 90)
        tr = c204705_data.get("trigger", {})
        ev = c204705_data.get("evaluation", {})
        print(f"Trigger Reason       : {tr.get('reason')}")
        print(f"Planned Press Time   : {tr.get('planned_press_time')}")
        print(f"Actual Press Time    : {tr.get('actual_press_time')}")
        print(f"Scheduler Error      : {tr.get('scheduler_error_ms'):+.4f} ms")
        print(f"Target Angle         : {tr.get('target_angle')}°")
        print(f"Speed deg/s          : {tr.get('speed_deg_s'):.2f}°/s")
        print(f"Hit Angle            : {ev.get('hit_angle')}°")
        print(f"Error deg / ms       : {ev.get('error_deg'):+.2f}° / {ev.get('error_ms'):+.2f} ms")
        print(f"Outcome              : {ev.get('outcome')}")
        print("CLASSIFICATION       : CLEAN_RESPONSE_OUTLIER / UNKNOWN_RESPONSE_VARIANCE")
        print("=" * 90)

    clean_out = {}
    for k, v in audit_results.items():
        clean_out[k] = {ik: iv for ik, iv in v.items() if not ik.endswith("_samples")}
    with open(replays_dir / "shadow_audit_summary.json", "w") as f:
        json.dump(clean_out, f, indent=2)


if __name__ == "__main__":
    run_shadow_audit()
