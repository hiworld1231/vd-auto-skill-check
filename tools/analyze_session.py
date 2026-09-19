#!/usr/bin/env python3
"""
Comprehensive Post-Session Telemetry & Replay Analyzer for Violent District Bot.
Analyzes check replays, trigger breakdowns, scheduler jitter, prediction lateness,
speed divergence (lock vs fire), per-tier performance, and deep dives into MISS/GOOD.
"""

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.learner import classify_miss


def compute_mad(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def analyze_session(
    replays_dir: Optional[Path] = None,
    since_timestamp: Optional[str] = None,
    until_timestamp: Optional[str] = None,
):
    replays_path = replays_dir or (ROOT / "replays")
    if not replays_path.exists():
        print(f"[ERROR] Replays directory {replays_path} does not exist.")
        return

    json_files = sorted(replays_path.glob("check_*.json"))
    if not json_files:
        print(f"[INFO] No replay JSON files found in {replays_path}.")
        return

    checks = []
    for p in json_files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            ts = data.get("timestamp", "")
            if since_timestamp and ts < since_timestamp:
                continue
            if until_timestamp and ts > until_timestamp:
                continue
            data["_file"] = p.name
            checks.append(data)
        except Exception:
            continue

    if not checks:
        print(f"[INFO] No matching checks found after timestamp {since_timestamp}.")
        return

    print("=" * 80)
    print(f"             VIOLENT DISTRICT SESSION DEEP AUDIT REPORT ({len(checks)} checks)")
    print("=" * 80)

    # LIVE MEASURED VALID CHECKS AUDIT SECTION
    valid_checks = [
        c for c in checks
        if (c.get("evaluation") or {}).get("outcome") in ("GREAT", "GOOD", "MISS")
        and (c.get("trigger") or {}).get("fired", False)
    ]

    print("\n" + "=" * 80)
    print(f"[LIVE MEASURED] CONTROLLED TEST BLOCK REPORT (N={len(valid_checks)} valid checks)")
    print("=" * 80)

    if valid_checks:
        v_great = sum(1 for c in valid_checks if (c.get("evaluation") or {}).get("outcome") == "GREAT")
        v_good = sum(1 for c in valid_checks if (c.get("evaluation") or {}).get("outcome") == "GOOD")
        v_miss = sum(1 for c in valid_checks if (c.get("evaluation") or {}).get("outcome") == "MISS")
        v_delays = [(c.get("trigger") or {}).get("latency_ms", 0.0) for c in valid_checks]
        v_errs_ms = [(c.get("evaluation") or {}).get("error_ms", 0.0) for c in valid_checks]
        v_sched = [(c.get("trigger") or {}).get("scheduler_error_ms", 0.0) for c in valid_checks if (c.get("trigger") or {}).get("scheduler_error_ms") is not None]

        frame_dts = []
        for c in valid_checks:
            frames = c.get("frames") or []
            if len(frames) >= 2:
                for i in range(1, len(frames)):
                    dt = frames[i].get("time_rel_ms", 0.0) - frames[i - 1].get("time_rel_ms", 0.0)
                    if dt > 0.01:
                        frame_dts.append(dt)

        fallback_count = sum(
            1 for c in valid_checks
            if "FALLBACK" in (c.get("trigger") or {}).get("reason", "")
            or (c.get("trigger") or {}).get("measured_speed_at_lock") is None
        )

        arr_err = np.array(v_errs_ms)
        arr_del = np.array(v_delays)
        arr_sch = np.array(v_sched) if v_sched else np.array([0.0])
        arr_fdt = np.array(frame_dts) if frame_dts else np.array([0.0])

        print(f"[LIVE MEASURED] Outcomes:")
        print(f"   N valid: {len(valid_checks)}")
        print(f"   GREAT:   {v_great} ({v_great / len(valid_checks) * 100.0:.1f}%)")
        print(f"   GOOD:    {v_good} ({v_good / len(valid_checks) * 100.0:.1f}%)")
        print(f"   MISS:    {v_miss} ({v_miss / len(valid_checks) * 100.0:.1f}%)")

        print(f"\n[LIVE MEASURED] Actual Used Delay:")
        print(f"   min={np.min(arr_del):.1f}ms | median={np.median(arr_del):.1f}ms | max={np.max(arr_del):.1f}ms")

        print(f"\n[LIVE MEASURED] Signed Error (ms):")
        print(
            f"   median={np.median(arr_err):+.1f}ms | MAD={compute_mad(arr_err):.1f}ms | "
            f"p25={np.percentile(arr_err, 25):+.1f}ms / p75={np.percentile(arr_err, 75):+.1f}ms | "
            f"min={np.min(arr_err):+.1f}ms / max={np.max(arr_err):+.1f}ms"
        )

        print(f"\n[LIVE MEASURED] Scheduler Lateness:")
        print(f"   p50={np.percentile(arr_sch, 50):+.2f}ms | p95={np.percentile(arr_sch, 95):+.2f}ms | max={np.max(np.abs(arr_sch)):.2f}ms")

        print(f"\n[LIVE MEASURED] Capture Frame Interval:")
        if frame_dts:
            print(f"   p50={np.percentile(arr_fdt, 50):.2f}ms | p95={np.percentile(arr_fdt, 95):.2f}ms | p99={np.percentile(arr_fdt, 99):.2f}ms")
        else:
            print("   p50=N/A | p95=N/A | p99=N/A (no frame timestamps)")

        print(f"\n[LIVE MEASURED] Detector Fallback Count: {fallback_count}/{len(valid_checks)}")

        print(f"\n[LIVE MEASURED] VALID CHECKS TABLE:")
        hdr = f"{'Timestamp':<19} | {'Outcome':<7} | {'Delay':<7} | {'Speed':<8} | {'Target':<6} | {'Hit':<6} | {'SignedErr':<10} | {'SchedErr':<9} | {'FrameAge':<8}"
        print(hdr)
        print("-" * len(hdr))
        for c in valid_checks:
            ev = c.get("evaluation") or {}
            tr = c.get("trigger") or {}
            ts_full = c.get("timestamp", "")
            ts_short = ts_full[11:19] if len(ts_full) >= 19 else ts_full
            out = ev.get("outcome", "UNKNOWN")
            lat = tr.get("latency_ms", 0.0)
            spd = tr.get("speed_deg_s", 0.0)
            tgt = ev.get("target_angle", 0.0)
            hit = ev.get("hit_angle")
            hit_str = f"{hit:5.1f}°" if hit is not None else " N/A "
            err = ev.get("error_ms", 0.0)
            sch = tr.get("scheduler_error_ms", 0.0)
            f_age = tr.get("last_frame_age_ms_at_fire", 0.0)
            print(
                f"{ts_short:<19} | {out:<7} | {lat:5.1f}ms | {spd:6.1f}°/s | {tgt:5.1f}° | {hit_str} | "
                f"{err:+8.1f}ms | {sch:+7.2f}ms | {f_age:6.1f}ms"
            )
        print("-" * len(hdr))
    else:
        print("[LIVE MEASURED] No valid checks in selected window (0 Great/Good/Miss fires).")

    # 1. Outcomes
    outcomes = {}
    for c in checks:
        ev = c.get("evaluation") or {}
        out = ev.get("outcome", "UNKNOWN")
        outcomes[out] = outcomes.get(out, 0) + 1

    great_c = outcomes.get("GREAT", 0)
    good_c = outcomes.get("GOOD", 0)
    miss_c = outcomes.get("MISS", 0)
    abort_c = sum(v for k, v in outcomes.items() if "ABORT" in k)
    valid_total = great_c + good_c + miss_c
    acc_pct = ((great_c + good_c) / valid_total * 100.0) if valid_total > 0 else 0.0
    great_pct = (great_c / valid_total * 100.0) if valid_total > 0 else 0.0

    print("\n1. OUTCOMES:")
    for k, v in sorted(outcomes.items()):
        pct = (v / len(checks) * 100.0)
        print(f"   {k:<16s}: {v:3d} ({pct:5.1f}%)")
    print(f"   --> Total Valid (Great+Good+Miss): {valid_total}")
    print(f"   --> Accuracy (Great+Good):        {acc_pct:5.1f}%")
    print(f"   --> Great Rate:                   {great_pct:5.1f}%")

    # 2. Latency Control & Experiment Audit
    req_latencies = []
    used_latencies = []
    override_count = 0
    for c in checks:
        conf_lat = c.get("configured_latency_ms")
        tr = c.get("trigger") or {}
        used_lat = tr.get("latency_ms")
        if conf_lat is not None:
            req_latencies.append(conf_lat)
        if used_lat is not None:
            used_latencies.append(used_lat)
            if conf_lat is not None and abs(used_lat - conf_lat) > 0.05:
                override_count += 1

    print("\n2. LATENCY CONTROL & EXPERIMENT AUDIT:")
    if req_latencies:
        req_arr = np.array(req_latencies)
        req_str = f"med={np.median(req_arr):.1f}ms (min={np.min(req_arr):.1f}ms, max={np.max(req_arr):.1f}ms)"
        if np.min(req_arr) == np.max(req_arr):
            req_str = f"{req_arr[0]:.1f}ms"
        print(f"   Requested Base Latency:  {req_str}")
    else:
        print("   Requested Base Latency:  None recorded")

    if used_latencies:
        used_arr = np.array(used_latencies)
        print(
            f"   Actual Used Delay:       min={np.min(used_arr):.1f}ms | "
            f"med={np.median(used_arr):.1f}ms | "
            f"max={np.max(used_arr):.1f}ms (N={len(used_arr)})"
        )
    else:
        print("   Actual Used Delay:       No triggers executed")

    override_pct = (override_count / len(used_latencies) * 100.0) if used_latencies else 0.0
    print(f"   Profile Overrides Count: {override_count}/{len(used_latencies)} ({override_pct:.1f}%)")

    # 3. Trigger Breakdown & Timing Jitter
    trigger_counts = {"SCHEDULED": 0, "IMMEDIATE": 0, "BACKUP": 0, "NOT_FIRED": 0}
    pred_lateness = []
    sched_jitters = []
    speed_divergences = []
    speed_lock_fire_pairs = []

    tier_stats = {}

    for c in checks:
        tr = c.get("trigger") or {}
        ev = c.get("evaluation") or {}
        fired = tr.get("fired", False)
        reason = tr.get("reason", "UNKNOWN")

        if not fired:
            trigger_counts["NOT_FIRED"] += 1
            continue

        if "СПИН" in reason or "SCHEDULED" in reason:
            trigger_counts["SCHEDULED"] += 1
        elif "BACKUP" in reason:
            trigger_counts["BACKUP"] += 1
        else:
            trigger_counts["IMMEDIATE"] += 1

        # Prediction lateness & scheduler jitter
        # scheduler_error_ms in replay can represent prediction lateness or jitter
        sched_err = tr.get("scheduler_error_ms")
        if sched_err is not None:
            pred_lateness.append(sched_err)
            if "СПИН" in reason:
                sched_jitters.append(sched_err)

        # Speed lock vs fire
        spd_fire = tr.get("speed_deg_s")
        spd_lock = tr.get("measured_speed_at_lock") or spd_fire
        tier = tr.get("selected_speed_tier") or (int(round(spd_fire / 25.0) * 25.0) if spd_fire else 275)

        if spd_fire is not None and spd_lock is not None:
            div = abs(spd_fire - spd_lock)
            speed_divergences.append(div)
            speed_lock_fire_pairs.append((c.get("check_id"), spd_lock, spd_fire, div))

        # Per-tier tracking
        if tier not in tier_stats:
            tier_stats[tier] = {
                "checks": 0,
                "great": 0,
                "good": 0,
                "miss": 0,
                "used_delays": [],
                "ideal_delays": [],
                "speeds": [],
                "divergences": [],
            }
        tst = tier_stats[tier]
        tst["checks"] += 1
        out = ev.get("outcome", "UNKNOWN")
        if out == "GREAT":
            tst["great"] += 1
        elif out == "GOOD":
            tst["good"] += 1
        elif out == "MISS":
            tst["miss"] += 1

        lat = tr.get("latency_ms")
        err_deg = ev.get("error_deg")
        if lat is not None:
            tst["used_delays"].append(lat)
            if spd_fire and err_deg is not None and ev.get("plateau_found"):
                ideal = lat + (err_deg / spd_fire) * 1000.0
                tst["ideal_delays"].append(ideal)

        if spd_fire:
            tst["speeds"].append(spd_fire)
        if spd_fire and spd_lock:
            tst["divergences"].append(abs(spd_fire - spd_lock))

    print("\n3. TRIGGER DISPATCH BREAKDOWN:")
    for k, v in trigger_counts.items():
        pct = (v / len(checks) * 100.0)
        print(f"   {k:<16s}: {v:3d} ({pct:5.1f}%)")

    print("\n4. TIMING & SCHEDULER JITTER:")
    if pred_lateness:
        arr_late = np.array(pred_lateness)
        print(
            f"   Prediction Lateness: p50={np.percentile(arr_late, 50):+5.2f}ms | "
            f"p95={np.percentile(arr_late, 95):+5.2f}ms | "
            f"p99={np.percentile(arr_late, 99):+5.2f}ms | "
            f"max={np.max(np.abs(arr_late)):5.2f}ms (N={len(arr_late)})"
        )
    else:
        print("   Prediction Lateness: No recorded samples")

    if sched_jitters:
        arr_jit = np.array(sched_jitters)
        print(
            f"   Scheduler Spin Jitter: p50={np.percentile(arr_jit, 50):+5.2f}ms | "
            f"p95={np.percentile(arr_jit, 95):+5.2f}ms | "
            f"p99={np.percentile(arr_jit, 99):+5.2f}ms | "
            f"max={np.max(np.abs(arr_jit)):5.2f}ms (N={len(arr_jit)})"
        )
    else:
        print("   Scheduler Spin Jitter: No spin timer triggers in sample")

    print("\n5. SPEED STABILITY & DIVERGENCE (LOCK vs FIRE):")
    if speed_divergences:
        arr_div = np.array(speed_divergences)
        unstable_25 = sum(1 for d in speed_divergences if d > 25.0)
        unstable_35 = sum(1 for d in speed_divergences if d > 35.0)
        print(
            f"   Speed Divergence: p50={np.percentile(arr_div, 50):5.1f}°/s | "
            f"p95={np.percentile(arr_div, 95):5.1f}°/s | "
            f"max={np.max(arr_div):5.1f}°/s"
        )
        print(
            f"   Unstable (>25°/s): {unstable_25}/{len(speed_divergences)} ({unstable_25/len(speed_divergences)*100.0:.1f}%)"
        )
        print(
            f"   Rejected (>35°/s): {unstable_35}/{len(speed_divergences)} ({unstable_35/len(speed_divergences)*100.0:.1f}%)"
        )
        if unstable_25 > 0:
            print("   Top speed divergences:")
            sorted_divs = sorted(speed_lock_fire_pairs, key=lambda x: x[3], reverse=True)[:5]
            for cid, sl, sf, d in sorted_divs:
                print(f"     - {cid}: lock={sl:.1f}°/s -> fire={sf:.1f}°/s (diff={d:+.1f}°/s)")
    else:
        print("   Speed Divergence: No speed pairs recorded")

    print("\n6. SPEED TIERS BREAKDOWN:")
    for tier in sorted(tier_stats.keys()):
        st = tier_stats[tier]
        n_chk = st["checks"]
        n_grt = st["great"]
        n_god = st["good"]
        n_mis = st["miss"]
        val_t = n_grt + n_god + n_mis
        t_acc = (n_grt / val_t * 100.0) if val_t > 0 else 0.0

        used_med = float(np.median(st["used_delays"])) if st["used_delays"] else 0.0
        if st["ideal_delays"]:
            ideal_arr = np.array(st["ideal_delays"])
            ideal_med = float(np.median(ideal_arr))
            ideal_mad = compute_mad(ideal_arr)
            ideal_std = float(np.std(ideal_arr))
            ideal_str = f"ideal_med={ideal_med:5.1f}ms (MAD={ideal_mad:4.1f}ms, std={ideal_std:4.1f}ms)"
        else:
            ideal_str = "no clean freeze plateaus"

        spd_med = float(np.median(st["speeds"])) if st["speeds"] else 0.0
        print(
            f"   Tier ~{tier:3d}°/s (N={n_chk:2d}): Great={n_grt} Good={n_god} Miss={n_mis} "
            f"[{t_acc:5.1f}% Great] | used_delay={used_med:5.1f}ms | {ideal_str} | spd_med={spd_med:5.1f}°/s"
        )

    # 7. Deep Dive into MISS and GOOD
    non_great = [c for c in checks if (c.get("evaluation") or {}).get("outcome") in ("MISS", "GOOD")]
    miss_categories = collections.defaultdict(int)
    print(f"\n7. DEEP DIVE INTO MISS & GOOD CHECKS ({len(non_great)} total):")
    if not non_great:
        print("   🎉 Zero MISS and zero GOOD checks! 100% Great hit accuracy!")
    else:
        for c in non_great:
            cid = c.get("check_id")
            ev = c.get("evaluation") or {}
            tr = c.get("trigger") or {}
            out = ev.get("outcome")
            spd = tr.get("speed_deg_s", 0.0)
            lat = tr.get("latency_ms", 0.0)
            reason = tr.get("reason", "UNKNOWN")
            tier = tr.get("selected_speed_tier") or int(round(spd / 25.0) * 25.0)
            err_deg = ev.get("error_deg", 0.0)
            err_ms = ev.get("error_ms", 0.0)
            sched_err = tr.get("scheduler_error_ms")
            sched_str = f"{sched_err:+.2f}ms" if sched_err is not None else "N/A"
            ideal = lat + (err_deg / spd * 1000.0) if spd > 0 else lat

            miss_cat = None
            miss_detail = None
            if out == "MISS":
                dur_ms = c.get("duration_ms", 0.0)
                spd_lock = tr.get("measured_speed_at_lock")
                tier_lock = tr.get("selected_speed_tier")
                tier_fire = int(round(spd / 25.0) * 25.0) if spd > 15.0 else None
                last_frame_age = tr.get("last_frame_age_ms_at_fire")
                timing = tr.get("timing") or {}
                frames = c.get("frames") or []
                max_dt = 0.0
                if len(frames) >= 2:
                    dts = [frames[i]["time_rel_ms"] - frames[i - 1]["time_rel_ms"] for i in range(1, len(frames))]
                    max_dt = max(dts) if dts else 0.0

                miss_cat, miss_detail = classify_miss(
                    reason=reason,
                    scheduler_error_ms=sched_err,
                    last_frame_age_ms=last_frame_age,
                    capture_wait_ms=timing.get("capture_wait_ms"),
                    duration_ms=dur_ms,
                    speed_at_lock=spd_lock,
                    speed_at_fire=spd,
                    tier_at_lock=tier_lock,
                    tier_at_fire=tier_fire,
                    error_deg=err_deg,
                    error_ms=err_ms,
                    max_frame_gap_ms=max_dt,
                )
                miss_categories[miss_cat] += 1

            cat_str = f" [{miss_cat}]" if miss_cat else ""
            print(
                f"   ❌ [{out}]{cat_str} {cid}: Tier=~{tier}°/s (speed={spd:.1f}°/s) | "
                f"used_lat={lat:.1f}ms -> ideal={ideal:.1f}ms | "
                f"error={err_deg:+.1f}° ({err_ms:+.1f}ms) | "
                f"trigger={reason} (sched_err={sched_str})"
            )
            if miss_detail:
                print(f"      └─ Root Cause: {miss_detail}")

    # 8. MISS Taxonomy Breakdown
    total_misses = sum(miss_categories.values())
    print(f"\n8. MISS TAXONOMY BREAKDOWN ({total_misses} misses):")
    for cat in ["GHOST/REACQUIRE", "UNSTABLE_SPEED", "SPEED/TIER WRONG", "SCHEDULER LATE", "VISION/DROPPED FRAMES", "CLEAN TIMING MISS"]:
        cnt = miss_categories.get(cat, 0)
        marker = "🎯" if cat == "CLEAN TIMING MISS" else "⚠️"
        print(f"   {marker} {cat:<22}: {cnt:2d}")
    print("   --> CRITERIA: Only 'CLEAN TIMING MISS' qualifies for physical latency / profile delay calibration.")

    print("\n" + "=" * 80)


def main():
    ap = argparse.ArgumentParser(description="Violent District Post-Session Telemetry & Replay Analyzer")
    ap.add_argument("--replays", help="Path to replays directory (default: replays/)")
    ap.add_argument("--since", help="ISO timestamp filter (e.g. 2026-09-17T08:00:00)")
    ap.add_argument("--until", help="ISO timestamp filter (e.g. 2026-09-17T09:00:00)")
    args = ap.parse_args()

    rdir = Path(args.replays) if args.replays else None
    analyze_session(replays_dir=rdir, since_timestamp=args.since, until_timestamp=args.until)


if __name__ == "__main__":
    main()
