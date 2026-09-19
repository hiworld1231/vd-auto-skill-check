#!/usr/bin/env python3
"""
A/B Block Calibration Evaluation Tool for Violent District Bot.

Simulates and evaluates candidate base latencies (e.g. 135ms, 140ms, 145ms, 150ms)
on clean, verified solo checks (chain_count=1, confirmed freeze plateau, low scheduler jitter,
measured base speed ~278°/s).

Computes:
- Great / Good / Miss distribution and rates
- Median signed error (ms) and MAD (ms)
- Projected hit angle relative to Great zone bounds
- Recommendation for optimal physical lead time
"""

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def compute_mad(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def is_angle_in_arc(angle: float, start: float, end: float) -> bool:
    """Checks whether an angle lies within [start, end] arc taking modulo 360 into account."""
    a = angle % 360.0
    s = start % 360.0
    e = end % 360.0
    if s <= e:
        return s <= a <= e
    else:
        return a >= s or a <= e


def evaluate_candidate_latency(
    checks: List[Dict[str, Any]],
    candidate_lat_ms: float,
) -> Dict[str, Any]:
    great_count = 0
    good_count = 0
    miss_count = 0
    proj_errors_ms = []
    proj_errors_deg = []

    details = []

    for c in checks:
        cid = c.get("check_id")
        tr = c.get("trigger") or {}
        ev = c.get("evaluation") or {}
        locked_zones = c.get("locked_zones") or {}
        w_zone = locked_zones.get("white")
        b_zone = locked_zones.get("black")

        used_lat = tr.get("latency_ms", 122.1)
        speed = tr.get("speed_deg_s", 278.0)
        target_ang = ev.get("target_angle", tr.get("target_angle", 0.0))
        actual_err_deg = ev.get("error_deg", 0.0)
        actual_err_ms = ev.get("error_ms", 0.0)

        # Delta in latency: higher latency = earlier press = needle arrives earlier (lower angle)
        delta_lat = candidate_lat_ms - used_lat
        proj_err_ms = actual_err_ms - delta_lat
        proj_err_deg = actual_err_deg - (delta_lat / 1000.0) * speed
        proj_hit_ang = (target_ang + proj_err_deg) % 360.0

        in_great = False
        in_good = False

        if w_zone:
            in_great = is_angle_in_arc(proj_hit_ang, w_zone["start"], w_zone["end"])
        if b_zone:
            in_good = is_angle_in_arc(proj_hit_ang, b_zone["start"], b_zone["end"])

        outcome = "GREAT" if in_great else ("GOOD" if in_good else "MISS")
        if outcome == "GREAT":
            great_count += 1
        elif outcome == "GOOD":
            good_count += 1
        else:
            miss_count += 1

        proj_errors_ms.append(proj_err_ms)
        proj_errors_deg.append(proj_err_deg)

        details.append({
            "check_id": cid,
            "actual_outcome": ev.get("outcome"),
            "actual_error_ms": actual_err_ms,
            "proj_outcome": outcome,
            "proj_error_ms": proj_err_ms,
            "proj_hit_ang": proj_hit_ang,
        })

    arr_ms = np.array(proj_errors_ms) if proj_errors_ms else np.array([0.0])
    total = len(checks)
    great_pct = (great_count / total * 100.0) if total > 0 else 0.0
    good_pct = (good_count / total * 100.0) if total > 0 else 0.0
    miss_pct = (miss_count / total * 100.0) if total > 0 else 0.0
    acc_pct = ((great_count + good_count) / total * 100.0) if total > 0 else 0.0

    return {
        "candidate_latency_ms": candidate_lat_ms,
        "total_checks": total,
        "great_count": great_count,
        "good_count": good_count,
        "miss_count": miss_count,
        "great_pct": great_pct,
        "good_pct": good_pct,
        "miss_pct": miss_pct,
        "accuracy_pct": acc_pct,
        "median_error_ms": float(np.median(arr_ms)),
        "mad_error_ms": compute_mad(arr_ms),
        "mean_error_ms": float(np.mean(arr_ms)),
        "std_error_ms": float(np.std(arr_ms)),
        "details": details,
    }


def find_clean_solo_checks(
    replays_dir: Path,
    since_timestamp: Optional[str] = None,
    until_timestamp: Optional[str] = None,
    max_speed_delta: float = 20.0,
    base_speed: float = 278.0,
) -> List[Dict[str, Any]]:
    json_files = sorted(replays_dir.glob("check_*.json"))
    clean = []

    for p in json_files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        ts = data.get("timestamp", "")
        if since_timestamp and ts < since_timestamp:
            continue
        if until_timestamp and ts > until_timestamp:
            continue

        # Criteria for clean solo check:
        # 1. chain_count == 1
        if data.get("chain_count", 1) != 1:
            continue

        ev = data.get("evaluation") or {}
        tr = data.get("trigger") or {}

        # 2. Must have fired and reached a confirmed freeze plateau
        if not tr.get("fired", False) or not ev.get("plateau_found", False):
            continue

        outcome = ev.get("outcome")
        if outcome not in ("GREAT", "GOOD", "MISS"):
            continue

        # 3. Clean scheduler execution (|scheduler_error_ms| <= 2.0ms)
        sched_err = tr.get("scheduler_error_ms")
        if sched_err is None or abs(sched_err) > 2.0:
            continue

        # 4. Triggered by spin timer (clean scheduled trigger)
        reason = tr.get("reason", "")
        if not ("СПИН" in reason or "SCHEDULED" in reason):
            continue

        # 5. Speed within base speed tolerance
        spd = tr.get("speed_deg_s", 0.0)
        if abs(spd - base_speed) > max_speed_delta:
            continue

        # 6. Must have locked zone geometry
        zones = data.get("locked_zones") or {}
        if not zones.get("white"):
            continue

        data["_file"] = p.name
        clean.append(data)

    return clean


def run_ab_blocks_analysis(
    replays_dir: Optional[Path] = None,
    candidates: Optional[List[float]] = None,
    since_timestamp: Optional[str] = None,
    until_timestamp: Optional[str] = None,
):
    rdir = replays_dir or (ROOT / "replays")
    candidate_list = candidates or [122.1, 130.0, 135.0, 140.0, 145.0, 150.0]

    clean_checks = find_clean_solo_checks(
        rdir,
        since_timestamp=since_timestamp,
        until_timestamp=until_timestamp,
    )

    if not clean_checks:
        print(f"[INFO] No clean solo scheduled checks found in {rdir}.")
        return

    print("=" * 80)
    print(f"       A/B CALIBRATION BLOCKS SIMULATION & EVALUATION ({len(clean_checks)} clean solo checks)")
    print("=" * 80)
    print(f"Sample selection criteria: chain_count=1, spin timer, sched_jitter <= 2ms, speed ~278°/s")
    print(f"Sample range: {clean_checks[0].get('check_id')} -> {clean_checks[-1].get('check_id')}\n")

    results = []
    for cand in candidate_list:
        res = evaluate_candidate_latency(clean_checks, cand)
        results.append(res)

    # Print comparative table
    print(f"{'Candidate':<12} | {'Great %':<9} | {'Good %':<8} | {'Miss %':<8} | {'Acc %':<7} | {'Median Error':<14} | {'MAD':<8} | {'Std':<8}")
    print("-" * 88)
    for r in results:
        cand_str = f"{r['candidate_latency_ms']:.1f}ms"
        grt_str = f"{r['great_pct']:5.1f}% ({r['great_count']})"
        god_str = f"{r['good_pct']:5.1f}% ({r['good_count']})"
        mis_str = f"{r['miss_pct']:5.1f}% ({r['miss_count']})"
        acc_str = f"{r['accuracy_pct']:5.1f}%"
        err_str = f"{r['median_error_ms']:+5.1f}ms"
        mad_str = f"{r['mad_error_ms']:4.1f}ms"
        std_str = f"{r['std_error_ms']:4.1f}ms"
        print(f"{cand_str:<12} | {grt_str:<9} | {god_str:<8} | {mis_str:<8} | {acc_str:<7} | {err_str:<14} | {mad_str:<8} | {std_str:<8}")

    print("-" * 88)

    # Find optimal candidate
    best_cand = max(results, key=lambda x: (x["great_pct"], -abs(x["median_error_ms"])))
    print(f"\n🎯 OPTIMAL LEAD TIME: {best_cand['candidate_latency_ms']:.1f}ms")
    print(f"   --> Projected Great Rate: {best_cand['great_pct']:.1f}% (Accuracy: {best_cand['accuracy_pct']:.1f}%)")
    print(f"   --> Projected Median Error: {best_cand['median_error_ms']:+.1f}ms (MAD: {best_cand['mad_error_ms']:.1f}ms)")
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser(description="A/B Calibration Blocks Simulator")
    ap.add_argument("--replays", help="Path to replays directory (default: replays/)")
    ap.add_argument("--candidates", nargs="+", type=float, default=[122.1, 130.0, 135.0, 140.0, 145.0, 150.0], help="Candidate latencies in ms")
    ap.add_argument("--since", help="Filter checks after timestamp")
    ap.add_argument("--until", help="Filter checks before timestamp")
    args = ap.parse_args()

    rdir = Path(args.replays) if args.replays else None
    run_ab_blocks_analysis(
        replays_dir=rdir,
        candidates=args.candidates,
        since_timestamp=args.since,
        until_timestamp=args.until,
    )


if __name__ == "__main__":
    main()
