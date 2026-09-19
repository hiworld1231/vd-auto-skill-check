#!/usr/bin/env python3
"""
Comprehensive Offline Replay Audit for High-Speed Skill Checks and Speed Lock Timing.

Replays every recorded check from replays/ and oldreplays/ frame-by-frame
through the new SkillCheckPredictor to evaluate:
1. Is LOCKED_SPEED achieved before the physical fire deadline (lock_margin_ms)?
2. Distribution of lock_margin_ms across discrete speed buckets (<300, 300-400, 400-500, 500-600, 600-750, 750-900, >=900°/s).
3. Detailed breakdown for high speed checks (>=400, >=500, >=600, >=750, >=900°/s).
4. Identification of checks that would require FALLBACK_NO_LOCK vs normal SCHEDULED fire.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.predictor import (
    SkillCheckPredictor,
    STATE_NO_MOTION,
    STATE_PROVISIONAL,
    STATE_LOCKED,
    STATE_COMMITTED,
)
from core.learner import SpeedProfileManager


def audit_single_replay(
    replay_path: Path,
    speed_mgr: SpeedProfileManager,
) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(replay_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    raw_frames = data.get("frames", [])
    frames = [
        fr for fr in raw_frames
        if not fr.get("is_pre_roll", False) and "needle_angle" in fr
    ]
    if len(frames) < 5:
        return None

    white_zone = data.get("locked_zones", {}).get("white")
    black_zone = data.get("locked_zones", {}).get("black")
    if not white_zone:
        for fr in frames:
            w = fr.get("white_zone")
            if w and w.get("start") is not None:
                white_zone = w
                black_zone = fr.get("black_zone")
                break

    target_mode = data.get("target_mode", "GREAT")
    target_ratio = data.get("target_ratio", 0.50)
    cfg_lat = data.get("configured_latency_ms", 122.1)
    chain_count = data.get("chain_count", 1)
    is_chain = chain_count > 1

    predictor = SkillCheckPredictor(latency_ms=cfg_lat, target_offset_ratio=target_ratio)
    if is_chain:
        prev_spd = data.get("trigger", {}).get("speed_deg_s", 270.3)
        predictor.reset(keep_speed=True, default_speed=prev_spd, is_chain=True)

    motion_onset_t: Optional[float] = None
    motion_onset_ang: Optional[float] = None
    time_to_target_at_onset: Optional[float] = None

    lock_t: Optional[float] = None
    speed_at_lock: Optional[float] = None
    tier_at_lock: Optional[int] = None
    time_to_target_at_lock: Optional[float] = None
    margin_at_lock: Optional[float] = None
    desired_press_time: Optional[float] = None
    earliest_candidate_press_t: Optional[float] = None
    latest_safe_commit_t: Optional[float] = None

    provisional_timeline: List[Tuple[float, float]] = []
    locked_tier_latency: Optional[float] = None

    t0 = frames[0]["time_rel_ms"] / 1000.0

    for fr in frames:
        t = fr["time_rel_ms"] / 1000.0
        ang = fr["needle_angle"]
        strn = fr.get("needle_strength", 75.0)

        valid = predictor.update(t, ang, strn, white_zone, black_zone)
        if not valid:
            continue

        if motion_onset_t is None and predictor.motion_onset:
            motion_onset_t = t
            motion_onset_ang = ang
            if white_zone and white_zone.get("start") is not None:
                tgt = (white_zone["start"] + target_ratio * white_zone.get("width", 10.0)) % 360.0
                dist = (tgt - ang + 360.0) % 360.0
                spd = predictor.speed_deg_s if predictor.speed_deg_s > 0 else 270.3
                time_to_target_at_onset = (dist / spd) * 1000.0

        if predictor.state == STATE_PROVISIONAL:
            provisional_timeline.append((t - t0, predictor.speed_deg_s))

        if lock_t is None and predictor.has_stable_speed():
            lock_t = t
            speed_at_lock = predictor.speed_deg_s
            tier_at_lock = speed_mgr.get_tier(speed_at_lock)
            locked_tier_latency = speed_mgr.get_latency_for_speed(speed_at_lock)
            predictor.latency_s = locked_tier_latency / 1000.0

            pred = predictor.predict(t, ang, target=target_mode)
            if pred:
                margin_at_lock = pred["time_until_press_ms"]
                desired_press_time = pred["press_timestamp"]
                time_to_target_at_lock = pred["time_to_hit_ms"]
                earliest_candidate_press_t = t
                latest_safe_commit_t = desired_press_time

    final_speed = predictor.speed_deg_s
    final_tier = speed_mgr.get_tier(final_speed)
    evaluation = data.get("evaluation", {})
    outcome = evaluation.get("outcome", "UNKNOWN")
    trigger = data.get("trigger", {})
    rec_speed = trigger.get("speed_deg_s")
    rec_reason = trigger.get("reason", "UNKNOWN")

    scheduled_fire_possible = bool(lock_t is not None and margin_at_lock is not None and margin_at_lock >= 0.0)
    needed_fallback = bool(lock_t is None or (margin_at_lock is not None and margin_at_lock < 0.0))

    return {
        "file": replay_path.name,
        "path": str(replay_path),
        "chain_count": chain_count,
        "is_chain": is_chain,
        "outcome": outcome,
        "final_speed": final_speed,
        "final_tier": final_tier,
        "rec_speed": rec_speed,
        "rec_reason": rec_reason,
        "frames_count": len(frames),
        "duration_ms": data.get("duration_ms", 0.0),
        "motion_onset_t_ms": (motion_onset_t - t0) * 1000.0 if motion_onset_t is not None else None,
        "time_to_target_at_onset_ms": time_to_target_at_onset,
        "lock_t_ms": (lock_t - t0) * 1000.0 if lock_t is not None else None,
        "speed_at_lock": speed_at_lock,
        "tier_at_lock": tier_at_lock,
        "time_to_target_at_lock_ms": time_to_target_at_lock,
        "margin_at_lock_ms": margin_at_lock,
        "desired_press_t_ms": (desired_press_time - t0) * 1000.0 if desired_press_time is not None else None,
        "scheduled_fire_possible": scheduled_fire_possible,
        "needed_fallback": needed_fallback,
        "white_zone": white_zone,
    }


def run_full_audit():
    speed_mgr = SpeedProfileManager(auto_bootstrap=False, save_to_disk=False)
    all_files = sorted(
        list((ROOT / "replays").glob("check_*.json")) +
        list((ROOT / "oldreplays").glob("check_*.json"))
    )
    print("================================================================================")
    print("     OFFLINE REPLAY AUDIT: SPEED LOCK TIMING & HIGH SPEED STABILITY")
    print("================================================================================")
    print(f"Total candidate replay files found: {len(all_files)}\n")

    results: List[Dict[str, Any]] = []
    for p in all_files:
        res = audit_single_replay(p, speed_mgr)
        if res is not None:
            results.append(res)

    print(f"Valid skill checks analyzed: {len(results)}\n")

    bucket_defs = [
        ("< 300°/s", lambda s: s < 300.0),
        ("300-400°/s", lambda s: 300.0 <= s < 400.0),
        ("400-500°/s", lambda s: 400.0 <= s < 500.0),
        ("500-600°/s", lambda s: 500.0 <= s < 600.0),
        ("600-750°/s", lambda s: 600.0 <= s < 750.0),
        ("750-900°/s", lambda s: 750.0 <= s < 900.0),
        (">= 900°/s", lambda s: s >= 900.0),
    ]

    print("--------------------------------------------------------------------------------")
    print("SPEED BUCKET DISTRIBUTION & LOCK MARGIN ANALYSIS:")
    print("--------------------------------------------------------------------------------")
    print(f"{'Bucket':12s} | {'N':4s} | {'Locked':6s} | {'NoLock':6s} | {'p50 Margin':10s} | {'p95 Margin':10s} | {'Min Margin':10s} | {'>30ms':5s} | {'10-30':5s} | {'0-10':5s} | {'<0(Late)':8s}")
    print("-------------+------+--------+--------+------------+------------+------------+-------+-------+------+---------")

    bucket_stats = {}

    for bname, bfilter in bucket_defs:
        b_items = [r for r in results if bfilter(r["final_speed"])]
        locked = [r for r in b_items if r["margin_at_lock_ms"] is not None]
        no_lock = [r for r in b_items if r["margin_at_lock_ms"] is None]
        margins = [r["margin_at_lock_ms"] for r in locked]

        c_good = sum(1 for m in margins if m > 30.0)
        c_small = sum(1 for m in margins if 10.0 < m <= 30.0)
        c_crit = sum(1 for m in margins if 0.0 <= m <= 10.0)
        c_late = sum(1 for m in margins if m < 0.0)

        p50_str = f"{np.median(margins):+6.1f}ms" if margins else "   N/A   "
        p95_str = f"{np.percentile(margins, 95):+6.1f}ms" if margins else "   N/A   "
        min_str = f"{min(margins):+6.1f}ms" if margins else "   N/A   "

        print(f"{bname:12s} | {len(b_items):4d} | {len(locked):6d} | {len(no_lock):6d} | {p50_str:10s} | {p95_str:10s} | {min_str:10s} | {c_good:5d} | {c_small:5d} | {c_crit:5d} | {c_late:8d}")

        bucket_stats[bname] = {
            "total": len(b_items),
            "locked": len(locked),
            "no_lock": len(no_lock),
            "margins": margins,
            "good": c_good,
            "small": c_small,
            "crit": c_crit,
            "late": c_late,
        }

    print("--------------------------------------------------------------------------------\n")

    print("--------------------------------------------------------------------------------")
    print("HIGH SPEED CUMULATIVE THRESHOLDS:")
    print("--------------------------------------------------------------------------------")
    thresh_defs = [
        (">= 400°/s", 400.0),
        (">= 500°/s", 500.0),
        (">= 600°/s", 600.0),
        (">= 750°/s", 750.0),
        (">= 900°/s", 900.0),
    ]
    print(f"{'Threshold':10s} | {'N':4s} | {'Locked':6s} | {'NoLock':6s} | {'SchedFire':9s} | {'Fallback':8s} | {'p50 Margin':10s} | {'Min Margin':10s} | {'Late (<0)':9s}")
    print("-----------+------+--------+--------+-----------+----------+------------+------------+----------")
    for tname, min_s in thresh_defs:
        t_items = [r for r in results if r["final_speed"] >= min_s]
        locked = [r for r in t_items if r["margin_at_lock_ms"] is not None]
        no_lock = [r for r in t_items if r["margin_at_lock_ms"] is None]
        margins = [r["margin_at_lock_ms"] for r in locked]
        sched = sum(1 for r in t_items if r["scheduled_fire_possible"])
        fallback = sum(1 for r in t_items if r["needed_fallback"])
        late = sum(1 for m in margins if m < 0.0)

        p50_str = f"{np.median(margins):+6.1f}ms" if margins else "   N/A   "
        min_str = f"{min(margins):+6.1f}ms" if margins else "   N/A   "
        print(f"{tname:10s} | {len(t_items):4d} | {len(locked):6d} | {len(no_lock):6d} | {sched:9d} | {fallback:8d} | {p50_str:10s} | {min_str:10s} | {late:9d}")
    print("--------------------------------------------------------------------------------\n")

    print("--------------------------------------------------------------------------------")
    print("DEEP DIVE: ALL RECORDED CHECKS WITH SPEED >= 500°/s:")
    print("--------------------------------------------------------------------------------")
    high_checks = [r for r in results if r["final_speed"] >= 480.0 or (r["rec_speed"] and r["rec_speed"] >= 480.0)]
    high_checks.sort(key=lambda x: x["final_speed"], reverse=True)

    for c in high_checks:
        fname = c["file"]
        spd = c["final_speed"]
        rec_spd = c["rec_speed"]
        out = c["outcome"]
        ch = f"Chain #{c['chain_count']}" if c["is_chain"] else "Solo"
        lock_str = (
            f"Lock at {c['lock_t_ms']:5.1f}ms (~{c['tier_at_lock']}°/s), Margin: {c['margin_at_lock_ms']:+6.1f}ms"
            if c["margin_at_lock_ms"] is not None
            else "NO LOCK"
        )
        action_str = "SCHEDULED" if c["scheduled_fire_possible"] else "FALLBACK/NO-FIRE"
        zone_str = f"Zone={c['white_zone'].get('start')}°" if c["white_zone"] else "No Zone"
        print(f"  • {fname:42s} | Spd={spd:5.1f}°/s (old_rec={rec_spd if rec_spd else 'N/A'}) | {ch:8s} | {out:11s} | {lock_str:38s} | {action_str:16s} | {zone_str}")

    print("================================================================================\n")


if __name__ == "__main__":
    run_full_audit()
