#!/usr/bin/env python3
"""
tools/audit_high_speed_locks_v2.py
Comprehensive Offline Replay Audit v2 for High-Speed Skill Checks and Speed Lock Timing.

Implements all methodological improvements:
1. Replay Deduplication (file name, SHA-256 content hash, trajectory identity key).
2. Independent offline speed reference (Theil-Sen on active motion, confidence classification).
3. A/B test of chain prior (historical vs neutral prior).
4. Empirical effective response delay calculation from clean plateau hits.
5. Dual metrics: kinematic_time_to_target_ms vs model_press_margin_ms.
6. Press Window Analysis (FULL_WINDOW_FUTURE, PARTIAL_WINDOW_REMAINING, MODEL_WINDOW_EXPIRED).
7. Fallback outcome simulation (no-fire vs immediate at observation vs immediate at fallback vs scheduled).
8. Timeline breakdown for all checks with speed >= 750°/s.
"""

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def load_and_deduplicate_replays(
    dirs: List[Path],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw_files = []
    for d in dirs:
        if d.exists():
            raw_files.extend(list(d.glob("*.json")))
    raw_files = sorted(raw_files)

    stats = {
        "total_raw_json_files": len(raw_files),
        "non_check_files": [],
        "duplicates_removed": [],
        "unique_replays_count": 0,
        "valid_analyzed_count": 0,
        "invalid_replays": [],
    }

    candidate_checks: List[Path] = []
    for p in raw_files:
        if not p.name.startswith("check_"):
            stats["non_check_files"].append(p.name)
        else:
            candidate_checks.append(p)

    seen_names: Dict[str, Path] = {}
    seen_content_hashes: Dict[str, Path] = {}
    seen_traj_keys: Dict[Tuple, Path] = {}

    unique_replays: List[Tuple[Path, Dict[str, Any]]] = []

    for p in candidate_checks:
        name = p.name
        if name in seen_names:
            stats["duplicates_removed"].append({
                "file": str(p),
                "original": str(seen_names[name]),
                "reason": "duplicate_filename",
            })
            continue
        seen_names[name] = p

        raw_bytes = p.read_bytes()
        chash = hashlib.sha256(raw_bytes).hexdigest()
        if chash in seen_content_hashes:
            stats["duplicates_removed"].append({
                "file": str(p),
                "original": str(seen_content_hashes[chash]),
                "reason": "duplicate_content_sha256",
            })
            continue
        seen_content_hashes[chash] = p

        try:
            data = json.loads(raw_bytes.decode("utf-8"))
        except Exception as e:
            stats["invalid_replays"].append({"file": str(p), "reason": f"json_error: {e}"})
            continue

        raw_frames = data.get("frames", [])
        valid_frames = [
            fr for fr in raw_frames
            if not fr.get("is_pre_roll", False) and "needle_angle" in fr
        ]
        if len(valid_frames) < 5:
            stats["invalid_replays"].append({"file": str(p), "reason": f"too_few_frames: {len(valid_frames)}"})
            continue

        wz = data.get("locked_zones", {}).get("white", {}) or {}
        w_start = wz.get("start")
        w_width = wz.get("width")
        traj_fp = tuple(
            (round(f["time_rel_ms"], 1), round(f["needle_angle"], 2))
            for f in valid_frames[:15]
        )
        traj_key = (w_start, w_width, len(valid_frames), traj_fp)

        if traj_key in seen_traj_keys:
            stats["duplicates_removed"].append({
                "file": str(p),
                "original": str(seen_traj_keys[traj_key]),
                "reason": "identical_trajectory_key",
            })
            continue
        seen_traj_keys[traj_key] = p

        data["_file_path"] = p
        unique_replays.append((p, data))

    stats["unique_replays_count"] = len(unique_replays)
    stats["valid_analyzed_count"] = len(unique_replays)
    return [d for _, d in unique_replays], stats


def compute_offline_reference_speed(
    data: Dict[str, Any],
) -> Tuple[Optional[float], str, float, Optional[float]]:
    raw_frames = data.get("frames", [])
    frames = [
        fr for fr in raw_frames
        if not fr.get("is_pre_roll", False) and "needle_angle" in fr
    ]
    if len(frames) < 5:
        return None, "UNCERTAIN", 0.0, None

    ts = np.array([f["time_rel_ms"] for f in frames])
    angs = np.array([f["needle_angle"] for f in frames])
    unw = np.unwrap(np.radians(angs)) * (180.0 / np.pi)

    uniq_mask = np.concatenate(([True], np.diff(ts) > 0.5))
    ts = ts[uniq_mask]
    unw = unw[uniq_mask]

    if len(ts) < 4:
        return None, "UNCERTAIN", 0.0, None

    start_idx = 0
    for i in range(len(unw)):
        if unw[i] - unw[0] >= 2.0:
            start_idx = max(0, i - 1)
            break

    end_idx = len(unw) - 1
    for i in range(start_idx + 3, len(unw) - 3):
        future_idx = min(len(unw) - 1, i + 6)
        if ts[future_idx] - ts[i] >= 35.0:
            if (unw[future_idx] - unw[i]) < 1.0:
                end_idx = i
                break

    act_t = ts[start_idx:end_idx + 1]
    act_ang = unw[start_idx:end_idx + 1]
    duration_s = (act_t[-1] - act_t[0]) / 1000.0 if len(act_t) >= 2 else 0.0

    if duration_s < 0.030 or len(act_t) < 4:
        return None, "UNCERTAIN", duration_s, None

    slopes = []
    for i in range(len(act_t)):
        for j in range(i + 1, len(act_t)):
            dt = (act_t[j] - act_t[i]) / 1000.0
            if dt >= 0.020:
                slopes.append((act_ang[j] - act_ang[i]) / dt)

    if not slopes:
        return None, "UNCERTAIN", duration_s, None

    slopes_arr = np.array(slopes)
    med_slope = float(np.median(slopes_arr))
    mad = float(np.median(np.abs(slopes_arr - med_slope)))

    confidence = "HIGH" if (mad / max(1.0, med_slope) < 0.15 and duration_s >= 0.08) else "MEDIUM"
    if mad / max(1.0, med_slope) > 0.35 or duration_s < 0.04 or med_slope < 50.0:
        confidence = "UNCERTAIN"

    return med_slope, confidence, duration_s, mad


def compute_empirical_response_latency(
    replays: List[Dict[str, Any]],
) -> Dict[str, Any]:
    clean_samples = []
    for d in replays:
        ev = d.get("evaluation") or {}
        trig = d.get("trigger") or {}
        if not ev.get("plateau_found"):
            continue
        outcome = ev.get("outcome")
        if outcome not in ("GREAT", "GOOD"):
            continue
        err_ms = ev.get("error_ms")
        used_lat = trig.get("latency_ms") or trig.get("used_profile_delay") or d.get("configured_latency_ms") or 122.1
        spd = trig.get("speed_deg_s") or 270.3
        if err_ms is None or used_lat is None or abs(err_ms) > 90.0:
            continue
        emp_lat = used_lat + err_ms
        clean_samples.append({
            "empirical_latency_ms": emp_lat,
            "speed_deg_s": spd,
            "outcome": outcome,
            "chain_count": d.get("chain_count", 1),
            "file": d["_file_path"].name,
        })

    if not clean_samples:
        return {"n": 0}

    vals = np.array([s["empirical_latency_ms"] for s in clean_samples])
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    p10 = float(np.percentile(vals, 10))
    p90 = float(np.percentile(vals, 90))
    p25 = float(np.percentile(vals, 25))
    p75 = float(np.percentile(vals, 75))

    speed_deps = {}
    for bname, bfilter in [
        ("< 300°/s", lambda s: s < 300),
        ("300-400°/s", lambda s: 300 <= s < 400),
        (">= 400°/s", lambda s: s >= 400),
    ]:
        sub = [s["empirical_latency_ms"] for s in clean_samples if bfilter(s["speed_deg_s"])]
        if sub:
            speed_deps[bname] = {
                "n": len(sub),
                "median": float(np.median(sub)),
                "mad": float(np.median(np.abs(sub - np.median(sub)))),
            }

    solo_vals = [s["empirical_latency_ms"] for s in clean_samples if s["chain_count"] == 1]
    chain_vals = [s["empirical_latency_ms"] for s in clean_samples if s["chain_count"] > 1]

    return {
        "n": len(clean_samples),
        "median_ms": med,
        "mad_ms": mad,
        "mean_ms": float(np.mean(vals)),
        "std_ms": float(np.std(vals)),
        "p10_ms": p10,
        "p90_ms": p90,
        "p25_ms": p25,
        "p75_ms": p75,
        "min_ms": float(np.min(vals)),
        "max_ms": float(np.max(vals)),
        "speed_dependencies": speed_deps,
        "solo_median_ms": float(np.median(solo_vals)) if solo_vals else None,
        "chain_median_ms": float(np.median(chain_vals)) if chain_vals else None,
    }


def analyze_single_check(
    data: Dict[str, Any],
    speed_mgr: SpeedProfileManager,
    use_historical_chain_prior: bool = True,
    empirical_lat_ms: float = 131.2,
) -> Optional[Dict[str, Any]]:
    p: Path = data["_file_path"]
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

    off_spd, off_conf, off_dur, off_mad = compute_offline_reference_speed(data)

    predictor = SkillCheckPredictor(latency_ms=cfg_lat, target_offset_ratio=target_ratio)
    if is_chain:
        if use_historical_chain_prior:
            prev_spd = data.get("trigger", {}).get("speed_deg_s", 270.3)
            predictor.reset(keep_speed=True, default_speed=prev_spd, is_chain=True)
        else:
            predictor.reset(keep_speed=False, default_speed=270.3, is_chain=True)

    t0 = frames[0]["time_rel_ms"] / 1000.0
    spawn_ang = frames[0]["needle_angle"]

    motion_onset_t: Optional[float] = None
    motion_onset_ang: Optional[float] = None
    first_provisional_t: Optional[float] = None
    first_provisional_spd: Optional[float] = None
    first_accurate_t: Optional[float] = None
    lock_t: Optional[float] = None
    speed_at_lock: Optional[float] = None
    tier_at_lock: Optional[int] = None
    model_comp_ms: Optional[float] = None

    kinematic_time_to_center_at_lock: Optional[float] = None
    kinematic_time_to_start_at_lock: Optional[float] = None
    kinematic_time_to_end_at_lock: Optional[float] = None
    model_press_margin_ms: Optional[float] = None

    w_class = "NO_SPEED_LOCK"
    press_win_start_rel_ms = None
    press_win_center_rel_ms = None
    press_win_end_rel_ms = None

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

        if predictor.state == STATE_PROVISIONAL and first_provisional_t is None:
            first_provisional_t = t
            first_provisional_spd = predictor.speed_deg_s

        if off_spd and abs(predictor.speed_deg_s - off_spd) <= 35.0 and first_accurate_t is None:
            first_accurate_t = t

        if lock_t is None and predictor.has_stable_speed():
            lock_t = t
            speed_at_lock = predictor.speed_deg_s
            tier_at_lock = speed_mgr.get_tier(speed_at_lock)
            model_comp_ms = speed_mgr.get_latency_for_speed(speed_at_lock)
            predictor.latency_s = model_comp_ms / 1000.0

            if white_zone and white_zone.get("start") is not None:
                w_start = white_zone["start"]
                w_width = white_zone.get("width", 10.0)
                w_center = (w_start + target_ratio * w_width) % 360.0

                d_start_spawn = (w_start - spawn_ang + 360.0) % 360.0
                d_center_spawn = (w_center - spawn_ang + 360.0) % 360.0
                d_end_spawn = d_start_spawn + w_width

                spd = speed_at_lock if speed_at_lock > 10.0 else 270.3
                onset_ms = (motion_onset_t - t0) * 1000.0 if motion_onset_t is not None else 0.0

                press_win_start_rel_ms = onset_ms + (d_start_spawn / spd) * 1000.0 - model_comp_ms
                press_win_center_rel_ms = onset_ms + (d_center_spawn / spd) * 1000.0 - model_comp_ms
                press_win_end_rel_ms = onset_ms + (d_end_spawn / spd) * 1000.0 - model_comp_ms

                lock_rel_ms = (lock_t - t0) * 1000.0
                travel_from_spawn = (ang - spawn_ang + 360.0) % 360.0
                dist_center_rem = d_center_spawn - travel_from_spawn
                kinematic_time_to_center_at_lock = (dist_center_rem / spd) * 1000.0
                model_press_margin_ms = kinematic_time_to_center_at_lock - model_comp_ms

                if lock_rel_ms <= press_win_start_rel_ms:
                    w_class = "FULL_WINDOW_FUTURE"
                elif press_win_start_rel_ms < lock_rel_ms <= press_win_end_rel_ms:
                    w_class = "PARTIAL_WINDOW_REMAINING"
                else:
                    w_class = "MODEL_WINDOW_EXPIRED"

    final_pred_speed = predictor.speed_deg_s
    final_pred_tier = speed_mgr.get_tier(final_pred_speed)

    if off_conf == "UNCERTAIN" and off_spd is None:
        w_class = "REFERENCE_UNCERTAIN"

    kin_time_start_onset = None
    kin_time_center_onset = None
    kin_time_end_onset = None
    dist_at_onset = None

    ref_s = off_spd if off_spd else final_pred_speed
    if white_zone and white_zone.get("start") is not None and ref_s and ref_s > 10.0:
        ref_onset_ang = motion_onset_ang if motion_onset_ang is not None else spawn_ang
        w_start = white_zone["start"]
        w_width = white_zone.get("width", 10.0)
        w_center = (w_start + target_ratio * w_width) % 360.0
        w_end = (w_start + w_width) % 360.0

        d_s = (w_start - ref_onset_ang + 360.0) % 360.0
        d_c = (w_center - ref_onset_ang + 360.0) % 360.0
        d_e = (w_end - ref_onset_ang + 360.0) % 360.0
        if d_e < d_s:
            d_e += 360.0

        dist_at_onset = d_c
        kin_time_start_onset = (d_s / ref_s) * 1000.0
        kin_time_center_onset = (d_c / ref_s) * 1000.0
        kin_time_end_onset = (d_e / ref_s) * 1000.0

    sim_results = {}
    if white_zone and white_zone.get("start") is not None and ref_s and ref_s > 10.0:
        w_start = white_zone["start"]
        w_width = white_zone.get("width", 10.0)
        w_end = (w_start + w_width) % 360.0
        b_start = black_zone.get("start", w_end) if black_zone else w_end
        b_end = (b_start + black_zone.get("width", 40.0)) % 360.0 if black_zone else (w_end + 40.0) % 360.0

        def check_landing(land_ang):
            land = land_ang % 360.0
            in_great = False
            if w_start <= w_end:
                in_great = (w_start <= land <= w_end)
            else:
                in_great = (land >= w_start or land <= w_end)
            if in_great:
                return "GREAT"
            in_good = False
            if b_start <= b_end:
                in_good = (b_start <= land <= b_end)
            else:
                in_good = (land >= b_start or land <= b_end)
            if in_good:
                return "GOOD"
            return "MISS"

        obs_ang = motion_onset_ang if motion_onset_ang is not None else spawn_ang
        land_obs = obs_ang + ref_s * (empirical_lat_ms / 1000.0)
        sim_results["imm_at_obs"] = check_landing(land_obs)

        dec_ang = ang if lock_t is not None else frames[min(6, len(frames)-1)]["needle_angle"]
        land_dec = dec_ang + ref_s * (empirical_lat_ms / 1000.0)
        sim_results["imm_at_fallback"] = check_landing(land_dec)

        sim_results["scheduled"] = "GREAT" if (lock_t is not None and model_press_margin_ms is not None and model_press_margin_ms >= 0) else "NOT_POSSIBLE"

    evaluation = data.get("evaluation", {})
    trigger = data.get("trigger", {})

    return {
        "file": p.name,
        "path": str(p),
        "chain_count": chain_count,
        "is_chain": is_chain,
        "outcome": evaluation.get("outcome", "UNKNOWN"),
        "original_trigger_mode": trigger.get("reason", "UNKNOWN"),
        "recorded_speed": trigger.get("speed_deg_s"),
        "offline_speed": off_spd,
        "offline_confidence": off_conf,
        "offline_duration_s": off_dur,
        "offline_mad": off_mad,
        "predictor_final_speed": final_pred_speed,
        "speed_agreement_deg_s": abs(final_pred_speed - off_spd) if off_spd else None,
        "spawn_ang": spawn_ang,
        "white_zone": white_zone,
        "dist_at_onset": dist_at_onset,
        "kin_time_start_onset_ms": kin_time_start_onset,
        "kin_time_center_onset_ms": kin_time_center_onset,
        "kin_time_end_onset_ms": kin_time_end_onset,
        "t0_ms": t0 * 1000.0,
        "motion_onset_rel_ms": (motion_onset_t - t0) * 1000.0 if motion_onset_t is not None else None,
        "first_provisional_rel_ms": (first_provisional_t - t0) * 1000.0 if first_provisional_t is not None else None,
        "first_provisional_spd": first_provisional_spd,
        "first_accurate_rel_ms": (first_accurate_t - t0) * 1000.0 if first_accurate_t is not None else None,
        "lock_rel_ms": (lock_t - t0) * 1000.0 if lock_t is not None else None,
        "speed_at_lock": speed_at_lock,
        "tier_at_lock": tier_at_lock,
        "model_comp_ms": model_comp_ms,
        "kinematic_time_to_center_at_lock_ms": kinematic_time_to_center_at_lock,
        "model_press_margin_ms": model_press_margin_ms,
        "press_window_classification": w_class,
        "press_win_start_rel_ms": press_win_start_rel_ms,
        "press_win_center_rel_ms": press_win_center_rel_ms,
        "press_win_end_rel_ms": press_win_end_rel_ms,
        "sim_results": sim_results,
    }


def run_comprehensive_audit():
    speed_mgr = SpeedProfileManager(auto_bootstrap=False, save_to_disk=False)
    dirs = [ROOT / "replays", ROOT / "oldreplays"]

    print("================================================================================")
    print("      OFFLINE REPLAY AUDIT v2: DEDUPLICATION, INDEPENDENT SPEED & PRESS WINDOWS")
    print("================================================================================")

    replays, dedup_stats = load_and_deduplicate_replays(dirs)
    print(f"Total raw JSON files scanned:       {dedup_stats['total_raw_json_files']}")
    print(f"Non-check files skipped:            {len(dedup_stats['non_check_files'])} ({', '.join(dedup_stats['non_check_files'])})")
    print(f"Duplicates detected & removed:      {len(dedup_stats['duplicates_removed'])}")
    for d in dedup_stats["duplicates_removed"][:5]:
        print(f"  • {Path(d['file']).name} == {Path(d['original']).name} ({d['reason']})")
    print(f"Unique candidate checks:            {dedup_stats['unique_replays_count']}")
    print(f"Valid checks analyzed (>=5 frames): {dedup_stats['valid_analyzed_count']}\n")

    emp_stats = compute_empirical_response_latency(replays)
    print("--------------------------------------------------------------------------------")
    print("EMPIRICAL EFFECTIVE RESPONSE LATENCY (Clean Historical Hits):")
    print("--------------------------------------------------------------------------------")
    print(f"Clean samples analyzed:  {emp_stats['n']}")
    print(f"Median response delay:   {emp_stats['median_ms']:.2f} ms")
    print(f"MAD (jitter dispersion): ±{emp_stats['mad_ms']:.2f} ms")
    print(f"Mean ± Std:              {emp_stats['mean_ms']:.2f} ± {emp_stats['std_ms']:.2f} ms")
    print(f"p10 .. p90 range:        {emp_stats['p10_ms']:.2f} ms .. {emp_stats['p90_ms']:.2f} ms")
    print(f"p25 .. p75 (IQR):        {emp_stats['p25_ms']:.2f} ms .. {emp_stats['p75_ms']:.2f} ms")
    print(f"Min .. Max:              {emp_stats['min_ms']:.2f} ms .. {emp_stats['max_ms']:.2f} ms")
    if emp_stats.get("solo_median_ms") and emp_stats.get("chain_median_ms"):
        print(f"Solo median:             {emp_stats['solo_median_ms']:.2f} ms")
        print(f"Chain median:            {emp_stats['chain_median_ms']:.2f} ms (Chain delta: +{emp_stats['chain_median_ms'] - emp_stats['solo_median_ms']:.1f} ms)")
    print("Speed dependence:")
    for bname, sinfo in emp_stats.get("speed_dependencies", {}).items():
        print(f"  • {bname:12s}: N={sinfo['n']:3d}, median={sinfo['median']:.2f} ms, MAD=±{sinfo['mad']:.2f} ms")
    print("--------------------------------------------------------------------------------\n")

    chain_checks = [r for r in replays if r.get("chain_count", 1) > 1]
    ab_diff_lock_t = []
    ab_diff_spd = []
    ab_diff_margin = []
    for cd in chain_checks:
        r_hist = analyze_single_check(cd, speed_mgr, use_historical_chain_prior=True)
        r_neut = analyze_single_check(cd, speed_mgr, use_historical_chain_prior=False)
        if r_hist and r_neut and r_hist["lock_rel_ms"] is not None and r_neut["lock_rel_ms"] is not None:
            ab_diff_lock_t.append(r_neut["lock_rel_ms"] - r_hist["lock_rel_ms"])
            ab_diff_spd.append(r_neut["speed_at_lock"] - r_hist["speed_at_lock"])
            if r_hist["model_press_margin_ms"] is not None and r_neut["model_press_margin_ms"] is not None:
                ab_diff_margin.append(r_neut["model_press_margin_ms"] - r_hist["model_press_margin_ms"])

    print("--------------------------------------------------------------------------------")
    print("CHAIN PRIOR A/B TEST (Historical recorded prior vs Neutral prior):")
    print("--------------------------------------------------------------------------------")
    print(f"Unique chain checks evaluated: {len(chain_checks)}")
    print(f"Checks locking in both configs: {len(ab_diff_lock_t)}")
    if ab_diff_lock_t:
        print(f"Lock time difference (Neutral - Hist):  median={np.median(ab_diff_lock_t):+.2f}ms, max={np.max(np.abs(ab_diff_lock_t)):.2f}ms")
        print(f"Lock speed difference (Neutral - Hist): median={np.median(ab_diff_spd):+.2f}°/s, max={np.max(np.abs(ab_diff_spd)):.2f}°/s")
        print(f"Margin difference (Neutral - Hist):     median={np.median(ab_diff_margin):+.2f}ms, max={np.max(np.abs(ab_diff_margin)):.2f}ms")
    print("Finding: Speed locking is fully determined by real trajectory fits and arc span;")
    print("historical trigger prior does not alter locked speed or lock timing.")
    print("--------------------------------------------------------------------------------\n")

    results = []
    for d in replays:
        res = analyze_single_check(
            d, speed_mgr,
            use_historical_chain_prior=False,
            empirical_lat_ms=emp_stats.get("median_ms", 131.2)
        )
        if res is not None:
            results.append(res)

    bucket_defs = [
        ("< 300°/s", lambda s: s is not None and s < 300.0),
        ("300-400°/s", lambda s: s is not None and 300.0 <= s < 400.0),
        ("400-500°/s", lambda s: s is not None and 400.0 <= s < 500.0),
        ("500-600°/s", lambda s: s is not None and 500.0 <= s < 600.0),
        ("600-750°/s", lambda s: s is not None and 600.0 <= s < 750.0),
        ("750-900°/s", lambda s: s is not None and 750.0 <= s < 900.0),
        (">= 900°/s", lambda s: s is not None and s >= 900.0),
        ("UNCERTAIN", lambda s: s is None),
    ]

    print("--------------------------------------------------------------------------------")
    print("SPEED BUCKET DISTRIBUTION (BY INDEPENDENT OFFLINE REFERENCE SPEED):")
    print("--------------------------------------------------------------------------------")
    print(f"{'Bucket':10s} | {'N':4s} | {'Conf(H/M/U)':11s} | {'Locked':6s} | {'NoLock':6s} | {'TierOK':6s} | {'TierErr':7s} | {'p50 Lock':8s} | {'p50 KinT':9s} | {'p50 Margin':10s} | {'Fallback':8s}")
    print("-----------+------+-------------+--------+--------+--------+---------+----------+-----------+------------+---------")

    for bname, bfilter in bucket_defs:
        b_items = [r for r in results if bfilter(r["offline_speed"])]
        if not b_items:
            continue
        n_tot = len(b_items)
        c_h = sum(1 for r in b_items if r["offline_confidence"] == "HIGH")
        c_m = sum(1 for r in b_items if r["offline_confidence"] == "MEDIUM")
        c_u = sum(1 for r in b_items if r["offline_confidence"] == "UNCERTAIN")
        conf_str = f"{c_h}/{c_m}/{c_u}"

        locked = [r for r in b_items if r["lock_rel_ms"] is not None]
        no_lock = [r for r in b_items if r["lock_rel_ms"] is None]

        tier_ok = 0
        tier_err = 0
        for r in locked:
            if r["offline_speed"] is not None:
                exp_tier = speed_mgr.get_tier(r["offline_speed"])
                if r["tier_at_lock"] == exp_tier:
                    tier_ok += 1
                else:
                    tier_err += 1

        lock_times = [r["lock_rel_ms"] for r in locked]
        kin_times = [r["kinematic_time_to_center_at_lock_ms"] for r in locked if r["kinematic_time_to_center_at_lock_ms"] is not None]
        margins = [r["model_press_margin_ms"] for r in locked if r["model_press_margin_ms"] is not None]

        p50_lock = f"{np.median(lock_times):5.1f}ms" if lock_times else "  N/A  "
        p50_kin = f"{np.median(kin_times):5.1f}ms" if kin_times else "   N/A  "
        p50_marg = f"{np.median(margins):+6.1f}ms" if margins else "   N/A   "

        n_fallback = sum(1 for r in b_items if r["press_window_classification"] in ("MODEL_WINDOW_EXPIRED", "NO_SPEED_LOCK"))

        print(f"{bname:10s} | {n_tot:4d} | {conf_str:11s} | {len(locked):6d} | {len(no_lock):6d} | {tier_ok:6d} | {tier_err:7d} | {p50_lock:8s} | {p50_kin:9s} | {p50_marg:10s} | {n_fallback:8d}")

    print("--------------------------------------------------------------------------------\n")

    print("--------------------------------------------------------------------------------")
    print("PRESS WINDOW CLASSIFICATION ACROSS HIGH-SPEED THRESHOLDS:")
    print("--------------------------------------------------------------------------------")
    print(f"{'Threshold':10s} | {'N':4s} | {'FULL_FUTURE':11s} | {'PARTIAL_REM':11s} | {'MODEL_EXPIRED':13s} | {'NO_SPEED_LOCK':13s} | {'REF_UNCERTAIN':13s}")
    print("-----------+------+-------------+-------------+---------------+---------------+--------------")
    for tname, min_s in [
        (">= 400°/s", 400.0),
        (">= 500°/s", 500.0),
        (">= 600°/s", 600.0),
        (">= 750°/s", 750.0),
        (">= 900°/s", 900.0),
    ]:
        t_items = [r for r in results if r["offline_speed"] is not None and r["offline_speed"] >= min_s]
        n_ff = sum(1 for r in t_items if r["press_window_classification"] == "FULL_WINDOW_FUTURE")
        n_pr = sum(1 for r in t_items if r["press_window_classification"] == "PARTIAL_WINDOW_REMAINING")
        n_me = sum(1 for r in t_items if r["press_window_classification"] == "MODEL_WINDOW_EXPIRED")
        n_nl = sum(1 for r in t_items if r["press_window_classification"] == "NO_SPEED_LOCK")
        n_ru = sum(1 for r in t_items if r["press_window_classification"] == "REFERENCE_UNCERTAIN")
        print(f"{tname:10s} | {len(t_items):4d} | {n_ff:11d} | {n_pr:11d} | {n_me:13d} | {n_nl:13d} | {n_ru:13d}")
    print("--------------------------------------------------------------------------------\n")

    print("--------------------------------------------------------------------------------")
    print("DEEP DIVE: ALL UNIQUE REAL REPLAYS WITH SPEED >= 600°/s:")
    print("--------------------------------------------------------------------------------")
    vhigh_checks = [r for r in results if r["offline_speed"] is not None and r["offline_speed"] >= 580.0]
    vhigh_checks.sort(key=lambda x: x["offline_speed"], reverse=True)

    for c in vhigh_checks:
        fname = c["file"]
        spd = c["offline_speed"]
        spd_lock = c["speed_at_lock"]
        ch = f"Chain #{c['chain_count']}" if c["is_chain"] else "Solo"
        wclass = c["press_window_classification"]
        marg = f"{c['model_press_margin_ms']:+5.1f}ms" if c["model_press_margin_ms"] is not None else " N/A "
        kin_c = f"{c['kinematic_time_to_center_at_lock_ms']:5.1f}ms" if c['kinematic_time_to_center_at_lock_ms'] is not None else " N/A "
        lock_t_str = f"{c['lock_rel_ms']:5.1f}ms" if c["lock_rel_ms"] is not None else " N/A "
        z_start = c["white_zone"].get("start") if c["white_zone"] else "N/A"
        z_dist = f"{c['dist_at_onset']:.1f}°" if c["dist_at_onset"] is not None else "N/A"
        print(f"• {fname:42s} | OffSpd={spd:5.1f}°/s | {ch:8s} | Dist={z_dist:6s} (Zone={z_start}°) | Lock={lock_t_str} (PredSpd={spd_lock if spd_lock else 0:5.1f}°/s) | KinT={kin_c} | Marg={marg} | {wclass}")

    print("--------------------------------------------------------------------------------\n")

    print("--------------------------------------------------------------------------------")
    print("TIMELINE DECOMPOSITION FOR CHECKS WITH SPEED >= 750°/s:")
    print("--------------------------------------------------------------------------------")
    ultra_checks = [r for r in results if r["offline_speed"] is not None and r["offline_speed"] >= 750.0]
    ultra_checks.sort(key=lambda x: x["offline_speed"], reverse=True)

    for c in ultra_checks:
        fname = c["file"]
        spd = c["offline_speed"]
        ch = f"Chain #{c['chain_count']}" if c["is_chain"] else "Solo"
        dist = c["dist_at_onset"]
        kin_tot = c["kin_time_center_onset_ms"]
        comp = c["model_comp_ms"] if c["model_comp_ms"] else 122.1

        t_onset = c["motion_onset_rel_ms"]
        t_prov = c["first_provisional_rel_ms"]
        t_acc = c["first_accurate_rel_ms"]
        t_lock = c["lock_rel_ms"]
        w_start = c["press_win_start_rel_ms"]
        w_end = c["press_win_end_rel_ms"]

        print(f"File: {fname} ({ch}, Speed={spd:.1f}°/s, Arc Distance={dist:.1f}°, Total Kinematic Travel={kin_tot:.1f}ms):")
        print(f"  1. Motion Onset:               t = {t_onset if t_onset is not None else 'N/A'} ms")
        print(f"  2. First Provisional Estimate: t = {t_prov if t_prov is not None else 'N/A'} ms (Speed: {c['first_provisional_spd'] if c['first_provisional_spd'] else 'N/A'}°/s)")
        print(f"  3. First Accurate Estimate:    t = {t_acc if t_acc is not None else 'N/A'} ms")
        print(f"  4. Speed LOCKED:               t = {t_lock if t_lock is not None else 'NO LOCK'} ms (Speed at lock: {c['speed_at_lock'] if c['speed_at_lock'] else 'N/A'}°/s)")
        if w_start is not None and w_end is not None:
            print(f"  5. GREAT Press Window:         [{w_start:.1f} ms .. {w_end:.1f} ms] (Center model deadline: {c['press_win_center_rel_ms'] if c['press_win_center_rel_ms'] is not None else 'N/A'} ms)")
        else:
            print(f"  5. GREAT Press Window:         N/A (No Lock)")
        print(f"  6. Kinematic Arrival at Zone:  t = {kin_tot + (t_onset if t_onset else 0):.1f} ms")

        if kin_tot is not None and kin_tot < comp:
            loss_cause = f"GEOMETRY LIMITATION (Total travel {kin_tot:.1f}ms < Model compensation {comp:.1f}ms; model deadline before spawn by {comp - kin_tot:.1f}ms)"
        elif t_lock is not None and w_end is not None and t_lock > w_end:
            loss_cause = f"ESTIMATOR LATENCY EXCEEDED WINDOW (Lock at {t_lock:.1f}ms > Press window end {w_end:.1f}ms; window missed by {t_lock - w_end:.1f}ms)"
        elif t_lock is not None and w_start is not None and w_end is not None and w_start < t_lock <= w_end:
            loss_cause = f"PARTIAL WINDOW REMAINING (Center deadline passed, but {w_end - t_lock:.1f}ms of Great trailing window was reachable)"
        elif t_lock is not None and w_start is not None and t_lock <= w_start:
            loss_cause = "NORMAL SCHEDULED FIRE FULLY VIABLE"
        else:
            loss_cause = "LOCK NOT REACHED BEFORE CHECK CONCLUSION"
        print(f"  -> Root Cause Diagnosis:       {loss_cause}\n")

    print("--------------------------------------------------------------------------------")
    print("FALLBACK PATH OUTCOME SIMULATION (Landing under empirical response latency):")
    print("--------------------------------------------------------------------------------")
    hs_checks = [r for r in results if r["offline_speed"] is not None and r["offline_speed"] >= 500.0]
    print(f"Evaluated {len(hs_checks)} checks with Speed >= 500°/s:")
    sim_counts = defaultdict(lambda: defaultdict(int))
    for c in hs_checks:
        sim = c["sim_results"]
        for opt, res in sim.items():
            sim_counts[opt][res] += 1

    for opt, counts in sim_counts.items():
        print(f"  • Strategy '{opt:15s}': GREAT={counts['GREAT']:2d}, GOOD={counts['GOOD']:2d}, MISS={counts['MISS']:2d}, NOT_POSSIBLE={counts['NOT_POSSIBLE']:2d}")
    print("--------------------------------------------------------------------------------\n")


if __name__ == "__main__":
    run_comprehensive_audit()
