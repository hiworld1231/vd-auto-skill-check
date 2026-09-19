#!/usr/bin/env python3
"""
tools/clean_response_audit.py
Rigorous empirical audit of skill check telemetry:
- Clean Response Dataset filtering with step-by-step funnel
- Method A, Method B, Method C empirical latency analysis
- 297 Tier Error dissection and 25°/s bin resolution analysis
- CURRENT vs ADAPTIVE_5PCT Pareto evaluation across all unique replays
"""

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.predictor import (
    SkillCheckPredictor,
    STATE_PROVISIONAL,
    STATE_LOCKED,
    STATE_COMMITTED,
)
from core.learner import SpeedProfileManager
from tools.audit_high_speed_locks_v2 import (
    load_and_deduplicate_replays,
    compute_offline_reference_speed,
)


class EvaluatedPredictor(SkillCheckPredictor):
    def __init__(self, mode="CURRENT", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode

    def has_stable_speed(self) -> bool:
        if self.state in (STATE_LOCKED, STATE_COMMITTED):
            return True
        min_hist = 3 if self.is_chain else 5
        if not self.has_adapted or len(self.history) < min_hist:
            return False
        span = self.history[-1][0] - self.history[0][0]
        min_span = 0.025 if self.is_chain else 0.040
        if span < min_span:
            return False
        start_angle = self.history[0][1]
        curr_angle = self.history[-1][1]
        arc_travel = (curr_angle - start_angle + 360.0) % 360.0
        min_arc = 15.0 if self.is_chain else 20.0
        if arc_travel < min_arc:
            return False
        min_fits = 2 if self.is_chain else 3
        if len(self.speed_fits) < min_fits:
            return False
        last_fits = [f[1] for f in list(self.speed_fits)[-min_fits:]]
        spread = max(last_fits) - min(last_fits)

        max_allowed = 25.0
        if self.mode == "ADAPTIVE_5PCT":
            max_allowed = max(25.0, 0.05 * self.speed_deg_s)

        if spread > max_allowed:
            return False
        self.state = STATE_LOCKED
        self.locked_speed = self.speed_deg_s
        return True


def run_clean_audit():
    dirs = [ROOT / "replays", ROOT / "oldreplays"]
    replays, dedup_stats = load_and_deduplicate_replays(dirs)
    speed_mgr = SpeedProfileManager(auto_bootstrap=False, save_to_disk=False)

    print("================================================================================")
    print("      RIGOROUS EMPIRICAL AUDIT: CLEAN RESPONSE DATASET & POLICY PARETO")
    print("================================================================================")
    print(f"Total raw JSON files scanned:       {dedup_stats['total_raw_json_files']}")
    print(f"Unique deduplicated replays:        {len(replays)}")

    # 1. Funnel & Trigger reason separation
    by_reason = Counter()
    for r in replays:
        trig = r.get("trigger") or {}
        reas = str(trig.get("reason"))
        by_reason[reas] += 1

    print("\n1. TRIGGER REASON DISTRIBUTION:")
    for k, v in by_reason.most_common():
        print(f"   • {k:22s}: {v:4d} checks")

    # Funnel for scheduled checks
    sched_replays = [r for r in replays if (r.get("trigger") or {}).get("reason") == "СПИН-ТАЙМЕР"]
    f0_total = len(replays)
    f1_sched = len(sched_replays)

    f2_outcome = []
    for r in sched_replays:
        ev = r.get("evaluation") or {}
        if ev.get("outcome") in ("GREAT", "GOOD"):
            f2_outcome.append(r)

    f3_plateau = []
    for r in f2_outcome:
        ev = r.get("evaluation") or {}
        if ev.get("plateau_found"):
            f3_plateau.append(r)

    f4_angles = []
    for r in f3_plateau:
        ev = r.get("evaluation") or {}
        ha = ev.get("hit_angle")
        ta = ev.get("target_angle")
        if ha is not None and ta is not None and not np.isnan(ha) and not np.isnan(ta):
            f4_angles.append(r)

    f5_dur = []
    off_speeds = {}
    for r in f4_angles:
        off_spd, off_conf, off_dur, off_mad = compute_offline_reference_speed(r)
        off_speeds[r["_file_path"].name] = (off_spd, off_conf, off_dur, off_mad)
        if off_dur >= 0.080 and off_spd is not None:
            f5_dur.append(r)

    f6_agree = []
    for r in f5_dur:
        off_spd = off_speeds[r["_file_path"].name][0]
        trig = r.get("trigger") or {}
        lock_spd = trig.get("measured_speed_at_lock")
        trig_spd = trig.get("speed_deg_s")
        ref_bot_spd = lock_spd if lock_spd is not None else trig_spd
        if ref_bot_spd is not None and abs(ref_bot_spd - off_spd) <= 35.0:
            f6_agree.append(r)

    print("\n2. CLEAN RESPONSE DATASET ATTRITION FUNNEL:")
    print("   -----------------------------------------------------------------------------")
    print(f"   Step 0: All deduplicated replays                  | N = {f0_total:4d}")
    print(f"   Step 1: Filter normal scheduled (СПИН-ТАЙМЕР)     | N = {f1_sched:4d} (dropped {f0_total - f1_sched:3d} non-scheduled / not fired)")
    print(f"   Step 2: Filter outcome GREAT or GOOD              | N = {len(f2_outcome):4d} (dropped {f1_sched - len(f2_outcome):3d} MISS/ABORTED/UNKNOWN)")
    print(f"   Step 3: Require plateau_found == True             | N = {len(f3_plateau):4d} (dropped {len(f2_outcome) - len(f3_plateau):3d} no plateau)")
    print(f"   Step 4: Valid numeric hit & target angles         | N = {len(f4_angles):4d} (dropped {len(f3_plateau) - len(f4_angles):3d} invalid angles)")
    print(f"   Step 5: Active trajectory duration >= 80ms        | N = {len(f5_dur):4d} (dropped {len(f4_angles) - len(f5_dur):3d} short checks)")
    print(f"   Step 6: Tracker/offline speed agreement <= 35°/s  | N = {len(f6_agree):4d} (dropped {len(f5_dur) - len(f6_agree):3d} speed disagreement)")
    print("   -----------------------------------------------------------------------------")
    print(f"   FINAL CLEAN RESPONSE DATASET SIZE:                 N = {len(f6_agree)}")

    # Split into Modern vs Legacy
    modern_clean = []
    modern_jitter = []
    legacy_clean = []
    for r in f6_agree:
        trig = r.get("trigger") or {}
        serr = trig.get("scheduler_error_ms")
        if serr is None:
            legacy_clean.append(r)
        else:
            if abs(serr) <= 2.0:
                modern_clean.append(r)
            else:
                modern_jitter.append(r)

    print("\n3. CLEAN DATASET TELEMETRY SUBSETS:")
    print(f"   • Modern Telemetry (|scheduler_error_ms| <= 2.0ms): N = {len(modern_clean)}")
    print(f"   • Modern Telemetry (|scheduler_error_ms| > 2.0ms):  N = {len(modern_jitter)}")
    print(f"   • Legacy Telemetry (scheduler_error_ms is None):   N = {len(legacy_clean)}")

    # Latency calculation helper
    def eval_latencies(sample_list):
        method_a = []
        method_b = []
        method_c = []
        speed_list = []
        chain_counts = []

        for r in sample_list:
            ev = r.get("evaluation") or {}
            trig = r.get("trigger") or {}
            off_spd = off_speeds[r["_file_path"].name][0]
            bot_spd = trig.get("speed_deg_s") or 270.3
            used_lat = trig.get("latency_ms") or trig.get("used_profile_delay") or 122.1
            err_ms = ev.get("error_ms", 0.0)
            err_deg = ev.get("error_deg", 0.0)
            hit_ang = ev.get("hit_angle")
            ch_cnt = r.get("chain_count", 1)

            # Method A
            lat_a = used_lat + err_ms
            method_a.append(lat_a)

            # Method B
            lat_b = used_lat + (err_deg / off_spd * 1000.0)
            method_b.append(lat_b)

            speed_list.append(off_spd)
            chain_counts.append(ch_cnt)

            # Method C (if modern with actual_press_time and last_frame_age_ms_at_fire)
            act_press = trig.get("actual_press_time")
            last_age = trig.get("last_frame_age_ms_at_fire")
            est_ang = trig.get("est_angle_at_trigger")

            if act_press is not None and last_age is not None and est_ang is not None and bot_spd > 0:
                raw_frames = [f for f in r["frames"] if not f.get("is_pre_roll") and "needle_angle" in f]
                if len(raw_frames) >= 5:
                    ts = np.array([f["time_rel_ms"] for f in raw_frames])
                    angs = np.array([f["needle_angle"] for f in raw_frames])
                    unw = np.unwrap(np.radians(angs)) * (180.0 / np.pi)

                    exp_last_ang = (est_ang - bot_spd * (last_age / 1000.0)) % 360.0
                    cand = [i for i, f in enumerate(raw_frames) if abs((f["needle_angle"] - exp_last_ang + 180.0) % 360.0 - 180.0) < 0.15]
                    if cand:
                        idx = cand[-1]
                        t_press_rel = ts[idx] + last_age
                        ang_dispatch = float(np.interp(t_press_rel, ts, unw)) % 360.0
                        delta_theta = (hit_ang - ang_dispatch + 360.0) % 360.0
                        if delta_theta > 180.0:
                            delta_theta -= 360.0
                        lat_c = (delta_theta / off_spd) * 1000.0
                        method_c.append({
                            "file": r["_file_path"].name,
                            "lat_a": lat_a,
                            "lat_b": lat_b,
                            "lat_c": lat_c,
                            "speed": off_spd,
                            "chain": ch_cnt,
                            "sched_err": trig.get("scheduler_error_ms"),
                        })

        def calc_stats(arr):
            a = np.array(arr)
            med = float(np.median(a))
            return {
                "n": len(a),
                "median": med,
                "mad": float(np.median(np.abs(a - med))),
                "mean": float(np.mean(a)),
                "std": float(np.std(a)),
                "p10": float(np.percentile(a, 10)),
                "p25": float(np.percentile(a, 25)),
                "p75": float(np.percentile(a, 75)),
                "p90": float(np.percentile(a, 90)),
                "min": float(np.min(a)),
                "max": float(np.max(a)),
            }

        return {
            "stats_a": calc_stats(method_a),
            "stats_b": calc_stats(method_b),
            "method_c_items": method_c,
            "speeds": speed_list,
            "chains": chain_counts,
            "method_a": method_a,
            "method_b": method_b,
        }

    modern_res = eval_latencies(modern_clean)
    all_clean_res = eval_latencies(modern_clean + legacy_clean)

    print("\n4. EMPIRICAL EFFECTIVE RESPONSE LATENCY ESTIMATION:")
    print("   =============================================================================")
    print("   [SUBSET 1: MODERN TELEMETRY ONLY (|scheduler_error_ms| <= 2.0ms, N=40)]")
    print("   -----------------------------------------------------------------------------")
    st_a = modern_res["stats_a"]
    st_b = modern_res["stats_b"]
    c_arr = np.array([x["lat_c"] for x in modern_res["method_c_items"]])
    c_med = float(np.median(c_arr))
    c_mad = float(np.median(np.abs(c_arr - c_med)))
    c_mean = float(np.mean(c_arr))
    c_std = float(np.std(c_arr))
    c_p10 = float(np.percentile(c_arr, 10))
    c_p90 = float(np.percentile(c_arr, 90))
    c_p25 = float(np.percentile(c_arr, 25))
    c_p75 = float(np.percentile(c_arr, 75))

    print(f"   Method A (Historical delay+error_ms):  N={st_a['n']:2d} | Median={st_a['median']:6.2f}ms | MAD=±{st_a['mad']:5.2f}ms | Mean={st_a['mean']:6.2f}±{st_a['std']:5.2f}ms | p10..p90=[{st_a['p10']:.1f}..{st_a['p90']:.1f}]")
    print(f"   Method B (Offline-speed corrected):    N={st_b['n']:2d} | Median={st_b['median']:6.2f}ms | MAD=±{st_b['mad']:5.2f}ms | Mean={st_b['mean']:6.2f}±{st_b['std']:5.2f}ms | p10..p90=[{st_b['p10']:.1f}..{st_b['p90']:.1f}]")
    print(f"   Method C (Dispatch-time reconstruction):N={len(c_arr):2d} | Median={c_med:6.2f}ms | MAD=±{c_mad:5.2f}ms | Mean={c_mean:6.2f}±{c_std:5.2f}ms | p10..p90=[{c_p10:.1f}..{c_p90:.1f}]")

    diff_ba = np.array([x["lat_b"] - x["lat_a"] for x in modern_res["method_c_items"]])
    diff_cb = np.array([x["lat_c"] - x["lat_b"] for x in modern_res["method_c_items"]])
    diff_ca = np.array([x["lat_c"] - x["lat_a"] for x in modern_res["method_c_items"]])
    print("\n   PAIRWISE METHOD BIAS (N=40):")
    print(f"   • Method B - Method A: Median bias = {np.median(diff_ba):+5.2f}ms, Mean bias = {np.mean(diff_ba):+5.2f}ms, Max |diff| = {np.max(np.abs(diff_ba)):.2f}ms")
    print(f"   • Method C - Method B: Median bias = {np.median(diff_cb):+5.2f}ms, Mean bias = {np.mean(diff_cb):+5.2f}ms, Max |diff| = {np.max(np.abs(diff_cb)):.2f}ms")
    print(f"   • Method C - Method A: Median bias = {np.median(diff_ca):+5.2f}ms, Mean bias = {np.mean(diff_ca):+5.2f}ms, Max |diff| = {np.max(np.abs(diff_ca)):.2f}ms")

    # Solo vs Chain in Modern
    c_solo = np.array([x["lat_c"] for x in modern_res["method_c_items"] if x["chain"] == 1])
    c_chain = np.array([x["lat_c"] for x in modern_res["method_c_items"] if x["chain"] > 1])
    print("\n   SOLO VS CHAIN BREAKDOWN (Method C, Modern Telemetry):")
    if len(c_solo) > 0:
        print(f"   • Solo checks (N={len(c_solo)}):  Median = {np.median(c_solo):6.2f}ms, MAD = ±{np.median(np.abs(c_solo - np.median(c_solo))):.2f}ms")
    if len(c_chain) > 0:
        print(f"   • Chain checks (N={len(c_chain)}): Median = {np.median(c_chain):6.2f}ms, MAD = ±{np.median(np.abs(c_chain - np.median(c_chain))):.2f}ms")
        if len(c_solo) > 0:
            print(f"   • Chain Kinematic Delta: +{np.median(c_chain) - np.median(c_solo):.2f}ms")

    print("\n   SPEED TIER BREAKDOWN (Method C, Modern Telemetry):")
    for s_name, s_filter in [
        ("< 300°/s", lambda s: s < 300.0),
        ("300 - 400°/s", lambda s: 300.0 <= s < 400.0),
        (">= 400°/s", lambda s: s >= 400.0),
    ]:
        sub = np.array([x["lat_c"] for x in modern_res["method_c_items"] if s_filter(x["speed"])])
        if len(sub) > 0:
            s_med = float(np.median(sub))
            s_mad = float(np.median(np.abs(sub - s_med)))
            print(f"   • {s_name:12s} (N={len(sub):2d}): Median = {s_med:6.2f}ms, MAD = ±{s_mad:.2f}ms")

    print("\n   -----------------------------------------------------------------------------")
    print(f"   [SUBSET 2: SENSITIVITY CHECK - MODERN + LEGACY (N={len(modern_clean + legacy_clean)})]")
    print("   -----------------------------------------------------------------------------")
    st_all_a = all_clean_res["stats_a"]
    st_all_b = all_clean_res["stats_b"]
    print(f"   Method A (All clean): N={st_all_a['n']:3d} | Median={st_all_a['median']:6.2f}ms | MAD=±{st_all_a['mad']:5.2f}ms | Mean={st_all_a['mean']:6.2f}±{st_all_a['std']:5.2f}ms | p10..p90=[{st_all_a['p10']:.1f}..{st_all_a['p90']:.1f}]")
    print(f"   Method B (All clean): N={st_all_b['n']:3d} | Median={st_all_b['median']:6.2f}ms | MAD=±{st_all_b['mad']:5.2f}ms | Mean={st_all_b['mean']:6.2f}±{st_all_b['std']:5.2f}ms | p10..p90=[{st_all_b['p10']:.1f}..{st_all_b['p90']:.1f}]")
    all_diff_ba = np.array([b - a for a, b in zip(all_clean_res["method_a"], all_clean_res["method_b"])])
    print(f"   Pairwise Method B - Method A across all clean: Median bias = {np.median(all_diff_ba):+5.2f}ms, Mean bias = {np.mean(all_diff_ba):+5.2f}ms")

    # 5. Immediate Statistics
    imm_replays = [r for r in replays if (r.get("trigger") or {}).get("reason") == "IMMEDIATE"]
    imm_outcomes = Counter((r.get("evaluation") or {}).get("outcome", "UNKNOWN") for r in imm_replays)
    imm_errs_deg = [(r.get("evaluation") or {}).get("error_deg") for r in imm_replays if (r.get("evaluation") or {}).get("error_deg") is not None]
    imm_errs_ms = [(r.get("evaluation") or {}).get("error_ms") for r in imm_replays if (r.get("evaluation") or {}).get("error_ms") is not None]
    imm_speeds = [(r.get("trigger") or {}).get("speed_deg_s") for r in imm_replays if (r.get("trigger") or {}).get("speed_deg_s") is not None]

    print("\n5. IMMEDIATE FIRE PATH TELEMETRY (ISOLATED FROM CALIBRATION):")
    print("   -----------------------------------------------------------------------------")
    print(f"   Total IMMEDIATE checks: {len(imm_replays)}")
    print(f"   Outcome breakdown:      {dict(imm_outcomes)}")
    conf_imm = imm_outcomes.get("GREAT", 0) + imm_outcomes.get("GOOD", 0)
    conf_tot = conf_imm + imm_outcomes.get("MISS", 0)
    print(f"   Success rate (GREAT/GOOD on confirmed): {conf_imm}/{conf_tot} ({conf_imm/conf_tot*100:.1f}%)")
    if imm_errs_deg:
        print(f"   Angular Error:          Median = {np.median(imm_errs_deg):+5.2f}°, Mean ± Std = {np.mean(imm_errs_deg):+5.2f} ± {np.std(imm_errs_deg):5.2f}°")
    if imm_errs_ms:
        print(f"   Temporal Error:         Median = {np.median(imm_errs_ms):+5.2f}ms, Mean ± Std = {np.mean(imm_errs_ms):+5.2f} ± {np.std(imm_errs_ms):5.2f}ms")
    if imm_speeds:
        print(f"   Speed range:            Median = {np.median(imm_speeds):6.1f}°/s, Min = {np.min(imm_speeds):5.1f}°/s, Max = {np.max(imm_speeds):6.1f}°/s")

    # 6. Tier Error Dissection
    print("\n6. DISSECTION OF THE 298 TIER MISMATCHES:")
    print("   -----------------------------------------------------------------------------")
    def sim_check(data, mode):
        raw_frames = data.get("frames", [])
        frames = [f for f in raw_frames if not f.get("is_pre_roll") and "needle_angle" in f]
        if len(frames) < 5:
            return None
        chain_count = data.get("chain_count", 1)
        is_chain = chain_count > 1
        wz = data.get("locked_zones", {}).get("white")
        bz = data.get("locked_zones", {}).get("black")
        if not wz:
            for fr in frames:
                w = fr.get("white_zone")
                if w and w.get("start") is not None:
                    wz = w
                    bz = fr.get("black_zone")
                    break
        t0 = frames[0]["time_rel_ms"] / 1000.0
        spawn_ang = frames[0]["needle_angle"]
        target_ratio = data.get("target_ratio", 0.50)

        pred = EvaluatedPredictor(mode=mode, latency_ms=122.1, target_offset_ratio=target_ratio)
        if is_chain:
            pred.reset(keep_speed=False, default_speed=270.3, is_chain=True)

        lock_t = None
        lock_spd = None
        w_class = "NO_SPEED_LOCK"

        for fr in frames:
            t = fr["time_rel_ms"] / 1000.0
            ang = fr["needle_angle"]
            strn = fr.get("needle_strength", 75.0)
            valid = pred.update(t, ang, strn, wz, bz)
            if not valid:
                continue
            if lock_t is None and pred.has_stable_speed():
                lock_t = t
                lock_spd = pred.speed_deg_s
                model_comp_ms = speed_mgr.get_latency_for_speed(lock_spd)
                if wz and wz.get("start") is not None:
                    w_start = wz["start"]
                    w_width = wz.get("width", 10.0)
                    w_center = (w_start + target_ratio * w_width) % 360.0
                    d_start = (w_start - spawn_ang + 360.0) % 360.0
                    d_end = d_start + w_width
                    spd = max(10.0, lock_spd)
                    p_win_start = (d_start / spd) * 1000.0 - model_comp_ms
                    p_win_end = (d_end / spd) * 1000.0 - model_comp_ms
                    lock_rel_ms = (lock_t - t0) * 1000.0
                    if lock_rel_ms <= p_win_start:
                        w_class = "FULL_WINDOW_FUTURE"
                    elif p_win_start < lock_rel_ms <= p_win_end:
                        w_class = "PARTIAL_WINDOW_REMAINING"
                    else:
                        w_class = "MODEL_WINDOW_EXPIRED"

        off_spd, off_conf, off_dur, off_mad = compute_offline_reference_speed(data)
        lock_rel_ms = (lock_t - t0) * 1000.0 if lock_t is not None else None
        tier_at_lock = speed_mgr.get_tier(lock_spd) if lock_spd is not None else None
        exp_tier = speed_mgr.get_tier(off_spd) if off_spd is not None else None

        return {
            "file": data["_file_path"].name,
            "lock_rel_ms": lock_rel_ms,
            "lock_spd": lock_spd,
            "tier_at_lock": tier_at_lock,
            "exp_tier": exp_tier,
            "offline_speed": off_spd,
            "offline_conf": off_conf,
            "w_class": w_class,
        }

    cur_results = [sim_check(r, "CURRENT") for r in replays]
    cur_valid = [r for r in cur_results if r is not None and r["lock_rel_ms"] is not None and r["offline_speed"] is not None]

    tier_matches = [r for r in cur_valid if r["tier_at_lock"] == r["exp_tier"]]
    tier_mismatches = [r for r in cur_valid if r["tier_at_lock"] != r["exp_tier"]]

    diff_arr = np.array([abs(r["lock_spd"] - r["offline_speed"]) for r in tier_mismatches])
    signed_diff_arr = np.array([r["lock_spd"] - r["offline_speed"] for r in tier_mismatches])
    bin_dists = np.array([abs(round(r["lock_spd"] / 25.0) - round(r["offline_speed"] / 25.0)) for r in tier_mismatches])

    print(f"   Total checks with locked & offline speed: {len(cur_valid)}")
    print(f"   • Exact Tier Matches:                     {len(tier_matches)} ({len(tier_matches)/len(cur_valid)*100:.1f}%)")
    print(f"   • Tier Mismatches:                        {len(tier_mismatches)} ({len(tier_mismatches)/len(cur_valid)*100:.1f}%)")
    print(f"   Speed Error in Mismatches: Median = {np.median(diff_arr):.2f}°/s, Mean ± Std = {np.mean(diff_arr):.2f} ± {np.std(diff_arr):.2f}°/s")
    print(f"   Signed Error in Mismatches: Median = {np.median(signed_diff_arr):+.2f}°/s, Mean = {np.mean(signed_diff_arr):+.2f}°/s")

    n_le_125 = sum(1 for d in diff_arr if d <= 12.5)
    n_le_25 = sum(1 for d in diff_arr if d <= 25.0)
    n_le_35 = sum(1 for d in diff_arr if d <= 35.0)
    n_gt_25 = sum(1 for d in diff_arr if d > 25.0)
    n_gt_35 = sum(1 for d in diff_arr if d > 35.0)
    n_gt_50 = sum(1 for d in diff_arr if d > 50.0)

    print("\n   ERROR THRESHOLDS IN TIER MISMATCHES:")
    print(f"   • Speed error <= 12.5°/s (< 0.5 bin, border roundoff): {n_le_125:3d} ({n_le_125/len(tier_mismatches)*100:.1f}%)")
    print(f"   • Speed error <= 25.0°/s (within 1 bin width):          {n_le_25:3d} ({n_le_25/len(tier_mismatches)*100:.1f}%)")
    print(f"   • Speed error <= 35.0°/s (acceptable tolerance):       {n_le_35:3d} ({n_le_35/len(tier_mismatches)*100:.1f}%)")
    print(f"   • Speed error > 25.0°/s:                                {n_gt_25:3d} ({n_gt_25/len(tier_mismatches)*100:.1f}%)")
    print(f"   • Speed error > 35.0°/s:                                {n_gt_35:3d} ({n_gt_35/len(tier_mismatches)*100:.1f}%)")
    print(f"   • Speed error > 50.0°/s:                                {n_gt_50:3d} ({n_gt_50/len(tier_mismatches)*100:.1f}%)")

    dist_counts = Counter(bin_dists)
    print("\n   TIER BIN DISTANCE DISTRIBUTION IN MISMATCHES:")
    print(f"   • Distance = 1 bin (adjacent tier ±25°/s):  {dist_counts[1]:3d} ({dist_counts[1]/len(tier_mismatches)*100:.1f}%)")
    print(f"   • Distance = 2 bins (±50°/s):               {dist_counts[2]:3d} ({dist_counts[2]/len(tier_mismatches)*100:.1f}%)")
    print(f"   • Distance >= 3 bins (> 50°/s):             {sum(v for k, v in dist_counts.items() if k >= 3):3d} ({sum(v for k, v in dist_counts.items() if k >= 3)/len(tier_mismatches)*100:.1f}%)")

    # 7. CURRENT vs ADAPTIVE_5PCT Pareto Evaluation
    adp_results = [sim_check(r, "ADAPTIVE_5PCT") for r in replays]

    def compute_policy_metrics(res_list):
        valid = [r for r in res_list if r is not None]
        locked = [r for r in valid if r["lock_rel_ms"] is not None]
        no_lock = [r for r in valid if r["lock_rel_ms"] is None]
        lock_times = [r["lock_rel_ms"] for r in locked]

        locked_with_off = [r for r in locked if r["offline_speed"] is not None]
        speed_errs = [abs(r["lock_spd"] - r["offline_speed"]) for r in locked_with_off]
        bin_d = [abs(round(r["lock_spd"] / 25.0) - round(r["offline_speed"] / 25.0)) for r in locked_with_off]

        err_gt_25 = sum(1 for e in speed_errs if e > 25.0)
        err_gt_35 = sum(1 for e in speed_errs if e > 35.0)
        err_gt_50 = sum(1 for e in speed_errs if e > 50.0)
        dist_ge_2 = sum(1 for d in bin_d if d >= 2)

        false_high_speed = sum(
            1 for r in locked_with_off
            if r["lock_spd"] >= 500.0 and r["offline_speed"] < 400.0
        )

        expired = sum(1 for r in locked if r["w_class"] == "MODEL_WINDOW_EXPIRED")
        full_future = sum(1 for r in locked if r["w_class"] == "FULL_WINDOW_FUTURE")
        partial_rem = sum(1 for r in locked if r["w_class"] == "PARTIAL_WINDOW_REMAINING")

        return {
            "total_checks": len(valid),
            "total_locked": len(locked),
            "no_lock": len(no_lock),
            "p50_lock_t": float(np.median(lock_times)) if lock_times else 0.0,
            "p95_lock_t": float(np.percentile(lock_times, 95)) if lock_times else 0.0,
            "spd_err_med": float(np.median(speed_errs)) if speed_errs else 0.0,
            "spd_err_p90": float(np.percentile(speed_errs, 90)) if speed_errs else 0.0,
            "err_gt_25": err_gt_25,
            "err_gt_35": err_gt_35,
            "err_gt_50": err_gt_50,
            "tier_dist_ge_2": dist_ge_2,
            "false_high_speed": false_high_speed,
            "window_expired": expired,
            "window_future": full_future,
            "window_partial": partial_rem,
        }

    m_cur = compute_policy_metrics(cur_results)
    m_adp = compute_policy_metrics(adp_results)

    print("\n7. POLICY PARETO COMPARISON (ALL DEDUPLICATED REPLAYS):")
    print("   -----------------------------------------------------------------------------")
    print(f"   Metric                               | CURRENT        | ADAPTIVE_5PCT  | Delta")
    print("   -------------------------------------+----------------+----------------+---------")
    print(f"   Total checks evaluated               | {m_cur['total_checks']:14d} | {m_adp['total_checks']:14d} | {m_adp['total_checks']-m_cur['total_checks']:+d}")
    print(f"   Total speed locks achieved           | {m_cur['total_locked']:14d} | {m_adp['total_locked']:14d} | {m_adp['total_locked']-m_cur['total_locked']:+d}")
    print(f"   No-lock count                        | {m_cur['no_lock']:14d} | {m_adp['no_lock']:14d} | {m_adp['no_lock']-m_cur['no_lock']:+d}")
    print(f"   Lock timing p50                      | {m_cur['p50_lock_t']:12.2f}ms | {m_adp['p50_lock_t']:12.2f}ms | {m_adp['p50_lock_t']-m_cur['p50_lock_t']:+6.2f}ms")
    print(f"   Lock timing p95                      | {m_cur['p95_lock_t']:12.2f}ms | {m_adp['p95_lock_t']:12.2f}ms | {m_adp['p95_lock_t']-m_cur['p95_lock_t']:+6.2f}ms")
    print(f"   Speed error median (vs offline)      | {m_cur['spd_err_med']:12.2f}°/s| {m_adp['spd_err_med']:12.2f}°/s| {m_adp['spd_err_med']-m_cur['spd_err_med']:+6.2f}°/s")
    print(f"   Speed error p90                      | {m_cur['spd_err_p90']:12.2f}°/s| {m_adp['spd_err_p90']:12.2f}°/s| {m_adp['spd_err_p90']-m_cur['spd_err_p90']:+6.2f}°/s")
    print(f"   Errors > 25°/s                       | {m_cur['err_gt_25']:14d} | {m_adp['err_gt_25']:14d} | {m_adp['err_gt_25']-m_cur['err_gt_25']:+d}")
    print(f"   Errors > 35°/s                       | {m_cur['err_gt_35']:14d} | {m_adp['err_gt_35']:14d} | {m_adp['err_gt_35']-m_cur['err_gt_35']:+d}")
    print(f"   Errors > 50°/s                       | {m_cur['err_gt_50']:14d} | {m_adp['err_gt_50']:14d} | {m_adp['err_gt_50']-m_cur['err_gt_50']:+d}")
    print(f"   Tier distance >= 2 bins              | {m_cur['tier_dist_ge_2']:14d} | {m_adp['tier_dist_ge_2']:14d} | {m_adp['tier_dist_ge_2']-m_cur['tier_dist_ge_2']:+d}")
    print(f"   False high-speed locks (>500 when <400)| {m_cur['false_high_speed']:14d} | {m_adp['false_high_speed']:14d} | {m_adp['false_high_speed']-m_cur['false_high_speed']:+d}")
    print(f"   MODEL_WINDOW_EXPIRED count           | {m_cur['window_expired']:14d} | {m_adp['window_expired']:14d} | {m_adp['window_expired']-m_cur['window_expired']:+d}")
    print(f"   Usable press windows (Future+Partial)| {m_cur['window_future']+m_cur['window_partial']:14d} | {m_adp['window_future']+m_adp['window_partial']:14d} | {m_adp['window_future']+m_adp['window_partial']-(m_cur['window_future']+m_cur['window_partial']):+d}")

    # Specific Benchmark Checks
    named_checks = [
        "check_20260917_192307_0013_MISS.json",
        "check_20260917_184757_0018_UNCONFIRMED.json",
        "check_20260917_184758_0019_UNCONFIRMED.json",
        "check_20260917_184758_0021_UNCONFIRMED.json",
        "check_20260917_184758_0022_UNCONFIRMED.json",
        "check_20260917_184758_0023_UNCONFIRMED.json",
    ]

    cur_dict = {r["file"]: r for r in cur_results if r}
    adp_dict = {r["file"]: r for r in adp_results if r}

    print("\n8. SPECIFIC BENCHMARK REPLAY TRACKING:")
    print("   -----------------------------------------------------------------------------")
    for name in named_checks:
        c = cur_dict.get(name)
        a = adp_dict.get(name)
        if not c:
            print(f"   • {name:42s}: NOT FOUND")
            continue
        c_t = f"{c['lock_rel_ms']:.1f}ms" if c['lock_rel_ms'] is not None else "NO_LOCK"
        a_t = f"{a['lock_rel_ms']:.1f}ms" if a['lock_rel_ms'] is not None else "NO_LOCK"
        c_spd = f"{c['lock_spd']:.1f}°/s" if c['lock_spd'] is not None else "N/A"
        a_spd = f"{a['lock_spd']:.1f}°/s" if a['lock_spd'] is not None else "N/A"
        off_s = f"{c['offline_speed']:.1f}°/s" if c['offline_speed'] is not None else "N/A"
        print(f"   • {name:42s} (Off={off_s:8s}):")
        print(f"     CURRENT:      Lock={c_t:8s} | Speed={c_spd:9s} | Window={c['w_class']}")
        print(f"     ADAPTIVE_5PCT:Lock={a_t:8s} | Speed={a_spd:9s} | Window={a['w_class']}")

    print("\n================================================================================")


if __name__ == "__main__":
    run_clean_audit()
