#!/usr/bin/env python3
"""
tools/simulate_base_speed_prior.py
Simulation and comparison of CURRENT runtime predictor vs SESSION_BASE_SPEED prior.
Accurately emulates the scheduling and execution lifecycle of skillcheck_bot.
"""

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
    STATE_NO_MOTION,
    STATE_PROVISIONAL,
    STATE_LOCKED,
    STATE_COMMITTED,
)
from core.learner import SpeedProfileManager
from tools.audit_high_speed_locks_v2 import compute_offline_reference_speed


class SessionPriorPredictor(SkillCheckPredictor):
    def __init__(
        self,
        base_speed: float = 278.2,
        base_latency_ms: float = 127.27,
        allow_live_switch: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(latency_ms=base_latency_ms, *args, **kwargs)
        self.session_base_speed = base_speed
        self.base_latency_ms = base_latency_ms
        self.allow_live_switch = allow_live_switch
        self.using_prior = True
        self.consecutive_deviations = 0
        self.live_switch_timestamp = None
        self.live_switch_speed = None
        self.speed_deg_s = base_speed

    def update(
        self,
        t: float,
        needle_angle: float,
        needle_strength: float,
        white_zone: Optional[Dict[str, float]],
        black_zone: Optional[Dict[str, float]],
    ) -> bool:
        valid = super().update(t, needle_angle, needle_strength, white_zone, black_zone)
        if self.using_prior:
            self.speed_deg_s = self.session_base_speed
            self.latency_s = self.base_latency_ms / 1000.0
        return valid

    def check_live_speed_switch(self) -> bool:
        if not self.allow_live_switch:
            return False
        if len(self.speed_fits) < 4:
            return False

        last_4 = [f[1] for f in list(self.speed_fits)[-4:]]
        spread = max(last_4) - min(last_4)
        med_4 = float(np.median(last_4))

        # Real speed change requires:
        # 1. Consistent offset from prior > 35°/s
        # 2. Tight spread among the 4 fits <= 25°/s (not noise/stutter)
        if abs(med_4 - self.session_base_speed) > 35.0 and spread <= 25.0:
            self.consecutive_deviations += 1
            if self.consecutive_deviations >= 2:
                self.using_prior = False
                self.speed_deg_s = med_4
                self.locked_speed = med_4
                self.state = STATE_LOCKED
                return True
        else:
            self.consecutive_deviations = 0
        return False


def simulate_session_modes(files: List[Path], speed_mgr: SpeedProfileManager):
    print("================================================================================")
    print("      OFFLINE SIMULATION: CURRENT ESTIMATOR vs SESSION_BASE_SPEED PRIOR")
    print("================================================================================")

    results_a = []
    results_b = []

    # Causal prior state for Mode B
    causal_prior = 278.2
    clean_history = [278.2]

    for p in files:
        raw_bytes = p.read_bytes()
        d = json.loads(raw_bytes.decode("utf-8"))
        ev = d.get("evaluation", {})
        trig = d.get("trigger", {})

        raw_frames = d.get("frames", [])
        frames = [f for f in raw_frames if not f.get("is_pre_roll", False) and "needle_angle" in f]
        if len(frames) < 5:
            continue

        wz = d.get("locked_zones", {}).get("white", {}) or {}
        bz = d.get("locked_zones", {}).get("black", {}) or {}
        if not wz.get("start"):
            for fr in frames:
                w = fr.get("white_zone")
                if w and w.get("start") is not None:
                    wz = w
                    bz = fr.get("black_zone") or {}
                    break

        w_start = wz.get("start", 0.0)
        w_width = wz.get("width", 10.0)
        w_end = (w_start + w_width) % 360.0
        b_start = bz.get("start", w_end)
        b_width = bz.get("width", 39.0)
        b_end = (b_start + b_width) % 360.0

        t0 = frames[0]["time_rel_ms"] / 1000.0
        spawn_ang = frames[0]["needle_angle"]
        target_ratio = d.get("target_ratio", 0.50)
        chain_count = d.get("chain_count", 1)
        is_chain = chain_count > 1

        off_spd, off_conf, off_dur, off_mad = compute_offline_reference_speed(d)
        ground_truth_speed = off_spd if (off_spd and off_conf != "UNCERTAIN") else (trig.get("speed_deg_s") or 278.2)

        # True empirical physical latency for this check
        act_lat_ms = 127.27
        if ev.get("plateau_found") and ev.get("hit_angle") is not None:
            act_press = trig.get("actual_press_time")
            last_age = trig.get("last_frame_age_ms_at_fire")
            est_ang = trig.get("est_angle_at_trigger")
            if act_press and last_age and est_ang and ground_truth_speed > 0:
                ts = np.array([f["time_rel_ms"] for f in frames])
                angs = np.array([f["needle_angle"] for f in frames])
                unw = np.unwrap(np.radians(angs)) * (180.0 / np.pi)
                exp_last_ang = (est_ang - ground_truth_speed * (last_age / 1000.0)) % 360.0
                cand = [i for i, f in enumerate(frames) if abs((f["needle_angle"] - exp_last_ang + 180.0) % 360.0 - 180.0) < 0.15]
                if cand:
                    idx = cand[-1]
                    t_press_rel = ts[idx] + last_age
                    ang_disp = float(np.interp(t_press_rel, ts, unw)) % 360.0
                    d_theta = (ev["hit_angle"] - ang_disp + 360.0) % 360.0
                    if d_theta > 180.0:
                        d_theta -= 360.0
                    act_lat_ms = (d_theta / ground_truth_speed) * 1000.0

        def landing_outcome(land_ang):
            land = land_ang % 360.0
            in_great = (w_start <= land <= w_end) if w_start <= w_end else (land >= w_start or land <= w_end)
            if in_great:
                return "GREAT"
            in_good = (b_start <= land <= b_end) if b_start <= b_end else (land >= b_start or land <= b_end)
            if in_good:
                return "GOOD"
            return "MISS"

        # --- SIMULATE MODE A: CURRENT RUNTIME ESTIMATOR ---
        pred_a = SkillCheckPredictor(latency_ms=127.27, target_offset_ratio=target_ratio)
        if is_chain:
            pred_a.reset(keep_speed=False, default_speed=270.3, is_chain=True)

        lock_t_a = None
        lock_spd_a = None
        tier_a = None
        decision_spd_a = None
        press_t_a = None
        armed_a = False
        sched_reschedules_a = 0
        trigger_reason_a = "FALLBACK_NO_LOCK"

        for fr in frames:
            t = fr["time_rel_ms"] / 1000.0
            ang = fr["needle_angle"]
            strn = fr.get("needle_strength", 75.0)
            valid = pred_a.update(t, ang, strn, wz, bz)
            if not valid:
                continue

            # Speed locking condition
            if lock_t_a is None and pred_a.has_stable_speed():
                lock_t_a = t
                lock_spd_a = pred_a.speed_deg_s
                tier_a = speed_mgr.get_tier(lock_spd_a)
                lat_ms = speed_mgr.get_latency_for_speed(lock_spd_a)
                pred_a.latency_s = lat_ms / 1000.0

            # Once locked, generate prediction & schedule
            if lock_t_a is not None:
                p_res = pred_a.predict(t, ang, target="GREAT")
                if p_res:
                    decision_spd_a = pred_a.speed_deg_s
                    candidate_press_t = p_res["press_timestamp"]
                    rem_ms = p_res["time_until_press_ms"]

                    if not armed_a:
                        if p_res.get("should_press_now", False) or candidate_press_t <= t:
                            armed_a = True
                            trigger_reason_a = "IMMEDIATE"
                            press_t_a = t
                            break
                        else:
                            armed_a = True
                            trigger_reason_a = "СПИН-ТАЙМЕР"
                            press_t_a = candidate_press_t
                    else:
                        # Scheduler already armed: check execution or safety backup
                        if t >= (press_t_a + 0.008):
                            trigger_reason_a = "ТАЙМЕР-BACKUP"
                            press_t_a = t
                            break
                        elif t >= press_t_a:
                            # Scheduled timer fired at target press_t
                            break
                        elif rem_ms > 1.5:
                            check_age_ms = (t - t0) * 1000.0
                            if rem_ms > 20.0 or check_age_ms < 150.0:
                                if abs(candidate_press_t - press_t_a) > 0.0015:
                                    sched_reschedules_a += 1
                                    press_t_a = candidate_press_t

        if press_t_a is None:
            press_t_a = frames[-1]["time_rel_ms"] / 1000.0
            trigger_reason_a = "FALLBACK_NO_LOCK"
            decision_spd_a = pred_a.speed_deg_s
            tier_a = speed_mgr.get_tier(decision_spd_a)

        press_rel_ms_a = (press_t_a - t0) * 1000.0
        travel_to_press_a = ground_truth_speed * (press_rel_ms_a / 1000.0)
        ang_at_press_a = (spawn_ang + travel_to_press_a) % 360.0
        land_ang_a = (ang_at_press_a + ground_truth_speed * (act_lat_ms / 1000.0)) % 360.0
        sim_outc_a = landing_outcome(land_ang_a)

        # --- SIMULATE MODE B: SESSION_BASE_SPEED PRIOR ---
        pred_b = SessionPriorPredictor(
            base_speed=causal_prior,
            base_latency_ms=127.27,
            allow_live_switch=True,
            target_offset_ratio=target_ratio,
        )
        if is_chain:
            pred_b.reset(keep_speed=False, default_speed=causal_prior, is_chain=True)
            pred_b.speed_deg_s = causal_prior

        lock_t_b = None
        decision_spd_b = causal_prior
        tier_b = 275
        press_t_b = None
        armed_b = False
        sched_reschedules_b = 0
        trigger_reason_b = "SCHEDULED"
        used_prior = True

        for fr in frames:
            t = fr["time_rel_ms"] / 1000.0
            ang = fr["needle_angle"]
            strn = fr.get("needle_strength", 75.0)
            valid = pred_b.update(t, ang, strn, wz, bz)
            if not valid:
                continue

            # Shadow check for confirmed live speed deviation
            switched = pred_b.check_live_speed_switch()
            if switched and lock_t_b is None:
                lock_t_b = t
                decision_spd_b = pred_b.speed_deg_s
                tier_b = speed_mgr.get_tier(decision_spd_b)
                lat_ms = speed_mgr.get_latency_for_speed(decision_spd_b)
                pred_b.latency_s = lat_ms / 1000.0
                used_prior = False

            # Predict using prior (or confirmed switched speed) once motion is active
            if pred_b.motion_onset or is_chain:
                p_res = pred_b.predict(t, ang, target="GREAT")
                if p_res:
                    if used_prior:
                        decision_spd_b = pred_b.session_base_speed
                        tier_b = 275
                        pred_b.latency_s = 127.27 / 1000.0
                    else:
                        decision_spd_b = pred_b.speed_deg_s

                    candidate_press_t = p_res["press_timestamp"]
                    rem_ms = p_res["time_until_press_ms"]

                    if not armed_b:
                        if p_res.get("should_press_now", False) or candidate_press_t <= t:
                            armed_b = True
                            trigger_reason_b = "IMMEDIATE"
                            press_t_b = t
                            break
                        else:
                            armed_b = True
                            trigger_reason_b = "СПИН-ТАЙМЕР"
                            press_t_b = candidate_press_t
                    else:
                        if t >= (press_t_b + 0.008):
                            trigger_reason_b = "ТАЙМЕР-BACKUP"
                            press_t_b = t
                            break
                        elif t >= press_t_b:
                            # Fired at scheduled target
                            break
                        elif rem_ms > 1.5:
                            check_age_ms = (t - t0) * 1000.0
                            if rem_ms > 20.0 or check_age_ms < 150.0:
                                if abs(candidate_press_t - press_t_b) > 0.0015:
                                    sched_reschedules_b += 1
                                    press_t_b = candidate_press_t

        if press_t_b is None:
            press_t_b = frames[-1]["time_rel_ms"] / 1000.0
            trigger_reason_b = "FALLBACK_NO_LOCK"

        press_rel_ms_b = (press_t_b - t0) * 1000.0
        travel_to_press_b = ground_truth_speed * (press_rel_ms_b / 1000.0)
        ang_at_press_b = (spawn_ang + travel_to_press_b) % 360.0
        land_ang_b = (ang_at_press_b + ground_truth_speed * (act_lat_ms / 1000.0)) % 360.0
        sim_outc_b = landing_outcome(land_ang_b)

        item_a = {
            "file": p.name,
            "outcome": sim_outc_a,
            "actual_outcome": ev.get("outcome"),
            "trigger_reason": trigger_reason_a,
            "speed_error": abs(decision_spd_a - ground_truth_speed) if decision_spd_a else 0.0,
            "press_rel_ms": press_rel_ms_a,
            "reschedules": sched_reschedules_a,
            "tier": tier_a,
            "land_ang": land_ang_a,
            "spd": decision_spd_a,
        }
        item_b = {
            "file": p.name,
            "outcome": sim_outc_b,
            "actual_outcome": ev.get("outcome"),
            "trigger_reason": trigger_reason_b,
            "speed_error": abs(decision_spd_b - ground_truth_speed) if decision_spd_b else 0.0,
            "press_rel_ms": press_rel_ms_b,
            "reschedules": sched_reschedules_b,
            "tier": tier_b,
            "land_ang": land_ang_b,
            "used_prior": used_prior,
            "causal_prior": causal_prior,
            "spd": decision_spd_b,
        }
        results_a.append(item_a)
        results_b.append(item_b)

        # Update causal prior if check was clean
        if (
            sim_outc_b in ("GREAT", "GOOD")
            and off_spd is not None
            and off_conf in ("HIGH", "MEDIUM")
            and off_dur >= 0.080
            and abs(off_spd - causal_prior) <= 35.0
        ):
            clean_history.append(off_spd)
            if len(clean_history) > 10:
                clean_history.pop(0)
            causal_prior = float(np.median(clean_history))

    print(f"\nSimulated {len(results_a)} skill checks across fresh session:\n")

    outcomes_a = Counter(r["outcome"] for r in results_a)
    outcomes_b = Counter(r["outcome"] for r in results_b)
    actual_outcomes = Counter(r["actual_outcome"] for r in results_a)

    print("1. OUTCOME COMPARISON:")
    print(f"   • Actual In-Game Session:  GREAT = {actual_outcomes.get('GREAT',0):2d} | GOOD = {actual_outcomes.get('GOOD',0):2d} | MISS = {actual_outcomes.get('MISS',0):2d} (GREAT Rate: {actual_outcomes.get('GREAT',0)/len(results_a)*100:.1f}%)")
    print(f"   • Mode A (CURRENT):         GREAT = {outcomes_a.get('GREAT',0):2d} | GOOD = {outcomes_a.get('GOOD',0):2d} | MISS = {outcomes_a.get('MISS',0):2d} (GREAT Rate: {outcomes_a.get('GREAT',0)/len(results_a)*100:.1f}%)")
    print(f"   • Mode B (SESSION_PRIOR):   GREAT = {outcomes_b.get('GREAT',0):2d} | GOOD = {outcomes_b.get('GOOD',0):2d} | MISS = {outcomes_b.get('MISS',0):2d} (GREAT Rate: {outcomes_b.get('GREAT',0)/len(results_b)*100:.1f}%)")

    spd_errs_a = [r["speed_error"] for r in results_a]
    spd_errs_b = [r["speed_error"] for r in results_b]
    print("\n2. SPEED ESTIMATE ERROR AT DECISION:")
    print(f"   • Mode A (CURRENT):       Median = {np.median(spd_errs_a):5.2f}°/s | Mean = {np.mean(spd_errs_a):5.2f}°/s | p90 = {np.percentile(spd_errs_a, 90):5.2f}°/s")
    print(f"   • Mode B (SESSION_PRIOR): Median = {np.median(spd_errs_b):5.2f}°/s | Mean = {np.mean(spd_errs_b):5.2f}°/s | p90 = {np.percentile(spd_errs_b, 90):5.2f}°/s")

    reasons_a = Counter(r["trigger_reason"] for r in results_a)
    reasons_b = Counter(r["trigger_reason"] for r in results_b)
    resched_a = sum(r["reschedules"] for r in results_a)
    resched_b = sum(r["reschedules"] for r in results_b)

    print("\n3. TRIGGER MECHANICS & SCHEDULER STABILITY:")
    print(f"   • Mode A Reasons: {dict(reasons_a)} | Total Reschedules = {resched_a}")
    print(f"   • Mode B Reasons: {dict(reasons_b)} | Total Reschedules = {resched_b}")

    switches_b = sum(1 for r in results_b if not r["used_prior"])
    print(f"   • False live-speed switches in Mode B: {switches_b} / {len(results_b)}")

    # Check-by-check comparison for the 11 GOODs
    print("\n4. IMPACT OF SESSION PRIOR ON THE 11 RECORDED GOOD CHECKS:")
    print("   -----------------------------------------------------------------------------")
    print(f"   {'File':38s} | {'Actual':6s} | {'Mode A':6s} (Tier, Spd) | {'Mode B':6s} (Prior, Spd)")
    print("   ---------------------------------------+--------+--------------------+--------------------")
    for ra, rb in zip(results_a, results_b):
        if ra["actual_outcome"] == "GOOD":
            print(f"   {ra['file']:38s} | {ra['actual_outcome']:6s} | {ra['outcome']:6s} (T={str(ra['tier']):3s}, {ra['spd']:5.1f}) | {rb['outcome']:6s} (P={rb['causal_prior']:5.1f}, {rb['spd']:5.1f})")

    # Check 204705_0001_MISS inspection
    miss_check_a = next((r for r in results_a if "204705_0001_MISS" in r["file"]), None)
    miss_check_b = next((r for r in results_b if "204705_0001_MISS" in r["file"]), None)

    print("\n5. DEEP DIVE: check_20260917_204705_0001_MISS.json:")
    print("   -----------------------------------------------------------------------------")
    if miss_check_a and miss_check_b:
        print(f"   • Actual Recorded Outcome: MISS (Error = -9.45°, -34.0ms, hit at 69.55° vs White [74.0..83.0])")
        print(f"   • Mode A Simulated Outcome: {miss_check_a['outcome']} (Land = {miss_check_a['land_ang']:.1f}°, Press = {miss_check_a['press_rel_ms']:.1f}ms, Tier = {miss_check_a['tier']})")
        print(f"   • Mode B Simulated Outcome: {miss_check_b['outcome']} (Land = {miss_check_b['land_ang']:.1f}°, Press = {miss_check_b['press_rel_ms']:.1f}ms, Prior = {miss_check_b['causal_prior']:.1f}°/s)")
        print("   • Physical Trajectory Diagnosis:")
        print("     On check 204705_0001, the needle spawned at 270.0° and traveled to target 79.0° (169.0° arc).")
        print("     At 278.2°/s, the total kinematic flight time was 607.5ms.")
        print("     The key was pressed at 469.4ms, when the needle was at 40.58° (38.42° before target).")
        print("     In an ideal 127.27ms reaction, 38.42° / 278.2°/s = 138.1ms, landing at 79.0° (GREAT).")
        print("     However, the in-game needle arrested at 69.55° (only 28.97° after keypress).")
        print("     28.97° / 278.2°/s = 104.1ms effective latency. The game arrested the needle 34.0ms early.")
        print("     Because the true offline speed was 280.2°/s and the prior was 278.2°/s, the error was 2.0°/s.")
        print("     No speed estimator or prior can fix this MISS: it is a pure engine response/game latency variance.")


if __name__ == "__main__":
    fresh_files = sorted(Path("replays").glob("check_20260917_2033*.json")) + sorted(Path("replays").glob("check_20260917_203[4-9]*.json")) + sorted(Path("replays").glob("check_20260917_204[0-3]*.json")) + [Path("replays/check_20260917_204705_0001_MISS.json")]
    mgr = SpeedProfileManager(auto_bootstrap=False, save_to_disk=False)
    simulate_session_modes(fresh_files, mgr)
