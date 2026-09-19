"""
Session-wide Performance and Training Telemetry Tracker for Violent District Bot.
Collects and formats statistical reports across gameplay sessions:
- Check outcomes (GREAT / GOOD / MISS / ABORTED / UNCONFIRMED)
- Locked speed tiers: accepted/rejected counts, median delay, MAD, p25/p75, status
- Trigger counts (Scheduled, Immediate, Backup)
- Prediction lateness, scheduler spin jitter percentiles (p50/p95/p99/max)
- Capture wait, decoded frame arrival, unique frame arrival, duplicate %
- Vision and predict execution times.
"""

import collections
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class SessionTelemetryTracker:
    def __init__(self):
        self.outcomes = collections.Counter()
        self.trigger_counts = collections.Counter()
        self.prediction_lateness_ms: List[float] = []
        self.scheduler_jitter_ms: List[float] = []
        self.grab_wait_ms: List[float] = []
        self.python_handoff_age_ms: List[float] = []
        self.decoded_intervals_ms: List[float] = []
        self.unique_intervals_ms: List[float] = []
        self.vision_times_ms: List[float] = []
        self.predict_times_ms: List[float] = []
        self.total_decoded_frames = 0
        self.unique_frames_count = 0
        self.last_decode_t: Optional[float] = None
        self.last_unique_t: Optional[float] = None
        self.prev_frame_thumb = None

        self.normal_locked_fires = 0
        self.fallback_no_lock_fires = 0
        self.no_fire_due_to_insufficient_lock = 0

        self.requested_base_latency: Optional[float] = None
        self.requested_latencies: List[float] = []
        self.actual_used_delays: List[float] = []
        self.profile_overrides_count: int = 0
        self.experiment_discrepancies: int = 0

        self.pending_fire_tier: Optional[int] = None
        self.tier_records: Dict[int, Dict[str, Any]] = collections.defaultdict(lambda: {
            "fires": 0,
            "accepted": 0,
            "rejected": 0,
            "not_evaluated": 0,
            "reasons": collections.Counter(),
            "ideal_delays": [],
            "speeds_at_lock": [],
            "speeds_at_fire": [],
        })

    def record_frame(
        self,
        t_grab_done: float,
        decode_ready_ts: Optional[float],
        grab_wait_ms: float,
        python_handoff_age_ms: float,
        frame_bgr: Optional[np.ndarray] = None,
    ):
        self.total_decoded_frames += 1
        self.grab_wait_ms.append(grab_wait_ms)
        self.python_handoff_age_ms.append(python_handoff_age_ms)

        if decode_ready_ts is not None:
            if self.last_decode_t is not None:
                self.decoded_intervals_ms.append((decode_ready_ts - self.last_decode_t) * 1000.0)
            self.last_decode_t = decode_ready_ts

        if frame_bgr is not None:
            # Subsample thumbnail for efficient real-time duplicate check
            thumb = frame_bgr[::6, ::6, :]
            is_unique = True
            if self.prev_frame_thumb is not None:
                diff = float(np.mean(np.abs(thumb.astype(np.int16) - self.prev_frame_thumb.astype(np.int16))))
                if diff < 0.40:
                    is_unique = False

            if is_unique:
                self.unique_frames_count += 1
                if self.last_unique_t is not None:
                    self.unique_intervals_ms.append((t_grab_done - self.last_unique_t) * 1000.0)
                self.last_unique_t = t_grab_done
                self.prev_frame_thumb = thumb

    def record_diagnostics(self, vision_ms: float, predict_ms: float):
        self.vision_times_ms.append(vision_ms)
        self.predict_times_ms.append(predict_ms)

    def record_fire(
        self,
        reason: str,
        locked_tier: Optional[int],
        speed_at_lock: Optional[float],
        speed_at_fire: Optional[float],
        prediction_lateness_ms: float,
        scheduler_jitter_ms: Optional[float] = None,
        trigger_mode: Optional[str] = None,
        requested_latency_ms: Optional[float] = None,
        actual_delay_ms: Optional[float] = None,
        is_profile_override: bool = False,
        is_experiment_valid: bool = True,
    ):
        is_fallback = (speed_at_lock is None) or (trigger_mode == "FALLBACK_NO_LOCK") or ("FALLBACK" in reason)
        if is_fallback:
            self.fallback_no_lock_fires += 1
            clean_reason = "FALLBACK_NO_LOCK"
        else:
            self.normal_locked_fires += 1
            clean_reason = "SCHEDULED" if ("СПИН" in reason or "SCHEDULED" in reason) else (
                "BACKUP" if "BACKUP" in reason else "IMMEDIATE"
            )
        self.trigger_counts[clean_reason] += 1
        self.prediction_lateness_ms.append(prediction_lateness_ms)
        if scheduler_jitter_ms is not None:
            self.scheduler_jitter_ms.append(scheduler_jitter_ms)

        if requested_latency_ms is not None:
            self.requested_latencies.append(requested_latency_ms)
            if self.requested_base_latency is None:
                self.requested_base_latency = requested_latency_ms
        if actual_delay_ms is not None:
            self.actual_used_delays.append(actual_delay_ms)
        if is_profile_override:
            self.profile_overrides_count += 1
        if not is_experiment_valid:
            self.experiment_discrepancies += 1

        tier = locked_tier
        if tier is None:
            tier = int(round(speed_at_fire / 25.0) * 25.0) if (speed_at_fire and speed_at_fire > 15.0) else 0

        self.pending_fire_tier = tier
        rec = self.tier_records[tier]
        rec["fires"] += 1
        if speed_at_lock is not None:
            rec["speeds_at_lock"].append(speed_at_lock)
        if speed_at_fire is not None:
            rec["speeds_at_fire"].append(speed_at_fire)

    def record_outcome(
        self,
        outcome: str,
        speed_tier: Optional[int],
        speed_at_lock: Optional[float],
        speed_at_fire: Optional[float],
        prof_res: Optional[Dict[str, Any]],
    ):
        self.outcomes[outcome] += 1
        if self.pending_fire_tier is not None:
            rec = self.tier_records[self.pending_fire_tier]
            if prof_res:
                if prof_res.get("rejected"):
                    rec["rejected"] += 1
                    rec["reasons"][prof_res.get("reason", "UNKNOWN")] += 1
                else:
                    rec["accepted"] += 1
                    if prof_res.get("ideal_delay") is not None:
                        rec["ideal_delays"].append(prof_res.get("ideal_delay"))
            else:
                rec["not_evaluated"] += 1
            self.pending_fire_tier = None
        else:
            if speed_at_lock is None:
                self.no_fire_due_to_insufficient_lock += 1

    @staticmethod
    def _pcts(vals: List[float]) -> Tuple[float, float, float]:
        if not vals:
            return 0.0, 0.0, 0.0
        arr = np.array(vals, dtype=float)
        return float(np.percentile(arr, 50)), float(np.percentile(arr, 95)), float(np.percentile(arr, 99))

    def generate_report(self, learner=None, grabber=None) -> str:
        lines = [
            "================================================================================",
            "           SESSION PERFORMANCE & SPEED PROFILES TRAINING REPORT                 ",
            "================================================================================",
        ]

        # 1. Outcomes
        total_checks = sum(self.outcomes.values())
        great = self.outcomes.get("GREAT", 0)
        good = self.outcomes.get("GOOD", 0)
        miss = self.outcomes.get("MISS", 0)
        aborted = self.outcomes.get("ABORTED_LMB", 0) + self.outcomes.get("ABORTED", 0)
        unconf = self.outcomes.get("UNCONFIRMED", 0)
        valid_checks = great + good + miss
        success_pct = ((great + good) / max(1, valid_checks) * 100.0) if valid_checks > 0 else 0.0

        lines.append(f"OUTCOMES ({total_checks} total checks):")
        lines.append(f"  GREAT:       {great:3d} ({great/max(1, total_checks)*100.0:.1f}%)")
        lines.append(f"  GOOD:        {good:3d} ({good/max(1, total_checks)*100.0:.1f}%)")
        lines.append(f"  MISS:        {miss:3d} ({miss/max(1, total_checks)*100.0:.1f}%)")
        lines.append(f"  ABORTED LMB: {aborted:3d}")
        lines.append(f"  UNCONFIRMED: {unconf:3d}")
        lines.append(f"  ACCURACY (Great+Good / valid): {success_pct:.1f}%")
        lines.append("--------------------------------------------------------------------------------")

        # 2. Trigger Method Breakdown
        sched_cnt = self.trigger_counts.get("SCHEDULED", 0)
        imm_cnt = self.trigger_counts.get("IMMEDIATE", 0)
        bak_cnt = self.trigger_counts.get("BACKUP", 0)
        fallback_cnt = self.trigger_counts.get("FALLBACK_NO_LOCK", 0)
        tot_fires = sched_cnt + imm_cnt + bak_cnt + fallback_cnt
        lines.append(f"TRIGGERS ({tot_fires} fires):")
        lines.append(f"  SCHEDULED (Spin Timer): {sched_cnt:3d}")
        lines.append(f"  IMMEDIATE (Direct):     {imm_cnt:3d}")
        lines.append(f"  TIMER-BACKUP (Safety):  {bak_cnt:3d}")
        lines.append(f"  FALLBACK (No Lock):     {fallback_cnt:3d}")
        lines.append(f"  NO-FIRE (Insufficient): {self.no_fire_due_to_insufficient_lock:3d}")
        lines.append(f"  BREAKDOWN: {self.normal_locked_fires} Locked + {self.fallback_no_lock_fires} Fallback = {tot_fires} Total Fires")
        lines.append("--------------------------------------------------------------------------------")

        # 3. Latency Control & Overrides Audit
        req_lat = self.requested_base_latency if self.requested_base_latency is not None else (self.requested_latencies[-1] if self.requested_latencies else 0.0)
        lines.append("LATENCY CONTROL & EXPERIMENT AUDIT:")
        lines.append(f"  Requested Base Latency:     {req_lat:.1f}ms")
        if self.actual_used_delays:
            arr_delays = np.array(self.actual_used_delays)
            d_min = float(np.min(arr_delays))
            d_med = float(np.median(arr_delays))
            d_max = float(np.max(arr_delays))
            lines.append(f"  Actual Used Delay (Fire):   min={d_min:.1f}ms | median={d_med:.1f}ms | max={d_max:.1f}ms (N={len(arr_delays)})")
        else:
            lines.append("  Actual Used Delay (Fire):   N/A (no fires recorded)")
        lines.append(f"  Profile Overrides Count:    {self.profile_overrides_count}")
        if self.experiment_discrepancies > 0:
            lines.append(f"  ⚠️ EXPERIMENT DISCREPANCY:  {self.experiment_discrepancies} checks had requested != actual used delay!")
        lines.append("--------------------------------------------------------------------------------")

        # 4. Latency & Scheduler Precision
        lat_p50, lat_p95, lat_p99 = self._pcts(self.prediction_lateness_ms)
        jit_p50, jit_p95, jit_p99 = self._pcts(self.scheduler_jitter_ms)
        max_jit = max(self.scheduler_jitter_ms) if self.scheduler_jitter_ms else 0.0
        lines.append("INPUT & SCHEDULER TIMING:")
        lines.append(f"  Prediction lateness:   p50={lat_p50:+.2f}ms | p95={lat_p95:+.2f}ms | p99={lat_p99:+.2f}ms")
        lines.append(f"  Scheduler spin jitter: p50={jit_p50:+.2f}ms | p95={jit_p95:+.2f}ms | p99={jit_p99:+.2f}ms | max={max_jit:+.2f}ms")
        lines.append("--------------------------------------------------------------------------------")

        # 4. Pipeline & Capture Timing
        gw_p50, gw_p95, gw_p99 = self._pcts(self.grab_wait_ms)
        dec_p50, dec_p95, dec_p99 = self._pcts(self.decoded_intervals_ms)
        u_p50, u_p95, u_p99 = self._pcts(self.unique_intervals_ms)
        vis_p50, vis_p95, vis_p99 = self._pcts(self.vision_times_ms)
        prd_p50, prd_p95, prd_p99 = self._pcts(self.predict_times_ms)

        dup_pct = ((self.total_decoded_frames - self.unique_frames_count) / max(1, self.total_decoded_frames) * 100.0)
        stale_dropped = grabber.get_diagnostics().get("discarded_stale_frames", 0) if grabber else 0

        lines.append("CAPTURE & VISION PIPELINE:")
        lines.append(f"  Grab wait:            p50={gw_p50:5.2f}ms | p95={gw_p95:5.2f}ms | p99={gw_p99:5.2f}ms")
        lines.append(f"  Decoded frame int:    p50={dec_p50:5.2f}ms | p95={dec_p95:5.2f}ms | p99={dec_p99:5.2f}ms")
        lines.append(f"  Unique frame int:     p50={u_p50:5.2f}ms | p95={u_p95:5.2f}ms | p99={u_p99:5.2f}ms")
        lines.append(f"  Vision process:       p50={vis_p50:5.2f}ms | p95={vis_p95:5.2f}ms | p99={vis_p99:5.2f}ms")
        lines.append(f"  Predict process:      p50={prd_p50:5.2f}ms | p95={prd_p95:5.2f}ms | p99={prd_p99:5.2f}ms")
        lines.append(f"  Total decoded frames: {self.total_decoded_frames} (Unique: {self.unique_frames_count}, Duplicates: {dup_pct:.1f}%)")
        lines.append(f"  Discarded stale frames: {stale_dropped}")
        lines.append("--------------------------------------------------------------------------------")

        # 5. Speed Profiles & Training Diagnostics
        lines.append("SPEED TIERS TRAINING & STABILITY:")
        if not self.tier_records and not (learner and learner.speed_profiles.profiles):
            lines.append("  (Нет зафиксированных проверок)")
        else:
            all_tiers = sorted(set(list(self.tier_records.keys()) + (list(learner.speed_profiles.profiles.keys()) if learner else [])))
            for tier in all_tiers:
                rec = self.tier_records.get(tier, {})
                fires = rec.get("fires", 0)
                acc = rec.get("accepted", 0)
                rej = rec.get("rejected", 0)
                reasons = rec.get("reasons", {})
                reasons_str = ", ".join(f"{k}: {v}" for k, v in reasons.items()) if reasons else "none"

                prof = learner.speed_profiles.profiles.get(tier) if learner else None
                status = "TRUSTED" if (prof and prof.trusted) else f"SHADOW (ep{prof.epoch if prof else 1})"
                cur_del = f"{prof.delay_ms:.1f}ms" if prof else "N/A"
                mad = f"{prof.mad_ms:.1f}ms" if prof else "N/A"
                p25_p75 = f"[{prof.p25_ms:.1f}, {prof.p75_ms:.1f}]" if prof else "N/A"

                lock_speeds = rec.get("speeds_at_lock", [])
                fire_speeds = rec.get("speeds_at_fire", [])
                unstable_cnt = sum(1 for l, f in zip(lock_speeds, fire_speeds) if abs(l - f) > 35.0)
                unstable_pct = (unstable_cnt / max(1, len(lock_speeds)) * 100.0) if lock_speeds else 0.0

                not_eval = rec.get("not_evaluated", 0)
                lines.append(f"  Tier ~{tier:3d}°/s: [{status}] Delay={cur_del} | MAD={mad} | p25/75={p25_p75}")
                lines.append(f"    Session Fires: {fires} | Accepted: {acc} | Rejected: {rej} | Not Evaluated: {not_eval} (Reasons: {reasons_str})")
                lines.append(f"    Speed stability: {unstable_pct:.1f}% unstable samples (lock vs fire diff > 35°/s)")

            tot_fires = sum(r.get("fires", 0) for r in self.tier_records.values())
            tot_acc = sum(r.get("accepted", 0) for r in self.tier_records.values())
            tot_rej = sum(r.get("rejected", 0) for r in self.tier_records.values())
            tot_not_eval = sum(r.get("not_evaluated", 0) for r in self.tier_records.values())
            lines.append(f"  TOTAL FIRES: {tot_fires} = {tot_acc} Accepted + {tot_rej} Rejected + {tot_not_eval} Not Evaluated")
            lines.append(f"  FIRE TYPES:  {tot_fires} = {self.normal_locked_fires} Locked + {self.fallback_no_lock_fires} Fallback No-Lock")

        lines.append("================================================================================")
        return "\n".join(lines)
