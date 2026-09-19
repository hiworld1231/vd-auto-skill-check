"""
Closed-Loop Adaptive Latency Learning Engine for Violent District Skill Check AI.
Observes post-trigger needle trajectories, detects the exact frozen needle registration
angle inside Roblox, computes error relative to the Great zone setpoint, and performs
robust, noise-resistant feedback adjustment with deadbands, outlier rejection,
frenzy protection, and safety bounds.
"""

import collections
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from core.vision import is_angle_in_arc

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
PROFILES_PATH = Path(__file__).resolve().parent.parent / "speed_profiles.json"
REPLAYS_DIR = Path(__file__).resolve().parent.parent / "replays"

SCHEMA_VERSION = 2
CURRENT_CALIBRATION_EPOCH = 2
_UNSPECIFIED = object()


def circular_spread_and_center(angles: List[float]) -> Tuple[float, float]:
    """
    Computes circular spread (max - min) and robust median center for a small cluster of angles.
    Handles 360°/0° wrap-around seamlessly.
    Returns (spread_deg, center_angle_deg).
    """
    if not angles:
        return 0.0, 0.0
    ref = angles[0]
    deltas = [(a - ref + 180.0) % 360.0 - 180.0 for a in angles]
    spread = float(max(deltas) - min(deltas))
    med_delta = float(np.median(deltas))
    center = float((ref + med_delta) % 360.0)
    return spread, center


def compute_mad(values: List[float]) -> float:
    """Computes Median Absolute Deviation (MAD) for a list of values."""
    if not values or len(values) < 2:
        return 0.0
    arr = np.array(values, dtype=float)
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


class SessionBaseSpeedTracker:
    """
    Maintains a strictly causal history of clean completed skill checks.
    Updates the session prior using the robust median of past clean observations.
    """

    def __init__(self, initial_seed: float = 278.0, max_history: int = 10):
        self.initial_seed = float(initial_seed)
        self.max_history = max_history
        self.history: collections.deque = collections.deque(maxlen=max_history)
        self.current_prior: float = self.initial_seed

    def record_completed_check(
        self,
        outcome: str,
        reason: str,
        scheduler_error_ms: Optional[float],
        speed_at_fire: Optional[float],
        offline_speed: Optional[float] = None,
        duration_ms: float = 0.0,
    ) -> bool:
        """
        Records a check strictly AFTER it has finished.
        Returns True if the check satisfied clean criteria and updated the prior.
        """
        if outcome not in ("GREAT", "GOOD"):
            return False
        if reason not in ("СПИН-ТАЙМЕР", "SCHEDULED"):
            return False
        if scheduler_error_ms is None or abs(scheduler_error_ms) > 2.0:
            return False
        if duration_ms < 80.0:
            return False

        ref_spd = offline_speed if offline_speed is not None else speed_at_fire
        if ref_spd is None:
            return False
        if abs(ref_spd - self.current_prior) > 35.0:
            return False

        self.history.append(float(ref_spd))
        self.current_prior = float(np.median(list(self.history)))
        return True


def classify_miss(
    reason: str = "UNKNOWN",
    scheduler_error_ms: Optional[float] = None,
    last_frame_age_ms: Optional[float] = None,
    capture_wait_ms: Optional[float] = None,
    duration_ms: float = 0.0,
    speed_at_lock: Optional[float] = None,
    speed_at_fire: float = 0.0,
    tier_at_lock: Optional[int] = None,
    tier_at_fire: Optional[int] = None,
    error_deg: float = 0.0,
    error_ms: float = 0.0,
    max_frame_gap_ms: Optional[float] = None,
) -> Tuple[str, str]:
    """
    Classifies a MISS into one of diagnostic root causes:
    0. CLEAN_RESPONSE_OUTLIER: Normal scheduled trigger with accurate speed and sub-2ms schedule,
       but abnormal early engine arrest (effective response << delay).
    1. GHOST/REACQUIRE: False re-arm on dying ring, immediate trigger on overdue spawn, duration < 120ms.
    2. SPEED/TIER WRONG: Locked tier differs from fire tier, or speed diverged > 25°/s between lock and fire.
    3. SCHEDULER LATE: Scheduler lateness or thread spin jitter > 8.0ms.
    4. VISION/DROPPED FRAMES: Stale frame at fire (> 35ms), capture wait > 35ms, or frame gap > 50ms.
    5. CLEAN TIMING MISS: Hardware timing calibration error. All systems functioned cleanly, but latency setpoint was incorrect.
    """
    # 0. CLEAN_RESPONSE_OUTLIER
    if (
        reason in ("СПИН-ТАЙМЕР", "SCHEDULED")
        and scheduler_error_ms is not None
        and abs(scheduler_error_ms) <= 2.0
        and speed_at_lock is not None
        and abs(speed_at_fire - speed_at_lock) <= 10.0
        and error_deg < -5.0
    ):
        return ("CLEAN_RESPONSE_OUTLIER", f"Abnormal early engine arrest (error={error_deg:+.1f}°, err_ms={error_ms:+.1f}ms, sched_err={scheduler_error_ms:+.2f}ms)")

    # 0b. UNSTABLE_SPEED / NO_SPEED_LOCK
    if tier_at_lock is None or speed_at_lock is None:
        return ("UNSTABLE_SPEED", f"Triggered without confirmed speed lock (tier_at_lock={tier_at_lock}, speed_at_lock={speed_at_lock})")

    # 1. GHOST / REACQUIRE
    if reason == "IMMEDIATE" and duration_ms < 120.0:
        return ("GHOST/REACQUIRE", f"Immediate fire on spawn/dying ring (dur={duration_ms:.0f}ms)")
    if reason == "IMMEDIATE" and scheduler_error_ms is not None and scheduler_error_ms > 20.0:
        return ("GHOST/REACQUIRE", f"Immediate fire on overdue target (lateness={scheduler_error_ms:+.1f}ms)")

    # 2. SPEED / TIER WRONG
    if speed_at_lock is not None and speed_at_fire > 0 and abs(speed_at_fire - speed_at_lock) > 25.0:
        return ("SPEED/TIER WRONG", f"Speed diverged (lock={speed_at_lock:.1f}°/s, fire={speed_at_fire:.1f}°/s, diff={abs(speed_at_fire-speed_at_lock):.1f}°/s)")
    if tier_at_lock is not None and tier_at_fire is not None and tier_at_lock != tier_at_fire:
        return ("SPEED/TIER WRONG", f"Tier changed (lock=~{tier_at_lock}°/s, fire=~{tier_at_fire}°/s)")

    # 3. SCHEDULER LATE
    if scheduler_error_ms is not None and abs(scheduler_error_ms) > 8.0:
        return ("SCHEDULER LATE", f"Scheduler lateness/jitter exceeded 8ms ({scheduler_error_ms:+.2f}ms)")

    # 4. VISION / DROPPED FRAMES
    if last_frame_age_ms is not None and last_frame_age_ms > 35.0:
        return ("VISION/DROPPED FRAMES", f"Stale frame at fire ({last_frame_age_ms:.1f}ms)")
    if capture_wait_ms is not None and capture_wait_ms > 35.0:
        return ("VISION/DROPPED FRAMES", f"Capture wait hitch ({capture_wait_ms:.1f}ms)")
    if max_frame_gap_ms is not None and max_frame_gap_ms > 50.0:
        return ("VISION/DROPPED FRAMES", f"Capture hitch during approach (max_gap={max_frame_gap_ms:.1f}ms)")

    # 5. CLEAN TIMING MISS
    direction = "EARLY" if error_deg < 0 else "LATE"
    return ("CLEAN TIMING MISS", f"Calibration error: {direction} by {abs(error_deg):.1f}° ({error_ms:+.1f}ms) at {speed_at_fire:.1f}°/s")


class SpeedProfile:
    """
    Profile for a discrete game speed tier (e.g. ~275°/s, ~375°/s, ~650°/s).
    Stores median delay, sample count, dispersion (std and MAD), confidence, and
    performs robust batch updates.
    """
    def __init__(
        self,
        tier_speed: int,
        delay_ms: float = 122.5,
        samples: int = 0,
        std_ms: float = 0.0,
        trusted: bool = False,
        recent_delays: Optional[List[float]] = None,
        mad_ms: float = 0.0,
        p25_ms: float = 0.0,
        p75_ms: float = 0.0,
        accepted_samples: int = 0,
        rejected_samples: int = 0,
        epoch: int = 2,
    ):
        self.tier_speed = int(tier_speed)
        self.delay_ms = float(delay_ms)
        self.samples = int(samples)
        self.std_ms = float(std_ms)
        self.mad_ms = float(mad_ms)
        self.p25_ms = float(p25_ms)
        self.p75_ms = float(p75_ms)
        self.trusted = bool(trusted)
        self.accepted_samples = int(accepted_samples if accepted_samples > 0 else samples)
        self.rejected_samples = int(rejected_samples)
        self.epoch = int(epoch)
        self.recent_delays: collections.deque = collections.deque(recent_delays or [], maxlen=20)
        self.batch_buffer: List[float] = []
        # CRITICAL: Always recalculate dispersion stats directly from recent_delays if >= 2 samples exist,
        # never trusting stale mad_ms=0.0 from legacy JSON files!
        if len(self.recent_delays) >= 2:
            self._recalculate_stats()

    def _recalculate_stats(self):
        if len(self.recent_delays) >= 2:
            vals = list(self.recent_delays)
            self.std_ms = float(np.std(vals))
            self.mad_ms = compute_mad(vals)
            self.p25_ms = float(np.percentile(vals, 25))
            self.p75_ms = float(np.percentile(vals, 75))

        # Bidirectional trusted transition:
        # A profile becomes trusted ONLY with sufficient samples, low MAD and low STD in current epoch.
        # It reverts to untrusted/unstable as soon as fresh dispersion exceeds bounds!
        if (
            self.samples >= 8
            and len(self.recent_delays) >= 5
            and self.mad_ms <= 15.0
            and self.std_ms <= 22.0
            and self.epoch >= CURRENT_CALIBRATION_EPOCH
        ):
            self.trusted = True
        elif self.samples < 8 or self.mad_ms > 18.0 or self.std_ms > 25.0:
            self.trusted = False

    @property
    def confidence(self) -> str:
        # Confidence must evaluate both sample size and low dispersion (MAD & STD)
        if self.samples >= 15 and len(self.recent_delays) >= 10 and self.mad_ms <= 12.0 and self.std_ms <= 16.0:
            return "high"
        elif self.samples >= 8 and len(self.recent_delays) >= 5 and self.mad_ms <= 18.0 and self.std_ms <= 24.0:
            return "medium"
        return "low"

    def add_ideal_delay(self, ideal_delay_ms: float, batch_size: int = 5) -> Optional[float]:
        """
        Adds an ideal delay observation from a verified hit.
        Performs batch median update when batch_size clean samples accumulate.
        Returns the adjustment applied if a batch step occurred, else None.
        """
        self.samples += 1
        self.recent_delays.append(ideal_delay_ms)
        self.batch_buffer.append(ideal_delay_ms)
        self._recalculate_stats()

        # Batch update: do not jerk latency after every single check.
        if len(self.batch_buffer) >= batch_size:
            target = float(np.median(self.recent_delays))
            diff = target - self.delay_ms
            step = float(np.clip(diff, -2.5, 2.5))
            self.delay_ms += step
            self.batch_buffer.clear()
            return step
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier_speed": self.tier_speed,
            "delay_ms": round(self.delay_ms, 2),
            "samples": self.samples,
            "accepted_samples": self.accepted_samples,
            "rejected_samples": self.rejected_samples,
            "epoch": self.epoch,
            "std_ms": round(self.std_ms, 2),
            "mad_ms": round(self.mad_ms, 2),
            "p25_ms": round(self.p25_ms, 2),
            "p75_ms": round(self.p75_ms, 2),
            "confidence": self.confidence,
            "trusted": self.trusted,
            "recent_delays": [round(x, 2) for x in self.recent_delays],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpeedProfile":
        return cls(
            tier_speed=data["tier_speed"],
            delay_ms=data.get("delay_ms", 122.5),
            samples=data.get("samples", 0),
            std_ms=data.get("std_ms", 0.0),
            trusted=data.get("trusted", False),
            recent_delays=data.get("recent_delays", []),
            mad_ms=data.get("mad_ms", 0.0),
            p25_ms=data.get("p25_ms", 0.0),
            p75_ms=data.get("p75_ms", 0.0),
            accepted_samples=data.get("accepted_samples", data.get("samples", 0)),
            rejected_samples=data.get("rejected_samples", 0),
            epoch=data.get("epoch", 1),
        )


class SpeedProfileManager:
    """
    Manages discrete speed tiers, shadow learning for new speeds,
    batch median updates, position bias tracking, and session summaries.
    In freeze mode, active profiles, delays, samples, and trust are strictly immutable.
    Hypothetical updates are tracked in shadow_profiles without affecting FIRE delays.
    """
    def __init__(
        self,
        profiles_path: Optional[Path] = None,
        auto_bootstrap: bool = True,
        save_to_disk: bool = True,
        freeze: bool = False,
    ):
        self.path = Path(profiles_path) if profiles_path else PROFILES_PATH
        self.freeze = bool(freeze)
        self.save_to_disk = bool(save_to_disk) and not self.freeze
        self.schema_version = SCHEMA_VERSION
        self.calibration_epoch = CURRENT_CALIBRATION_EPOCH
        self.profiles: Dict[int, SpeedProfile] = {}
        self.shadow_profiles: Dict[int, SpeedProfile] = {}
        self.sector_errors: Dict[int, List[float]] = {0: [], 90: [], 180: [], 270: []}
        self.step_size = 25  # tier rounding step: 225, 250, 275, 300, 325, 350, 375, 400, etc.

        self.load()
        if not self.profiles and auto_bootstrap:
            self.bootstrap_from_replays(REPLAYS_DIR)
            if self.save_to_disk:
                self.save()
        self._init_shadow_profiles()

    def _init_shadow_profiles(self):
        self.shadow_profiles = {}
        for tier, prof in self.profiles.items():
            self.shadow_profiles[tier] = SpeedProfile(
                tier_speed=prof.tier_speed,
                delay_ms=prof.delay_ms,
                samples=prof.samples,
                std_ms=prof.std_ms,
                trusted=prof.trusted,
                epoch=prof.epoch,
                accepted_samples=prof.accepted_samples,
                rejected_samples=prof.rejected_samples,
                mad_ms=prof.mad_ms,
                p25_ms=prof.p25_ms,
                p75_ms=prof.p75_ms,
                recent_delays=list(prof.recent_delays),
            )

    def _get_shadow_profile(self, tier: int) -> SpeedProfile:
        if tier not in self.shadow_profiles:
            if tier in self.profiles:
                p = self.profiles[tier]
                self.shadow_profiles[tier] = SpeedProfile(
                    tier_speed=p.tier_speed,
                    delay_ms=p.delay_ms,
                    samples=p.samples,
                    std_ms=p.std_ms,
                    trusted=p.trusted,
                    epoch=p.epoch,
                    accepted_samples=p.accepted_samples,
                    rejected_samples=p.rejected_samples,
                    mad_ms=p.mad_ms,
                    p25_ms=p.p25_ms,
                    p75_ms=p.p75_ms,
                    recent_delays=list(p.recent_delays),
                )
            else:
                self.shadow_profiles[tier] = SpeedProfile(
                    tier_speed=tier,
                    delay_ms=self.get_latency_for_speed(tier),
                    samples=0,
                    std_ms=0.0,
                    trusted=False,
                    epoch=CURRENT_CALIBRATION_EPOCH,
                )
        return self.shadow_profiles[tier]

    def get_tier(self, speed: float, tolerance: float = 20.0) -> int:
        base_tier = int(round(speed / self.step_size) * self.step_size)
        if base_tier in self.profiles:
            return base_tier
        if self.profiles:
            closest_existing = min(self.profiles.keys(), key=lambda t: abs(t - speed))
            if abs(closest_existing - speed) <= tolerance:
                return closest_existing
        return base_tier

    def get_profile(self, speed: float) -> SpeedProfile:
        tier = self.get_tier(speed)
        if tier not in self.profiles:
            closest_delay = self.get_latency_for_speed(speed)
            new_p = SpeedProfile(
                tier_speed=tier,
                delay_ms=closest_delay,
                samples=0,
                std_ms=0.0,
                trusted=False,
                epoch=CURRENT_CALIBRATION_EPOCH,
            )
            if not self.freeze:
                self.profiles[tier] = new_p
            else:
                return new_p
        return self.profiles[tier]

    def get_latency_for_speed(self, speed: float) -> float:
        tier = self.get_tier(speed)
        if tier in self.profiles and self.profiles[tier].trusted:
            return self.profiles[tier].delay_ms

        # If tier is untrusted or missing, find the nearest trusted tier
        trusted_tiers = [p for p in self.profiles.values() if p.trusted and p.samples >= 5]
        if trusted_tiers:
            closest = min(trusted_tiers, key=lambda p: abs(p.tier_speed - speed))
            return closest.delay_ms

        # Fallback to any profile or global default
        if self.profiles:
            closest = min(self.profiles.values(), key=lambda p: abs(p.tier_speed - speed))
            return closest.delay_ms
        return 122.5

    def record_hit(
        self,
        speed: float,
        actual_delay_ms: float,
        error_deg: float,
        outcome: str,
        plateau_found: bool,
        target_angle: Optional[float] = None,
        is_aborted: bool = False,
        scheduler_error_ms: Optional[float] = None,
        prediction_lateness_ms: Optional[float] = None,
        target_tier: Any = _UNSPECIFIED,
        speed_at_lock: Any = _UNSPECIFIED,
        last_frame_age_ms: Optional[float] = None,
        trigger_reason: Optional[str] = None,
        duration_ms: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Records a confirmed hit into the exact speed tier used at trigger.
        Filters out noise: only trains on verified freeze plateaus (GREAT/GOOD/MISS).
        Rejects learning if target_tier differs from fire tier (TIER_CHANGED).
        Rejects learning if OS/thread scheduler jitter or prediction lateness exceeded 8.0ms.
        Rejects learning if speed diverged significantly between lock and fire (> 25.0°/s).
        Rejects learning on MISS unless it is a verified CLEAN TIMING MISS.
        """
        if is_aborted or not plateau_found or outcome not in ("GREAT", "GOOD", "MISS") or speed <= 15.0:
            return None

        # If caller did not specify tier/lock (e.g. direct test calls), default from speed
        if target_tier is _UNSPECIFIED and speed_at_lock is _UNSPECIFIED:
            target_tier = self.get_tier(speed)
            speed_at_lock = speed

        # Reject learning unconditionally if trigger was fired without confirmed speed lock
        if target_tier is None or speed_at_lock is None:
            return {
                "tier": None,
                "rejected": True,
                "reason": "NO_SPEED_LOCK",
            }

        # Train the EXACT locked tier that was physically used at FIRE!
        tier = int(target_tier)
        profile = self.get_profile(tier)

        # 1. Speed divergence check: reject if speed changed significantly between lock and fire (> 25.0°/s)
        if speed_at_lock is not None and abs(speed - speed_at_lock) > 25.0:
            if not self.freeze:
                profile.rejected_samples += 1
                if self.save_to_disk:
                    self.save()
            else:
                shadow_prof = self._get_shadow_profile(tier)
                shadow_prof.rejected_samples += 1
            return {
                "tier": tier,
                "rejected": True,
                "reason": f"SPEED_DIVERGENCE (lock={speed_at_lock:.1f}, fire={speed:.1f})",
                "frozen": self.freeze,
            }

        # 2. Tier changed between lock and fire: reject cross-tier contamination
        tier_at_fire = self.get_tier(speed)
        if target_tier is not None and int(target_tier) != tier_at_fire:
            if not self.freeze:
                profile.rejected_samples += 1
                if self.save_to_disk:
                    self.save()
            else:
                shadow_prof = self._get_shadow_profile(tier)
                shadow_prof.rejected_samples += 1
            return {
                "tier": tier,
                "rejected": True,
                "reason": f"TIER_CHANGED (lock_tier={target_tier}, fire_tier={tier_at_fire})",
                "frozen": self.freeze,
            }

        # 3. Scheduler jitter check / prediction lateness check:
        effective_jitter = prediction_lateness_ms if prediction_lateness_ms is not None else scheduler_error_ms
        if effective_jitter is not None and abs(effective_jitter) > 8.0:
            if not self.freeze:
                profile.rejected_samples += 1
                if self.save_to_disk:
                    self.save()
            else:
                shadow_prof = self._get_shadow_profile(tier)
                shadow_prof.rejected_samples += 1
            return {
                "tier": tier,
                "rejected": True,
                "reason": f"EXCESSIVE_JITTER ({effective_jitter:+.1f}ms)",
                "frozen": self.freeze,
            }

        # 4. Stale frame check at fire:
        if last_frame_age_ms is not None and last_frame_age_ms > 35.0:
            if not self.freeze:
                profile.rejected_samples += 1
                if self.save_to_disk:
                    self.save()
            else:
                shadow_prof = self._get_shadow_profile(tier)
                shadow_prof.rejected_samples += 1
            return {
                "tier": tier,
                "rejected": True,
                "reason": f"STALE_FRAME ({last_frame_age_ms:.1f}ms)",
                "frozen": self.freeze,
            }

        # 5. On MISS: Only CLEAN TIMING MISS is accepted into training
        if outcome == "MISS":
            miss_cat, miss_detail = classify_miss(
                reason=trigger_reason or "UNKNOWN",
                scheduler_error_ms=effective_jitter,
                last_frame_age_ms=last_frame_age_ms,
                duration_ms=duration_ms,
                speed_at_lock=speed_at_lock,
                speed_at_fire=speed,
                tier_at_lock=target_tier,
                tier_at_fire=tier_at_fire,
                error_deg=error_deg,
                error_ms=(error_deg / speed * 1000.0) if speed > 0 else 0.0,
            )
            if miss_cat != "CLEAN TIMING MISS":
                if not self.freeze:
                    profile.rejected_samples += 1
                    if self.save_to_disk:
                        self.save()
                else:
                    shadow_prof = self._get_shadow_profile(tier)
                    shadow_prof.rejected_samples += 1
                return {
                    "tier": tier,
                    "rejected": True,
                    "reason": f"{miss_cat}: {miss_detail}",
                    "frozen": self.freeze,
                }

        # Calculate ideal physical delay for this specific hit:
        ideal_delay = actual_delay_ms + (error_deg / speed) * 1000.0

        # Physical safety bounds: reject extreme physical anomalies (< 45ms or > 250ms)
        if not (45.0 <= ideal_delay <= 250.0):
            if not self.freeze:
                profile.rejected_samples += 1
                if self.save_to_disk:
                    self.save()
            else:
                shadow_prof = self._get_shadow_profile(tier)
                shadow_prof.rejected_samples += 1
            return {
                "tier": tier,
                "rejected": True,
                "reason": f"PHYSICAL_OUTLIER ({ideal_delay:.1f}ms)",
                "frozen": self.freeze,
            }

        if self.freeze:
            # FREEZE MODE: Active runtime profile is strictly IMMUTABLE.
            # Active delay, N, trust, statistics must NEVER change.
            # Hypothetical training is calculated strictly on shadow profile.
            shadow_prof = self._get_shadow_profile(tier)
            shadow_prof.accepted_samples += 1
            shadow_prof.epoch = CURRENT_CALIBRATION_EPOCH
            shadow_step = shadow_prof.add_ideal_delay(ideal_delay, batch_size=5)
            return {
                "tier": profile.tier_speed,
                "ideal_delay": ideal_delay,
                "profile_delay": profile.delay_ms,
                "shadow_delay": shadow_prof.delay_ms,
                "step": None,
                "shadow_step": shadow_step,
                "samples": profile.samples,
                "shadow_samples": shadow_prof.samples,
                "accepted_samples": profile.accepted_samples,
                "rejected_samples": profile.rejected_samples,
                "confidence": profile.confidence,
                "trusted": profile.trusted,
                "is_new": False,
                "rejected": False,
                "frozen": True,
            }

        profile.accepted_samples += 1
        profile.epoch = CURRENT_CALIBRATION_EPOCH
        is_new = (profile.samples == 0)
        step = profile.add_ideal_delay(ideal_delay, batch_size=5)

        # Track sector position bias
        if target_angle is not None:
            sec = (int(target_angle // 90) * 90) % 360
            error_ms = (error_deg / speed) * 1000.0
            self.sector_errors[sec].append(error_ms)

        if self.save_to_disk:
            self.save()
        return {
            "tier": profile.tier_speed,
            "ideal_delay": ideal_delay,
            "profile_delay": profile.delay_ms,
            "step": step,
            "samples": profile.samples,
            "accepted_samples": profile.accepted_samples,
            "rejected_samples": profile.rejected_samples,
            "confidence": profile.confidence,
            "trusted": profile.trusted,
            "is_new": is_new,
            "rejected": False,
            "frozen": False,
        }

    def bootstrap_from_replays(self, replays_dir: Path):
        """Pre-seeds profiles from clean historical replays as epoch 1 shadow priors."""
        if not replays_dir.exists():
            return

        replay_paths = sorted(replays_dir.glob("check_*.json"))
        oldreplays = replays_dir.parent / "oldreplays"
        if oldreplays.exists():
            known_names = {p.name for p in replay_paths}
            for p in sorted(oldreplays.glob("check_*.json")):
                if p.name not in known_names:
                    replay_paths.append(p)

        bins: Dict[int, List[float]] = collections.defaultdict(list)
        for p in replay_paths:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                ev = data.get("evaluation") or {}
                tr = data.get("trigger") or {}
                outcome = ev.get("outcome", "UNKNOWN")
                plateau = ev.get("plateau_found", False)
                spd = tr.get("speed_deg_s")
                lat = tr.get("latency_ms")
                err_deg = ev.get("error_deg")

                if outcome in ("GREAT", "GOOD", "MISS") and plateau and spd and lat is not None and err_deg is not None:
                    ideal = lat + (err_deg / spd) * 1000.0
                    if 60.0 <= ideal <= 220.0:
                        tier = self.get_tier(spd)
                        bins[tier].append(ideal)
                        t_ang = ev.get("target_angle")
                        if t_ang is not None:
                            sec = (int(t_ang // 90) * 90) % 360
                            self.sector_errors[sec].append((err_deg / spd) * 1000.0)
            except Exception:
                continue

        for tier, vals in bins.items():
            if not vals:
                continue
            med = float(np.median(vals))
            std = float(np.std(vals))
            # Historical replays are marked as epoch 1 shadow priors (untrusted until re-verified in epoch 2)
            self.profiles[tier] = SpeedProfile(
                tier_speed=tier,
                delay_ms=round(med, 2),
                samples=len(vals),
                std_ms=round(std, 2),
                trusted=False,
                recent_delays=vals[-20:],
                accepted_samples=len(vals),
                rejected_samples=0,
                epoch=1,
            )

    def save(self):
        if not self.save_to_disk:
            return
        try:
            data = {
                "schema_version": self.schema_version,
                "calibration_epoch": self.calibration_epoch,
                "tiers": {str(k): v.to_dict() for k, v in sorted(self.profiles.items())}
            }
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def load(self):
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.schema_version = data.get("schema_version", 1)
            self.calibration_epoch = data.get("calibration_epoch", 1)
            tiers = data.get("tiers", {})
            for k, v in tiers.items():
                self.profiles[int(k)] = SpeedProfile.from_dict(v)
        except Exception:
            pass

    def get_summary(self) -> str:
        lines = [
            "================== SKILL CHECK SPEED PROFILES ==================",
        ]
        if not self.profiles:
            lines.append("Нет данных по скоростям.")
        else:
            for tier in sorted(self.profiles.keys()):
                p = self.profiles[tier]
                status_tag = " [TRUSTED]" if p.trusted else f" [SHADOW ep{p.epoch}]"
                lines.append(
                    f"speed ~{p.tier_speed:3d}:  N={p.samples:<4d} (acc={p.accepted_samples}/rej={p.rejected_samples}) "
                    f"delay={p.delay_ms:5.1f} ms (std={p.std_ms:4.1f} ms, MAD={p.mad_ms:4.1f} ms) "
                    f"p25/p75=[{p.p25_ms:5.1f}, {p.p75_ms:5.1f}]  conf={p.confidence:<6s}{status_tag}"
                )

        lines.append("----------------------------------------------------------------")
        lines.append("POSITION BIAS:")
        for sec in [0, 90, 180, 270]:
            errs = self.sector_errors.get(sec, [])
            if len(errs) >= 5:
                med_err = float(np.median(errs))
                sign = "+" if med_err >= 0 else ""
                flag = " ← подозрительно" if abs(med_err) > 4.0 else ""
                lines.append(f"{sec:3d}–{sec+90:3d}°:    {sign}{med_err:4.1f} ms (N={len(errs)}){flag}")
            else:
                lines.append(f"{sec:3d}–{sec+90:3d}°:    недостаточно данных (N={len(errs)})")
        lines.append("================================================================")
        return "\n".join(lines)


class AdaptiveLatencyLearner:
    """
    Self-learning controller that automatically calibrates hardware latency compensation
    based on real-world hit feedback in live gameplay.
    Features:
    - Circular freeze-plateau verification (handles 360°/0° boundary wrapping).
    - Deadband filtering (does not chase sub-frame quantization noise).
    - Expanded valid error range (adapts on all Good hits and errors up to 48.0°).
    - Frenzy immunity (does not corrupt latency during rapid multi-check bursts).
    - Rolling median smoothing for stable convergence.
    - Safety clamping within physical display limits [45ms..160ms].
    """

    def __init__(
        self,
        initial_latency_ms: float = 108.0,
        learning_rate: float = 0.25,
        max_step_ms: float = 6.0,
        min_latency_ms: float = 45.0,
        max_latency_ms: float = 155.0,
        deadband_deg: float = 1.20,
        max_valid_error_deg: float = 30.0,
        auto_save: bool = True,
        verbose: bool = True,
        profiles_path: Optional[Path] = None,
        use_speed_profiles: bool = False,
        config_path: Optional[Path] = None,
        save_to_disk: bool = True,
        freeze: bool = False,
    ):
        self.min_latency_ms = float(min_latency_ms)
        self.max_latency_ms = float(max_latency_ms)
        self.latency_ms = max(self.min_latency_ms, min(self.max_latency_ms, float(initial_latency_ms)))
        self.learning_rate = learning_rate
        self.max_step_ms = max_step_ms
        self.deadband_deg = deadband_deg
        self.max_valid_error_deg = max_valid_error_deg
        self.freeze = bool(freeze)
        self.auto_save = auto_save and not self.freeze
        self.save_to_disk = bool(save_to_disk) and not self.freeze
        self.verbose = verbose
        self.use_speed_profiles = use_speed_profiles
        self.config_path = Path(config_path) if config_path else CONFIG_PATH

        # Multi-speed profiling manager:
        # CRITICAL ISOLATION: When use_speed_profiles is False (unit tests / default) or save_to_disk is False,
        # never bootstrap production replays and NEVER save to production speed_profiles.json!
        save_profiles_to_disk = bool(save_to_disk) and self.use_speed_profiles and not self.freeze
        if self.use_speed_profiles:
            self.speed_profiles = SpeedProfileManager(
                profiles_path=profiles_path,
                auto_bootstrap=True,
                save_to_disk=save_profiles_to_disk,
                freeze=self.freeze,
            )
        else:
            self.speed_profiles = SpeedProfileManager(
                profiles_path=profiles_path,
                auto_bootstrap=False,
                save_to_disk=False,
                freeze=self.freeze,
            )

        # Statistics
        self.total_evals = 0
        self.great_hits = 0
        self.good_hits = 0
        self.miss_hits = 0
        self.history_errors_deg: collections.deque = collections.deque(maxlen=10)
        self.recent_valid_errors_deg: collections.deque = collections.deque(maxlen=5)

        # Current active episode tracking
        self.pending_hit: Optional[Dict[str, Any]] = None
        self.post_hit_samples: List[Tuple[float, float]] = []

        # Session base speed causal tracker
        self.base_speed_tracker = SessionBaseSpeedTracker(initial_seed=278.0)

    def on_trigger(
        self,
        press_time: float,
        target_angle: float,
        speed_deg_s: float,
        white_zone: Optional[Dict[str, float]],
        black_zone: Optional[Dict[str, float]],
        is_frenzy: bool = False,
        trajectory_samples: int = 10,
        used_latency_ms: Optional[float] = None,
        scheduler_error_ms: Optional[float] = None,
        prediction_lateness_ms: Optional[float] = None,
        selected_speed_tier: Any = _UNSPECIFIED,
        speed_at_lock: Any = _UNSPECIFIED,
        last_frame_age_ms: Optional[float] = None,
        trigger_reason: Optional[str] = None,
        duration_ms: float = 0.0,
    ):
        """Records a trigger event to observe post-hit registration."""
        if selected_speed_tier is _UNSPECIFIED and speed_at_lock is _UNSPECIFIED:
            if not self.use_speed_profiles:
                selected_speed_tier = self.speed_profiles.get_tier(speed_deg_s) if self.speed_profiles else int(round(speed_deg_s / 25.0) * 25)
                speed_at_lock = speed_deg_s
            else:
                selected_speed_tier = None
                speed_at_lock = None

        self.pending_hit = {
            "press_time": press_time,
            "target_angle": target_angle,
            "speed_deg_s": speed_deg_s,
            "white_zone": white_zone,
            "black_zone": black_zone,
            "is_frenzy": is_frenzy,
            "trajectory_samples": trajectory_samples,
            "used_latency_ms": used_latency_ms if used_latency_ms is not None else self.latency_ms,
            "scheduler_error_ms": scheduler_error_ms,
            "prediction_lateness_ms": prediction_lateness_ms,
            "selected_speed_tier": selected_speed_tier,
            "speed_at_lock": speed_at_lock,
            "last_frame_age_ms": last_frame_age_ms,
            "trigger_reason": trigger_reason,
            "duration_ms": duration_ms,
        }
        self.post_hit_samples = []

    def observe_sample(self, t: float, needle_angle: Optional[float], needle_strength: float = 20.0):
        """Accumulates post-hit samples to calculate exact registration position."""
        if self.pending_hit is not None and needle_angle is not None and needle_strength >= 15.0:
            self.post_hit_samples.append((t, float(needle_angle)))

    def conclude_check(self, is_aborted: bool = False, abort_reason: str = "") -> Optional[Dict[str, Any]]:
        """
        Analyzes the completed check, detects the frozen needle position in Roblox,
        evaluates hit outcome, and adapts latency compensation.
        Returns evaluation dict or None if no pending hit.
        """
        if self.pending_hit is None:
            return None

        hit_info = self.pending_hit
        self.pending_hit = None

        press_time = hit_info["press_time"]
        target_angle = hit_info["target_angle"]
        speed = hit_info["speed_deg_s"]
        white_zone = hit_info["white_zone"]
        black_zone = hit_info["black_zone"]
        is_frenzy = hit_info.get("is_frenzy", False)

        hit_angle = None
        plateau_found = False
        plateau_time = None

        # 1. Search for genuine frozen needle plateau in post-hit samples.
        # When Roblox registers Space, the needle remains stationary for >= 3 frames (spread < 2.0°).
        # We search from the latest samples backwards (since freeze happens until UI despawn).
        if len(self.post_hit_samples) >= 3:
            for w_size in (4, 3):
                if len(self.post_hit_samples) >= w_size:
                    for i in range(len(self.post_hit_samples) - w_size, -1, -1):
                        window = [self.post_hit_samples[k][1] for k in range(i, i + w_size)]
                        spread, center = circular_spread_and_center(window)
                        if spread < 2.0:
                            # CRITICAL: Verify that the needle actually FROZE and did NOT continue forward!
                            # In 60 FPS Roblox captured at 120 FPS, 2-3 duplicate frames occur mid-flight.
                            subsequent = [self.post_hit_samples[k][1] for k in range(i + w_size, len(self.post_hit_samples))]
                            if subsequent:
                                max_forward = max((a - center + 180.0) % 360.0 - 180.0 for a in subsequent)
                                if max_forward > 3.0:
                                    continue  # False plateau: needle kept rotating!

                            # Reject obvious noise plateaus that are far away from any zone (>max_dist_w away)
                            dist_w = abs((center - (white_zone["center"] if white_zone else target_angle) + 180.0) % 360.0 - 180.0)
                            max_dist_w = max(55.0, (speed * 0.18))
                            if dist_w <= max_dist_w:
                                hit_angle = center
                                plateau_found = True
                                plateau_time = self.post_hit_samples[i][0]
                                break
                if plateau_found:
                    break
        elif len(self.post_hit_samples) == 2:
            spread, center = circular_spread_and_center([s[1] for s in self.post_hit_samples])
            if spread < 1.5:
                dist_w = abs((center - (white_zone["center"] if white_zone else target_angle) + 180.0) % 360.0 - 180.0)
                max_dist_w = max(55.0, (speed * 0.18))
                if dist_w <= max_dist_w:
                    hit_angle = center
                    plateau_found = True
                    plateau_time = self.post_hit_samples[0][0]

        # 2. If no freeze plateau was found:
        # Needle never confirmed frozen (despawned instantly, passed through, or user missed completely).
        if not plateau_found:
            if self.post_hit_samples:
                hit_angle = float(self.post_hit_samples[-1][1])
            else:
                hit_angle = None

        # Automatic detection of aborted checks (e.g. player released LMB):
        # Only declare early freeze abort if it happened in < 30ms (physically impossible for input arrival)
        if not is_aborted and plateau_found:
            if plateau_time is not None and (plateau_time - press_time) < 0.030:
                is_aborted = True
                abort_reason = "EARLY_FREEZE_LMB"

        # 3. Determine outcome (GREAT / GOOD / MISS / ABORTED) using robust arc wrapping
        outcome = "UNKNOWN"
        in_great = False
        in_good = False

        if hit_angle is not None:
            if white_zone:
                in_great = is_angle_in_arc(hit_angle, white_zone["start"], white_zone["end"], tol_start=0.5, tol_end=0.5)
                if in_great and plateau_found:
                    outcome = "GREAT"

            if not in_great and black_zone:
                # In Roblox, the Good zone is contiguous with the Great zone.
                # Visual segmentation often produces a small 1-3 deg seam gap between white["end"] and black["start"].
                effective_start = black_zone["start"]
                if white_zone:
                    gap = (black_zone["start"] - white_zone["end"] + 360.0) % 360.0
                    if gap <= 10.0:
                        effective_start = white_zone["end"]
                in_good = is_angle_in_arc(hit_angle, effective_start, black_zone["end"], tol_start=0.5, tol_end=0.5)
                if in_good and plateau_found:
                    outcome = "GOOD"

            if not in_great and not in_good:
                if is_aborted:
                    outcome = "ABORTED_LMB" if "LMB" in abort_reason else "ABORTED"
                elif plateau_found:
                    outcome = "MISS"
                else:
                    outcome = "UNCONFIRMED"
            elif not plateau_found:
                outcome = "UNCONFIRMED"
        else:
            if is_aborted:
                outcome = "ABORTED_LMB" if "LMB" in abort_reason else "ABORTED"
            else:
                outcome = "UNCONFIRMED"

        # Handle aborted checks: NEVER adjust latency, do NOT record as player/bot miss
        if is_aborted:
            outcome = "ABORTED_LMB" if "LMB" in abort_reason else "ABORTED"
            reason_str = f"🛑 ПРЕРВАНО ({abort_reason})"
            great_pct = (self.great_hits / self.total_evals * 100.0) if self.total_evals > 0 else 100.0
            w_s = white_zone.get("start") if white_zone else None
            w_e = white_zone.get("end") if white_zone else None
            w_c = white_zone.get("center") if white_zone else None
            w_w = white_zone.get("width") if white_zone else None
            result = {
                "hit_angle": hit_angle if hit_angle is not None else target_angle,
                "target_angle": target_angle,
                "error_deg": 0.0,
                "error_ms": 0.0,
                "outcome": outcome,
                "plateau_found": plateau_found,
                "prev_latency_ms": self.latency_ms,
                "new_latency_ms": self.latency_ms,
                "adjustment_ms": 0.0,
                "total_evals": self.total_evals,
                "great_pct": great_pct,
                "white_start": w_s,
                "white_end": w_e,
                "white_center": w_c,
                "white_width": w_w,
                "center_error_deg": 0.0,
                "center_error_ms": 0.0,
                "entry_edge_error_deg": 0.0,
                "entry_edge_error_ms": 0.0,
            }
            if self.verbose:
                print(f"[AI LEARNER] {outcome} | {reason_str} (адаптация сохранена на {self.latency_ms:.1f}мс)")
            return result

        # Update stats for valid completed checks
        if plateau_found:
            self.total_evals += 1
            if in_great:
                self.great_hits += 1
            elif in_good:
                self.good_hits += 1
            else:
                self.miss_hits += 1

        # 4. Calculate angular error relative to target setpoint (center of Great zone)
        # positive error = hit late (needle moved past target towards black)
        # negative error = hit early (needle arrived before target)
        if hit_angle is not None:
            error_deg = (hit_angle - target_angle + 180.0) % 360.0 - 180.0
            error_ms = (error_deg / speed) * 1000.0 if speed > 0 else 0.0
        else:
            error_deg = 0.0
            error_ms = 0.0
        self.history_errors_deg.append(error_deg)

        # Precise geometry calculations for white zone center and entry edge
        w_start = white_zone.get("start") if white_zone else None
        w_end = white_zone.get("end") if white_zone else None
        w_center = white_zone.get("center") if white_zone else None
        w_width = white_zone.get("width") if white_zone else None
        if w_center is None and w_start is not None and w_width is not None:
            w_center = (w_start + w_width / 2.0) % 360.0
        if w_width is None and w_start is not None and w_end is not None:
            w_width = (w_end - w_start + 360.0) % 360.0

        if hit_angle is not None and w_center is not None:
            center_error_deg = (hit_angle - w_center + 180.0) % 360.0 - 180.0
            center_error_ms = (center_error_deg / speed) * 1000.0 if speed > 0 else 0.0
        else:
            center_error_deg = error_deg
            center_error_ms = error_ms

        if hit_angle is not None and w_start is not None:
            entry_edge_error_deg = (hit_angle - w_start + 180.0) % 360.0 - 180.0
            entry_edge_error_ms = (entry_edge_error_deg / speed) * 1000.0 if speed > 0 else 0.0
        else:
            entry_edge_error_deg = 0.0
            entry_edge_error_ms = 0.0

        great_pct = (self.great_hits / self.total_evals) * 100.0 if self.total_evals > 0 else 0.0
        prev_latency = self.latency_ms
        latency_adjustment_ms = 0.0

        used_latency = hit_info.get("used_latency_ms", self.latency_ms)
        sched_err = hit_info.get("scheduler_error_ms")

        # 5. ROBUST ADAPTIVE LEARNING LOGIC:
        # If speed profiles are enabled, per-check single-step adaptation on self.latency_ms
        # is bypassed. Speed profiles learn exclusively via batch medians per discrete speed tier.
        eff_max_error_deg = max(self.max_valid_error_deg, (speed * 0.12))
        is_outlier = (not in_good and not in_great and abs(error_deg) > eff_max_error_deg)
        traj_count = hit_info.get("trajectory_samples", 10)

        if self.use_speed_profiles:
            tier = self.speed_profiles.get_tier(speed)
            if is_frenzy:
                reason_str = "⚡ FRENZY (заморозка адаптации на серии)"
            elif not plateau_found:
                reason_str = "⏸ Замирание не зафиксировано (пропуск)"
            elif traj_count < 5:
                reason_str = f"⏸ Мало точек траектории ({traj_count} < 5) — пропуск адаптации"
            elif is_outlier:
                reason_str = f"⚠️ Выброс отброшен ({error_deg:+.1f}° > {eff_max_error_deg:.1f}°)"
            elif abs(error_deg) <= self.deadband_deg:
                self.recent_valid_errors_deg.append(error_deg)
                reason_str = f"🎯 ИДЕАЛЬНО В ЦЕНТРЕ (ошибка {error_deg:+.1f}° в зоне допуска)"
            else:
                self.recent_valid_errors_deg.append(error_deg)
                reason_str = f"📊 Speed Profile ~{tier}°/с (использовано {used_latency:.1f}мс, baseline {self.latency_ms:.1f}мс)"
        else:
            # Legacy single-delay adaptation
            if is_frenzy:
                reason_str = "⚡ FRENZY (заморозка адаптации на серии)"
            elif not plateau_found:
                reason_str = "⏸ Замирание не зафиксировано (пропуск)"
            elif traj_count < 5:
                reason_str = f"⏸ Мало точек траектории ({traj_count} < 5) — пропуск адаптации"
            elif is_outlier:
                reason_str = f"⚠️ Выброс отброшен ({error_deg:+.1f}° > {eff_max_error_deg:.1f}°)"
            elif abs(error_deg) <= self.deadband_deg:
                self.recent_valid_errors_deg.append(error_deg)
                reason_str = f"🎯 ИДЕАЛЬНО В ЦЕНТРЕ (ошибка {error_deg:+.1f}° в зоне допуска)"
            else:
                self.recent_valid_errors_deg.append(error_deg)
                excess_deg = abs(error_deg) - self.deadband_deg * 0.5
                excess_ms = (excess_deg / speed) * 1000.0

                # Only adjust online legacy latency if Great/Good or if verified CLEAN TIMING MISS
                is_clean_miss = (outcome != "MISS")
                if outcome == "MISS":
                    miss_cat, miss_detail = classify_miss(
                        reason=hit_info.get("trigger_reason", "UNKNOWN"),
                        scheduler_error_ms=sched_err,
                        last_frame_age_ms=hit_info.get("last_frame_age_ms"),
                        duration_ms=hit_info.get("duration_ms", 0.0),
                        speed_at_lock=hit_info.get("speed_at_lock"),
                        speed_at_fire=speed,
                        tier_at_lock=hit_info.get("selected_speed_tier"),
                        tier_at_fire=self.speed_profiles.get_tier(speed) if self.speed_profiles else None,
                        error_deg=error_deg,
                        error_ms=error_ms,
                    )
                    is_clean_miss = (miss_cat == "CLEAN TIMING MISS")

                if is_clean_miss:
                    if not in_great:
                        step_cap = self.max_step_ms
                        lr = self.learning_rate
                    else:
                        step_cap = min(1.5, self.max_step_ms * 0.5)
                        lr = self.learning_rate * 0.75

                    raw_adj = lr * excess_ms
                    step_mag = min(step_cap, max(0.15, raw_adj))
                    latency_adjustment_ms = math.copysign(step_mag, error_deg)

                    if not self.freeze:
                        new_latency = self.latency_ms + latency_adjustment_ms
                        self.latency_ms = max(self.min_latency_ms, min(self.max_latency_ms, new_latency))

                        if self.auto_save and abs(self.latency_ms - prev_latency) >= 0.05:
                            self._save_config()

                    sign_str = "+" if latency_adjustment_ms >= 0 else ""
                    type_str = "Good (опоздание)" if in_good else ("Miss (недолёт)" if not in_great else "Great (балансировка)")
                    freeze_tag = " [FREEZE - IMMUTABLE]" if self.freeze else ""
                    reason_str = f"🧠 Коррекция {type_str}{freeze_tag}: {prev_latency:.1f}мс -> {self.latency_ms:.1f}мс ({sign_str}{latency_adjustment_ms:.1f}мс)"
                else:
                    latency_adjustment_ms = 0.0
                    reason_str = f"⚠️ Промах [{miss_cat}] ({miss_detail}) — задержка не изменена"

        result = {
            "hit_angle": hit_angle,
            "target_angle": target_angle,
            "error_deg": error_deg,
            "error_ms": error_ms,
            "outcome": outcome,
            "plateau_found": plateau_found,
            "prev_latency_ms": prev_latency,
            "new_latency_ms": self.latency_ms,
            "adjustment_ms": 0.0 if self.freeze else latency_adjustment_ms,
            "hypothetical_adjustment_ms": latency_adjustment_ms if self.freeze else 0.0,
            "total_evals": self.total_evals,
            "great_pct": great_pct,
            "session_base_speed": self.base_speed_tracker.current_prior,
            "white_start": w_start,
            "white_end": w_end,
            "white_center": w_center,
            "white_width": w_width,
            "center_error_deg": center_error_deg,
            "center_error_ms": center_error_ms,
            "entry_edge_error_deg": entry_edge_error_deg,
            "entry_edge_error_ms": entry_edge_error_ms,
        }

        # Strictly causal session base speed update on clean completed checks
        clean_sched = sched_err is not None and abs(sched_err) <= 2.0
        if outcome in ("GREAT", "GOOD") and clean_sched:
            if not self.freeze:
                self.base_speed_tracker.record_completed_check(
                    outcome=outcome,
                    reason=hit_info.get("trigger_reason", "SCHEDULED"),
                    scheduler_error_ms=sched_err,
                    speed_at_fire=speed,
                    duration_ms=hit_info.get("duration_ms", 0.0),
                )
            result["session_base_speed"] = self.base_speed_tracker.current_prior

        if outcome == "MISS":
            miss_cat, miss_detail = classify_miss(
                reason=hit_info.get("trigger_reason", "UNKNOWN"),
                scheduler_error_ms=sched_err,
                last_frame_age_ms=hit_info.get("last_frame_age_ms"),
                duration_ms=hit_info.get("duration_ms", 0.0),
                speed_at_lock=hit_info.get("speed_at_lock"),
                speed_at_fire=speed,
                tier_at_lock=hit_info.get("selected_speed_tier"),
                tier_at_fire=self.speed_profiles.get_tier(speed) if self.speed_profiles else None,
                error_deg=error_deg,
                error_ms=error_ms,
            )
            result["miss_category"] = miss_cat
            result["miss_detail"] = miss_detail

        # Record into discrete Speed Profile Manager using the ACTUAL used latency at FIRE
        # CRITICAL ISOLATION: Only record when use_speed_profiles is explicitly True!
        if plateau_found and self.use_speed_profiles and self.speed_profiles is not None:
            prof_res = self.speed_profiles.record_hit(
                speed=speed,
                actual_delay_ms=used_latency,
                error_deg=error_deg,
                outcome=outcome,
                plateau_found=plateau_found,
                target_angle=target_angle,
                is_aborted=is_aborted,
                scheduler_error_ms=sched_err,
                prediction_lateness_ms=hit_info.get("prediction_lateness_ms"),
                target_tier=hit_info.get("selected_speed_tier"),
                speed_at_lock=hit_info.get("speed_at_lock"),
                last_frame_age_ms=hit_info.get("last_frame_age_ms"),
                trigger_reason=hit_info.get("trigger_reason"),
                duration_ms=hit_info.get("duration_ms", 0.0),
            )
            if prof_res:
                result["speed_profile"] = prof_res
                if prof_res.get("rejected"):
                    if self.verbose:
                        print(
                            f"[AI LEARNER] ⚠️ Speed Profile ~{prof_res['tier']}°/с образец отклонён от обучения: "
                            f"{prof_res.get('reason')}"
                        )
                elif prof_res.get("step") is not None:
                    step_v = prof_res["step"]
                    sign_v = "+" if step_v >= 0 else ""
                    if self.verbose:
                        print(
                            f"[AI LEARNER] 📊 Speed Profile ~{prof_res['tier']}°/с обновлён: "
                            f"{prof_res['profile_delay']:.1f}мс ({sign_v}{step_v:.1f}мс) | "
                            f"N={prof_res['samples']} (confidence={prof_res['confidence']})"
                        )
                elif prof_res.get("is_new") and self.verbose:
                    print(
                        f"[AI LEARNER] ⚡ Новая скорость ~{prof_res['tier']}°/с (shadow mode): "
                        f"предложено delay={prof_res['profile_delay']:.1f}мс (confidence={prof_res['confidence']}, N={prof_res['samples']})"
                    )

        sign_err = "+" if error_deg >= 0 else ""
        hit_str = f"{hit_angle:.1f}°" if hit_angle is not None else "N/A"
        if self.verbose:
            print(
                f"[AI LEARNER] Попадание: {outcome} | Факт={hit_str}, Цель={target_angle:.1f}° (ошибка: {sign_err}{error_deg:.1f}° / {sign_err}{error_ms:.1f}мс) "
                f"| {reason_str} | Статистика: {self.great_hits}/{self.total_evals} GREAT ({great_pct:.1f}%)"
            )
        return result

    def get_latency_for_speed(self, speed: float) -> float:
        """Returns the calibrated or nearest trusted latency for a given speed tier."""
        return self.speed_profiles.get_latency_for_speed(speed)

    def get_training_summary(self) -> str:
        """Generates structured training summary of all speed profiles and position bias."""
        return self.speed_profiles.get_summary()

    def print_training_summary(self):
        """Prints the end-of-session training summary."""
        print(self.get_training_summary())

    def _save_config(self):
        if not self.auto_save or not self.save_to_disk:
            return
        try:
            if self.config_path and self.config_path.exists():
                cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
                cfg["latency_ms"] = round(self.latency_ms, 1)
                self.config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        except Exception:
            pass
