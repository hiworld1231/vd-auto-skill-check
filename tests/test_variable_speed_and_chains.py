#!/usr/bin/env python3
"""
Automated Unit Tests for Variable Speeds, Close/Distant Zone Spawns (Top vs Bottom),
and Continuous Multi-Check Frenzy Chains.
Tests the exact issues reported by the user:
1. Multi-speed support (20°/s slow hexes/perks, 220°/s, 275°/s normal, 380°/s frenzy).
2. Top vs Bottom symmetry (spawns close to 270° vs 180° away at 90°).
3. Frenzy continuous chains with rapid re-arming (90ms) and speed preservation.
4. False plateau rejection (no adaptation on mid-flight moving needles).
"""

import math
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.predictor import SkillCheckPredictor, DEFAULT_SPEED_DEG_S
from core.learner import AdaptiveLatencyLearner
from core.vision import is_angle_in_arc


class TestVariableSpeedAndChains(unittest.TestCase):
    """Tests variable speeds, frenzy chains, and top/bottom zone geometry."""

    def test_multi_speed_adaptation(self):
        """Tests that predictor quickly and accurately adapts across all speed tiers: 20°/s, 220°/s, 275°/s, 380°/s, 550°/s, 750°/s, 1000°/s."""
        speeds_to_test = [20.0, 220.0, 275.0, 380.0, 550.0, 750.0, 1000.0]
        fps = 120.0
        dt = 1.0 / fps

        for target_speed in speeds_to_test:
            predictor = SkillCheckPredictor(latency_ms=60.0, target_offset_ratio=0.50)
            t0 = 100.0
            
            # Feed 10 frames
            for i in range(10):
                t = t0 + i * dt
                ang = (i * target_speed * dt) % 360.0
                predictor.update(t, ang, 80.0, {"start": 100.0, "end": 110.0, "width": 10.0, "center": 105.0}, None)

            # Check convergence
            self.assertAlmostEqual(
                predictor.speed_deg_s, target_speed, delta=target_speed * 0.08 + 2.0,
                msg=f"Predictor must adapt to {target_speed}°/s within 8% tolerance, got {predictor.speed_deg_s:.1f}°/s"
            )

    def test_top_vs_bottom_press_accuracy(self):
        """
        Tests that both TOP zone (spawning close to needle: 20° away)
        and BOTTOM zone (spawning 180° away) are accurately scheduled and hit within Great zone.
        """
        latency_ms = 70.0
        fps = 120.0
        dt = 1.0 / fps
        speed = 275.0

        test_cases = [
            {"name": "TOP (Close)", "start_ang": 270.0, "w_start": 295.0, "w_end": 305.0, "w_center": 300.0},
            {"name": "BOTTOM (Distant)", "start_ang": 270.0, "w_start": 85.0, "w_end": 95.0, "w_center": 90.0},
        ]

        for tc in test_cases:
            predictor = SkillCheckPredictor(latency_ms=latency_ms, target_offset_ratio=0.50)
            w_zone = {"start": tc["w_start"], "end": tc["w_end"], "width": 10.0, "center": tc["w_center"]}
            
            t0 = 100.0
            scheduled_press_t = None
            pressed = False
            hit_needle = None

            for f in range(int(360.0 / (speed / fps))):
                t = t0 + f * dt
                needle = (tc["start_ang"] + speed * (f * dt)) % 360.0

                predictor.update(t, needle, 80.0, w_zone, None)

                if not pressed:
                    pred = predictor.predict(t, needle, target="GREAT")
                    if pred is not None:
                        if scheduled_press_t is None or pred["time_until_press_ms"] > 4.0:
                            scheduled_press_t = pred["press_timestamp"]

                        if scheduled_press_t is not None and t >= scheduled_press_t:
                            pressed = True
                            # Needle position in game at registration (press time + physical latency):
                            reg_t = scheduled_press_t + (latency_ms / 1000.0)
                            hit_needle = (tc["start_ang"] + speed * (reg_t - t0)) % 360.0
                            break

            self.assertTrue(pressed, f"{tc['name']}: Trigger must fire")
            self.assertIsNotNone(hit_needle)
            in_great = is_angle_in_arc(hit_needle, tc["w_start"], tc["w_end"], tol_start=0.5, tol_end=0.5)
            err_from_center = (hit_needle - tc["w_center"] + 180.0) % 360.0 - 180.0
            print(f"\n[TEST ZONE GEOMETRY] {tc['name']}: hit={hit_needle:.1f}°, zone=[{tc['w_start']:.1f}°, {tc['w_end']:.1f}°], err={err_from_center:+.1f}°")
            self.assertTrue(in_great, f"{tc['name']} must land inside Great zone (hit={hit_needle:.1f}°, zone=[{tc['w_start']:.1f}°, {tc['w_end']:.1f}°])")

    def test_rapid_frenzy_chain_hit_rate(self):
        """
        Tests 5 consecutive rapid frenzy checks where speed is 370°/s and zones appear quickly.
        Verifies 100% Great hit rate with 90ms re-arming and speed preservation.
        """
        fps = 120.0
        dt = 1.0 / fps
        frenzy_speed = 370.0
        latency_ms = 70.0

        chains = [
            {"w_start": 80.0, "w_end": 89.0, "w_center": 84.5},
            {"w_start": 210.0, "w_end": 219.0, "w_center": 214.5},
            {"w_start": 45.0, "w_end": 54.0, "w_center": 49.5},   # Challenging close-range zone
            {"w_start": 320.0, "w_end": 329.0, "w_center": 324.5},
            {"w_start": 130.0, "w_end": 139.0, "w_center": 134.5},
        ]

        hits = []
        sim_t = 100.0

        predictor = SkillCheckPredictor(latency_ms=latency_ms, target_offset_ratio=0.50)
        
        for c_idx, c_info in enumerate(chains):
            w_zone = {"start": c_info["w_start"], "end": c_info["w_end"], "width": 9.0, "center": c_info["w_center"]}
            start_needle = (c_info["w_center"] - 140.0) % 360.0 if c_idx > 0 else 270.0
            
            # Reset with keep_speed=True for chains
            if c_idx > 0:
                predictor.reset(keep_speed=True, default_speed=frenzy_speed)
            else:
                predictor.reset(keep_speed=False)

            pressed = False
            scheduled_press_t = None
            hit_needle = None

            for f in range(int(360.0 / (frenzy_speed / fps))):
                sim_t += dt
                needle = (start_needle + frenzy_speed * (f * dt)) % 360.0

                predictor.update(sim_t, needle, 85.0, w_zone, None)

                if not pressed:
                    pred = predictor.predict(sim_t, needle, target="GREAT")
                    if pred is not None:
                        if scheduled_press_t is None or pred["time_until_press_ms"] > 4.0:
                            scheduled_press_t = pred["press_timestamp"]

                        if scheduled_press_t is not None and sim_t >= scheduled_press_t:
                            pressed = True
                            reg_t = scheduled_press_t + (latency_ms / 1000.0)
                            hit_needle = (start_needle + frenzy_speed * (reg_t - (sim_t - f * dt))) % 360.0
                            hits.append({
                                "chain": c_idx + 1,
                                "hit": hit_needle,
                                "zone": w_zone,
                            })
                            break

            self.assertTrue(pressed, f"Chain #{c_idx+1} must be triggered")

        print(f"\n[TEST FRENZY CHAINS] Simulated {len(chains)} rapid chains at {frenzy_speed}°/s:")
        for h in hits:
            z = h["zone"]
            in_great = is_angle_in_arc(h["hit"], z["start"], z["end"], tol_start=0.5, tol_end=0.5)
            print(f"  Chain #{h['chain']}: hit={h['hit']:.1f}°, zone=[{z['start']:.1f}°, {z['end']:.1f}°] -> {'GREAT' if in_great else 'MISS'}")
            self.assertTrue(in_great, f"Chain #{h['chain']} must be Great hit")

    def test_false_plateau_rejection(self):
        """Tests that moving needles with duplicate frames are rejected and not treated as freeze."""
        learner = AdaptiveLatencyLearner(initial_latency_ms=75.0, auto_save=False)
        learner.on_trigger(
            press_time=1.0,
            target_angle=100.0,
            speed_deg_s=270.0,
            white_zone={"start": 95.0, "end": 105.0, "width": 10.0, "center": 100.0},
            black_zone={"start": 105.0, "end": 145.0, "width": 40.0, "center": 125.0},
            is_frenzy=False,
        )

        # Feed moving samples with duplicate frame in middle:
        # [80.0, 80.0, 80.0 (duplicate/stutter), 85.0, 92.0, 98.0, 100.0 (freeze), 100.0, 100.0, 100.0]
        learner.observe_sample(1.02, 80.0)
        learner.observe_sample(1.04, 80.0)
        learner.observe_sample(1.06, 80.0)  # Old learner would falsely stop here at 80.0°!
        learner.observe_sample(1.08, 85.0)
        learner.observe_sample(1.10, 92.0)
        learner.observe_sample(1.12, 98.0)
        learner.observe_sample(1.14, 100.0)
        learner.observe_sample(1.16, 100.0)
        learner.observe_sample(1.18, 100.0)
        learner.observe_sample(1.20, 100.0)

        res = learner.conclude_check()
        self.assertTrue(res["plateau_found"])
        self.assertAlmostEqual(res["hit_angle"], 100.0, delta=0.5,
                               msg="Must pick the TRUE freeze at 100.0°, rejecting false mid-flight stutter at 80.0°")

    def test_high_speed_perk_hit_accuracy(self):
        """
        Tests end-to-end press scheduling and hit accuracy on high-speed perk checks (550°/s, 750°/s, 1000°/s).
        Ensures that despite extreme needle speed, the bot adapts within frames and lands cleanly in the GREAT zone.
        """
        high_speeds = [550.0, 750.0, 1000.0]
        fps = 120.0
        dt = 1.0 / fps
        latency_ms = 70.0

        for perk_speed in high_speeds:
            predictor = SkillCheckPredictor(latency_ms=latency_ms, target_offset_ratio=0.50)
            w_zone = {"start": 85.0, "end": 95.0, "width": 10.0, "center": 90.0}
            start_ang = 270.0
            t0 = 200.0
            scheduled_press_t = None
            pressed = False
            hit_needle = None

            total_frames = int(360.0 / (perk_speed / fps)) + 5
            for f in range(total_frames):
                t = t0 + f * dt
                needle = (start_ang + perk_speed * (f * dt)) % 360.0
                predictor.update(t, needle, 80.0, w_zone, None)

                if not pressed:
                    pred = predictor.predict(t, needle, target="GREAT")
                    if pred is not None:
                        if scheduled_press_t is None or pred["time_until_press_ms"] > 4.0:
                            scheduled_press_t = pred["press_timestamp"]

                        if scheduled_press_t is not None and t >= scheduled_press_t:
                            pressed = True
                            reg_t = scheduled_press_t + (latency_ms / 1000.0)
                            hit_needle = (start_ang + perk_speed * (reg_t - t0)) % 360.0
                            break

            self.assertTrue(pressed, f"Perk speed {perk_speed}°/s: Trigger must fire")
            self.assertIsNotNone(hit_needle)
            in_great = is_angle_in_arc(hit_needle, w_zone["start"], w_zone["end"], tol_start=0.5, tol_end=0.5)
            err = (hit_needle - w_zone["center"] + 180.0) % 360.0 - 180.0
            print(f"\n[TEST HIGH PERK SPEED] {perk_speed}°/s: hit={hit_needle:.1f}°, zone=[{w_zone['start']}°, {w_zone['end']}°], err={err:+.2f}°")
            self.assertTrue(in_great, f"Perk speed {perk_speed}°/s must land inside Great zone (hit={hit_needle:.1f}°)")

    def test_continuous_frenzy_chain_hits_without_backward_jump(self):
        """
        Simulates rapid continuous skill checks (Merciless Storm / Frenzy) where:
        1. The needle NEVER jumps backwards; it continues forward seamlessly at 450°/s.
        2. New zones relocate 60°-80° ahead.
        3. Re-arming occurs on Frame 1 without arbitrary 80ms lockout.
        4. Verifies 100% GREAT hit rate across all 4 chained checks at true physical latency.
        """
        speed = 450.0
        fps = 120.0
        dt = 1.0 / fps
        latency_ms = 70.0
        phys_latency_s = latency_ms / 1000.0

        zones = [
            {"start": 85.0, "end": 95.0, "width": 10.0, "center": 90.0},
            {"start": 165.0, "end": 175.0, "width": 10.0, "center": 170.0},
            {"start": 250.0, "end": 260.0, "width": 10.0, "center": 255.0},
            {"start": 335.0, "end": 345.0, "width": 10.0, "center": 340.0},
        ]

        hits = []
        current_zone_idx = 0
        current_needle = 0.0
        t = 100.0
        last_trigger_t = -1.0
        pressed = False
        scheduled_press_t = None
        chain_count = 1

        predictor = SkillCheckPredictor(latency_ms=latency_ms, target_offset_ratio=0.50)

        for step in range(600):
            t += dt
            current_needle = (current_needle + speed * dt) % 360.0

            active_zone = zones[current_zone_idx]

            # Re-arm immediately on Frame 1 when new zone relocates (>15° difference)
            if pressed and (current_zone_idx + 1 < len(zones)):
                next_zone = zones[current_zone_idx + 1]
                zone_diff = abs((next_zone["center"] - active_zone["center"] + 180.0) % 360.0 - 180.0)
                if zone_diff > 15.0:
                    current_zone_idx += 1
                    chain_count += 1
                    pressed = False
                    scheduled_press_t = None
                    active_zone = zones[current_zone_idx]
                    predictor = SkillCheckPredictor(latency_ms=latency_ms, target_offset_ratio=0.50)
                    predictor.reset(keep_speed=True, default_speed=speed)

            predictor.update(t, current_needle, 80.0, active_zone, None)

            if not pressed:
                pred = predictor.predict(t, current_needle, target="GREAT")
                if pred is not None:
                    # Emergency overdue fire does not require history gate
                    if pred.get("should_press_now", False):
                        pressed = True
                        last_trigger_t = t
                        hit_pos = (current_needle + speed * phys_latency_s) % 360.0
                        hits.append((chain_count, hit_pos, active_zone))
                    elif scheduled_press_t is None or pred["time_until_press_ms"] > 4.0:
                        scheduled_press_t = pred["press_timestamp"]

                    if scheduled_press_t is not None and t >= scheduled_press_t and not pressed:
                        pressed = True
                        last_trigger_t = t
                        # Needle position in game at registration (press time + physical latency):
                        hit_pos = (current_needle + speed * phys_latency_s) % 360.0
                        hits.append((chain_count, hit_pos, active_zone))

            if len(hits) == len(zones):
                break

        self.assertEqual(len(hits), len(zones), f"Must trigger all {len(zones)} checks in continuous chain")
        print(f"\n[TEST MERCILESS CHAIN] Completed {len(hits)} continuous checks at {speed}°/s:")
        for ch, hit_pos, z in hits:
            in_great = is_angle_in_arc(hit_pos, z["start"], z["end"], tol_start=0.5, tol_end=0.5)
            err = (hit_pos - z["center"] + 180.0) % 360.0 - 180.0
            print(f"  Chain #{ch}: hit={hit_pos:.1f}°, zone=[{z['start']}°, {z['end']}°], err={err:+.1f}° -> {'GREAT' if in_great else 'MISS'}")
            self.assertTrue(in_great, f"Chain #{ch} must hit Great zone")

    def test_duplicate_frame_rejection_in_predictor(self):
        """Verifies predictor.update() returns True on valid new frames and False on duplicate frames."""
        predictor = SkillCheckPredictor(latency_ms=60.0, target_offset_ratio=0.50)
        zone = {"start": 100.0, "end": 110.0, "width": 10.0, "center": 105.0}

        # First frame is valid
        self.assertTrue(predictor.update(1.0, 50.0, 80.0, zone, None))

        # Duplicate frame (delta < 0.10°) must return False
        self.assertFalse(predictor.update(1.008, 50.05, 80.0, zone, None))

        # Frame with movement must return True
        self.assertTrue(predictor.update(1.016, 52.5, 80.0, zone, None))

        # Weak signal (spark spike noise) must return False
        self.assertFalse(predictor.update(1.024, 55.0, 5.0, zone, None))

    def test_overdue_target_emergency_fire_without_history_gate(self):
        """
        Verifies overdue targets fire immediately within the widened overdue recovery window
        (min(-45.0, -speed * 0.10)) even without 4 history points.
        """
        predictor = SkillCheckPredictor(latency_ms=60.0, target_offset_ratio=0.50)
        zone = {"start": 100.0, "end": 110.0, "width": 10.0, "center": 105.0}

        # Single frame past target setpoint (target is 105.0, needle at 125.0 -> direct_diff = -20.0°)
        predictor.update(1.0, 125.0, 80.0, zone, None)
        pred = predictor.predict(1.0, 125.0, target="GREAT")

        self.assertIsNotNone(pred)
        self.assertTrue(pred["should_press_now"], "Must fire immediately on overdue target without requiring 4 history points")
        self.assertEqual(pred["time_until_press_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
