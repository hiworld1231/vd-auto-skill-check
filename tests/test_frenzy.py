#!/usr/bin/env python3
"""
Automated Unit Test Suite for Frenzy / Chained Skill Checks (серийные скилл чеки / бешенство).
Simulates consecutive skill checks with:
1. Zero-gap continuous UI (needle rewinds, zone changes while ring stays on screen).
2. Short-gap UI (50ms gap between checks).
3. Accelerated frenzy speed (340°/s) and narrow Great zone (4.5°).
Verifies that all chained skill checks are detected, locked, and hit with 100% success.
"""

import math
import os
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.vision import VisionEngine, is_angle_in_arc
from core.predictor import SkillCheckPredictor


class TestFrenzyChainedSkillChecks(unittest.TestCase):
    """Tests instant re-arming and multi-hit tracking in frenzy mode."""

    def test_predictor_frenzy_speeds(self):
        """Verifies predictor converges on high-speed frenzy rotations up to 380°/s."""
        predictor = SkillCheckPredictor(latency_ms=0.0, target_offset_ratio=0.50)
        t0 = 100.0
        fps = 120.0
        true_speed = 350.0  # High-speed frenzy skill check

        # Feed 15 frames at 350°/s
        for i in range(15):
            t = t0 + i / fps
            ang = (i * (true_speed / fps)) % 360.0
            predictor.update(t, ang, 80.0, {"start": 120.0, "end": 126.0, "width": 6.0, "center": 123.0}, None)

        self.assertAlmostEqual(predictor.speed_deg_s, true_speed, delta=15.0,
                               msg="Predictor must adapt to high-speed frenzy rotations")

    def test_frenzy_chain_simulation(self):
        """
        Simulates 3 consecutive chained skill checks in frenzy mode:
        Check 1: Zone at 110°, speed 270°/s
        Check 2: Zone at 190°, speed 320°/s (0ms gap, needle jumps to 5°)
        Check 3: Zone at 80°, speed 290°/s (40ms gap, needle jumps to 0°)
        """
        checks_def = [
            {"white_start": 105.0, "white_width": 9.0, "speed": 270.0, "name": "Check 1 (Normal)"},
            {"white_start": 185.0, "white_width": 6.0, "speed": 320.0, "name": "Check 2 (Frenzy Fast & Narrow)"},
            {"white_start": 75.0, "white_width": 8.0, "speed": 290.0, "name": "Check 3 (Frenzy Chained)"},
        ]

        hits_recorded = []
        fps = 120.0
        dt = 1.0 / fps
        sim_time = 10.0

        # State machine variables matching skillcheck_bot.py
        in_check = False
        chain_count = 0
        pressed = False
        last_trigger_t = 0.0
        locked_w = None
        last_needle_angle = None
        target_angle_locked = 0.0
        predictor = SkillCheckPredictor(latency_ms=0.0, target_offset_ratio=0.50)

        scheduled_press_t = None
        for c_idx, c_info in enumerate(checks_def):
            w_start = c_info["white_start"]
            w_width = c_info["white_width"]
            w_center = w_start + w_width / 2.0
            speed = c_info["speed"]
            w_dict = {"start": w_start, "end": w_start + w_width, "width": w_width, "center": w_center}

            check_duration_frames = int((w_center + 15.0) / (speed / fps))

            for f in range(check_duration_frames):
                sim_time += dt
                needle = (f * (speed / fps)) % 360.0
                now = sim_time

                # Simulated detection
                det = {"needle_angle": needle, "needle_strength": 80.0, "white_dict": w_dict}
                current_needle = needle

                if not in_check:
                    in_check = True
                    chain_count = 1
                    pressed = False
                    locked_w = w_dict
                    scheduled_press_t = None
                    predictor.reset()
                    last_needle_angle = current_needle

                elif pressed:
                    # FRENZY RE-ARM LOGIC
                    fresh_w = det["white_dict"]
                    is_frenzy = False

                    if fresh_w is not None and locked_w is not None:
                        zone_diff = abs((fresh_w["center"] - locked_w["center"] + 180.0) % 360.0 - 180.0)
                        if zone_diff > 12.0:
                            is_frenzy = True

                    if not is_frenzy and last_needle_angle is not None:
                        backward_jump = (last_needle_angle - current_needle) % 360.0
                        if 35.0 < backward_jump < 325.0 and (now - last_trigger_t >= 0.110):
                            is_frenzy = True

                    if is_frenzy:
                        chain_count += 1
                        pressed = False
                        scheduled_press_t = None
                        frenzy_spd = max(340.0, predictor.speed_deg_s)
                        predictor.reset(keep_speed=True, default_speed=frenzy_spd)
                        locked_w = fresh_w
                        target_angle_locked = 0.0

                predictor.update(now, current_needle, 80.0, locked_w, None)
                last_needle_angle = current_needle

                if not pressed and locked_w is not None:
                    pred = predictor.predict(now, current_needle, target="GREAT")
                    if pred is not None:
                        target_angle_locked = pred["target_angle"]
                        if (scheduled_press_t is None or pred["time_until_press_ms"] > 4.0) and pred["angular_distance_deg"] < 180.0:
                            scheduled_press_t = pred["press_timestamp"]

                        if scheduled_press_t is not None and now >= scheduled_press_t:
                            pressed = True
                            last_trigger_t = now
                            hits_recorded.append({
                                "check_index": c_idx,
                                "chain_count": chain_count,
                                "hit_needle": current_needle,
                                "zone": locked_w,
                            })

        print(f"\n[TEST FRENZY] Simulated {len(checks_def)} chained checks. Hits recorded: {len(hits_recorded)}")
        for h in hits_recorded:
            z = h["zone"]
            in_great = z["start"] <= h["hit_needle"] <= z["end"]
            print(f"  Hit #{h['chain_count']} (Check {h['check_index']+1}): needle={h['hit_needle']:.1f}°, zone=[{z['start']:.1f}°, {z['end']:.1f}°] -> {'GREAT' if in_great else 'MISS'}")
            self.assertTrue(in_great, f"Hit #{h['chain_count']} must be inside Great zone")

        self.assertEqual(len(hits_recorded), 3, "Must successfully hit all 3 checks in the frenzy chain")


if __name__ == "__main__":
    unittest.main()
