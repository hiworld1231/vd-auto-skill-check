#!/usr/bin/env python3
import json
import math
import os
import re
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPLAYS_DIR = ROOT / "replays"
RECORDINGS_DIR = ROOT / "recordings"

def analyze_replays():
    print("=" * 80)
    print("ANALYZING ALL REPLAY JSON FILES IN replays/")
    print("=" * 80)

    json_files = sorted(REPLAYS_DIR.glob("check_*_*.json"))
    print(f"Total replay JSON files: {len(json_files)}")

    episodes = []
    for f in json_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["_filename"] = f.name
            episodes.append(data)
        except Exception as e:
            print(f"Error reading {f.name}: {e}")

    outcomes = defaultdict(list)
    by_chain = defaultdict(list)
    by_quadrant = defaultdict(list)
    
    # Analyze each episode
    for ep in episodes:
        ev = ep.get("evaluation") or {}
        outcome = ev.get("outcome", "UNKNOWN")
        chain = ep.get("chain_count", 1)
        outcomes[outcome].append(ep)
        by_chain[chain].append(ep)

        # White zone position
        w_zone = (ep.get("locked_zones") or {}).get("white") or {}
        w_center = w_zone.get("center")
        if w_center is not None:
            # Quadrants:
            # Right: 315..360, 0..45
            # Bottom (снизу): 45..135 (around 90°)
            # Left: 135..225 (around 180°)
            # Top (сверху): 225..315 (around 270°)
            if 45 <= w_center < 135:
                quad = "BOTTOM (45°-135°)"
            elif 135 <= w_center < 225:
                quad = "LEFT (135°-225°)"
            elif 225 <= w_center < 315:
                quad = "TOP (225°-315°)"
            else:
                quad = "RIGHT (315°-45°)"
            by_quadrant[quad].append(ep)

    print("\n--- OUTCOME BREAKDOWN ---")
    for k, v in outcomes.items():
        print(f"  {k}: {len(v)} ({len(v)/len(episodes)*100:.1f}%)")

    print("\n--- OUTCOMES BY CHAIN COUNT ---")
    for chain, eps in sorted(by_chain.items()):
        chain_outcomes = defaultdict(int)
        for ep in eps:
            chain_outcomes[(ep.get("evaluation") or {}).get("outcome", "UNKNOWN")] += 1
        print(f"  Chain #{chain}: total {len(eps)} -> {dict(chain_outcomes)}")

    print("\n--- OUTCOMES BY ZONE QUADRANT ---")
    for quad, eps in sorted(by_quadrant.items()):
        quad_outcomes = defaultdict(int)
        errors = []
        for ep in eps:
            ev = ep.get("evaluation") or {}
            quad_outcomes[ev.get("outcome", "UNKNOWN")] += 1
            if ev.get("error_deg") is not None and ev.get("outcome") in ("GREAT", "GOOD", "MISS"):
                errors.append(ev["error_deg"])
        mean_err = np.mean(errors) if errors else 0
        print(f"  {quad}: total {len(eps)} -> {dict(quad_outcomes)}, mean_err_deg: {mean_err:+.2f}°")

    print("\n--- DETAILED INSPECTION OF EVERY MISS & ABORT ---")
    for ep in episodes:
        ev = ep.get("evaluation") or {}
        outcome = ev.get("outcome", "UNKNOWN")
        if outcome in ("MISS", "UNKNOWN") or "ABORT" in outcome:
            trig = ep.get("trigger") or {}
            w_zone = (ep.get("locked_zones") or {}).get("white") or {}
            b_zone = (ep.get("locked_zones") or {}).get("black") or {}
            print(f"\nEpisode: {ep['_filename']}")
            print(f"  Outcome: {outcome}, Chain: {ep.get('chain_count')}, Duration: {ep.get('duration_ms')}ms")
            print(f"  Configured Latency: {ep.get('configured_latency_ms'):.1f}ms")
            print(f"  White zone: {w_zone.get('start')}° - {w_zone.get('end')}° (center {w_zone.get('center')}°)")
            print(f"  Black zone: {b_zone.get('start')}° - {b_zone.get('end')}°")
            speed_s = trig.get('speed_deg_s')
            speed_str = f"{speed_s:.1f}°/s" if speed_s is not None else "N/A"
            print(f"  Trigger: fired={trig.get('fired')}, reason={trig.get('reason')}, target_angle={trig.get('target_angle')}, speed={speed_str}")
            print(f"  Evaluation: hit_angle={ev.get('hit_angle')}, target={ev.get('target_angle')}, err_deg={ev.get('error_deg')}, plateau={ev.get('plateau_found')}")
            # Count active frames
            frames = ep.get("frames") or []
            active_frames = [f for f in frames if not f.get("is_pre_roll")]
            print(f"  Total frames: {len(frames)}, Active frames: {len(active_frames)}")
            if active_frames:
                first_f = active_frames[0]
                last_f = active_frames[-1]
                print(f"  First active: t_rel={first_f.get('time_rel_ms')}ms, needle={first_f.get('needle_angle')}°")
                print(f"  Last active: t_rel={last_f.get('time_rel_ms')}ms, needle={last_f.get('needle_angle')}°")

    print("\n--- MEASURING ACTUAL NEEDLE SPEED IN EACH REPLAY ---")
    speeds_by_quadrant = defaultdict(list)
    speeds_by_chain = defaultdict(list)
    all_measured_speeds = []

    for ep in episodes:
        frames = ep.get("frames") or []
        active_frames = [f for f in frames if not f.get("is_pre_roll") and f.get("needle_angle") is not None]
        # Only take frames before trigger
        trig = ep.get("trigger") or {}
        target_ang = trig.get("target_angle")
        w_zone = (ep.get("locked_zones") or {}).get("white") or {}
        w_center = w_zone.get("center")

        pre_press_frames = []
        for f in active_frames:
            pre_press_frames.append(f)
            # stop around trigger
            if f.get("time_until_press_ms") is not None and f["time_until_press_ms"] <= 0:
                break

        if len(pre_press_frames) >= 5:
            ts = [f["time_rel_ms"] / 1000.0 for f in pre_press_frames]
            angs = [f["needle_angle"] for f in pre_press_frames]
            # unwrap
            unwrapped = np.unwrap(np.array(angs) * np.pi / 180.0) * 180.0 / np.pi
            poly = np.polyfit(ts, unwrapped, 1)
            measured_speed = poly[0]
            if 150 <= measured_speed <= 450:
                all_measured_speeds.append(measured_speed)
                chain = ep.get("chain_count", 1)
                speeds_by_chain[chain].append(measured_speed)
                if w_center is not None:
                    if 45 <= w_center < 135:
                        quad = "BOTTOM (45°-135°)"
                    elif 135 <= w_center < 225:
                        quad = "LEFT (135°-225°)"
                    elif 225 <= w_center < 315:
                        quad = "TOP (225°-315°)"
                    else:
                        quad = "RIGHT (315°-45°)"
                    speeds_by_quadrant[quad].append(measured_speed)

    print(f"Overall measured speed: mean={np.mean(all_measured_speeds):.2f}°/s, std={np.std(all_measured_speeds):.2f}°/s, min={np.min(all_measured_speeds):.2f}°/s, max={np.max(all_measured_speeds):.2f}°/s")
    print("Speeds by quadrant:")
    for q, spds in sorted(speeds_by_quadrant.items()):
        print(f"  {q}: mean={np.mean(spds):.2f}°/s, std={np.std(spds):.2f}°/s (n={len(spds)})")
    print("Speeds by chain count:")
    for c, spds in sorted(speeds_by_chain.items()):
        print(f"  Chain #{c}: mean={np.mean(spds):.2f}°/s, std={np.std(spds):.2f}°/s (n={len(spds)})")

if __name__ == "__main__":
    analyze_replays()
