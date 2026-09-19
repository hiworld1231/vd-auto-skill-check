#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPLAYS_DIR = ROOT / "replays"

def inspect_post_trigger_trajectories():
    json_files = sorted(REPLAYS_DIR.glob("check_*_*.json"))
    
    print("=== INSPECTING POST-TRIGGER TRAJECTORIES ===")
    for f in json_files:
        try:
            ep = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
            
        trig = ep.get("trigger") or {}
        if not trig.get("fired"):
            continue
            
        ev = ep.get("evaluation") or {}
        outcome = ev.get("outcome")
        w_zone = (ep.get("locked_zones") or {}).get("white") or {}
        w_ctr = w_zone.get("center")
        
        frames = ep.get("frames") or []
        active = [fr for fr in frames if not fr.get("is_pre_roll") and fr.get("needle_angle") is not None]
        
        # Find frames around trigger
        post_trig = []
        passed_trig = False
        for fr in active:
            if fr.get("time_until_press_ms") is not None and fr["time_until_press_ms"] <= 0:
                passed_trig = True
            if passed_trig:
                post_trig.append((fr["time_rel_ms"], fr["needle_angle"]))
                
        if len(post_trig) >= 3 and outcome in ("MISS", "GOOD", "ABORTED", "ABORTED_LMB"):
            print(f"\n{ep['check_id']} ({outcome}, ch#{ep.get('chain_count')}, tgt={trig.get('target_angle')}°, w_ctr={w_ctr}°):")
            print(f"  Trigger est_angle: {trig.get('est_angle_at_trigger'):.1f}°, hit_angle: {ev.get('hit_angle')}")
            for t_rel, ang in post_trig[:12]:
                print(f"    t_rel={t_rel:6.1f}ms: needle={ang:5.1f}°")

if __name__ == "__main__":
    inspect_post_trigger_trajectories()
