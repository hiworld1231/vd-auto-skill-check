#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPLAYS_DIR = ROOT / "replays"

def check_false_plateaus():
    json_files = sorted(REPLAYS_DIR.glob("check_*_*.json"))
    print("=== CHECKING FOR FALSE PLATEAUS (NEEDLE MOVED AFTER 'HIT' ANGLE) ===")
    
    false_plateau_count = 0
    total_plateau_count = 0
    
    for f in json_files:
        try:
            ep = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
            
        ev = ep.get("evaluation") or {}
        hit_ang = ev.get("hit_angle")
        if not ev.get("plateau_found") or hit_ang is None:
            continue
            
        total_plateau_count += 1
        
        frames = ep.get("frames") or []
        active = [fr for fr in frames if not fr.get("is_pre_roll") and fr.get("needle_angle") is not None]
        
        # Check all frames after the hit_angle was recorded
        # Find where hit_angle was in active frames
        hit_idx = None
        for i, fr in enumerate(active):
            if abs((fr["needle_angle"] - hit_ang + 180.0) % 360.0 - 180.0) < 1.5:
                hit_idx = i
                break
                
        if hit_idx is not None and hit_idx < len(active) - 1:
            later_frames = active[hit_idx + 1:]
            max_forward_motion = 0.0
            final_ang = active[-1]["needle_angle"]
            for lfr in later_frames:
                diff = (lfr["needle_angle"] - hit_ang) % 360.0
                if 0 < diff < 180:
                    max_forward_motion = max(max_forward_motion, diff)
                    
            if max_forward_motion > 5.0:
                false_plateau_count += 1
                outcome = ev.get("outcome")
                print(f"\nFALSE PLATEAU in {ep['check_id']} ({outcome}):")
                print(f"  Detected hit_angle: {hit_ang:.1f}°, but needle moved +{max_forward_motion:.1f}° later!")
                print(f"  Frame at detected hit: idx={active[hit_idx]['idx']}, t_rel={active[hit_idx]['time_rel_ms']}ms")
                print(f"  Final frame in check: needle={final_ang:.1f}°, t_rel={active[-1]['time_rel_ms']}ms")
                w_zone = (ep.get("locked_zones") or {}).get("white") or {}
                print(f"  White zone: {w_zone.get('start')}° - {w_zone.get('end')}° (center {w_zone.get('center')}°)")

    print("\n" + "=" * 80)
    print(f"TOTAL EVALUATED: {total_plateau_count} plateaus found.")
    print(f"FALSE PLATEAUS (needle continued moving >5°): {false_plateau_count} ({(false_plateau_count/total_plateau_count*100.0) if total_plateau_count else 0:.1f}%)")

if __name__ == "__main__":
    check_false_plateaus()
