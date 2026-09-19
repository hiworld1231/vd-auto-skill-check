#!/usr/bin/env python3
import json
import math
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
REPLAYS_DIR = ROOT / "replays"

def analyze_needle_and_geometry():
    json_files = sorted(REPLAYS_DIR.glob("check_*_*.json"))
    
    print("=== NEEDLE SPAWN ANGLE & TRAJECTORY ANALYSIS ===")
    spawn_angles_chain1 = []
    spawn_angles_frenzy = []
    
    centers_x = []
    centers_y = []
    radii = []

    quadrant_data = defaultdict(list)

    for f in json_files:
        try:
            ep = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        
        frames = ep.get("frames") or []
        active = [fr for fr in frames if not fr.get("is_pre_roll") and fr.get("needle_angle") is not None]
        chain = ep.get("chain_count", 1)
        w_zone = (ep.get("locked_zones") or {}).get("white") or {}
        w_center = w_zone.get("center")
        w_start = w_zone.get("start")
        w_end = w_zone.get("end")
        ev = ep.get("evaluation") or {}
        trig = ep.get("trigger") or {}

        if active:
            first_needle = active[0]["needle_angle"]
            if chain == 1:
                spawn_angles_chain1.append((ep["check_id"], first_needle, w_center))
            else:
                spawn_angles_frenzy.append((ep["check_id"], chain, first_needle, w_center))

            for fr in active:
                if fr.get("cx") is not None and fr.get("cy") is not None:
                    centers_x.append(fr["cx"])
                    centers_y.append(fr["cy"])

        # Detailed breakdown of quadrant accuracy
        if w_center is not None and ev.get("hit_angle") is not None:
            # angle distance from spawn to white
            hit_ang = ev["hit_angle"]
            target_ang = trig.get("target_angle") or ev.get("target_angle") or w_center
            err_deg = (hit_ang - target_ang + 180.0) % 360.0 - 180.0
            
            quadrant_data[f"w_center_{int(w_center//30)*30}"].append({
                "check_id": ep["check_id"],
                "chain": chain,
                "w_center": w_center,
                "hit_ang": hit_ang,
                "target_ang": target_ang,
                "err_deg": err_deg,
                "outcome": ev.get("outcome"),
                "speed": trig.get("speed_deg_s"),
                "lat": ep.get("configured_latency_ms"),
                "first_needle": active[0]["needle_angle"] if active else None,
            })

    print(f"Centers: cx mean={np.mean(centers_x):.2f}, std={np.std(centers_x):.2f} | cy mean={np.mean(centers_y):.2f}, std={np.std(centers_y):.2f}")
    
    print("\n--- CHAIN 1 SPAWN ANGLES (first 15) ---")
    for cid, ang, wc in spawn_angles_chain1[:15]:
        print(f"  {cid}: spawn={ang:.1f}°, zone_center={wc if wc is not None else 'None'}")

    print("\n--- FRENZY SPAWN ANGLES ---")
    for cid, ch, ang, wc in spawn_angles_frenzy[:20]:
        print(f"  {cid} (chain #{ch}): spawn={ang:.1f}°, zone_center={wc if wc is not None else 'None'}")

    print("\n--- ERROR BY ZONE ANGLE SECTOR (every 30°) ---")
    for sector in sorted(quadrant_data.keys(), key=lambda s: int(s.split('_')[-1])):
        items = quadrant_data[sector]
        errors = [it["err_deg"] for it in items if it["err_deg"] is not None]
        speeds = [it["speed"] for it in items if it["speed"] is not None]
        outcomes = defaultdict(int)
        for it in items:
            outcomes[it["outcome"]] += 1
        print(f"Sector {sector}: n={len(items)}, mean_err={np.mean(errors):+.2f}°, std_err={np.std(errors):.2f}°, mean_speed={np.mean(speeds):.1f}°/s -> {dict(outcomes)}")

if __name__ == "__main__":
    analyze_needle_and_geometry()
