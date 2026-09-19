#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPLAYS_DIR = ROOT / "replays"

def test_speed_convergence():
    json_files = sorted(REPLAYS_DIR.glob("check_*_*.json"))
    
    print("=== TESTING VELOCITY CONVERGENCE ON REPLAY EPISODES ===")
    
    errors_old = []
    errors_new = []
    
    for f in json_files:
        try:
            ep = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
            
        frames = ep.get("frames") or []
        active = [fr for fr in frames if not fr.get("is_pre_roll") and fr.get("needle_angle") is not None]
        
        # We need at least 10 active frames to have a ground truth speed
        if len(active) < 10:
            continue
            
        ts_all = [fr["time_rel_ms"] / 1000.0 for fr in active[:20]]
        angs_all = [fr["needle_angle"] for fr in active[:20]]
        unwrapped_all = np.unwrap(np.array(angs_all) * np.pi / 180.0) * 180.0 / np.pi
        true_speed = np.polyfit(ts_all, unwrapped_all, 1)[0]
        
        if not (100 <= true_speed <= 450):
            continue
            
        # Simulate OLD update logic frame by frame
        history_old = []
        speed_old = 270.3
        speed_old_at_frame4 = 270.3
        speed_old_at_frame8 = 270.3
        
        # Simulate NEW update logic
        history_new = []
        speed_new = 270.3
        speed_new_at_frame4 = 270.3
        speed_new_at_frame8 = 270.3
        
        for i, fr in enumerate(active[:15]):
            t = fr["time_rel_ms"] / 1000.0
            ang = fr["needle_angle"]
            
            # Filter duplicates
            if history_old:
                step = (ang - history_old[-1][1] + 180.0) % 360.0 - 180.0
                if abs(step) < 0.25:
                    continue
            history_old.append((t, ang))
            
            # OLD logic
            if len(history_old) >= 4:
                recent = history_old[-14:]
                ts = np.array([x[0] for x in recent], dtype=np.float64)
                angs = np.array([x[1] for x in recent], dtype=np.float64)
                unw = np.unwrap(angs * (np.pi / 180.0)) * (180.0 / np.pi)
                poly = np.polyfit(ts, unw, 1)
                m_spd = float(poly[0])
                if 160.0 <= m_spd <= 420.0:
                    diff = abs(m_spd - speed_old)
                    weight = 0.55 if diff > 25.0 else (0.25 if len(history_old) >= 8 else 0.15)
                    speed_old = (1.0 - weight) * speed_old + weight * m_spd
            
            if len(history_old) == 4:
                speed_old_at_frame4 = speed_old
            if len(history_old) == 8:
                speed_old_at_frame8 = speed_old

            # NEW logic: immediate robust regression once >= 3 non-duplicate points, direct fit
            if len(history_old) >= 3:
                recent = history_old[-12:]
                ts = np.array([x[0] for x in recent], dtype=np.float64)
                angs = np.array([x[1] for x in recent], dtype=np.float64)
                unw = np.unwrap(angs * (np.pi / 180.0)) * (180.0 / np.pi)
                poly = np.polyfit(ts, unw, 1)
                m_spd = float(poly[0])
                if 15.0 <= m_spd <= 480.0:
                    # Direct assignment when fit is solid (>= 4 points), gentle for 3 points
                    w = 0.75 if len(recent) == 3 else 0.95
                    speed_new = (1.0 - w) * speed_new + w * m_spd
                    
            if len(history_old) == 4:
                speed_new_at_frame4 = speed_new
            if len(history_old) == 8:
                speed_new_at_frame8 = speed_new
                
        errors_old.append((abs(speed_old_at_frame4 - true_speed), abs(speed_old_at_frame8 - true_speed)))
        errors_new.append((abs(speed_new_at_frame4 - true_speed), abs(speed_new_at_frame8 - true_speed)))

    print(f"Total evaluated episodes: {len(errors_old)}")
    old_f4 = [x[0] for x in errors_old]
    old_f8 = [x[1] for x in errors_old]
    new_f4 = [x[0] for x in errors_new]
    new_f8 = [x[1] for x in errors_new]
    
    print(f"Frame 4 Speed Error: OLD mean={np.mean(old_f4):.1f}°/s -> NEW mean={np.mean(new_f4):.1f}°/s (improved by {((np.mean(old_f4)-np.mean(new_f4))/np.mean(old_f4)*100):.1f}%)")
    print(f"Frame 8 Speed Error: OLD mean={np.mean(old_f8):.1f}°/s -> NEW mean={np.mean(new_f8):.1f}°/s (improved by {((np.mean(old_f8)-np.mean(new_f8))/np.mean(old_f8)*100):.1f}%)")

if __name__ == "__main__":
    test_speed_convergence()
