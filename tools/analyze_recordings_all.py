#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from pathlib import Path
import cv2
import numpy as np
from core.vision import VisionEngine, is_angle_in_arc
from tools.analyze_skillcheck import parse_events, associate_labels

ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = ROOT / "recordings"

def analyze_all_recordings():
    files = sorted(RECORDINGS_DIR.glob("session_*.mkv"))
    print(f"=== ANALYZING ALL {len(files)} RECORDING SESSIONS ===")
    
    vision = VisionEngine()
    
    total_checks = 0
    checks_by_label = {"WHITE": 0, "BLACK": 0}
    
    all_speeds = []
    speeds_by_zone_angle = []
    
    for vf in files:
        if vf.stat().st_size < 10000:
            continue
        events = parse_events(str(vf))
        labeled_checks, _ = associate_labels(events)
        if not labeled_checks:
            continue
            
        cap = cv2.VideoCapture(str(vf))
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n{vf.name} (fps={fps:.1f}, frames={total_frames}, labeled_checks={len(labeled_checks)}):")
        
        for c in labeled_checks:
            sp_t = c["space_time"]
            lbl = c.get("label") or "UNKNOWN"
            checks_by_label[lbl] = checks_by_label.get(lbl, 0) + 1
            total_checks += 1
            
            # Read 20 frames before space_time to 10 frames after
            start_f = max(0, int((sp_t - 0.5) * fps))
            end_f = min(total_frames, int((sp_t + 0.3) * fps))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            
            check_trajectory = []
            locked_w = None
            locked_b = None
            
            for f_idx in range(start_f, end_f):
                ret, frame = cap.read()
                if not ret:
                    break
                t_frame = f_idx / fps
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                det = vision.detect_frame(gray, frame)
                if det:
                    if locked_w is None:
                        w_d, b_d = vision.extract_zones(det)
                        if w_d:
                            locked_w, locked_b = w_d, b_d
                    check_trajectory.append((t_frame, det["needle_angle"], det["needle_strength"]))
            
            # Measure speed before Space
            pre_space = [p for p in check_trajectory if p[0] <= sp_t and p[2] >= 15.0]
            if len(pre_space) >= 4:
                ts = [p[0] for p in pre_space]
                angs = [p[1] for p in pre_space]
                unwrapped = np.unwrap(np.array(angs) * np.pi / 180.0) * 180.0 / np.pi
                poly = np.polyfit(ts, unwrapped, 1)
                spd = poly[0]
                all_speeds.append(spd)
                w_ctr = locked_w["center"] if locked_w else None
                if w_ctr is not None:
                    speeds_by_zone_angle.append((w_ctr, spd))
                print(f"  Check at t={sp_t:.3f}s: label={lbl}, speed={spd:.1f}°/s, white_zone={locked_w['center'] if locked_w else 'N/A'}")
            else:
                print(f"  Check at t={sp_t:.3f}s: label={lbl}, tracking frames={len(pre_space)}")
                
        cap.release()

    print("\n" + "=" * 80)
    print(f"SUMMARY OF RECORDINGS: total checks={total_checks}, labels={checks_by_label}")
    if all_speeds:
        print(f"Measured speeds from recordings: mean={np.mean(all_speeds):.1f}°/s, std={np.std(all_speeds):.1f}°/s, min={np.min(all_speeds):.1f}°/s, max={np.max(all_speeds):.1f}°/s")

if __name__ == "__main__":
    analyze_all_recordings()
