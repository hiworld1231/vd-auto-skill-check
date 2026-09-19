#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPLAYS_DIR = ROOT / "replays"

def analyze_all_replays_az():
    json_files = sorted(REPLAYS_DIR.glob("check_*_*.json"))
    print(f"=== FULL A-to-Z REPLAY REPORT: {len(json_files)} EPISODES ===")
    
    rows = []
    for f in json_files:
        try:
            ep = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            continue
            
        cid = ep["check_id"]
        chain = ep.get("chain_count", 1)
        dur = ep.get("duration_ms", 0)
        cfg_lat = ep.get("configured_latency_ms", 0)
        w = (ep.get("locked_zones") or {}).get("white") or {}
        w_start = w.get("start")
        w_end = w.get("end")
        w_center = w.get("center")
        w_width = w.get("width")
        
        trig = ep.get("trigger") or {}
        fired = trig.get("fired", False)
        reason = trig.get("reason", "NONE")
        trig_tgt = trig.get("target_angle")
        speed = trig.get("speed_deg_s")
        est_trig = trig.get("est_angle_at_trigger")
        
        ev = ep.get("evaluation") or {}
        outcome = ev.get("outcome", "UNKNOWN")
        hit_ang = ev.get("hit_angle")
        err_deg = ev.get("error_deg")
        err_ms = ev.get("error_ms")
        plateau = ev.get("plateau_found", False)
        
        # Position label:
        # 0° = Right, 90° = Bottom, 180° = Left, 270° = Top
        pos = "UNKNOWN"
        if w_center is not None:
            if 45 <= w_center < 135: pos = "BOTTOM"
            elif 135 <= w_center < 225: pos = "LEFT"
            elif 225 <= w_center < 315: pos = "TOP"
            else: pos = "RIGHT"
            
        rows.append({
            "cid": cid,
            "chain": chain,
            "outcome": outcome,
            "pos": pos,
            "w_center": w_center,
            "w_width": w_width,
            "speed": speed,
            "cfg_lat": cfg_lat,
            "target": trig_tgt,
            "hit": hit_ang,
            "err_deg": err_deg,
            "err_ms": err_ms,
            "reason": reason,
            "plateau": plateau,
            "dur": dur,
        })

    print(f"{'CHECK ID':<28} {'CH':<3} {'OUTCOME':<12} {'POS':<7} {'W_CTR':<6} {'SPEED':<6} {'LAT_MS':<7} {'TGT':<6} {'HIT':<6} {'ERR_DEG':<8} {'REASON':<12}")
    print("-" * 115)
    for r in rows:
        w_c = f"{r['w_center']:.1f}°" if r['w_center'] is not None else "N/A"
        spd = f"{r['speed']:.1f}" if r['speed'] is not None else "N/A"
        tgt = f"{r['target']:.1f}°" if r['target'] is not None else "N/A"
        hit = f"{r['hit']:.1f}°" if r['hit'] is not None else "N/A"
        err = f"{r['err_deg']:+.1f}°" if r['err_deg'] is not None else "N/A"
        reason_str = str(r['reason']) if r['reason'] is not None else "NONE"
        lat_str = f"{r['cfg_lat']:.1f}" if r['cfg_lat'] is not None else "N/A"
        print(f"{r['cid']:<28} #{r['chain']:<2} {r['outcome']:<12} {r['pos']:<7} {w_c:<6} {spd:<6} {lat_str:<7} {tgt:<6} {hit:<6} {err:<8} {reason_str:<12}")

    print("\n" + "=" * 80)
    print("ANALYSIS & SUMMARY")
    print("=" * 80)
    
    # 1. Group by Outcome
    outcomes = {}
    for r in rows:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    print("Outcome counts:", outcomes)
    
    # 2. Group by Position
    by_pos = {}
    for r in rows:
        p = r["pos"]
        if p not in by_pos: by_pos[p] = []
        by_pos[p].append(r)
    print("\nBreakdown by Position on Circle:")
    for p, prs in by_pos.items():
        p_outcomes = {}
        errs = [x["err_deg"] for x in prs if x["err_deg"] is not None and x["outcome"] in ("GREAT", "GOOD", "MISS")]
        spds = [x["speed"] for x in prs if x["speed"] is not None]
        for x in prs:
            p_outcomes[x["outcome"]] = p_outcomes.get(x["outcome"], 0) + 1
        mean_e = f"{np.mean(errs):+.2f}°" if errs else "N/A"
        mean_s = f"{np.mean(spds):.1f}°/s" if spds else "N/A"
        print(f"  {p:<8} (n={len(prs):<2}): mean_speed={mean_s:<8} mean_err={mean_e:<8} outcomes={p_outcomes}")

    # 3. Group by Chain
    by_ch = {}
    for r in rows:
        c = r["chain"]
        if c not in by_ch: by_ch[c] = []
        by_ch[c].append(r)
    print("\nBreakdown by Chain Count:")
    for c, crs in sorted(by_ch.items()):
        c_outcomes = {}
        spds = [x["speed"] for x in crs if x["speed"] is not None]
        for x in crs:
            c_outcomes[x["outcome"]] = c_outcomes.get(x["outcome"], 0) + 1
        mean_s = f"{np.mean(spds):.1f}°/s" if spds else "N/A"
        print(f"  Chain #{c:<2} (n={len(crs):<2}): mean_speed={mean_s:<8} outcomes={c_outcomes}")

if __name__ == "__main__":
    analyze_all_replays_az()
