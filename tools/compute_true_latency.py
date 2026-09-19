#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPLAYS_DIR = ROOT / "replays"

def compute_true_latency():
    json_files = sorted(REPLAYS_DIR.glob("check_*_*.json"))
    
    true_latencies_ms = []
    
    for f in json_files:
        try:
            ep = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
            
        ev = ep.get("evaluation") or {}
        trig = ep.get("trigger") or {}
        
        if not trig.get("fired") or not ev.get("plateau_found"):
            continue
            
        hit_ang = ev.get("hit_angle")
        est_at_trig = trig.get("est_angle_at_trigger")
        speed = trig.get("speed_deg_s")
        configured_lat = ep.get("configured_latency_ms")
        outcome = ev.get("outcome")
        chain = ep.get("chain_count", 1)
        
        # Only take valid hits with good tracking
        if hit_ang is not None and est_at_trig is not None and speed and speed > 100:
            # Degrees moved between trigger and freeze:
            deg_moved = (hit_ang - est_at_trig + 180.0) % 360.0 - 180.0
            if deg_moved > 0:
                actual_lag_ms = (deg_moved / speed) * 1000.0
                if 20.0 <= actual_lag_ms <= 200.0:
                    true_latencies_ms.append({
                        "check_id": ep["check_id"],
                        "outcome": outcome,
                        "chain": chain,
                        "speed": speed,
                        "configured_lat": configured_lat,
                        "deg_moved": deg_moved,
                        "actual_lag_ms": actual_lag_ms,
                        "error_deg": ev.get("error_deg")
                    })

    print(f"Total evaluated hits with freeze: {len(true_latencies_ms)}")
    all_lags = [x["actual_lag_ms"] for x in true_latencies_ms]
    print(f"ACTUAL PHYSICAL LATENCY (time from Space keypress to Roblox needle freeze):")
    print(f"  Mean: {np.mean(all_lags):.1f} ms")
    print(f"  Median: {np.median(all_lags):.1f} ms")
    print(f"  Std: {np.std(all_lags):.1f} ms")
    print(f"  Min: {np.min(all_lags):.1f} ms")
    print(f"  Max: {np.max(all_lags):.1f} ms")
    print(f"  25th percentile: {np.percentile(all_lags, 25):.1f} ms")
    print(f"  75th percentile: {np.percentile(all_lags, 75):.1f} ms")

    print("\nBreakdown by Chain Count:")
    by_chain = {}
    for x in true_latencies_ms:
        c = x["chain"]
        by_chain.setdefault(c, []).append(x["actual_lag_ms"])
    for c, lags in sorted(by_chain.items()):
        print(f"  Chain #{c}: mean={np.mean(lags):.1f}ms, median={np.median(lags):.1f}ms (n={len(lags)})")

    print("\nRecent 15 checks:")
    for x in true_latencies_ms[-15:]:
        print(f"  {x['check_id']} (ch#{x['chain']}, {x['outcome']}): cfg_lat={x['configured_lat']:.1f}ms, actual_lag={x['actual_lag_ms']:.1f}ms, err={x['error_deg']:+.1f}°")

if __name__ == "__main__":
    compute_true_latency()
