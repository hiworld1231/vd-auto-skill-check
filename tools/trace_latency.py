#!/usr/bin/env python3
from pathlib import Path
import re

LOG_PATH = Path(__file__).resolve().parent.parent / "bot.log"

def trace_latency():
    lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    print(f"Total log lines: {len(lines)}")
    
    learner_changes = []
    for line in lines:
        if "lat=" in line:
            m = re.search(r"lat=([0-9.]+)->([0-9.]+)ms", line)
            if m:
                p_lat, n_lat = float(m.group(1)), float(m.group(2))
                learner_changes.append((line[:19], p_lat, n_lat, line))
    
    print(f"Total latency adjustments in log: {len(learner_changes)}")
    for dt, p, n, l in learner_changes[-40:]:
        print(f"{dt}: {p:.1f} -> {n:.1f} | {l}")

if __name__ == "__main__":
    trace_latency()
