#!/usr/bin/env python3
import subprocess
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = ROOT / "recordings"

def inspect_mkv_events():
    files = sorted(RECORDINGS_DIR.glob("session_*.mkv"))
    print(f"Found {len(files)} session MKV files")
    
    for vf in files[-5:]:
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-dump_attachment:t:0", "pipe:1", "-i", str(vf), "-y"],
            capture_output=True,
            text=True,
            timeout=5
        )
        events = []
        if p.returncode == 0 and p.stdout:
            for line in p.stdout.splitlines():
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    events.append((float(parts[0]), parts[1]))
        
        spaces = [e for e in events if "SPACE" in e[1]]
        print(f"\n{vf.name}: total events={len(events)}, SPACE events={len(spaces)}")
        for sp in spaces[:5]:
            print(f"  t={sp[0]:.4f}s: {sp[1]}")

if __name__ == "__main__":
    inspect_mkv_events()
