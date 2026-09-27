import json
from pathlib import Path
import re
import struct
import subprocess


PROJECT = Path(__file__).resolve().parents[1]


def test_capture_converts_only_the_selected_roi_before_emitting_bgr_frames():
    result = subprocess.run(
        ["/usr/bin/python", str(PROJECT / "native" / "pipewire_worker.py"),
         "--synthetic", "--frames", "2", "--roi", "800,420,320,240"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15, check=False,
    )

    assert result.returncode == 0, result.stderr.decode(errors="replace")
    match = re.search(rb"Capture negotiated: ([^\r\n]+)", result.stderr)
    assert match, result.stderr.decode(errors="replace")
    assert "width=(int)320" in match.group(1).decode()
    assert "height=(int)240" in match.group(1).decode()

    header_size, = struct.unpack("!I", result.stdout[:4])
    header = json.loads(result.stdout[4:4 + header_size])
    assert (header["width"], header["height"], header["bytes"]) == (320, 240, 320 * 240 * 3)
    assert header["worker_cpu_time_ns"] > 0
