#!/usr/bin/env python3
"""
A/B Capture Benchmark for Violent District Screen Grabber.
Evaluates:
1. GSR CFR vs VFR (-fm cfr vs -fm vfr)
2. Keyframe interval: keyint=0.05s vs default keyint=2.0s
3. Pipe sizes: 64KB vs 256KB
4. Decoded FPS vs UNIQUE content FPS (distinguishing CFR duplicates)
5. Frame arrival intervals p50/p95/p99, bursts, and discarded stale frames.
"""

import argparse
import array
import fcntl
import os
import shutil
import subprocess
import sys
import termios
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VENV_PY = ROOT.parent / ".venv" / "bin" / "python"
if VENV_PY.exists() and sys.prefix != str(VENV_PY.parent.parent):
    os.execv(str(VENV_PY), [str(VENV_PY)] + sys.argv)

from core.capture import CAPTURE_REGION, F_SETPIPE_SZ, F_GETPIPE_SZ


def run_benchmark_config(
    name: str,
    framerate_mode: str = "cfr",
    keyint: str = "2.0",
    fps: int = 120,
    pipe_size_kb: int = 64,
    duration_s: float = 3.0,
    ffmpeg_r: bool = True,
) -> Dict[str, Any]:
    if not shutil.which("gpu-screen-recorder"):
        return {"error": "gpu-screen-recorder not found"}

    w, h = CAPTURE_REGION["width"], CAPTURE_REGION["height"]
    x, y = CAPTURE_REGION["left"], CAPTURE_REGION["top"]
    reg_str = f"{w}x{h}+{x}+{y}"
    frame_bytes = w * h * 3

    gsr_cmd = [
        "gpu-screen-recorder",
        "-w", reg_str,
        "-f", str(fps),
        "-c", "h264",
        "-fm", framerate_mode,
        "-tune", "performance",
        "-keyint", keyint,
        "-o", "/dev/stdout",
    ]

    gsr_p = subprocess.Popen(
        gsr_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    in_fd = gsr_p.stdout.fileno()
    try:
        fcntl.fcntl(in_fd, F_SETPIPE_SZ, pipe_size_kb * 1024)
        actual_in_pipe = fcntl.fcntl(in_fd, F_GETPIPE_SZ)
    except Exception:
        actual_in_pipe = 65536

    ff_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-fflags", "nobuffer+discardcorrupt",
        "-flags", "low_delay",
        "-avioflags", "direct",
        "-threads", "1",
        "-f", "h264",
    ]
    if ffmpeg_r:
        ff_cmd.extend(["-r", str(fps)])
    ff_cmd.extend([
        "-i", "pipe:0",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-fps_mode", "passthrough",
        "-",
    ])

    ff_p = subprocess.Popen(
        ff_cmd,
        stdin=gsr_p.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=262144,
        start_new_session=True,
    )
    gsr_p.stdout.close()

    out_fd = ff_p.stdout.fileno()
    os.set_blocking(out_fd, False)

    try:
        fcntl.fcntl(out_fd, F_SETPIPE_SZ, 262144)
        actual_out_pipe = fcntl.fcntl(out_fd, F_GETPIPE_SZ)
    except Exception:
        actual_out_pipe = 65536

    # Warm-up phase (0.8s)
    t_start = time.monotonic()
    buf = bytearray()
    ionread = array.array("i", [0])

    while time.monotonic() - t_start < 0.8:
        fcntl.ioctl(out_fd, termios.FIONREAD, ionread)
        ready = ionread[0]
        if ready > 0:
            chunk = os.read(out_fd, ready)
            buf.extend(chunk)
            if len(buf) >= frame_bytes:
                buf = buf[len(buf) % frame_bytes:]
        time.sleep(0.002)

    # Measurement phase
    decoded_count = 0
    unique_count = 0
    burst_count = 0
    discarded_frames = 0
    intervals_ms: List[float] = []
    unique_intervals_ms: List[float] = []

    last_arrival_t = None
    last_unique_t = None
    prev_frame = None

    t_measure_start = time.monotonic()
    while time.monotonic() - t_measure_start < duration_s:
        fcntl.ioctl(out_fd, termios.FIONREAD, ionread)
        ready = ionread[0]
        if ready == 0:
            time.sleep(0.0005)
            continue

        chunk = os.read(out_fd, ready)
        if not chunk:
            break
        buf.extend(chunk)

        if len(buf) >= frame_bytes:
            now = time.monotonic()
            num_frames = len(buf) // frame_bytes
            if num_frames > 1:
                burst_count += 1
                discarded_frames += (num_frames - 1)

            # Process latest frame
            start_off = (num_frames - 1) * frame_bytes
            frame_raw = buf[start_off:start_off + frame_bytes]
            buf = buf[num_frames * frame_bytes:]
            decoded_count += num_frames

            arr = np.frombuffer(frame_raw, dtype=np.uint8).reshape((h, w, 3))

            if last_arrival_t is not None:
                intervals_ms.append((now - last_arrival_t) * 1000.0)
            last_arrival_t = now

            # Detect content changes vs duplicates
            is_unique = True
            if prev_frame is not None:
                diff = float(np.mean(np.abs(arr.astype(np.int16) - prev_frame.astype(np.int16))))
                if diff < 0.40:
                    is_unique = False

            if is_unique:
                unique_count += 1
                if last_unique_t is not None:
                    unique_intervals_ms.append((now - last_unique_t) * 1000.0)
                last_unique_t = now
                prev_frame = arr

    elapsed = time.monotonic() - t_measure_start

    # Cleanup
    try:
        ff_p.kill()
        ff_p.wait(timeout=0.2)
    except Exception:
        pass
    try:
        gsr_p.kill()
        gsr_p.wait(timeout=0.2)
    except Exception:
        pass

    arr_int = np.array(intervals_ms) if intervals_ms else np.array([0.0])
    arr_u_int = np.array(unique_intervals_ms) if unique_intervals_ms else np.array([0.0])

    duplicate_pct = ((decoded_count - unique_count) / decoded_count * 100.0) if decoded_count > 0 else 0.0

    return {
        "name": name,
        "framerate_mode": framerate_mode,
        "keyint": keyint,
        "pipe_in_kb": actual_in_pipe // 1024,
        "pipe_out_kb": actual_out_pipe // 1024,
        "elapsed_s": round(elapsed, 2),
        "decoded_fps": round(decoded_count / elapsed, 1) if elapsed > 0 else 0.0,
        "unique_fps": round(unique_count / elapsed, 1) if elapsed > 0 else 0.0,
        "duplicate_pct": round(duplicate_pct, 1),
        "burst_count": burst_count,
        "discarded_stale": discarded_frames,
        "arrival_p50_ms": round(float(np.percentile(arr_int, 50)), 2),
        "arrival_p95_ms": round(float(np.percentile(arr_int, 95)), 2),
        "arrival_p99_ms": round(float(np.percentile(arr_int, 99)), 2),
        "unique_p50_ms": round(float(np.percentile(arr_u_int, 50)), 2) if len(unique_intervals_ms) > 0 else 0.0,
    }


def main():
    print("=========================================================================")
    print("   A/B BENCHMARK: SCREEN CAPTURE PIPELINE OPTIMIZATION & TELEMETRY       ")
    print("=========================================================================")
    configs = [
        {"name": "1. GSR CFR + keyint 0.05s (Baseline)", "framerate_mode": "cfr", "keyint": "0.05", "pipe_size_kb": 64},
        {"name": "2. GSR CFR + keyint 2.00s (Default)",  "framerate_mode": "cfr", "keyint": "2.0",  "pipe_size_kb": 64},
        {"name": "3. GSR VFR + keyint 2.00s (Variable)", "framerate_mode": "vfr", "keyint": "2.0",  "pipe_size_kb": 64},
        {"name": "4. GSR CFR + keyint 2.0s (128KB Pipe)","framerate_mode": "cfr", "keyint": "2.0",  "pipe_size_kb": 128},
    ]

    results = []
    for cfg in configs:
        print(f"Тестирование конфигурации: {cfg['name']} (3 секунды)...")
        res = run_benchmark_config(**cfg)
        results.append(res)
        time.sleep(0.5)

    print("\n---------------------------------------------------------------------------------------------------------")
    print(f"{'Конфигурация':<35} | {'Dec FPS':<7} | {'Uniq FPS':<8} | {'Dup %':<6} | {'Burst':<5} | {'p50 ms':<6} | {'p95 ms':<6}")
    print("---------------------------------------------------------------------------------------------------------")
    for r in results:
        if "error" in r:
            print(f"{r.get('name', 'N/A'):<35} | ERROR: {r['error']}")
        else:
            print(
                f"{r['name']:<35} | {r['decoded_fps']:<7.1f} | {r['unique_fps']:<8.1f} | "
                f"{r['duplicate_pct']:<5.1f}% | {r['burst_count']:<5d} | {r['arrival_p50_ms']:<6.2f} | {r['arrival_p95_ms']:<6.2f}"
            )
    print("---------------------------------------------------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
