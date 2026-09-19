"""
Benchmark synchronous overhead of FlightRecorder.on_frame on the production thread.
Measures: median, p95, p99, max latency in milliseconds across 5000 calls.
"""

import gc
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.flight_recorder import FlightRecorder

def benchmark_recorder(n_iter: int = 5000):
    recorder = FlightRecorder(save_video=False, save_diagnostic_strip=False, record_all=False)
    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    dummy_det = {"needle_angle": 120.0, "needle_strength": 50.0, "ring_present": True}
    dummy_pred = {"target_angle": 124.0, "time_until_press_ms": 100.0}

    # Warmup
    for _ in range(100):
        recorder.on_frame(time.monotonic(), dummy_frame, dummy_det, dummy_pred)

    # 1. Measure on_frame when idle (maintaining pre-roll circular buffer)
    idle_timings = []
    gc.disable()
    try:
        for _ in range(n_iter):
            t0 = time.perf_counter_ns()
            recorder.on_frame(time.monotonic(), dummy_frame, dummy_det, dummy_pred)
            t1 = time.perf_counter_ns()
            idle_timings.append((t1 - t0) / 1e6)
    finally:
        gc.enable()

    # 2. Measure on_frame during active skill check
    recorder.start_check(now=time.monotonic(), chain_count=1)
    active_timings = []
    gc.disable()
    try:
        for _ in range(n_iter):
            t0 = time.perf_counter_ns()
            recorder.on_frame(time.monotonic(), dummy_frame, dummy_det, dummy_pred)
            t1 = time.perf_counter_ns()
            active_timings.append((t1 - t0) / 1e6)
    finally:
        gc.enable()

    recorder.close()

    idle_arr = np.array(idle_timings)
    active_arr = np.array(active_timings)

    print("=" * 75)
    print(f"       FLIGHT RECORDER SYNCHRONOUS OVERHEAD AUDIT ({n_iter} iterations)")
    print("=" * 75)
    print(f"{'State':<18} | {'Median ms':<10} | {'p95 ms':<8} | {'p99 ms':<8} | {'Max ms':<8}")
    print("-" * 75)
    print(f"{'Idle (Pre-roll)':<18} | {np.median(idle_arr):<10.4f} | {np.percentile(idle_arr, 95):<8.4f} | {np.percentile(idle_arr, 99):<8.4f} | {np.max(idle_arr):<8.4f}")
    print(f"{'Active Episode':<18} | {np.median(active_arr):<10.4f} | {np.percentile(active_arr, 95):<8.4f} | {np.percentile(active_arr, 99):<8.4f} | {np.max(active_arr):<8.4f}")
    print("=" * 75)

if __name__ == "__main__":
    benchmark_recorder()
