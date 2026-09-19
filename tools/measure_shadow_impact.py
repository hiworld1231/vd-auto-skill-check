"""
Rigorous Production Main-Path Timing Audit:
Compares main-thread timings under three conditions:
A. Shadow OFF
B. Async Shadow HYBRID ON (non-blocking bounded worker)
C. Sync Shadow HYBRID ON (synchronous baseline for comparison)

Measures across 1500 realistic gameplay frames:
- Main-thread frame processing time
- Baseline vision time
- Predictor time
- Shadow dispatch / submit overhead on main thread
- Frame throughput (FPS)
- Shadow frames dropped / queue backlog
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.vision import VisionEngine
from core.predictor import SkillCheckPredictor, SPEED_MODE_BASE
from core.shadow_detector import AsyncShadowDetectorWorker
from core.detectors import get_detector


def load_test_frames(n_frames: int = 1000):
    video_paths = [
        Path("/home/oae/violence-district-data/session_0002.mkv"),
        Path("/home/oae/violence-district-data/session_0004.mkv"),
    ]
    frames = []
    base_det = get_detector("baseline")
    for vp in video_paths:
        if not vp.exists():
            continue
        cap = cv2.VideoCapture(str(vp))
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in range(tot):
            ret, bgr = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            d = base_det.detect(bgr, gray)
            if d is not None and d.get("ring_present") and d.get("needle_valid"):
                frames.append((gray, bgr))
            if len(frames) >= n_frames:
                break
        cap.release()
        if len(frames) >= n_frames:
            break
    return frames


def run_benchmark():
    frames = load_test_frames(1000)
    print(f"[AUDIT] Loaded {len(frames)} active skill check frames.")

    results = {}

    # -------------------------------------------------------------
    # Scenario A: Shadow OFF
    # -------------------------------------------------------------
    vision = VisionEngine(backend_name="baseline")
    predictor = SkillCheckPredictor(latency_ms=127.27, speed_mode=SPEED_MODE_BASE)

    total_times_a = []
    vision_times_a = []
    pred_times_a = []

    # Warmup
    for g, b in frames[:20]:
        vision.detect_frame(g, b)

    t_start_a = time.perf_counter()
    for g, b in frames:
        t0 = time.perf_counter_ns()
        det = vision.detect_frame(g, b)
        t_vis = time.perf_counter_ns()

        pred = None
        if det is not None:
            w_d, b_d = vision.extract_zones(det, None)
            predictor.update(time.monotonic(), det["needle_angle"], det["needle_strength"], w_d, b_d)
            pred = predictor.predict(time.monotonic(), det["needle_angle"], target="GREAT")
        t_pred = time.perf_counter_ns()

        vision_times_a.append((t_vis - t0) / 1e6)
        pred_times_a.append((t_pred - t_vis) / 1e6)
        total_times_a.append((t_pred - t0) / 1e6)
    t_end_a = time.perf_counter()

    results["SHADOW_OFF"] = {
        "total_ms": np.array(total_times_a),
        "vision_ms": np.array(vision_times_a),
        "pred_ms": np.array(pred_times_a),
        "submit_ms": np.zeros(len(frames)),
        "fps": len(frames) / (t_end_a - t_start_a),
        "dropped": 0,
    }

    # -------------------------------------------------------------
    # Scenario B: Async Shadow HYBRID ON (our architecture)
    # -------------------------------------------------------------
    vision.reset()
    predictor.reset()
    shadow_worker = AsyncShadowDetectorWorker(backend_name="hybrid", latency_ms=127.27)
    shadow_worker.on_check_start(time.monotonic(), target_angle=124.0)

    total_times_b = []
    vision_times_b = []
    pred_times_b = []
    submit_times_b = []

    t_start_b = time.perf_counter()
    for g, b in frames:
        t0 = time.perf_counter_ns()
        det = vision.detect_frame(g, b)
        t_vis = time.perf_counter_ns()
        vis_ms = (t_vis - t0) / 1e6

        # Submit to async shadow worker
        t_sub0 = time.perf_counter_ns()
        shadow_worker.submit_frame(
            now=time.monotonic(),
            frame_gray=g,
            frame_bgr=b,
            in_check=True,
            baseline_det=det,
            dt_frame=1.0 / 120.0,
            expected_speed=278.0,
            baseline_vision_ms=vis_ms,
        )
        t_sub1 = time.perf_counter_ns()

        pred = None
        if det is not None:
            w_d, b_d = vision.extract_zones(det, None)
            predictor.update(time.monotonic(), det["needle_angle"], det["needle_strength"], w_d, b_d)
            pred = predictor.predict(time.monotonic(), det["needle_angle"], target="GREAT")
        t_pred = time.perf_counter_ns()

        vision_times_b.append(vis_ms)
        submit_times_b.append((t_sub1 - t_sub0) / 1e6)
        pred_times_b.append((t_pred - t_sub1) / 1e6)
        total_times_b.append((t_pred - t0) / 1e6)
    t_end_b = time.perf_counter()

    summary_b = shadow_worker.conclude_check(time.monotonic(), "BENCHMARK")
    shadow_worker.close()

    results["ASYNC_SHADOW_HYBRID"] = {
        "total_ms": np.array(total_times_b),
        "vision_ms": np.array(vision_times_b),
        "pred_ms": np.array(pred_times_b),
        "submit_ms": np.array(submit_times_b),
        "fps": len(frames) / (t_end_b - t_start_b),
        "dropped": shadow_worker.frames_dropped,
    }

    # -------------------------------------------------------------
    # Scenario C: Synchronous Shadow HYBRID ON (old baseline)
    # -------------------------------------------------------------
    vision.reset()
    predictor.reset()
    sync_hybrid = get_detector("hybrid")
    sync_hybrid.reset()

    total_times_c = []
    vision_times_c = []
    pred_times_c = []
    sync_shadow_times_c = []

    t_start_c = time.perf_counter()
    for g, b in frames:
        t0 = time.perf_counter_ns()
        det = vision.detect_frame(g, b)
        t_vis = time.perf_counter_ns()

        # Synchronous execution in main thread
        t_syn0 = time.perf_counter_ns()
        sh_det = sync_hybrid.detect(b, g)
        t_syn1 = time.perf_counter_ns()

        pred = None
        if det is not None:
            w_d, b_d = vision.extract_zones(det, None)
            predictor.update(time.monotonic(), det["needle_angle"], det["needle_strength"], w_d, b_d)
            pred = predictor.predict(time.monotonic(), det["needle_angle"], target="GREAT")
        t_pred = time.perf_counter_ns()

        vision_times_c.append((t_vis - t0) / 1e6)
        sync_shadow_times_c.append((t_syn1 - t_syn0) / 1e6)
        pred_times_c.append((t_pred - t_syn1) / 1e6)
        total_times_c.append((t_pred - t0) / 1e6)
    t_end_c = time.perf_counter()

    results["SYNC_SHADOW_HYBRID"] = {
        "total_ms": np.array(total_times_c),
        "vision_ms": np.array(vision_times_c),
        "pred_ms": np.array(pred_times_c),
        "submit_ms": np.array(sync_shadow_times_c),
        "fps": len(frames) / (t_end_c - t_start_c),
        "dropped": 0,
    }

    # Print results
    print("\n" + "=" * 90)
    print("           PRODUCTION MAIN-PATH CRITICAL TIMING AUDIT (1000 FRAMES)")
    print("=" * 90)
    print(f"{'Configuration':<22} | {'Main Med':<9} | {'Main p95':<9} | {'Vision Med':<10} | {'Predict Med':<11} | {'Shadow Cost':<11} | {'Throughput'}")
    print("-" * 90)
    for name, d in results.items():
        tot_med = np.median(d["total_ms"])
        tot_p95 = np.percentile(d["total_ms"], 95)
        vis_med = np.median(d["vision_ms"])
        prd_med = np.median(d["pred_ms"])
        sub_med = np.median(d["submit_ms"])
        fps = d["fps"]
        print(f"{name:<22} | {tot_med:<9.4f} | {tot_p95:<9.4f} | {vis_med:<10.4f} | {prd_med:<11.4f} | {sub_med:<11.4f} | {fps:.1f} FPS")
    print("=" * 90)


if __name__ == "__main__":
    run_benchmark()
