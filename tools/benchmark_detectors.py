"""
Exhaustive Automated Benchmark Suite for Skill Check Detector Backends.
Evaluates all detector backends on identical raw gameplay video frames:
1. CONTOUR_BASELINE: Original matchTemplate + warpPolar.
2. FIXED_POLAR: Precomputed polar coordinate lookup table.
3. PRECOMPUTED_RAYS: Discrete radial ray scoring.
4. LOCAL_TRACKER: Dynamic local angular window tracking.
5. HYBRID: Primary candidate combining fixed geometry, local search, and lifecycle zone caching.
6. MINIMAL_COLOR: Direct integer BGR indexing without cvtColor.

Measures:
- Latency (median, p95, p99, max in milliseconds)
- Accuracy vs baseline (median, p95, max angular error)
- Needle stability (frame-to-frame angular jitter, teleports, impossible backward jumps)
- Zone stability (white zone center error, width error, zone losses)
- Reliability (lost frames, reacquires)
"""

import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from core.detectors import DETECTOR_REGISTRY, get_detector
from core.detectors.base import is_angle_in_arc


def load_dataset(max_episodes: int = 25) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Loads raw video frames grouped by continuous skill check episodes from recorded session MKVs.
    Returns list of episodes, where each episode is a list of (frame_gray, frame_bgr).
    """
    video_paths = [
        Path("/home/oae/violence-district-data/session_0002.mkv"),
        Path("/home/oae/violence-district-data/session_0004.mkv"),
    ]
    episodes: List[List[Tuple[np.ndarray, np.ndarray]]] = []
    baseline_detector = get_detector("baseline")

    for vp in video_paths:
        if not vp.exists():
            continue
        cap = cv2.VideoCapture(str(vp))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_ep: List[Tuple[np.ndarray, np.ndarray]] = []

        for f_idx in range(total):
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            det = baseline_detector.detect(frame_bgr, frame_gray)
            if det is not None and det.get("ring_present") and det.get("needle_valid"):
                current_ep.append((frame_gray, frame_bgr))
            else:
                if len(current_ep) >= 8:  # Valid active check with at least 8 frames
                    episodes.append(current_ep)
                    if len(episodes) >= max_episodes:
                        break
                current_ep = []

        if current_ep and len(current_ep) >= 8:
            episodes.append(current_ep)
        cap.release()
        if len(episodes) >= max_episodes:
            break

    return episodes


def run_detector_benchmark(episodes: List[List[Tuple[np.ndarray, np.ndarray]]]) -> Dict[str, Any]:
    backends = [
        ("CONTOUR_BASELINE", "baseline"),
        ("DETECTOR_A_POLAR", "polar"),
        ("DETECTOR_B_RAYS", "rays"),
        ("DETECTOR_C_LOCAL", "local"),
        ("DETECTOR_D_HYBRID", "hybrid"),
        ("DETECTOR_E_COLOR", "color_index"),
    ]

    # Pre-evaluate baseline ground truth per episode
    baseline_detector = get_detector("baseline")
    baseline_episodes = []
    for ep in episodes:
        baseline_detector.reset()
        base_ep = []
        for g, b in ep:
            base_ep.append(baseline_detector.detect(b, g))
        baseline_episodes.append(base_ep)

    results = {}

    for label, b_key in backends:
        det_inst = get_detector(b_key)
        det_inst.reset()

        timings_ms = []
        angle_diffs = []
        jitters = []
        teleports = 0
        backward_jumps = 0
        lost_frames = 0
        reacquires = 0
        zone_center_errors = []
        zone_losses = 0

        for ep_idx, ep in enumerate(episodes):
            base_ep = baseline_episodes[ep_idx]
            det_inst.reset()
            prev_angle = None
            prev_delta = None

            for frame_idx, (g, b) in enumerate(ep):
                base_det = base_ep[frame_idx]
                exp_ang = prev_angle if prev_angle is not None else (base_det["needle_angle"] if base_det else None)

                t0 = time.perf_counter_ns()
                det = det_inst.detect(b, g, expected_angle=exp_ang)
                t1 = time.perf_counter_ns()
                timings_ms.append((t1 - t0) / 1e6)

                if base_det is not None and base_det.get("ring_present"):
                    if det is None or not det.get("ring_present"):
                        lost_frames += 1
                    else:
                        if det.get("status") == "REACQUIRE":
                            reacquires += 1

                        # Compare needle angle
                        base_ang = base_det["needle_angle"]
                        cur_ang = det["needle_angle"]
                        if base_ang is not None and cur_ang is not None:
                            diff = abs((cur_ang - base_ang + 180.0) % 360.0 - 180.0)
                            angle_diffs.append(diff)

                            # Check temporal stability
                            if prev_angle is not None:
                                step = (cur_ang - prev_angle + 180.0) % 360.0 - 180.0
                                if step < -3.0:
                                    backward_jumps += 1
                                elif step > 45.0:
                                    teleports += 1

                                if prev_delta is not None:
                                    jitter = abs(step - prev_delta)
                                    jitters.append(jitter)
                                prev_delta = step
                            prev_angle = cur_ang

                    # Compare white zone
                    base_w = base_det.get("white_zone")
                    cur_w = det.get("white_zone")
                    if base_w is not None:
                        if cur_w is not None:
                            w_err = abs((cur_w["center"] - base_w["center"] + 180.0) % 360.0 - 180.0)
                            zone_center_errors.append(w_err)
                        else:
                            zone_losses += 1
            else:
                prev_angle = None
                prev_delta = None

        t_arr = np.array(timings_ms, dtype=float)
        a_arr = np.array(angle_diffs, dtype=float) if angle_diffs else np.array([0.0])
        z_arr = np.array(zone_center_errors, dtype=float) if zone_center_errors else np.array([0.0])
        j_arr = np.array(jitters, dtype=float) if jitters else np.array([0.0])

        med_ms = float(np.median(t_arr))
        p95_ms = float(np.percentile(t_arr, 95))
        p99_ms = float(np.percentile(t_arr, 99))
        max_ms = float(np.max(t_arr))

        med_angle = float(np.median(a_arr))
        p95_angle = float(np.percentile(a_arr, 95))
        p99_angle = float(np.percentile(a_arr, 99))
        mean_angle = float(np.mean(a_arr))
        max_angle = float(np.max(a_arr))

        # Mathematical sanity assertions
        assert p95_angle >= med_angle - 1e-9, f"p95 < median in {label}: {p95_angle} < {med_angle}"
        assert p99_angle >= p95_angle - 1e-9, f"p99 < p95 in {label}: {p99_angle} < {p95_angle}"
        assert p95_ms >= med_ms - 1e-9, f"p95_ms < med_ms in {label}: {p95_ms} < {med_ms}"
        assert p99_ms >= p95_ms - 1e-9, f"p99_ms < p95_ms in {label}: {p99_ms} < {p95_ms}"

        results[label] = {
            "median_ms": med_ms,
            "p95_ms": p95_ms,
            "p99_ms": p99_ms,
            "max_ms": max_ms,
            "median_angle_err": med_angle,
            "p95_angle_err": p95_angle,
            "p99_angle_err": p99_angle,
            "mean_angle_err": mean_angle,
            "max_angle_err": max_angle,
            "mean_jitter": float(np.mean(j_arr)),
            "p95_jitter": float(np.percentile(j_arr, 95)),
            "lost_frames": lost_frames,
            "reacquires": reacquires,
            "teleports": teleports,
            "backward_jumps": backward_jumps,
            "zone_center_err_mean": float(np.mean(z_arr)),
            "zone_center_err_p95": float(np.percentile(z_arr, 95)),
            "zone_losses": zone_losses,
        }

    return results


def print_comparison_table(results: Dict[str, Any]):
    print("\n" + "=" * 125)
    print("                          SKILL CHECK DETECTOR ARCHITECTURE BENCHMARK")
    print("=" * 125)
    header = (
        f"{'Backend':<19} | {'Med ms':<7} | {'p95 ms':<7} | {'Med Err':<8} | {'p95 Err':<8} | {'Mean Err':<8} | "
        f"{'Jitter':<7} | {'Loss':<4} | {'Reacq':<5} | {'Zone Err':<8} | {'Complexity'}"
    )
    print(header)
    print("-" * 125)

    complexity_map = {
        "CONTOUR_BASELINE": "Heavy (warpPolar + morphology + template)",
        "DETECTOR_A_POLAR": "Medium (precomputed 360-ray polar LUT)",
        "DETECTOR_B_RAYS": "Low (precomputed 8-pt discrete rays)",
        "DETECTOR_C_LOCAL": "Medium (adaptive local window)",
        "DETECTOR_D_HYBRID": "Minimal (local rays + locked zone cache)",
        "DETECTOR_E_COLOR": "Low (direct BGR channels, no cvtColor)",
    }

    for label, m in results.items():
        med_err_str = f"{m['median_angle_err']:.2f}°"
        p95_err_str = f"{m['p95_angle_err']:.2f}°"
        mean_err_str = f"{m['mean_angle_err']:.2f}°"
        jit_str = f"{m['mean_jitter']:.2f}°"
        zne_str = f"{m['zone_center_err_mean']:.2f}°"
        cplx = complexity_map.get(label, "Standard")
        row = (
            f"{label:<19} | {m['median_ms']:<7.3f} | {m['p95_ms']:<7.3f} | {med_err_str:<8} | {p95_err_str:<8} | {mean_err_str:<8} | "
            f"{jit_str:<7} | {m['lost_frames']:<4} | {m['reacquires']:<5} | {zne_str:<8} | {cplx}"
        )
        print(row)
    print("=" * 125)


def main():
    print("[BENCHMARK] Loading raw gameplay video episodes from session MKVs...")
    episodes = load_dataset(max_episodes=25)
    total_frames = sum(len(ep) for ep in episodes)
    print(f"[BENCHMARK] Loaded {len(episodes)} continuous skill check episodes ({total_frames} frames).")

    print("[BENCHMARK] Running comparative evaluations on identical frames...")
    results = run_detector_benchmark(episodes)
    print_comparison_table(results)


if __name__ == "__main__":
    main()
