#!/usr/bin/env python3
"""Scan MKV screen-session captures without treating them as outcome labels.

These are screen recordings, so any live overlay visible at capture time may
also be present. They are not postprocessed replay MP4s, but not guaranteed to
be game-only pixels.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import statistics

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vd.vision import Detector


def _circular_distance(left, right):
    return abs((right - left + 180) % 360 - 180)


def _same_target(left, right, *, angle_tolerance_deg, width_tolerance_deg,
                 center_tolerance_px):
    for key in ("great_start_median_deg", "good_start_median_deg"):
        if left.get(key) is None or right.get(key) is None:
            return False
        if _circular_distance(left[key], right[key]) > angle_tolerance_deg:
            return False
    for key in ("great_width_median_deg", "good_width_median_deg"):
        if left.get(key) is None or right.get(key) is None:
            return False
        if abs(left[key] - right[key]) > width_tolerance_deg:
            return False
    left_center, right_center = left.get("center_median_px"), right.get("center_median_px")
    if left_center is None or right_center is None:
        return False
    return math.dist(left_center, right_center) <= center_tolerance_px


def merge_target_spans(spans, *, max_gap_s=0.15, angle_tolerance_deg=2.0,
                       width_tolerance_deg=2.0, center_tolerance_px=3.0):
    """Join short detector dropouts only when both measured rings stay the same."""
    merged = []
    for source in spans:
        span = dict(source)
        span["merged_segments"] = span.get("merged_segments", 1)
        if not merged:
            merged.append(span)
            continue
        previous = merged[-1]
        gap = span["start_s"] - previous["end_s"]
        if (0 <= gap <= max_gap_s and _same_target(
                previous, span, angle_tolerance_deg=angle_tolerance_deg,
                width_tolerance_deg=width_tolerance_deg,
                center_tolerance_px=center_tolerance_px)):
            old_frames = previous["sampled_frames"]
            new_frames = span["sampled_frames"]
            total_frames = old_frames + new_frames
            for key in ("great_width_median_deg", "good_width_median_deg"):
                previous[key] = (previous[key] * old_frames + span[key] * new_frames) / total_frames
            for key in ("great_start_median_deg", "good_start_median_deg"):
                delta = (span[key] - previous[key] + 180) % 360 - 180
                previous[key] = (previous[key] + delta * new_frames / total_frames) % 360
            previous["center_median_px"] = [
                (a * old_frames + b * new_frames) / total_frames
                for a, b in zip(previous["center_median_px"], span["center_median_px"])
            ]
            previous["end_s"] = span["end_s"]
            previous["duration_s"] = previous["end_s"] - previous["start_s"]
            previous["sampled_frames"] = total_frames
            previous["merged_segments"] += span["merged_segments"]
            sweeps = (previous.get("needle_sweep_deg"), span.get("needle_sweep_deg"))
            if all(value is not None for value in sweeps):
                previous["needle_sweep_deg"] = sum(sweeps)
                previous["needle_speed_deg_s"] = (
                    previous["needle_sweep_deg"] / previous["duration_s"]
                    if previous["duration_s"] > 0 else None)
            else:
                previous["needle_sweep_deg"] = None
                previous["needle_speed_deg_s"] = None
        else:
            merged.append(span)
    return merged


def radial_stroke_width(frame, center, arc, *, light):
    """Estimate the radial pixel thickness near the middle of an observed arc."""
    cx, cy = center
    radii = np.arange(58.0, 75.01, .25)
    angles = np.linspace(arc.center-arc.width*.25, arc.center+arc.width*.25, 11)
    profiles = []
    for angle in angles:
        radians = math.radians(angle)
        xs = np.rint(cx + radii*np.cos(radians)).astype(int)
        ys = np.rint(cy + radii*np.sin(radians)).astype(int)
        profiles.append(frame[ys, xs].mean(axis=1))
    profile = np.median(profiles, axis=0)
    mask = profile > 185 if light else profile < 40
    center_index = int(round((66.5-58.0)/.25))
    if not mask[center_index]:
        return None
    left = right = center_index
    while left > 0 and mask[left-1]:
        left -= 1
    while right+1 < len(mask) and mask[right+1]:
        right += 1
    return (right-left+1)*.25


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    videos = sorted({p.resolve() for root in args.roots if root.exists()
                     for p in ([root] if root.is_file() and root.suffix.lower() == ".mkv"
                               else root.rglob("*.mkv") if root.is_dir() else [])
                     if p.is_file()})
    detector = Detector()
    hashes = {}
    rows = []
    for path in videos:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest in hashes:
            rows.append({"path": str(path), "sha256": digest,
                         "duplicate_of": hashes[digest], "screen_session_capture": True})
            continue
        hashes[digest] = str(path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            rows.append({"path": str(path), "sha256": digest, "error": "VIDEO_OPEN_FAILED",
                         "screen_session_capture": True})
            continue
        n = prompts = whites = blacks = pairs = needle_candidates = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        bouts = []
        current = []

        def finish_bout():
            nonlocal current
            if len(current) >= 5:
                angles = [(row[0], row[1]) for row in current if row[1] is not None]
                sweep = speed = None
                if len(angles) >= 3:
                    unwrapped = [angles[0][1]]
                    for _, angle in angles[1:]:
                        step = (angle - unwrapped[-1] + 180) % 360 - 180
                        unwrapped.append(unwrapped[-1] + step)
                    elapsed = angles[-1][0] - angles[0][0]
                    sweep = unwrapped[-1] - unwrapped[0]
                    if elapsed > 0:
                        speed = sweep / elapsed
                start, end = current[0][0], current[-1][0]
                great_widths = [row[2] for row in current if row[2] is not None]
                good_widths = [row[3] for row in current if row[3] is not None]
                centers = [row[4] for row in current if row[4] is not None]
                great_strokes = [row[5] for row in current if row[5] is not None]
                good_strokes = [row[6] for row in current if row[6] is not None]
                great_starts = [row[7] for row in current if row[7] is not None]
                good_starts = [row[8] for row in current if row[8] is not None]
                bouts.append({"start_s": round(start, 4), "end_s": round(end, 4),
                              "duration_s": round(end-start, 4),
                              "sampled_frames": len(current),
                              "needle_sweep_deg": round(sweep, 2) if sweep is not None else None,
                              "needle_speed_deg_s": round(speed, 2) if speed is not None else None,
                              "great_start_median_deg": round(statistics.median(great_starts), 2) if great_starts else None,
                              "good_start_median_deg": round(statistics.median(good_starts), 2) if good_starts else None,
                              "great_width_median_deg": round(statistics.median(great_widths), 2) if great_widths else None,
                              "good_width_median_deg": round(statistics.median(good_widths), 2) if good_widths else None,
                              "great_stroke_median_px": round(statistics.median(great_strokes), 2) if great_strokes else None,
                              "good_stroke_median_px": round(statistics.median(good_strokes), 2) if good_strokes else None,
                              "center_median_px": [round(statistics.median(c[0] for c in centers), 2),
                                                   round(statistics.median(c[1] for c in centers), 2)] if centers else None})
            current = []
        center_hint = None
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                m = detector.measure(frame, n / fps,
                                     center_hint=center_hint)
                n += 1
                if m.center is not None:
                    prompts += 1
                    center_hint = m.center
                whites += m.great is not None
                blacks += m.good is not None
                pairs += m.great is not None and m.good is not None
                needle_candidates += bool(m.candidates)
                if (m.prompt_score >= .85 and m.center is not None
                        and m.great is not None and m.good is not None):
                    angle = m.candidates[0].angle if len(m.candidates) == 1 else None
                    current.append((n / fps, angle,
                                    m.great.width if m.great else None,
                                    m.good.width if m.good else None,
                                    m.center,
                                    radial_stroke_width(frame, m.center, m.great, light=True),
                                    radial_stroke_width(frame, m.center, m.good, light=False),
                                    m.great.start if m.great else None,
                                    m.good.start if m.good else None))
                else:
                    finish_bout()
        finally:
            cap.release()
        finish_bout()
        rows.append({"path": str(path), "sha256": digest, "screen_session_capture": True,
                     "frames": n, "prompt_frames": prompts, "white_great_frames": whites,
                     "black_good_frames": blacks, "paired_zone_frames": pairs,
                     "needle_candidate_frames": needle_candidates,
                     "candidate_skillcheck_spans": bouts,
                     "candidate_target_spans": merge_target_spans(bouts)})
        print(f"scanned {path.name}: {n} frames", flush=True)
    valid = [r for r in rows if "frames" in r]
    plausible_spans = [span for row in valid for span in row.get("candidate_target_spans", [])
                       if span["duration_s"] >= .25 and span["needle_speed_deg_s"] is not None
                       and 180 <= span["needle_speed_deg_s"] <= 450]
    plausible_durations = [span["duration_s"] for span in plausible_spans]
    plausible_sweeps = [span["needle_sweep_deg"] for span in plausible_spans
                        if span["needle_sweep_deg"] is not None]
    plausible_speeds = [span["needle_speed_deg_s"] for span in plausible_spans]
    plausible_great_widths = [span["great_width_median_deg"] for span in plausible_spans
                              if span["great_width_median_deg"] is not None]
    plausible_good_widths = [span["good_width_median_deg"] for span in plausible_spans
                             if span["good_width_median_deg"] is not None]
    plausible_great_strokes = [span["great_stroke_median_px"] for span in plausible_spans
                               if span["great_stroke_median_px"] is not None]
    plausible_good_strokes = [span["good_stroke_median_px"] for span in plausible_spans
                              if span["good_stroke_median_px"] is not None]
    intercheck_gaps = []
    for row in valid:
        spans = [span for span in row.get("candidate_target_spans", [])
                 if span["duration_s"] >= .25 and span["needle_speed_deg_s"] is not None
                 and 180 <= span["needle_speed_deg_s"] <= 450]
        intercheck_gaps.extend(max(0, b["start_s"] - a["end_s"])
                               for a, b in zip(spans, spans[1:]))
    gap_quartiles = statistics.quantiles(intercheck_gaps, n=4) if len(intercheck_gaps) >= 4 else []
    report = {"session_mkv_files": len(videos),
              "unique_sessions": len(hashes), "decoded_sessions": len(valid),
              "total_frames": sum(r["frames"] for r in valid), "sessions": rows,
              "candidate_span_count": sum(len(r.get("candidate_skillcheck_spans", [])) for r in valid),
              "candidate_target_count": sum(len(r.get("candidate_target_spans", [])) for r in valid),
              "span_merge_policy": {"maximum_detector_dropout_s": .15,
                                    "great_good_angle_tolerance_deg": 2.0,
                                    "zone_width_tolerance_deg": 2.0,
                                    "center_tolerance_px": 3.0,
                                    "meaning": "Adjacent detector spans merge only when both measured zones and center match; this reduces brief CV dropout splits, but target tracks remain visual candidates, not game labels."},
              "candidate_span_duration_median_s": statistics.median(
                  [span["duration_s"] for r in valid for span in r.get("candidate_skillcheck_spans", [])]
              ) if any(r.get("candidate_skillcheck_spans") for r in valid) else None,
              "plausible_span_filter": {"minimum_duration_s": .25,
                  "needle_speed_deg_s_range": [180, 450], "span_count": len(plausible_spans),
                  "duration_median_s": statistics.median(plausible_durations) if plausible_durations else None,
                  "needle_sweep_median_deg": statistics.median(plausible_sweeps) if plausible_sweeps else None,
                  "needle_speed_median_deg_s": statistics.median(plausible_speeds) if plausible_speeds else None,
                  "great_width_median_deg": statistics.median(plausible_great_widths) if plausible_great_widths else None,
                  "good_width_median_deg": statistics.median(plausible_good_widths) if plausible_good_widths else None,
                  "great_stroke_median_px": statistics.median(plausible_great_strokes) if plausible_great_strokes else None,
                  "good_stroke_median_px": statistics.median(plausible_good_strokes) if plausible_good_strokes else None,
                  "intercheck_gap_median_s": statistics.median(intercheck_gaps) if intercheck_gaps else None,
                  "intercheck_gap_quartiles_s": [gap_quartiles[0], gap_quartiles[2]] if gap_quartiles else None},
              "interpretation": "These are screen recordings without postprocessed replay-MP4 annotations; live overlays visible at capture time may be included. Candidate detections are exploratory, similar scenery can create false positives, and the files have no per-check outcome labels."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("session_mkv_files", "unique_sessions",
                                               "decoded_sessions", "total_frames",
                                               "candidate_span_count",
                                               "candidate_target_count",
                                               "candidate_span_duration_median_s",
                                               "plausible_span_filter")}, indent=2))


if __name__ == "__main__":
    main()
