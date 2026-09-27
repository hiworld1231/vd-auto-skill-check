#!/usr/bin/env python3
"""Inspect archived, program-annotated replay MP4s.

Pass one or more replay roots. Paired JSON labels are used only for comparison;
the report preserves missing/ambiguous samples instead of counting them as hits.
The source recorder paints zone and needle annotations over each frame, so this
is not a raw gameplay-pixel accuracy test. Use audit_session_captures.py for
session MKVs, which lack per-check outcome labels and may include live overlays.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vd.vision import Detector, retained_target


def arc_pair(data):
    zones = data.get("locked_zones") or {}
    if not zones:
        zones = {"white": data.get("locked_w"), "black": data.get("locked_b")}
    great = zones.get("white") or zones.get("w")
    good = zones.get("black") or zones.get("b")
    return (great if isinstance(great, dict) else None,
            good if isinstance(good, dict) else None)


def cdist(a, b):
    return abs((a-b+180) % 360 - 180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    videos = sorted({p.resolve() for root in args.roots if root.exists()
                     for p in ([root] if root.is_file() and root.suffix.lower()==".mp4"
                               else root.rglob("*.mp4") if root.is_dir() else [])
                     if p.is_file()})
    detector = Detector()
    results = []
    hashes = {}
    for index, path in enumerate(videos, 1):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        duplicate_of = hashes.get(digest)
        if duplicate_of is not None:
            results.append({"path": str(path), "sha256": digest,
                            "duplicate_of": duplicate_of})
            continue
        hashes[digest] = str(path)
        label_path = path.with_suffix(".json")
        expected = None
        if label_path.exists():
            try:
                label = json.loads(label_path.read_text(encoding="utf-8"))
                expected = arc_pair(label)
            except (OSError, json.JSONDecodeError):
                label = None
        else:
            label = None
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            results.append({"path": str(path), "sha256": digest, "error": "VIDEO_OPEN_FAILED"})
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        frames = prompt_frames = paired = candidates = 0
        great_centers, great_widths, good_centers, good_widths = [], [], [], []
        center_hint = None
        great_hint = good_hint = None
        initial_pair = None
        target_transition_frames = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                measurement = detector.measure(frame, frames/fps, center_hint=center_hint)
                frames += 1
                if measurement.center is not None:
                    center_hint = measurement.center
                    prompt_frames += 1
                if measurement.candidates:
                    candidates += 1
                measurement = retained_target(measurement, great=great_hint,
                                              good=good_hint, center=center_hint)
                if measurement.great is not None and measurement.good is not None:
                    pair=(measurement.great,measurement.good)
                    if initial_pair is None:
                        initial_pair=pair
                        great_hint,good_hint=pair
                    stable=(cdist(pair[0].center,initial_pair[0].center)<=3
                            and abs(pair[0].width-initial_pair[0].width)<=3
                            and cdist(pair[1].center,initial_pair[1].center)<=3
                            and abs(pair[1].width-initial_pair[1].width)<=3)
                    if stable:
                        paired += 1
                        great_centers.append(pair[0].center)
                        great_widths.append(pair[0].width)
                        good_centers.append(pair[1].center)
                        good_widths.append(pair[1].width)
                    else:
                        target_transition_frames += 1
        finally:
            cap.release()
        row = {"path": str(path), "sha256": digest,
               "pixel_source": "legacy_program_annotated_replay",
               "raw_gameplay_cv_validation": False, "frames": frames,
               "prompt_frames": prompt_frames, "paired_target_frames": paired,
               "needle_candidate_frames": candidates,
               "target_transition_frames": target_transition_frames,
               "paired_target_rate": paired/frames if frames else None,
               "label_outcome": ((label.get("evaluation") or label.get("outcome_info") or {}).get("outcome")
                                 if label else None)}
        if expected and expected[0] and expected[1] and great_centers:
            eg, eb = expected
            row["zone_comparison"] = {
                "great_center_error_deg": round(statistics.median(
                    cdist(c, float(eg.get("center", (float(eg["start"])+float(eg["end"]))/2)))
                    for c in great_centers), 3),
                "great_width_error_deg": round(statistics.median(great_widths)-float(eg.get("width", float(eg["end"])-float(eg["start"]))), 3),
                "good_center_error_deg": round(statistics.median(
                    cdist(c, float(eb.get("center", (float(eb["start"])+float(eb["end"]))/2)))
                    for c in good_centers), 3),
                "good_width_error_deg": round(statistics.median(good_widths)-float(eb.get("width", float(eb["end"])-float(eb["start"]))), 3),
            }
        results.append(row)
        if index % 25 == 0:
            print(f"scanned {index}/{len(videos)} videos", flush=True)

    distinct = [r for r in results if "duplicate_of" not in r]
    valid = [r for r in distinct if "error" not in r and "frames" in r]
    with_labels = [r for r in valid if "zone_comparison" in r]
    report = {"videos": len(videos), "unique_videos": len(distinct),
              "valid_videos": len(valid), "paired_labeled_videos": len(with_labels),
              "outcomes": dict(Counter(str(r.get("label_outcome", "UNLABELED"))
                                         for r in with_labels)),
              "video_results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "video_results"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
