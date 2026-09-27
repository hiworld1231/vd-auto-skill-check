#!/usr/bin/env python3
"""Run the screen detector and planner over recorded, prompt-labeled episodes.

Episode boundaries are inherited from screen_video_geometry_audit.json and
remain heuristic. This is offline CV/planner coverage, not proof of a Roblox
outcome or a physical key press.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vd.engine import Engine
from vd.vision import Detector, retained_target


def median(values):
    return round(statistics.median(values), 4) if values else None


def parse_crop(value):
    source, coords = value.split("=", 1)
    x, y = map(int, coords.split(","))
    if x < 0 or y < 0:
        raise argparse.ArgumentTypeError("Crop coordinates must be non-negative")
    return source, (x, y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True,
                        help="screen_video_geometry_audit.json")
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--crop", type=parse_crop, action="append", required=True,
                        metavar="SOURCE=X,Y", help="320x240 ROI origin per source stem")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    crops = dict(args.crop)
    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    sessions = source["full_video_prompt_scan"]["sessions"]
    results = []
    detector = Detector()

    for session in sessions:
        stem = session["source"]
        if stem not in crops:
            raise ValueError(f"No crop supplied for source {stem}")
        path = args.video_root / f"Запись экрана_{stem}.mp4"
        if not path.is_file():
            results.append({"source": str(path), "missing": True})
            continue
        x, y = crops[stem]
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            results.append({"source": str(path), "decode_error": True})
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        episodes = session["episodes"]
        rows = [{"frames": 0, "prompt": 0, "great": 0, "good": 0,
                 "needle": 0, "multiple_needles": 0, "planned": 0,
                 "centers_x": [], "centers_y": [], "prompt_scores": []}
                for _ in episodes]
        episode_for_frame = {}
        for index, episode in enumerate(episodes):
            first = max(0, int(episode["start_s"] * fps))
            last = int(episode["end_s"] * fps)
            for frame_index in range(first, last + 1):
                episode_for_frame[frame_index] = index

        engine = Engine()
        begin_times = []
        reason_counts = Counter()
        frame_index = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            at = frame_index / fps
            roi = image[y:y + 240, x:x + 320]
            if roi.shape[:2] != (240, 320):
                raise ValueError(f"ROI {x},{y},320,240 is outside {path} frame {frame_index}")
            measurement = detector.measure(roi, at, center_hint=engine.center)
            measurement = retained_target(measurement, great=engine.target,
                                          good=engine.good, center=engine.center)
            engine.observe(measurement, now=at, held=True)
            reason_counts[engine.reason] += 1
            for event in engine.take_events():
                if event["kind"] == "BEGIN":
                    begin_times.append(event["at"])
            episode_index = episode_for_frame.get(frame_index)
            if episode_index is not None:
                row = rows[episode_index]
                row["frames"] += 1
                row["prompt_scores"].append(measurement.prompt_score)
                if measurement.prompt_score >= .80 and measurement.center is not None:
                    row["prompt"] += 1
                    row["centers_x"].append(measurement.center[0])
                    row["centers_y"].append(measurement.center[1])
                    row["great"] += measurement.great is not None
                    row["good"] += measurement.good is not None
                    row["needle"] += bool(measurement.candidates)
                    row["multiple_needles"] += len(measurement.candidates) > 1
                row["planned"] += engine.reason == "PLANNED"
            frame_index += 1
        cap.release()
        episode_results = []
        for index, (episode, row) in enumerate(zip(episodes, rows)):
            begins = [at for at in begin_times
                      if episode["start_s"] - 1 / fps <= at <= episode["end_s"] + 1 / fps]
            episode_results.append({
                "interval_s": [episode["start_s"], episode["end_s"]],
                "reported_center_px": episode["center_px"],
                "frames": row["frames"], "prompt_frames": row["prompt"],
                "great_frames": row["great"], "good_frames": row["good"],
                "needle_candidate_frames": row["needle"],
                "multiple_candidate_frames": row["multiple_needles"],
                "planned_frames": row["planned"],
                "median_center_px": [median(row["centers_x"]), median(row["centers_y"])],
                "median_prompt_score": median(row["prompt_scores"]),
                "engine_begin_times_s": [round(at, 4) for at in begins],
            })
        results.append({
            "source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "fps": fps, "decoded_frames": frame_index, "roi_xywh": [x, y, 320, 240],
            "candidate_prompt_episodes": len(episodes),
            "engine_begins": len(begin_times),
            "engine_begins_in_candidate_episodes": sum(bool(row["engine_begin_times_s"])
                                                       for row in episode_results),
            "engine_begin_times_s": [round(at, 4) for at in begin_times],
            "episode_results": episode_results,
            "engine_reason_frames": dict(reason_counts),
        })

    valid = [row for row in results if "episode_results" in row]
    report = {
        "method": "Current Detector and Engine processed every decoded frame from two user screen recordings. Candidate episode intervals come from the screen_video_geometry_audit prompt scan; each source uses its supplied fixed 320x240 ROI.",
        "limitations": [
            "The interval boundaries are prompt-template detections, not game-labeled outcomes.",
            "The user identified the red arrow in these recordings as a legacy-program overlay; detected needle candidates may be that annotation rather than the in-game needle.",
            "PLANNED means the internal planner produced a candidate only; this audit does not send Space or verify a game result.",
            "Flawless Execution equipped/active, repair state, and alone state are not available in these recordings.",
        ],
        "candidate_prompt_episodes": sum(len(s["episodes"]) for s in sessions),
        "engine_begin_episodes": sum(s.get("engine_begins_in_candidate_episodes", 0) for s in valid),
        "videos": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "videos"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
