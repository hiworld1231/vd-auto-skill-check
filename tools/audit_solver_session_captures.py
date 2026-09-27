#!/usr/bin/env python3
"""Evaluate the current screen detector against every archived MKV session.

Candidate intervals are inherited from audit_session_captures.py and are
heuristic, not game-labeled ground truth. This report measures detector
availability and background activity; it does not claim hit accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vd.vision import Detector
from vd.engine import Engine
from vd.vision import retained_target


def candidate_targets(session):
    """Use merged visual targets when available; keep old reports readable."""
    return session.get("candidate_target_spans", session.get("candidate_skillcheck_spans", []))


def median(values):
    return statistics.median(values) if values else None


def summarize(rows):
    total = len(rows)
    result = {
        "frames": total,
        "prompt_score_ge_0_80": sum(r["prompt"] >= 0.80 for r in rows),
        "great_present": sum(r["great"] for r in rows),
        "good_present": sum(r["good"] for r in rows),
        "needle_candidate_present": sum(r["needles"] > 0 for r in rows),
        "multiple_needle_candidates": sum(r["needles"] > 1 for r in rows),
        "median_prompt_score": median([r["prompt"] for r in rows]),
        "candidate_angles": [r["angle"] for r in rows if r["angle"] is not None],
    }
    for key in ("prompt_score_ge_0_80", "great_present", "good_present",
                "needle_candidate_present", "multiple_needle_candidates"):
        result[key + "_rate"] = result[key] / total if total else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    report = {
        "method": "Current vd.vision.Detector and vd.engine.Engine evaluated on every decoded frame. The source intervals are heuristic candidate spans from audit_session_captures.py, not verified check labels; engine begin events are matched to spans with a one-frame edge tolerance.",
        "limitations": [
            "MKVs have no authoritative per-frame check labels or FE/repair/alone state.",
            "Desktop overlays may be present in captured frames; detector hits are not proof of original game pixels.",
            "The user identifies the red arrow as a legacy-program annotation, while collector.py does not draw into its MKV frames; the actual needle color is therefore not established for every source.",
            "Coverage and background activity do not establish successful physical inputs or game-authoritative outcomes.",
        ],
        "sessions": [],
    }
    for session in source["sessions"]:
        path = Path(session["path"])
        if not path.is_file():
            report["sessions"].append({"path": str(path), "missing": True})
            continue
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            report["sessions"].append({"path": str(path), "decode_error": True})
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        spans = candidate_targets(session)
        all_rows = []
        active_rows = []
        outside_rows = []
        engine = Engine()
        begin_events = []
        index = 0
        detector = Detector()
        while True:
            ok, image = cap.read()
            if not ok:
                break
            at = index / fps
            m = detector.measure(image, at, center_hint=engine.center)
            m = retained_target(m, great=engine.target, good=engine.good,
                                center=engine.center)
            engine.observe(m, now=at, held=True)
            begin_events.extend(e for e in engine.take_events() if e["kind"] == "BEGIN")
            row = {
                "prompt": m.prompt_score,
                "great": m.great is not None,
                "good": m.good is not None,
                "needles": len(m.candidates),
                "angle": m.candidates[0].angle if m.candidates else None,
                "reason": m.reason,
            }
            active = any(s["start_s"] <= at <= s["end_s"] for s in spans)
            (active_rows if active else outside_rows).append(row)
            all_rows.append(row)
            index += 1
        cap.release()
        if index:
            engine.cancel("AUDIT_ENDED", index / fps)
            begin_events.extend(e for e in engine.take_events() if e["kind"] == "BEGIN")
        edge_tolerance = 1.0 / fps
        span_begin_events = [e for e in begin_events
                             if any(s["start_s"] - edge_tolerance <= e["at"]
                                    <= s["end_s"] + edge_tolerance for s in spans)]
        active = summarize(active_rows)
        outside = summarize(outside_rows)
        overall = summarize(all_rows)
        speeds = []
        for span in spans:
            seq = [r for r in active_rows if r["angle"] is not None]
            # Use original timestamps from the full decoded sequence for speeds.
            start, end = span["start_s"], span["end_s"]
            seq = [(i / fps, all_rows[i]["angle"]) for i in range(len(all_rows))
                   if start <= i / fps <= end and all_rows[i]["angle"] is not None]
            unwrapped = []
            for at, angle in seq:
                if unwrapped:
                    prev_at, prev_angle = unwrapped[-1]
                    delta = (angle - prev_angle + 180.0) % 360.0 - 180.0
                    if delta < -5 or delta > 80:
                        continue
                    unwrapped.append((at, prev_angle + delta))
                else:
                    unwrapped.append((at, angle))
            for (t0, a0), (t1, a1) in zip(unwrapped, unwrapped[1:]):
                if t1 > t0:
                    speed = (a1 - a0) / (t1 - t0)
                    if 100 <= speed <= 700:
                        speeds.append(speed)
        for group in (active, outside, overall):
            group.pop("candidate_angles")
        session_result = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "fps": fps,
            "decoded_frames": index,
            "candidate_span_count": len(spans),
            "candidate_span_intervals_s": [[s["start_s"], s["end_s"]] for s in spans],
            "engine_span_edge_tolerance_s": edge_tolerance,
            "engine_begin_count": len(begin_events),
            "engine_begins_in_candidate_spans": len(span_begin_events),
            "engine_begins_outside_candidate_spans": len(begin_events)-len(span_begin_events),
            "engine_begin_times_s": [round(e["at"], 4) for e in begin_events],
            "candidate_spans": active,
            "outside_candidate_spans": outside,
            "all_frames": overall,
            "active_needle_speed_deg_s_median": median(speeds),
        }
        report["sessions"].append(session_result)
        print(f"scanned {path.name}: {index} frames, {len(spans)} candidate spans",
              flush=True)
    report["aggregate_summary"] = {}
    for label, key in (("candidate_spans", "candidate_spans"),
                       ("outside_candidate_spans", "outside_candidate_spans"),
                       ("all_frames", "all_frames")):
        groups = [s[key] for s in report["sessions"] if key in s]
        frames = sum(group["frames"] for group in groups)
        counts = {name: sum(group[name] for group in groups)
                  for name in ("prompt_score_ge_0_80", "great_present", "good_present",
                               "needle_candidate_present", "multiple_needle_candidates")}
        report["aggregate_summary"][label] = {
            "frames": frames, **counts,
            **{name + "_rate": count / frames if frames else None
               for name, count in counts.items()},
        }
    report["aggregate_summary"]["engine"] = {
        "begin_events": sum(s.get("engine_begin_count", 0) for s in report["sessions"]),
        "begins_outside_candidate_spans_with_one_frame_tolerance": sum(
            s.get("engine_begins_outside_candidate_spans", 0) for s in report["sessions"]),
        "candidate_spans": sum(s.get("candidate_span_count", 0) for s in report["sessions"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    summary = [{"path": s["path"], "frames": s.get("decoded_frames"),
                "spans": s.get("candidate_span_count"),
                "active_prompt_rate": s.get("candidate_spans", {}).get("prompt_score_ge_0_80_rate"),
                "active_needle_rate": s.get("candidate_spans", {}).get("needle_candidate_present_rate"),
                "outside_needle_rate": s.get("outside_candidate_spans", {}).get("needle_candidate_present_rate"),
                "engine_begins": s.get("engine_begin_count"),
                "begins_outside": s.get("engine_begins_outside_candidate_spans")}
               for s in report["sessions"]]
    print(json.dumps({"sessions": len(report["sessions"]), "results": summary},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
