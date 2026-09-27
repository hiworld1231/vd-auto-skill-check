#!/usr/bin/env python3
"""Summarize detector-recorded geometry, needle speeds, and check cadence.

These values were produced by the legacy detector and are not raw-pixel ground
truth. Exact-content duplicates are counted once for distribution summaries.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def zone(value):
    if not isinstance(value, dict):
        return None
    try:
        start = float(value["start"]) % 360
        width = (float(value["width"]) if value.get("width") is not None else
                 (float(value["end"]) - start) % 360)
    except (KeyError, TypeError, ValueError):
        return None
    if not 0 < width < 360:
        return None
    return {"start": start, "width": width,
            "center": float(value.get("center", (start + width / 2) % 360)) % 360,
            "source": value.get("source", "UNSPECIFIED")}


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def stats(values):
    values = [float(v) for v in values if v is not None]
    return {"n": len(values),
            "median": statistics.median(values) if values else None,
            "p10": percentile(values, .10), "p90": percentile(values, .90)}


def normalize_replay(data):
    evaluation = data.get("evaluation") or data.get("outcome_info") or {}
    locked = data.get("locked_zones") or {}
    great = zone(locked.get("white") if locked else data.get("locked_w"))
    good = zone(locked.get("black") if locked else data.get("locked_b"))
    if not great:
        for frame in data.get("frames", []):
            great = zone(frame.get("white_zone"))
            if great:
                break
    if not good:
        for frame in data.get("frames", []):
            good = zone(frame.get("black_zone"))
            if good:
                break
    trigger = data.get("trigger") or data.get("trigger_event") or {}
    check_id = str(data.get("check_id", ""))
    trigger_reason = str(trigger.get("reason", "")).upper()
    session = data.get("session")
    session_id = session.get("session_id") if isinstance(session, dict) else None
    speeds = []
    for key in ("speed_deg_s", "speed_at_fire", "live_speed", "prediction_speed_used"):
        value = trigger.get(key)
        if isinstance(value, (int, float)) and 0 < value < 1000:
            speeds.append(float(value))
    for frame in data.get("frames", []):
        for key in ("speed_deg_s",):
            value = frame.get(key)
            if isinstance(value, (int, float)) and 0 < value < 1000:
                speeds.append(float(value))
        pred = frame.get("pred")
        if isinstance(pred, dict):
            value = pred.get("speed_deg_s")
            if isinstance(value, (int, float)) and 0 < value < 1000:
                speeds.append(float(value))
    timed_angles = []
    for frame in data.get("frames", []):
        if frame.get("is_pre_roll"):
            continue
        angle = frame.get("needle_angle")
        timestamp = frame.get("t")
        if timestamp is None and frame.get("time_rel_ms") is not None:
            timestamp = float(frame["time_rel_ms"]) / 1000
        if isinstance(angle, (int, float)) and isinstance(timestamp, (int, float)):
            timed_angles.append((float(timestamp), float(angle) % 360))
    observed_speeds = []
    left = 0
    while left < len(timed_angles) - 1:
        t0, a0 = timed_angles[left]
        right = left + 1
        while right < len(timed_angles) and timed_angles[right][0] - t0 < .08:
            right += 1
        if right == len(timed_angles):
            break
        t1, a1 = timed_angles[right]
        dt = t1 - t0
        step = (a1 - a0 + 180) % 360 - 180
        speed = step / dt if dt > 0 else 0
        if .08 <= dt <= .18 and 100 <= speed <= 700:
            observed_speeds.append(speed)
        left = right
    outcome = str(evaluation.get("outcome", data.get("outcome", "UNLABELED"))).upper()
    synthetic = (check_id.lower().startswith(("bench", "test"))
                 or trigger_reason in {"TEST", "BENCH", "BENCHMARK"})
    return {"check_id": check_id, "trigger_reason": trigger_reason,
            "synthetic_test_record": synthetic,
            "session_id": session_id,
            "start_monotonic": data.get("start_monotonic"),
            "outcome": outcome, "chain_count": int(data.get("chain_count", 1)),
            "great": great, "good": good,
            "gap": (good["start"] - great["start"] - great["width"]) % 360
                   if great and good else None,
            "recorded_speed_median": statistics.median(speeds) if speeds else None,
            "observed_speed_median": statistics.median(observed_speeds) if observed_speeds else None,
            "zone_source": great["source"] if great else None,
            "schema": "modern" if data.get("locked_zones") else "legacy"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--zone-samples", type=Path,
                        help="optional JS output for sampling plausible paired zone geometry")
    args = parser.parse_args()
    files = sorted({path.resolve() for root in args.roots if root.exists()
                    for path in root.rglob("*.json") if path.is_file()})
    by_digest = {}
    invalid = []
    for path in files:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest in by_digest:
            continue
        try:
            data = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            invalid.append({"path": str(path), "error": str(exc)})
            continue
        if not isinstance(data, dict) or not isinstance(data.get("frames"), list) or not data.get("check_id"):
            continue
        by_digest[digest] = {"path": str(path), "metrics": normalize_replay(data)}

    records = list(by_digest.values())
    metrics = [r["metrics"] for r in records]
    outcomes = Counter(m["outcome"] for m in metrics)
    grouped = defaultdict(list)
    by_chain = defaultdict(list)
    for item in metrics:
        grouped[item["schema"]].append(item)
        by_chain[item["chain_count"]].append(item)

    speed_by_schema_and_chain = {}
    for schema, items in grouped.items():
        speed_by_schema_and_chain[schema] = {}
        for label, selected in (
            ("chain_count_1", [m for m in items if m["chain_count"] == 1]),
            ("chain_count_2_plus", [m for m in items if m["chain_count"] > 1]),
        ):
            speed_by_schema_and_chain[schema][label] = stats(
                [m["observed_speed_median"] for m in selected
                 if not m["synthetic_test_record"]]
            )

    session_checks = defaultdict(list)
    for metric in metrics:
        if (not metric["synthetic_test_record"] and metric["session_id"]
                and isinstance(metric["start_monotonic"], (int, float))):
            session_checks[metric["session_id"]].append(metric)
    adjacent_intervals = {"all": [], "next_chain_count_1": [], "next_chain_count_2_plus": []}
    for checks in session_checks.values():
        checks.sort(key=lambda row: row["start_monotonic"])
        for previous, following in zip(checks, checks[1:]):
            delta = following["start_monotonic"] - previous["start_monotonic"]
            if 0 < delta < 60:
                adjacent_intervals["all"].append(delta)
                category = "next_chain_count_2_plus" if following["chain_count"] > 1 else "next_chain_count_1"
                adjacent_intervals[category].append(delta)

    def geometry_summary(items):
        return {"records": len(items),
                "great_width_deg": stats([m["great"]["width"] if m["great"] else None for m in items]),
                "good_width_deg": stats([m["good"]["width"] if m["good"] else None for m in items]),
                "paired_gap_deg": stats([m["gap"] for m in items]),
                "recorded_needle_speed_deg_s": stats([m["recorded_speed_median"] for m in items]),
                "frame_derived_needle_speed_deg_s": stats([m["observed_speed_median"] for m in items]),
                "great_geometry_source_counts": dict(Counter(m["zone_source"] for m in items if m["great"]))}

    outcome_geometry = {key: geometry_summary([m for m in metrics if m["outcome"] == key])
                        for key in sorted(outcomes)}
    sample_rows = [[round(m["great"]["width"], 3), round(m["good"]["width"], 3),
                    round(m["gap"], 3), round(m["great"]["start"], 3)]
                   for m in metrics if not m["synthetic_test_record"]
                   and m["great"] and m["good"]
                   and 9.5 <= m["great"]["width"] <= 11.5
                   and 40 <= m["good"]["width"] <= 44
                   and 0 <= m["gap"] <= 2]
    report = {"json_files_scanned": len(files), "unique_replay_jsons": len(records),
              "synthetic_test_records": sum(m["synthetic_test_record"] for m in metrics),
              "plausible_nonbenchmark_zone_samples": len(sample_rows),
              "invalid_json": invalid, "outcomes": dict(sorted(outcomes.items())),
              "all_unique_records": geometry_summary(metrics),
              "by_schema": {key: geometry_summary(value) for key, value in grouped.items()},
              "by_chain_count": {str(key): geometry_summary(value)
                                 for key, value in sorted(by_chain.items())},
              "frame_derived_speed_by_schema_and_chain_count": speed_by_schema_and_chain,
              "adjacent_check_start_intervals_with_explicit_session_ids": {
                  "session_count": len(session_checks),
                  "all": stats(adjacent_intervals["all"]),
                  "next_chain_count_1": stats(adjacent_intervals["next_chain_count_1"]),
                  "next_chain_count_2_plus": stats(adjacent_intervals["next_chain_count_2_plus"]),
                  "chain_count_note": "The old bot increments chain_count when it sees the next check within 1.5 seconds; this is bot session metadata, not a Flawless Execution label.",
              },
              "by_outcome": outcome_geometry,
              "interpretation": "Zone and speed values are detector telemetry saved by the legacy program. Replay media contains annotations; these distributions are not raw-pixel ground truth or game-authoritative settings.",
              "records": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.zone_samples is not None:
        args.zone_samples.parent.mkdir(parents=True, exist_ok=True)
        args.zone_samples.write_text("// Plausible paired-zone geometry sampled from unique replay JSON telemetry.\n"
                                     "window.VDZoneSamples = " +
                                     json.dumps(sample_rows, separators=(",", ":")) + ";\n",
                                     encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k in
                      ("json_files_scanned", "unique_replay_jsons", "synthetic_test_records",
                       "plausible_nonbenchmark_zone_samples", "invalid_json",
                       "outcomes", "all_unique_records", "by_schema")},
                     ensure_ascii=False, indent=2))
    print(f"plausible paired zone samples: {len(sample_rows)}")


if __name__ == "__main__":
    main()
