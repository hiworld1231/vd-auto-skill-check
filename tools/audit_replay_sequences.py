#!/usr/bin/env python3
"""Audit ordered check angles and adjacent Frenzy-chain zones from replay telemetry."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import re
import statistics
from datetime import datetime


def _angle(item):
    great = item.get("great")
    if not isinstance(great, dict):
        return None
    try:
        value = float(great["start"])
    except (KeyError, TypeError, ValueError):
        return None
    return value % 360 if math.isfinite(value) else None


def _quantile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] * (high - position) + values[high] * (position - low)


def _distribution(values):
    if not values:
        return {"count": 0, "min": None, "p10": None, "median": None,
                "p90": None, "max": None, "under_15": 0, "under_50": 0}
    return {"count": len(values), "min": round(min(values), 3),
            "p10": round(_quantile(values, .1), 3),
            "median": round(statistics.median(values), 3),
            "p90": round(_quantile(values, .9), 3),
            "max": round(max(values), 3),
            "under_15": sum(value < 15 for value in values),
            "under_50": sum(value < 50 for value in values)}


def _circular_distance(left, right):
    return abs((right - left + 180) % 360 - 180)


def _shuffle_test(pair_groups, observed_close, *, threshold=50, permutations=1000, seed=20260927):
    pairs = [pair for group in pair_groups for pair in group]
    if not pairs or not permutations:
        return {"threshold_deg": threshold, "permutations": 0,
                "observed_close_pairs": observed_close, "null_mean_rate": None,
                "null_p05_rate": None, "null_p95_rate": None,
                "lower_tail_p": None}
    rng = random.Random(seed)
    null_counts = []
    for _ in range(permutations):
        close = 0
        for group in pair_groups:
            following = [right for _, right in group]
            rng.shuffle(following)
            close += sum(_circular_distance(left, right) < threshold
                         for (left, _), right in zip(group, following))
        null_counts.append(close / len(pairs))
    observed_rate = observed_close / len(pairs)
    return {"threshold_deg": threshold, "permutations": permutations,
            "observed_close_pairs": observed_close,
            "observed_rate": round(observed_rate, 6),
            "null_mean_rate": round(statistics.mean(null_counts), 6),
            "null_p05_rate": round(_quantile(null_counts, .05), 6),
            "null_p95_rate": round(_quantile(null_counts, .95), 6),
            "lower_tail_p": round((sum(rate <= observed_rate for rate in null_counts) + 1)
                                  / (permutations + 1), 6)}


def _sessions(records):
    groups = defaultdict(list)
    for row in records:
        item = row.get("metrics", row)
        session = item.get("session_id")
        at = item.get("start_monotonic")
        if (not item.get("synthetic_test_record") and session is not None
                and isinstance(at, (int, float)) and math.isfinite(at)):
            groups[str(session)].append((float(at), item))
    for rows in groups.values():
        rows.sort(key=lambda pair: pair[0])
    return groups


def _normal_sequences(groups):
    sequences = []
    pair_groups = []
    all_angles = 0
    for session, rows in groups.items():
        checks = [(at, item, _angle(item)) for at, item in rows
                  if item.get("chain_count") == 1
                  and str(item.get("outcome", "")).upper() in {"GREAT", "GOOD", "MISS"}
                  and _angle(item) is not None]
        all_angles += len(checks)
        if len(checks) < 2:
            continue
        pairs = []
        edges = []
        for previous, following in zip(checks, checks[1:]):
            interval = following[0] - previous[0]
            if 1.5 <= interval <= 60:
                pair = (previous[2], following[2])
                pairs.append(pair)
                edges.append((previous[2], following[2]))
        if pairs:
            pair_groups.append(pairs)
        # Long idle gaps split ordered samples; do not treat them as adjacent.
        current = []
        for at, item, angle in checks:
            if current and at - current[-1][0] > 60:
                if len(current) >= 3:
                    sequences.append((session, [entry[1] for entry in current]))
                current = []
            current.append((at, angle))
        if len(current) >= 3:
            sequences.append((session, [entry[1] for entry in current]))
    return all_angles, pair_groups, sequences


def _repeated_triplets(sequences, tolerance):
    motifs = []
    for session, angles in sequences:
        for start in range(len(angles) - 2):
            motifs.append((session, angles[start:start + 3]))
    matches = 0
    for index, (session, motif) in enumerate(motifs):
        for other_session, other in motifs[index + 1:]:
            if session != other_session and all(
                    _circular_distance(left, right) <= tolerance
                    for left, right in zip(motif, other)):
                matches += 1
    return {"tolerance_deg_per_position": tolerance,
            "candidate_triplets": len(motifs),
            "matching_cross_session_pairs": matches}


def strict_normal_placements(records):
    """Count only measured normal checks with explicit sessions and adjacent IDs."""
    groups = defaultdict(list)
    eligible = 0
    for row in records:
        item = row.get("metrics", row)
        great, good = item.get("great") or {}, item.get("good") or {}
        check_id = str(item.get("check_id", ""))
        match = re.search(r"_(\d+)$", check_id)
        if (not item.get("session_id") or item.get("synthetic_test_record")
                or item.get("chain_count") != 1
                or str(item.get("outcome", "")).upper() not in {"GREAT", "GOOD", "MISS"}
                or great.get("source") != "MEASURED"
                or good.get("source") != "MEASURED" or match is None):
            continue
        try:
            at = float(item["start_monotonic"])
            angle = float(great["start"]) % 360
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(at) or not math.isfinite(angle):
            continue
        eligible += 1
        groups[str(item["session_id"])].append(
            (at, int(match.group(1)), angle, check_id))

    pairs = []
    for session, checks in groups.items():
        checks.sort(key=lambda row: row[0])
        for previous, following in zip(checks, checks[1:]):
            gap = following[0] - previous[0]
            if (following[1] != previous[1] + 1 or not 1.5 <= gap <= 60):
                continue
            pairs.append({
                "distance_deg": _circular_distance(previous[2], following[2]),
                "gap_seconds": gap,
                "session_id": session,
                "left_check_id": previous[3],
                "right_check_id": following[3],
            })
    distances = [row["distance_deg"] for row in pairs]
    closest = sorted(pairs, key=lambda row: row["distance_deg"])[:10]
    return {
        "method": "explicit session; chain_count 1; GREAT/GOOD/MISS telemetry; both zone sources MEASURED; sequential check ID; 1.5–60 second interval",
        "eligible_checks": eligible,
        "explicit_sessions": len(groups),
        "adjacent_pairs": len(pairs),
        "min_separation_deg": round(min(distances), 4) if distances else None,
        "median_separation_deg": round(statistics.median(distances), 3) if distances else None,
        "pairs_under_1deg": sum(value < 1 for value in distances),
        "pairs_under_15deg": sum(value < 15 for value in distances),
        "pairs_under_50deg": sum(value < 50 for value in distances),
        "closest_pairs": [{**row, "distance_deg": round(row["distance_deg"], 4),
                            "gap_seconds": round(row["gap_seconds"], 3)} for row in closest],
        "limitations": "Sequence and zones are recorded by the legacy detector; outcome telemetry is not game-source ground truth.",
    }


def _frenzy_sequences(groups, *, permutations=1000):
    runs = []
    consecutive_pairs = 0
    missing_geometry_pairs = 0
    separation_groups = []
    for rows in groups.values():
        current = []
        for at, item in rows:
            if current:
                previous_at, previous = current[-1]
                continues = (item.get("chain_count") == previous.get("chain_count", 1) + 1
                             and 0 < at - previous_at < 1.5)
                if not continues:
                    runs.append(current)
                    current = []
            current.append((at, item))
        if current:
            runs.append(current)
    for run in runs:
        angles = [_angle(item) if (item.get("great") or {}).get("source") == "MEASURED" else None
                  for _, item in run]
        measured_pairs = []
        for previous, following in zip(angles, angles[1:]):
            consecutive_pairs += 1
            if previous is None or following is None:
                missing_geometry_pairs += 1
                continue
            measured_pairs.append((_circular_distance(previous, following)))
        if measured_pairs:
            separation_groups.append(angles)
    separations = [distance for run in separation_groups
                   for distance in (_circular_distance(left, right)
                                    for left, right in zip(run, run[1:])
                                    if left is not None and right is not None)]
    # Shuffle saved target angles only within each contiguous recorded Frenzy run.
    pair_groups = []
    for run in separation_groups:
        pair_groups.append([(left, right) for left, right in zip(run, run[1:])
                            if left is not None and right is not None])
    observed_close = sum(value < 50 for value in separations)
    return {"runs": len(runs), "consecutive_pairs": consecutive_pairs,
            "pairs_with_two_measured_great_zones": len(separations),
            "missing_geometry_pairs": missing_geometry_pairs,
            "great_start_separation_deg": _distribution(separations),
            "within_run_shuffle_under_50_deg": _shuffle_test(
                pair_groups, observed_close, permutations=permutations)}


def analyze_records(records, *, permutations=1000):
    groups = _sessions(records)
    normal_count, normal_pairs, normal_sequences = _normal_sequences(groups)
    normal_separations = [_circular_distance(left, right)
                          for group in normal_pairs for left, right in group]
    normal_close = sum(value < 50 for value in normal_separations)
    frenzy = _frenzy_sequences(groups, permutations=permutations)
    return {
        "unique_replay_records": len(records),
        "explicit_sessions": len(groups),
        "records_without_session_id": sum(
            1 for row in records
            if row.get("metrics", row).get("session_id") is None),
        "normal": {
            "eligible_checks": normal_count,
            "adjacent_pairs": len(normal_separations),
            "great_start_separation_deg": _distribution(normal_separations),
            "within_session_shuffle_under_50_deg": _shuffle_test(
                normal_pairs, normal_close, permutations=permutations),
            "repeated_three_check_angles_across_sessions": [
                _repeated_triplets(normal_sequences, tolerance)
                for tolerance in (1, 2, 5)],
            "limitations": "Only replay records with explicit session IDs can establish reliable within-session order. These are detector records, not game-source outcomes.",
        },
        "strict_normal_measured_adjacent": strict_normal_placements(records),
        "frenzy": frenzy,
        "limitations": "Records without an explicit session ID are excluded from ordered analysis. Missing geometry remains a chain break/unknown angle and is never bridged. The old bot's chain_count and FRENZY_TRANSITION are telemetry labels, not game-source state.",
    }


def infer_legacy_sessions(records, *, max_idle_seconds=300):
    """Reconstruct conservative order from old check IDs when session IDs are absent.

    A group is kept only while check IDs increment by exactly one and adjacent
    embedded wall-clock times remain within five minutes. Monotonic timestamps
    are intentionally ignored because they reset when the recorder restarts.
    """
    dated = defaultdict(list)
    for row in records:
        item = row.get("metrics", row)
        if item.get("session_id") is not None or item.get("synthetic_test_record"):
            continue
        match = re.search(r"check_(\d{8})_(\d{6})_(\d+)",
                          str(item.get("check_id", row.get("path", ""))))
        if not match:
            continue
        at = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S").timestamp()
        monotonic = item.get("start_monotonic")
        order_at = (float(monotonic) if isinstance(monotonic, (int, float))
                    and math.isfinite(monotonic) else at)
        dated[match.group(1)].append((int(match.group(3)), at, order_at, row))

    inferred = []
    for date, entries in dated.items():
        entries.sort(key=lambda entry: (entry[0], entry[1]))
        group = []
        for index, wall_at, order_at, row in entries:
            if group:
                prev_index, prev_wall_at, prev_order_at, _ = group[-1]
                wall_gap = wall_at - prev_wall_at
                order_gap = order_at - prev_order_at
                if (index != prev_index + 1 or wall_gap < 0
                        or wall_gap > max_idle_seconds or order_gap < 0
                        or order_gap > max_idle_seconds):
                    if group:
                        inferred.append(group)
                    group = []
            group.append((index, wall_at, order_at, row))
        if group:
            inferred.append(group)

    output = []
    for group_index, group in enumerate(inferred):
        if len(group) < 2:
            continue
        normalized = []
        for _, _, at, row in group:
            item = dict(row.get("metrics", row))
            item["session_id"] = f"inferred:{group_index}"
            item["start_monotonic"] = at
            normalized.append({"metrics": item})
        output.extend(normalized)
    return output, {
        "max_idle_seconds": max_idle_seconds,
        "candidate_records_without_session_id": sum(
            1 for row in records
            if row.get("metrics", row).get("session_id") is None
            and not row.get("metrics", row).get("synthetic_test_record")),
        "records_with_parseable_legacy_check_id": sum(map(len, inferred)),
        "inferred_groups_before_singleton_drop": len(inferred),
        "inferred_sessions_kept": len({row["metrics"]["session_id"] for row in output}),
        "records_in_kept_sessions": len(output),
        "ordering_source": "check_id sequence number and embedded wall-clock timestamp; process-relative monotonic clocks are ignored",
        "confidence": "inferred; gaps, restarts, and missing check IDs split sessions; filename counters may still collide",
    }


def _frenzy_chain_angle_summary(records):
    by_chain = defaultdict(list)
    for row in records:
        item = row.get("metrics", row)
        try:
            chain = int(item.get("chain_count", 1))
        except (TypeError, ValueError):
            continue
        if chain == 1 and str(item.get("outcome", "")).upper() != "FRENZY_TRANSITION":
            continue
        angle = _angle(item)
        if angle is not None:
            by_chain[chain].append(angle)
    summaries = {}
    for chain, values in sorted(by_chain.items()):
        x = sum(math.cos(math.radians(value)) for value in values) / len(values)
        y = sum(math.sin(math.radians(value)) for value in values) / len(values)
        summaries[str(chain)] = {
            "count": len(values), "circular_mean_deg": round(math.degrees(math.atan2(y, x)) % 360, 3),
            "resultant_length": round(math.hypot(x, y), 6),
        }
    return summaries


def _legacy_frenzy_geometry_hint(records):
    groups = _sessions(records)
    pairs = []
    rejected_geometry = 0
    source_pairs = defaultdict(int)
    for rows in groups.values():
        for (at_a, a), (at_b, b) in zip(rows, rows[1:]):
            try:
                consecutive = int(b.get("chain_count", 1)) == int(a.get("chain_count", 1)) + 1
            except (TypeError, ValueError):
                consecutive = False
            if not consecutive or not 0 < at_b - at_a < 1.5:
                continue
            great_a, great_b = a.get("great") or {}, b.get("great") or {}
            good_a, good_b = a.get("good") or {}, b.get("good") or {}
            try:
                widths = (float(great_a.get("width", 0)), float(great_b.get("width", 0)),
                          float(good_a.get("width", 0)), float(good_b.get("width", 0)))
            except (TypeError, ValueError):
                widths = ()
            if not (len(widths) == 4 and 9 <= widths[0] <= 12 and 9 <= widths[1] <= 12
                    and 40 <= widths[2] <= 44 and 40 <= widths[3] <= 44):
                rejected_geometry += 1
                continue
            left, right = _angle(a), _angle(b)
            if left is None or right is None:
                rejected_geometry += 1
                continue
            distance = _circular_distance(left, right)
            source_pair = (great_a.get("source", "UNSPECIFIED"),
                           great_b.get("source", "UNSPECIFIED"))
            source_pairs[str(source_pair)] += 1
            pairs.append({
                "distance_deg": distance,
                "left_angle_deg": left,
                "right_angle_deg": right,
                "delta_seconds": at_b - at_a,
                "same_zone_within_1deg": distance <= 1,
                "source_pair": source_pair,
                "left_check_id": a.get("check_id"),
                "right_check_id": b.get("check_id"),
                "outcomes": [a.get("outcome"), b.get("outcome")],
            })

    def summary(items):
        values = [item["distance_deg"] for item in items]
        return {
            "count": len(values),
            "min": round(min(values), 3) if values else None,
            "median": round(statistics.median(values), 3) if values else None,
            "max": round(max(values), 3) if values else None,
            "under_15": sum(value < 15 for value in values),
            "under_50": sum(value < 50 for value in values),
        }
    distinct = [item for item in pairs if not item["same_zone_within_1deg"]]
    return {
        "geometry_status": "plausible zone widths; legacy source labels are unspecified",
        "paired_zone_transitions": len(pairs),
        "rejected_implausible_geometry_pairs": rejected_geometry,
        "source_pair_counts": dict(source_pairs),
        "all_pairs": summary(pairs),
        "pairs_excluding_repeated_same_zone_within_1deg": summary(distinct),
        "repeated_same_zone_pairs": sum(item["same_zone_within_1deg"] for item in pairs),
        "close_pairs_under_50deg": [item for item in pairs if item["distance_deg"] < 50],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry_report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=1000)
    args = parser.parse_args()
    source = json.loads(args.geometry_report.read_text(encoding="utf-8"))
    report = analyze_records(source["records"], permutations=args.permutations)
    inferred_records, inference = infer_legacy_sessions(source["records"])
    inferred_report = analyze_records(inferred_records, permutations=args.permutations)
    report["legacy_order_inference"] = {
        **inference,
        "normal": inferred_report["normal"],
        "frenzy": inferred_report["frenzy"],
        "frenzy_chain_angle_distribution": _frenzy_chain_angle_summary(inferred_records),
        "frenzy_geometry_hint": _legacy_frenzy_geometry_hint(inferred_records),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("unique_replay_records", "explicit_sessions", "normal",
                       "strict_normal_measured_adjacent", "frenzy")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
