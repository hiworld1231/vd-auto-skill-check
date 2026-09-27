#!/usr/bin/env python3
"""Audit check ordering from original replay manifests and paired JSON records."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics
from datetime import datetime

if __package__:
    from .audit_replay_sequences import _circular_distance, _distribution, _repeated_triplets, _shuffle_test
else:
    from audit_replay_sequences import _circular_distance, _distribution, _repeated_triplets, _shuffle_test


def _index(check_id):
    match = re.search(r"_(\d+)$", str(check_id))
    return int(match.group(1)) if match else None


def _timestamp(value):
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def _geometry(data, manifest_row):
    metrics = data.get("metrics", data)
    zones = metrics.get("locked_zones", {}) or {}
    great = zones.get("white", zones.get("w", metrics.get("locked_w", metrics.get("great"))))
    good = zones.get("black", zones.get("b", metrics.get("locked_b", metrics.get("good"))))
    great = great if isinstance(great, dict) else {}
    good = good if isinstance(good, dict) else {}
    evaluation = metrics.get("evaluation", metrics.get("outcome_info", {})) or {}
    try:
        angle = float(great["start"]) % 360
    except (KeyError, TypeError, ValueError):
        angle = None
    def numeric(zone, key):
        try:
            return float(zone[key])
        except (KeyError, TypeError, ValueError):
            return None
    try:
        chain = int(metrics.get("chain_count", manifest_row.get("chain_count", 1)))
    except (TypeError, ValueError):
        chain = 1
    gap = numeric(metrics, "gap")
    if gap is None and angle is not None:
        great_width = numeric(great, "width")
        good_start = numeric(good, "start")
        if great_width is not None and good_start is not None:
            gap = (good_start - angle - great_width) % 360
    return {
        "check_id": manifest_row.get("check_id", metrics.get("check_id")),
        "timestamp": _timestamp(manifest_row.get("timestamp")
                                  or metrics.get("timestamp_iso")
                                  or metrics.get("timestamp")),
        "chain_count": chain,
        "outcome": str(evaluation.get("outcome", evaluation.get("result",
                         manifest_row.get("outcome", "UNKNOWN")))).upper(),
        "great_start": angle,
        "great_width": numeric(great, "width"),
        "good_width": numeric(good, "width"),
        "gap": gap,
        "geometry_source": str(great.get("source", "UNSPECIFIED")).upper(),
    }


def load_manifest_sessions(root):
    """Read each distinct manifest once, retaining original row order."""
    manifest_hashes = set()
    replay_cache = {}
    raw_sessions = []
    missing_json = 0
    invalid_json = 0
    for manifest in sorted(Path(root).rglob("manifest.jsonl")):
        raw_manifest = manifest.read_bytes()
        digest = hashlib.sha256(raw_manifest).hexdigest()
        if digest in manifest_hashes:
            continue
        manifest_hashes.add(digest)
        rows = []
        for line in raw_manifest.decode(errors="replace").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            json_name = (entry.get("files") or {}).get("json")
            if not json_name:
                missing_json += 1
                continue
            path = (manifest.parent / json_name).resolve()
            if not path.is_file():
                missing_json += 1
                continue
            if path not in replay_cache:
                try:
                    payload = path.read_bytes()
                    replay_cache[path] = (json.loads(payload), hashlib.sha256(payload).hexdigest())
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    invalid_json += 1
                    continue
            payload, record_hash = replay_cache[path]
            item = _geometry(payload, entry)
            item["sequence_index"] = _index(item["check_id"])
            item["replay_sha256"] = record_hash
            rows.append(item)

        group = []
        for item in rows:
            if group:
                previous = group[-1]
                gap = (item["timestamp"] - previous["timestamp"]
                       if item["timestamp"] is not None and previous["timestamp"] is not None
                       else None)
                if (item["sequence_index"] is None or previous["sequence_index"] is None
                        or item["sequence_index"] != previous["sequence_index"] + 1
                        or gap is None or gap < 0 or gap > 300):
                    raw_sessions.append(group)
                    group = []
            group.append(item)
        if group:
            raw_sessions.append(group)

    unique_sessions = {}
    for session in raw_sessions:
        if not session:
            continue
        signature = tuple((row["check_id"], row["replay_sha256"]) for row in session)
        unique_sessions.setdefault(signature, session)
    sessions = list(unique_sessions.values())
    return sessions, {
        "source_root": str(Path(root).resolve()),
        "unique_manifest_contents": len(manifest_hashes),
        "raw_session_segments": len(raw_sessions),
        "deduplicated_ordered_sessions": len(sessions),
        "records_in_deduplicated_sessions": sum(map(len, sessions)),
        "missing_json_references": missing_json,
        "invalid_json_records": invalid_json,
        "ordering_source": "original manifest row order, checked against incrementing check IDs and wall-clock timestamps",
    }


def analyze_normal(sessions, *, permutations=10000):
    routes = []
    eligible = 0
    for session_id, session in enumerate(sessions):
        route = []
        for row in session:
            valid = (row["chain_count"] == 1
                     and row["outcome"] in {"GREAT", "GOOD", "MISS"}
                     and row["great_start"] is not None
                     and row["great_width"] is not None
                     and 9 <= row["great_width"] <= 12)
            eligible += int(valid)
            adjacent_to_previous = False
            if route and valid:
                prev = route[-1]
                gap = row["timestamp"] - prev["timestamp"]
                adjacent_to_previous = (
                    row["sequence_index"] == prev["sequence_index"] + 1
                    and 1.5 <= gap <= 60)
            if valid and adjacent_to_previous:
                route.append(row)
            else:
                if len(route) >= 2:
                    routes.append((session_id, route))
                route = [row] if valid else []
        if len(route) >= 2:
            routes.append((session_id, route))

    pair_groups = [[(a["great_start"], b["great_start"])
                    for a, b in zip(route, route[1:])]
                   for _, route in routes]
    pair_groups = [group for group in pair_groups if group]
    distances = [_circular_distance(a, b) for group in pair_groups for a, b in group]
    close = sum(value < 50 for value in distances)
    motifs = [(session_id, [row["great_start"] for row in route])
              for session_id, route in routes]
    return {
        "eligible_normal_checks": eligible,
        "ordered_routes_with_at_least_two_checks": len(routes),
        "strict_adjacent_pairs": len(distances),
        "great_start_separation_deg": _distribution(distances),
        "within_route_shuffle_under_50_deg": _shuffle_test(
            pair_groups, close, permutations=permutations),
        "repeated_three_check_sequences_across_routes": [
            _repeated_triplets(motifs, tolerance) for tolerance in (1, 2, 5)],
        "ordering_rule": "Only adjacent manifest rows with consecutive check IDs, chain_count=1, completed outcomes, plausible GREAT width, and 1.5–60 s interval are paired.",
    }


def build_normal_routes(sessions):
    """Export exact contiguous ordinary check geometry in manifest order."""
    routes = []
    for session in sessions:
        route = []
        for row in session:
            valid = (row["chain_count"] == 1
                     and row["outcome"] in {"GREAT", "GOOD", "MISS"}
                     and row["great_start"] is not None
                     and row["great_width"] is not None and 9 <= row["great_width"] <= 12
                     and row["good_width"] is not None and 40 <= row["good_width"] <= 44
                     and row["gap"] is not None and 0 <= row["gap"] <= 2)
            adjacent = False
            if route and valid:
                previous = route[-1]
                interval = row["timestamp"] - previous["timestamp"]
                adjacent = (row["sequence_index"] == previous["sequence_index"] + 1
                            and 1.5 <= interval <= 60)
            if valid and adjacent:
                route.append(row)
            else:
                if len(route) >= 2:
                    routes.append(route)
                route = [row] if valid else []
        if len(route) >= 2:
            routes.append(route)
    return [[[
        round(row["great_width"], 3), round(row["good_width"], 3),
        round(row["gap"], 3), round(row["great_start"], 3),
    ] for row in route] for route in routes]


def analyze_rapid_chains(sessions):
    routes = []
    for session_id, session in enumerate(sessions):
        route = []
        for index, row in enumerate(session):
            if route:
                prev = route[-1]
                dt = row["timestamp"] - prev["timestamp"]
                connected = (row["chain_count"] == prev["chain_count"] + 1
                             and 0 < dt < 1.5)
                if connected:
                    route.append(row)
                    continue
                if len(route) >= 2:
                    routes.append((session_id, route))
                route = []
            next_is_fast_chain = False
            if index + 1 < len(session):
                following = session[index + 1]
                next_is_fast_chain = (
                    row["chain_count"] == 1 and following["chain_count"] == 2
                    and 0 < following["timestamp"] - row["timestamp"] < 1.5)
            if (row["chain_count"] > 1 or row["outcome"] == "FRENZY_TRANSITION"
                    or next_is_fast_chain):
                route = [row]
        if len(route) >= 2:
            routes.append((session_id, route))

    transitions = []
    by_source = defaultdict(list)
    for session_id, route in routes:
        for left, right in zip(route, route[1:]):
            if left["great_start"] is None or right["great_start"] is None:
                continue
            distance = _circular_distance(left["great_start"], right["great_start"])
            plausible = (left["great_width"] is not None and right["great_width"] is not None
                         and left["good_width"] is not None and right["good_width"] is not None
                         and 9 <= left["great_width"] <= 12 and 9 <= right["great_width"] <= 12
                         and 40 <= left["good_width"] <= 44 and 40 <= right["good_width"] <= 44)
            measured = (left["geometry_source"] == "MEASURED"
                        and right["geometry_source"] == "MEASURED")
            item = {
                "from_check": left["check_id"], "to_check": right["check_id"],
                "from_chain": left["chain_count"], "to_chain": right["chain_count"],
                "from_angle": left["great_start"], "to_angle": right["great_start"],
                "separation_deg": round(distance, 3),
                "interval_s": round(right["timestamp"] - left["timestamp"], 3),
                "outcomes": [left["outcome"], right["outcome"]],
                "geometry_sources": [left["geometry_source"], right["geometry_source"]],
                "plausible_paired_geometry": plausible,
                "measured_geometry": measured,
            }
            transitions.append(item)
            by_source["measured" if measured else "unspecified"].append(item)

    def summary(items):
        distances = [item["separation_deg"] for item in items]
        return {
            "pairs": len(items),
            "min_deg": min(distances) if distances else None,
            "median_deg": round(statistics.median(distances), 3) if distances else None,
            "under_15_deg": sum(value < 15 for value in distances),
            "under_50_deg": sum(value < 50 for value in distances),
            "close_examples": [item for item in items if item["separation_deg"] < 50][:20],
        }
    return {
        "rapid_chain_routes": len(routes),
        "checks_in_routes": sum(len(route) for _, route in routes),
        "all_paired_geometry_candidates": summary([x for x in transitions if x["plausible_paired_geometry"]]),
        "measured_geometry_candidates": summary([x for x in transitions if x["plausible_paired_geometry"] and x["measured_geometry"]]),
        "unspecified_geometry_candidates": summary([x for x in transitions if x["plausible_paired_geometry"] and not x["measured_geometry"]]),
        "all_transitions": transitions,
        "limits": "Rapid chain_count/time labels come from the legacy recorder; they do not prove which perk was active. Old source labels may be UNSPECIFIED.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=10000)
    args = parser.parse_args()
    sessions, inventory = load_manifest_sessions(args.manifest_root)
    report = {
        "inventory": inventory,
        "normal": analyze_normal(sessions, permutations=args.permutations),
        "rapid_chains": analyze_rapid_chains(sessions),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "inventory": inventory,
        "normal": {key: value for key, value in report["normal"].items()
                   if key not in {"repeated_three_check_sequences_across_routes"}},
        "rapid_chains": {key: value for key, value in report["rapid_chains"].items()
                         if key not in {"all_transitions"}},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
