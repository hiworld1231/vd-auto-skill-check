#!/usr/bin/env python3
"""Audit every JSON replay under the supplied roots, including old schemas.

The report is deliberately descriptive: it keeps source paths, counts exact
duplicates, validates every JSON file, and normalizes outcome/chain fields.
It does not treat a CV estimate as a confirmed game outcome.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def first_dict(data, *keys):
    for key in keys:
        value = data.get(key)
        if isinstance(value, dict):
            return value
    return {}


def normalize(data):
    evaluation = first_dict(data, "evaluation", "outcome_info")
    locked = first_dict(data, "locked_zones")
    if not locked:
        locked = {"white": data.get("locked_w"), "black": data.get("locked_b")}
    trigger = first_dict(data, "trigger", "trigger_event")
    if not trigger:
        trigger = first_dict(evaluation, "trigger")
    outcome = evaluation.get("outcome", evaluation.get("result", "UNLABELED"))
    if outcome == "UNLABELED" and data.get("outcome"):
        outcome = data["outcome"]
    return {
        "check_id": data.get("check_id"),
        "timestamp": data.get("timestamp", data.get("timestamp_iso")),
        "chain_count": data.get("chain_count", 1),
        "outcome": str(outcome).upper(),
        "great": locked.get("white", locked.get("w")),
        "good": locked.get("black", locked.get("b")),
        "trigger": trigger,
        "frame_count": len(data.get("frames", [])) if isinstance(data.get("frames"), list) else 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="replay directories to scan recursively")
    parser.add_argument("--out", type=Path, required=True, help="JSON report path")
    args = parser.parse_args()

    files = sorted({p.resolve() for root in args.roots if root.exists()
                    for p in root.rglob("*.json") if p.is_file()
                    and p.resolve() != args.out.resolve()})
    rows, invalid, auxiliary = [], [], []
    digest_sources = {}
    digest_unique = {}
    for path in files:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            invalid.append({"path": str(path), "error": str(exc)})
            continue
        if not isinstance(value, dict):
            auxiliary.append(str(path))
            continue
        normalized = normalize(value)
        is_replay = isinstance(value.get("frames"), list) and bool(value.get("check_id"))
        digest_sources.setdefault(digest, []).append(str(path))
        replay_row = {"path": str(path), "sha256": digest, "is_replay": is_replay,
                      "schema": sorted(value), **normalized}
        if is_replay:
            frames = value.get("frames", [])
            replay_row["has_needle_angle"] = any(
                isinstance(frame, dict) and isinstance(frame.get("needle_angle"), (int, float))
                for frame in frames)
            replay_row["trigger_fired"] = bool(normalized["trigger"].get("fired"))
            digest_unique.setdefault(digest, replay_row)
        rows.append(replay_row)

    replay_rows = [r for r in rows if r["is_replay"]]
    unique_hashes = {r["sha256"] for r in replay_rows}
    outcomes = Counter(r["outcome"] for r in replay_rows)
    chains = Counter(str(r["chain_count"]) for r in replay_rows)
    schemas = Counter(tuple(r["schema"]) for r in replay_rows)
    unique_rows = list(digest_unique.values())

    def zone_present(value):
        return isinstance(value, dict) and all(
            isinstance(value.get(key), (int, float))
            for key in ("start", "width"))

    def summarize_replays(items):
        outcomes_local = Counter(r["outcome"] for r in items)
        chain_local = Counter(str(r["chain_count"]) for r in items)
        return {
            "replay_records": len(items),
            "outcomes": dict(sorted(outcomes_local.items())),
            "chain_count_distribution": dict(sorted(chain_local.items(), key=lambda x: int(x[0]))),
            "paired_locked_zones": sum(zone_present(r["great"]) and zone_present(r["good"]) for r in items),
            "great_only": sum(zone_present(r["great"]) and not zone_present(r["good"]) for r in items),
            "good_only": sum(zone_present(r["good"]) and not zone_present(r["great"]) for r in items),
            "no_locked_zones": sum(not zone_present(r["great"]) and not zone_present(r["good"]) for r in items),
            "trigger_fired": sum(r["trigger_fired"] for r in items),
            "replays_with_needle_angles": sum(r["has_needle_angle"] for r in items),
        }

    # The full index above is per physical file. These summaries separately
    # describe every replay copy and every exact-content-unique replay.
    report = {
        "roots": [str(p.resolve()) for p in args.roots],
        "json_files": len(files), "valid_json": len(rows),
        "auxiliary_json_files": auxiliary, "invalid_json": invalid,
        "replay_records": len(replay_rows), "unique_replay_hashes": len(unique_hashes),
        "exact_duplicate_replay_copies": len(replay_rows) - len(unique_hashes),
        "outcomes": dict(sorted(outcomes.items())),
        "chain_count_distribution": dict(sorted(chains.items(), key=lambda x: int(x[0]))),
        "schemas": [{"count": n, "keys": list(keys)} for keys, n in schemas.most_common()],
        "all_replay_copies_summary": summarize_replays(replay_rows),
        "unique_replays_summary": summarize_replays(unique_rows),
        "unique_outcome_summaries": {
            outcome: summarize_replays([row for row in unique_rows if row["outcome"] == outcome])
            for outcome in sorted({row["outcome"] for row in unique_rows})
        },
        "records": rows,
        "duplicate_groups": [paths for paths in digest_sources.values() if len(paths) > 1],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k in {
        "json_files", "valid_json", "replay_records", "unique_replay_hashes",
        "exact_duplicate_replay_copies", "outcomes", "chain_count_distribution"}},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
