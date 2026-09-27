#!/usr/bin/env python3
"""Export observed between-check intervals for the local practice stand.

The intervals are human/session recordings, not game-authoritative spawn timers.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="deduplicated replay geometry report")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    sessions = {}
    for row in report["records"]:
        item = row["metrics"]
        if (item.get("synthetic_test_record") or not item.get("session_id")
                or item.get("start_monotonic") is None):
            continue
        sessions.setdefault(item["session_id"], []).append(item)
    intervals = []
    for session_id in sorted(sessions):
        records = sorted(sessions[session_id], key=lambda item: item["start_monotonic"])
        for previous, following in zip(records, records[1:]):
            delta = following["start_monotonic"] - previous["start_monotonic"]
            if 1.5 <= delta < 60 and following.get("chain_count", 1) == 1:
                intervals.append(round(delta, 4))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("// Observed normal-chain start intervals; session timing is approximate.\n"
                        "window.VDDefaultCheckIntervals = " + json.dumps(intervals, separators=(",", ":")) + ";\n",
                        encoding="utf-8")
    print(json.dumps({"intervals": len(intervals), "minimum_s": min(intervals),
                      "maximum_s": max(intervals)}, indent=2))


if __name__ == "__main__":
    main()
