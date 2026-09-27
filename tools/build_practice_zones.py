#!/usr/bin/env python3
"""Build the normal-mode zone pool, excluding fast-chain/Frenzy records."""
import argparse
import json
from pathlib import Path


def build_samples(records):
    """Keep one sample for each completed normal check with paired geometry."""
    rows = []
    for record in records:
        item = record.get("metrics", record)
        great, good = item.get("great"), item.get("good")
        if (item.get("synthetic_test_record") or item.get("chain_count") != 1
                or item.get("outcome") not in {"GREAT", "GOOD", "MISS"}
                or not great or not good):
            continue
        width, good_width, gap = great.get("width"), good.get("width"), item.get("gap")
        try:
            width, good_width, gap, start = (float(width), float(good_width), float(gap),
                                              float(great["start"]) % 360)
        except (KeyError, TypeError, ValueError):
            continue
        if (9.5 <= width <= 11.5 and 40 <= good_width <= 44
                and 0 <= gap <= 2):
            rows.append([round(width, 3), round(good_width, 3), round(gap, 3),
                         round(start, 3)])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="audit_replay_geometry.py JSON report")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = build_samples(report["records"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "// Completed normal-chain replay checks only; unresolved and rapid-chain records excluded.\n"
        "window.VDDefaultZoneSamples = " + json.dumps(rows, separators=(",", ":")) + ";\n",
        encoding="utf-8")
    starts = [row[3] for row in rows]
    print(json.dumps({"normal_zone_samples": len(rows),
                      "min_great_start_deg": min(starts) if starts else None,
                      "max_great_start_deg": max(starts) if starts else None}, indent=2))


if __name__ == "__main__":
    main()
