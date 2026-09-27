#!/usr/bin/env python3
"""Check every unique replay's logged outcome against its saved hit angle/zones.

This measures internal consistency of detector-produced JSON only. It is not
game-authoritative ground truth: zones and hit angles can be refined by the
legacy detector, and outcomes may have been calculated with another version.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def as_zone(value):
    if not isinstance(value, dict):
        return None
    try:
        start = float(value["start"]) % 360
        end = float(value["end"]) % 360
    except (KeyError, TypeError, ValueError):
        return None
    return {"start": start, "end": end,
            "width": float(value.get("width", (end - start) % 360)),
            "source": value.get("source", "UNSPECIFIED")}


def in_arc(angle, zone, tolerance=0.5):
    if angle is None or zone is None:
        return None
    start = (zone["start"] - tolerance) % 360
    end = (zone["end"] + tolerance) % 360
    angle = float(angle) % 360
    return start <= angle <= end if start <= end else angle >= start or angle <= end


def inspect(path, digest):
    try:
        data = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("frames"), list) or not data.get("check_id"):
        return None
    evaluation = data.get("evaluation") or data.get("outcome_info") or {}
    trigger = data.get("trigger") or data.get("trigger_event") or {}
    outcome = str(evaluation.get("outcome", data.get("outcome", "UNKNOWN"))).upper()
    check_id = str(data.get("check_id", ""))
    synthetic = check_id.lower().startswith(("bench", "test")) or str(trigger.get("reason", "")).upper() in {
        "TEST", "BENCH", "BENCHMARK"
    }
    if synthetic:
        return {"path": str(path), "sha256": digest, "synthetic_test_record": True,
                "outcome": outcome, "check_id": check_id}
    locked = data.get("locked_zones") or {}
    great = as_zone(locked.get("white") if locked else data.get("locked_w"))
    good = as_zone(locked.get("black") if locked else data.get("locked_b"))
    angle = evaluation.get("hit_angle")
    great_hit = in_arc(angle, great)
    good_hit = in_arc(angle, good)
    gap = ((good["start"] - great["end"]) % 360) if great and good else None
    merged_good = good
    if gap is not None and gap <= 10:
        merged_good = dict(good, start=great["end"])
    merged_good_hit = in_arc(angle, merged_good)
    if angle is None or great_hit is None or good_hit is None:
        expected_strict = expected_merged = "UNASSESSABLE"
    elif great_hit:
        expected_strict = expected_merged = "GREAT"
    elif good_hit:
        expected_strict = expected_merged = "GOOD"
    elif merged_good_hit:
        expected_strict = "MISS"
        expected_merged = "GOOD"
    else:
        expected_strict = expected_merged = "MISS"
    relevant = outcome in {"GREAT", "GOOD", "MISS"}
    row = {
        "path": str(path), "sha256": digest, "synthetic_test_record": False,
        "schema": "modern" if data.get("locked_zones") else "legacy",
        "outcome": outcome, "check_id": check_id,
        "hit_angle": angle, "plateau_found": evaluation.get("plateau_found"),
        "great": great, "good": good, "gap_deg": gap,
        "angle_in_great": great_hit, "angle_in_recorded_good": good_hit,
        "angle_in_gap_merged_good": merged_good_hit,
        "expected_from_saved_arcs_strict": expected_strict,
        "expected_from_saved_arcs_merged_gap_le_10_deg": expected_merged,
        "label_matches_strict_saved_arcs": outcome == expected_strict if relevant else None,
        "label_matches_merged_gap_rule": outcome == expected_merged if relevant else None,
    }
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    by_digest = {}
    invalid = []
    files = sorted({p.resolve() for root in args.roots if root.exists()
                    for p in root.rglob("*.json") if p.is_file()})
    for path in files:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest not in by_digest:
            row = inspect(path, digest)
            if row:
                by_digest[digest] = row
            else:
                try:
                    value = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    invalid.append({"path": str(path), "error": str(exc)})
                    continue
                if isinstance(value, dict) and isinstance(value.get("frames"), list) and value.get("check_id"):
                    invalid.append({"path": str(path), "error": "replay could not be normalized"})
    rows = list(by_digest.values())
    usable = [r for r in rows if not r["synthetic_test_record"]]
    scored = [r for r in usable if r["outcome"] in {"GREAT", "GOOD", "MISS"}
              and r["expected_from_saved_arcs_strict"] != "UNASSESSABLE"]
    summaries = {}
    for schema in ("modern", "legacy"):
        sub = [r for r in scored if r["schema"] == schema]
        summaries[schema] = {
            "assessed_labels": len(sub),
            "outcomes": dict(Counter(r["outcome"] for r in sub)),
            "matches_strict_saved_arcs": sum(r["label_matches_strict_saved_arcs"] for r in sub),
            "matches_merged_gap_rule": sum(r["label_matches_merged_gap_rule"] for r in sub),
            "mismatches_strict": sum(not r["label_matches_strict_saved_arcs"] for r in sub),
            "mismatches_merged": sum(not r["label_matches_merged_gap_rule"] for r in sub),
        }
    mismatch_rows = [r for r in scored if not r["label_matches_merged_gap_rule"]]
    report = {
        "method": "For every unique replay JSON, compare evaluation.hit_angle with locked GREAT/GOOD arcs using the evaluator's 0.5-degree edge tolerance; compute both strict recorded-arc and legacy <=10-degree gap-merged results.",
        "limitation": "This is an internal consistency check of detector-generated data, not game-authoritative labeling. Zone geometry may be refined after the hit, and files may reflect different evaluator versions.",
        "json_files_scanned": len(files), "unique_replay_jsons": len(rows),
        "synthetic_test_records": sum(r["synthetic_test_record"] for r in rows),
        "invalid_replays": invalid,
        "assessable_labeled_replays": len(scored),
        "summary_by_schema": summaries,
        "outcome_counts": dict(Counter(r["outcome"] for r in usable)),
        "mismatches_against_merged_gap_rule": mismatch_rows,
        "records": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k in {
        "json_files_scanned", "unique_replay_jsons", "synthetic_test_records",
        "assessable_labeled_replays", "summary_by_schema", "outcome_counts"
    }}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
