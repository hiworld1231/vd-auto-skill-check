#!/usr/bin/env python3
"""Check every unique replay JSON for explicit Flawless Execution context.

This scans JSON object keys (including nested frame/evaluation records) rather
than guessing state from chain_count, timestamps, or detector outcomes.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


STATE_KEY_TERMS = {
    "flawless", "perk_equipped", "perk_active", "perk_tier", "alone",
    "is_alone", "solo", "is_solo", "teammate_count", "repairing",
    "is_repairing", "repair_state", "fe_active",
}


def walk(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            key = str(key)
            path = f"{prefix}.{key}" if prefix else key
            normalized = key.lower().replace("-", "_")
            if any(term in normalized for term in STATE_KEY_TERMS):
                yield "state", path, key
            if "perk_latency" in normalized or ("perk" in normalized and "latency" in normalized):
                yield "latency_setting", path, key
            yield from walk(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, f"{prefix}[{index}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    files = sorted({p.resolve() for root in args.roots if root.exists()
                    for p in root.rglob("*.json") if p.is_file()})
    by_digest = {}
    invalid = []
    for path in files:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest in by_digest:
            by_digest[digest]["copies"].append(str(path))
            continue
        try:
            data = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            invalid.append({"path": str(path), "error": str(exc)})
            continue
        if not isinstance(data, dict) or not isinstance(data.get("frames"), list) or not data.get("check_id"):
            continue
        key_hits = sorted(set(walk(data)))
        state_hits = [(p, k) for kind, p, k in key_hits if kind == "state"]
        latency_hits = [(p, k) for kind, p, k in key_hits if kind == "latency_setting"]
        by_digest[digest] = {
            "sha256": digest, "path": str(path), "copies": [str(path)],
            "schema": "modern" if data.get("locked_zones") else "legacy",
            "check_id": data.get("check_id"),
            "chain_count": data.get("chain_count"),
            "state_key_paths": [{"path": p, "key": k} for p, k in state_hits],
            "latency_setting_paths": [{"path": p, "key": k} for p, k in latency_hits],
            "has_explicit_perk_or_context_state": bool(state_hits),
        }
    records = list(by_digest.values())
    key_counts = Counter()
    latency_key_counts = Counter()
    schema_counts = defaultdict(Counter)
    for record in records:
        for item in record["state_key_paths"]:
            key_counts[item["key"]] += 1
            schema_counts[record["schema"]][item["key"]] += 1
        for item in record["latency_setting_paths"]:
            latency_key_counts[item["key"]] += 1
    report = {
        "method": "Recursively scan every unique replay JSON object key for explicit Flawless Execution, perk-equipped/tier/active, solo/teammate, or repair-state metadata. Perk-latency settings are counted separately as solver configuration; chain_count is never treated as a perk/state label.",
        "limitation": "Absence of state keys means replay telemetry cannot establish whether Flawless Execution was equipped/active, whether the generator was being repaired alone, or whether a hit qualified for the perk.",
        "json_files_scanned": len(files), "unique_replay_jsons": len(records),
        "invalid_json": invalid,
        "records_with_explicit_perk_or_context_state": sum(r["has_explicit_perk_or_context_state"] for r in records),
        "matching_key_counts": dict(sorted(key_counts.items())),
        "records_with_perk_latency_setting": sum(bool(r["latency_setting_paths"]) for r in records),
        "perk_latency_setting_key_counts": dict(sorted(latency_key_counts.items())),
        "matching_keys_by_schema": {k: dict(v) for k, v in schema_counts.items()},
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
