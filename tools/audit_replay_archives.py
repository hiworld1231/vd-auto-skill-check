#!/usr/bin/env python3
"""Inventory replay ZIPs and verify whether their payloads are restored.

Replay media is matched by filename and SHA-256 against an extraction root.
Every ZIP member is also read, which checks its CRC. ZIPs are never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ASSETS = {".json", ".mp4", ".mkv", ".png"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zip_root", type=Path)
    ap.add_argument("restored_root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    restored = {}
    for path in args.restored_root.rglob("*"):
        if path.is_file():
            restored.setdefault(path.name, set()).add(
                hashlib.sha256(path.read_bytes()).hexdigest())

    reports = []
    for path in sorted(p for p in args.zip_root.rglob("*")
                       if p.is_file() and (p.suffix.lower() == ".zip" or ".zip" in p.name)):
        row = {"zip": str(path), "members": 0, "replay_assets": 0,
               "crc_error": None, "assets": []}
        try:
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    row["members"] += 1
                    raw = archive.read(info)
                    if Path(info.filename).suffix.lower() not in ASSETS:
                        continue
                    row["replay_assets"] += 1
                    digest = hashlib.sha256(raw).hexdigest()
                    candidates = restored.get(Path(info.filename).name, set())
                    status = ("same_content" if digest in candidates else
                              "same_name_different_content" if candidates else "missing")
                    row["assets"].append({"member": info.filename, "sha256": digest,
                                          "size": len(raw), "status": status})
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            row["crc_error"] = str(exc)
        reports.append(row)

    summary = {"zip_count": len(reports),
               "bad_archives": sum(bool(r["crc_error"]) for r in reports),
               "replay_asset_entries": sum(r["replay_assets"] for r in reports),
               "missing_entries": sum(a["status"] == "missing" for r in reports for a in r["assets"]),
               "different_entries": sum(a["status"] == "same_name_different_content"
                                         for r in reports for a in r["assets"])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "archives": reports},
                                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
