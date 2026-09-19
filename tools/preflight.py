#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from core.preflight import run_preflight

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
report = run_preflight(
    dry_run=False,
    require_lmb=bool(config.get("require_lmb", True)),
    require_capture_tools=False,
)
print("PRE-FLIGHT:", "OK" if report.ok else "FAILED")
for item in report.errors:
    print("ERROR:", item)
for item in report.warnings:
    print("WARN :", item)
raise SystemExit(0 if report.ok else 1)
