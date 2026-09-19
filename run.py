#!/usr/bin/env python3
"""Interactive launcher for Violence District CLEAN V5."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
VENV_PY = VENV_DIR / "bin" / "python"

# Prefer the repo-local virtualenv created by ./setup.sh.
if VENV_PY.exists() and Path(sys.executable).resolve() != VENV_PY.resolve():
    os.execv(str(VENV_PY), [str(VENV_PY)] + sys.argv)

PYTHON_BIN = str(VENV_PY) if VENV_PY.exists() else sys.executable


def _bot(*args: str) -> int:
    return subprocess.run([PYTHON_BIN, str(ROOT / "skillcheck_bot.py"), *args], cwd=ROOT).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="Violence District Skill Check — CLEAN V5")
    ap.add_argument("--gen-rush", action="store_true", help="run CLEAN V5 directly")
    ap.add_argument("--hud", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-require-lmb", action="store_true")
    args, unknown = ap.parse_known_args()

    if not VENV_PY.exists():
        print("[WARN] Repo-local .venv not found.")
        print("       Run: ./setup.sh")
        print("       (or: python -m venv .venv && .venv/bin/python -m pip install -r requirements.txt)")
        print()

    if args.gen_rush:
        cmd = ["--gen-rush"]
        if args.hud:
            cmd.append("--hud")
        if args.dry_run:
            cmd.append("--dry-run")
        if args.no_require_lmb:
            cmd.append("--no-require-lmb")
        return _bot(*(cmd + unknown))

    print("==================================================")
    print("  VIOLENCE DISTRICT — SKILL CHECK CLEAN V5")
    print("==================================================")
    print("1. START            — основной runtime (normal + perk)")
    print("2. START + HUD      — debug HUD")
    print("3. DRY-RUN          — без реального Space")
    print("4. PREFLIGHT        — проверить окружение")
    print("5. TESTS            — release test suite")
    print()
    while True:
        choice = input("Выбери режим [1-5]: ").strip()
        if choice in {"1", "2", "3", "4", "5"}:
            break
        print("Введи число от 1 до 5.")

    if choice == "1":
        return _bot("--gen-rush")
    if choice == "2":
        return _bot("--gen-rush", "--hud")
    if choice == "3":
        return _bot("--gen-rush", "--dry-run")
    if choice == "4":
        return subprocess.run([PYTHON_BIN, str(ROOT / "tools" / "preflight.py")], cwd=ROOT).returncode
    return subprocess.run([PYTHON_BIN, "-m", "pytest", "-q"], cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
