#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[1/4] Creating repo-local virtual environment: $ROOT/.venv"
"$PYTHON_BIN" -m venv .venv

echo "[2/4] Upgrading pip"
.venv/bin/python -m pip install --upgrade pip

echo "[3/4] Installing Python dependencies"
.venv/bin/python -m pip install -r requirements.txt

echo "[4/4] Running preflight"
.venv/bin/python tools/preflight.py

echo
echo "Setup complete."
echo "Start with: python run.py"
