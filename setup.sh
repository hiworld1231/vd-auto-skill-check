#!/usr/bin/env bash
set -euo pipefail
vd_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
/usr/bin/python -m venv "$vd_root/.venv"
"$vd_root/.venv/bin/python" -m pip install --disable-pip-version-check -r "$vd_root/requirements.txt"
"$vd_root/.venv/bin/python" "$vd_root/run.py" preflight
