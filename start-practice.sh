#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_file="$project_dir/simulator/skillcheck.html"
if (($#)); then
  printf 'Usage: %s\n' "$0" >&2
  exit 2
fi

if [[ ! -f "$practice_file" ]]; then
  printf 'Practice stand not found: %s\n' "$practice_file" >&2
  exit 1
fi
if ! command -v xdg-open >/dev/null 2>&1; then
  printf 'xdg-open is required to open the practice stand in a browser.\n' >&2
  exit 1
fi

practice_uri="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).as_uri())' "$practice_file")"
exec xdg-open "${practice_uri}?solver-test=1"
