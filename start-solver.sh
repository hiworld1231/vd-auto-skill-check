#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
seconds_args=()
args=()
while (($#)); do
  case "$1" in
    --seconds)
      if (($# < 2)); then
        printf '%s\n' '--seconds requires a value' >&2
        exit 2
      fi
      seconds_args=(--seconds "$2")
      shift 2
      ;;
    --seconds=*)
      seconds_args=(--seconds "${1#*=}")
      shift
      ;;
    --recording|--recording=*)
      args+=("$1")
      shift
      ;;
    --no-recording)
      printf '%s\n' 'Recording is required in solver mode' >&2
      exit 2
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done
exec python "$project_dir/run.py" run "${seconds_args[@]}" "${args[@]}"
