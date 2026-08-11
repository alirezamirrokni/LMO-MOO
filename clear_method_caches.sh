#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/mnt/data/LMO-MOO-runs}"
shift || true
METHODS=("$@")
find "$ROOT" -type d -name cache -print0 2>/dev/null | while IFS= read -r -d '' c; do
  if (( ${#METHODS[@]} > 0 )); then
    for method in "${METHODS[@]}"; do
      safe="${method//\//_}"; safe="${safe// /_}"
      rm -rf "$c/$safe"
    done
  else
    rm -rf "$c"; mkdir -p "$c"
  fi
done
