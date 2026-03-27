#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-cpu}"

python -m pip install --upgrade pip

if [[ "$MODE" == "cpu" ]]; then
  python -m pip install -r requirements.txt
elif [[ "$MODE" == "cu124" ]]; then
  python -m pip install -r requirements-cu124.txt
else
  echo "Usage: bash install_deps.sh [cpu|cu124]"
  exit 1
fi

echo "[done] dependency install mode: $MODE"
