#!/usr/bin/env bash
# Run from anywhere: creates .venv, installs deps, starts scripts/run_dashboard.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv/bin/python"
PIP="${ROOT}/.venv/bin/pip"

if [[ ! -x "$PY" ]]; then
  echo "Creating virtualenv at ${ROOT}/.venv ..."
  python3 -m venv "${ROOT}/.venv"
fi

if [[ -f "${ROOT}/requirements.txt" ]]; then
  echo "Installing requirements.txt ..."
  "$PIP" install -r "${ROOT}/requirements.txt"
fi

exec "${ROOT}/scripts/run_dashboard.sh"
