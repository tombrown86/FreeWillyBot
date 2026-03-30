#!/usr/bin/env bash
# Run from anywhere: creates .venv, installs deps, starts scripts/run_dashboard.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  if [[ -d "${ROOT}/.venv" ]]; then
    echo "Removing incomplete .venv ..."
    rm -rf "${ROOT}/.venv"
  fi
  echo "Creating virtualenv at ${ROOT}/.venv ..."
  if ! python3 -m venv "${ROOT}/.venv"; then
    echo ""
    echo "venv creation failed. On Debian/Ubuntu, install the matching venv package, e.g.:"
    echo "  sudo apt install python3.13-venv"
    echo "(Match the major.minor of: python3 --version)"
    exit 1
  fi
fi

if [[ -f "${ROOT}/requirements.txt" ]]; then
  echo "Installing requirements.txt ..."
  if ! "$PY" -m pip --version &>/dev/null; then
    echo "Bootstrapping pip in venv ..."
    "$PY" -m ensurepip --upgrade
  fi
  "$PY" -m pip install -r "${ROOT}/requirements.txt"
fi

exec "${ROOT}/scripts/run_dashboard.sh"
