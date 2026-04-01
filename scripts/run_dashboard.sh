#!/bin/bash
# Launch FreeWillyBot dashboard using project venv.
# Stops any existing dashboard on the same port (and stray run_dashboard.py), then starts a new one.
# Usage: ./scripts/run_dashboard.sh [--port 8080]

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="${ROOT}/.venv/bin/python"
SCRIPT="${ROOT}/scripts/run_dashboard.py"

PORT=5050
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  if [[ "${args[$i]}" == "--port" && $((i + 1)) -lt ${#args[@]} ]]; then
    PORT="${args[$((i + 1))]}"
    break
  fi
done

_stop_listeners_on_port() {
  local p="$1"
  if command -v lsof >/dev/null 2>&1; then
    # shellcheck disable=SC2046
    local pids
    pids=$(lsof -ti ":$p" 2>/dev/null) || true
    if [[ -n "$pids" ]]; then
      # shellcheck disable=SC2086
      kill $pids 2>/dev/null || true
    fi
  fi
}

_stop_listeners_on_port "$PORT"
sleep 0.4

# Any previous run_dashboard.py (e.g. different port in args, or stale process)
if command -v pkill >/dev/null 2>&1; then
  pkill -f "scripts/run_dashboard.py" 2>/dev/null || true
  sleep 0.4
fi

exec "$PY" "$SCRIPT" "$@"
