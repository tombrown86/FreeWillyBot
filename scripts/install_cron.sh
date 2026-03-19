#!/bin/bash
# Install FreeWillyBot cron jobs (same schedule as Mac launchd).
# Usage: ./scripts/install_cron.sh [--print]
#   --print   only print the cron lines, do not install
#   PROJECT_ROOT=/path/to/FreeWillyBot ./scripts/install_cron.sh   to override path

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/home/tom/dev/FreeWillyBot}"
PRINT_ONLY=false
for arg in "$@"; do
  if [ "$arg" = "--print" ]; then PRINT_ONLY=true; fi
done

PY="$PROJECT_ROOT/.venv/bin/python"
CRON_BLOCK="# FreeWillyBot — same as Mac launchd (livetick every 2 min, daily refresh 00:00, daily retrain 00:30)
PROJECT=$PROJECT_ROOT
PY=$PY

# Livetick: every 2 min
*/2 * * * * cd \$PROJECT && \$PY -m scripts.run_live_tick >> \$PROJECT/data/logs/livetick_stdout.log 2>> \$PROJECT/data/logs/livetick_stderr.log

# Data refresh: midnight daily
0 0 * * * cd \$PROJECT && \$PY -m scripts.run_daily_data_refresh --skip-if-recent 20 >> \$PROJECT/data/logs/data_refresh_stdout.log 2>> \$PROJECT/data/logs/data_refresh_stderr.log

# Retrain: 00:30 daily
30 0 * * * cd \$PROJECT && \$PY -m scripts.run_daily_retrain --skip-if-recent 20 >> \$PROJECT/data/logs/retrain_stdout.log 2>> \$PROJECT/data/logs/retrain_stderr.log
"

if [ "$PRINT_ONLY" = true ]; then
  echo "$CRON_BLOCK"
  exit 0
fi

echo "Project root: $PROJECT_ROOT"
if [ ! -x "$PY" ]; then
  echo "Error: $PY not found or not executable. Create venv: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
mkdir -p "$PROJECT_ROOT/data/logs"

# Append to crontab (merge with existing)
( crontab -l 2>/dev/null; echo "$CRON_BLOCK" ) | crontab -
echo "Crontab updated. Current entries:"
crontab -l
