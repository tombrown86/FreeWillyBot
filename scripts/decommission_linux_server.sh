#!/usr/bin/env bash
# Decommission FreeWillyBot on the Linux host: stop services, remove cron, close demo positions.
# Run ON the Linux box (or pipe over SSH from your Mac on the LAN):
#   ssh tom@192.168.0.10 'bash -s' < scripts/decommission_linux_server.sh
#
# Optional env overrides:
#   PROJECT=/home/tom/dev/FreeWillyBot

set -euo pipefail

PROJECT="${PROJECT:-/home/tom/dev/FreeWillyBot}"
PY="${PROJECT}/.venv/bin/python"
TS="$(date -u +%Y%m%d_%H%M%S)"
ARCHIVE="${PROJECT}/data/migration_archive_linux_${TS}"

if [ ! -d "$PROJECT" ]; then
  echo "ERROR: PROJECT not found: $PROJECT" >&2
  exit 1
fi

cd "$PROJECT"

echo "=== FreeWillyBot decommission (${TS}) ==="

# ── 1. Stop running processes ────────────────────────────────────────────────
echo "--- Stopping processes ---"
pkill -f "scripts.run_live_tick" 2>/dev/null || true
pkill -f "scripts.run_daily_data_refresh" 2>/dev/null || true
pkill -f "scripts.run_daily_retrain" 2>/dev/null || true
pkill -f "scripts.run_dashboard" 2>/dev/null || true
pkill -f "run_dashboard.sh" 2>/dev/null || true
if command -v screen >/dev/null 2>&1; then
  screen -S dashboard -X quit 2>/dev/null || true
fi
sleep 2
pgrep -af "FreeWillyBot|run_live_tick|run_dashboard|run_daily_data" || echo "(no bot processes running)"

# ── 2. Remove cron jobs ────────────────────────────────────────────────────
echo "--- Removing crontab entries ---"
if crontab -l >/dev/null 2>&1; then
  crontab -l | grep -v "FreeWillyBot" | grep -v "run_live_tick" | grep -v "run_daily_data_refresh" | grep -v "run_daily_retrain" | grep -v "run_dashboard" | crontab - || true
  echo "Crontab after cleanup:"
  crontab -l 2>/dev/null || echo "(empty crontab)"
else
  echo "(no crontab)"
fi

# ── 3. Disable nginx dashboard site (optional) ───────────────────────────────
if command -v systemctl >/dev/null 2>&1 && [ -f /etc/nginx/sites-enabled/freewillybot ]; then
  echo "--- Disabling nginx freewillybot site ---"
  sudo rm -f /etc/nginx/sites-enabled/freewillybot 2>/dev/null || true
  sudo systemctl reload nginx 2>/dev/null || true
fi

# ── 4. Archive useful local data before wipe ─────────────────────────────────
echo "--- Archiving state to ${ARCHIVE} ---"
mkdir -p "${ARCHIVE}"/{execution,predictions,logs,config}
cp -a "${PROJECT}/.env" "${ARCHIVE}/config/.env" 2>/dev/null || true
cp -a data/logs/execution/*.json "${ARCHIVE}/execution/" 2>/dev/null || true
cp -a data/logs/execution/*.csv "${ARCHIVE}/execution/" 2>/dev/null || true
cp -a data/predictions/predictions_live*.csv "${ARCHIVE}/predictions/" 2>/dev/null || true
cp -a data/logs/livetick_*.log data/logs/data_refresh_*.log "${ARCHIVE}/logs/" 2>/dev/null || true
echo "Archive written under ${ARCHIVE}"

# ── 5. Close all demo cTrader positions ────────────────────────────────────
if [ -x "$PY" ]; then
  echo "--- Closing demo positions ---"
  "$PY" scripts/verify_demo_accounts.py 2>&1 || true
  "$PY" scripts/reset_paper_demo_state.py --close-all-accounts --also-strategy-state --signals 2>&1 || true
  for acct in 4247810 4243419 4247812 4247811; do
    echo "Net position check ${acct}:"
    "$PY" scripts/ctrader_net_position.py "$acct" 2>&1 || true
  done
else
  echo "WARN: ${PY} missing — skip broker close/reset"
fi

echo ""
echo "=== Decommission complete ==="
echo "Tar the archive for copy to Mac:"
echo "  tar -czvf ~/fwb_linux_archive_${TS}.tar.gz -C ${PROJECT}/data $(basename ${ARCHIVE})"
echo "Then from Mac:"
echo "  scp tom@192.168.0.10:~/fwb_linux_archive_${TS}.tar.gz ."
