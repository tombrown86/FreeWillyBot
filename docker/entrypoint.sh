#!/bin/sh
# FreeWillyBot — 24/7 live-tick loop (every 2 min by default).
# Override with: docker run ... freewillybot python -m scripts.run_daily_data_refresh

set -e
cd /app
INTERVAL="${LIVETICK_INTERVAL_SEC:-120}"

# If args given, run once and exit (e.g. data refresh, train)
if [ -n "$1" ]; then
  exec python -m "$@"
fi

echo "FreeWillyBot livetick loop: every ${INTERVAL}s (set LIVETICK_INTERVAL_SEC to change)"
while true; do
  python -m scripts.run_live_tick || true
  sleep "$INTERVAL"
done
