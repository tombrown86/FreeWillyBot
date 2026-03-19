#!/bin/bash
# Stop and unload FreeWillyBot launchd jobs on the Mac.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
LAUNCHD_DIR="$SCRIPT_DIR/launchd"

for template in "$LAUNCHD_DIR"/*.plist.template; do
  [ -f "$template" ] || continue
  basename=$(basename "$template" .plist.template)
  plist="$LAUNCH_AGENTS/$basename.plist"
  if [ -f "$plist" ]; then
    launchctl unload "$plist" 2>/dev/null || true
    echo "Unloaded $basename"
  fi
done

echo "Done. FreeWillyBot launchd jobs are stopped."
