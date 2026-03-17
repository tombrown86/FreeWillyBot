#!/bin/bash
# Phase 16 — Install launchd jobs for FreeWillyBot automation.
# Copies plists to ~/Library/LaunchAgents/, substitutes project path, loads with launchctl.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
LAUNCHD_DIR="$SCRIPT_DIR/launchd"

echo "Project root: $PROJECT_ROOT"
echo "LaunchAgents: $LAUNCH_AGENTS"
mkdir -p "$LAUNCH_AGENTS"

for template in "$LAUNCHD_DIR"/*.plist.template; do
  basename=$(basename "$template" .plist.template)
  plist="$LAUNCH_AGENTS/$basename.plist"
  echo "Installing $basename..."
  sed "s|@PROJECT_ROOT@|$PROJECT_ROOT|g" "$template" > "$plist"
  launchctl unload "$plist" 2>/dev/null || true
  launchctl load "$plist"
  echo "  Loaded $plist"
done

echo "Done. Jobs: com.freewillybot.livetick (every 5 min), com.freewillybot.data_refresh (00:00), com.freewillybot.retrain (00:30)"
