#!/bin/bash
# Launch FreeWillyBot dashboard using project venv.
# Usage: ./scripts/run_dashboard.sh [--port 8080]

cd "$(dirname "$0")/.."
.venv/bin/python scripts/run_dashboard.py "$@"
