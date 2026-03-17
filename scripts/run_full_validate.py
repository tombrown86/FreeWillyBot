"""
Phase 17 — Full-stack validation on custom range.

Runs: data refresh (optional custom --start/--end), build, train, then
rolling-window backtest. Use when you want to validate from scratch.
"""

import logging
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run(
    months: int = 6,
    start: datetime | None = None,
    end: datetime | None = None,
) -> int:
    """
    Run full pipeline then validate.
    months: window sizes for run_validate (default 6)
    start/end: optional custom download range (UTC)
    Returns 0 on success, 1 on failure.
    """
    refresh_cmd = [sys.executable, "-m", "scripts.run_daily_data_refresh"]
    if start:
        refresh_cmd.extend(["--start", start.strftime("%Y-%m-%d")])
    if end:
        refresh_cmd.extend(["--end", end.strftime("%Y-%m-%d")])

    steps = [
        ("run_daily_data_refresh", refresh_cmd),
        ("run_daily_retrain", [sys.executable, "-m", "scripts.run_daily_retrain"]),
        ("run_validate", [sys.executable, "-m", "scripts.run_validate", "--months", str(months)]),
    ]

    for name, cmd in steps:
        logging.info("Running %s...", name)
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        if result.returncode != 0:
            logging.error("%s failed with code %d", name, result.returncode)
            return 1

    return 0


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="Full-stack validation")
    parser.add_argument("--months", type=int, default=6, help="Validation window months")
    parser.add_argument("--start", type=str, help="Download start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="Download end date (YYYY-MM-DD)")
    args = parser.parse_args()

    start_dt = None
    end_dt = None
    if args.start:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if args.end:
        end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    exit_code = run(months=args.months, start=start_dt, end=end_dt)
    sys.exit(exit_code)
