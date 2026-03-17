"""
Phase 16 — Live tick script (every 5 min).

Produce one signal for the latest bar. Append to predictions_live.csv.
With --execute: call execution and append to trade_decisions.csv.
"""

import csv
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PREDICTIONS_LIVE_CSV, TRADE_DECISIONS_CSV


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _append_predictions_row(row: dict, run_at: str) -> None:
    """Append one row to predictions_live.csv. Create file with header if missing."""
    csv_path = PROJECT_ROOT / PREDICTIONS_LIVE_CSV
    _ensure_dir(csv_path)
    write_row = {
        "timestamp": row.get("timestamp", ""),
        "signal": row.get("signal", ""),
        "confidence": row.get("confidence", ""),
        "blocked": row.get("blocked", ""),
        "reason": row.get("reason", ""),
        "action": row.get("action", ""),
        "P_buy": row.get("P_buy", ""),
        "P_sell": row.get("P_sell", ""),
        "run_at": run_at,
    }
    fieldnames = list(write_row.keys())
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(write_row)


def _append_trade_decision(
    timestamp: str,
    signal: str,
    action_taken: str,
    broker_response: str,
    run_at: str,
) -> None:
    """Append one row to trade_decisions.csv."""
    csv_path = PROJECT_ROOT / TRADE_DECISIONS_CSV
    _ensure_dir(csv_path)
    write_row = {
        "timestamp": timestamp,
        "signal": signal,
        "action_taken": action_taken,
        "broker_response": broker_response,
        "run_at": run_at,
    }
    fieldnames = list(write_row.keys())
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(write_row)


def run(refresh: bool = False, execute: bool = False) -> int:
    """
    Run live tick: get latest signal, append to CSVs, optionally execute.
    Returns 0 on success, 1 on failure.
    """
    try:
        if refresh:
            logging.info("Running data refresh first...")
            result = subprocess.run(
                [sys.executable, "-m", "scripts.run_daily_data_refresh"],
                cwd=PROJECT_ROOT,
                check=False,
            )
            if result.returncode != 0:
                logging.error("Data refresh failed with code %d", result.returncode)
                return 1

        from src.live_signal import run as run_live_signal

        rows = run_live_signal(n_bars=1)
        if not rows:
            logging.error("No signal produced")
            return 1

        row = rows[0]
        run_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        _append_predictions_row(row, run_at)

        if execute:
            from src.execution import process_signal

            current_position = "flat"
            action_taken, broker_response = process_signal(
                row, current_position, dry_run=True
            )
            _append_trade_decision(
                row.get("timestamp", ""),
                row.get("signal", ""),
                action_taken,
                broker_response,
                run_at,
            )

        return 0
    except Exception as e:
        logging.exception("Live tick failed: %s", e)
        return 1


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="Live tick: produce signal for latest bar")
    parser.add_argument("--refresh", action="store_true", help="Run data refresh first")
    parser.add_argument("--execute", action="store_true", help="Execute demo orders and log to trade_decisions.csv")
    args = parser.parse_args()

    exit_code = run(refresh=args.refresh, execute=args.execute)
    sys.exit(exit_code)
