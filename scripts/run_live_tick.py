"""
Phase 16 — Live tick script (every 5 min).

Runs all registered strategies, appending signals to predictions_live.csv
and (with --execute) trade decisions to trade_decisions.csv.
Each row is tagged with a strategy_id so multiple strategies can be compared.
"""

import csv
import importlib
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PREDICTIONS_LIVE_CSV, TRADE_DECISIONS_CSV

# Strategy registry — add new strategies here.
# Each entry: id (string label), module path, function name.
STRATEGIES = [
    {"id": "classifier_v1", "module": "src.live_signal", "fn": "run"},
    {"id": "regression_v1", "module": "src.live_signal_regression", "fn": "run"},
]


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _append_predictions_row(row: dict, run_at: str, strategy_id: str) -> None:
    """Append one signal row to predictions_live.csv."""
    csv_path = PROJECT_ROOT / PREDICTIONS_LIVE_CSV
    _ensure_dir(csv_path)
    write_row = {
        "strategy_id": strategy_id,
        "timestamp": row.get("timestamp", ""),
        "signal": row.get("signal", ""),
        "confidence": row.get("confidence", ""),
        "blocked": row.get("blocked", ""),
        "reason": row.get("reason", ""),
        "action": row.get("action", ""),
        "P_buy": row.get("P_buy", ""),
        "P_sell": row.get("P_sell", ""),
        "pred": row.get("pred", ""),
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
    strategy_id: str,
) -> None:
    """Append one trade decision row to trade_decisions.csv."""
    csv_path = PROJECT_ROOT / TRADE_DECISIONS_CSV
    _ensure_dir(csv_path)
    write_row = {
        "strategy_id": strategy_id,
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


def _run_strategy(strategy: dict, execute: bool, run_at: str) -> bool:
    """Run a single strategy, append its output. Returns True on success."""
    sid = strategy["id"]
    try:
        mod = importlib.import_module(strategy["module"])
        fn = getattr(mod, strategy["fn"])
        rows = fn(n_bars=1)
        if not rows:
            logging.warning("[%s] No signal produced", sid)
            return False

        for row in rows:
            _append_predictions_row(row, run_at, strategy_id=sid)
            logging.info(
                "[%s] signal=%s action=%s blocked=%s",
                sid,
                row.get("signal", "?"),
                row.get("action", "?"),
                row.get("blocked", "?"),
            )

            if execute:
                from src.execution import process_signal

                current_position = "flat"
                action_taken, broker_response = process_signal(row, current_position, dry_run=True)
                _append_trade_decision(
                    row.get("timestamp", ""),
                    row.get("signal", ""),
                    action_taken,
                    broker_response,
                    run_at,
                    strategy_id=sid,
                )
        return True
    except Exception as e:
        logging.exception("[%s] strategy failed: %s", sid, e)
        return False


def run(refresh: bool = False, execute: bool = False) -> int:
    """Run all strategies for the latest bar. Returns 0 if all succeed, 1 if any fail."""
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

        run_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        results = [_run_strategy(s, execute, run_at) for s in STRATEGIES]
        return 0 if all(results) else 1

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

    parser = argparse.ArgumentParser(description="Live tick: produce signal for all strategies")
    parser.add_argument("--refresh", action="store_true", help="Run data refresh first")
    parser.add_argument("--execute", action="store_true", help="Execute demo orders and log to trade_decisions.csv")
    args = parser.parse_args()

    sys.exit(run(refresh=args.refresh, execute=args.execute))
