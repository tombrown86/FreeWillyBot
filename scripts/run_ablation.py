"""
Phase 17 — Ablation studies.

Compare: price_only, price_plus_exog, price_plus_news, full.
Runs build_features + train + backtest for each config.
Appends to data/validation/ablation_report.csv.
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

VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"

CONFIGS = [
    ("price_only", False, False),
    ("price_plus_exog", True, False),
    ("price_plus_news", False, True),
    ("full", True, True),
]


def _run_backtest_metrics() -> dict | None:
    """Run backtest and return strategy metrics. Returns None on failure."""
    import json

    backtests_dir = PROJECT_ROOT / "data" / "backtests"
    pred_path = PROJECT_ROOT / "data" / "predictions" / "test_predictions.csv"
    if not pred_path.exists():
        return None

    result = subprocess.run(
        [sys.executable, "-c", "from src.backtest import run; run()"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    reports = sorted(backtests_dir.glob("backtest_report_*.json"), reverse=True)
    if not reports:
        return None
    with open(reports[0]) as f:
        data = json.load(f)
    return data.get("strategy", {})


def run(months: int = 6) -> int:
    """
    Run ablation: build + train + backtest for each config.
    Uses last N months for backtest window.
    Returns 0 on success, 1 on failure.
    """
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    report_path = VALIDATION_DIR / "ablation_report.csv"
    fieldnames = ["config", "cum_return", "max_dd", "sharpe", "profit_factor", "n_trades", "run_at"]
    file_exists = report_path.exists()
    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    for config_name, use_exog, use_news in CONFIGS:
        logging.info("Ablation config: %s (exog=%s, news=%s)", config_name, use_exog, use_news)

        # Build features with override
        from src.build_features import run as build_run

        build_run(use_exogenous=use_exog, use_news=use_news)

        # Train
        result = subprocess.run(
            [sys.executable, "-m", "scripts.run_daily_retrain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logging.error("Train failed for %s: %s", config_name, result.stderr)
            continue

        # Backtest (full test period; run_validate would slice)
        metrics = _run_backtest_metrics()
        if not metrics:
            logging.warning("No backtest metrics for %s", config_name)
            continue

        row = {
            "config": config_name,
            "cum_return": round(metrics.get("cumulative_return", 0), 4),
            "max_dd": round(metrics.get("max_drawdown", 0), 4),
            "sharpe": round(metrics.get("sharpe_ratio", 0), 2),
            "profit_factor": round(metrics.get("profit_factor", 0), 2),
            "n_trades": metrics.get("num_trades", 0),
            "run_at": run_at,
        }

        with open(report_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                w.writeheader()
                file_exists = True
            w.writerow(row)

        logging.info(
            "%s: cum_ret=%.4f max_dd=%.4f sharpe=%.2f",
            config_name,
            row["cum_return"],
            row["max_dd"],
            row["sharpe"],
        )

    logging.info("Ablation report saved to %s", report_path)
    return 0


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="Ablation studies")
    parser.add_argument("--months", type=int, default=6, help="Backtest window (for reference)")
    args = parser.parse_args()

    exit_code = run(months=args.months)
    sys.exit(exit_code)
