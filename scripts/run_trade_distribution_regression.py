"""
Final validation — Trade distribution analysis for regression strategy.

Runs backtest with best config, plots trades per hour/day/session/weekday.
Output: data/backtests_regression/trade_distribution.png + summary stats.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions_regression"

TOP_PCT = 0.25
VOL_PCT = 20
PRED_THRESHOLD = 0.00005


def _session_from_hour(hour_utc: int) -> str:
    if 8 <= hour_utc < 16:
        return "London"
    if 13 <= hour_utc < 21:
        return "NY"
    return "Other"


def run() -> int:
    """Run trade distribution with best config. Returns 0 on success, 1 on failure."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    test_pred_path = PREDICTIONS_DIR / "test_predictions.parquet"
    if not test_pred_path.exists():
        logging.info("Test predictions not found. Running predict_regression_test...")
        r = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "predict_regression_test.py")],
            cwd=str(PROJECT_ROOT),
        )
        if r.returncode != 0:
            return 1

    import numpy as np
    import pandas as pd
    from src.backtest_regression import _positions_from_pred, plot_trade_clustering

    df = pd.read_parquet(test_pred_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    ret = df["target_ret"].values.astype(float)
    pred = df["pred"].values.astype(float)
    vol = df["vol_6"].fillna(0).values.astype(float)

    positions = _positions_from_pred(pred, vol, TOP_PCT, VOL_PCT, PRED_THRESHOLD)
    changes = np.diff(positions, prepend=positions[0]) != 0
    trade_idx = np.where(changes)[0]

    if len(trade_idx) == 0:
        logging.warning("No trades; skipping distribution")
        return 0

    timestamps = df["timestamp"].iloc[trade_idx]
    trades_per_day = timestamps.dt.date.value_counts()
    hour_utc = timestamps.dt.hour
    session = hour_utc.apply(_session_from_hour)
    session_counts = session.value_counts()

    logging.info("Trades per day: mean=%.2f std=%.2f", trades_per_day.mean(), trades_per_day.std())
    logging.info("Session breakdown: %s", dict(session_counts))

    plot_trade_clustering(
        pred_path=test_pred_path,
        top_pct=TOP_PCT,
        vol_pct=VOL_PCT,
        pred_threshold=PRED_THRESHOLD,
        save_distribution=True,
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    sys.exit(run())
