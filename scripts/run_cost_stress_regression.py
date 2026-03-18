"""
Final validation — Cost stress test for regression strategy.

Runs backtest with cost_mult = 1.0, 1.5, 2.0.
Fixed config: top_pct=0.25, vol_pct=20, pred_threshold=0.00005.
Output: data/backtests_regression/cost_stress.csv
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions_regression"
BACKTESTS_DIR = PROJECT_ROOT / "data" / "backtests_regression"

TOP_PCT = 0.25
VOL_PCT = 20
PRED_THRESHOLD = 0.00005


def run() -> int:
    """Run cost stress with cost_mult 1, 1.5, 2. Returns 0 on success, 1 on failure."""
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

    from src.backtest_regression import run_grid

    results = []
    for cost_mult in [1.0, 1.5, 2.0]:
        grid = run_grid(
            pred_path=test_pred_path,
            top_pct_list=[TOP_PCT],
            pred_threshold_list=[PRED_THRESHOLD],
            vol_pct_list=[VOL_PCT],
            min_bars_between=0,
            cost_mult=cost_mult,
        )
        row = grid[0]
        row["cost_mult"] = cost_mult
        results.append(row)
        logging.info(
            "cost_mult=%.1f: net_return=%.4f n_trades=%d PF=%.2f",
            cost_mult,
            row["net_return"],
            row["n_trades"],
            row["profit_factor"],
        )

    BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    df = pd.DataFrame(results)
    df = df[["cost_mult", "net_return", "n_trades", "profit_factor", "max_dd"]]
    df.to_csv(BACKTESTS_DIR / "cost_stress.csv", index=False)
    logging.info("Saved %s", BACKTESTS_DIR / "cost_stress.csv")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    sys.exit(run())
