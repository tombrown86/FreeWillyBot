"""
Final validation — Parameter stability sweep for regression strategy.

Perturb around best config:
  top_pct: 0.2, 0.25, 0.3
  vol_pct: 15, 20, 25
  pred_threshold: 0.00004, 0.00005, 0.00006 (±20%)
Output: data/backtests_regression/stability_sweep.csv
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

TOP_PCT_OPTIONS = [0.2, 0.25, 0.3]
VOL_PCT_OPTIONS = [15, 20, 25]
PRED_THRESHOLD_OPTIONS = [0.00004, 0.00005, 0.00006]


def run() -> int:
    """Run stability sweep. Returns 0 on success, 1 on failure."""
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

    results = run_grid(
        pred_path=test_pred_path,
        top_pct_list=TOP_PCT_OPTIONS,
        pred_threshold_list=PRED_THRESHOLD_OPTIONS,
        vol_pct_list=VOL_PCT_OPTIONS,
        min_bars_between=0,
        cost_mult=1.0,
    )

    BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    df = pd.DataFrame(results)
    df.to_csv(BACKTESTS_DIR / "stability_sweep.csv", index=False)
    logging.info("Saved %s (%d rows)", BACKTESTS_DIR / "stability_sweep.csv", len(df))

    n_positive = (df["net_return"] > 0).sum()
    logging.info("Positive net_return: %d / %d", n_positive, len(df))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    sys.exit(run())
