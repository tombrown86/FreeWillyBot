"""
Run Batch 4 regression backtest grid.

1. Run predict_regression_test if test_predictions.parquet missing
2. Run backtest_regression grid (top_pct x pred_threshold x vol_pct)
3. Plot trade clustering; save results

Exact experiment: top_pct 0.5, 1; pred_threshold 0.00005, 0.00008, 0.0001; vol_pct 20, 30, 40
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions_regression"

EXACT_TOP_PCT = [0.25, 0.5, 1.0]
EXACT_PRED_THRESHOLD = [0.00005, 0.00008, 0.0001]
EXACT_VOL_PCT = [20, 30, 40]
EXACT_MIN_BARS = [0, 3, 6]


def run(use_exact: bool = True, min_bars_list: list[int] | None = None) -> int:
    """Orchestrate predict + backtest. Returns 0 on success, 1 on failure."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    test_pred_path = PREDICTIONS_DIR / "test_predictions.parquet"
    if not test_pred_path.exists():
        logging.info("Test predictions not found. Running predict_regression_test...")
        r = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "predict_regression_test.py")], cwd=str(PROJECT_ROOT))
        if r.returncode != 0:
            return 1

    from src.backtest_regression import run as backtest_run

    min_bars = min_bars_list or [0]
    best = {"net_return": float("-inf")}
    for mb in min_bars:
        if use_exact:
            res = backtest_run(
                pred_path=test_pred_path,
                top_pct_list=EXACT_TOP_PCT,
                pred_threshold_list=EXACT_PRED_THRESHOLD,
                vol_pct_list=EXACT_VOL_PCT,
                min_bars_between=mb,
                plot_clustering=(mb == 0),
            )
        else:
            res = backtest_run(pred_path=test_pred_path, min_bars_between=mb, plot_clustering=(mb == 0))
        if res["net_return"] > best["net_return"]:
            best = res

    logging.info(
        "Grid complete. Best: top_pct=%.2f pred_thresh=%.5f vol_pct=%d net_return=%.4f trades=%d",
        best["top_pct"],
        best["pred_threshold"],
        best["vol_pct"],
        best["net_return"],
        best["n_trades"],
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run full grid (all top_pct, pred_threshold, vol_pct)")
    parser.add_argument("--min-bars", type=int, nargs="+", default=[0], help="Min bars between trades (e.g. 0 3 6)")
    args = parser.parse_args()
    sys.exit(run(use_exact=not args.full, min_bars_list=args.min_bars))
