"""
Model ablation: full, no_forecasters, price_only, chronos_only, momentum.

Runs backtest with each signal source on frozen test (top 5%, vol 30%).
Output: data/validation/model_ablation_report.csv
"""

import csv
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
FROZEN_DIR = PROJECT_ROOT / "data" / "frozen_test"


def _load_frozen_test() -> pd.DataFrame:
    import json

    manifest_path = FROZEN_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run freeze_test_set.py first")
    with open(manifest_path) as f:
        manifest = json.load(f)
    df = pd.read_parquet(FROZEN_DIR / manifest["file"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _run_backtest_for_ablation(
    pred_path: str | None = None,
    pred_df: pd.DataFrame | None = None,
    positions_override: np.ndarray | None = None,
) -> dict | None:
    """Run backtest, return strategy metrics."""
    from src.backtest import run

    try:
        result = run(
            use_frozen=True,
            top_pct=5,
            pred_path=pred_path,
            pred_df=pred_df,
            positions_override=positions_override,
            return_only=True,
        )
        if result is None:
            return None
        metrics, _, _ = result
        return metrics
    except Exception as e:
        logging.warning("Backtest failed: %s", e)
        return None


def run() -> int:
    """Run model ablations. Returns 0 on success."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    report_path = VALIDATION_DIR / "model_ablation_report.csv"
    fieldnames = ["config", "cum_return", "max_dd", "sharpe", "profit_factor", "n_trades"]
    rows = []

    # 1. Full (default predictions)
    logging.info("Ablation: full (meta + forecasters + exogenous)")
    m = _run_backtest_for_ablation()
    if m:
        rows.append({
            "config": "full",
            "cum_return": round(m["cumulative_return"], 4),
            "max_dd": round(m["max_drawdown"], 4),
            "sharpe": round(m["sharpe_ratio"], 2),
            "profit_factor": round(m["profit_factor"], 2),
            "n_trades": m["num_trades"],
        })

    # 2. No forecasters (retrain first)
    logging.info("Ablation: no_forecasters")
    r = subprocess.run(
        [sys.executable, "-m", "src.train_meta_model", "--no-forecasters", "--suffix", "_no_forecasters"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        m = _run_backtest_for_ablation(pred_path="test_predictions_no_forecasters.csv")
        if m:
            rows.append({
                "config": "no_forecasters",
                "cum_return": round(m["cumulative_return"], 4),
                "max_dd": round(m["max_drawdown"], 4),
                "sharpe": round(m["sharpe_ratio"], 2),
                "profit_factor": round(m["profit_factor"], 2),
                "n_trades": m["num_trades"],
            })
    else:
        logging.warning("Train no_forecasters failed: %s", r.stderr[:200])

    # 3. Price only (retrain first)
    logging.info("Ablation: price_only")
    r = subprocess.run(
        [sys.executable, "-m", "src.train_meta_model", "--price-only", "--suffix", "_price_only"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        m = _run_backtest_for_ablation(pred_path="test_predictions_price_only.csv")
        if m:
            rows.append({
                "config": "price_only",
                "cum_return": round(m["cumulative_return"], 4),
                "max_dd": round(m["max_drawdown"], 4),
                "sharpe": round(m["sharpe_ratio"], 2),
                "profit_factor": round(m["profit_factor"], 2),
                "n_trades": m["num_trades"],
            })
    else:
        logging.warning("Train price_only failed: %s", r.stderr[:200])

    # 4. Chronos only (no retrain; use forecaster signal)
    logging.info("Ablation: chronos_only")
    frozen = _load_frozen_test()
    pred_full = pd.read_csv(PREDICTIONS_DIR / "test_predictions.csv")
    pred_full["timestamp"] = pd.to_datetime(pred_full["timestamp"], utc=True)
    fc_path = FEATURES_DIR / "forecaster_predictions.csv"
    if fc_path.exists():
        fc = pd.read_csv(fc_path)
        fc["timestamp"] = pd.to_datetime(fc["timestamp"], utc=True)
        pred_full = pred_full.merge(fc, on="timestamp", how="left")
    pred_full["chronos_pred_return"] = pred_full.get("chronos_pred_return", pd.Series(0.0)).fillna(0)
    pred = pred_full.merge(
        frozen[["timestamp", "return_5", "volatility_20", "hour", "weekday"]],
        on="timestamp",
        how="inner",
    )
    chronos = pred["chronos_pred_return"].values.astype(float)
    pos = np.where(chronos > 0, 1, np.where(chronos < 0, -1, 0))
    m = _run_backtest_for_ablation(pred_df=pred, positions_override=pos)
    if m:
        rows.append({
            "config": "chronos_only",
            "cum_return": round(m["cumulative_return"], 4),
            "max_dd": round(m["max_drawdown"], 4),
            "sharpe": round(m["sharpe_ratio"], 2),
            "profit_factor": round(m["profit_factor"], 2),
            "n_trades": m["num_trades"],
        })

    # 5. Momentum (return_5 signal with same filters)
    logging.info("Ablation: momentum")
    pred_full = pd.read_csv(PREDICTIONS_DIR / "test_predictions.csv")
    pred_full["timestamp"] = pd.to_datetime(pred_full["timestamp"], utc=True)
    pred = pred_full.merge(
        frozen[["timestamp", "return_5", "volatility_20", "hour", "weekday"]],
        on="timestamp",
        how="inner",
    )
    return_5 = pred["return_5"].fillna(0).values.astype(float)
    pos = np.where(return_5 > 0, 1, np.where(return_5 < 0, -1, 0))
    m = _run_backtest_for_ablation(pred_df=pred, positions_override=pos)
    if m:
        rows.append({
            "config": "momentum",
            "cum_return": round(m["cumulative_return"], 4),
            "max_dd": round(m["max_drawdown"], 4),
            "sharpe": round(m["sharpe_ratio"], 2),
            "profit_factor": round(m["profit_factor"], 2),
            "n_trades": m["num_trades"],
        })

    with open(report_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    logging.info("Model ablation report saved to %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(run())
