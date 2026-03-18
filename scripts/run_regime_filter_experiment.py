"""
Final validation — Regime filter quick experiment.

Tests: trade only when vol_6 >= median AND abs(ret_12) >= threshold.
Grid over ret_12_threshold values.
Also re-runs per-month walk-forward for each regime config.
Output: data/backtests_regression/regime_filter_experiment.csv
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions_regression"
BACKTESTS_DIR = PROJECT_ROOT / "data" / "backtests_regression"

TOP_PCT = 0.25
VOL_PCT = 20
PRED_THRESHOLD = 0.00005
RET12_THRESHOLDS = [0.0, 0.0001, 0.0002, 0.0003]


def _run_walk_forward_with_mask(df: pd.DataFrame, regime_mask_fn) -> list[dict]:
    """Re-run 1-month walk-forward applying a per-window regime mask."""
    from src.backtest_regression import _apply_min_bars_between, _positions_from_pred, _profit_factor, _run_single
    from src.config import FX_SPREAD_PIPS

    cost_per_leg = FX_SPREAD_PIPS * 0.0001
    months = sorted(df["month"].dropna().unique())
    results = []
    for m in months[:12]:
        w = df[df["month"] == m]
        if len(w) < 100:
            continue
        ret = w["target_ret"].values.astype(float)
        pred = w["pred"].values.astype(float)
        vol = w["vol_6"].fillna(0).values.astype(float)
        ret_12 = w["ret_12"].fillna(0).values.astype(float)
        vol_change = w["vol_change"].fillna(0).values.astype(float)

        rm = regime_mask_fn(vol, ret_12, vol_change)
        valid = np.isfinite(ret) & np.isfinite(pred)
        net_ret, n_trades, max_dd = _run_single(
            ret[valid], pred[valid], vol[valid], TOP_PCT, VOL_PCT, PRED_THRESHOLD, 0, cost_per_leg,
            with_costs=True, regime_mask=rm[valid],
        )
        positions = _positions_from_pred(pred[valid], vol[valid], TOP_PCT, VOL_PCT, PRED_THRESHOLD, rm[valid])
        pf = _profit_factor(ret[valid], positions)
        results.append({"month": m, "net_return": float(net_ret), "profit_factor": float(pf), "max_dd": float(max_dd), "n_trades": int(n_trades)})
    return results


def run(kill_switch_n: int = 0) -> int:
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

    df = pd.read_parquet(test_pred_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["month"] = df["timestamp"].dt.strftime("%Y-%m")

    if "ret_12" not in df.columns or "vol_change" not in df.columns:
        logging.error("ret_12 / vol_change missing from test_predictions.parquet — re-run predict_regression_test")
        return 1

    from src.backtest_regression import _regime_mask_simple, run_grid

    experiment_rows = []

    for ret12_thresh in RET12_THRESHOLDS:
        vol = df["vol_6"].fillna(0).values.astype(float)
        ret_12 = df["ret_12"].fillna(0).values.astype(float)
        vol_change = df["vol_change"].fillna(0).values.astype(float)

        rm = _regime_mask_simple(vol, ret_12, vol_change, vol_above_median=True, ret_12_threshold=ret12_thresh)
        pct_active = rm.mean() * 100

        grid = run_grid(
            pred_path=test_pred_path,
            top_pct_list=[TOP_PCT],
            pred_threshold_list=[PRED_THRESHOLD],
            vol_pct_list=[VOL_PCT],
            min_bars_between=0,
            cost_mult=1.0,
            regime_mask=rm,
            kill_switch_n=kill_switch_n,
        )
        row = grid[0]

        # Monthly walk-forward
        wf = _run_walk_forward_with_mask(
            df,
            lambda v, r12, vc, t=ret12_thresh: _regime_mask_simple(v, r12, vc, vol_above_median=True, ret_12_threshold=t),
        )
        n_positive = sum(1 for w in wf if w["net_return"] > 0)

        logging.info(
            "ret12_thresh=%.4f: pct_active=%.1f%% net=%.4f PF=%.2f trades=%d months_positive=%d/12",
            ret12_thresh, pct_active, row["net_return"], row["profit_factor"], row["n_trades"], n_positive,
        )

        experiment_rows.append({
            "ret12_threshold": ret12_thresh,
            "pct_bars_active": float(pct_active),
            "net_return": row["net_return"],
            "profit_factor": row["profit_factor"],
            "n_trades": row["n_trades"],
            "max_dd": row["max_dd"],
            "months_positive": n_positive,
        })

    BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(experiment_rows)
    out.to_csv(BACKTESTS_DIR / "regime_filter_experiment.csv", index=False)
    logging.info("Saved %s", BACKTESTS_DIR / "regime_filter_experiment.csv")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kill-switch", type=int, default=0, metavar="N",
                        help="Pause trading when last N trades PF < 0.9 (0 = disabled)")
    args = parser.parse_args()
    sys.exit(run(kill_switch_n=args.kill_switch))
