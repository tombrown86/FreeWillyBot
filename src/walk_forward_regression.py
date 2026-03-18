"""
Final validation — Rolling walk-forward for regression strategy.

12 × 1-month windows, 6 × 2-month windows.
Fixed config: top_pct=0.25, vol_pct=20, pred_threshold=0.00005.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest_regression import (
    _apply_min_bars_between,
    _positions_from_pred,
    _profit_factor,
    _run_single,
)
from src.config import FX_SPREAD_PIPS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions_regression"
BACKTESTS_DIR = PROJECT_ROOT / "data" / "backtests_regression"

FX_SPREAD = FX_SPREAD_PIPS * 0.0001
TOP_PCT = 0.25
VOL_PCT = 20
PRED_THRESHOLD = 0.00005
MIN_BARS_BETWEEN = 0


def run_backtest_on_arrays(
    ret: np.ndarray,
    pred: np.ndarray,
    vol: np.ndarray,
    cost_mult: float = 1.0,
    kill_switch_n: int = 0,
    kill_switch_pf: float = 0.9,
    dd_kill: float = 0.0,
    pause_bars: int = 0,
) -> dict:
    """Run backtest on given arrays. Percentiles computed within the arrays. Returns net_return, pf, max_dd, n_trades."""
    valid = np.isfinite(ret) & np.isfinite(pred) & np.isfinite(vol)
    ret = ret[valid]
    pred = pred[valid]
    vol = vol[valid]
    if len(ret) < 100:
        return {"net_return": 0.0, "profit_factor": 0.0, "max_dd": 0.0, "n_trades": 0}

    cost_per_leg = FX_SPREAD * cost_mult
    net_ret, n_trades, max_dd = _run_single(
        ret, pred, vol, TOP_PCT, VOL_PCT, PRED_THRESHOLD, MIN_BARS_BETWEEN, cost_per_leg,
        with_costs=True, kill_switch_n=kill_switch_n, kill_switch_pf=kill_switch_pf,
        dd_kill=dd_kill, pause_bars=pause_bars,
    )
    positions = _positions_from_pred(pred, vol, TOP_PCT, VOL_PCT, PRED_THRESHOLD)
    positions = _apply_min_bars_between(positions, MIN_BARS_BETWEEN)
    pf = _profit_factor(ret, positions)
    return {"net_return": float(net_ret), "profit_factor": float(pf), "max_dd": float(max_dd), "n_trades": int(n_trades)}


def run_walk_forward(
    pred_path: Path | None = None,
    kill_switch_n: int = 0,
    kill_switch_pf: float = 0.9,
    dd_kill: float = 0.0,
    pause_bars: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Run 12×1m and 6×2m rolling windows. Returns (results_1m, results_2m)."""
    path = pred_path or (PREDICTIONS_DIR / "test_predictions.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Test predictions not found: {path}. Run predict_regression_test first.")

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    results_1m = []
    results_2m = []

    # 12 × 1-month
    df["month"] = df["timestamp"].dt.strftime("%Y-%m")
    months = sorted(df["month"].dropna().unique())
    for i, m in enumerate(months[:12]):
        w = df[df["month"] == m]
        if len(w) < 100:
            continue
        ret = w["target_ret"].values.astype(float)
        pred = w["pred"].values.astype(float)
        vol = w["vol_6"].fillna(0).values.astype(float)
        r = run_backtest_on_arrays(ret, pred, vol, kill_switch_n=kill_switch_n, kill_switch_pf=kill_switch_pf,
                                    dd_kill=dd_kill, pause_bars=pause_bars)
        r["window_start"] = str(m)
        r["window_months"] = 1
        results_1m.append(r)
        logging.info("1m window %s: net=%.4f PF=%.2f trades=%d", m, r["net_return"], r["profit_factor"], r["n_trades"])

    # 6 × 2-month
    for i in range(0, min(12, len(months) - 1), 2):
        m1, m2 = months[i], months[i + 1]
        w = df[df["month"].isin([m1, m2])]
        if len(w) < 100:
            continue
        ret = w["target_ret"].values.astype(float)
        pred = w["pred"].values.astype(float)
        vol = w["vol_6"].fillna(0).values.astype(float)
        r = run_backtest_on_arrays(ret, pred, vol, kill_switch_n=kill_switch_n, kill_switch_pf=kill_switch_pf,
                                    dd_kill=dd_kill, pause_bars=pause_bars)
        r["window_start"] = str(m1)
        r["window_months"] = 2
        results_2m.append(r)
        logging.info("2m window %s-%s: net=%.4f PF=%.2f trades=%d", m1, m2, r["net_return"], r["profit_factor"], r["n_trades"])

    return results_1m, results_2m
