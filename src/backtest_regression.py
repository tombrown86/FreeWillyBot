"""
Batch 4 — Backtest regression predictions with extreme selection and volatility filter.

Symmetric extremes: top X% long, bottom X% short.
Vol filter: trade only when volatility in top Y%.
Pred threshold: |pred| > threshold.
Min bars between trades: structural cooldown.
Grid: top_pct x pred_threshold x vol_pct.
Reports gross vs net PnL.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FX_SPREAD_PIPS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions_regression"
BACKTESTS_DIR = PROJECT_ROOT / "data" / "backtests_regression"

FX_SPREAD = FX_SPREAD_PIPS * 0.0001

TOP_PCT_OPTIONS = [0.5, 0.75, 1, 2, 3, 5]
PRED_THRESHOLD_OPTIONS = [0.0, 0.00005, 0.00008, 0.0001]
VOL_PCT_OPTIONS = [20, 30, 40]
MIN_BARS_BETWEEN_OPTIONS = [0, 3, 6]
VOL_COL = "vol_6"


def _positions_from_pred(
    pred: np.ndarray,
    vol: np.ndarray,
    top_pct: float,
    vol_pct: int,
    pred_threshold: float = 0.0,
    regime_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Long when pred in top top_pct%, short when pred in bottom top_pct%. Trade only when vol in top vol_pct% and |pred| > threshold."""
    th_long = np.percentile(pred, 100 - top_pct)
    th_short = np.percentile(pred, top_pct)
    th_vol = np.percentile(vol, 100 - vol_pct)

    pos = np.where(pred >= th_long, 1, np.where(pred <= th_short, -1, 0))
    vol_ok = vol >= th_vol
    pred_ok = np.abs(pred) > pred_threshold
    pos[~vol_ok] = 0
    pos[~pred_ok] = 0
    if regime_mask is not None:
        pos[~regime_mask] = 0
    return pos.astype(int)


def _regime_mask_simple(
    vol: np.ndarray,
    ret_12: np.ndarray,
    vol_change: np.ndarray,
    vol_above_median: bool = True,
    ret_12_threshold: float = 0.0,
    vol_rising: bool | None = None,
) -> np.ndarray:
    """Rule-based regime filter. Returns bool array: True = trade allowed."""
    mask = np.ones(len(vol), dtype=bool)
    if vol_above_median:
        mask &= vol >= np.nanmedian(vol)
    if ret_12_threshold > 0:
        mask &= np.abs(ret_12) >= ret_12_threshold
    if vol_rising is True:
        mask &= vol_change > 0
    elif vol_rising is False:
        mask &= vol_change < 0
    return mask


def _apply_min_bars_between(positions: np.ndarray, min_bars: int) -> np.ndarray:
    """Block position changes within min_bars of the previous change."""
    if min_bars <= 0:
        return positions
    n = len(positions)
    out = positions.copy()
    prev_pos = 0
    last_change = -min_bars - 1
    for i in range(n):
        p = positions[i]
        if p != prev_pos:
            if i - last_change < min_bars:
                out[i] = prev_pos
            else:
                last_change = i
                prev_pos = p
                out[i] = p
        else:
            out[i] = p
    return out


def _run_single(
    ret: np.ndarray,
    pred: np.ndarray,
    vol: np.ndarray,
    top_pct: float,
    vol_pct: int,
    pred_threshold: float,
    min_bars_between: int,
    cost_per_leg: float,
    with_costs: bool,
    regime_mask: np.ndarray | None = None,
    kill_switch_n: int = 0,
    kill_switch_pf: float = 0.9,
    dd_kill: float = 0.0,
    pause_bars: int = 0,
) -> tuple[float, int, float]:
    """Run backtest. Returns (cum_return, n_trades, max_dd).

    Kill mechanisms (combined with OR logic):
      kill_switch_n / kill_switch_pf : pause when rolling PF of last N trades < threshold
      dd_kill                         : pause when current drawdown from peak > threshold
      pause_bars                      : after triggering either kill, pause for N bars then resume
                                        (0 = permanent pause for the window)
    """
    positions = _positions_from_pred(pred, vol, top_pct, vol_pct, pred_threshold, regime_mask)
    positions = _apply_min_bars_between(positions, min_bars_between)

    n = len(positions)
    equity = np.ones(n + 1)
    prev_pos = 0
    n_trades = 0
    paused = False
    pause_remaining = 0

    # Kill switch: track per-trade returns
    trade_rets: list[float] = []
    trade_start_equity = 1.0
    peak_equity = 1.0

    for i in range(n):
        # Resume after timed pause
        if paused and pause_bars > 0:
            pause_remaining -= 1
            if pause_remaining <= 0:
                paused = False

        p = positions[i] if not paused else 0

        if p != prev_pos and not paused:
            # Check trade-based kill switch
            if kill_switch_n > 0 and n_trades >= kill_switch_n:
                window = trade_rets[-kill_switch_n:]
                gains = sum(r for r in window if r > 0)
                losses = abs(sum(r for r in window if r < 0))
                rolling_pf = (gains / losses) if losses > 0 else float("inf")
                if rolling_pf < kill_switch_pf:
                    paused = True
                    pause_remaining = pause_bars
                    p = 0

        if p != prev_pos and not paused:
            n_trades += 1
            if with_costs:
                legs = 1 if (prev_pos == 0 or p == 0) else 2
                equity[i + 1] = equity[i] * (1 - legs * cost_per_leg)
            else:
                equity[i + 1] = equity[i]
            # Record completed trade return when closing
            if prev_pos != 0 and p == 0:
                trade_rets.append(equity[i + 1] / trade_start_equity - 1)
            if p != 0:
                trade_start_equity = equity[i + 1]
        else:
            equity[i + 1] = equity[i]

        if p != 0:
            equity[i + 1] *= 1 + p * ret[i]
        prev_pos = p

        # Update peak and check DD-based kill switch (checked after each bar)
        peak_equity = max(peak_equity, equity[i + 1])
        if not paused and dd_kill > 0:
            current_dd = (peak_equity - equity[i + 1]) / peak_equity if peak_equity > 0 else 0.0
            if current_dd >= dd_kill:
                # Close any open position next bar
                paused = True
                pause_remaining = pause_bars

    cum_ret = equity[-1] - 1.0
    eq = equity[1:]
    peak = np.maximum.accumulate(eq)
    drawdown = (peak - eq) / np.where(peak > 0, peak, 1)
    max_dd = float(np.max(drawdown))
    return cum_ret, n_trades, max_dd


def _profit_factor(ret: np.ndarray, positions: np.ndarray) -> float:
    """Profit factor from period returns when in position."""
    period_ret = np.where(positions != 0, positions * ret, 0)
    gains = period_ret[period_ret > 0].sum()
    losses = np.abs(period_ret[period_ret < 0].sum())
    return float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)


def run_grid(
    pred_path: Path | None = None,
    top_pct_list: list[float] | None = None,
    pred_threshold_list: list[float] | None = None,
    vol_pct_list: list[int] | None = None,
    min_bars_between: int = 0,
    cost_mult: float = 1.0,
    regime_mask: np.ndarray | None = None,
    kill_switch_n: int = 0,
    kill_switch_pf: float = 0.9,
    dd_kill: float = 0.0,
    pause_bars: int = 0,
) -> list[dict]:
    """Run grid over top_pct x pred_threshold x vol_pct. Returns list of result dicts."""
    path = pred_path or (PREDICTIONS_DIR / "test_predictions.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Test predictions not found: {path}. Run predict_regression_test first.")

    df = pd.read_parquet(path)
    ret = df["target_ret"].values.astype(float)
    pred = df["pred"].values.astype(float)

    if VOL_COL not in df.columns:
        raise ValueError(f"Volatility column {VOL_COL} not in {path}")
    vol = df[VOL_COL].fillna(0).values.astype(float)

    valid = np.isfinite(ret) & np.isfinite(pred) & np.isfinite(vol)
    ret = ret[valid]
    pred = pred[valid]
    vol = vol[valid]
    rm = regime_mask[valid] if regime_mask is not None else None

    top_list = top_pct_list or TOP_PCT_OPTIONS
    thresh_list = pred_threshold_list or PRED_THRESHOLD_OPTIONS
    vol_list = vol_pct_list or VOL_PCT_OPTIONS
    cost_per_leg = FX_SPREAD * cost_mult

    results = []
    for top_pct in top_list:
        for pred_threshold in thresh_list:
            for vol_pct in vol_list:
                gross_ret, n_trades, max_dd = _run_single(
                    ret, pred, vol, top_pct, vol_pct, pred_threshold, min_bars_between, cost_per_leg,
                    with_costs=False, regime_mask=rm, kill_switch_n=kill_switch_n, kill_switch_pf=kill_switch_pf,
                    dd_kill=dd_kill, pause_bars=pause_bars,
                )
                net_ret, _, _ = _run_single(
                    ret, pred, vol, top_pct, vol_pct, pred_threshold, min_bars_between, cost_per_leg,
                    with_costs=True, regime_mask=rm, kill_switch_n=kill_switch_n, kill_switch_pf=kill_switch_pf,
                    dd_kill=dd_kill, pause_bars=pause_bars,
                )

                positions = _positions_from_pred(pred, vol, top_pct, vol_pct, pred_threshold, rm)
                positions = _apply_min_bars_between(positions, min_bars_between)
                pf = _profit_factor(ret, positions)

                row = {
                    "top_pct": top_pct,
                    "pred_threshold": pred_threshold,
                    "vol_pct": vol_pct,
                    "min_bars_between": min_bars_between,
                    "gross_return": float(gross_ret),
                    "net_return": float(net_ret),
                    "n_trades": int(n_trades),
                    "max_dd": float(max_dd),
                    "profit_factor": float(pf),
                }
                results.append(row)
                logging.info(
                    "top=%.2f%% thresh=%.5f vol=%d%%: gross=%.4f net=%.4f trades=%d max_dd=%.4f PF=%.2f",
                    top_pct,
                    pred_threshold,
                    vol_pct,
                    gross_ret,
                    net_ret,
                    n_trades,
                    max_dd,
                    pf,
                )

    return results


def _session_from_hour(hour_utc: int) -> str:
    """Map UTC hour to session: London 8-16, NY 13-21, other."""
    if 8 <= hour_utc < 16:
        return "London"
    if 13 <= hour_utc < 21:
        return "NY"
    return "Other"


def plot_trade_clustering(
    pred_path: Path | None = None,
    top_pct: float = 1.0,
    vol_pct: int = 30,
    pred_threshold: float = 0.00005,
    save_distribution: bool = False,
) -> None:
    """Plot trades per hour, per day, per session (London/NY/Other), per weekday. Optionally save enhanced plot to trade_distribution.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not installed; skipping trade clustering plot")
        return

    path = pred_path or (PREDICTIONS_DIR / "test_predictions.parquet")
    if not path.exists():
        return

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    ret = df["target_ret"].values.astype(float)
    pred = df["pred"].values.astype(float)
    vol = df[VOL_COL].fillna(0).values.astype(float)

    positions = _positions_from_pred(pred, vol, top_pct, vol_pct, pred_threshold)
    changes = np.diff(positions, prepend=positions[0]) != 0
    trade_idx = np.where(changes)[0]

    if len(trade_idx) == 0:
        return

    timestamps = df["timestamp"].iloc[trade_idx]
    trades_per_hour = timestamps.dt.floor("h").value_counts().sort_index()
    trades_per_day = timestamps.dt.date.value_counts().sort_index()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].bar(range(len(trades_per_hour)), trades_per_hour.values)
    axes[0].set_title("Trades per hour")
    axes[0].set_xlabel("Hour index")
    axes[1].bar(range(len(trades_per_day)), trades_per_day.values)
    axes[1].set_title("Trades per day")
    axes[1].set_xlabel("Day index")
    fig.suptitle(f"Trade clustering (top={top_pct}% vol={vol_pct}% thresh={pred_threshold})")
    fig.tight_layout()
    BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(BACKTESTS_DIR / "trade_clustering.png", dpi=100)
    plt.close(fig)
    logging.info("Saved trade clustering plot to %s", BACKTESTS_DIR / "trade_clustering.png")

    if save_distribution:
        _plot_trade_distribution(timestamps, top_pct, vol_pct, pred_threshold)


def _plot_trade_distribution(timestamps: pd.Series, top_pct: float, vol_pct: int, pred_threshold: float) -> None:
    """Multi-panel: trades per hour, per day, per session, per weekday. Saves to trade_distribution.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    hour_utc = timestamps.dt.hour
    session = hour_utc.apply(_session_from_hour)
    trades_per_session = session.value_counts().reindex(["London", "NY", "Other"], fill_value=0)
    trades_per_weekday = timestamps.dt.dayofweek.value_counts().sort_index()
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    trades_per_weekday = trades_per_weekday.reindex(range(7), fill_value=0)

    trades_per_hour = timestamps.dt.floor("h").value_counts().sort_index()
    trades_per_day = timestamps.dt.date.value_counts().sort_index()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].bar(range(len(trades_per_hour)), trades_per_hour.values)
    axes[0, 0].set_title("Trades per hour")
    axes[0, 0].set_xlabel("Hour index")
    axes[0, 1].bar(range(len(trades_per_day)), trades_per_day.values)
    axes[0, 1].set_title("Trades per day")
    axes[0, 1].set_xlabel("Day index")
    axes[1, 0].bar(trades_per_session.index, trades_per_session.values)
    axes[1, 0].set_title("Trades per session")
    axes[1, 0].set_xlabel("Session")
    axes[1, 1].bar([weekday_names[i] for i in trades_per_weekday.index], trades_per_weekday.values)
    axes[1, 1].set_title("Trades per weekday")
    axes[1, 1].set_xlabel("Weekday")
    fig.suptitle(f"Trade distribution (top={top_pct}% vol={vol_pct}% thresh={pred_threshold})")
    fig.tight_layout()
    fig.savefig(BACKTESTS_DIR / "trade_distribution.png", dpi=100)
    plt.close(fig)
    logging.info("Saved trade distribution plot to %s", BACKTESTS_DIR / "trade_distribution.png")


def run(
    pred_path: Path | None = None,
    top_pct_list: list[float] | None = None,
    pred_threshold_list: list[float] | None = None,
    vol_pct_list: list[int] | None = None,
    min_bars_between: int = 0,
    cost_mult: float = 1.0,
    plot_clustering: bool = True,
    regime_mask: np.ndarray | None = None,
    kill_switch_n: int = 0,
    kill_switch_pf: float = 0.9,
    dd_kill: float = 0.0,
    pause_bars: int = 0,
) -> dict:
    """Run grid, save results, plot clustering, return best cell."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )

    results = run_grid(
        pred_path=pred_path,
        top_pct_list=top_pct_list,
        pred_threshold_list=pred_threshold_list,
        vol_pct_list=vol_pct_list,
        min_bars_between=min_bars_between,
        cost_mult=cost_mult,
        regime_mask=regime_mask,
        kill_switch_n=kill_switch_n,
        kill_switch_pf=kill_switch_pf,
        dd_kill=dd_kill,
        pause_bars=pause_bars,
    )

    BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(BACKTESTS_DIR / "grid_results.csv", index=False)
    with open(BACKTESTS_DIR / "grid_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if plot_clustering:
        plot_trade_clustering(pred_path, top_pct=1.0, vol_pct=30, pred_threshold=0.00005)

    best = max(results, key=lambda r: (r["net_return"], -r["n_trades"]))
    logging.info(
        "Best: top_pct=%.2f pred_thresh=%.5f vol_pct=%d net=%.4f trades=%d",
        best["top_pct"],
        best["pred_threshold"],
        best["vol_pct"],
        best["net_return"],
        best["n_trades"],
    )
    return best
