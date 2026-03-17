"""
Phase 12 — Honest backtest on test period only.

Load test predictions, apply no-trade threshold, simulate returns with costs.
Compute metrics: cumulative return, max drawdown, Sharpe, profit factor, num trades.
Compare against always-flat and simple momentum baselines.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CONFIDENCE_MARGIN_PCT,
    COOLDOWN_BARS_AFTER_LOSS,
    CRYPTO_FEE_PCT,
    CRYPTO_SLIPPAGE_PCT,
    CRYPTO_SKIP_WEEKEND,
    FX_SPREAD_PIPS,
    MACRO_EVENT_BLACKOUT_MIN,
    MAX_DAILY_LOSS_PCT,
    MIN_CONFIDENCE_PCT,
    NO_TRADE_THRESHOLD_PCT,
    SESSION_EXCLUDE_HOURS,
    SPREAD_PROXY_VOLATILITY_PCT,
    SYMBOL,
    TEST_START_DATE,
    VOL_REGIME_TOP_PCT,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
FROZEN_DIR = PROJECT_ROOT / "data" / "frozen_test"
MACRO_DIR = PROJECT_ROOT / "data" / "raw" / "macro"
BACKTESTS_DIR = PROJECT_ROOT / "data" / "backtests"
LOG_FILE = PROJECT_ROOT / "data" / "logs" / "backtest.log"

FX_SPREAD = FX_SPREAD_PIPS * 0.0001
BARS_PER_YEAR = 252 * 288  # 5-min bars


def _setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _cost_per_leg() -> float:
    is_crypto = "USDT" in SYMBOL
    return (CRYPTO_FEE_PCT + CRYPTO_SLIPPAGE_PCT) if is_crypto else FX_SPREAD


def _positions_from_probs(
    P_buy: np.ndarray,
    P_sell: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Return position: 1=long, -1=short, 0=no-trade."""
    pos = np.where(P_buy > P_sell, 1, np.where(P_sell > P_buy, -1, 0))
    below = np.maximum(P_buy, P_sell) < threshold
    pos[below] = 0
    return pos.astype(int)


def _positions_from_probs_percentile(
    P_buy: np.ndarray,
    P_sell: np.ndarray,
    top_pct: int,
) -> np.ndarray:
    """Return position: 1=long, -1=short, 0=no-trade. Trade only top X% of directional signals by confidence."""
    conf = np.maximum(P_buy, P_sell)
    pos = np.where(P_buy > P_sell, 1, np.where(P_sell > P_buy, -1, 0))
    directional = pos != 0
    if directional.sum() == 0:
        return pos.astype(int)
    threshold = np.percentile(conf[directional], 100 - top_pct)
    keep = (conf >= threshold) & directional
    pos[~keep] = 0
    return pos.astype(int)


def _compute_event_window(
    timestamps: pd.DatetimeIndex,
    events_path: Path,
    blackout_min: int,
) -> np.ndarray:
    """Return 1 where bar is within ±blackout_min of a high-importance event, else 0."""
    if not events_path.exists():
        return np.zeros(len(timestamps), dtype=int)

    events = pd.read_csv(events_path)
    events = events[events["importance"] == "high"]
    event_times = pd.to_datetime(events["event_timestamp_utc"], utc=True).values
    event_minutes = event_times.astype("datetime64[m]").astype(np.int64)

    bar_minutes = timestamps.values.astype("datetime64[m]").astype(np.int64)
    bar_minutes = np.expand_dims(bar_minutes, axis=1)
    diff = np.abs(bar_minutes - event_minutes)
    min_dist = np.min(diff, axis=1)
    return (min_dist <= blackout_min).astype(int)


def _run_backtest(
    ret: np.ndarray,
    positions: np.ndarray,
    cost_per_leg: float,
    filter_data: dict | None = None,
    filter_overrides: dict[str, bool] | None = None,
) -> tuple[dict, dict]:
    """Compute strategy metrics with costs. Optionally apply safety filters during the loop."""
    n = len(positions)
    equity = np.ones(n + 1)
    effective_pos = np.zeros(n, dtype=int)
    prev_pos = 0
    n_trades = 0
    filter_stats = {
        "macro_blackout_bars": 0,
        "spread_proxy_bars": 0,
        "weak_confidence_bars": 0,
        "max_daily_loss_bars": 0,
        "cooldown_bars": 0,
        "weekend_bars": 0,
        "session_exclude_bars": 0,
    }
    cooldown_remaining = 0
    day_start_equity = 1.0
    current_day = -1
    entry_eq = 1.0

    overrides = filter_overrides or {}

    for i in range(n):
        p = positions[i]

        if filter_data is not None:
            macro_ok = filter_data["macro_ok"][i] if not overrides.get("macro_ok") else True
            vol_ok = filter_data["vol_ok"][i] if not overrides.get("vol_ok") else True
            conf_ok = filter_data["confidence_ok"][i] if not overrides.get("confidence_ok") else True
            sess_ok = filter_data["session_ok"][i] if not overrides.get("session_ok") else True
            wkd_ok = filter_data["weekend_ok"][i] if not overrides.get("weekend_ok") else True
            static_ok = macro_ok & vol_ok & conf_ok & sess_ok & wkd_ok

            if not static_ok:
                if not (overrides.get("macro_ok") or filter_data["macro_ok"][i]):
                    filter_stats["macro_blackout_bars"] += 1
                if not (overrides.get("vol_ok") or filter_data["vol_ok"][i]):
                    filter_stats["spread_proxy_bars"] += 1
                if not (overrides.get("confidence_ok") or filter_data["confidence_ok"][i]):
                    filter_stats["weak_confidence_bars"] += 1
                if not (overrides.get("session_ok") or filter_data["session_ok"][i]):
                    filter_stats["session_exclude_bars"] += 1
                if not (overrides.get("weekend_ok") or filter_data["weekend_ok"][i]):
                    filter_stats["weekend_bars"] += 1
                p = 0
            elif not overrides.get("cooldown") and cooldown_remaining > 0:
                filter_stats["cooldown_bars"] += 1
                p = 0
                cooldown_remaining -= 1
            else:
                bar_day = filter_data["day"][i]
                if bar_day != current_day:
                    current_day = bar_day
                    day_start_equity = equity[i]
                if not overrides.get("daily_loss") and equity[i] < day_start_equity * (1 - MAX_DAILY_LOSS_PCT):
                    filter_stats["max_daily_loss_bars"] += 1
                    p = 0

        effective_pos[i] = p
        if p != prev_pos:
            legs = 1 if (prev_pos == 0 or p == 0) else 2
            equity[i + 1] = equity[i] * (1 - legs * cost_per_leg)
            n_trades += 1
            if prev_pos != 0:
                trade_ret = (equity[i + 1] / entry_eq - 1) * prev_pos
                if trade_ret < 0:
                    cooldown_remaining = COOLDOWN_BARS_AFTER_LOSS
            if p != 0:
                entry_eq = equity[i + 1]
        else:
            equity[i + 1] = equity[i]
        # ret[i] = future_return_30m = 6-bar forward return (close[i] to close[i+6]). Strategy uses rolling
        # horizon: re-evaluate every bar; when position p != 0, credit that bar's 6-bar return.
        if p != 0:
            equity[i + 1] *= 1 + p * ret[i]
        prev_pos = p

    cum_ret = equity[-1] - 1.0

    eq = equity[1:]
    peak = np.maximum.accumulate(eq)
    drawdown = (peak - eq) / np.where(peak > 0, peak, 1)
    max_dd = float(np.max(drawdown))

    period_ret = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1)
    period_ret = period_ret[np.isfinite(period_ret)]
    sharpe = float(np.sqrt(BARS_PER_YEAR) * np.mean(period_ret) / np.std(period_ret)) if len(period_ret) > 1 and np.std(period_ret) > 0 else 0.0

    gains = period_ret[period_ret > 0].sum()
    losses = np.abs(period_ret[period_ret < 0].sum())
    profit_factor = float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)

    trade_returns = []
    prev_pos = 0
    entry_eq = 1.0
    for i in range(n):
        p = effective_pos[i]
        if p != prev_pos:
            if prev_pos != 0:
                trade_ret = (equity[i + 1] / entry_eq - 1) * prev_pos
                trade_returns.append(trade_ret)
            if p != 0:
                entry_eq = equity[i + 1]
        prev_pos = p
    if prev_pos != 0:
        trade_returns.append((equity[-1] / entry_eq - 1) * prev_pos)

    avg_ret_per_trade = float(np.mean(trade_returns)) if trade_returns else 0.0

    metrics = {
        "cumulative_return": cum_ret,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "profit_factor": profit_factor,
        "num_trades": n_trades,
        "avg_return_per_trade": avg_ret_per_trade,
    }
    return metrics, filter_stats


def _load_frozen_test() -> pd.DataFrame:
    """Load frozen test set. Returns DataFrame with timestamp and merge columns."""
    manifest_path = FROZEN_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Frozen test not found. Run freeze_test_set.py first.")
    with open(manifest_path) as f:
        manifest = json.load(f)
    parquet_path = FROZEN_DIR / manifest["file"]
    if not parquet_path.exists():
        raise FileNotFoundError(f"Frozen parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def run(
    use_frozen: bool = False,
    top_pct: int | None = None,
    cost_mult: float = 1.0,
    vol_top_pct: int | None = -1,
    filter_overrides: dict[str, bool] | None = None,
    return_only: bool = False,
    pred_path: Path | str | None = None,
    pred_df: pd.DataFrame | None = None,
    positions_override: np.ndarray | None = None,
):
    """Run backtest and save report."""
    if not return_only:
        _setup_logging()

    if pred_df is not None:
        pred = pred_df.copy()
    else:
        if pred_path:
            p = Path(pred_path)
            if not p.is_absolute():
                p = PREDICTIONS_DIR / p
        else:
            p = PREDICTIONS_DIR / "test_predictions.csv"
        if not p.exists():
            raise FileNotFoundError("Run train_meta_model first to generate test_predictions.csv")
        pred = pd.read_csv(p)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True)
    pred = pred[pred["timestamp"] >= pd.Timestamp(TEST_START_DATE, tz="UTC")].reset_index(drop=True)

    if pred.empty:
        raise ValueError("No test predictions in test period")

    merge_cols = ["timestamp", "return_5", "volatility_20", "hour", "weekday"]
    if pred_df is None:
        if use_frozen:
            frozen = _load_frozen_test()
            pred = pred.merge(frozen[merge_cols], on="timestamp", how="inner")
            if pred.empty:
                raise ValueError("No predictions overlap with frozen test timestamps")
            logging.info("Using frozen test: %d bars", len(pred))
        else:
            test = pd.read_csv(FEATURES_DIR / "test.csv")
            test["timestamp"] = pd.to_datetime(test["timestamp"], utc=True)
            test = test[test["timestamp"] >= pd.Timestamp(TEST_START_DATE, tz="UTC")].reset_index(drop=True)
            pred = pred.merge(test[merge_cols], on="timestamp", how="left")
    else:
        if use_frozen and not all(c in pred.columns for c in merge_cols[1:]):
            raise ValueError("pred_df must have return_5, volatility_20, hour, weekday when use_frozen")
        if use_frozen:
            logging.info("Using frozen test: %d bars", len(pred))

    ret = pred["future_return_30m"].values.astype(float)
    P_buy = pred["P_buy"].values.astype(float)
    P_sell = pred["P_sell"].values.astype(float)
    return_5 = pred["return_5"].fillna(0).values.astype(float)
    volatility_20 = pred["volatility_20"].fillna(0).values.astype(float)
    hour = pred["hour"].fillna(0).values.astype(int)
    weekday = pred["weekday"].fillna(0).values.astype(int)
    timestamps = pred["timestamp"]

    cost = _cost_per_leg() * cost_mult

    # Strategy positions: override, percentile filter, or threshold
    if positions_override is not None:
        pos_strategy = np.asarray(positions_override, dtype=int)
        confidence_ok = np.ones(len(ret), dtype=bool)
    elif top_pct is not None:
        pos_strategy = _positions_from_probs_percentile(P_buy, P_sell, top_pct)
        confidence_ok = np.ones(len(ret), dtype=bool)  # positions already encode percentile filter
    else:
        pos_strategy = _positions_from_probs(P_buy, P_sell, NO_TRADE_THRESHOLD_PCT)
        confidence_ok = (np.maximum(P_buy, P_sell) >= MIN_CONFIDENCE_PCT)

    # Build filter data for strategy backtest
    is_event_window = _compute_event_window(
        timestamps, MACRO_DIR / "event_calendar.csv", MACRO_EVENT_BLACKOUT_MIN
    )
    macro_ok = (is_event_window == 0)
    vol_regime: int | None = None if vol_top_pct == -1 else (vol_top_pct if vol_top_pct is not None else VOL_REGIME_TOP_PCT)
    if vol_regime is not None:
        vol_threshold = np.percentile(volatility_20, 100 - vol_regime)
        vol_ok = (volatility_20 >= vol_threshold)
    else:
        vol_ok = (volatility_20 <= SPREAD_PROXY_VOLATILITY_PCT)
    session_ok = np.ones(len(ret), dtype=bool)
    if SESSION_EXCLUDE_HOURS:
        session_ok = np.array([h not in SESSION_EXCLUDE_HOURS for h in hour])
    weekend_ok = np.ones(len(ret), dtype=bool)
    if CRYPTO_SKIP_WEEKEND and "USDT" in SYMBOL:
        weekend_ok = (weekday != 5) & (weekday != 6)
    day = timestamps.dt.normalize().astype(np.int64)

    filter_data = {
        "macro_ok": macro_ok,
        "vol_ok": vol_ok,
        "confidence_ok": confidence_ok,
        "session_ok": session_ok,
        "weekend_ok": weekend_ok,
        "day": day,
    }
    filters_applied = [
        "macro_blackout",
        "vol_regime_top" if vol_regime is not None else "spread_proxy",
        "weak_confidence",
        "max_daily_loss",
        "cooldown_after_loss",
        "weekend" if (CRYPTO_SKIP_WEEKEND and "USDT" in SYMBOL) else None,
        "session" if SESSION_EXCLUDE_HOURS else None,
    ]
    filters_applied = [f for f in filters_applied if f is not None]

    # Momentum baseline: long if return_5 > 0, short if return_5 < 0
    pos_momentum = np.where(return_5 > 0, 1, np.where(return_5 < 0, -1, 0))

    # Always-flat
    pos_flat = np.zeros(len(ret), dtype=int)

    metrics_strategy, filter_stats = _run_backtest(
        ret, pos_strategy, cost, filter_data, filter_overrides=filter_overrides
    )
    metrics_momentum, _ = _run_backtest(ret, pos_momentum, cost, None)
    metrics_flat, _ = _run_backtest(ret, pos_flat, cost, None)

    config = {
        "symbol": SYMBOL,
        "test_start": str(TEST_START_DATE),
        "use_frozen": use_frozen,
        "no_trade_threshold": NO_TRADE_THRESHOLD_PCT,
        "cost_per_leg": cost,
        "n_bars": len(pred),
    }
    if top_pct is not None:
        config["top_pct"] = top_pct
    if cost_mult != 1.0:
        config["cost_mult"] = cost_mult
    if vol_regime is not None:
        config["vol_regime_top_pct"] = vol_regime
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "filters_applied": filters_applied,
        "filter_stats": filter_stats,
        "strategy": metrics_strategy,
        "baseline_momentum": metrics_momentum,
        "baseline_flat": metrics_flat,
    }

    if return_only:
        return metrics_strategy, filter_stats, report

    BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = BACKTESTS_DIR / f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logging.info("Backtest report saved to %s", report_path)
    logging.info("Strategy: cum_ret=%.4f max_dd=%.4f sharpe=%.2f pf=%.2f trades=%d",
                 metrics_strategy["cumulative_return"], metrics_strategy["max_drawdown"],
                 metrics_strategy["sharpe_ratio"], metrics_strategy["profit_factor"],
                 metrics_strategy["num_trades"])
    logging.info("Momentum: cum_ret=%.4f max_dd=%.4f sharpe=%.2f",
                 metrics_momentum["cumulative_return"], metrics_momentum["max_drawdown"],
                 metrics_momentum["sharpe_ratio"])
    logging.info("Flat: cum_ret=%.4f", metrics_flat["cumulative_return"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest on test period")
    parser.add_argument("--use-frozen", action="store_true", help="Evaluate only on frozen test set")
    parser.add_argument("--top-pct", type=int, choices=[5, 10, 15, 20, 30], metavar="PCT",
                        help="Trade only top X%% of signals by confidence (percentile filter)")
    parser.add_argument("--cost-mult", type=float, default=1.0, choices=[1.0, 1.5, 2.0],
                        help="Cost multiplier (1.5x, 2.0x for stress test)")
    parser.add_argument("--vol-top-pct", type=int, choices=[20, 30, 40, 50, 60], metavar="PCT",
                        help="Trade only when volatility_20 in top X%% (high vol regime)")
    parser.add_argument("--no-vol-regime", action="store_true",
                        help="Disable vol regime filter; use legacy SPREAD_PROXY")
    args = parser.parse_args()
    vol_top: int | None = -1 if args.no_vol_regime else args.vol_top_pct
    run(use_frozen=args.use_frozen, top_pct=args.top_pct, cost_mult=args.cost_mult, vol_top_pct=vol_top)
