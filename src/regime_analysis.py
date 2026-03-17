"""
Phase 17 — Regime-change analysis.

Segment test period by volatility (or trend), run backtest per regime.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    CRYPTO_SKIP_WEEKEND,
    MACRO_EVENT_BLACKOUT_MIN,
    MIN_CONFIDENCE_PCT,
    NO_TRADE_THRESHOLD_PCT,
    SESSION_EXCLUDE_HOURS,
    SPREAD_PROXY_VOLATILITY_PCT,
    SYMBOL,
    VOL_REGIME_TOP_PCT,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MACRO_DIR = PROJECT_ROOT / "data" / "raw" / "macro"


def segment_by_regime(df: pd.DataFrame, method: str = "volatility") -> pd.Series:
    """
    Segment rows by regime. Returns Series of regime labels (low/mid/high or up/down).
    method="volatility": terciles of rolling 20-bar std of returns
    method="trend": positive vs negative rolling 50-bar return
    """
    if "volatility_20" not in df.columns and "return_5" not in df.columns:
        return pd.Series(["unknown"] * len(df), index=df.index)

    if method == "volatility":
        vol = df["volatility_20"].fillna(0)
        if vol.max() == 0:
            return pd.Series(["mid"] * len(df), index=df.index)
        terciles = vol.quantile([1/3, 2/3]).values
        low, high = terciles[0], terciles[1]
        regime = np.where(vol <= low, "low", np.where(vol >= high, "high", "mid"))
        return pd.Series(regime, index=df.index)

    if method == "trend":
        ret = df["return_5"].fillna(0)
        regime = np.where(ret > 0, "up", "down")
        return pd.Series(regime, index=df.index)

    return pd.Series(["unknown"] * len(df), index=df.index)


def backtest_by_regime(pred_df: pd.DataFrame) -> dict[str, dict]:
    """
    Run backtest within each regime. pred_df must have timestamp, P_buy, P_sell,
    future_return_30m, return_5, volatility_20, hour, weekday.
    Returns dict[regime_label, metrics].
    """
    from src.backtest import (
        _compute_event_window,
        _cost_per_leg,
        _positions_from_probs,
        _run_backtest,
    )

    regime_series = segment_by_regime(pred_df, method="volatility")
    cost = _cost_per_leg()
    results = {}
    full_vol = pred_df["volatility_20"].fillna(0).values.astype(float)
    vol_threshold = np.percentile(full_vol, 100 - VOL_REGIME_TOP_PCT) if VOL_REGIME_TOP_PCT else None

    for regime_label in regime_series.unique():
        mask = regime_series == regime_label
        sub = pred_df.loc[mask].reset_index(drop=True)
        if len(sub) < 10:
            continue

        ret = sub["future_return_30m"].values.astype(float)
        P_buy = sub["P_buy"].values.astype(float)
        P_sell = sub["P_sell"].values.astype(float)
        volatility_20 = sub["volatility_20"].fillna(0).values.astype(float)
        hour = sub["hour"].fillna(0).values.astype(int)
        weekday = sub["weekday"].fillna(0).values.astype(int)
        timestamps = sub["timestamp"]

        pos = _positions_from_probs(P_buy, P_sell, NO_TRADE_THRESHOLD_PCT)
        is_event_window = _compute_event_window(
            timestamps, MACRO_DIR / "event_calendar.csv", MACRO_EVENT_BLACKOUT_MIN
        )
        macro_ok = is_event_window == 0
        if vol_threshold is not None:
            vol_ok = volatility_20 >= vol_threshold
        else:
            vol_ok = volatility_20 <= SPREAD_PROXY_VOLATILITY_PCT
        confidence_ok = np.maximum(P_buy, P_sell) >= MIN_CONFIDENCE_PCT
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
        metrics, _ = _run_backtest(ret, pos, cost, filter_data)
        results[regime_label] = metrics

    return results
