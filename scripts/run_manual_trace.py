"""
Manual trace: inspect 20-30 consecutive bars for one day.

Verify features, prediction, block reason, and trade outcome.
Usage: python -m scripts.run_manual_trace --date 2024-01-15 --n-bars 25
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    COOLDOWN_BARS_AFTER_LOSS,
    CRYPTO_SKIP_WEEKEND,
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

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
MACRO_DIR = PROJECT_ROOT / "data" / "raw" / "macro"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"


def _compute_event_window(
    timestamps: pd.DatetimeIndex,
    events_path: Path,
    blackout_min: int,
) -> np.ndarray:
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


def _positions_from_probs(P_buy: np.ndarray, P_sell: np.ndarray, threshold: float) -> np.ndarray:
    pos = np.where(P_buy > P_sell, 1, np.where(P_sell > P_buy, -1, 0))
    below = np.maximum(P_buy, P_sell) < threshold
    pos[below] = 0
    return pos.astype(int)


def _action_label(p: int, prev_pos: int, blocked: int) -> str:
    if blocked:
        return "NONE"
    if p == prev_pos:
        return "NONE"
    if prev_pos != 0 and p == 0:
        return "CLOSE"
    if p == 1:
        return "OPEN_LONG"
    if p == -1:
        return "OPEN_SHORT"
    return "NONE"


def run(date_str: str = "2024-01-15", n_bars: int = 25) -> int:
    """Run manual trace for one day. Returns 0 on success."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    test_path = FEATURES_DIR / "test.csv"
    if not test_path.exists():
        logging.error("test.csv not found. Run build_features first.")
        return 1

    meta_path = MODELS_DIR / "meta_model.pkl"
    meta_cols_path = MODELS_DIR / "meta_feature_cols.json"
    if not meta_path.exists() or not meta_cols_path.exists():
        logging.error("Meta model not found. Run train_meta_model first.")
        return 1

    test = pd.read_csv(test_path)
    test["timestamp"] = pd.to_datetime(test["timestamp"], utc=True)
    test = test[test["timestamp"] >= pd.Timestamp(TEST_START_DATE, tz="UTC")].reset_index(drop=True)

    forecaster_path = FEATURES_DIR / "forecaster_predictions.csv"
    if forecaster_path.exists():
        fc = pd.read_csv(forecaster_path)
        fc["timestamp"] = pd.to_datetime(fc["timestamp"], utc=True)
        test = test.merge(fc, on="timestamp", how="left")
        for c in ["chronos_pred_return", "timesfm_pred_return"]:
            if c not in test.columns:
                test[c] = 0.0
    else:
        test["chronos_pred_return"] = 0.0
        test["timesfm_pred_return"] = 0.0

    target_date = pd.Timestamp(date_str, tz="UTC").date()
    mask = test["timestamp"].dt.date == target_date
    day_df = test.loc[mask].head(n_bars).reset_index(drop=True)
    if day_df.empty:
        logging.error("No bars for date %s in test period", date_str)
        return 1

    with open(meta_path, "rb") as f:
        meta_model = pickle.load(f)
    with open(meta_cols_path) as f:
        meta_cols = json.load(f)

    missing = [c for c in meta_cols if c not in day_df.columns]
    if missing:
        for c in missing:
            day_df[c] = 0.0

    X = day_df[meta_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    probs = meta_model.predict_proba(X)
    P_buy = probs[:, 2]
    P_sell = probs[:, 0]
    confidence = np.maximum(P_buy, P_sell)
    base_pos = _positions_from_probs(P_buy, P_sell, NO_TRADE_THRESHOLD_PCT)

    volatility_20 = day_df["volatility_20"].fillna(0).values.astype(float)
    hour = day_df["hour"].fillna(0).values.astype(int)
    weekday = day_df["weekday"].fillna(0).values.astype(int)
    timestamps = day_df["timestamp"]
    ret = day_df["future_return_30m"].fillna(np.nan).values.astype(float)

    is_event_window = _compute_event_window(
        timestamps, MACRO_DIR / "event_calendar.csv", MACRO_EVENT_BLACKOUT_MIN
    )
    macro_ok = is_event_window == 0
    if VOL_REGIME_TOP_PCT is not None:
        full_vol = test["volatility_20"].fillna(0).values.astype(float)
        vol_threshold = np.percentile(full_vol, 100 - VOL_REGIME_TOP_PCT)
        vol_ok = volatility_20 >= vol_threshold
    else:
        vol_ok = volatility_20 <= SPREAD_PROXY_VOLATILITY_PCT
    confidence_ok = confidence >= MIN_CONFIDENCE_PCT
    session_ok = np.ones(len(day_df), dtype=bool)
    if SESSION_EXCLUDE_HOURS:
        session_ok = np.array([h not in SESSION_EXCLUDE_HOURS for h in hour])
    weekend_ok = np.ones(len(day_df), dtype=bool)
    if CRYPTO_SKIP_WEEKEND and "USDT" in SYMBOL:
        weekend_ok = (weekday != 5) & (weekday != 6)
    day_norm = timestamps.dt.normalize().astype(np.int64)

    pred_labels = {0: "sell", 1: "no_trade", 2: "buy"}
    pred_class = np.argmax(probs, axis=1)

    cooldown_remaining = 0
    day_start_equity = 1.0
    current_day = -1
    equity = 1.0
    prev_pos = 0
    entry_eq = 1.0

    logging.info("Manual trace: %s, %d bars", date_str, len(day_df))
    logging.info("")

    for i in range(len(day_df)):
        p = base_pos[i]
        prev_pos_at_bar = prev_pos
        reasons = []

        if not macro_ok[i]:
            if base_pos[i] != 0:
                reasons.append("macro_blackout")
            p = 0
        if not vol_ok[i]:
            if base_pos[i] != 0:
                reasons.append("spread_proxy")
            p = 0
        if not confidence_ok[i]:
            if base_pos[i] != 0:
                reasons.append("weak_confidence")
            p = 0
        if not session_ok[i]:
            if base_pos[i] != 0:
                reasons.append("session")
            p = 0
        if not weekend_ok[i]:
            if base_pos[i] != 0:
                reasons.append("weekend")
            p = 0

        bar_day = day_norm.iloc[i]
        if bar_day != current_day:
            current_day = bar_day
            day_start_equity = equity
        if p != 0 and equity < day_start_equity * (1 - MAX_DAILY_LOSS_PCT):
            reasons.append("daily_loss")
            p = 0
        if p != 0 and cooldown_remaining > 0:
            reasons.append("cooldown")
            p = 0

        if p != prev_pos:
            if prev_pos != 0:
                trade_ret = (equity / entry_eq - 1) * prev_pos
                if trade_ret < 0:
                    cooldown_remaining = COOLDOWN_BARS_AFTER_LOSS
            if p != 0:
                entry_eq = equity
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        if p != 0:
            equity *= 1 + p * (ret[i] if not np.isnan(ret[i]) else 0)
        prev_pos = p

        reason_str = ",".join(reasons) if reasons else ""
        blocked = 1 if p == 0 and base_pos[i] != 0 else 0
        action = _action_label(int(p), int(prev_pos_at_bar), int(blocked))

        r1 = day_df["return_1"].iloc[i] if "return_1" in day_df.columns else np.nan
        r5 = day_df["return_5"].iloc[i] if "return_5" in day_df.columns else np.nan
        vol = volatility_20[i]
        fut = ret[i]

        logging.info("Bar %d: %s", i, timestamps.iloc[i].strftime("%Y-%m-%d %H:%M:%S UTC"))
        logging.info("  return_1=%.6f return_5=%.6f vol_20=%.6f hour=%d weekday=%d", r1, r5, vol, hour[i], weekday[i])
        logging.info("  P_buy=%.2f P_sell=%.2f predicted=%s (%s)", P_buy[i], P_sell[i], pred_class[i], pred_labels[pred_class[i]])
        logging.info("  blocked=%d reason=%s", blocked, reason_str or "none")
        logging.info("  action=%s", action)
        logging.info("  future_return_30m=%.6f (actual)", fut if not np.isnan(fut) else float("nan"))
        logging.info("")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual trace: inspect bars for one day")
    parser.add_argument("--date", default="2024-01-15", help="Date YYYY-MM-DD")
    parser.add_argument("--n-bars", type=int, default=25, help="Number of bars to show")
    args = parser.parse_args()
    sys.exit(run(date_str=args.date, n_bars=args.n_bars))
