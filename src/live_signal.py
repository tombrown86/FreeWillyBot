"""
Phase 14 — Live signal generation and validation.

Run on recent data as if live. Output: timestamp, signal, confidence, blocked, reason_blocked.
Log every block reason separately for Phase 17/19 analysis.
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    COOLDOWN_BARS_AFTER_LOSS,
    CRYPTO_SKIP_WEEKEND,
    MACRO_EVENT_BLACKOUT_MIN,
    MAX_DAILY_LOSS_PCT,
    MIN_CONFIDENCE_PCT,
    NO_TRADE_THRESHOLD_PCT,
    SESSION_EXCLUDE_HOURS,
    SPREAD_PROXY_VOLATILITY_PCT,
    VOL_REGIME_TOP_PCT,
    SYMBOL,
    TEST_START_DATE,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
MACRO_DIR = PROJECT_ROOT / "data" / "raw" / "macro"
LOG_FILE = PROJECT_ROOT / "data" / "logs" / "live_signal.log"

# Block reason labels for logging
BLOCK_REASONS = [
    "weak_confidence",
    "macro_blackout",
    "spread_proxy",
    "vol_regime",
    "cooldown",
    "daily_loss",
    "weekend",
    "session",
]


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


def _signal_label(pos: int) -> str:
    if pos == 1:
        return "BUY"
    if pos == -1:
        return "SELL"
    return "FLAT"


def _action_label(p: int, prev_pos: int, blocked: int) -> str:
    """Explicit action for Phase 15 execution: NONE, OPEN_LONG, OPEN_SHORT, CLOSE."""
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


def run(n_bars: int = 500) -> list[dict]:
    """
    Run live-style signal generation on the last n_bars of test data.
    Output: timestamp | signal | conf | blocked | action | size | reason
    Returns list of dicts with timestamp, signal, confidence, blocked, reason, action.
    """
    _setup_logging()

    # Load meta-model
    meta_path = MODELS_DIR / "meta_model.pkl"
    meta_cols_path = MODELS_DIR / "meta_feature_cols.json"
    if not meta_path.exists() or not meta_cols_path.exists():
        raise FileNotFoundError("Run train_meta_model first")

    with open(meta_path, "rb") as f:
        meta_model = pickle.load(f)
    with open(meta_cols_path) as f:
        meta_cols = json.load(f)

    # Load test features + forecaster predictions
    test = pd.read_csv(FEATURES_DIR / "test.csv")
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

    # Take last n_bars
    test = test.tail(n_bars).reset_index(drop=True)
    if test.empty:
        raise ValueError("No test data in range")

    # Ensure meta_cols exist
    missing = [c for c in meta_cols if c not in test.columns]
    if missing:
        logging.warning("Missing meta cols (filling 0): %s", missing)
        for c in missing:
            test[c] = 0.0

    X = test[meta_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    probs = meta_model.predict_proba(X)
    P_buy = probs[:, 2]
    P_sell = probs[:, 0]
    confidence = np.maximum(P_buy, P_sell)

    base_pos = _positions_from_probs(P_buy, P_sell, NO_TRADE_THRESHOLD_PCT)

    # Filter inputs
    volatility_20 = test["volatility_20"].fillna(0).values.astype(float)
    hour = test["hour"].fillna(0).values.astype(int)
    weekday = test["weekday"].fillna(0).values.astype(int)
    timestamps = test["timestamp"]
    ret = test["future_return_30m"].fillna(0).values.astype(float)

    is_event_window = _compute_event_window(
        timestamps, MACRO_DIR / "event_calendar.csv", MACRO_EVENT_BLACKOUT_MIN
    )
    macro_ok = is_event_window == 0
    if VOL_REGIME_TOP_PCT is not None:
        vol_threshold = np.percentile(volatility_20, 100 - VOL_REGIME_TOP_PCT)
        vol_ok = volatility_20 >= vol_threshold
    else:
        vol_ok = volatility_20 <= SPREAD_PROXY_VOLATILITY_PCT
    confidence_ok = confidence >= MIN_CONFIDENCE_PCT
    session_ok = np.ones(len(test), dtype=bool)
    if SESSION_EXCLUDE_HOURS:
        session_ok = np.array([h not in SESSION_EXCLUDE_HOURS for h in hour])
    weekend_ok = np.ones(len(test), dtype=bool)
    if CRYPTO_SKIP_WEEKEND and "USDT" in SYMBOL:
        weekend_ok = (weekday != 5) & (weekday != 6)
    day = timestamps.dt.normalize().astype(np.int64)

    # Block reason counters (for Phase 17/19 analysis)
    block_counts = {r: 0 for r in BLOCK_REASONS}

    # Sequential pass: apply stateful filters, output each row
    cooldown_remaining = 0
    day_start_equity = 1.0
    current_day = -1
    equity = 1.0
    prev_pos = 0
    entry_eq = 1.0

    logging.info("Live-style signal run: last %d bars", n_bars)
    logging.info("Output format: timestamp | signal | conf | blocked | action | size | reason")

    rows_out = []

    for i in range(len(test)):
        p = base_pos[i]
        prev_pos_at_bar = prev_pos
        reasons = []

        # Static filters
        if not macro_ok[i]:
            block_counts["macro_blackout"] += 1
            if base_pos[i] != 0:
                reasons.append("macro_blackout")
            p = 0
        if not vol_ok[i]:
            block_counts["vol_regime" if VOL_REGIME_TOP_PCT else "spread_proxy"] += 1
            if base_pos[i] != 0:
                reasons.append("vol_regime" if VOL_REGIME_TOP_PCT else "spread_proxy")
            p = 0
        if not confidence_ok[i]:
            block_counts["weak_confidence"] += 1
            if base_pos[i] != 0:
                reasons.append("weak_confidence")
            p = 0
        if not session_ok[i]:
            block_counts["session"] += 1
            if base_pos[i] != 0:
                reasons.append("session")
            p = 0
        if not weekend_ok[i]:
            block_counts["weekend"] += 1
            if base_pos[i] != 0:
                reasons.append("weekend")
            p = 0

        # Stateful filters
        bar_day = day[i]
        if bar_day != current_day:
            current_day = bar_day
            day_start_equity = equity
        if p != 0 and equity < day_start_equity * (1 - MAX_DAILY_LOSS_PCT):
            block_counts["daily_loss"] += 1
            reasons.append("daily_loss")
            p = 0
        if p != 0 and cooldown_remaining > 0:
            block_counts["cooldown"] += 1
            reasons.append("cooldown")
            p = 0

        # Cooldown trigger: did we just close a losing trade?
        if p != prev_pos:
            if prev_pos != 0:
                trade_ret = (equity / entry_eq - 1) * prev_pos
                if trade_ret < 0:
                    cooldown_remaining = COOLDOWN_BARS_AFTER_LOSS
            if p != 0:
                entry_eq = equity
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        # Update equity for next bar
        if p != 0:
            equity *= 1 + p * ret[i]
        prev_pos = p

        reason_str = ",".join(reasons) if reasons else ""
        blocked = 1 if p == 0 and base_pos[i] != 0 else 0
        action = _action_label(int(p), int(prev_pos_at_bar), int(blocked))
        size_str = "size=1.0" if p != 0 else ""
        reason_out = f"reason={reason_str}" if blocked and reason_str else ""

        ts_str = timestamps.iloc[i].strftime("%Y-%m-%d %H:%M:%S UTC")
        conf_str = f"conf={confidence[i]:.2f}"
        block_str = f"blocked={blocked}"
        action_str = f"action={action}"
        parts = [ts_str, _signal_label(p), conf_str, block_str, action_str]
        if size_str:
            parts.append(size_str)
        if reason_out:
            parts.append(reason_out)

        line = " | ".join(parts)
        print(line)

        rows_out.append({
            "timestamp": ts_str,
            "signal": _signal_label(p),
            "confidence": float(confidence[i]),
            "blocked": blocked,
            "reason": reason_str,
            "action": action,
            "P_buy": float(P_buy[i]),
            "P_sell": float(P_sell[i]),
        })

    # Log block reason summary
    logging.info("Block reason counts: %s", block_counts)
    for r in BLOCK_REASONS:
        if block_counts[r] > 0:
            logging.info("  %s: %d", r, block_counts[r])

    return rows_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Live-style signal generation on recent data")
    parser.add_argument("-n", "--n-bars", type=int, default=500, help="Number of recent bars to process")
    args = parser.parse_args()
    run(n_bars=args.n_bars)
