"""
Trade review: 10 best, 10 worst, 10 blocked high-confidence, 10 executed low-confidence.

Runs backtest with trade-level logging. Output: data/validation/trade_review.csv
"""

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
FROZEN_DIR = PROJECT_ROOT / "data" / "frozen_test"
MACRO_DIR = PROJECT_ROOT / "data" / "raw" / "macro"

from src.backtest import (
    _compute_event_window,
    _cost_per_leg,
    _load_frozen_test,
    _positions_from_probs_percentile,
)
from src.config import (
    COOLDOWN_BARS_AFTER_LOSS,
    MACRO_EVENT_BLACKOUT_MIN,
    MAX_DAILY_LOSS_PCT,
    SESSION_EXCLUDE_HOURS,
    VOL_REGIME_TOP_PCT,
)


def run() -> int:
    """Run trade review. Returns 0 on success."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    pred = pd.read_csv(PREDICTIONS_DIR / "test_predictions.csv")
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True)
    frozen = _load_frozen_test()
    pred = pred.merge(
        frozen[["timestamp", "return_5", "volatility_20", "hour", "weekday"]],
        on="timestamp",
        how="inner",
    )

    ret = pred["future_return_30m"].values.astype(float)
    P_buy = pred["P_buy"].values.astype(float)
    P_sell = pred["P_sell"].values.astype(float)
    return_5 = pred["return_5"].fillna(0).values.astype(float)
    volatility_20 = pred["volatility_20"].fillna(0).values.astype(float)
    timestamps = pred["timestamp"]
    hour = pred["hour"].fillna(0).values.astype(int)
    weekday = pred["weekday"].fillna(0).values.astype(int)

    base_pos = _positions_from_probs_percentile(P_buy, P_sell, 5)
    confidence = np.maximum(P_buy, P_sell)

    is_event_window = _compute_event_window(
        timestamps, MACRO_DIR / "event_calendar.csv", MACRO_EVENT_BLACKOUT_MIN
    )
    macro_ok = (is_event_window == 0)
    vol_threshold = np.percentile(volatility_20, 100 - VOL_REGIME_TOP_PCT)
    vol_ok = (volatility_20 >= vol_threshold)
    session_ok = np.ones(len(ret), dtype=bool)
    if SESSION_EXCLUDE_HOURS:
        session_ok = np.array([h not in SESSION_EXCLUDE_HOURS for h in hour])
    weekend_ok = np.ones(len(ret), dtype=bool)
    day = timestamps.dt.normalize().astype(np.int64)

    cost = _cost_per_leg()
    n = len(ret)
    equity = np.ones(n + 1)
    prev_pos = 0
    cooldown_remaining = 0
    day_start_equity = 1.0
    current_day = -1
    entry_eq = 1.0
    entry_idx = -1

    trades = []
    blocked_high_conf = []

    for i in range(n):
        p = base_pos[i]
        reasons = []

        if not macro_ok[i]:
            reasons.append("macro_blackout")
            p = 0
        if not vol_ok[i]:
            reasons.append("vol_regime")
            p = 0
        if not session_ok[i]:
            reasons.append("session")
            p = 0
        if not weekend_ok[i]:
            reasons.append("weekend")
            p = 0

        bar_day = day[i]
        if bar_day != current_day:
            current_day = bar_day
            day_start_equity = equity[i]
        if p != 0 and equity[i] < day_start_equity * (1 - MAX_DAILY_LOSS_PCT):
            reasons.append("daily_loss")
            p = 0
        if p != 0 and cooldown_remaining > 0:
            reasons.append("cooldown")
            p = 0

        if base_pos[i] != 0 and p == 0 and reasons:
            blocked_high_conf.append({
                "timestamp": timestamps.iloc[i],
                "confidence": confidence[i],
                "volatility_20": volatility_20[i],
                "base_pos": int(base_pos[i]),
                "reason": ",".join(reasons),
            })

        if p != prev_pos:
            legs = 1 if (prev_pos == 0 or p == 0) else 2
            equity[i + 1] = equity[i] * (1 - legs * cost)
            if prev_pos != 0:
                trade_ret = (equity[i + 1] / entry_eq - 1) * prev_pos
                trades.append({
                    "timestamp_exit": timestamps.iloc[i],
                    "position": int(prev_pos),
                    "entry_eq": entry_eq,
                    "exit_eq": equity[i + 1],
                    "trade_return": trade_ret,
                    "confidence": confidence[entry_idx] if entry_idx >= 0 else np.nan,
                    "volatility_20": volatility_20[entry_idx] if entry_idx >= 0 else np.nan,
                })
            if p != 0:
                entry_eq = equity[i + 1]
                entry_idx = i
            if prev_pos != 0 and (equity[i + 1] / entry_eq - 1) * prev_pos < 0:
                cooldown_remaining = COOLDOWN_BARS_AFTER_LOSS
        else:
            equity[i + 1] = equity[i]
        if p != 0:
            equity[i + 1] *= 1 + p * ret[i]
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
        prev_pos = p

    if prev_pos != 0:
        trade_ret = (equity[-1] / entry_eq - 1) * prev_pos
        trades.append({
            "timestamp_exit": timestamps.iloc[-1],
            "position": int(prev_pos),
            "entry_eq": entry_eq,
            "exit_eq": equity[-1],
            "trade_return": trade_ret,
            "confidence": confidence[entry_idx] if entry_idx >= 0 else np.nan,
            "volatility_20": volatility_20[entry_idx] if entry_idx >= 0 else np.nan,
        })

    trades_sorted = sorted(trades, key=lambda t: t["trade_return"], reverse=True)
    best_10 = trades_sorted[:10]
    worst_10 = trades_sorted[-10:]
    blocked_sorted = sorted(blocked_high_conf, key=lambda b: b["confidence"], reverse=True)
    blocked_10 = blocked_sorted[:10]

    executed_low_conf = sorted(
        [t for t in trades if t["confidence"] < np.percentile([x["confidence"] for x in trades], 25)],
        key=lambda t: t["trade_return"],
    )[:10]

    rows = []
    for t in best_10:
        rows.append({"category": "best", "reason": "", **t})
    for t in worst_10:
        rows.append({"category": "worst", "reason": "", **t})
    for b in blocked_10:
        rows.append({
            "category": "blocked_high_conf",
            "timestamp_exit": b["timestamp"],
            "position": b["base_pos"],
            "entry_eq": np.nan,
            "exit_eq": np.nan,
            "trade_return": np.nan,
            "confidence": b["confidence"],
            "volatility_20": b["volatility_20"],
            "reason": b["reason"],
        })
    for t in executed_low_conf:
        rows.append({"category": "executed_low_conf", "reason": "", **t})

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    report_path = VALIDATION_DIR / "trade_review.csv"
    fieldnames = ["category", "timestamp_exit", "position", "entry_eq", "exit_eq", "trade_return", "confidence", "volatility_20", "reason"]
    with open(report_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            wr = {k: r.get(k, "") for k in fieldnames}
            w.writerow(wr)

    logging.info("Trade review saved to %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(run())
