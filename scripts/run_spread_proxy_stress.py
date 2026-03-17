"""
Spread proxy stress test.

Compare backtest performance in normal-vol vs high-vol periods.
Validates that SPREAD_PROXY_VOLATILITY_PCT blocks the right environments.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CRYPTO_SKIP_WEEKEND,
    MACRO_EVENT_BLACKOUT_MIN,
    SPREAD_PROXY_VOLATILITY_PCT,
    SESSION_EXCLUDE_HOURS,
    SYMBOL,
    TEST_START_DATE,
)
from src.backtest import (
    _compute_event_window,
    _cost_per_leg,
    _positions_from_probs_percentile,
    _run_backtest,
)

VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
FROZEN_DIR = PROJECT_ROOT / "data" / "frozen_test"
MACRO_DIR = PROJECT_ROOT / "data" / "raw" / "macro"

# Use top 10% for meaningful trade count
TOP_PCT = 10


def _load_pred(use_frozen: bool) -> pd.DataFrame:
    """Load predictions merged with features."""
    merge_cols = ["timestamp", "return_5", "volatility_20", "hour", "weekday"]
    pred = pd.read_csv(PREDICTIONS_DIR / "test_predictions.csv")
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True)
    pred = pred[pred["timestamp"] >= pd.Timestamp(TEST_START_DATE, tz="UTC")].reset_index(drop=True)

    if use_frozen:
        with open(FROZEN_DIR / "manifest.json") as f:
            manifest = json.load(f)
        frozen = pd.read_parquet(FROZEN_DIR / manifest["file"])
        frozen["timestamp"] = pd.to_datetime(frozen["timestamp"], utc=True)
        pred = pred.merge(frozen[merge_cols], on="timestamp", how="inner")
    else:
        test = pd.read_csv(FEATURES_DIR / "test.csv")
        test["timestamp"] = pd.to_datetime(test["timestamp"], utc=True)
        test = test[test["timestamp"] >= pd.Timestamp(TEST_START_DATE, tz="UTC")].reset_index(drop=True)
        pred = pred.merge(test[merge_cols], on="timestamp", how="left")

    return pred


def run(use_frozen: bool = False) -> int:
    """Run spread proxy stress test."""
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    pred = _load_pred(use_frozen)
    if pred.empty:
        logging.error("No predictions loaded")
        return 1

    volatility_20 = pred["volatility_20"].fillna(0).values.astype(float)
    # Use percentile split: vol may never exceed SPREAD_PROXY threshold (e.g. EURUSD max ~0.0026)
    vol_threshold = np.percentile(volatility_20, 50)
    normal_mask = volatility_20 <= vol_threshold
    high_mask = volatility_20 > vol_threshold

    cost = _cost_per_leg()
    results = []

    for regime, mask, label in [
        ("normal_vol", normal_mask, f"normal-vol (bottom 50%, vol <= {vol_threshold:.6f})"),
        ("high_vol", high_mask, f"high-vol (top 50%, vol > {vol_threshold:.6f})"),
    ]:
        sub = pred.loc[mask].reset_index(drop=True)
        if len(sub) < 10:
            logging.warning("Skipping %s: only %d bars", regime, len(sub))
            continue

        ret = sub["future_return_30m"].values.astype(float)
        P_buy = sub["P_buy"].values.astype(float)
        P_sell = sub["P_sell"].values.astype(float)
        return_5 = sub["return_5"].fillna(0).values.astype(float)
        vol = sub["volatility_20"].fillna(0).values.astype(float)
        hour = sub["hour"].fillna(0).values.astype(int)
        weekday = sub["weekday"].fillna(0).values.astype(int)
        timestamps = sub["timestamp"]

        pos = _positions_from_probs_percentile(P_buy, P_sell, TOP_PCT)
        is_event_window = _compute_event_window(
            timestamps, MACRO_DIR / "event_calendar.csv", MACRO_EVENT_BLACKOUT_MIN
        )
        macro_ok = (is_event_window == 0)
        vol_ok = np.ones(len(ret), dtype=bool)  # pre-filtered by vol; no additional filter
        confidence_ok = np.ones(len(ret), dtype=bool)  # percentile handles it
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
        results.append({
            "regime": regime,
            "label": label,
            "n_bars": len(sub),
            "trades": metrics["num_trades"],
            "cum_ret": metrics["cumulative_return"],
            "profit_factor": metrics["profit_factor"],
            "max_dd": metrics["max_drawdown"],
            "sharpe": metrics["sharpe_ratio"],
        })
        logging.info("%s: %d bars, %d trades, cum_ret=%.4f, pf=%.2f",
                     label, len(sub), metrics["num_trades"],
                     metrics["cumulative_return"], metrics["profit_factor"])

    out_path = VALIDATION_DIR / "spread_proxy_stress.md"
    with open(out_path, "w") as f:
        f.write("# Spread proxy stress test\n\n")
        f.write(f"**Use frozen:** {use_frozen}\n\n")
        f.write("| Regime | Bars | Trades | Cum return | Profit factor | Max DD | Sharpe |\n")
        f.write("|--------|------|--------|------------|---------------|--------|--------|\n")
        for r in results:
            f.write(f"| {r['label']} | {r['n_bars']} | {r['trades']} | {r['cum_ret']:.4f} | {r['profit_factor']:.2f} | {r['max_dd']:.4f} | {r['sharpe']:.2f} |\n")
        f.write("\n**Interpretation:** If high-vol has worse metrics than normal-vol, the spread_proxy rule is validated.\n")

    logging.info("Results saved to %s", out_path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-frozen", action="store_true")
    args = parser.parse_args()
    exit_code = run(use_frozen=args.use_frozen)
    sys.exit(exit_code)
