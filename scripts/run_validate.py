"""
Phase 17 — Rolling-window validation.

Run backtest on configurable windows (1m, 6m, 12m) of recent test data.
Appends metrics to data/validation/validation_report.csv.
"""

import csv
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TEST_START_DATE

VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
FROZEN_DIR = PROJECT_ROOT / "data" / "frozen_test"
MACRO_DIR = PROJECT_ROOT / "data" / "raw" / "macro"


def _load_frozen_test() -> "pd.DataFrame":
    """Load frozen test set. Returns DataFrame with timestamp and merge columns."""
    import json

    import pandas as pd

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


def _run_window_backtest(pred, cost_per_leg: float, top_pct: int | None = None) -> dict:
    """Run backtest on a sliced window. Returns metrics dict."""
    import numpy as np

    from src.backtest import (
        _compute_event_window,
        _cost_per_leg,
        _positions_from_probs,
        _positions_from_probs_percentile,
        _run_backtest,
    )
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

    ret = pred["future_return_30m"].values.astype(float)
    P_buy = pred["P_buy"].values.astype(float)
    P_sell = pred["P_sell"].values.astype(float)
    return_5 = pred["return_5"].fillna(0).values.astype(float)
    volatility_20 = pred["volatility_20"].fillna(0).values.astype(float)
    hour = pred["hour"].fillna(0).values.astype(int)
    weekday = pred["weekday"].fillna(0).values.astype(int)
    timestamps = pred["timestamp"]

    if top_pct is not None:
        pos_strategy = _positions_from_probs_percentile(P_buy, P_sell, top_pct)
        confidence_ok = np.ones(len(ret), dtype=bool)
    else:
        pos_strategy = _positions_from_probs(P_buy, P_sell, NO_TRADE_THRESHOLD_PCT)
        confidence_ok = np.asarray(np.maximum(P_buy, P_sell) >= MIN_CONFIDENCE_PCT)

    is_event_window = _compute_event_window(
        timestamps, MACRO_DIR / "event_calendar.csv", MACRO_EVENT_BLACKOUT_MIN
    )
    macro_ok = np.asarray(is_event_window == 0)
    if VOL_REGIME_TOP_PCT is not None:
        vol_threshold = np.percentile(volatility_20, 100 - VOL_REGIME_TOP_PCT)
        vol_ok = np.asarray(volatility_20 >= vol_threshold)
    else:
        vol_ok = np.asarray(volatility_20 <= SPREAD_PROXY_VOLATILITY_PCT)
    session_ok = np.ones(len(ret), dtype=bool)
    if SESSION_EXCLUDE_HOURS:
        session_ok = np.array([h not in SESSION_EXCLUDE_HOURS for h in hour])
    weekend_ok = np.ones(len(ret), dtype=bool)
    if CRYPTO_SKIP_WEEKEND and "USDT" in SYMBOL:
        weekend_ok = np.asarray((weekday != 5) & (weekday != 6))
    day = np.asarray(timestamps.dt.normalize().astype(np.int64))

    filter_data = {
        "macro_ok": macro_ok,
        "vol_ok": vol_ok,
        "confidence_ok": confidence_ok,
        "session_ok": session_ok,
        "weekend_ok": weekend_ok,
        "day": day,
    }

    metrics_strategy, _ = _run_backtest(ret, pos_strategy, cost_per_leg, filter_data)
    return metrics_strategy


def _run_monthly(pred, cost_per_leg: float, top_pct: int | None = None) -> int:
    """Run backtest on each calendar month in test. Write monthly_windows.csv."""
    import pandas as pd

    pred["month"] = pred["timestamp"].dt.strftime("%Y-%m")
    months = sorted(pred["month"].unique())
    if not months:
        logging.error("No months in test period")
        return 1

    monthly_path = VALIDATION_DIR / "monthly_windows.csv"
    fieldnames = ["month", "start_date", "end_date", "cum_return", "max_dd", "sharpe", "n_trades", "run_at"]
    file_exists = monthly_path.exists()
    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    n_positive = 0
    cum_returns: list[tuple[str, float]] = []
    for m in months:
        window = pred[pred["month"] == m].copy()
        window = window.drop(columns=["month"])
        if len(window) < 10:
            continue
        metrics = _run_window_backtest(window, cost_per_leg, top_pct)
        start_date = window["timestamp"].min().strftime("%Y-%m-%d")
        end_date = window["timestamp"].max().strftime("%Y-%m-%d")
        cum_ret = metrics["cumulative_return"]
        if cum_ret > 0:
            n_positive += 1
        cum_returns.append((str(m), cum_ret))
        row = {
            "month": str(m),
            "start_date": start_date,
            "end_date": end_date,
            "cum_return": round(cum_ret, 4),
            "max_dd": round(metrics["max_drawdown"], 4),
            "sharpe": round(metrics["sharpe_ratio"], 2),
            "n_trades": metrics["num_trades"],
            "run_at": run_at,
        }
        with open(monthly_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                w.writeheader()
                file_exists = True
            w.writerow(row)
        logging.info("Month %s: cum_ret=%.4f max_dd=%.4f sharpe=%.2f trades=%d", str(m), row["cum_return"], row["max_dd"], row["sharpe"], row["n_trades"])

    n_total = len(months)
    pct = 100 * n_positive / n_total if n_total else 0
    logging.info("Monthly summary: %d positive / %d months (%.1f%%)", n_positive, n_total, pct)

    # Check whether results come from one exceptional period
    if len(cum_returns) >= 2:
        total_positive = sum(max(0, r) for _, r in cum_returns)
        if total_positive > 0:
            best_month, best_ret = max(cum_returns, key=lambda x: x[1])
            best_contrib = max(0, best_ret) / total_positive
            if best_contrib > 0.5:
                logging.warning(
                    "Exceptional period: %s contributes %.0f%% of positive return. Results may rely on one good month.",
                    best_month, 100 * best_contrib,
                )

    return 0


def run(months_list: list[int], regime: bool = False, use_frozen: bool = False, mode: str = "default", top_pct: int | None = None) -> int:
    """
    Run backtest on each window. Append to validation_report.csv.
    Returns 0 on success, 1 on failure.
    """
    import pandas as pd

    from src.backtest import _cost_per_leg

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    pred_path = PREDICTIONS_DIR / "test_predictions.csv"
    test_path = FEATURES_DIR / "test.csv"
    if not pred_path.exists():
        logging.error("Run train_meta_model first: %s not found", pred_path)
        return 1
    if not use_frozen and not test_path.exists():
        logging.error("Run build_features first: %s not found", test_path)
        return 1

    pred = pd.read_csv(pred_path)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True)
    pred = pred[pred["timestamp"] >= pd.Timestamp(TEST_START_DATE, tz="UTC")].reset_index(drop=True)

    if use_frozen:
        frozen = _load_frozen_test()
        merge_cols = ["timestamp", "return_5", "volatility_20", "hour", "weekday"]
        pred = pred.merge(frozen[merge_cols], on="timestamp", how="inner")
        if pred.empty:
            logging.error("No predictions overlap with frozen test timestamps")
            return 1
        logging.info("Using frozen test: %d bars", len(pred))

    merge_cols = ["timestamp", "return_5", "volatility_20", "hour", "weekday"]
    if not use_frozen:
        test = pd.read_csv(test_path)
        test["timestamp"] = pd.to_datetime(test["timestamp"], utc=True)
        test = test[test["timestamp"] >= pd.Timestamp(TEST_START_DATE, tz="UTC")].reset_index(drop=True)
        pred = pred.merge(test[merge_cols], on="timestamp", how="left")

    if pred.empty:
        logging.error("No test predictions in test period")
        return 1

    cost_per_leg = _cost_per_leg()

    if mode == "monthly":
        return _run_monthly(pred, cost_per_leg, top_pct)

    end_ts = pred["timestamp"].max()
    report_path = VALIDATION_DIR / "validation_report.csv"
    fieldnames = [
        "window_months",
        "start_date",
        "end_date",
        "cum_return",
        "max_dd",
        "sharpe",
        "profit_factor",
        "n_trades",
        "run_at",
    ]
    file_exists = report_path.exists()

    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    for months in months_list:
        start_ts = end_ts - timedelta(days=months * 30)
        window = pred[(pred["timestamp"] >= start_ts) & (pred["timestamp"] <= end_ts)]
        if window.empty:
            logging.warning("No data for %d-month window", months)
            continue

        metrics = _run_window_backtest(window, cost_per_leg, top_pct)
        row = {
            "window_months": months,
            "start_date": start_ts.strftime("%Y-%m-%d"),
            "end_date": end_ts.strftime("%Y-%m-%d"),
            "cum_return": round(metrics["cumulative_return"], 4),
            "max_dd": round(metrics["max_drawdown"], 4),
            "sharpe": round(metrics["sharpe_ratio"], 2),
            "profit_factor": round(metrics["profit_factor"], 2),
            "n_trades": metrics["num_trades"],
            "run_at": run_at,
        }

        with open(report_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                w.writeheader()
                file_exists = True
            w.writerow(row)

        logging.info(
            "Window %dm: cum_ret=%.4f max_dd=%.4f sharpe=%.2f trades=%d",
            months,
            row["cum_return"],
            row["max_dd"],
            row["sharpe"],
            row["n_trades"],
        )

    if regime:
        from src.regime_analysis import backtest_by_regime

        regime_metrics = backtest_by_regime(pred)
        regime_path = VALIDATION_DIR / "regime_report.csv"
        regime_fieldnames = ["regime", "cum_return", "max_dd", "sharpe", "n_trades", "run_at"]
        regime_exists = regime_path.exists()
        with open(regime_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=regime_fieldnames)
            if not regime_exists:
                w.writeheader()
            for reg, m in regime_metrics.items():
                w.writerow({
                    "regime": reg,
                    "cum_return": round(m["cumulative_return"], 4),
                    "max_dd": round(m["max_drawdown"], 4),
                    "sharpe": round(m["sharpe_ratio"], 2),
                    "n_trades": m["num_trades"],
                    "run_at": run_at,
                })
        logging.info("Regime report saved to %s", regime_path)

    return 0


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="Rolling-window validation")
    parser.add_argument("--months", type=int, nargs="+", default=[1, 6, 12], help="Window sizes in months")
    parser.add_argument("--regime", action="store_true", help="Run regime analysis")
    parser.add_argument("--use-frozen", action="store_true", help="Evaluate only on frozen test set")
    parser.add_argument("--mode", choices=["default", "monthly"], default="default",
                        help="default: 1m/6m/12m windows; monthly: rolling 1-month windows")
    parser.add_argument("--top-pct", type=int, choices=[10, 20, 30], metavar="PCT",
                        help="Trade only top X%% of signals by confidence (percentile filter)")
    args = parser.parse_args()

    exit_code = run(months_list=args.months, regime=args.regime, use_frozen=args.use_frozen, mode=args.mode, top_pct=args.top_pct)
    sys.exit(exit_code)
