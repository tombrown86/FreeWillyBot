"""
Batch 2 — Core feature set for regression.

Minimal feature set: price/returns, volatility, trend, cross-asset, time, simple macro.
No news, no complex macro transforms, no sparse features.
Saves to data/features_regression_core/.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.build_features import load_inputs
from src.config import (
    MACRO_EVENT_BLACKOUT_MIN,
    REGRESSION_HORIZONS,
    TEST_START_DATE,
    USE_EXOGENOUS,
    VALIDATION_START_DATE,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_REGRESSION_CORE_DIR = PROJECT_ROOT / "data" / "features_regression_core"
LOG_FILE = PROJECT_ROOT / "data" / "logs" / "build_features_regression_core.log"


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


def build_regression_targets(price: pd.DataFrame) -> pd.DataFrame:
    """Multi-horizon return targets: target_ret_h = (close[t+h] / close[t]) - 1."""
    df = price[["timestamp", "close"]].copy()
    close = df["close"]
    max_h = max(REGRESSION_HORIZONS)
    for h in REGRESSION_HORIZONS:
        df[f"target_ret_{h}"] = (close.shift(-h) / close) - 1
    df = df.drop(columns=["close"])
    df = df.iloc[:-max_h]
    return df


def build_core_price_features(price: pd.DataFrame) -> pd.DataFrame:
    """Core price features: ret_1/3/6/12, vol_6/12/24, ma gaps, ma_slope_10."""
    df = price[["timestamp"]].copy()
    close = price["close"]
    ret_1 = close.pct_change(1)

    for h in [1, 3, 6, 12]:
        df[f"ret_{h}"] = close.pct_change(h)

    for w in [6, 12, 24]:
        df[f"vol_{w}"] = ret_1.rolling(w).std()

    for h in [10, 20, 50]:
        ma = close.rolling(h).mean()
        df[f"ma_{h}_gap"] = (close - ma) / close.replace(0, np.nan)

    ma_10 = close.rolling(10).mean()
    df["ma_slope_10"] = ma_10.diff(1)

    return df


def build_core_cross_asset_features(
    price: pd.DataFrame, cross: pd.DataFrame | None, macro: pd.DataFrame | None
) -> pd.DataFrame:
    """Core cross-asset: sp500_ret, vix_ret, gold_ret, oil_ret, us10y_change (1-bar only)."""
    df = price[["timestamp"]].copy()
    if cross is None and macro is None:
        return df

    merged = price[["timestamp"]].copy()
    if cross is not None:
        merged = merged.merge(cross, on="timestamp", how="left")
    if macro is not None and "TREASURY_10Y" in macro.columns:
        m = macro[["timestamp", "TREASURY_10Y"]].copy()
        merged = merged.merge(m, on="timestamp", how="left")

    mapping = [
        ("SP500", "sp500_ret", "pct_change"),
        ("VIX", "vix_ret", "pct_change"),
        ("GOLD", "gold_ret", "pct_change"),
        ("OIL", "oil_ret", "pct_change"),
    ]
    for col, out_name, op in mapping:
        if col not in merged.columns:
            continue
        s = merged[col].ffill()
        if s.isna().all():
            continue
        if op == "pct_change":
            ret = s.pct_change(1)
            ret = ret.replace([np.inf, -np.inf], np.nan)
            df[out_name] = ret

    if "TREASURY_10Y" in merged.columns:
        s = merged["TREASURY_10Y"].ffill()
        df["us10y_change"] = s.diff(1)

    return df


def build_core_time_features(price: pd.DataFrame) -> pd.DataFrame:
    """Time features: hour, weekday, London session, NY session."""
    df = price[["timestamp"]].copy()
    ts = pd.DatetimeIndex(df["timestamp"])
    df["hour"] = ts.hour
    df["weekday"] = ts.weekday
    df["is_london_session"] = ((ts.hour >= 8) & (ts.hour < 17)).astype(int)
    df["is_ny_session"] = ((ts.hour >= 13) & (ts.hour < 22)).astype(int)
    return df


def build_core_macro_features(
    price: pd.DataFrame, events: pd.DataFrame | None
) -> pd.DataFrame:
    """Simple macro: is_event_day, is_event_window (within ±N min of any event)."""
    df = price[["timestamp"]].copy()
    df["is_event_day"] = 0
    df["is_event_window"] = 0

    if events is None or events.empty:
        return df

    ev = events.copy()
    ev["event_timestamp_utc"] = pd.to_datetime(ev["event_timestamp_utc"], utc=True)
    ts = pd.DatetimeIndex(df["timestamp"])

    event_dates = set(ev["event_timestamp_utc"].dropna().dt.date)
    df["is_event_day"] = pd.Series(ts.date).isin(event_dates).astype(int).values

    all_events = ev["event_timestamp_utc"].dropna().sort_values()
    if len(all_events) > 0:
        window_sec = MACRO_EVENT_BLACKOUT_MIN * 60
        for e in all_events:
            diff_sec = np.abs((ts - e).total_seconds())
            df.loc[diff_sec <= window_sec, "is_event_window"] = 1

    return df


def run() -> None:
    """Build core regression features and targets, split by time, save."""
    _setup_logging()

    FEATURES_REGRESSION_CORE_DIR.mkdir(parents=True, exist_ok=True)

    use_exog = USE_EXOGENOUS
    price, cross, macro, events, _ = load_inputs()
    if not use_exog:
        cross = None
        macro = None
        events = None
        logging.info("Exogenous disabled: skipping cross-asset, macro, events")
    logging.info("Loaded price: %d rows", len(price))

    df = build_core_price_features(price)
    cross_feat = build_core_cross_asset_features(price, cross, macro) if use_exog else price[["timestamp"]].copy()
    time_feat = build_core_time_features(price)
    macro_feat = build_core_macro_features(price, events) if use_exog else build_core_macro_features(price, None)
    targets = build_regression_targets(price)

    for name, feat_df in [("cross", cross_feat), ("time", time_feat), ("macro", macro_feat)]:
        drop = [c for c in feat_df.columns if c != "timestamp"]
        if drop:
            right = feat_df[["timestamp"] + drop].drop_duplicates(subset=["timestamp"], keep="first")
            before_len = len(df)
            df = df.merge(right, on="timestamp", how="left")
            if len(df) != before_len:
                logging.warning("Merge %s produced row count change %d -> %d; deduped right", name, before_len, len(df))
                df = df.drop_duplicates(subset=["timestamp"], keep="first")

    target_cols = [f"target_ret_{h}" for h in REGRESSION_HORIZONS]
    df = df.merge(targets, on="timestamp", how="inner")
    feature_cols = [c for c in df.columns if c not in ("timestamp",) + tuple(target_cols)]
    df = df.dropna(subset=target_cols + feature_cols)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    train = df[df["timestamp"] < pd.Timestamp(VALIDATION_START_DATE, tz="UTC")]
    val = df[
        (df["timestamp"] >= pd.Timestamp(VALIDATION_START_DATE, tz="UTC"))
        & (df["timestamp"] < pd.Timestamp(TEST_START_DATE, tz="UTC"))
    ]
    test = df[df["timestamp"] >= pd.Timestamp(TEST_START_DATE, tz="UTC")]

    for name, split in [("train", train), ("validation", val), ("test", test)]:
        split.to_csv(FEATURES_REGRESSION_CORE_DIR / f"{name}.csv", index=False)
        split.to_parquet(FEATURES_REGRESSION_CORE_DIR / f"{name}.parquet", index=False)

    logging.info("Saved train: %d | validation: %d | test: %d", len(train), len(val), len(test))
    logging.info("Targets: %s", target_cols)
    logging.info("Features: %s", feature_cols)


if __name__ == "__main__":
    run()
