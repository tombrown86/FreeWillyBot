"""
Regression experiment (Batch 1) — Multi-horizon return targets.

Builds same features as classifier path (price, cross-asset, macro, news, time)
but with regression targets: target_ret_3, target_ret_6, target_ret_12.
No Chronos/TimesFM. Saves to data/features_regression/.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.build_features import (
    build_cross_asset_features,
    build_macro_event_features,
    build_news_features,
    build_price_features,
    build_time_features,
    load_inputs,
)
from src.config import (
    REGRESSION_HORIZONS,
    TEST_START_DATE,
    TRAINING_START_DATE,
    USE_EXOGENOUS,
    USE_NEWS,
    VALIDATION_START_DATE,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_REGRESSION_DIR = PROJECT_ROOT / "data" / "features_regression"
LOG_FILE = PROJECT_ROOT / "data" / "logs" / "build_features_regression.log"


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


def run() -> None:
    """Build regression features and targets, split by time, save."""
    _setup_logging()

    FEATURES_REGRESSION_DIR.mkdir(parents=True, exist_ok=True)

    use_exog = USE_EXOGENOUS
    use_news_val = USE_NEWS

    price, cross, macro, events, news_files = load_inputs()
    if not use_exog:
        cross = None
        macro = None
        events = None
        logging.info("Exogenous disabled: skipping cross-asset, macro, events")
    if not use_news_val:
        news_files = []
    logging.info("Loaded price: %d rows", len(price))

    df = build_price_features(price)
    cross_feat = build_cross_asset_features(price, cross, macro) if use_exog else price[["timestamp"]].copy()
    macro_feat = build_macro_event_features(price, macro, events) if use_exog else price[["timestamp"]].copy()
    time_feat = build_time_features(price)
    news_feat = build_news_features(price, news_files)
    targets = build_regression_targets(price)

    feature_cols = [c for c in df.columns if c not in ("timestamp", "open", "high", "low", "close", "volume")]
    df = df[["timestamp"] + feature_cols]

    for name, feat_df in [("cross", cross_feat), ("macro", macro_feat), ("time", time_feat), ("news", news_feat)]:
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
    df = df.dropna(subset=target_cols)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    train = df[df["timestamp"] < pd.Timestamp(VALIDATION_START_DATE, tz="UTC")]
    val = df[
        (df["timestamp"] >= pd.Timestamp(VALIDATION_START_DATE, tz="UTC"))
        & (df["timestamp"] < pd.Timestamp(TEST_START_DATE, tz="UTC"))
    ]
    test = df[df["timestamp"] >= pd.Timestamp(TEST_START_DATE, tz="UTC")]

    for name, split in [("train", train), ("validation", val), ("test", test)]:
        split.to_csv(FEATURES_REGRESSION_DIR / f"{name}.csv", index=False)
        split.to_parquet(FEATURES_REGRESSION_DIR / f"{name}.parquet", index=False)

    logging.info("Saved train: %d | validation: %d | test: %d", len(train), len(val), len(test))
    logging.info("Targets: %s", target_cols)
    logging.info("Features: %s", [c for c in df.columns if c not in ("timestamp",) + tuple(target_cols)])


if __name__ == "__main__":
    run()
