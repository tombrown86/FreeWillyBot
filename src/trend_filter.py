"""
Trend filter utility for research experiments.

Loads the 5-minute EURUSD price series, resamples to a configurable period
(default 1h), computes a simple trend indicator (close > MA{ma_window} =
uptrend), then maps the trend state back onto any set of 5-minute timestamps
via forward-fill (merge_asof).

Public API
----------
compute_trend_mask(timestamps_utc, ma_window=20, resample_period="1h") -> pd.DataFrame
    Columns returned:
        timestamp         – same as input (UTC)
        trend_up_1h       – float 1.0 when close > MA, else 0.0 (NaN during warmup)
        trend_down_1h     – float 1.0 when close < MA, else 0.0 (NaN during warmup)
        trend_strength_1h – close / MA - 1 (positive = above MA, NaN during warmup)
    Note: column names always use the "_1h" suffix regardless of resample_period
    so that downstream code can use a single naming convention.

print_trend_stats(df) -> None
    Prints % bars in uptrend / downtrend, NaN count after warmup.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRICE_PATH = PROJECT_ROOT / "data" / "processed" / "price" / "EURUSD_5min_clean.parquet"


def compute_trend_mask(
    timestamps_utc: "pd.DatetimeIndex | pd.Series",
    ma_window: int = 20,
    resample_period: str = "1h",
) -> pd.DataFrame:
    """Compute higher-timeframe MA trend state aligned to ``timestamps_utc``.

    Parameters
    ----------
    timestamps_utc  : DatetimeIndex or Series of UTC timestamps
    ma_window       : rolling window on the resampled bars (default 20)
    resample_period : pandas offset string — "1h" (default) or "4h"

    Returns
    -------
    DataFrame with one row per input timestamp and columns:
        trend_up_1h, trend_down_1h, trend_strength_1h
    Column names always use the "_1h" suffix so downstream code is uniform.
    NaN values appear only during the initial MA warmup period.
    """
    df_5m = pd.read_parquet(PRICE_PATH, columns=["timestamp", "close"])
    df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"], utc=True)
    df_5m = df_5m.sort_values("timestamp").drop_duplicates("timestamp")

    df_htf = (
        df_5m.set_index("timestamp")
        .resample(resample_period)["close"]
        .last()
        .dropna()
        .reset_index()
    )
    df_htf["ma"] = df_htf["close"].rolling(ma_window, min_periods=ma_window).mean()

    warmup = df_htf["ma"].isna()
    df_htf["trend_up_1h"] = (df_htf["close"] > df_htf["ma"]).astype(float)
    df_htf["trend_down_1h"] = (df_htf["close"] < df_htf["ma"]).astype(float)
    df_htf["trend_strength_1h"] = df_htf["close"] / df_htf["ma"] - 1.0

    # During warmup MA is NaN — propagate NaN to all trend columns
    df_htf.loc[warmup, ["trend_up_1h", "trend_down_1h", "trend_strength_1h"]] = float("nan")

    target = pd.DataFrame({"timestamp": pd.to_datetime(timestamps_utc, utc=True)})
    merged = pd.merge_asof(
        target.sort_values("timestamp"),
        df_htf[["timestamp", "trend_up_1h", "trend_down_1h", "trend_strength_1h"]].sort_values(
            "timestamp"
        ),
        on="timestamp",
        direction="backward",
    )
    return merged.reset_index(drop=True)


def print_trend_stats(df: pd.DataFrame) -> None:
    """Print a quick diagnostic of trend state distribution.

    Parameters
    ----------
    df : DataFrame returned by compute_trend_mask (must contain trend_up_1h etc.)
    """
    total = len(df)
    nan_count = int(df["trend_up_1h"].isna().sum())
    valid = df.dropna(subset=["trend_up_1h"])
    n_up = int((valid["trend_up_1h"] == 1.0).sum())
    n_down = int((valid["trend_down_1h"] == 1.0).sum())
    n_neutral = len(valid) - n_up - n_down

    print(f"Total bars  : {total:>8,}")
    print(f"NaN (warmup): {nan_count:>8,}  ({100 * nan_count / max(total, 1):.1f}%)")
    print(f"Uptrend     : {n_up:>8,}  ({100 * n_up / max(len(valid), 1):.1f}%)")
    print(f"Downtrend   : {n_down:>8,}  ({100 * n_down / max(len(valid), 1):.1f}%)")
    print(f"Neutral     : {n_neutral:>8,}  ({100 * n_neutral / max(len(valid), 1):.1f}%)")
    if "trend_strength_1h" in df.columns:
        s = valid["trend_strength_1h"].dropna()
        print(
            f"Strength    : mean={s.mean():.5f}  std={s.std():.5f}"
            f"  min={s.min():.5f}  max={s.max():.5f}"
        )
