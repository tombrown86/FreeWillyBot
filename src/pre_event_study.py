"""
Pre-event drift study utility — EUR/USD pre-announcement drift analysis.

For each high-impact macro event at scheduled time T, this module computes
returns in the windows BEFORE the release using only information available
before T. No lookahead: every bar index used is strictly < T bar index.

Windows computed
----------------
  T-12 → T-1  (55 min, "60-min window") : ret_12_to_1
  T-6  → T-1  (25 min, "30-min window") : ret_6_to_1
  T-12 → T-6  (25 min, "early half")    : ret_12_to_6   ← momentum signal input

Signal columns
--------------
  drift_dir_6     = sign(ret_6_to_1)    — direction of 30-min pre-event window
  drift_dir_12    = sign(ret_12_to_1)   — direction of 60-min pre-event window
  momentum_signal = sign(ret_12_to_6)   — does early drift predict the late window?
                                           independent of ret_6_to_1 (no window overlap)

Context columns
---------------
  pre_vol         = std of bar returns T-24 to T-13 (background vol, no overlap)
  trend_label     = 4h MA10 at T-6 (backward-filled, safe)
  trend_strength  = close/MA-1 at T-6

Public API
----------
build_pre_event_dataset(pre_windows, pre_vol_window, add_trend) -> pd.DataFrame
load_pre_event_dataset() -> pd.DataFrame
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVENTS_CSV   = PROJECT_ROOT / "data" / "raw" / "macro" / "event_calendar.csv"
PRICE_PATH   = PROJECT_ROOT / "data" / "processed" / "price" / "EURUSD_5min_clean.parquet"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "events"
OUTPUT_RAW   = OUTPUT_DIR / "pre_event_study_raw.csv"

BAR_MATCH_WINDOW_MIN: int = 30   # max minutes after scheduled event to accept bar as T


def build_pre_event_dataset(
    pre_windows: list[int] | None = None,
    pre_vol_window: int = 24,
    add_trend: bool = True,
) -> pd.DataFrame:
    """Build one row per event with pre-event drift features.

    Parameters
    ----------
    pre_windows    : look-back bar counts to compute (default [6, 12])
                     ret_{A}_to_{B} is computed for the range A→B
    pre_vol_window : bars for background volatility (T-24 to T-{pre_vol_window+1})
                     window never overlaps the drift measurement window
    add_trend      : attach 4h MA10 trend state at T-6

    Returns
    -------
    DataFrame, also saved to data/events/pre_event_study_raw.csv
    """
    if pre_windows is None:
        pre_windows = [6, 12]

    max_lookback = max(pre_windows) + pre_vol_window   # bars needed before T

    # ── Load events ───────────────────────────────────────────────────────
    events = pd.read_csv(EVENTS_CSV)
    events["event_timestamp_utc"] = pd.to_datetime(events["event_timestamp_utc"], utc=True)
    events = events[events["importance"] == "high"].reset_index(drop=True)

    # ── Load price ────────────────────────────────────────────────────────
    price = pd.read_parquet(PRICE_PATH, columns=["timestamp", "close"])
    price["timestamp"] = pd.to_datetime(price["timestamp"], utc=True)
    price = price.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    price["ret_1bar"] = price["close"].pct_change()

    # ── Trend at T-6 for all events (batch) ───────────────────────────────
    trend_df = None
    if add_trend:
        from src.trend_filter import compute_trend_mask
        # We want the trend state at T-6 (30 min before event), not AT T
        # We'll attach it per-event below after finding T-6 bar timestamp
        # Store the full trend series separately and merge by timestamp
        t6_timestamps = []
        t6_event_idx = []
        # We'll compute after finding T bar indices

    # ── Extract per-event rows ────────────────────────────────────────────
    rows: list[dict] = []
    skipped = 0

    for i, ev in events.iterrows():
        ev_ts = ev["event_timestamp_utc"]

        # Find the T bar (first bar at/after event time within 30-min window)
        cutoff = ev_ts + pd.Timedelta(minutes=BAR_MATCH_WINDOW_MIN)
        candidates = price[(price["timestamp"] >= ev_ts) & (price["timestamp"] <= cutoff)]
        if candidates.empty:
            skipped += 1
            continue

        t_idx = candidates.index[0]   # integer position in price DataFrame

        # Require enough history before T
        if t_idx < max_lookback:
            skipped += 1
            continue

        # Require at least 1 bar after T (sanity check)
        if t_idx >= len(price) - 1:
            skipped += 1
            continue

        t_ts    = price.loc[t_idx, "timestamp"]
        t_close = float(price.loc[t_idx, "close"])

        row: dict = {
            "event_name":       ev["event_name"],
            "country":          ev["country"],
            "event_time_utc":   ev_ts,
            "bar_time_utc":     t_ts,
            "bar_lag_min":      (t_ts - ev_ts).total_seconds() / 60.0,
            "close_at_T":       t_close,
        }

        # ── Pre-event returns ─────────────────────────────────────────────
        # T-1: last bar before the event
        t_minus_1_close  = float(price.loc[t_idx - 1,  "close"])
        t_minus_6_close  = float(price.loc[t_idx - 6,  "close"])
        t_minus_12_close = float(price.loc[t_idx - 12, "close"])

        row["ret_6_to_1"]  = (t_minus_1_close  - t_minus_6_close)  / t_minus_6_close
        row["ret_12_to_1"] = (t_minus_1_close  - t_minus_12_close) / t_minus_12_close
        row["ret_12_to_6"] = (t_minus_6_close  - t_minus_12_close) / t_minus_12_close

        # ── Signal columns ────────────────────────────────────────────────
        row["drift_dir_6"]    = int(np.sign(row["ret_6_to_1"]))
        row["drift_dir_12"]   = int(np.sign(row["ret_12_to_1"]))
        row["momentum_signal"] = int(np.sign(row["ret_12_to_6"]))  # early-half direction

        # ── Pre-event background volatility (no overlap with drift window) ─
        # Use bars T-pre_vol_window to T-max(pre_windows)-1  (before the 60-min window)
        vol_start = t_idx - pre_vol_window - max(pre_windows)
        vol_end   = t_idx - max(pre_windows) - 1
        pre_rets = price.loc[vol_start:vol_end, "ret_1bar"].dropna()
        row["pre_vol"] = float(pre_rets.std()) if len(pre_rets) >= 5 else float("nan")

        # Bar timestamps for trend lookup (T-6)
        row["t6_ts"] = price.loc[t_idx - 6, "timestamp"]

        rows.append(row)

    if not rows:
        print(f"WARNING: No events extracted (skipped={skipped})")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ── Attach 4h MA10 trend state at T-6 ────────────────────────────────
    if add_trend:
        from src.trend_filter import compute_trend_mask
        trend_df = compute_trend_mask(
            df["t6_ts"],
            resample_period="4h",
            ma_window=10,
        )
        df["trend_label"] = trend_df.apply(
            lambda r: (
                "up"      if r["trend_up_1h"]   == 1.0
                else "down" if r["trend_down_1h"] == 1.0
                else "warmup" if pd.isna(r["trend_up_1h"])
                else "neutral"
            ),
            axis=1,
        )
        df["trend_strength"] = trend_df["trend_strength_1h"].fillna(0.0)
    else:
        df["trend_label"]    = "n/a"
        df["trend_strength"] = 0.0

    # Drop the helper timestamp column
    df = df.drop(columns=["t6_ts"])

    # ── Save ──────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_RAW, index=False)
    print(f"Saved {len(df)} events → {OUTPUT_RAW}  (skipped {skipped})")
    return df


def load_pre_event_dataset() -> pd.DataFrame:
    """Load the pre-built CSV. Call build_pre_event_dataset() first."""
    if not OUTPUT_RAW.exists():
        raise FileNotFoundError(f"Run build_pre_event_dataset() first — {OUTPUT_RAW} not found")
    df = pd.read_csv(OUTPUT_RAW)
    df["event_time_utc"] = pd.to_datetime(df["event_time_utc"], utc=True)
    df["bar_time_utc"]   = pd.to_datetime(df["bar_time_utc"],   utc=True)
    return df
