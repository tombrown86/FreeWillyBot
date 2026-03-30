"""
Event study utility — EUR/USD post-news drift analysis.

For each high-impact macro event (CPI, NFP, FOMC, ECB, BoE) in the calendar,
this module:
  1. Locates the nearest 5-minute bar at or after the event timestamp
  2. Computes forward returns at T+1, T+3, T+6, T+12 bars
  3. Records the initial move direction (sign of ret_1)
  4. Computes pre-event volatility baseline (std of last N bar returns before T)
  5. Attaches the 4h MA10 trend state at event time (from src.trend_filter)

Public API
----------
build_event_dataset(fwd_bars, pre_vol_window, add_trend) -> pd.DataFrame
    Returns one row per event with forward returns and context.
    Saves to data/events/event_study_raw.csv.

load_event_dataset() -> pd.DataFrame
    Load the pre-built CSV (must call build_event_dataset first).
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVENTS_CSV   = PROJECT_ROOT / "data" / "raw" / "macro" / "event_calendar.csv"
PRICE_PATH   = PROJECT_ROOT / "data" / "processed" / "price" / "EURUSD_5min_clean.parquet"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "events"
OUTPUT_RAW   = OUTPUT_DIR / "event_study_raw.csv"

# Max minutes after the scheduled event time to accept a bar as "T"
BAR_MATCH_WINDOW_MIN: int = 30


def build_event_dataset(
    fwd_bars: list[int] | None = None,
    pre_vol_window: int = 12,
    add_trend: bool = True,
) -> pd.DataFrame:
    """Build the event-level dataset with forward returns and context.

    Parameters
    ----------
    fwd_bars       : forward horizons in bars (default [1, 3, 6, 12])
    pre_vol_window : bars before T used to compute pre-event volatility baseline
    add_trend      : attach 4h MA10 trend state at event time

    Returns
    -------
    DataFrame with one row per event that has full price coverage.
    Also saves to data/events/event_study_raw.csv.
    """
    if fwd_bars is None:
        fwd_bars = [1, 3, 6, 12]

    # ── Load inputs ────────────────────────────────────────────────────────
    events = pd.read_csv(EVENTS_CSV)
    events["event_timestamp_utc"] = pd.to_datetime(events["event_timestamp_utc"], utc=True)
    events = events[events["importance"] == "high"].reset_index(drop=True)

    price = pd.read_parquet(PRICE_PATH, columns=["timestamp", "close"])
    price["timestamp"] = pd.to_datetime(price["timestamp"], utc=True)
    price = price.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    # Pre-compute bar-level returns for pre-event volatility
    price["ret_1bar"] = price["close"].pct_change()

    # ── Batch trend lookup ────────────────────────────────────────────────
    trend_df = None
    if add_trend:
        from src.trend_filter import compute_trend_mask
        trend_df = compute_trend_mask(
            events["event_timestamp_utc"],
            resample_period="4h",
            ma_window=10,
        )

    # ── Extract per-event rows ────────────────────────────────────────────
    max_fwd = max(fwd_bars)
    rows: list[dict] = []
    skipped = 0

    for i, ev in events.iterrows():
        ev_ts = ev["event_timestamp_utc"]

        # Find nearest bar at or after event time within the match window
        cutoff = ev_ts + pd.Timedelta(minutes=BAR_MATCH_WINDOW_MIN)
        candidates = price[(price["timestamp"] >= ev_ts) & (price["timestamp"] <= cutoff)]
        if candidates.empty:
            skipped += 1
            continue

        t_idx = candidates.index[0]          # integer index into price DataFrame

        # Need enough forward bars
        if t_idx + max_fwd >= len(price):
            skipped += 1
            continue

        # Need enough backward bars for pre-vol
        if t_idx < pre_vol_window:
            skipped += 1
            continue

        t_close = float(price.loc[t_idx, "close"])
        t_ts    = price.loc[t_idx, "timestamp"]

        row: dict = {
            "event_name":      ev["event_name"],
            "event_time_utc":  ev_ts,
            "bar_time_utc":    t_ts,
            "bar_lag_min":     (t_ts - ev_ts).total_seconds() / 60.0,
            "close_at_T":      t_close,
        }

        # Forward returns (fractional)
        for h in fwd_bars:
            fwd_close = float(price.loc[t_idx + h, "close"])
            row[f"ret_{h}"] = (fwd_close - t_close) / t_close

        # Initial move direction: sign of ret_1
        row["initial_move"] = int(np.sign(row["ret_1"])) if "ret_1" in row else 0

        # Pre-event volatility baseline
        pre_rets = price.loc[t_idx - pre_vol_window : t_idx - 1, "ret_1bar"].dropna()
        row["pre_vol"] = float(pre_rets.std()) if len(pre_rets) >= 3 else float("nan")

        # Trend state at event time (from the batch lookup)
        if trend_df is not None:
            tr = trend_df.iloc[i]
            row["trend_label"]    = (
                "up"      if tr["trend_up_1h"]   == 1.0
                else "down" if tr["trend_down_1h"] == 1.0
                else "warmup" if pd.isna(tr["trend_up_1h"])
                else "neutral"
            )
            row["trend_strength"] = float(tr["trend_strength_1h"]) if pd.notna(tr.get("trend_strength_1h")) else 0.0
        else:
            row["trend_label"]    = "n/a"
            row["trend_strength"] = 0.0

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"WARNING: No events extracted (skipped={skipped})")
        return df

    # ── Derived columns ───────────────────────────────────────────────────
    # Directional returns: positive when the initial move continued (momentum)
    # and negative when it reversed.
    for h in fwd_bars:
        df[f"dir_ret_{h}"] = df[f"ret_{h}"] * df["initial_move"]

    # Label: momentum (drift continued) or reversion (drift reversed)
    df["behavior_6"] = df["dir_ret_6"].apply(
        lambda x: "drift" if x > 0 else ("reversion" if x < 0 else "flat")
    )

    # Trend alignment: does the 4h trend agree with the initial move direction?
    df["trend_agrees"] = (
        ((df["initial_move"] == 1)  & (df["trend_label"] == "up")) |
        ((df["initial_move"] == -1) & (df["trend_label"] == "down"))
    )

    # ── Save ──────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_RAW, index=False)
    print(f"Saved {len(df)} events → {OUTPUT_RAW}  (skipped {skipped})")
    return df


def load_event_dataset() -> pd.DataFrame:
    """Load the pre-built event study CSV."""
    if not OUTPUT_RAW.exists():
        raise FileNotFoundError(f"Run build_event_dataset() first — {OUTPUT_RAW} not found")
    df = pd.read_csv(OUTPUT_RAW)
    df["event_time_utc"] = pd.to_datetime(df["event_time_utc"], utc=True)
    df["bar_time_utc"]   = pd.to_datetime(df["bar_time_utc"],   utc=True)
    return df
