"""
Phase 9 — Build clean bars and aligned tables.

1. Load raw price data, sort, dedupe, resample to 5min, save to data/processed/price/.
2. Build one aligned time index from clean bars.
3. Reindex cross-asset and macro onto main time index with forward-fill.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BAR_INTERVAL, SYMBOL, TRAINING_START_DATE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PRICE_DIR = PROJECT_ROOT / "data" / "raw" / "price"
PROCESSED_PRICE_DIR = PROJECT_ROOT / "data" / "processed" / "price"
RAW_CROSS_ASSET_DIR = PROJECT_ROOT / "data" / "raw" / "cross_asset"
RAW_MACRO_DIR = PROJECT_ROOT / "data" / "raw" / "macro"
ALIGNED_DIR = PROJECT_ROOT / "data" / "processed" / "aligned"
LOG_FILE = PROJECT_ROOT / "data" / "logs" / "build_price_bars.log"


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


def load_raw_price_data() -> pd.DataFrame:
    """Load all raw price CSVs, concatenate, sort, remove duplicates."""
    files = sorted(RAW_PRICE_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw price files in {RAW_PRICE_DIR}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Normalize column names
        df.columns = [c.lower().strip() for c in df.columns]
        if "timestamp" not in df.columns and "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    combined = combined.sort_values("timestamp")
    combined = combined.drop_duplicates(subset=["timestamp"], keep="first")

    return combined


def resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample or confirm bars at exactly 5-minute frequency. Compute OHLCV."""
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="first")]

    # Resample to 5min: first=open, max=high, min=low, last=close, sum=volume
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    # Only include columns that exist
    agg = {k: v for k, v in agg.items() if k in df.columns}
    resampled = df.resample("5min").agg(agg).dropna(how="all")

    # Drop rows where OHLC are all NaN (no trading)
    ohlc_cols = [c for c in ["open", "high", "low", "close"] if c in resampled.columns]
    if ohlc_cols:
        resampled = resampled.dropna(subset=ohlc_cols, how="all")

    return resampled.reset_index()


def build_clean_bars() -> pd.DataFrame:
    """Load raw price, clean, resample to 5min, save to data/processed/price/."""
    _setup_logging()
    PROCESSED_PRICE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_price_data()
    logging.info("Loaded %d raw rows from %d files", len(df), len(list(RAW_PRICE_DIR.glob("*.csv"))))

    df = resample_to_5min(df)

    # Phase 18: Drop rows with zero or negative OHLC
    ohlc_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    if ohlc_cols:
        invalid = (df[ohlc_cols] <= 0).any(axis=1)
        if invalid.any():
            n_dropped = invalid.sum()
            logging.warning("Dropping %d rows with zero/negative OHLC", n_dropped)
            df = df[~invalid].reset_index(drop=True)
    logging.info("Resampled to %d clean 5min bars", len(df))

    # Stamp provenance so downstream consumers know these bars came from Dukascopy.
    # The live tail pipeline overwrites recent rows with cTrader bars (bar_source="ctrader")
    # without touching this file — training data stays pure Dukascopy.
    df["bar_source"] = "dukascopy"

    out_path = PROCESSED_PRICE_DIR / f"{SYMBOL}_{BAR_INTERVAL}_clean.csv"
    parquet_path = PROCESSED_PRICE_DIR / f"{SYMBOL}_{BAR_INTERVAL}_clean.parquet"
    df.to_csv(out_path, index=False)
    df.to_parquet(parquet_path, index=False)
    logging.info("Saved clean bars to %s and %s", out_path.name, parquet_path.name)

    # Phase 18: Versioned copy
    from datetime import datetime, timezone
    version = datetime.now(timezone.utc).strftime("%Y%m%d")
    versions_dir = PROCESSED_PRICE_DIR / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    versioned_path = versions_dir / f"{SYMBOL}_{BAR_INTERVAL}_clean_{version}.parquet"
    df.to_parquet(versioned_path, index=False)
    logging.info("Saved versioned copy to %s", versioned_path.name)

    return df


def build_aligned_tables(price_df: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame]:
    """
    Build main time index from price bars. Reindex cross-asset and macro onto it.
    Forward-fill slow series (macro, cross-asset) appropriately.
    """
    time_index = pd.DatetimeIndex(price_df["timestamp"]).unique().sort_values()
    time_index.name = "timestamp"

    # Load cross-asset (daily): each file has date, value; filename = series name.
    # Daily close for date D is available only after market close; shift +1 day so value D is used for bars on D+1.
    cross_series = []
    for f in sorted(RAW_CROSS_ASSET_DIR.glob("*.csv")):
        df = pd.read_csv(f)
        df.columns = [c.lower().strip() for c in df.columns]
        df["timestamp"] = pd.to_datetime(df["date"], utc=True)
        name = f.stem
        s = df.set_index("timestamp")["value"].rename(name)
        s.index = s.index + pd.Timedelta(days=1)  # Release lag: D's close available D+1
        cross_series.append(s)

    if cross_series:
        cross_combined = pd.concat(cross_series, axis=1, join="outer")
        cross_aligned = cross_combined.reindex(time_index, method="ffill")
    else:
        cross_aligned = pd.DataFrame(index=time_index)

    # Load macro (monthly/daily): use release dates to avoid lookahead.
    # CPI: match to event_calendar CPI release times; fallback ref_date + 45 days.
    # FED_FUNDS, TREASURY_10Y, UNEMPLOYMENT: daily, shift +1 day (released same day; 1d conservative).
    macro_aligned = pd.DataFrame(index=time_index)
    events_path = RAW_MACRO_DIR / "event_calendar.csv"
    cpi_releases = {}
    if events_path.exists():
        ev = pd.read_csv(events_path)
        ev["event_timestamp_utc"] = pd.to_datetime(ev["event_timestamp_utc"], utc=True)
        cpi_ev = ev[ev["event_name"] == "CPI"].sort_values("event_timestamp_utc")
        for _, row in cpi_ev.iterrows():
            ts = row["event_timestamp_utc"]
            key = (ts.year, ts.month)
            if key not in cpi_releases:
                cpi_releases[key] = ts
    # (year, month) of release -> timestamp; CPI for ref month M releases in month M+1

    for f in sorted(RAW_MACRO_DIR.glob("*.csv")):
        if f.name == "event_calendar.csv":
            continue
        df = pd.read_csv(f)
        df.columns = [c.lower().strip() for c in df.columns]
        df["timestamp"] = pd.to_datetime(df["date"], utc=True)
        name = df["series_name"].iloc[0] if "series_name" in df.columns else f.stem
        s = df.set_index("timestamp")["value"].rename(name).dropna()

        if name == "CPI" and cpi_releases:
            # CPI: reference month M releases in M+1; map each row to release timestamp
            release_ts = []
            for ref_ts in s.index:
                ref_dt = ref_ts.to_pydatetime()
                release_key = (ref_dt.year, ref_dt.month + 1) if ref_dt.month < 12 else (ref_dt.year + 1, 1)  # M releases in M+1
                ts = cpi_releases.get(release_key, ref_ts + pd.Timedelta(days=45))
                release_ts.append(ts if isinstance(ts, pd.Timestamp) else pd.Timestamp(ts, tz="UTC"))
            s_release = pd.Series(s.values, index=pd.DatetimeIndex(release_ts, tz="UTC")).sort_index()
        else:
            # Daily series: shift +1 day
            s.index = s.index + pd.Timedelta(days=1)
            s_release = s

        macro_aligned[name] = s_release.reindex(time_index, method="ffill")
    return time_index, cross_aligned, macro_aligned


def run() -> None:
    """Build clean bars and aligned tables. Never writes to data/raw/."""
    _setup_logging()
    assert str(PROCESSED_PRICE_DIR).startswith(str(PROJECT_ROOT / "data" / "processed")), "Must write to processed, not raw"
    assert str(ALIGNED_DIR).startswith(str(PROJECT_ROOT / "data" / "processed")), "Must write to processed, not raw"
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Clean price bars
    price_df = build_clean_bars()

    # 2. Build aligned tables
    time_index, cross_aligned, macro_aligned = build_aligned_tables(price_df)

    # Save aligned tables (CSV and Parquet)
    cross_aligned = cross_aligned.reset_index()
    cross_aligned.to_csv(ALIGNED_DIR / "cross_asset_aligned.csv", index=False)
    cross_aligned.to_parquet(ALIGNED_DIR / "cross_asset_aligned.parquet", index=False)
    logging.info("Saved cross-asset aligned: %d rows, %d cols", len(cross_aligned), len(cross_aligned.columns) - 1)

    macro_aligned = macro_aligned.reset_index()
    macro_aligned.to_csv(ALIGNED_DIR / "macro_aligned.csv", index=False)
    macro_aligned.to_parquet(ALIGNED_DIR / "macro_aligned.parquet", index=False)
    logging.info("Saved macro aligned: %d rows, %d cols", len(macro_aligned), len(macro_aligned.columns) - 1)

    # Save main time index for reference
    time_df = pd.DataFrame({"timestamp": time_index})
    time_df.to_csv(ALIGNED_DIR / "time_index.csv", index=False)
    time_df.to_parquet(ALIGNED_DIR / "time_index.parquet", index=False)
    logging.info("Saved time index: %d bars", len(time_index))

    # Phase 18: Versioned copies for aligned tables
    from datetime import datetime, timezone
    version = datetime.now(timezone.utc).strftime("%Y%m%d")
    for name, df_save in [("cross_asset_aligned", cross_aligned), ("macro_aligned", macro_aligned), ("time_index", time_df)]:
        versions_dir = ALIGNED_DIR / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)
        versioned_path = versions_dir / f"{name}_{version}.parquet"
        df_save.to_parquet(versioned_path, index=False)
    logging.info("Saved versioned aligned tables")


if __name__ == "__main__":
    run()
