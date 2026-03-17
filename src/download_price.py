"""
Phase 5 — Fetch market price data.

For EUR/USD: Dukascopy historical data (common source for FX research).
For BTCUSDT: Binance klines (looped over date ranges in chunks).

Raw files saved to data/raw/price/. Never edit raw files manually.
Logs each download to data/logs/download_price.log.
"""

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    BAR_INTERVAL,
    DATA_SOURCE,
    SYMBOL,
    TEST_START_DATE,
    TRAINING_START_DATE,
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "price"
LOG_FILE = PROJECT_ROOT / "data" / "logs" / "download_price.log"

# Dukascopy limit per request (5min bars: ~288/day, 100 days = 28800)
DUKASCOPY_CHUNK_DAYS = 100
# Binance limit per request
BINANCE_KLINES_LIMIT = 1000


def _setup_logging() -> None:
    """Configure logging to file. Each download writes a line."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _get_dukascopy_interval() -> str:
    """Map config bar interval to Dukascopy constant."""
    mapping = {
        "1min": "INTERVAL_MIN_1",
        "5min": "INTERVAL_MIN_5",
        "10min": "INTERVAL_MIN_10",
        "15min": "INTERVAL_MIN_15",
        "30min": "INTERVAL_MIN_30",
        "1h": "INTERVAL_HOUR_1",
        "4h": "INTERVAL_HOUR_4",
        "1d": "INTERVAL_DAY_1",
    }
    key = BAR_INTERVAL.lower().replace(" ", "")
    if key not in mapping:
        raise ValueError(f"Unsupported bar interval for Dukascopy: {BAR_INTERVAL}")
    return mapping[key]


def _get_dukascopy_instrument() -> str:
    """Map config symbol to Dukascopy instrument constant."""
    symbol_map = {
        "EURUSD": "INSTRUMENT_FX_MAJORS_EUR_USD",
        "EUR/USD": "INSTRUMENT_FX_MAJORS_EUR_USD",
    }
    key = SYMBOL.upper().replace("/", "")
    if key not in symbol_map:
        raise ValueError(
            f"Dukascopy supports EUR/USD for FX. Symbol '{SYMBOL}' not mapped."
        )
    return symbol_map[key]


def _binance_interval() -> str:
    """Map config bar interval to Binance kline interval."""
    mapping = {
        "1min": "1m",
        "5min": "5m",
        "15min": "15m",
        "30min": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }
    key = BAR_INTERVAL.lower().replace(" ", "")
    return mapping.get(key, "5m")


def fetch_dukascopy_chunk(
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Fetch OHLC bars from Dukascopy for one date range."""
    import dukascopy_python
    from dukascopy_python import instruments

    # Suppress dukascopy's verbose INFO logs
    logging.getLogger("dukascopy_python").setLevel(logging.WARNING)

    interval = getattr(dukascopy_python, _get_dukascopy_interval())
    instrument = getattr(instruments, _get_dukascopy_instrument())

    df = dukascopy_python.fetch(
        instrument,
        interval,
        dukascopy_python.OFFER_SIDE_BID,
        start,
        end,
    )
    return df


def fetch_binance_klines(
    start: datetime,
    end: datetime,
) -> list[pd.DataFrame]:
    """
    Fetch klines from Binance in chunks of 1000 (API limit).
    Returns list of DataFrames, one per chunk.
    """
    import requests

    symbol = SYMBOL.upper().replace("/", "")
    interval = _binance_interval()
    url = "https://api.binance.com/api/v3/klines"

    chunks = []
    current_start = start

    while current_start < end:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(current_start.timestamp() * 1000),
            "endTime": int(end.timestamp() * 1000),
            "limit": BINANCE_KLINES_LIMIT,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json()

        if not rows:
            break

        df = pd.DataFrame(
            rows,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
        chunks.append(df)

        last_ts = int(rows[-1][0])
        current_start = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
        current_start += timedelta(minutes=1)  # move past last candle

    return chunks


def fetch_yfinance(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch OHLC from yfinance (fallback)."""
    import yfinance as yf

    symbol_map = {"EURUSD": "EURUSD=X", "EUR/USD": "EURUSD=X", "BTCUSDT": "BTC-USD"}
    key = SYMBOL.upper().replace("/", "")
    yf_symbol = symbol_map.get(key, f"{SYMBOL}=X")

    interval_map = {"1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m", "1h": "1h", "1d": "1d"}
    yf_interval = interval_map.get(BAR_INTERVAL.lower().replace(" ", ""), "1h")

    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(start=start.date(), end=end.date(), interval=yf_interval, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data for {yf_symbol}")

    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]]
    df.index.name = "timestamp"
    df = df.reset_index()
    return df


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure lowercase columns and timestamp index/column."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" not in df.columns and df.index.name == "timestamp":
        df = df.reset_index()
    return df


def _save_chunk(df: pd.DataFrame, out_path: Path, start: datetime, end: datetime) -> None:
    """Save chunk to CSV and log."""
    df = _normalize_df(df)
    df.to_csv(out_path, index=False)
    logging.info(
        "Downloaded %s | %s | %s | %s -> %s | %d bars",
        SYMBOL,
        BAR_INTERVAL,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        out_path.name,
        len(df),
    )


def download_price_data(
    start: datetime | None = None,
    end: datetime | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """
    Download price data for the configured symbol and timeframe.

    Dukascopy: chunks by ~100 days, saves each to data/raw/price/.
    Binance: fetches klines in 1000-bar chunks, saves each chunk.
    """
    _setup_logging()
    start = start or datetime.combine(TRAINING_START_DATE, datetime.min.time(), tzinfo=timezone.utc)
    end = end or datetime.now(timezone.utc)
    output_dir = output_dir or OUTPUT_DIR
    raw_dir = PROJECT_ROOT / "data" / "raw"
    assert str(output_dir.resolve()).startswith(str(raw_dir.resolve())), "Download must write to data/raw/ only"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    if DATA_SOURCE == "binance" and SYMBOL.upper().replace("/", "") == "BTCUSDT":
        # Binance: loop over date ranges, fetch klines in chunks
        chunks = fetch_binance_klines(start, end)
        for i, df in enumerate(chunks):
            chunk_start = df["timestamp"].min()
            chunk_end = df["timestamp"].max()
            out_path = output_dir / f"{SYMBOL}_{BAR_INTERVAL}_{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}_chunk{i:04d}.csv"
            _save_chunk(df, out_path, chunk_start, chunk_end)
            saved_paths.append(out_path)

    elif DATA_SOURCE == "dukascopy" and SYMBOL.upper().replace("/", "") in ("EURUSD",):
        # Dukascopy: chunk by days to stay under API limit
        current = start
        chunk_idx = 0
        while current < end:
            chunk_end = min(current + timedelta(days=DUKASCOPY_CHUNK_DAYS), end)
            df = fetch_dukascopy_chunk(current, chunk_end)
            if df.empty:
                current = chunk_end
                continue
            df = _normalize_df(df.reset_index())
            first_ts = df["timestamp"].min()
            last_ts = df["timestamp"].max()
            out_path = output_dir / f"{SYMBOL}_{BAR_INTERVAL}_{first_ts:%Y%m%d}_{last_ts:%Y%m%d}_chunk{chunk_idx:04d}.csv"
            _save_chunk(df, out_path, first_ts, last_ts)
            saved_paths.append(out_path)
            current = chunk_end
            chunk_idx += 1

    else:
        # yfinance fallback
        df = fetch_yfinance(start, end)
        out_path = output_dir / f"{SYMBOL}_{BAR_INTERVAL}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
        _save_chunk(df, out_path, start, end)
        saved_paths.append(out_path)

    return saved_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download price data")
    parser.add_argument("--test", action="store_true", help="Short test range (1 week)")
    parser.add_argument("--full", action="store_true", help="Full range from config")
    args = parser.parse_args()

    if args.test:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=7)
        download_price_data(start=start, end=end)
    elif args.full:
        download_price_data()
    else:
        # Default: full range
        download_price_data()
