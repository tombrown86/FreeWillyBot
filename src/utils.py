"""
Shared utilities for FreeWillyBot.
"""

import logging
import time
from pathlib import Path

LOG = logging.getLogger(__name__)

# Minimum Parquet file size (footer is 8 bytes). Smaller = corrupt/truncated.
MIN_PARQUET_BYTES = 8


def load_processed_price(processed_price_dir: Path, symbol: str, bar_interval: str) -> "pd.DataFrame":
    """
    Load clean price bars. Prefer Parquet; on corrupt/tiny file, delete and fall back to CSV.
    Raises FileNotFoundError if neither exists.
    """
    import pandas as pd

    csv_path = processed_price_dir / f"{symbol}_{bar_interval}_clean.csv"
    parquet_path = processed_price_dir / f"{symbol}_{bar_interval}_clean.parquet"

    if parquet_path.exists():
        if parquet_path.stat().st_size < MIN_PARQUET_BYTES:
            LOG.warning(
                "Corrupt Parquet (size %d < %d): deleting %s",
                parquet_path.stat().st_size,
                MIN_PARQUET_BYTES,
                parquet_path,
            )
            parquet_path.unlink()
        else:
            try:
                return pd.read_parquet(parquet_path)
            except Exception as e:
                LOG.warning("Parquet read failed (%s): deleting %s, falling back to CSV", e, parquet_path)
                parquet_path.unlink()

    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Price file not found: {csv_path} or {parquet_path}")


def retry_with_backoff(
    func,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple = (Exception,),
) -> any:
    """
    Retry func with exponential backoff on transient failures.
    """
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return func()
        except exceptions as e:
            last_exc = e
            if attempt == max_attempts - 1:
                raise
            delay = base_delay * (2**attempt)
            LOG.warning("Attempt %d failed: %s. Retrying in %.1fs", attempt + 1, e, delay)
            time.sleep(delay)
    raise last_exc
