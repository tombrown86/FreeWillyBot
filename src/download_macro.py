"""
Phase 6/7 — Fetch macro data from FRED.

CPI, unemployment, policy rate, 10Y yield.
API key from FRED_API_KEY environment variable.
Saves each series to data/raw/macro/.
"""

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env if present (FRED_API_KEY)
_env = PROJECT_ROOT / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env)
    except ImportError:
        pass

from src.config import TEST_START_DATE, TRAINING_START_DATE
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "macro"
LOG_FILE = PROJECT_ROOT / "data" / "logs" / "download_macro.log"

# FRED series: series_id -> (display_name, description)
FRED_SERIES = {
    "CPIAUCSL": ("CPI", "Consumer Price Index All Urban Consumers"),
    "UNRATE": ("UNEMPLOYMENT", "Unemployment Rate"),
    "FEDFUNDS": ("FED_FUNDS", "Federal Funds Effective Rate"),
    "DGS10": ("TREASURY_10Y", "10-Year Treasury Constant Maturity Rate"),
}


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


def download_macro_data(
    start: datetime | None = None,
    end: datetime | None = None,
    output_dir: Path | None = None,
    test_one: bool = False,
) -> list[Path]:
    """
    Download macro series from FRED. Requires FRED_API_KEY in environment.
    Each series saved as own CSV with date and value columns.
    """
    _setup_logging()
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        logging.error("FRED_API_KEY not set. Create key at https://fred.stlouisfed.org/docs/api/api_key.html")
        return []

    start = start or datetime.combine(TRAINING_START_DATE, datetime.min.time(), tzinfo=timezone.utc)
    end = end or datetime.now(timezone.utc)
    output_dir = output_dir or OUTPUT_DIR
    raw_dir = PROJECT_ROOT / "data" / "raw"
    assert str(output_dir.resolve()).startswith(str(raw_dir.resolve())), "Download must write to data/raw/ only"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from fredapi import Fred
    except ImportError:
        logging.error("fredapi not installed. Run: pip install fredapi")
        return []

    fred = Fred(api_key=api_key)
    saved_paths: list[Path] = []
    series_ids = list(FRED_SERIES.keys())

    if test_one:
        series_ids = series_ids[:1]
        logging.info("Testing one series: %s", series_ids[0])

    for series_id in series_ids:
        try:
            s = fred.get_series(series_id, start=start.date(), end=end.date())
            if s is None or s.empty:
                logging.warning("No data for %s", series_id)
                continue

            df = pd.DataFrame({"date": s.index, "value": s.values})
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            name, _ = FRED_SERIES[series_id]
            df["series_name"] = name

            out_path = output_dir / f"{name}.csv"
            df.to_csv(out_path, index=False)
            logging.info("Downloaded %s | %s | %d rows", name, series_id, len(df))
            saved_paths.append(out_path)
        except Exception as e:
            logging.error("Failed %s: %s", series_id, e)

    return saved_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download macro data from FRED")
    parser.add_argument("--test", action="store_true", help="Test one series only")
    args = parser.parse_args()

    download_macro_data(test_one=args.test)
