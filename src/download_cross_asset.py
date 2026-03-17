"""
Phase 6 — Fetch cross-asset data.

S&P 500, VIX, gold, oil from yfinance.
Saves to data/raw/cross_asset/.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TEST_START_DATE, TRAINING_START_DATE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "cross_asset"
LOG_FILE = PROJECT_ROOT / "data" / "logs" / "download_cross_asset.log"

# Cross-asset symbols (yfinance)
CROSS_ASSET_SYMBOLS = {
    "^GSPC": "SP500",
    "^VIX": "VIX",
    "GC=F": "GOLD",
    "CL=F": "OIL",
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


def download_cross_asset_data(
    start: datetime | None = None,
    end: datetime | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """Download ^GSPC, ^VIX, GC=F, CL=F from yfinance. Save each to data/raw/cross_asset/."""
    _setup_logging()
    start = start or datetime.combine(TRAINING_START_DATE, datetime.min.time(), tzinfo=timezone.utc)
    end = end or datetime.now(timezone.utc)
    output_dir = output_dir or OUTPUT_DIR
    raw_dir = PROJECT_ROOT / "data" / "raw"
    assert str(output_dir.resolve()).startswith(str(raw_dir.resolve())), "Download must write to data/raw/ only"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for yf_symbol, name in CROSS_ASSET_SYMBOLS.items():
        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(start=start.date(), end=end.date(), interval="1d", auto_adjust=True)
            if df.empty:
                logging.warning("No data for %s (%s)", name, yf_symbol)
                continue

            df = df.reset_index()
            df = df.rename(columns={"Date": "date", "Close": "value"})
            df = df[["date", "value"]]
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

            out_path = output_dir / f"{name}.csv"
            df.to_csv(out_path, index=False)
            logging.info("Downloaded %s | %s | %d rows", name, yf_symbol, len(df))
            saved_paths.append(out_path)
        except Exception as e:
            logging.error("Failed %s (%s): %s", name, yf_symbol, e)

    return saved_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download cross-asset data")
    parser.add_argument("--test", action="store_true", help="Short test range (1 week)")
    args = parser.parse_args()

    if args.test:
        from datetime import timedelta

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=7)
        download_cross_asset_data(start=start, end=end)
    else:
        download_cross_asset_data()
