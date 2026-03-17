"""
Phase 8 — Fetch news data from GDELT GKG raw files.

Uses GKG daily files (not DOC 2.0) for bulk historical coverage.
Downloads YYYYMMDD.gkg.csv.zip, parses, filters by FX themes, extracts tone.
Output: timestamp (daily), sentiment_score (GKG tone normalized to [-1,1]), fx_theme_count.
"""

import logging
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
import requests

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TRAINING_START_DATE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "news"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "news"
LOG_FILE = PROJECT_ROOT / "data" / "logs" / "download_news.log"

# GKG base URL for daily files
GKG_BASE_URL = "http://data.gdeltproject.org/gkg/{date}.gkg.csv.zip"

# FX-related theme keywords (match case-insensitive against THEMES column)
# GDELT themes: ECONOMICS, CURRENCY, EXCHANGE_RATE, WB/UN/IMF codes, etc.
FX_THEME_KEYWORDS = [
    "eur", "euro", "ecb", "eurozone", "euro_zone",
    "usd", "dollar", "fed", "federal_reserve", "fomc",
    "inflation", "interest_rate", "rate_hike", "rate_cut",
    "recession", "forex", "fx", "currency", "exchange_rate",
    "monetary_policy", "central_bank", "gdp", "employment",
    "economic", "economics", "exchange", "reserve", "trade",
]

# GKG column names (header row exists in files)
GKG_COLS = ["DATE", "NUMARTS", "COUNTS", "THEMES", "LOCATIONS", "PERSONS", "ORGANIZATIONS", "TONE", "CAMEOEVENTIDS", "SOURCES", "SOURCEURLS"]

# Request settings
REQUEST_TIMEOUT = 180  # large files (25-50 MB)
MAX_RETRIES = 3
REQUEST_HEADERS = {"User-Agent": "FreeWillyBot-GKG/1.0 (research)"}
MAX_WORKERS = 5  # parallel downloads


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


def _matches_fx_themes(themes_str: str) -> bool:
    """True if any theme matches FX keywords."""
    if pd.isna(themes_str) or not str(themes_str).strip():
        return False
    themes_lower = str(themes_str).lower()
    for kw in FX_THEME_KEYWORDS:
        if kw in themes_lower:
            return True
    return False


def _parse_tone(tone_str: str) -> float | None:
    """
    Parse GKG TONE field. First value is document tone (-100 to +100).
    Returns normalized score in [-1, 1] or None if unparseable.
    """
    if pd.isna(tone_str) or not str(tone_str).strip():
        return None
    parts = str(tone_str).split(",")
    if not parts:
        return None
    try:
        raw = float(parts[0].strip())
        # Clamp and normalize to [-1, 1]
        clamped = max(-100, min(100, raw))
        return clamped / 100.0
    except (ValueError, TypeError):
        return None


def _download_gkg_day(date: datetime) -> pd.DataFrame | None:
    """Download and parse one day's GKG file. Returns DataFrame or None on failure."""
    date_str = date.strftime("%Y%m%d")
    url = GKG_BASE_URL.format(date=date_str)

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=REQUEST_HEADERS)
            if resp.status_code == 404:
                # Date not yet available (e.g. future or not yet published)
                return None
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            logging.warning("GKG download failed %s (attempt %d/%d): %s", date_str, attempt + 1, MAX_RETRIES, e)
            if attempt == MAX_RETRIES - 1:
                return None
            continue

    try:
        with zipfile.ZipFile(BytesIO(resp.content), "r") as zf:
            names = zf.namelist()
            if not names:
                return None
            with zf.open(names[0]) as f:
                content = f.read().decode("utf-8", errors="replace")
    except Exception as e:
        logging.warning("GKG unzip/parse failed %s: %s", date_str, e)
        return None

    try:
        df = pd.read_csv(StringIO(content), sep="\t", header=0, dtype=str, on_bad_lines="skip")
    except Exception as e:
        logging.warning("GKG CSV parse failed %s: %s", date_str, e)
        return None

    # Normalize column names
    df.columns = [c.strip().upper() for c in df.columns]
    if "THEMES" not in df.columns or "TONE" not in df.columns:
        logging.warning("GKG %s missing THEMES or TONE columns", date_str)
        return None

    return df


def _process_gkg_day(df: pd.DataFrame, date: datetime) -> pd.DataFrame:
    """Filter FX rows, parse tone, aggregate by date."""
    mask = df["THEMES"].apply(_matches_fx_themes)
    fx = df.loc[mask].copy()

    if fx.empty:
        return pd.DataFrame()

    fx["sentiment_score"] = fx["TONE"].apply(_parse_tone)
    fx = fx.dropna(subset=["sentiment_score"])

    if fx.empty:
        return pd.DataFrame()

    ts = datetime(date.year, date.month, date.day, 0, 0, 0, tzinfo=timezone.utc)
    agg = pd.DataFrame([{
        "timestamp": ts,
        "sentiment_score": fx["sentiment_score"].mean(),
        "fx_theme_count": len(fx),
    }])
    return agg


def _download_and_process_day(current: datetime) -> tuple[datetime, pd.DataFrame | None]:
    """Download one day, process, return (date, agg_df or None)."""
    df = _download_gkg_day(current)
    if df is None or df.empty:
        return current, None
    agg = _process_gkg_day(df, current)
    return current, agg if not agg.empty else None


def download_news_data(
    start: datetime | None = None,
    end: datetime | None = None,
) -> tuple[list[Path], Path | None]:
    """
    Download GKG files by day, filter FX themes, aggregate sentiment.
    Raw GKG CSVs to data/raw/news/, processed aggregates to data/processed/news/.
    """
    _setup_logging()
    start = start or datetime.combine(TRAINING_START_DATE, datetime.min.time(), tzinfo=timezone.utc)
    end = end or datetime.now(timezone.utc)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    current = start.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = end.replace(hour=0, minute=0, second=0, microsecond=0)
    total_days = (end_date - current).days + 1

    logging.info("GKG download: %s to %s (%d days, %d workers)", start.date(), end.date(), total_days, MAX_WORKERS)

    all_agg: list[pd.DataFrame] = []
    dates = [current + timedelta(days=i) for i in range(total_days)]
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_download_and_process_day, d): d for d in dates}
        for future in as_completed(futures):
            done += 1
            try:
                _, agg = future.result()
                if agg is not None:
                    all_agg.append(agg)
            except Exception as e:
                d = futures[future]
                logging.warning("GKG day %s failed: %s", d.strftime("%Y%m%d"), e)
            if done % 50 == 0 or done == total_days:
                logging.info("Progress: %d/%d days, %d FX days", done, total_days, len(all_agg))

    if not all_agg:
        logging.warning("No GKG FX data retrieved for %s to %s", start.date(), end.date())
        return [], None

    result = pd.concat(all_agg, ignore_index=True)
    result = result.sort_values("timestamp").reset_index(drop=True)

    # Save processed (compatible with build_features)
    clean_path = PROCESSED_DIR / f"news_gkg_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    result["timestamp"] = result["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    result.to_csv(clean_path, index=False)
    logging.info("Saved GKG news %d days to %s", len(result), clean_path.name)

    return [], clean_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download news from GDELT GKG")
    parser.add_argument("--test", action="store_true", help="Short test range (7 days)")
    args = parser.parse_args()

    if args.test:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=7)
        download_news_data(start=start, end=end)
    else:
        download_news_data()
