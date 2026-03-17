"""
Phase 16 — Daily data refresh script.

Download latest data, rebuild bars and features, refresh forecaster predictions.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import retry_with_backoff

try:
    import requests
    NETWORK_EXCEPTIONS = (requests.RequestException, ConnectionError, OSError, TimeoutError)
except ImportError:
    NETWORK_EXCEPTIONS = (ConnectionError, OSError, TimeoutError)


def _run_step(name: str, func, use_retry: bool = False) -> bool:
    """Run one step. Return True on success, False on failure."""
    try:
        if use_retry:
            retry_with_backoff(
                func,
                max_attempts=3,
                base_delay=1.0,
                exceptions=NETWORK_EXCEPTIONS,
            )
        else:
            func()
        logging.info("Completed: %s", name)
        return True
    except Exception as e:
        logging.exception("Failed %s: %s", name, e)
        return False


def run(start: datetime | None = None, end: datetime | None = None) -> int:
    """
    Run full data refresh: downloads, build bars, build features, run forecasters.
    start/end: optional custom range (UTC). Default: TRAINING_START_DATE to now.
    Returns 0 on success, 1 on failure.
    """
    from src.config import TRAINING_START_DATE

    end = end or datetime.now(timezone.utc)
    start = start or datetime.combine(TRAINING_START_DATE, datetime.min.time(), tzinfo=timezone.utc)

    def do_download_price():
        from src.download_price import download_price_data

        download_price_data(start=start, end=end)

    def do_download_cross_asset():
        from src.download_cross_asset import download_cross_asset_data

        download_cross_asset_data(start=start, end=end)

    def do_download_macro():
        from src.download_macro import download_macro_data

        download_macro_data(start=start, end=end)

    def do_download_news():
        from src.download_news import download_news_data

        download_news_data(start=start, end=end)

    def do_build_bars():
        from src.build_price_bars import run as build_bars_run

        build_bars_run()

    def do_build_features():
        from src.build_features import run as build_features_run

        build_features_run()

    def do_run_forecasters():
        from src.train_price_model import run_forecasters

        run_forecasters()

    steps = [
        ("download_price", do_download_price, True),
        ("download_cross_asset", do_download_cross_asset, True),
        ("download_macro", do_download_macro, True),
        ("download_news", do_download_news, True),
        ("build_price_bars", do_build_bars, False),
        ("build_features", do_build_features, False),
        ("run_forecasters", do_run_forecasters, True),
    ]

    for name, func, use_retry in steps:
        if not _run_step(name, func, use_retry):
            return 1
    return 0


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="Daily data refresh")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD) for download range")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD) for download range")
    args = parser.parse_args()

    start_dt = None
    end_dt = None
    if args.start:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if args.end:
        end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    exit_code = run(start=start_dt, end=end_dt)
    sys.exit(exit_code)
