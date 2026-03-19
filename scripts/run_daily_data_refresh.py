"""
Phase 16 — Daily data refresh script.

Download latest data, rebuild bars and features, refresh forecaster predictions.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

HEARTBEAT_PATH = PROJECT_ROOT / "data" / "logs" / "execution" / "livetick_heartbeat.json"


def _read_heartbeat() -> dict:
    HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not HEARTBEAT_PATH.exists():
        return {}
    try:
        with open(HEARTBEAT_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _write_heartbeat(**kwargs: str) -> None:
    data = _read_heartbeat()
    data.update({k: v for k, v in kwargs.items() if v is not None})
    with open(HEARTBEAT_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _last_run_age_hours() -> float | None:
    """Return how many hours ago the last successful data refresh ran, or None if never."""
    h = _read_heartbeat()
    ts_str = h.get("last_data_refresh_utc")
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
    except Exception:
        return None

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
    Run full data refresh in two phases:

    Phase A — fast (~5 min): price + cross-asset + macro downloads, build bars,
      build all feature files, write heartbeat. Live tick gets fresh bars immediately.

    Phase B — slow (~1-2 h): news download + forecasters (Chronos-Bolt).
      Failures here do NOT block Phase A or the live tick.

    Returns 0 on success, 1 if any Phase A step fails.
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

    def do_build_features_regression_core():
        """Keep features_regression_core/test in sync with price (live regression bar timestamps)."""
        from src.build_features_regression_core import run as build_reg_core_run
        build_reg_core_run()

    def do_run_forecasters():
        from src.train_price_model import run_forecasters
        run_forecasters()

    def do_live_tail_ctrader():
        """Rebuild live tail from cTrader bars (execution broker). Non-fatal if cTrader is down."""
        from src.build_features_regression_core import run_live_tail_ctrader
        run_live_tail_ctrader()

    # ── Phase A: fast steps — must all succeed ────────────────────────────────
    phase_a = [
        ("download_price", do_download_price, True),
        ("download_cross_asset", do_download_cross_asset, True),
        ("download_macro", do_download_macro, True),
        ("build_price_bars", do_build_bars, False),
        ("build_features", do_build_features, False),
        ("build_features_regression_core", do_build_features_regression_core, False),
        # Pull live tail from cTrader (execution broker) so live signals use the same
        # price feed we trade on.  Listed as non-critical (failure logged, not fatal)
        # so a cTrader outage never blocks the rest of the refresh.
        ("live_tail_ctrader", do_live_tail_ctrader, False),
    ]

    for name, func, use_retry in phase_a:
        if name == "live_tail_ctrader":
            # Non-fatal: cTrader downtime should not abort the refresh
            _run_step(name, func, use_retry)
        elif not _run_step(name, func, use_retry):
            logging.error("Phase A step '%s' failed — aborting refresh", name)
            return 1

    # Features are now current; write heartbeat so live tick sees fresh data
    _write_heartbeat(
        last_data_refresh_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    )
    logging.info("Phase A complete — features updated, live tick will use fresh bars")

    # ── Phase B: slow steps — failures are logged but don't fail the job ─────
    phase_b = [
        ("download_news", do_download_news, True),
        ("run_forecasters", do_run_forecasters, True),
    ]

    phase_b_ok = True
    for name, func, use_retry in phase_b:
        if not _run_step(name, func, use_retry):
            logging.warning("Phase B step '%s' failed — continuing (live tick not affected)", name)
            phase_b_ok = False

    if phase_b_ok:
        logging.info("Phase B complete — news and forecasters updated")
    else:
        logging.warning("Phase B had failures — news/forecasters may be stale, but features are current")

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
    parser.add_argument(
        "--skip-if-recent",
        type=float,
        metavar="HOURS",
        default=None,
        help="Skip if a successful refresh already ran within this many hours (used by RunAtLoad catch-up)",
    )
    args = parser.parse_args()

    if args.skip_if_recent is not None:
        age = _last_run_age_hours()
        if age is not None and age < args.skip_if_recent:
            logging.info(
                "Skipping data refresh — last run %.1fh ago (skip threshold %.1fh)",
                age,
                args.skip_if_recent,
            )
            sys.exit(0)

    start_dt = None
    end_dt = None
    if args.start:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if args.end:
        end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    exit_code = run(start=start_dt, end=end_dt)
    sys.exit(exit_code)
