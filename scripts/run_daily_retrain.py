"""
Phase 16 — Daily retrain script.

Retrain baselines and meta-model. Ensures features exist first.
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


def _last_retrain_age_hours() -> float | None:
    """Return how many hours ago the last successful retrain ran, or None if never."""
    h = _read_heartbeat()
    ts_str = h.get("last_retrain_utc")
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
    except Exception:
        return None


def run() -> int:
    """
    Run full retrain: train_price_model.run(), train_meta_model.run().
    Returns 0 on success, 1 on failure.
    """
    features_dir = PROJECT_ROOT / "data" / "features"
    test_path = features_dir / "test.csv"
    if not test_path.exists():
        logging.error(
            "Features not found: %s. Run run_daily_data_refresh first.",
            test_path,
        )
        return 1

    try:
        from src.train_price_model import run as train_price_run
        from src.train_meta_model import run as train_meta_run

        train_price_run()
        train_meta_run()
        logging.info("Retrain complete")
        _write_heartbeat(
            last_retrain_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        )
        return 0
    except Exception as e:
        logging.exception("Retrain failed: %s", e)
        return 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    import argparse

    parser = argparse.ArgumentParser(description="Daily retrain")
    parser.add_argument(
        "--skip-if-recent",
        type=float,
        metavar="HOURS",
        default=None,
        help="Skip if a successful retrain already ran within this many hours (used by RunAtLoad catch-up)",
    )
    args = parser.parse_args()

    if args.skip_if_recent is not None:
        age = _last_retrain_age_hours()
        if age is not None and age < args.skip_if_recent:
            logging.info(
                "Skipping retrain — last run %.1fh ago (skip threshold %.1fh)",
                age,
                args.skip_if_recent,
            )
            sys.exit(0)

    exit_code = run()
    sys.exit(exit_code)
