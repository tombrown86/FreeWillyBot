"""
Phase 16 — Daily retrain script.

Retrain baselines and meta-model. Ensures features exist first.
"""

import logging
import sys
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


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

    exit_code = run()
    sys.exit(exit_code)
