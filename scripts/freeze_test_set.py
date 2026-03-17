"""
Phase 18 — Freeze test set before serious tuning.

Copies current test.csv/test.parquet to data/frozen_test/ with date suffix.
Records manifest. Run once before hyperparameter tuning.
"""

import json
import logging
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TEST_START_DATE

FROZEN_DIR = PROJECT_ROOT / "data" / "frozen_test"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"


def run() -> int:
    """
    Copy test set to frozen_test/. Create manifest.
    Returns 0 on success, 1 on failure.
    """
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)

    freeze_date = date.today()
    date_str = freeze_date.isoformat().replace("-", "")

    test_csv = FEATURES_DIR / "test.csv"
    test_parquet = FEATURES_DIR / "test.parquet"
    source = test_parquet if test_parquet.exists() else test_csv
    if not source.exists():
        logging.error("Test set not found: %s or %s", test_csv, test_parquet)
        return 1

    dest_name = f"test_{date_str}.{source.suffix[1:]}"
    dest_path = FROZEN_DIR / dest_name
    shutil.copy2(source, dest_path)
    logging.info("Frozen test set to %s", dest_path)

    manifest = {
        "freeze_date": freeze_date.isoformat(),
        "freeze_run_at": datetime.now(timezone.utc).isoformat(),
        "test_start_date": str(TEST_START_DATE),
        "file": dest_name,
        "source": str(source.name),
    }
    manifest_path = FROZEN_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logging.info("Manifest saved to %s", manifest_path)

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    exit_code = run()
    sys.exit(exit_code)
