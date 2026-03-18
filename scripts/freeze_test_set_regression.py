"""
Freeze regression test set before tuning.

Copies data/features_regression/test.parquet to data/frozen_test_regression/ with date suffix.
Records manifest. Run after build_features_regression.
Use --core to freeze from data/features_regression_core/ to data/frozen_test_regression_core/.
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TEST_START_DATE


def run(use_core: bool = False) -> int:
    """
    Copy regression test set to frozen_test_regression/. Create manifest.
    Returns 0 on success, 1 on failure.
    """
    if use_core:
        features_dir = PROJECT_ROOT / "data" / "features_regression_core"
        frozen_dir = PROJECT_ROOT / "data" / "frozen_test_regression_core"
        build_cmd = "build_features_regression_core"
    else:
        features_dir = PROJECT_ROOT / "data" / "features_regression"
        frozen_dir = PROJECT_ROOT / "data" / "frozen_test_regression"
        build_cmd = "build_features_regression"

    frozen_dir.mkdir(parents=True, exist_ok=True)

    freeze_date = date.today()
    date_str = freeze_date.isoformat().replace("-", "")

    test_csv = features_dir / "test.csv"
    test_parquet = features_dir / "test.parquet"
    source = test_parquet if test_parquet.exists() else test_csv
    if not source.exists():
        logging.error("Regression test set not found. Run %s first: %s or %s", build_cmd, test_csv, test_parquet)
        return 1

    dest_name = f"test_{date_str}.{source.suffix[1:]}"
    dest_path = frozen_dir / dest_name
    shutil.copy2(source, dest_path)
    logging.info("Frozen regression test set to %s", dest_path)

    manifest = {
        "freeze_date": freeze_date.isoformat(),
        "freeze_run_at": datetime.now(timezone.utc).isoformat(),
        "test_start_date": str(TEST_START_DATE),
        "file": dest_name,
        "source": str(source.name),
        "feature_set": "core" if use_core else "full",
    }
    manifest_path = frozen_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logging.info("Manifest saved to %s", manifest_path)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freeze regression test set before tuning")
    parser.add_argument("--core", action="store_true", help="Use core feature set (features_regression_core)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    exit_code = run(use_core=args.core)
    sys.exit(exit_code)
