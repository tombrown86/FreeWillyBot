"""
Run Batch 3 regression training pipeline.

Trains LinearRegression, Ridge, ElasticNet, XGBRegressor for horizons 3, 6, 12.
Evaluates on validation, saves predictions, plots, and best model.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.train_regression import run


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run()
