"""
Predict on frozen regression test set using best model.

Loads regression_best.pkl and regression_feature_cols.json from data/models/.
Loads frozen test from data/frozen_test_regression_core/ (via manifest).
Saves to data/predictions_regression/test_predictions.parquet.
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "data" / "models"
FROZEN_DIR = PROJECT_ROOT / "data" / "frozen_test_regression_core"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions_regression"


def run() -> int:
    """Load model, predict on frozen test, save. Returns 0 on success, 1 on failure."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger()

    model_path = MODELS_DIR / "regression_best.pkl"
    cols_path = MODELS_DIR / "regression_feature_cols.json"
    config_path = MODELS_DIR / "regression_best_config.json"

    if not model_path.exists():
        log.error("Model not found: %s. Run run_train_regression first.", model_path)
        return 1
    if not cols_path.exists() or not config_path.exists():
        log.error("Config not found. Run run_train_regression first.")
        return 1

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(cols_path) as f:
        feature_cols = json.load(f)
    with open(config_path) as f:
        config = json.load(f)

    horizon = config["horizon"]
    target_col = config["target_col"]

    manifest_path = FROZEN_DIR / "manifest.json"
    if not manifest_path.exists():
        log.error("Frozen test manifest not found: %s. Run freeze_test_set_regression --core first.", manifest_path)
        return 1

    with open(manifest_path) as f:
        manifest = json.load(f)
    frozen_file = manifest.get("file", "test.parquet")
    frozen_path = FROZEN_DIR / frozen_file

    if not frozen_path.exists():
        log.error("Frozen test file not found: %s", frozen_path)
        return 1

    df = pd.read_parquet(frozen_path) if frozen_path.suffix == ".parquet" else pd.read_csv(frozen_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        log.error("Missing feature columns in frozen test: %s", missing)
        return 1

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    pred = model.predict(X)

    out = df[["timestamp", target_col]].copy()
    out = out.rename(columns={target_col: "target_ret"})
    out["pred"] = pred
    for v in ["vol_6", "vol_12", "vol_24"]:
        if v in df.columns:
            out[v] = df[v].values

    # Regime signals
    if "ret_12" in df.columns:
        out["ret_12"] = df["ret_12"].values
    if "vol_6" in df.columns:
        out["vol_change"] = df["vol_6"].diff().values

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PREDICTIONS_DIR / "test_predictions.parquet", index=False)
    log.info("Saved test predictions to %s (%d rows)", PREDICTIONS_DIR / "test_predictions.parquet", len(out))
    return 0


if __name__ == "__main__":
    sys.exit(run())
