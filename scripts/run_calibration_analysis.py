"""
Probability calibration: bin predictions by confidence, verify higher confidence → better outcomes.

Bins: 0.35–0.45, 0.45–0.55, 0.55–0.65, 0.65–0.75, 0.75+
Within vol regime (top 30%). Output: data/validation/calibration_report.csv
"""

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
FROZEN_DIR = PROJECT_ROOT / "data" / "frozen_test"
from src.config import VOL_REGIME_TOP_PCT


def _load_frozen_test() -> pd.DataFrame:
    manifest_path = FROZEN_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run freeze_test_set.py first")
    with open(manifest_path) as f:
        manifest = json.load(f)
    df = pd.read_parquet(FROZEN_DIR / manifest["file"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def run() -> int:
    """Run calibration analysis. Returns 0 on success."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    pred = pd.read_csv(PREDICTIONS_DIR / "test_predictions.csv")
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True)
    frozen = _load_frozen_test()
    pred = pred.merge(
        frozen[["timestamp", "return_5", "volatility_20", "hour", "weekday"]],
        on="timestamp",
        how="inner",
    )

    vol = pred["volatility_20"].fillna(0).values.astype(float)
    vol_threshold = np.percentile(vol, 100 - VOL_REGIME_TOP_PCT)
    vol_mask = vol >= vol_threshold
    pred_regime = pred.loc[vol_mask].reset_index(drop=True)

    conf = np.maximum(pred_regime["P_buy"].values, pred_regime["P_sell"].values)
    pos = np.where(
        pred_regime["P_buy"].values > pred_regime["P_sell"].values,
        1,
        np.where(pred_regime["P_sell"].values > pred_regime["P_buy"].values, -1, 0),
    )
    ret = pred_regime["future_return_30m"].values.astype(float)

    bins = [(0.35, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 1.01)]
    rows = []

    for lo, hi in bins:
        mask = (conf >= lo) & (conf < hi) & (pos != 0)
        if mask.sum() == 0:
            rows.append({
                "bin": f"{lo}-{hi}",
                "n": 0,
                "hit_rate": np.nan,
                "mean_return": np.nan,
                "mean_return_signed": np.nan,
            })
            continue
        r = ret[mask]
        p = pos[mask]
        correct = (np.sign(r) == np.sign(p)).sum()
        hit_rate = correct / len(r)
        mean_return = np.mean(r)
        mean_return_signed = np.mean(p * r)
        rows.append({
            "bin": f"{lo}-{hi}",
            "n": int(mask.sum()),
            "hit_rate": round(hit_rate, 4),
            "mean_return": round(mean_return, 6),
            "mean_return_signed": round(mean_return_signed, 6),
        })

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    report_path = VALIDATION_DIR / "calibration_report.csv"
    with open(report_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bin", "n", "hit_rate", "mean_return", "mean_return_signed"])
        w.writeheader()
        w.writerows(rows)

    logging.info("Calibration report saved to %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(run())
