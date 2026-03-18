"""
Batch 2 — Sanity checks for core regression features.

Loads data/features_regression_core/train, checks:
- No NaNs in features
- Feature distributions (summary stats)
- Outliers (abs z-score > 5)
- Feature-feature correlations (flag |corr| > 0.95)
- corr(pred_features, target_ret_6) diagnostic

Exits with non-zero if NaNs or absurd correlations.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_DIR = PROJECT_ROOT / "data" / "features_regression_core"
LOG_FILE = PROJECT_ROOT / "data" / "logs" / "sanity_check_core.log"

TARGET_COL = "target_ret_6"
ZSCORE_THRESHOLD = 5.0
CORR_DUPLICATE_THRESHOLD = 0.95
SIGNAL_WARN_THRESHOLD = 0.01  # warn if max |corr(feature, target)| < this


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


def load_train() -> pd.DataFrame:
    """Load train split from parquet or csv."""
    for ext in ["parquet", "csv"]:
        path = FEATURES_DIR / f"train.{ext}"
        if path.exists():
            if ext == "parquet":
                return pd.read_parquet(path)
            return pd.read_csv(path)
    raise FileNotFoundError(f"Train not found in {FEATURES_DIR}")


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Feature columns (exclude timestamp and targets)."""
    exclude = {"timestamp", "target_ret_3", "target_ret_6", "target_ret_12"}
    return [c for c in df.columns if c not in exclude]


def run() -> int:
    """
    Run sanity checks. Returns 0 on success, 1 on failure.
    """
    _setup_logging()
    log = logging.getLogger()

    if not FEATURES_DIR.exists():
        log.error("Features dir not found: %s. Run build_features_regression_core first.", FEATURES_DIR)
        return 1

    df = load_train()
    feature_cols = get_feature_cols(df)
    log.info("Loaded train: %d rows, %d features", len(df), len(feature_cols))

    failed = False

    # 1. NaN check
    nan_counts = df[feature_cols].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        log.error("NaNs found in features: %s", nan_cols.to_dict())
        failed = True
    else:
        log.info("NaN check: PASS (no NaNs)")

    # 2. Distributions
    log.info("Feature distributions (mean, std, min, max):")
    stats = df[feature_cols].describe().loc[["mean", "std", "min", "max"]]
    for col in feature_cols:
        row = stats[col]
        log.info("  %s: mean=%.6g std=%.6g min=%.6g max=%.6g", col, row["mean"], row["std"], row["min"], row["max"])

    # 3. Outliers (abs z-score > 5)
    outlier_counts = {}
    for col in feature_cols:
        s = df[col]
        mean, std = s.mean(), s.std()
        if std == 0 or np.isnan(std):
            continue
        z = np.abs((s - mean) / std)
        n = (z > ZSCORE_THRESHOLD).sum()
        if n > 0:
            outlier_counts[col] = int(n)
    if outlier_counts:
        log.warning("Outliers (|z|>%.1f): %s", ZSCORE_THRESHOLD, outlier_counts)
    else:
        log.info("Outlier check: PASS (no extreme outliers)")

    # 4. Feature-feature correlation (near-duplicates)
    corr_mat = df[feature_cols].corr()
    high_corr_pairs = []
    for i, c1 in enumerate(feature_cols):
        for c2 in feature_cols[i + 1 :]:
            r = corr_mat.loc[c1, c2]
            if abs(r) > CORR_DUPLICATE_THRESHOLD:
                high_corr_pairs.append((c1, c2, float(r)))
    if high_corr_pairs:
        log.warning("Near-duplicate feature pairs (|corr|>%.2f): %s", CORR_DUPLICATE_THRESHOLD, high_corr_pairs)
        failed = True
    else:
        log.info("Feature-feature correlation: PASS (no near-duplicates)")

    # 5. Diagnostic: corr(pred_features, target_ret_6)
    if TARGET_COL not in df.columns:
        log.error("Target %s not found", TARGET_COL)
        failed = True
    else:
        target_corrs = df[feature_cols].corrwith(df[TARGET_COL])
        target_corrs = target_corrs.reindex(target_corrs.abs().sort_values(ascending=False).index)
        log.info("corr(feature, %s) sorted by |corr|:", TARGET_COL)
        for feat, r in target_corrs.items():
            log.info("  %s: %.6f", feat, r)
        valid_corrs = target_corrs.dropna()
        max_abs = valid_corrs.abs().max() if len(valid_corrs) > 0 else 0.0
        if max_abs < SIGNAL_WARN_THRESHOLD:
            log.warning(
                "SIGNAL WARNING: max |corr(feature, %s)| = %.6f < %.2f — no signal detected",
                TARGET_COL,
                max_abs,
                SIGNAL_WARN_THRESHOLD,
            )
        else:
            log.info("Signal check: max |corr| = %.6f", max_abs)

    return 1 if failed else 0


if __name__ == "__main__":
    exit_code = run()
    sys.exit(exit_code)
