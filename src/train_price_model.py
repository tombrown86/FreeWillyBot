"""
Phase 11 — Train baselines and run pretrained forecasters.

1. Train XGBoost and Logistic Regression on engineered features only.
2. Compare via walk-forward validation, calibration, stability, perf after costs.
3. Choose baseline (tie-break to LogReg).
4. Run Chronos-Bolt and TimesFM on price history; save pred returns as features.
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    BAR_INTERVAL,
    BUY_THRESHOLD_PCT,
    CRYPTO_FEE_PCT,
    CRYPTO_SLIPPAGE_PCT,
    FX_SPREAD_PIPS,
    FORECASTER_CONTEXT_BARS,
    FORECASTER_HORIZON_BARS,
    LABEL_HORIZON_BARS,
    SELL_THRESHOLD_PCT,
    SYMBOL,
    TEST_START_DATE,
    TRAINING_START_DATE,
    VALIDATION_START_DATE,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
PROCESSED_PRICE = PROJECT_ROOT / "data" / "processed" / "price"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
LOG_FILE = PROJECT_ROOT / "data" / "logs" / "train_price_model.log"

# Feature columns (exclude target, timestamp, forecaster preds)
EXCLUDE_COLS = {"timestamp", "future_return_30m", "label_class", "chronos_pred_return", "timesfm_pred_return"}

# Walk-forward: 3 months train, 1 month test
WF_TRAIN_MONTHS = 3
WF_TEST_MONTHS = 1
# Step between folds (days) — 7 gives 6–12 windows from Oct 2023 to Jan 2024
WF_STEP_DAYS = 7

# FX spread as decimal (1 pip = 0.0001 for EURUSD)
FX_SPREAD = FX_SPREAD_PIPS * 0.0001


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


def _brier_score_multi(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int = 3) -> float:
    """Multiclass Brier score (lower = better calibrated). y_true: 0,1,2 (sell, no-trade, buy)."""
    idx = np.clip(y_true.astype(int), 0, n_classes - 1)
    one_hot = np.zeros((len(y_true), n_classes))
    one_hot[np.arange(len(y_true)), idx] = 1
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def _mini_backtest_returns(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    spread: float,
    is_crypto: bool,
) -> tuple[float, int]:
    """Cumulative return after costs and trade count. y_pred: 1=long, -1=short, 0=no-trade."""
    ret = df["future_return_30m"].values
    cost_per_leg = spread if not is_crypto else (CRYPTO_FEE_PCT + CRYPTO_SLIPPAGE_PCT)
    positions = np.sign(y_pred)
    prev_pos = 0
    cum = 1.0
    n_trades = 0
    for i in range(len(positions)):
        p = positions[i]
        if p != prev_pos:
            n_trades += 1
            legs = (1 if prev_pos == 0 or p == 0 else 2)
            cum *= 1 - legs * cost_per_leg
        if p != 0:
            cum *= 1 + p * ret[i]
        prev_pos = p
    return cum - 1.0, n_trades


def _run_walk_forward(
    train_val: pd.DataFrame,
    model_type: str,
    feature_cols: list[str],
) -> dict:
    """Walk-forward validation on train+val period. Steps by WF_STEP_DAYS to get 6–12 folds."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from xgboost import XGBClassifier

    train_val = train_val.copy()
    train_val["timestamp"] = pd.to_datetime(train_val["timestamp"], utc=True)
    train_val = train_val.sort_values("timestamp").reset_index(drop=True)

    X = train_val[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = train_val["label_class"]

    # Map -1,0,1 to 0,1,2 for sklearn
    y_sk = np.where(y == -1, 0, np.where(y == 0, 1, 2))

    results = []
    validation_dir = PROJECT_ROOT / "data" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    wf_path = validation_dir / "walk_forward_windows.csv"

    current = pd.Timestamp(VALIDATION_START_DATE, tz="UTC")
    test_end_limit = pd.Timestamp(TEST_START_DATE, tz="UTC")
    while current < test_end_limit:
        train_start = current - pd.DateOffset(months=WF_TRAIN_MONTHS)
        train_end = current
        test_start = current
        test_end = current + pd.DateOffset(months=WF_TEST_MONTHS)

        if test_end > test_end_limit:
            break
        assert test_end <= test_end_limit, "Walk-forward must never include test set"

        mask_train = (train_val["timestamp"] >= train_start) & (train_val["timestamp"] < train_end)
        mask_test = (train_val["timestamp"] >= test_start) & (train_val["timestamp"] < test_end)

        if mask_train.sum() < 1000 or mask_test.sum() < 100:
            current = current + pd.Timedelta(days=WF_STEP_DAYS)
            continue

        X_tr = X.loc[mask_train].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_tr = y_sk[mask_train]
        X_te = X.loc[mask_test].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_te = y_sk[mask_test]
        df_te = train_val.loc[mask_test]

        if model_type == "xgboost":
            model = XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
            model.fit(X_tr, y_tr)
        else:
            model = LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            )
            model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)

        # Map back to -1,0,1 for metrics
        y_pred_orig = np.where(y_pred == 0, -1, np.where(y_pred == 1, 0, 1))

        acc = accuracy_score(y_te, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="macro", zero_division=0)
        brier = _brier_score_multi(y_te, y_prob)
        is_crypto = "USDT" in SYMBOL
        net_ret, n_trades = _mini_backtest_returns(df_te, y_pred_orig, FX_SPREAD, is_crypto)

        row = {
            "fold": len(results) + 1,
            "model_type": model_type,
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "brier": brier,
            "net_return": net_ret,
            "n_trades": n_trades,
        }
        results.append(row)
        current = current + pd.Timedelta(days=WF_STEP_DAYS)

    if results:
        import csv
        from datetime import datetime, timezone
        fieldnames = ["fold", "model_type", "train_start", "train_end", "test_start", "test_end",
                      "accuracy", "precision", "recall", "f1", "brier", "net_return", "n_trades", "run_at"]
        file_exists = wf_path.exists()
        with open(wf_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                w.writeheader()
            run_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            for r in results:
                row = {k: r[k] for k in fieldnames if k in r}
                row["run_at"] = run_at
                w.writerow(row)
        logging.info("Walk-forward results saved to %s (%d folds)", wf_path, len(results))

    if not results:
        return {"mean_accuracy": 0, "mean_precision": 0, "mean_recall": 0, "std_precision": 999, "std_recall": 999, "mean_brier": 999, "mean_net_return": -999, "n_folds": 0}

    df_r = pd.DataFrame(results)
    return {
        "mean_accuracy": df_r["accuracy"].mean(),
        "mean_precision": df_r["precision"].mean(),
        "mean_recall": df_r["recall"].mean(),
        "std_precision": df_r["precision"].std() or 0,
        "std_recall": df_r["recall"].std() or 0,
        "mean_brier": df_r["brier"].mean(),
        "mean_net_return": df_r["net_return"].mean(),
        "n_folds": len(results),
    }


def _select_baseline(xgb_scores: dict, logreg_scores: dict) -> str:
    """Choose baseline: more stable wins; tie-break to LogReg."""
    if xgb_scores["n_folds"] == 0 and logreg_scores["n_folds"] == 0:
        return "logreg"
    if xgb_scores["n_folds"] == 0:
        return "logreg"
    if logreg_scores["n_folds"] == 0:
        return "xgboost"

    # Stability: lower variance
    xgb_stability = 1 / (1 + xgb_scores["std_precision"] + xgb_scores["std_recall"])
    logreg_stability = 1 / (1 + logreg_scores["std_precision"] + logreg_scores["std_recall"])

    # Better calibration (lower Brier)
    xgb_cal = 1 - xgb_scores["mean_brier"]
    logreg_cal = 1 - logreg_scores["mean_brier"]

    # OOS net return
    xgb_ret = xgb_scores["mean_net_return"]
    logreg_ret = logreg_scores["mean_net_return"]

    # XGB wins only if clearly better on OOS precision, trade quality, regime robustness
    xgb_better = (
        xgb_scores["mean_precision"] > logreg_scores["mean_precision"] * 1.05
        and xgb_ret > logreg_ret * 1.1
        and xgb_stability >= logreg_stability * 0.95
    )

    if xgb_better:
        return "xgboost"
    return "logreg"


def run_baselines() -> str:
    """Train both baselines, compare, return chosen model type."""
    _setup_logging()

    def _load_feature(name: str) -> pd.DataFrame:
        p = FEATURES_DIR / f"{name}.parquet"
        c = FEATURES_DIR / f"{name}.csv"
        if p.exists():
            return pd.read_parquet(p)
        return pd.read_csv(c)

    train = _load_feature("train")
    val = _load_feature("validation")
    train_val = pd.concat([train, val], ignore_index=True)

    feature_cols = [c for c in train.columns if c not in EXCLUDE_COLS]
    feature_cols = [c for c in feature_cols if train[c].dtype in ("float64", "int64") or "float" in str(train[c].dtype)]

    logging.info("Baseline features: %d cols", len(feature_cols))

    logging.info("Running XGBoost walk-forward...")
    xgb_scores = _run_walk_forward(train_val, "xgboost", feature_cols)
    logging.info("XGBoost: acc=%.4f prec=%.4f rec=%.4f brier=%.4f net_ret=%.4f std_prec=%.4f std_rec=%.4f",
                 xgb_scores["mean_accuracy"], xgb_scores["mean_precision"], xgb_scores["mean_recall"],
                 xgb_scores["mean_brier"], xgb_scores["mean_net_return"],
                 xgb_scores["std_precision"], xgb_scores["std_recall"])

    logging.info("Running Logistic Regression walk-forward...")
    logreg_scores = _run_walk_forward(train_val, "logreg", feature_cols)
    logging.info("LogReg: acc=%.4f prec=%.4f rec=%.4f brier=%.4f net_ret=%.4f std_prec=%.4f std_rec=%.4f",
                 logreg_scores["mean_accuracy"], logreg_scores["mean_precision"], logreg_scores["mean_recall"],
                 logreg_scores["mean_brier"], logreg_scores["mean_net_return"],
                 logreg_scores["std_precision"], logreg_scores["std_recall"])

    chosen = _select_baseline(xgb_scores, logreg_scores)
    logging.info("Chosen baseline: %s", chosen)

    # Train final model on full train+val
    X = train_val[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = train_val["label_class"]
    y_sk = np.where(y == -1, 0, np.where(y == 0, 1, 2))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "baseline_feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    if chosen == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
        model.fit(X, y_sk)
        with open(MODELS_DIR / "baseline_xgb.pkl", "wb") as f:
            pickle.dump(model, f)
    else:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
        model.fit(X, y_sk)
        with open(MODELS_DIR / "baseline_logreg.pkl", "wb") as f:
            pickle.dump(model, f)

    with open(MODELS_DIR / "baseline_choice.json", "w") as f:
        json.dump({"baseline": chosen, "xgb_scores": xgb_scores, "logreg_scores": logreg_scores}, f, indent=2)

    return chosen


def run_chronos_bolt(price: pd.DataFrame) -> pd.DataFrame:
    """Run Chronos-Bolt on close prices; return DataFrame with chronos_pred_return."""
    from chronos import ChronosBoltPipeline
    import torch

    close = price["close"].astype(float).values
    n = len(close)
    ctx = FORECASTER_CONTEXT_BARS
    horizon = FORECASTER_HORIZON_BARS

    pipeline = ChronosBoltPipeline.from_pretrained("amazon/chronos-bolt-small")
    pipeline.model.eval()

    pred_returns = np.full(n, np.nan)
    batch_size = 64
    total = n - ctx - horizon

    for start in range(ctx, n - horizon, batch_size):
        end_batch = min(start + batch_size, n - horizon)
        if (start - ctx) % 5000 < batch_size:
            logging.info("Chronos-Bolt progress: %d / %d", start - ctx, total)

        contexts = []
        for i in range(start, end_batch):
            ctx_slice = close[i - ctx : i]
            contexts.append(torch.tensor(ctx_slice, dtype=torch.float32))

        with torch.no_grad():
            _, mean = pipeline.predict_quantiles(contexts, prediction_length=horizon, quantile_levels=[0.5])

        mean_np = mean.numpy()
        for j, i in enumerate(range(start, end_batch)):
            pred_close_6 = mean_np[j, horizon - 1]
            curr_close = close[i]
            if curr_close > 0:
                pred_returns[i] = (pred_close_6 - curr_close) / curr_close

    df = price[["timestamp"]].copy()
    df["chronos_pred_return"] = pred_returns
    return df


def run_timesfm(price: pd.DataFrame) -> pd.DataFrame | None:
    """Run TimesFM on close prices; return DataFrame with timesfm_pred_return or None if failed."""
    try:
        import timesfm
    except ImportError as e:
        logging.warning("TimesFM not available: %s", e)
        return None

    close = price["close"].astype(float).values
    n = len(close)
    ctx = min(FORECASTER_CONTEXT_BARS, 512)
    horizon = FORECASTER_HORIZON_BARS

    model = None
    try:
        if hasattr(timesfm, "TimesFM_2p5_200M_torch"):
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
            if hasattr(model, "compile") and hasattr(timesfm, "ForecastConfig"):
                model.compile(timesfm.ForecastConfig(max_context=ctx, max_horizon=horizon))
        elif hasattr(timesfm, "TimesFm"):
            model = timesfm.TimesFm.from_pretrained("google/timesfm-1.0-200m-pytorch")
    except Exception as e:
        logging.warning("TimesFM load failed: %s", e)
        return None

    if model is None or not hasattr(model, "forecast"):
        logging.warning("TimesFM: no compatible model found")
        return None

    pred_returns = np.full(n, np.nan)
    batch_size = 32
    total = n - ctx - horizon

    for start in range(ctx, n - horizon, batch_size):
        end_batch = min(start + batch_size, n - horizon)
        if (start - ctx) % 5000 < batch_size:
            logging.info("TimesFM progress: %d / %d", start - ctx, total)

        inputs = [close[i - ctx : i].astype(np.float32) for i in range(start, end_batch)]

        try:
            point, _ = model.forecast(horizon=horizon, inputs=inputs)
        except Exception as e:
            logging.warning("TimesFM forecast failed at %d: %s", start, e)
            continue

        point = np.asarray(point)
        for j, i in enumerate(range(start, end_batch)):
            if j < point.shape[0] and horizon - 1 < point.shape[1]:
                pred_close_6 = float(point[j, horizon - 1])
                curr_close = close[i]
                if curr_close > 0:
                    pred_returns[i] = (pred_close_6 - curr_close) / curr_close

    df = price[["timestamp"]].copy()
    df["timesfm_pred_return"] = pred_returns
    return df


def run_forecasters() -> None:
    """Run Chronos-Bolt and TimesFM; save forecaster_predictions.csv."""
    _setup_logging()

    from src.utils import load_processed_price

    price = load_processed_price(PROCESSED_PRICE, SYMBOL, BAR_INTERVAL)
    price["timestamp"] = pd.to_datetime(price["timestamp"], utc=True)
    price = price.sort_values("timestamp").reset_index(drop=True)

    logging.info("Running Chronos-Bolt on %d bars...", len(price))
    chronos_df = run_chronos_bolt(price)

    logging.info("Running TimesFM on %d bars...", len(price))
    timesfm_df = run_timesfm(price)

    # Merge
    result = chronos_df.copy()
    if timesfm_df is not None:
        result = result.merge(timesfm_df[["timestamp", "timesfm_pred_return"]], on="timestamp", how="left")
    else:
        result["timesfm_pred_return"] = np.nan

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(FEATURES_DIR / "forecaster_predictions.csv", index=False)
    logging.info("Saved forecaster_predictions.csv with %d rows", len(result))


def run() -> None:
    """Full pipeline: baselines + forecasters."""
    chosen = run_baselines()
    run_forecasters()
    logging.info("Phase 11 complete. Baseline: %s", chosen)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines-only", action="store_true", help="Run only baselines")
    parser.add_argument("--forecasters-only", action="store_true", help="Run only forecasters")
    args = parser.parse_args()

    if args.baselines_only:
        run_baselines()
    elif args.forecasters_only:
        _setup_logging()
        run_forecasters()
    else:
        run()
