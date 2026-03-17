"""
Phase 11 — Train meta-model on engineered features + forecaster predictions.

Uses the chosen baseline (XGBoost or LogReg) with forecaster features added.
Outputs probabilities for buy/sell/no-trade. Applies no-trade threshold.
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
    NO_TRADE_THRESHOLD_PCT,
    SYMBOL,
    TEST_START_DATE,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
LOG_FILE = PROJECT_ROOT / "data" / "logs" / "train_meta_model.log"

# Class mapping: 0=sell, 1=no-trade, 2=buy
CLASS_TO_LABEL = {0: -1, 1: 0, 2: 1}

PRICE_ONLY_COLS = [
    "return_1", "return_5", "volatility_20", "rsi_14", "macd", "macd_signal", "macd_hist",
    "atr_14", "ma_gap_20", "ma_gap_50", "is_london_session", "is_ny_session",
]


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


def run(
    no_forecasters: bool = False,
    price_only: bool = False,
    save_importance: bool = False,
    suffix: str = "",
) -> None:
    """Train meta-model, save test predictions."""
    _setup_logging()

    # Load baseline choice
    choice_path = MODELS_DIR / "baseline_choice.json"
    if not choice_path.exists():
        raise FileNotFoundError("Run train_price_model first to select baseline")

    with open(choice_path) as f:
        choice_data = json.load(f)
    baseline = choice_data["baseline"]

    # Load feature cols
    with open(MODELS_DIR / "baseline_feature_cols.json") as f:
        base_cols = json.load(f)

    if price_only:
        meta_cols = [c for c in PRICE_ONLY_COLS if c in base_cols]
    else:
        meta_cols = base_cols.copy()

    # Add forecaster cols unless disabled
    forecaster_path = FEATURES_DIR / "forecaster_predictions.csv"
    forecaster_cols = []
    if not no_forecasters and forecaster_path.exists():
        forecaster_cols = ["chronos_pred_return", "timesfm_pred_return"]
    meta_cols = meta_cols + forecaster_cols

    # Load data
    def _load_feature(name: str) -> pd.DataFrame:
        p = FEATURES_DIR / f"{name}.parquet"
        c = FEATURES_DIR / f"{name}.csv"
        if p.exists():
            return pd.read_parquet(p)
        return pd.read_csv(c)

    train = _load_feature("train")
    val = _load_feature("validation")
    test = _load_feature("test")

    if forecaster_path.exists():
        fc = pd.read_csv(forecaster_path)
        fc["timestamp"] = pd.to_datetime(fc["timestamp"], utc=True)
        train["timestamp"] = pd.to_datetime(train["timestamp"], utc=True)
        val["timestamp"] = pd.to_datetime(val["timestamp"], utc=True)
        test["timestamp"] = pd.to_datetime(test["timestamp"], utc=True)
        train = train.merge(fc, on="timestamp", how="left")
        val = val.merge(fc, on="timestamp", how="left")
        test = test.merge(fc, on="timestamp", how="left")

    train_val = pd.concat([train, val], ignore_index=True)

    X_train = train_val[meta_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = train_val["label_class"]
    y_sk = np.where(y == -1, 0, np.where(y == 0, 1, 2))

    if baseline == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
    else:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)

    model.fit(X_train, y_sk)
    logging.info("Meta-model trained with %d features", len(meta_cols))

    # Feature importance
    if save_importance:
        imp_path = MODELS_DIR / "meta_feature_importance.json"
        if hasattr(model, "feature_importances_"):
            imp = dict(zip(meta_cols, model.feature_importances_.tolist()))
        elif hasattr(model, "coef_"):
            imp = dict(zip(meta_cols, np.abs(model.coef_).sum(axis=0).tolist()))
        else:
            imp = {}
        with open(imp_path, "w") as f:
            json.dump(imp, f, indent=2)
        logging.info("Saved feature importance to %s", imp_path)

    # Save meta-model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_name = f"meta_model{suffix}.pkl" if suffix else "meta_model.pkl"
    cols_name = f"meta_feature_cols{suffix}.json" if suffix else "meta_feature_cols.json"
    with open(MODELS_DIR / model_name, "wb") as f:
        pickle.dump(model, f)
    with open(MODELS_DIR / cols_name, "w") as f:
        json.dump(meta_cols, f, indent=2)
    with open(MODELS_DIR / "meta_config.json", "w") as f:
        json.dump({"no_trade_threshold": NO_TRADE_THRESHOLD_PCT, "baseline": baseline}, f, indent=2)

    # Predict on test
    X_test = test[meta_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    probs = model.predict_proba(X_test)
    pred_class = model.predict(X_test)

    # probs: columns 0=sell, 1=no-trade, 2=buy
    P_sell = probs[:, 0]
    P_no_trade = probs[:, 1]
    P_buy = probs[:, 2]

    pred_label = np.array([CLASS_TO_LABEL[c] for c in pred_class])

    # Apply no-trade threshold: if max(P_buy, P_sell) < threshold, force no-trade
    below_threshold = np.maximum(P_buy, P_sell) < NO_TRADE_THRESHOLD_PCT
    pred_label[below_threshold] = 0

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pred_name = f"test_predictions{suffix}.csv" if suffix else "test_predictions.csv"
    out = pd.DataFrame({
        "timestamp": test["timestamp"],
        "P_buy": P_buy,
        "P_sell": P_sell,
        "P_no_trade": P_no_trade,
        "predicted_class": pred_label,
        "actual_class": test["label_class"].values,
        "future_return_30m": test["future_return_30m"].values,
    })
    out.to_csv(PREDICTIONS_DIR / pred_name, index=False)
    logging.info("Saved test predictions to %s", PREDICTIONS_DIR / pred_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-forecasters", action="store_true", help="Exclude chronos/timesfm features")
    parser.add_argument("--price-only", action="store_true", help="Use price+time features only")
    parser.add_argument("--save-importance", action="store_true", help="Save feature importance to JSON")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for model/pred filenames (e.g. _price_only)")
    args = parser.parse_args()
    run(
        no_forecasters=args.no_forecasters,
        price_only=args.price_only,
        save_importance=args.save_importance,
        suffix=args.suffix,
    )
