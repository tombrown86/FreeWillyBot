"""
Batch 3 — Train regression models for return prediction.

Trains LinearRegression, Ridge, ElasticNet, XGBRegressor for each horizon (3, 6, 12).
Evaluates on validation: correlation, MAE, RMSE, directional hit rate, net trading result.
Saves predictions, distribution plots, model comparison, best horizon-model pair, and best model for inference.
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FX_SPREAD_PIPS, REGRESSION_HORIZONS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / "data" / "features_regression_core"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions_regression"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
LOG_FILE = PROJECT_ROOT / "data" / "logs" / "train_regression.log"

FX_SPREAD = FX_SPREAD_PIPS * 0.0001
P_LONG = 90
P_SHORT = 10

MODEL_ORDER = ["linear", "ridge", "elasticnet", "xgb"]


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


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load train and validation from features_regression_core. Return feature columns."""
    def _load(name: str) -> pd.DataFrame:
        p = FEATURES_DIR / f"{name}.parquet"
        c = FEATURES_DIR / f"{name}.csv"
        if p.exists():
            return pd.read_parquet(p)
        return pd.read_csv(c)

    train = _load("train")
    val = _load("validation")

    target_cols = [f"target_ret_{h}" for h in REGRESSION_HORIZONS]
    feature_cols = [c for c in train.columns if c not in ("timestamp",) + tuple(target_cols)]
    feature_cols = [c for c in feature_cols if train[c].dtype in ("float64", "int64") or "float" in str(train[c].dtype)]

    return train, val, feature_cols


def _mini_backtest_regression(
    ret: np.ndarray,
    pred: np.ndarray,
    spread: float,
    p_long: float = 90,
    p_short: float = 10,
) -> tuple[float, int]:
    """Cumulative return after costs. Long when pred > p_long percentile, short when pred < p_short percentile."""
    th_long = np.percentile(pred, p_long)
    th_short = np.percentile(pred, p_short)
    positions = np.where(pred >= th_long, 1, np.where(pred <= th_short, -1, 0))

    prev_pos = 0
    cum = 1.0
    n_trades = 0
    for i in range(len(positions)):
        p = positions[i]
        if p != prev_pos:
            n_trades += 1
            legs = 1 if (prev_pos == 0 or p == 0) else 2
            cum *= 1 - legs * spread
        if p != 0:
            cum *= 1 + p * ret[i]
        prev_pos = p
    return cum - 1.0, n_trades


def _get_models() -> dict:
    """Return dict of model_name -> (model_instance, display_name)."""
    from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
    from xgboost import XGBRegressor

    return {
        "linear": (LinearRegression(), "LinearRegression"),
        "ridge": (Ridge(alpha=1.0, random_state=42), "Ridge"),
        "elasticnet": (ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42), "ElasticNet"),
        "xgb": (
            XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
            "XGBRegressor",
        ),
    }


def _compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    spread: float,
) -> dict:
    """Compute correlation, MAE, RMSE, directional hit rate, net return, n_trades."""
    valid = np.isfinite(pred) & np.isfinite(target)
    pred_v = pred[valid]
    target_v = target[valid]

    if len(pred_v) < 10:
        return {
            "correlation": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "directional_hit_rate": np.nan,
            "net_return": np.nan,
            "n_trades": 0,
        }

    corr = np.corrcoef(pred_v, target_v)[0, 1] if np.std(pred_v) > 0 and np.std(target_v) > 0 else np.nan
    mae = np.mean(np.abs(pred_v - target_v))
    rmse = np.sqrt(np.mean((pred_v - target_v) ** 2))

    sign_pred = np.sign(pred_v)
    sign_target = np.sign(target_v)
    nonzero = (sign_pred != 0) | (sign_target != 0)
    if nonzero.sum() > 0:
        dir_hit = np.mean(sign_pred[nonzero] == sign_target[nonzero])
    else:
        dir_hit = np.nan

    net_ret, n_trades = _mini_backtest_regression(target_v, pred_v, spread, P_LONG, P_SHORT)

    return {
        "correlation": float(corr),
        "mae": float(mae),
        "rmse": float(rmse),
        "directional_hit_rate": float(dir_hit),
        "net_return": float(net_ret),
        "n_trades": int(n_trades),
    }


def _top_vs_middle(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Mean realized return in top 10% vs middle 40-60% of predictions."""
    valid = np.isfinite(pred) & np.isfinite(target)
    pred_v = pred[valid]
    target_v = target[valid]
    if len(pred_v) < 100:
        return np.nan, np.nan

    p90 = np.percentile(pred_v, 90)
    p40 = np.percentile(pred_v, 40)
    p60 = np.percentile(pred_v, 60)

    top_mask = pred_v >= p90
    mid_mask = (pred_v >= p40) & (pred_v <= p60)

    top_mean = np.mean(target_v[top_mask]) if top_mask.sum() > 0 else np.nan
    mid_mean = np.mean(target_v[mid_mask]) if mid_mask.sum() > 0 else np.nan
    return float(top_mean), float(mid_mean)


def _plot_distributions(pred_df: pd.DataFrame, log: logging.Logger) -> None:
    """Plot predicted return distribution for each model, one figure per model with 3 subplots (h=3,6,12)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping distribution plots")
        return

    plots_dir = PREDICTIONS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for model in MODEL_ORDER:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for idx, h in enumerate(REGRESSION_HORIZONS):
            col = f"pred_{model}_{h}"
            if col not in pred_df.columns:
                continue
            ax = axes[idx]
            ax.hist(pred_df[col].dropna(), bins=80, density=True, alpha=0.7, edgecolor="black", linewidth=0.3)
            ax.set_title(f"h={h}")
            ax.set_xlabel("Predicted return")
            ax.set_ylabel("Density")
        fig.suptitle(f"Predicted return distribution — {model}")
        fig.tight_layout()
        fig.savefig(plots_dir / f"pred_dist_{model}.png", dpi=100)
        plt.close(fig)
        log.info("Saved plot: %s", plots_dir / f"pred_dist_{model}.png")


def _select_best(comparison: pd.DataFrame) -> dict:
    """Select single best horizon-model. Primary: correlation; secondary: dir hit, net return; tie-break: simpler model."""
    model_rank = {"linear": 0, "ridge": 1, "elasticnet": 2, "xgb": 3}
    df = comparison.copy()
    df["model_rank"] = df["model"].map(model_rank)
    df = df.sort_values(
        by=["correlation", "directional_hit_rate", "net_return", "model_rank", "horizon"],
        ascending=[False, False, False, True, True],
        na_position="last",
    )
    best = df.iloc[0]
    return {
        "horizon": int(best["horizon"]),
        "model": str(best["model"]),
        "correlation": float(best["correlation"]),
        "mae": float(best["mae"]),
        "rmse": float(best["rmse"]),
        "directional_hit_rate": float(best["directional_hit_rate"]),
        "net_return": float(best["net_return"]),
        "n_trades": int(best["n_trades"]),
    }


def run() -> None:
    """Full pipeline: load, train, evaluate, save predictions, plots, comparison, best model."""
    _setup_logging()
    log = logging.getLogger()

    if not FEATURES_DIR.exists():
        log.error("Features dir not found: %s. Run build_features_regression_core first.", FEATURES_DIR)
        raise FileNotFoundError(f"Run build_features_regression_core first: {FEATURES_DIR}")

    train, val, feature_cols = _load_data()
    log.info("Loaded train: %d rows, val: %d rows, features: %d", len(train), len(val), len(feature_cols))

    X_train = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_val = val[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    pred_cols = {}
    results = []
    models = _get_models()

    for h in REGRESSION_HORIZONS:
        target_col = f"target_ret_{h}"
        y_train = train[target_col].values
        y_val = val[target_col].values

        for model_name, (model, _) in models.items():
            model.fit(X_train, y_train)
            pred_val = model.predict(X_val)
            pred_cols[f"pred_{model_name}_{h}"] = pred_val

            metrics = _compute_metrics(pred_val, y_val, FX_SPREAD)
            top_mean, mid_mean = _top_vs_middle(pred_val, y_val)

            log.info(
                "%s h=%d: corr=%.4f MAE=%.6f RMSE=%.6f dir_hit=%.4f net_ret=%.4f n_trades=%d | top10%%=%.6f mid40-60%%=%.6f",
                model_name,
                h,
                metrics["correlation"],
                metrics["mae"],
                metrics["rmse"],
                metrics["directional_hit_rate"],
                metrics["net_return"],
                metrics["n_trades"],
                top_mean,
                mid_mean,
            )

            results.append({
                "horizon": h,
                "model": model_name,
                **metrics,
                "top10_mean_realized": top_mean,
                "mid40_60_mean_realized": mid_mean,
            })

    pred_df = val[["timestamp"] + [f"target_ret_{h}" for h in REGRESSION_HORIZONS]].copy()
    for col, vals in pred_cols.items():
        pred_df[col] = vals

    pred_df.to_parquet(PREDICTIONS_DIR / "validation_predictions.parquet", index=False)
    pred_df.to_csv(PREDICTIONS_DIR / "validation_predictions.csv", index=False)
    log.info("Saved predictions to %s", PREDICTIONS_DIR / "validation_predictions.parquet")

    _plot_distributions(pred_df, log)

    comparison = pd.DataFrame(results)
    comparison.to_csv(PREDICTIONS_DIR / "model_comparison.csv", index=False)
    log.info("Saved model comparison to %s", PREDICTIONS_DIR / "model_comparison.csv")

    best = _select_best(comparison)
    with open(PREDICTIONS_DIR / "best_model.json", "w") as f:
        json.dump(best, f, indent=2)
    log.info("Best model: horizon=%d model=%s corr=%.4f", best["horizon"], best["model"], best["correlation"])

    # Retrain best model on train+val for inference (Batch 4)
    train_val = pd.concat([train, val], ignore_index=True)
    X_full = train_val[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    target_col = f"target_ret_{best['horizon']}"
    y_full = train_val[target_col].values

    model_name = best["model"]
    model, _ = _get_models()[model_name]
    model.fit(X_full, y_full)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "regression_best.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODELS_DIR / "regression_feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(MODELS_DIR / "regression_best_config.json", "w") as f:
        json.dump({"horizon": best["horizon"], "model": model_name, "target_col": target_col}, f, indent=2)
    log.info("Saved best model to %s (horizon=%d, model=%s)", MODELS_DIR / "regression_best.pkl", best["horizon"], model_name)


if __name__ == "__main__":
    run()
