"""
Month-level regime analysis — compare good vs bad months.

Per month: avg_vol_6, avg_trend_strength (abs(ret_12)), avg_return_dispersion (std(target_ret)).
Compares characteristics of positive vs negative months to validate regime hypothesis.
Output: data/backtests_regression/month_regime_analysis.csv + summary stats + optional scatter plot.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions_regression"
BACKTESTS_DIR = PROJECT_ROOT / "data" / "backtests_regression"

TOP_PCT = 0.25
VOL_PCT = 20
PRED_THRESHOLD = 0.00005


def run(plot: bool = True) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    test_pred_path = PREDICTIONS_DIR / "test_predictions.parquet"
    if not test_pred_path.exists():
        logging.error("test_predictions.parquet not found. Run predict_regression_test first.")
        return 1

    df = pd.read_parquet(test_pred_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["month"] = df["timestamp"].dt.strftime("%Y-%m")

    if "ret_12" not in df.columns:
        logging.error("ret_12 missing — re-run predict_regression_test after the regime enrichment patch.")
        return 1

    from src.backtest_regression import _apply_min_bars_between, _positions_from_pred, _profit_factor, _run_single
    from src.config import FX_SPREAD_PIPS

    cost_per_leg = FX_SPREAD_PIPS * 0.0001
    months = sorted(df["month"].dropna().unique())

    rows = []
    for m in months[:12]:
        w = df[df["month"] == m]
        if len(w) < 100:
            continue

        ret = w["target_ret"].values.astype(float)
        pred = w["pred"].values.astype(float)
        vol = w["vol_6"].fillna(0).values.astype(float)
        ret_12 = w["ret_12"].fillna(0).values.astype(float)

        valid = np.isfinite(ret) & np.isfinite(pred)
        net_ret, n_trades, max_dd = _run_single(
            ret[valid], pred[valid], vol[valid], TOP_PCT, VOL_PCT, PRED_THRESHOLD, 0, cost_per_leg, with_costs=True
        )

        rows.append({
            "month": m,
            "net_return": float(net_ret),
            "n_trades": int(n_trades),
            "max_dd": float(max_dd),
            "avg_vol_6": float(np.nanmean(vol)),
            "avg_trend_strength": float(np.nanmean(np.abs(ret_12))),
            "avg_return_dispersion": float(np.nanstd(ret)),
            "good_month": int(net_ret > 0),
        })

    out = pd.DataFrame(rows)
    BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(BACKTESTS_DIR / "month_regime_analysis.csv", index=False)
    logging.info("Saved %s", BACKTESTS_DIR / "month_regime_analysis.csv")

    # Summary comparison
    good = out[out["good_month"] == 1]
    bad = out[out["good_month"] == 0]
    logging.info("Good months (%d): avg_vol_6=%.6f  avg_trend=%.6f  avg_dispersion=%.6f",
                 len(good), good["avg_vol_6"].mean(), good["avg_trend_strength"].mean(), good["avg_return_dispersion"].mean())
    logging.info("Bad  months (%d): avg_vol_6=%.6f  avg_trend=%.6f  avg_dispersion=%.6f",
                 len(bad), bad["avg_vol_6"].mean(), bad["avg_trend_strength"].mean(), bad["avg_return_dispersion"].mean())

    if plot:
        _scatter_plot(out)

    return 0


def _scatter_plot(out: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    colors = ["#2ecc71" if g else "#e74c3c" for g in out["good_month"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(out["avg_vol_6"], out["avg_trend_strength"], c=colors, s=80, edgecolors="k", linewidths=0.5)
    for _, row in out.iterrows():
        axes[0].annotate(row["month"], (row["avg_vol_6"], row["avg_trend_strength"]), fontsize=7, ha="center", va="bottom")
    axes[0].set_xlabel("Avg vol_6")
    axes[0].set_ylabel("Avg trend strength (abs ret_12)")
    axes[0].set_title("Vol vs trend strength by month")

    axes[1].scatter(out["avg_vol_6"], out["avg_return_dispersion"], c=colors, s=80, edgecolors="k", linewidths=0.5)
    for _, row in out.iterrows():
        axes[1].annotate(row["month"], (row["avg_vol_6"], row["avg_return_dispersion"]), fontsize=7, ha="center", va="bottom")
    axes[1].set_xlabel("Avg vol_6")
    axes[1].set_ylabel("Return dispersion (std)")
    axes[1].set_title("Vol vs return dispersion by month")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#2ecc71", label="Positive month"), Patch(facecolor="#e74c3c", label="Negative month")]
    fig.legend(handles=legend_elements, loc="upper right")
    fig.suptitle("Month regime analysis — good vs bad months")
    fig.tight_layout()
    BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(BACKTESTS_DIR / "month_regime_analysis.png", dpi=100)
    plt.close(fig)
    logging.info("Saved scatter plot to %s", BACKTESTS_DIR / "month_regime_analysis.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plot", action="store_true", help="Skip scatter plot")
    args = parser.parse_args()
    sys.exit(run(plot=not args.no_plot))
