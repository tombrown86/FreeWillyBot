"""
Trend filter experiment — regression_v1 (strict alignment) validation suite.

Modes
-----
experiment  : baseline vs MA20 filtered, both strategies (default from initial run)
sweep       : MA 10/15/20/25/30 on 1h + MA 10/20 on 4h, regression only
walk-forward: month-by-month baseline vs filtered — checks consistency of filter benefit
cost-stress : filtered regression at 1x / 1.5x / 2x costs
stability   : full MA window sweep 5–40 (step 5) to confirm plateau, not spike

Usage
-----
python -m scripts.run_trend_filter_experiment --mode sweep
python -m scripts.run_trend_filter_experiment --mode walk-forward --ma-window 20
python -m scripts.run_trend_filter_experiment --mode cost-stress --ma-window 20
python -m scripts.run_trend_filter_experiment --mode stability

Outputs written to data/validation/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest_regression import (  # noqa: E402  (research script — private API ok)
    _apply_min_bars_between,
    _positions_from_pred,
    _profit_factor,
)
from src.config import (  # noqa: E402
    FX_SPREAD_PIPS,
    MR_HOLD_BARS,
    MR_LOOKBACK_BARS,
    MR_VOL_PCT,
    MR_ZSCORE_THRESHOLD,
)
from src.trend_filter import compute_trend_mask, print_trend_stats  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions_regression" / "test_predictions.parquet"
FEATURES_PATH = PROJECT_ROOT / "data" / "features_regression_core" / "test.parquet"
VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"

# ── Costs ─────────────────────────────────────────────────────────────────────
FX_SPREAD = FX_SPREAD_PIPS * 0.0001

# ── Regression defaults ───────────────────────────────────────────────────────
REG_TOP_PCT: float = 1.0
REG_PRED_THRESHOLD: float = 0.0
REG_VOL_PCT: int = 30
REG_MIN_BARS: int = 0

# ── MR soft-alignment threshold ───────────────────────────────────────────────
MR_SOFT_THRESHOLD: float = 0.0005


# ── Core helpers ──────────────────────────────────────────────────────────────

def _load_regression_data() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Predictions not found: {PREDICTIONS_PATH}")
    df = pd.read_parquet(PREDICTIONS_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    valid = np.isfinite(df["target_ret"]) & np.isfinite(df["pred"]) & np.isfinite(df["vol_6"])
    return df[valid].reset_index(drop=True)


def _build_filtered_positions(
    df: pd.DataFrame,
    ma_window: int = 20,
    resample_period: str = "1h",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pos_baseline, pos_filtered) for regression data."""
    pred = df["pred"].values.astype(float)
    vol = df["vol_6"].values.astype(float)

    pos_base = _positions_from_pred(pred, vol, REG_TOP_PCT, REG_VOL_PCT, REG_PRED_THRESHOLD)
    pos_base = _apply_min_bars_between(pos_base, REG_MIN_BARS)

    trend = compute_trend_mask(df["timestamp"], ma_window=ma_window, resample_period=resample_period)
    trend_up = trend["trend_up_1h"].fillna(False).astype(bool).values
    trend_down = trend["trend_down_1h"].fillna(False).astype(bool).values

    pos_filt = pos_base.copy()
    pos_filt[(pos_filt == 1) & ~trend_up] = 0
    pos_filt[(pos_filt == -1) & ~trend_down] = 0

    return pos_base, pos_filt


def _run_equity(
    positions: np.ndarray,
    ret: np.ndarray,
    cost_per_leg: float,
    with_costs: bool = True,
) -> tuple[float, int, float]:
    """Equity simulation for pre-computed positions. Returns (cum_ret, n_trades, max_dd)."""
    n = len(positions)
    equity = np.ones(n + 1)
    prev_pos = 0
    n_trades = 0

    for i in range(n):
        p = positions[i]
        if p != prev_pos:
            n_trades += 1
            if with_costs:
                legs = 1 if (prev_pos == 0 or p == 0) else 2
                equity[i + 1] = equity[i] * (1 - legs * cost_per_leg)
            else:
                equity[i + 1] = equity[i]
            prev_pos = p
        else:
            equity[i + 1] = equity[i]
        if p != 0:
            equity[i + 1] *= 1 + p * ret[i]

    cum_ret = float(equity[-1] - 1.0)
    eq = equity[1:]
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, 1.0)
    return cum_ret, n_trades, float(np.max(dd))


def _monthly_breakdown(
    positions: np.ndarray,
    ret: np.ndarray,
    timestamps: pd.Series,
    cost_per_leg: float,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {"timestamp": pd.to_datetime(timestamps, utc=True), "position": positions, "ret": ret}
    )
    df["month"] = df["timestamp"].dt.tz_convert(None).dt.to_period("M")
    df["period_ret"] = np.where(df["position"] != 0, df["position"] * df["ret"], 0.0)
    df["pos_change"] = (df["position"] != df["position"].shift(1).fillna(0)).astype(int)
    rows = []
    for month, grp in df.groupby("month"):
        gross = float((1 + grp["period_ret"]).prod() - 1.0)
        n_changes = int(grp["pos_change"].sum())
        net = gross - n_changes * cost_per_leg
        rows.append({"month": str(month), "net_ret": round(net, 6), "n_trades": n_changes})
    return pd.DataFrame(rows)


def _summary_metrics(
    cum_ret: float,
    n_trades: int,
    max_dd: float,
    positions: np.ndarray,
    ret: np.ndarray,
    timestamps: pd.Series,
    cost_per_leg: float,
) -> dict:
    pf = _profit_factor(ret, positions)
    monthly = _monthly_breakdown(positions, ret, timestamps, cost_per_leg)
    pos_months = int((monthly["net_ret"] > 0).sum())
    total_months = len(monthly)
    return {
        "cum_ret": round(cum_ret, 6),
        "profit_factor": round(pf, 4),
        "max_dd": round(max_dd, 6),
        "n_trades": n_trades,
        "positive_months": pos_months,
        "total_months": total_months,
        "pct_positive_months": round(100 * pos_months / max(total_months, 1), 1),
    }


def _print_comparison(rows: list[dict], labels: list[str] | None = None) -> None:
    if len(rows) < 2:
        return
    base, filt = rows[0], rows[1]
    lbl_b = (labels or ["baseline", "filtered"])[0]
    lbl_f = (labels or ["baseline", "filtered"])[1]
    print(f"  {'metric':<22} {lbl_b:>14} {lbl_f:>14} {'delta':>10}")
    print("  " + "-" * 62)
    for key, label in [
        ("cum_ret", "cum_ret"),
        ("profit_factor", "profit_factor"),
        ("max_dd", "max_dd"),
        ("n_trades", "n_trades"),
        ("pct_positive_months", "% positive months"),
    ]:
        bv = base.get(key, 0)
        fv = filt.get(key, 0)
        if isinstance(bv, float) and isinstance(fv, float):
            print(f"  {label:<22} {bv:>14.4f} {fv:>14.4f} {fv - bv:>+10.4f}")
        else:
            print(f"  {label:<22} {bv:>14} {fv:>14} {fv - bv:>+10}")
    pf_ok = filt.get("profit_factor", 0) > base.get("profit_factor", 0)
    dd_ok = filt.get("max_dd", 1) < base.get("max_dd", 1)
    tc_ok = filt.get("n_trades", 0) > 0.1 * base.get("n_trades", 1)
    if pf_ok and dd_ok and tc_ok:
        verdict = "PASS"
    elif not tc_ok:
        verdict = "REJECT — trade count collapsed"
    elif not pf_ok and not dd_ok:
        verdict = "REJECT — no PF or DD improvement"
    else:
        verdict = "MARGINAL"
    print(f"\n  Verdict: {verdict}")


# ── Regression experiment (original mode) ────────────────────────────────────

def run_regression_experiment(ma_window: int = 20) -> tuple[list[dict], pd.DataFrame]:
    """Baseline vs strict-trend-filtered regression_v1."""
    df = _load_regression_data()
    ret = df["target_ret"].values.astype(float)
    timestamps = df["timestamp"]

    pos_base, pos_filt = _build_filtered_positions(df, ma_window=ma_window)

    rows: list[dict] = []
    monthly_frames: list[pd.DataFrame] = []
    for label, positions in [("baseline", pos_base), ("strict_trend", pos_filt)]:
        cum_ret, n_trades, max_dd = _run_equity(positions, ret, FX_SPREAD)
        m = _summary_metrics(cum_ret, n_trades, max_dd, positions, ret, timestamps, FX_SPREAD)
        m.update({"strategy": "regression_v1", "mode": label, "ma_window": ma_window, "resample": "1h"})
        rows.append(m)
        monthly = _monthly_breakdown(positions, ret, timestamps, FX_SPREAD)
        monthly["strategy"] = "regression_v1"
        monthly["mode"] = label
        monthly["ma_window"] = ma_window
        monthly_frames.append(monthly)
    return rows, pd.concat(monthly_frames, ignore_index=True)


# ── MR experiment (original mode) ────────────────────────────────────────────

def _mr_positions_vectorized(
    df: pd.DataFrame,
    zscore_threshold: float = MR_ZSCORE_THRESHOLD,
    hold_bars: int = MR_HOLD_BARS,
    lookback: int = MR_LOOKBACK_BARS,
    vol_pct: int = MR_VOL_PCT,
    trend_strength: "np.ndarray | None" = None,
    soft_threshold: float = MR_SOFT_THRESHOLD,
) -> np.ndarray:
    n = len(df)
    gap = df["ma_20_gap"].values.astype(float)
    zscore = np.zeros(n, dtype=float)
    for i in range(lookback, n):
        window = gap[i - lookback : i + 1]
        mu = float(np.mean(window))
        std = float(np.std(window, ddof=0))
        if std > 1e-12:
            zscore[i] = (gap[i] - mu) / std

    is_london = df.get("is_london_session", pd.Series(1, index=df.index)) == 1
    is_ny = df.get("is_ny_session", pd.Series(0, index=df.index)) == 1
    in_session = (is_london | is_ny).values
    in_event = (df.get("is_event_window", pd.Series(0, index=df.index)) == 1).values
    vol = df["vol_6"].fillna(0).values.astype(float)
    vol_ok = (
        vol >= float(np.percentile(vol[vol > 0], 100 - vol_pct))
        if vol_pct > 0 and (vol > 0).any()
        else np.ones(n, dtype=bool)
    )
    can_trade = in_session & ~in_event & vol_ok

    if trend_strength is not None:
        ts = np.nan_to_num(trend_strength, nan=0.0)
        suppress_long = ts < -soft_threshold
        suppress_short = ts > soft_threshold
    else:
        suppress_long = np.zeros(n, dtype=bool)
        suppress_short = np.zeros(n, dtype=bool)

    positions = np.zeros(n, dtype=int)
    current_pos = 0
    bars_held = 0
    for i in range(n):
        if current_pos != 0:
            bars_held += 1
            if bars_held >= hold_bars:
                current_pos = 0
                bars_held = 0
        if can_trade[i] and current_pos == 0:
            if zscore[i] >= zscore_threshold and not suppress_short[i]:
                current_pos = -1
                bars_held = 1
            elif zscore[i] <= -zscore_threshold and not suppress_long[i]:
                current_pos = 1
                bars_held = 1
        positions[i] = current_pos
    return positions


def run_mr_experiment(ma_window: int = 20) -> tuple[list[dict], pd.DataFrame]:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features not found: {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    ret = np.nan_to_num(df["ret_1"].values.astype(float), nan=0.0)
    timestamps = df["timestamp"]
    trend = compute_trend_mask(df["timestamp"], ma_window=ma_window)
    trend_strength = trend["trend_strength_1h"].values.astype(float)
    rows: list[dict] = []
    monthly_frames: list[pd.DataFrame] = []
    for label, ts_arg in [("baseline", None), ("soft_trend", trend_strength)]:
        positions = _mr_positions_vectorized(df, trend_strength=ts_arg)
        cum_ret, n_trades, max_dd = _run_equity(positions, ret, FX_SPREAD)
        m = _summary_metrics(cum_ret, n_trades, max_dd, positions, ret, timestamps, FX_SPREAD)
        m.update({"strategy": "mean_reversion_v1", "mode": label, "ma_window": ma_window, "resample": "1h"})
        rows.append(m)
        monthly = _monthly_breakdown(positions, ret, timestamps, FX_SPREAD)
        monthly["strategy"] = "mean_reversion_v1"
        monthly["mode"] = label
        monthly["ma_window"] = ma_window
        monthly_frames.append(monthly)
    return rows, pd.concat(monthly_frames, ignore_index=True)


# ── Mode: sweep ───────────────────────────────────────────────────────────────

def mode_sweep() -> None:
    """Run regression with MA 10/15/20/25/30 on 1h and MA 10/20 on 4h."""
    df = _load_regression_data()
    ret = df["target_ret"].values.astype(float)
    timestamps = df["timestamp"]

    combos: list[tuple[str, int]] = [
        ("1h", 10), ("1h", 15), ("1h", 20), ("1h", 25), ("1h", 30),
        ("4h", 10), ("4h", 20),
    ]

    # Pre-compute baseline once
    pos_base = _positions_from_pred(
        df["pred"].values.astype(float),
        df["vol_6"].values.astype(float),
        REG_TOP_PCT, REG_VOL_PCT, REG_PRED_THRESHOLD,
    )
    pos_base = _apply_min_bars_between(pos_base, REG_MIN_BARS)
    cum_base, n_base, dd_base = _run_equity(pos_base, ret, FX_SPREAD)
    pf_base = _profit_factor(ret, pos_base)
    monthly_base = _monthly_breakdown(pos_base, ret, timestamps, FX_SPREAD)
    pm_base = int((monthly_base["net_ret"] > 0).sum())

    print(f"\n  {'resample':<10} {'MA':>4} {'PF':>8} {'cum_ret':>10} {'max_dd':>10} {'n_trades':>10} {'%pos_mo':>9}  verdict")
    print("  " + "-" * 80)
    print(f"  {'baseline':<10} {'—':>4} {pf_base:>8.4f} {cum_base:>10.4f} {dd_base:>10.4f} {n_base:>10} {100*pm_base/max(len(monthly_base),1):>8.1f}%  (reference)")

    sweep_rows: list[dict] = []
    for resample, ma in combos:
        pos_base_i, pos_filt = _build_filtered_positions(df, ma_window=ma, resample_period=resample)
        cum, n_trades, max_dd = _run_equity(pos_filt, ret, FX_SPREAD)
        pf = _profit_factor(ret, pos_filt)
        monthly = _monthly_breakdown(pos_filt, ret, timestamps, FX_SPREAD)
        pm = int((monthly["net_ret"] > 0).sum())
        pct_pm = 100 * pm / max(len(monthly), 1)
        dpf = pf - pf_base
        verdict = "PASS" if (pf > pf_base and max_dd < dd_base and n_trades > 0.1 * n_base) else "—"
        print(
            f"  {resample:<10} {ma:>4} {pf:>8.4f} {cum:>10.4f} {max_dd:>10.4f} {n_trades:>10} {pct_pm:>8.1f}%  {verdict}  (dPF={dpf:+.4f})"
        )
        sweep_rows.append({
            "resample": resample, "ma_window": ma,
            "profit_factor": round(pf, 4), "cum_ret": round(cum, 6),
            "max_dd": round(max_dd, 6), "n_trades": n_trades,
            "positive_months": pm, "total_months": len(monthly),
            "pct_positive_months": round(pct_pm, 1), "delta_pf": round(dpf, 4),
        })

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    path = VALIDATION_DIR / "trend_filter_sweep.csv"
    pd.DataFrame(sweep_rows).to_csv(path, index=False)
    print(f"\n  Saved → {path}")


# ── Mode: walk-forward ────────────────────────────────────────────────────────

def mode_walk_forward(ma_window: int = 20, resample_period: str = "1h") -> None:
    """Per-month baseline vs filtered — checks consistency of filter benefit."""
    df = _load_regression_data()
    ret = df["target_ret"].values.astype(float)
    timestamps = df["timestamp"]

    pos_base, pos_filt = _build_filtered_positions(df, ma_window=ma_window, resample_period=resample_period)

    monthly_base = _monthly_breakdown(pos_base, ret, timestamps, FX_SPREAD)
    monthly_filt = _monthly_breakdown(pos_filt, ret, timestamps, FX_SPREAD)

    merged = monthly_base.merge(monthly_filt, on="month", suffixes=("_base", "_filt"))
    merged["filter_better"] = merged["net_ret_filt"] > merged["net_ret_base"]
    merged["delta_net_ret"] = merged["net_ret_filt"] - merged["net_ret_base"]

    n_months = len(merged)
    n_filter_better = int(merged["filter_better"].sum())
    base_pos_months = int((merged["net_ret_base"] > 0).sum())
    filt_pos_months = int((merged["net_ret_filt"] > 0).sum())

    print(f"\n  MA{ma_window} {resample_period} — monthly walk-forward")
    print(f"  {'month':<10} {'baseline':>10} {'filtered':>10} {'delta':>10} {'filter_wins':>12}")
    print("  " + "-" * 56)
    for _, row in merged.iterrows():
        mark = "✓" if row["filter_better"] else " "
        print(
            f"  {row['month']:<10} {row['net_ret_base']:>10.4f} {row['net_ret_filt']:>10.4f}"
            f" {row['delta_net_ret']:>+10.4f}  {mark}"
        )
    print("  " + "-" * 56)
    cum_base_monthly = merged["net_ret_base"].sum()
    cum_filt_monthly = merged["net_ret_filt"].sum()
    print(f"  {'TOTAL':<10} {cum_base_monthly:>10.4f} {cum_filt_monthly:>10.4f} {cum_filt_monthly-cum_base_monthly:>+10.4f}")
    print(f"\n  Filter wins   : {n_filter_better}/{n_months} months ({100*n_filter_better/max(n_months,1):.0f}%)")
    print(f"  Positive months: baseline={base_pos_months}/{n_months}  filtered={filt_pos_months}/{n_months}")

    verdict = (
        "PASS — filter improves majority of months"
        if n_filter_better > n_months * 0.5
        else "MARGINAL — filter helps fewer than half the months"
    )
    print(f"  Verdict: {verdict}")

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    path = VALIDATION_DIR / "trend_filter_walkforward.csv"
    merged.to_csv(path, index=False)
    print(f"  Saved → {path}")


# ── Mode: cost-stress ─────────────────────────────────────────────────────────

def mode_cost_stress(ma_window: int = 20, resample_period: str = "1h") -> None:
    """Run filtered regression at 1x / 1.5x / 2x spread to check edge fragility."""
    df = _load_regression_data()
    ret = df["target_ret"].values.astype(float)
    timestamps = df["timestamp"]

    pos_base, pos_filt = _build_filtered_positions(df, ma_window=ma_window, resample_period=resample_period)

    multipliers = [1.0, 1.5, 2.0]

    print(f"\n  MA{ma_window} {resample_period} — cost stress (base spread = {FX_SPREAD_PIPS} pips)")
    print(f"  {'run':<22} {'cost_mult':>10} {'PF':>8} {'cum_ret':>10} {'max_dd':>10} {'n_trades':>10}")
    print("  " + "-" * 72)

    rows = []
    for label, positions in [("baseline", pos_base), (f"strict_trend_MA{ma_window}", pos_filt)]:
        for mult in multipliers:
            cost = FX_SPREAD * mult
            cum, n_trades, max_dd = _run_equity(positions, ret, cost)
            pf = _profit_factor(ret, positions)
            monthly = _monthly_breakdown(positions, ret, timestamps, cost)
            pm = int((monthly["net_ret"] > 0).sum())
            run_label = f"{label} {mult:.1f}x"
            print(f"  {run_label:<22} {mult:>10.1f} {pf:>8.4f} {cum:>10.4f} {max_dd:>10.4f} {n_trades:>10}")
            rows.append({
                "mode": label, "cost_mult": mult, "cost_pips": round(FX_SPREAD_PIPS * mult, 2),
                "profit_factor": round(pf, 4), "cum_ret": round(cum, 6),
                "max_dd": round(max_dd, 6), "n_trades": n_trades,
                "positive_months": pm, "total_months": len(monthly),
            })
        print()

    # Verdict: filtered at 2x should still beat baseline at 1x
    filt_2x = next(r for r in rows if r["mode"] != "baseline" and r["cost_mult"] == 2.0)
    base_1x = next(r for r in rows if r["mode"] == "baseline" and r["cost_mult"] == 1.0)
    if filt_2x["profit_factor"] > base_1x["profit_factor"]:
        verdict = "ROBUST — filtered at 2x cost still beats unfiltered baseline at 1x cost"
    elif filt_2x["cum_ret"] > 0:
        verdict = "MODERATE — edge survives at 2x but margin is thin"
    else:
        verdict = "FRAGILE — edge disappears at 2x cost"
    print(f"  Verdict: {verdict}")

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    path = VALIDATION_DIR / "trend_filter_cost_stress.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved → {path}")


# ── Mode: stability ───────────────────────────────────────────────────────────

def mode_stability() -> None:
    """Sweep MA windows 5–40 (step 5) on 1h to confirm the benefit is a plateau."""
    df = _load_regression_data()
    ret = df["target_ret"].values.astype(float)
    timestamps = df["timestamp"]

    pos_base = _positions_from_pred(
        df["pred"].values.astype(float),
        df["vol_6"].values.astype(float),
        REG_TOP_PCT, REG_VOL_PCT, REG_PRED_THRESHOLD,
    )
    pos_base = _apply_min_bars_between(pos_base, REG_MIN_BARS)
    cum_base, n_base, dd_base = _run_equity(pos_base, ret, FX_SPREAD)
    pf_base = _profit_factor(ret, pos_base)
    monthly_base = _monthly_breakdown(pos_base, ret, timestamps, FX_SPREAD)
    pm_base = int((monthly_base["net_ret"] > 0).sum())

    windows = list(range(5, 45, 5))

    print(f"\n  1h MA stability sweep (regression_v1 strict alignment)")
    print(f"  {'MA':>5} {'PF':>8} {'cum_ret':>10} {'max_dd':>10} {'n_trades':>10} {'%pos_mo':>9}  dPF")
    print("  " + "-" * 70)
    print(f"  {'base':>5} {pf_base:>8.4f} {cum_base:>10.4f} {dd_base:>10.4f} {n_base:>10} {100*pm_base/max(len(monthly_base),1):>8.1f}%  (reference)")

    rows = []
    for ma in windows:
        _, pos_filt = _build_filtered_positions(df, ma_window=ma, resample_period="1h")
        cum, n_trades, max_dd = _run_equity(pos_filt, ret, FX_SPREAD)
        pf = _profit_factor(ret, pos_filt)
        monthly = _monthly_breakdown(pos_filt, ret, timestamps, FX_SPREAD)
        pm = int((monthly["net_ret"] > 0).sum())
        pct_pm = 100 * pm / max(len(monthly), 1)
        dpf = pf - pf_base
        flag = " ◄" if pf > 1.2 else ""
        print(f"  {ma:>5} {pf:>8.4f} {cum:>10.4f} {max_dd:>10.4f} {n_trades:>10} {pct_pm:>8.1f}%  {dpf:+.4f}{flag}")
        rows.append({
            "ma_window": ma, "profit_factor": round(pf, 4), "cum_ret": round(cum, 6),
            "max_dd": round(max_dd, 6), "n_trades": n_trades,
            "positive_months": pm, "total_months": len(monthly),
            "pct_positive_months": round(pct_pm, 1), "delta_pf": round(dpf, 4),
        })

    # Plateau check: count how many windows beat baseline PF
    n_above = sum(1 for r in rows if r["profit_factor"] > pf_base)
    verdict = (
        f"PLATEAU — {n_above}/{len(windows)} MA windows beat baseline PF ({pf_base:.4f})"
        if n_above >= len(windows) // 2
        else f"SPIKE — only {n_above}/{len(windows)} windows beat baseline; benefit may be window-specific"
    )
    print(f"\n  Verdict: {verdict}")

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    path = VALIDATION_DIR / "trend_filter_stability.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Trend filter v1 validation suite")
    parser.add_argument(
        "--mode",
        choices=["experiment", "sweep", "walk-forward", "cost-stress", "stability", "all"],
        default="experiment",
        help="Test mode (default: experiment)",
    )
    parser.add_argument(
        "--strategy",
        choices=["regression", "mean_reversion", "both"],
        default="both",
        help="Strategy for experiment mode (default: both)",
    )
    parser.add_argument(
        "--ma-window", type=int, default=20, metavar="N",
        help="MA window on higher-timeframe bars (default: 20)",
    )
    parser.add_argument(
        "--resample", default="1h", metavar="PERIOD",
        help="Resample period for walk-forward / cost-stress (default: 1h)",
    )
    args = parser.parse_args()

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode in ("experiment", "all"):
        all_rows: list[dict] = []
        all_monthly: list[pd.DataFrame] = []
        if args.strategy in ("regression", "both"):
            print(f"\n{'='*60}")
            print(f"  regression_v1  [strict alignment, MA{args.ma_window}]")
            print(f"{'='*60}")
            print("  Computing 1h trend mask...")
            trend_preview = compute_trend_mask(
                _load_regression_data()["timestamp"], ma_window=args.ma_window
            )
            print_trend_stats(trend_preview)
            rows, monthly = run_regression_experiment(ma_window=args.ma_window)
            all_rows.extend(rows)
            all_monthly.append(monthly)
            _print_comparison(rows)
        if args.strategy in ("mean_reversion", "both"):
            print(f"\n{'='*60}")
            print(f"  mean_reversion_v1  [soft alignment, MA{args.ma_window}]")
            print(f"{'='*60}")
            rows, monthly = run_mr_experiment(ma_window=args.ma_window)
            all_rows.extend(rows)
            all_monthly.append(monthly)
            _print_comparison(rows)
        col_order = [
            "strategy", "mode", "ma_window", "resample", "cum_ret", "profit_factor",
            "max_dd", "n_trades", "positive_months", "total_months", "pct_positive_months",
        ]
        report_df = pd.DataFrame(all_rows)
        report_df = report_df[[c for c in col_order if c in report_df.columns]]
        report_df.to_csv(VALIDATION_DIR / "trend_filter_report.csv", index=False)
        pd.concat(all_monthly, ignore_index=True).to_csv(
            VALIDATION_DIR / "trend_filter_monthly.csv", index=False
        )

    if args.mode in ("sweep", "all"):
        print(f"\n{'='*60}")
        print("  MA SWEEP — regression_v1")
        print(f"{'='*60}")
        mode_sweep()

    if args.mode in ("walk-forward", "all"):
        print(f"\n{'='*60}")
        print("  WALK-FORWARD — regression_v1")
        print(f"{'='*60}")
        mode_walk_forward(ma_window=args.ma_window, resample_period=args.resample)

    if args.mode in ("cost-stress", "all"):
        print(f"\n{'='*60}")
        print("  COST STRESS — regression_v1")
        print(f"{'='*60}")
        mode_cost_stress(ma_window=args.ma_window, resample_period=args.resample)

    if args.mode in ("stability", "all"):
        print(f"\n{'='*60}")
        print("  STABILITY SWEEP — regression_v1")
        print(f"{'='*60}")
        mode_stability()

    print(f"\nAll outputs saved to {VALIDATION_DIR}/\n")


if __name__ == "__main__":
    main()
