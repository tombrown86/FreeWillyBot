"""
Portfolio engine backtest — replay regression_v2_trendfilter signals with
three sizing modes and compare key performance metrics.

Modes
-----
fixed     : always 1.0 unit (current baseline behaviour)
vol_only  : base × volatility_targeting_multiplier
full      : base × vol × trend_strength × drawdown × loss_streak multipliers

The backtest uses:
  data/predictions_regression/test_predictions.parquet  — model predictions
  data/processed/price/EURUSD_5min_clean.parquet        — close prices for vol
  src/trend_filter.compute_trend_mask()                 — 4h MA10 trend state
  src/config.py   RV2_*                                 — signal thresholds
  src/config_portfolio.py                               — sizing params

Signal generation exactly mirrors live_signal_regression_v2_trendfilter:
  1. vol_6 in top RV2_VOL_PCT %
  2. pred in top/bottom RV2_TOP_PCT % (based on full-dataset percentiles)
  3. |pred| >= RV2_PRED_THRESHOLD
  4. Trend filter: long only when 4h MA10 is up, short only when down
P&L = target_ret (6-bar forward return) minus FX_SPREAD_PIPS round-trip.

Usage
-----
python -m scripts.run_portfolio_backtest
python -m scripts.run_portfolio_backtest --mode fixed vol_only full
python -m scripts.run_portfolio_backtest --detail        # monthly breakdown

Output
------
Printed table + data/validation/portfolio_engine_backtest.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    FX_SPREAD_PIPS,
    RV2_DD_KILL,
    RV2_KILL_SWITCH_N,
    RV2_KILL_SWITCH_PF,
    RV2_PAUSE_BARS,
    RV2_PRED_THRESHOLD,
    RV2_TOP_PCT,
    RV2_TREND_MA_WINDOW,
    RV2_TREND_RESAMPLE,
    RV2_VOL_PCT,
)
from src.config_portfolio import (  # noqa: E402
    PORTFOLIO_BASE_SIZE,
    PORTFOLIO_DD_TIER1,
    PORTFOLIO_DD_TIER2,
    PORTFOLIO_LOSS_STREAK_MULT,
    PORTFOLIO_LOSS_STREAK_N,
    PORTFOLIO_MAX_SIZE,
    PORTFOLIO_MIN_SIZE,
    PORTFOLIO_TREND_MEDIUM_THRESH,
    PORTFOLIO_TREND_STRONG_THRESH,
    PORTFOLIO_VOL_CLIP_HIGH,
    PORTFOLIO_VOL_CLIP_LOW,
    PORTFOLIO_VOL_LOOKBACK_BARS,
    PORTFOLIO_VOL_TARGET,
)
from src.portfolio_engine import _realized_vol  # noqa: E402
from src.trend_filter import compute_trend_mask  # noqa: E402

PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions_regression" / "test_predictions.parquet"
PRICE_PATH       = PROJECT_ROOT / "data" / "processed" / "price" / "EURUSD_5min_clean.parquet"
OUTPUT_PATH      = PROJECT_ROOT / "data" / "validation" / "portfolio_engine_backtest.csv"
FX_SPREAD        = FX_SPREAD_PIPS * 0.0001


# ── Signal generation helpers ─────────────────────────────────────────────────

def _build_positions(pred_df: pd.DataFrame, trend_df: pd.DataFrame) -> pd.Series:
    """Apply RV2 signal rules, return int Series in {-1, 0, +1}."""
    n = len(pred_df)
    all_pred = pred_df["pred"].values.astype(float)
    all_vol  = pred_df["vol_6"].fillna(0).values.astype(float)

    th_long  = np.percentile(all_pred, 100 - RV2_TOP_PCT)
    th_short = np.percentile(all_pred, RV2_TOP_PCT)
    th_vol   = np.percentile(all_vol,  100 - RV2_VOL_PCT)

    pred = pred_df["pred"].values.astype(float)
    vol  = all_vol

    desired = np.zeros(n, dtype=int)
    desired[pred >= th_long]  =  1
    desired[pred <= th_short] = -1
    desired[vol  <  th_vol]   =  0
    if RV2_PRED_THRESHOLD > 0:
        desired[np.abs(pred) <= RV2_PRED_THRESHOLD] = 0

    # Trend filter gate
    trend_up   = trend_df["trend_up_1h"].values
    trend_down = trend_df["trend_down_1h"].values

    for i in range(n):
        if desired[i] == 0:
            continue
        tu = trend_up[i]
        td = trend_down[i]
        if np.isnan(tu):
            continue   # warmup: pass through (no filter yet)
        if desired[i] ==  1 and tu != 1.0:
            desired[i] = 0
        if desired[i] == -1 and td != 1.0:
            desired[i] = 0

    return pd.Series(desired, index=pred_df.index)


def _apply_kill_switch(
    positions: pd.Series,
    trade_rets: list[float],
    pause_remaining: list[int],
) -> tuple[pd.Series, list[float], list[int]]:
    """Apply per-strategy kill switch + DD kill to position series.

    Mutates positions in place. Returns updated positions.
    This replicates the exact logic from live_signal_regression_v2_trendfilter.
    """
    # Stateful simulation: iterate bar by bar
    pos = positions.values.copy()
    n = len(pos)
    current_pos = 0
    entry_equity = 1.0
    equity = 1.0
    peak   = 1.0
    t_rets: list[float] = []
    paused = 0

    for i in range(n):
        if paused > 0:
            paused -= 1
            pos[i] = 0
            continue

        # Check kill switch and DD kill before new entries
        if len(t_rets) >= RV2_KILL_SWITCH_N:
            window = t_rets[-RV2_KILL_SWITCH_N:]
            gains  = sum(r for r in window if r > 0)
            losses = abs(sum(r for r in window if r < 0))
            pf = gains / losses if losses > 0 else float("inf")
            if pf < RV2_KILL_SWITCH_PF:
                paused = RV2_PAUSE_BARS
                pos[i] = 0
                continue

        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd >= RV2_DD_KILL:
            paused = RV2_PAUSE_BARS
            pos[i] = 0
            continue

        # Track positions and record trade returns
        new_pos = pos[i]
        if current_pos != 0 and (new_pos == 0 or new_pos != current_pos):
            # Trade closes — approximate as 1-bar return (positions are 6-bar hold in backtest)
            pass
        current_pos = new_pos

    return pd.Series(pos, index=positions.index)


# ── Sizing helpers ────────────────────────────────────────────────────────────

def _trend_mult(trend_strength: float) -> float:
    ts = abs(trend_strength)
    if ts >= PORTFOLIO_TREND_STRONG_THRESH:
        return 1.0
    if ts >= PORTFOLIO_TREND_MEDIUM_THRESH:
        return 0.75
    return 0.5


def _dd_mult(dd: float) -> float:
    if dd >= PORTFOLIO_DD_TIER2:
        return 0.5
    if dd >= PORTFOLIO_DD_TIER1:
        return 0.75
    return 1.0


def _streak_mult(streak: int) -> float:
    return PORTFOLIO_LOSS_STREAK_MULT if streak >= PORTFOLIO_LOSS_STREAK_N else 1.0


# ── Backtest simulation ───────────────────────────────────────────────────────

def _simulate(
    pred_df: pd.DataFrame,
    positions: pd.Series,
    price_df: pd.DataFrame,
    trend_df: pd.DataFrame,
    mode: str,
) -> dict:
    """Simulate equity curve for one sizing mode.

    Returns dict with trade_rets, equity_curve, sizes, monthly stats.
    """
    target_rets = pred_df["target_ret"].values.astype(float)
    ts_arr      = pred_df["timestamp"].values

    # Pre-compute realized vol for each bar (lookback from price data)
    # Merge price close onto prediction timestamps for vol calculation
    price_df = price_df.set_index("timestamp")["close"]

    # Build a lookup from timestamp → realized_vol using a rolling window on price
    # For efficiency: compute rolling std on price series, then map to pred timestamps
    price_sorted = price_df.sort_index()
    price_log_ret = np.log(price_sorted.values[1:] / price_sorted.values[:-1])
    price_ts_idx  = price_sorted.index[1:]

    # Rolling std in a sliding window — pre-compute for speed
    log_ret_s = pd.Series(price_log_ret, index=price_ts_idx)
    roll_vol   = log_ret_s.rolling(PORTFOLIO_VOL_LOOKBACK_BARS, min_periods=PORTFOLIO_VOL_LOOKBACK_BARS).std()
    # Now merge_asof: for each prediction bar, get the last available realized vol
    roll_vol_df = roll_vol.reset_index()
    roll_vol_df.columns = ["timestamp", "realized_vol"]
    pred_ts_df = pd.DataFrame({"timestamp": pred_df["timestamp"]})
    merged_vol = pd.merge_asof(
        pred_ts_df.sort_values("timestamp"),
        roll_vol_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )["realized_vol"].values

    # Per-bar simulation
    n        = len(positions)
    pos      = positions.values
    trend_s  = trend_df["trend_strength_1h"].fillna(0.0).values
    realized_vols = merged_vol

    equity         = 1.0
    peak           = 1.0
    trade_rets: list[float] = []
    sizes: list[float]      = []
    loss_streak    = 0
    prev_pos       = 0

    equity_curve: list[float] = [1.0]

    for i in range(n):
        cur_pos = int(pos[i])
        if cur_pos == 0:
            equity_curve.append(equity)
            continue

        # Determine size for this bar
        if mode == "fixed":
            size = PORTFOLIO_BASE_SIZE

        elif mode == "vol_only":
            rv = realized_vols[i]
            if not np.isnan(rv) and rv > 0:
                vm = float(np.clip(PORTFOLIO_VOL_TARGET / rv, PORTFOLIO_VOL_CLIP_LOW, PORTFOLIO_VOL_CLIP_HIGH))
            else:
                vm = 1.0
            size = float(np.clip(PORTFOLIO_BASE_SIZE * vm, PORTFOLIO_MIN_SIZE, PORTFOLIO_MAX_SIZE))

        else:   # full
            rv = realized_vols[i]
            vm = 1.0
            if not np.isnan(rv) and rv > 0:
                vm = float(np.clip(PORTFOLIO_VOL_TARGET / rv, PORTFOLIO_VOL_CLIP_LOW, PORTFOLIO_VOL_CLIP_HIGH))
            tm  = _trend_mult(trend_s[i])
            dd  = (peak - equity) / peak if peak > 0 else 0.0
            dm  = _dd_mult(dd)
            sm  = _streak_mult(loss_streak)
            size = float(np.clip(PORTFOLIO_BASE_SIZE * vm * tm * dm * sm, PORTFOLIO_MIN_SIZE, PORTFOLIO_MAX_SIZE))

        sizes.append(size)

        # Per-unit net return (direction × price_return − round-trip cost)
        net_per_unit = float(target_rets[i]) * cur_pos - 2 * FX_SPREAD

        # Equity update: size scales how much capital is at risk each trade
        # size=1.0 → identical to baseline; size=0.5 → half gain/loss;
        # size=2.0 → double gain/loss.
        equity_change = net_per_unit * size
        equity *= (1.0 + equity_change)
        if equity > peak:
            peak = equity

        net = net_per_unit * size   # size-scaled net return for recording

        # Track loss streak for "full" mode
        if net_per_unit < 0:
            loss_streak += 1
        else:
            loss_streak = 0

        trade_rets.append(net)
        equity_curve.append(equity)
        prev_pos = cur_pos

    if not trade_rets:
        return {
            "mode": mode, "n_trades": 0, "cum_ret": 0.0,
            "pf": 0.0, "max_dd": 0.0, "win_rate": 0.0,
            "avg_size": 0.0, "size_std": 0.0, "avg_pip": 0.0,
            "sharpe": 0.0, "ann_vol_equity": 0.0, "turnover_units": 0.0,
            "monthly": pd.DataFrame(),
        }

    tr_arr  = np.array(trade_rets)
    gains   = tr_arr[tr_arr > 0].sum()
    losses  = abs(tr_arr[tr_arr < 0].sum())
    pf      = float(gains / losses) if losses > 0 else float("inf")
    eq_arr  = np.array(equity_curve)
    peaks   = np.maximum.accumulate(eq_arr)
    dd_arr  = (peaks - eq_arr) / np.where(peaks > 0, peaks, 1.0)
    max_dd  = float(dd_arr.max())
    cum_ret = float(eq_arr[-1] - 1.0)
    win_rate = float((tr_arr > 0).mean())
    avg_pip  = float(tr_arr.mean() * 10000)
    sizes_arr = np.array(sizes) if sizes else np.array([0.0])

    # Sharpe: annualised return / annualised std of per-step equity changes
    eq_diff = np.diff(eq_arr)
    sharpe  = float(eq_diff.mean() / eq_diff.std() * np.sqrt(252 * 24 * 12)) if eq_diff.std() > 0 else 0.0

    # Annualised equity volatility (fractional returns between trade steps)
    eq_ret = np.diff(eq_arr) / np.where(eq_arr[:-1] > 0, eq_arr[:-1], 1.0)
    ann_vol_equity = (
        float(np.std(eq_ret) * np.sqrt(252 * 24 * 12)) if len(eq_ret) > 1 and np.std(eq_ret) > 0 else 0.0
    )

    # Turnover proxy: sum of absolute lot-units deployed per trade (batch 7)
    turnover_units = float(np.sum(np.abs(sizes_arr)))

    # Monthly breakdown
    pred_ts = pred_df["timestamp"].values
    pos_mask = positions.values != 0
    if pos_mask.any():
        trade_ts   = pred_ts[pos_mask][:len(trade_rets)]
        trade_ts_s = pd.Series(
            pd.to_datetime(trade_ts, utc=True).tz_convert(None)
        ).dt.to_period("M").astype(str)
        month_df = pd.DataFrame({"month": trade_ts_s, "ret": tr_arr})
        monthly  = month_df.groupby("month").agg(
            net_ret=("ret", lambda x: float((1 + x).prod() - 1)),
            n_trades=("ret", "count"),
        ).reset_index()
    else:
        monthly = pd.DataFrame()

    return {
        "mode": mode,
        "n_trades": len(trade_rets),
        "cum_ret":  round(cum_ret,  6),
        "pf":       round(pf,       4),
        "max_dd":   round(max_dd,   6),
        "win_rate": round(win_rate, 4),
        "avg_size": round(float(sizes_arr.mean()), 4),
        "size_std": round(float(sizes_arr.std()),  4),
        "avg_pip":  round(avg_pip,  2),
        "sharpe":   round(sharpe,   4),
        "ann_vol_equity": round(ann_vol_equity, 6),
        "turnover_units": round(turnover_units, 4),
        "monthly":  monthly,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio engine backtest")
    parser.add_argument(
        "--mode", nargs="+",
        choices=["fixed", "vol_only", "full"],
        default=["fixed", "vol_only", "full"],
    )
    parser.add_argument("--detail", action="store_true", help="Print monthly breakdown")
    args = parser.parse_args()

    print(f"\n  Loading predictions ({PREDICTIONS_PATH.name})...")
    pred_df = pd.read_parquet(PREDICTIONS_PATH)
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"], utc=True)
    pred_df = pred_df.sort_values("timestamp").reset_index(drop=True)
    print(f"  Rows: {len(pred_df):,}  |  {pred_df['timestamp'].min()} → {pred_df['timestamp'].max()}")

    print("  Loading price data...")
    price_df = pd.read_parquet(PRICE_PATH, columns=["timestamp", "close"])
    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"], utc=True)
    price_df = price_df.sort_values("timestamp").drop_duplicates("timestamp")

    print(f"  Computing 4h MA{RV2_TREND_MA_WINDOW} trend filter...")
    trend_df = compute_trend_mask(
        pred_df["timestamp"],
        resample_period=RV2_TREND_RESAMPLE,
        ma_window=RV2_TREND_MA_WINDOW,
    )

    print("  Generating signals (RV2 rules)...")
    positions = _build_positions(pred_df, trend_df)
    n_signals = int((positions != 0).sum())
    n_long    = int((positions ==  1).sum())
    n_short   = int((positions == -1).sum())
    print(f"  Signals: {n_signals:,} total  (long={n_long:,}, short={n_short:,})")
    print(f"  Cost assumption: {FX_SPREAD_PIPS} pip one-way ({FX_SPREAD_PIPS*2} pip RT)\n")

    results = []
    for mode in args.mode:
        print(f"  Simulating mode={mode}...")
        r = _simulate(pred_df, positions, price_df.copy(), trend_df, mode)
        results.append(r)

    # ── Print comparison table ─────────────────────────────────────────────
    print(f"\n{'='*74}")
    print("  SIZING MODE COMPARISON")
    print(f"{'='*74}")
    print(f"  {'mode':<12} {'n':>7} {'PF':>7} {'cum_ret':>9} {'max_dd':>9} {'win%':>7}"
          f" {'avg_pip':>8} {'sharpe':>8} {'annVol':>8} {'turnover':>9} {'avg_sz':>8} {'sz_std':>8}")
    print("  " + "-" * 88)
    for r in results:
        pf_s = f"{r['pf']:.4f}" if not np.isinf(r["pf"]) else "   inf"
        print(
            f"  {r['mode']:<12} {r['n_trades']:>7,} {pf_s:>7}"
            f" {r['cum_ret']:>+9.4f} {r['max_dd']:>9.4f}"
            f" {100*r['win_rate']:>6.1f}% {r['avg_pip']:>7.2f}p"
            f" {r['sharpe']:>8.3f} {r['ann_vol_equity']:>8.4f} {r['turnover_units']:>9.2f}"
            f" {r['avg_size']:>8.3f} {r['size_std']:>8.3f}"
        )

    # ── Verdict ────────────────────────────────────────────────────────────
    print(f"\n{'='*74}")
    print("  VERDICT")
    print(f"{'='*74}")
    baseline = next((r for r in results if r["mode"] == "fixed"), None)
    for r in results:
        if r["mode"] == "fixed":
            continue
        if baseline is None:
            continue
        dd_ok = r["max_dd"] <= baseline["max_dd"] * 1.10   # at most 10% worse DD
        pf_ok = r["pf"]     >= baseline["pf"]     * 0.95   # at most 5% worse PF
        ret_ok = r["cum_ret"] >= baseline["cum_ret"] * 0.90
        promote = dd_ok and pf_ok and ret_ok
        status  = "PASS — promote" if promote else "MARGINAL — do not promote yet"
        print(
            f"  {r['mode']:>10} vs fixed: "
            f"DD_ok={dd_ok}  PF_ok={pf_ok}  ret_ok={ret_ok}  → {status}"
        )
        if promote:
            dd_delta = (r["max_dd"] - baseline["max_dd"]) * 100
            pf_delta = (r["pf"]    - baseline["pf"])
            print(f"           DD delta={dd_delta:+.2f}%  PF delta={pf_delta:+.4f}")

    # ── Monthly detail ─────────────────────────────────────────────────────
    if args.detail:
        for r in results:
            m = r.get("monthly")
            if m is None or m.empty:
                continue
            n_pos  = (m["net_ret"] > 0).sum()
            n_tot  = len(m)
            print(f"\n  Walk-forward [{r['mode']}]  positive months: {n_pos}/{n_tot}"
                  f" ({100*n_pos/max(n_tot,1):.0f}%)")
            print(f"  {'month':<10} {'n':>5} {'net_ret':>10}")
            print("  " + "-" * 32)
            for _, row in m.iterrows():
                print(f"  {row['month']:<10} {int(row['n_trades']):>5} {row['net_ret']:>+10.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    save_rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "monthly"}
        save_rows.append(row)

    out_df = pd.DataFrame(save_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Results saved → {OUTPUT_PATH}\n")


if __name__ == "__main__":
    main()
