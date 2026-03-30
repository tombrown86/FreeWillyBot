"""
Event drift backtest runner — full validation suite.

Modes
-----
baseline   : run all configurations at hold=6, print comparison table
sweep      : sweep hold_bars [1,3,6,12] across best configs
walk-forward: monthly breakdown for best config
cost-stress: 1x / 1.5x / 2x spread at best config
all        : run all modes

Usage
-----
python -m scripts.run_backtest_event_drift
python -m scripts.run_backtest_event_drift --mode all

Outputs saved to data/events/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest_event_drift import run_all, run_hold_sweep, run_single  # noqa: E402
from src.config import FX_SPREAD_PIPS  # noqa: E402
from src.event_study import OUTPUT_RAW, build_event_dataset, load_event_dataset  # noqa: E402

EVENTS_DIR   = PROJECT_ROOT / "data" / "events"
FX_SPREAD    = FX_SPREAD_PIPS * 0.0001


def _load_df() -> pd.DataFrame:
    if OUTPUT_RAW.exists():
        return load_event_dataset()
    print("Raw dataset not found — building...")
    return build_event_dataset()


def _print_results(results: list[dict], title: str) -> None:
    print(f"\n  {'config':<26} {'hold':>5} {'PF':>7} {'cum_ret':>9} {'max_dd':>9} {'n':>5} {'win%':>7} {'avg_pip':>9}")
    print("  " + "-" * 80)
    for r in results:
        pf_str = f"{r['profit_factor']:.4f}" if r['profit_factor'] != float("inf") else "  inf "
        print(
            f"  {r['config']:<26} {r['hold_bars']:>5} {pf_str:>7}"
            f" {r['cum_ret']:>+9.4f} {r['max_dd']:>9.4f}"
            f" {r['n_trades']:>5} {100*r['win_rate']:>6.1f}%"
            f" {r['avg_trade_pips']:>8.2f}p"
        )


def _verdict(results: list[dict]) -> None:
    print(f"\n{'='*70}")
    print("  VERDICT")
    print(f"{'='*70}")
    passing = [r for r in results if r["profit_factor"] > 1.1 and r["n_trades"] >= 5 and r["cum_ret"] > 0]
    if passing:
        best = max(passing, key=lambda r: r["profit_factor"])
        print(f"  {len(passing)} config(s) pass PF>1.1 + positive return + n≥5:")
        for r in passing:
            print(f"    {r['config']}: PF={r['profit_factor']:.4f}  cum={r['cum_ret']:+.4f}  n={r['n_trades']}")
        print(f"\n  Best: {best['config']}  (PF={best['profit_factor']:.4f})")
        print("  Verdict: PASS — proceed to walk-forward and cost stress")
    else:
        pf_best = max(results, key=lambda r: r["profit_factor"]) if results else {}
        print(f"  No config passes all gates. Best PF: {pf_best.get('profit_factor', 0):.4f} ({pf_best.get('config','')})")
        print("  Verdict: MARGINAL or FAIL — check walk-forward before promoting")


def mode_baseline(df: pd.DataFrame) -> list[dict]:
    print(f"\n{'='*70}")
    print("  BASELINE — all configurations, hold=6 bars (30 min)")
    print(f"{'='*70}")
    results = run_all(df, hold_bars=6, cost_per_leg=FX_SPREAD)
    _print_results(results, "Baseline")
    _verdict(results)
    return results


def mode_sweep(df: pd.DataFrame) -> list[dict]:
    print(f"\n{'='*70}")
    print("  HOLD SWEEP — [1,3,6,12] bars × best configs")
    print(f"{'='*70}")
    results = run_hold_sweep(df, hold_bars_list=[1, 3, 6, 12], cost_per_leg=FX_SPREAD)
    _print_results(results, "Hold sweep")
    return results


def mode_walk_forward(df: pd.DataFrame, config: str = "no_cpi_up", hold_bars: int = 6) -> None:
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD — {config}, hold={hold_bars} bars")
    print(f"{'='*70}")

    pre_vol_median = df["pre_vol"].median()
    filter_map = {
        "baseline":   None,
        "no_cpi_up":  lambda d: ~((d["event_name"] == "CPI") & (d["initial_move"] == 1)),
        "high_vol":   lambda d: d["pre_vol"] >= pre_vol_median,
        "high_vol_no_cpi_up": lambda d: (d["pre_vol"] >= pre_vol_median)
                                          & ~((d["event_name"] == "CPI") & (d["initial_move"] == 1)),
    }
    fn = filter_map.get(config)

    ret_col = f"ret_{hold_bars}"
    sub = df.copy()
    if fn is not None:
        sub = sub[fn(sub)].reset_index(drop=True)

    if sub.empty:
        print("  No events after filter.")
        return

    sub["trade_ret_net"] = sub["initial_move"] * sub[ret_col] - 2 * FX_SPREAD
    sub["month"] = pd.to_datetime(sub["event_time_utc"]).dt.tz_convert(None).dt.to_period("M").astype(str)

    monthly = sub.groupby("month").agg(
        net_ret=("trade_ret_net", lambda x: float((1 + x).prod() - 1)),
        n_trades=("trade_ret_net", "count"),
        win_rate=("trade_ret_net", lambda x: (x > 0).mean()),
        avg_pip=("trade_ret_net", lambda x: x.mean() * 10000),
    ).reset_index()

    n_pos = (monthly["net_ret"] > 0).sum()
    total = len(monthly)

    print(f"  {'month':<10} {'n':>5} {'net_ret':>10} {'win%':>8} {'avg_pip':>9}")
    print("  " + "-" * 48)
    for _, row in monthly.iterrows():
        print(
            f"  {row['month']:<10} {int(row['n_trades']):>5}"
            f" {row['net_ret']:>+10.4f} {100*row['win_rate']:>7.1f}%"
            f" {row['avg_pip']:>8.2f}p"
        )
    print("  " + "-" * 48)
    print(f"  Positive months: {n_pos}/{total} ({100*n_pos/max(total,1):.0f}%)")

    cum = float((1 + monthly["net_ret"]).prod() - 1)
    avg_n = monthly["n_trades"].mean()
    print(f"  Cum return: {cum:+.4f}  |  Avg trades/month: {avg_n:.1f}")

    verdict = "PASS" if n_pos >= total * 0.5 and cum > 0 else "MARGINAL" if cum > 0 else "FAIL"
    print(f"  Verdict: {verdict}")

    monthly.to_csv(EVENTS_DIR / f"event_drift_monthly_{config}.csv", index=False)
    print(f"  Saved → {EVENTS_DIR / f'event_drift_monthly_{config}.csv'}")


def mode_cost_stress(df: pd.DataFrame, config: str = "no_cpi_up", hold_bars: int = 6) -> None:
    print(f"\n{'='*70}")
    print(f"  COST STRESS — {config}, hold={hold_bars} bars")
    print(f"{'='*70}")

    pre_vol_median = df["pre_vol"].median()
    filter_map = {
        "baseline":   None,
        "no_cpi_up":  lambda d: ~((d["event_name"] == "CPI") & (d["initial_move"] == 1)),
        "high_vol":   lambda d: d["pre_vol"] >= pre_vol_median,
        "high_vol_no_cpi_up": lambda d: (d["pre_vol"] >= pre_vol_median)
                                          & ~((d["event_name"] == "CPI") & (d["initial_move"] == 1)),
    }
    fn = filter_map.get(config)

    results = []
    for mult in [1.0, 1.5, 2.0, 3.0]:
        cost = FX_SPREAD * mult
        res = run_single(df, hold_bars=hold_bars, cost_per_leg=cost, filter_fn=fn)
        res["cost_mult"] = mult
        res["cost_pips"] = round(FX_SPREAD_PIPS * mult * 2, 2)
        results.append(res)

    print(f"  {'cost_mult':>10} {'rt_pips':>9} {'PF':>7} {'cum_ret':>9} {'n':>5} {'win%':>7}")
    print("  " + "-" * 52)
    for r in results:
        print(
            f"  {r['cost_mult']:>10.1f} {r['cost_pips']:>8.2f}p"
            f" {r['profit_factor']:>7.4f} {r['cum_ret']:>+9.4f}"
            f" {r['n_trades']:>5} {100*r['win_rate']:>6.1f}%"
        )

    pf_1x = results[0]["profit_factor"]
    pf_2x = results[2]["profit_factor"]
    pf_3x = results[3]["profit_factor"]
    if pf_2x > 1.0 and results[2]["cum_ret"] > 0:
        verdict = "ROBUST — edge survives at 2× spread"
    elif pf_2x > pf_1x * 0.7:
        verdict = "MODERATE — edge degrades at 2× but PF stays above 1"
    else:
        verdict = "FRAGILE — edge disappears above 1.5× spread"
    print(f"\n  Verdict: {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Event drift backtest runner")
    parser.add_argument(
        "--mode",
        choices=["baseline", "sweep", "walk-forward", "cost-stress", "all"],
        default="all",
    )
    parser.add_argument("--rebuild", action="store_true", help="Re-extract raw event data")
    parser.add_argument("--hold", type=int, default=6, metavar="N")
    parser.add_argument("--config", default="no_cpi_up", help="Config for walk-forward / cost-stress")
    args = parser.parse_args()

    EVENTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.rebuild or not OUTPUT_RAW.exists():
        df = build_event_dataset()
    else:
        df = _load_df()

    print(f"\n  Events loaded: {len(df)}  |  cost assumption: {FX_SPREAD_PIPS} pips (1-way)")

    if args.mode in ("baseline", "all"):
        results = mode_baseline(df)
        pd.DataFrame(results).to_csv(EVENTS_DIR / "event_drift_baseline.csv", index=False)

    if args.mode in ("sweep", "all"):
        sweep_results = mode_sweep(df)
        pd.DataFrame(sweep_results).to_csv(EVENTS_DIR / "event_drift_sweep.csv", index=False)

    # Determine best config for walk-forward / cost-stress
    best_config = args.config
    if args.mode == "all":
        all_res = run_all(df, hold_bars=6, cost_per_leg=FX_SPREAD)
        eligible = [r for r in all_res if r["profit_factor"] > 1.1 and r["n_trades"] >= 5]
        if eligible:
            best_config = max(eligible, key=lambda r: r["profit_factor"])["config"]
        print(f"\n  Best config for further tests: {best_config}")

    if args.mode in ("walk-forward", "all"):
        mode_walk_forward(df, config=best_config, hold_bars=args.hold)

    if args.mode in ("cost-stress", "all"):
        mode_cost_stress(df, config=best_config, hold_bars=args.hold)

    print(f"\n  All outputs saved to {EVENTS_DIR}/\n")


if __name__ == "__main__":
    main()
