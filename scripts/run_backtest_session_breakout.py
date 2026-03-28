"""
Session Breakout backtest CLI runner — v1 (rolling) and v2 (ORB).

Modes (--mode):
    baseline      Single run. Prints metrics.
    grid          Parameter grid search.
    walk-forward  12×1m rolling windows.
    stability     Perturb ±1 step around chosen config.
    cost-stress   Repeat config at 1×, 1.5×, 2× spread.
    event-study   [v2 only] Forward-return distribution at ORB signal events
                  vs v1 rolling-breakout events. Go/no-go gate before grid.

Strategy (--strategy):
    v1  Rolling N-bar high/low breakout (default, original)
    v2  Opening Range Breakout — frozen range from session-open window

Usage examples:
    # ── v1 (unchanged) ────────────────────────────────────────────────────────
    python scripts/run_backtest_session_breakout.py --mode baseline
    python scripts/run_backtest_session_breakout.py --mode grid
    python scripts/run_backtest_session_breakout.py --mode walk-forward --n 8 --min-range 0.0002 --hold 12
    python scripts/run_backtest_session_breakout.py --mode stability --n 8 --min-range 0.0002 --hold 12
    python scripts/run_backtest_session_breakout.py --mode cost-stress --n 8 --min-range 0.0002 --hold 12

    # ── v2 ORB ────────────────────────────────────────────────────────────────
    # Gate 0 — event study (go/no-go before any grid)
    python scripts/run_backtest_session_breakout.py --strategy v2 --mode event-study --start-date 2024-01-01

    # Phase 1 — baseline (London 30m OR, hold 12)
    python scripts/run_backtest_session_breakout.py --strategy v2 --mode baseline --or-minutes 30 --hold 12

    # Phase 2 — small grid (18 combos)
    python scripts/run_backtest_session_breakout.py --strategy v2 --mode grid

    # Phase 3 — walk-forward on top configs (auto from grid_results_v2.csv)
    python scripts/run_backtest_session_breakout.py --strategy v2 --mode walk-forward

    # Phase 3 — walk-forward on a specific config
    python scripts/run_backtest_session_breakout.py --strategy v2 --mode walk-forward --or-minutes 30 --hold 12

    # Phase 4 — stability
    python scripts/run_backtest_session_breakout.py --strategy v2 --mode stability --or-minutes 30 --hold 12

    # Phase 5 — cost stress
    python scripts/run_backtest_session_breakout.py --strategy v2 --mode cost-stress --or-minutes 30 --hold 12
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BACKTESTS_DIR = PROJECT_ROOT / "data" / "backtests_session_breakout"


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _load_price(start_date: str | None) -> pd.DataFrame:
    from src.backtest_session_breakout import load_price
    return load_price(start_date=start_date)


def _print_metrics_v1(r: dict) -> None:
    print(
        f"  N={r['n']}  min_range={r['min_range']:.4f}  hold={r['hold_bars']}"
        f"  cost_mult={r.get('cost_mult', 1.0):.1f}"
        f"\n  net_return={r['net_return']:.4f}"
        f"  PF={r['profit_factor']:.3f}"
        f"  sharpe={r['sharpe']:.2f}"
        f"  max_dd={r['max_dd']:.4f}"
        f"  trades={r['n_trades']}"
    )


def _print_metrics_v2(r: dict) -> None:
    print(
        f"  OR={r['or_minutes']}m  buffer={r['entry_buffer']:.4f}  hold={r['hold_bars']}"
        f"  cost_mult={r.get('cost_mult', 1.0):.1f}"
        f"\n  net_return={r['net_return']:.4f}"
        f"  PF={r['profit_factor']:.3f}"
        f"  sharpe={r['sharpe']:.2f}"
        f"  max_dd={r['max_dd']:.4f}"
        f"  trades={r['n_trades']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# v1 modes (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _mode_baseline_v1(args: argparse.Namespace) -> int:
    from src.backtest_session_breakout import run_single
    df = _load_price(args.start_date)
    logging.info("Baseline v1: %d bars (%.1f years)", len(df), len(df) / (252 * 288))
    r = run_single(df, n=args.n, min_range=args.min_range, hold_bars=args.hold)
    print("\n── Baseline result (v1) ─────────────────────────────")
    _print_metrics_v1(r)
    print()
    return 0


def _mode_grid_v1(args: argparse.Namespace) -> int:
    from src.backtest_session_breakout import run_grid
    df = _load_price(args.start_date)
    logging.info("Grid search v1: %d bars", len(df))
    grid_df = run_grid(df, cost_mult=args.cost_mult)

    print("\n── Grid results v1 (top 10 by PF) ──────────────────")
    cols = ["n", "min_range", "hold_bars", "net_return", "profit_factor", "sharpe", "max_dd", "n_trades"]
    print(grid_df[cols].head(10).to_string(index=False))

    shortlist = grid_df[(grid_df["profit_factor"] >= 1.05) & (grid_df["n_trades"] >= 50)]
    print(f"\n── Shortlist (PF≥1.05, trades≥50): {len(shortlist)} config(s) ──")
    if not shortlist.empty:
        print(shortlist[cols].head(5).to_string(index=False))
    print()
    return 0


def _mode_walk_forward_v1(args: argparse.Namespace) -> int:
    from src.backtest_session_breakout import run_walk_forward
    df = _load_price(args.start_date)

    configs_to_run: list[tuple[int, float, int, str]] = []
    if args.n is not None and args.min_range is not None and args.hold is not None:
        configs_to_run.append((args.n, args.min_range, args.hold, "manual"))
    else:
        grid_path = BACKTESTS_DIR / "grid_results.csv"
        if not grid_path.exists():
            logging.error("grid_results.csv not found. Run --mode grid first.")
            return 1
        grid_df = pd.read_csv(grid_path)
        shortlist = grid_df[(grid_df["profit_factor"] >= 1.05) & (grid_df["n_trades"] >= 50)].head(3)
        if shortlist.empty:
            logging.warning("No shortlisted configs (PF≥1.05). Using top 3 by PF.")
            shortlist = grid_df.head(3)
        for _, row in shortlist.iterrows():
            n = int(row["n"])
            mr = float(row["min_range"])
            hb = int(row["hold_bars"])
            configs_to_run.append((n, mr, hb, f"n{n}_mr{str(mr).replace('.', '')}_h{hb}"))

    for n, mr, hb, label in configs_to_run:
        logging.info("Walk-forward v1: N=%d min_range=%.4f hold=%d", n, mr, hb)
        results = run_walk_forward(df, n=n, min_range=mr, hold_bars=hb, label=label)
        n_pos = sum(1 for r in results if r["net_return"] > 0)
        print(
            f"\n── Walk-forward v1 N={n} mr={mr:.4f} hold={hb} ──"
            f"\n   {n_pos}/{len(results)} positive months"
            f"  avg_ret={sum(r['net_return'] for r in results) / max(len(results), 1):.4f}"
        )
    print()
    return 0


def _mode_stability_v1(args: argparse.Namespace) -> int:
    from src.backtest_session_breakout import run_stability
    if args.n is None or args.min_range is None or args.hold is None:
        logging.error("--n, --min-range, and --hold are required for stability mode.")
        return 1
    df = _load_price(args.start_date)
    stab_df = run_stability(df, center_n=args.n, center_min_range=args.min_range, center_hold=args.hold)

    center_pf = stab_df[stab_df["is_center"] == True]["profit_factor"].values
    center_pf_val = float(center_pf[0]) if len(center_pf) > 0 else 0.0
    min_pf = float(stab_df["profit_factor"].min())
    drop_pct = (1 - min_pf / center_pf_val) * 100 if center_pf_val > 0 else 0.0

    print("\n── Stability results v1 ─────────────────────────────")
    cols = ["n", "min_range", "hold_bars", "net_return", "profit_factor", "max_dd", "n_trades", "is_center"]
    print(stab_df[cols].to_string(index=False))
    print(f"\nCenter PF: {center_pf_val:.3f}  Min neighbour PF: {min_pf:.3f}  Max drop: {drop_pct:.1f}%")
    print("⚠ PF drops >20% on perturbation — config may be fragile" if drop_pct > 20
          else "✓ PF stable across perturbations")
    print()
    return 0


def _mode_cost_stress_v1(args: argparse.Namespace) -> int:
    from src.backtest_session_breakout import run_cost_stress
    if args.n is None or args.min_range is None or args.hold is None:
        logging.error("--n, --min-range, and --hold are required for cost-stress mode.")
        return 1
    df = _load_price(args.start_date)
    stress_df = run_cost_stress(df, n=args.n, min_range=args.min_range, hold_bars=args.hold)

    print("\n── Cost stress results v1 ───────────────────────────")
    print(stress_df[["cost_mult", "net_return", "profit_factor", "max_dd", "n_trades"]].to_string(index=False))
    pf_2x = stress_df[stress_df["cost_mult"] == 2.0]["profit_factor"].values
    if len(pf_2x) > 0:
        print("✓ PF ≥ 1.0 at 2× cost" if float(pf_2x[0]) >= 1.0
              else "⚠ PF < 1.0 at 2× cost — edge is thin")
    print()
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# v2 modes (ORB)
# ══════════════════════════════════════════════════════════════════════════════

def _mode_event_study(args: argparse.Namespace) -> int:
    """
    Gate 0: measure raw forward-return distribution at ORB events vs v1 events.
    Run this before any grid to check whether the hypothesis is worth pursuing.
    """
    from src.backtest_session_breakout import run_event_study
    df = _load_price(args.start_date)
    logging.info("Event study: %d bars", len(df))

    event_df = run_event_study(
        df,
        or_minutes_list=[15, 30, 60],
        entry_buffer=args.entry_buffer,
        fwd_bars=[3, 6, 12],
        entry_cutoff_bars=args.entry_cutoff,
    )

    if event_df.empty:
        logging.error("Event study returned no results.")
        return 1

    print("\n── Event study: mean signed forward return after signal ─────────────")
    print("(positive = trade direction wins; negative = trade direction loses)\n")

    pivot = event_df.pivot_table(
        index=["strategy", "or_minutes", "session", "direction"],
        columns="horizon_bars",
        values="mean_fwd_ret_net",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    for col in [3, 6, 12]:
        if col in pivot.columns:
            pivot[col] = pivot[col].map(lambda x: f"{x:.6f}" if pd.notna(x) else "n/a")
    print(pivot.to_string(index=False))

    print("\n── Hit rates (% of events where trade direction is correct) ─────────")
    pivot_hr = event_df.pivot_table(
        index=["strategy", "or_minutes", "session", "direction"],
        columns="horizon_bars",
        values="hit_rate",
        aggfunc="first",
    ).reset_index()
    pivot_hr.columns.name = None
    for col in [3, 6, 12]:
        if col in pivot_hr.columns:
            pivot_hr[col] = pivot_hr[col].map(lambda x: f"{x:.1%}" if pd.notna(x) else "n/a")
    print(pivot_hr.to_string(index=False))

    # Simple verdict
    orb_6bar = event_df[
        (event_df["strategy"] == "v2_orb")
        & (event_df["horizon_bars"] == 6)
    ]["mean_fwd_ret_net"]
    v1_6bar = event_df[
        (event_df["strategy"] == "v1_rolling")
        & (event_df["horizon_bars"] == 6)
    ]["mean_fwd_ret_net"]

    orb_pos = (orb_6bar > 0).sum()
    v1_pos = (v1_6bar > 0).sum()
    orb_total = len(orb_6bar)
    v1_total = len(v1_6bar)

    print(f"\n── Verdict (6-bar horizon, after cost) ──────────────────────────────")
    print(f"  ORB groups with positive net fwd return: {orb_pos}/{orb_total}")
    print(f"  v1  groups with positive net fwd return: {v1_pos}/{v1_total}")
    if orb_pos > v1_pos:
        print("  Directional improvement over v1 — proceed to baseline")
    elif orb_pos == v1_pos:
        print("  No clear improvement over v1 — review by session/direction before proceeding")
    else:
        print("  ORB worse than v1 at event level — hypothesis not supported; stop here")
    print()
    return 0


def _mode_baseline_v2(args: argparse.Namespace) -> int:
    from src.backtest_session_breakout import run_single_v2
    df = _load_price(args.start_date)
    logging.info("Baseline v2 ORB: %d bars", len(df))
    r = run_single_v2(
        df, or_minutes=args.or_minutes, entry_buffer=args.entry_buffer,
        hold_bars=args.hold, entry_cutoff_bars=args.entry_cutoff,
    )
    print("\n── Baseline result (v2 ORB) ─────────────────────────")
    _print_metrics_v2(r)
    print()
    return 0


def _mode_grid_v2(args: argparse.Namespace) -> int:
    from src.backtest_session_breakout import run_grid_v2
    df = _load_price(args.start_date)
    logging.info("Grid search v2 ORB: %d bars", len(df))
    grid_df = run_grid_v2(df, entry_cutoff_bars=args.entry_cutoff, cost_mult=args.cost_mult)

    print("\n── Grid results v2 ORB (top 10 by PF) ──────────────")
    cols = ["or_minutes", "entry_buffer", "hold_bars", "net_return", "profit_factor", "sharpe", "max_dd", "n_trades"]
    print(grid_df[cols].head(10).to_string(index=False))

    shortlist = grid_df[(grid_df["profit_factor"] >= 1.0) & (grid_df["n_trades"] >= 20)]
    print(f"\n── Shortlist (PF≥1.0, trades≥20): {len(shortlist)} config(s) ──")
    if not shortlist.empty:
        print(shortlist[cols].head(5).to_string(index=False))
    print()
    return 0


def _mode_walk_forward_v2(args: argparse.Namespace) -> int:
    from src.backtest_session_breakout import run_walk_forward_v2
    df = _load_price(args.start_date)

    configs_to_run: list[tuple[int, float, int, str]] = []

    # Use explicit args if all three are supplied
    or_supplied = args.or_minutes != 30  # user explicitly set it (not default)
    hold_supplied = args.hold != 12

    if or_supplied or hold_supplied:
        label = f"or{args.or_minutes}_buf{str(args.entry_buffer).replace('.','')}_h{args.hold}"
        configs_to_run.append((args.or_minutes, args.entry_buffer, args.hold, label))
    else:
        grid_path = BACKTESTS_DIR / "grid_results_v2.csv"
        if not grid_path.exists():
            logging.error("grid_results_v2.csv not found. Run --strategy v2 --mode grid first.")
            return 1
        grid_df = pd.read_csv(grid_path)
        shortlist = grid_df[(grid_df["profit_factor"] >= 1.0) & (grid_df["n_trades"] >= 20)].head(3)
        if shortlist.empty:
            logging.warning("No shortlisted v2 configs (PF≥1.0). Using top 3 by PF.")
            shortlist = grid_df.head(3)
        for _, row in shortlist.iterrows():
            orm = int(row["or_minutes"])
            eb = float(row["entry_buffer"])
            hb = int(row["hold_bars"])
            label = f"or{orm}_buf{str(eb).replace('.','')}_h{hb}"
            configs_to_run.append((orm, eb, hb, label))

    for orm, eb, hb, label in configs_to_run:
        logging.info("Walk-forward v2: OR=%dm buffer=%.4f hold=%d", orm, eb, hb)
        results = run_walk_forward_v2(
            df, or_minutes=orm, entry_buffer=eb, hold_bars=hb,
            entry_cutoff_bars=args.entry_cutoff, label=label,
        )
        n_pos = sum(1 for r in results if r["net_return"] > 0)
        print(
            f"\n── Walk-forward v2 OR={orm}m buf={eb:.4f} hold={hb} ──"
            f"\n   {n_pos}/{len(results)} positive months"
            f"  avg_ret={sum(r['net_return'] for r in results) / max(len(results), 1):.4f}"
        )
    print()
    return 0


def _mode_stability_v2(args: argparse.Namespace) -> int:
    from src.backtest_session_breakout import run_stability_v2
    df = _load_price(args.start_date)
    logging.info("Stability v2: OR=%dm buffer=%.4f hold=%d", args.or_minutes, args.entry_buffer, args.hold)
    stab_df = run_stability_v2(
        df, center_or_minutes=args.or_minutes, center_entry_buffer=args.entry_buffer,
        center_hold=args.hold, entry_cutoff_bars=args.entry_cutoff,
    )

    center_pf = stab_df[stab_df["is_center"] == True]["profit_factor"].values
    center_pf_val = float(center_pf[0]) if len(center_pf) > 0 else 0.0
    min_pf = float(stab_df["profit_factor"].min())
    drop_pct = (1 - min_pf / center_pf_val) * 100 if center_pf_val > 0 else 0.0

    print("\n── Stability results v2 ORB ─────────────────────────")
    cols = ["or_minutes", "entry_buffer", "hold_bars", "net_return", "profit_factor", "max_dd", "n_trades", "is_center"]
    print(stab_df[cols].to_string(index=False))
    print(f"\nCenter PF: {center_pf_val:.3f}  Min neighbour PF: {min_pf:.3f}  Max drop: {drop_pct:.1f}%")
    print("⚠ PF drops >20% on perturbation — config may be fragile" if drop_pct > 20
          else "✓ PF stable across perturbations")
    print()
    return 0


def _mode_cost_stress_v2(args: argparse.Namespace) -> int:
    from src.backtest_session_breakout import run_cost_stress_v2
    df = _load_price(args.start_date)
    logging.info("Cost stress v2: OR=%dm buffer=%.4f hold=%d", args.or_minutes, args.entry_buffer, args.hold)
    stress_df = run_cost_stress_v2(
        df, or_minutes=args.or_minutes, entry_buffer=args.entry_buffer,
        hold_bars=args.hold, entry_cutoff_bars=args.entry_cutoff,
    )

    print("\n── Cost stress results v2 ORB ───────────────────────")
    print(stress_df[["cost_mult", "net_return", "profit_factor", "max_dd", "n_trades"]].to_string(index=False))
    pf_2x = stress_df[stress_df["cost_mult"] == 2.0]["profit_factor"].values
    if len(pf_2x) > 0:
        print("✓ PF ≥ 1.0 at 2× cost" if float(pf_2x[0]) >= 1.0
              else "⚠ PF < 1.0 at 2× cost — edge is thin")
    print()
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Session Breakout backtest runner (v1 rolling / v2 ORB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--strategy", choices=["v1", "v2"], default="v1",
        help="v1 = rolling breakout (default); v2 = Opening Range Breakout",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "grid", "walk-forward", "stability", "cost-stress", "event-study"],
        default="baseline",
        help="Which phase to run (default: baseline)",
    )
    # v1 params
    parser.add_argument("--n", type=int, default=None, help="[v1] Lookback bars")
    parser.add_argument("--min-range", type=float, default=None, dest="min_range",
                        help="[v1] Min range size in price units")
    # v2 params
    parser.add_argument("--or-minutes", type=int, default=30, dest="or_minutes",
                        help="[v2] Opening range window length in minutes (default: 30)")
    parser.add_argument("--entry-buffer", type=float, default=0.0, dest="entry_buffer",
                        help="[v2] Buffer beyond OR boundary before entry (default: 0)")
    parser.add_argument("--entry-cutoff", type=int, default=18, dest="entry_cutoff",
                        help="[v2] Max bars after OR closes to accept entry (default: 18)")
    # shared
    parser.add_argument("--hold", type=int, default=12,
                        help="Hold bars before forced close (default: 12)")
    parser.add_argument("--cost-mult", type=float, default=1.0, dest="cost_mult",
                        help="Cost multiplier for grid/baseline (default: 1.0)")
    parser.add_argument("--start-date", type=str, default=None, dest="start_date",
                        help="Filter price data to this start date (YYYY-MM-DD)")

    args = parser.parse_args()

    # v1 fallback defaults
    if args.n is None:
        args.n = 12
    if args.min_range is None:
        args.min_range = 0.0003

    if args.strategy == "v2":
        if args.mode == "event-study":
            return _mode_event_study(args)
        dispatch_v2 = {
            "baseline": _mode_baseline_v2,
            "grid": _mode_grid_v2,
            "walk-forward": _mode_walk_forward_v2,
            "stability": _mode_stability_v2,
            "cost-stress": _mode_cost_stress_v2,
        }
        if args.mode not in dispatch_v2:
            logging.error("Mode '%s' not supported for --strategy v2", args.mode)
            return 1
        return dispatch_v2[args.mode](args)

    # v1 (default)
    if args.mode == "event-study":
        logging.error("--mode event-study requires --strategy v2")
        return 1
    dispatch_v1 = {
        "baseline": _mode_baseline_v1,
        "grid": _mode_grid_v1,
        "walk-forward": _mode_walk_forward_v1,
        "stability": _mode_stability_v1,
        "cost-stress": _mode_cost_stress_v1,
    }
    return dispatch_v1[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
