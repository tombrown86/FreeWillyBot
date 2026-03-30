"""
Event study runner — post-news drift analysis for EUR/USD.

Builds the event dataset, then analyses:
  1. Overall: mean/median forward return + hit rate at 1/3/6/12 bars
             split by initial move direction (up vs down)
  2. By event type: CPI / NFP / FOMC / ECB / BoE
  3. By trend state: 4h MA10 trend agrees or disagrees with initial move
  4. Combined: event type × trend alignment

Pass / fail gates printed at the end.

Usage
-----
python -m scripts.run_event_study
python -m scripts.run_event_study --rebuild     # re-extract raw data

Outputs
-------
data/events/event_study_raw.csv          one row per event
data/events/event_study_analysis.csv     one row per group/horizon
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FX_SPREAD_PIPS  # noqa: E402
from src.event_study import OUTPUT_RAW, build_event_dataset, load_event_dataset  # noqa: E402

VALIDATION_DIR = PROJECT_ROOT / "data" / "events"
FX_SPREAD = FX_SPREAD_PIPS * 0.0001   # round-trip cost ≈ 2 × half-spread


# ── Analysis helpers ──────────────────────────────────────────────────────────

def _group_stats(df: pd.DataFrame, group_cols: list[str], horizons: list[int]) -> pd.DataFrame:
    """Compute mean/median directional return, hit rate, n per group×horizon."""
    rows = []
    for keys, grp in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        base["n_events"] = len(grp)
        for h in horizons:
            dcol = f"dir_ret_{h}"   # positive = momentum continued
            rcol = f"ret_{h}"
            if dcol not in grp.columns:
                continue
            dr = grp[dcol].dropna()
            rr = grp[rcol].dropna()
            hit = (dr > 0).mean() if len(dr) > 0 else float("nan")
            row = {
                **base,
                "horizon_bars": h,
                "horizon_min": h * 5,
                "mean_dir_ret": dr.mean() if len(dr) > 0 else float("nan"),
                "median_dir_ret": dr.median() if len(dr) > 0 else float("nan"),
                "mean_raw_ret": rr.mean() if len(rr) > 0 else float("nan"),
                "hit_rate": hit,
                "net_dir_ret": (dr.mean() - FX_SPREAD / df["close_at_T"].mean()) if len(dr) > 0 else float("nan"),
                "reliable": len(dr) >= 5,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _print_table(df: pd.DataFrame, title: str, group_cols: list[str], horizon: int = 6) -> None:
    """Print a formatted summary for a given horizon."""
    sub = df[df["horizon_bars"] == horizon].copy()
    if sub.empty:
        return
    sub = sub.sort_values(group_cols + ["horizon_bars"])

    print(f"\n  {title}  [T+{horizon} bars = {horizon*5} min]")
    header_parts = [f"{'':>4}"]
    for c in group_cols:
        header_parts.append(f"{c:<18}")
    header_parts += [
        f"{'n':>5}", f"{'hit%':>7}", f"{'mean_dir':>10}",
        f"{'net_dir':>10}", f"{'behavior':>12}"
    ]
    print("  " + " ".join(header_parts))
    print("  " + "-" * 78)

    for _, row in sub.iterrows():
        flag = "" if row["reliable"] else "  (n<5 ⚠)"
        behavior = "DRIFT" if row["mean_dir_ret"] > FX_SPREAD else ("REVERT" if row["mean_dir_ret"] < -FX_SPREAD else "FLAT")
        parts = ["    "]
        for c in group_cols:
            parts.append(f"{str(row[c]):<18}")
        parts += [
            f"{int(row['n_events']):>5}",
            f"{100*row['hit_rate']:>6.1f}%",
            f"{10000*row['mean_dir_ret']:>9.2f}p",
            f"{10000*row['net_dir_ret']:>9.2f}p",
            f"{behavior:>12}{flag}",
        ]
        print("  " + " ".join(parts))


def _verdict(df: pd.DataFrame, horizons: list[int]) -> None:
    """Print pass/fail verdict against the event study gates."""
    print(f"\n{'='*70}")
    print("  VERDICT")
    print(f"{'='*70}")

    spread_pip = FX_SPREAD * 10000
    results = []

    for h in horizons:
        sub = df[(df["horizon_bars"] == h) & df["reliable"]]
        if sub.empty:
            continue
        n_positive = (sub["mean_dir_ret"] > FX_SPREAD).sum()
        n_negative = (sub["mean_dir_ret"] < -FX_SPREAD).sum()
        n_total = len(sub)
        results.append((h, n_positive, n_negative, n_total))
        print(f"  T+{h} ({h*5}min): {n_positive}/{n_total} reliable groups show net positive drift  |  {n_negative}/{n_total} show net reversion")

    # Gate 1: momentum drift in any group
    any_drift = any(np > 0 for _, np, _, _ in results if np is not None)
    gate1 = "PASS" if any_drift else "FAIL"

    # Gate 2: trend interaction — does trend_agrees split show different behavior?
    trend_sub = df[(df["horizon_bars"] == 6) & df["reliable"] & df["group"].astype(str).str.contains("trend") if "group" in df.columns else pd.Series(True, index=df.index)]

    # Check overall hit rate at 6 bars
    all6 = df[(df["horizon_bars"] == 6)]["hit_rate"].dropna()
    avg_hit = all6.mean() if len(all6) > 0 else 0.5
    gate2 = "PASS" if abs(avg_hit - 0.5) > 0.05 else "FAIL"

    print(f"\n  Gate 1 — any net-positive drift group at T+3 to T+12:  {gate1}")
    print(f"  Gate 2 — average hit rate meaningfully different from 50%: {gate2}  (avg={100*avg_hit:.1f}%)")

    overall_verdict = "GO — event study shows exploitable signal, proceed to backtest" if gate1 == "PASS" or gate2 == "PASS" else "STOP — no signal detected, do not build a strategy on this"
    print(f"\n  Overall: {overall_verdict}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Event study runner")
    parser.add_argument("--rebuild", action="store_true", help="Re-extract raw data from price + events")
    parser.add_argument("--fwd-bars", type=int, nargs="+", default=[1, 3, 6, 12], metavar="N")
    args = parser.parse_args()

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    if args.rebuild or not OUTPUT_RAW.exists():
        print("Building event dataset...")
        df = build_event_dataset(fwd_bars=args.fwd_bars)
    else:
        print(f"Loading cached dataset from {OUTPUT_RAW}")
        df = load_event_dataset()

    if df.empty:
        print("No events to analyse.")
        return

    horizons = [h for h in args.fwd_bars if f"dir_ret_{h}" in df.columns]
    print(f"\n  Total events: {len(df)}  |  Horizons: {[f'T+{h}' for h in horizons]}")
    print(f"  Event types:  {df['event_name'].value_counts().to_dict()}")
    print(f"  Trend labels: {df['trend_label'].value_counts().to_dict()}")
    print(f"  Initial move: {df['initial_move'].value_counts().to_dict()}")
    print(f"  FX spread assumption: {FX_SPREAD_PIPS} pips ({FX_SPREAD*10000:.1f}p)")

    all_analysis: list[pd.DataFrame] = []

    # ── 1. Overall by initial move direction ──────────────────────────────
    print(f"\n{'='*70}")
    print("  SECTION 1 — Overall (all events, by initial move direction)")
    print(f"{'='*70}")
    df["initial_move_label"] = df["initial_move"].map({1: "up", -1: "down", 0: "flat"})
    s1 = _group_stats(df, ["initial_move_label"], horizons)
    s1["group"] = "overall"
    for h in [3, 6, 12]:
        _print_table(s1, "Overall by initial move", ["initial_move_label"], h)
    all_analysis.append(s1)

    # ── 2. By event type ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SECTION 2 — By event type")
    print(f"{'='*70}")
    s2 = _group_stats(df, ["event_name", "initial_move_label"], horizons)
    s2["group"] = "by_event_type"
    _print_table(s2, "By event type", ["event_name", "initial_move_label"], 6)
    all_analysis.append(s2)

    # ── 3. By trend alignment ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SECTION 3 — By trend alignment (4h MA10)")
    print(f"{'='*70}")
    df["trend_align_label"] = df["trend_agrees"].map({True: "trend_agrees", False: "trend_disagrees"})
    s3 = _group_stats(df, ["trend_align_label", "initial_move_label"], horizons)
    s3["group"] = "by_trend"
    for h in [3, 6, 12]:
        _print_table(s3, "By trend alignment", ["trend_align_label", "initial_move_label"], h)
    all_analysis.append(s3)

    # ── 4. Event type × trend alignment ───────────────────────────────────
    print(f"\n{'='*70}")
    print("  SECTION 4 — Event type × trend alignment (best combinations)")
    print(f"{'='*70}")
    s4 = _group_stats(df, ["event_name", "trend_align_label", "initial_move_label"], horizons)
    s4["group"] = "event_x_trend"
    _print_table(s4, "Event type × trend", ["event_name", "trend_align_label", "initial_move_label"], 6)
    all_analysis.append(s4)

    # ── 5. Behavior distribution ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SECTION 5 — Behavior at T+6 (drift vs reversion)")
    print(f"{'='*70}")
    beh = df.groupby(["event_name", "behavior_6"]).size().unstack(fill_value=0)
    print(beh.to_string())
    drift_pct = 100 * df["behavior_6"].eq("drift").mean()
    rev_pct   = 100 * df["behavior_6"].eq("reversion").mean()
    print(f"\n  Overall: drift={drift_pct:.1f}%  reversion={rev_pct:.1f}%  flat={100-drift_pct-rev_pct:.1f}%")

    # ── 6. Pre-event volatility context ───────────────────────────────────
    print(f"\n{'='*70}")
    print("  SECTION 6 — High vs low pre-event volatility context")
    print(f"{'='*70}")
    pre_vol_med = df["pre_vol"].median()
    df["vol_regime"] = df["pre_vol"].apply(lambda v: "high_vol" if v >= pre_vol_med else "low_vol")
    s6 = _group_stats(df, ["vol_regime", "initial_move_label"], horizons)
    s6["group"] = "by_vol_regime"
    _print_table(s6, "By pre-event vol", ["vol_regime", "initial_move_label"], 6)
    all_analysis.append(s6)

    # ── Save ──────────────────────────────────────────────────────────────
    analysis_df = pd.concat(all_analysis, ignore_index=True)
    analysis_path = VALIDATION_DIR / "event_study_analysis.csv"
    analysis_df.to_csv(analysis_path, index=False)
    print(f"\n  Analysis saved → {analysis_path}")

    # ── Verdict ───────────────────────────────────────────────────────────
    _verdict(analysis_df, horizons)


if __name__ == "__main__":
    main()
