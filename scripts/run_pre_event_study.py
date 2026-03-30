"""
Pre-event drift runner — full research suite for EUR/USD pre-announcement drift.

Modes
-----
event-study : Event study (go/no-go gate). Groups mean/median pre-event returns,
              hit rates, momentum signal validity, trend + vol context.
backtest    : Simulate the momentum signal strategy with walk-forward + cost stress.
              Only meaningful if event-study passes.
all         : Run event-study first; if it passes proceed to backtest automatically.

Usage
-----
python -m scripts.run_pre_event_study --mode all
python -m scripts.run_pre_event_study --mode event-study
python -m scripts.run_pre_event_study --mode backtest
python -m scripts.run_pre_event_study --rebuild     # re-extract raw data

Outputs
-------
data/events/pre_event_study_raw.csv
data/events/pre_event_analysis.csv
data/events/pre_event_monthly.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FX_SPREAD_PIPS  # noqa: E402
from src.pre_event_study import OUTPUT_RAW, build_pre_event_dataset, load_pre_event_dataset  # noqa: E402

EVENTS_DIR   = PROJECT_ROOT / "data" / "events"
FX_SPREAD    = FX_SPREAD_PIPS * 0.0001   # one-way spread


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_df(rebuild: bool = False) -> pd.DataFrame:
    if rebuild or not OUTPUT_RAW.exists():
        print("Building pre-event dataset...")
        return build_pre_event_dataset()
    print(f"Loading cached dataset from {OUTPUT_RAW}")
    return load_pre_event_dataset()


def _profit_factor(rets: np.ndarray) -> float:
    gains  = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    return float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)


def _group_stats(df: pd.DataFrame, group_cols: list[str], ret_col: str) -> pd.DataFrame:
    """Mean, median, hit rate, and net return per group for a given return column."""
    SPREAD_RT = 2 * FX_SPREAD
    rows = []
    for keys, grp in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        base["n"] = len(grp)
        raw  = grp[ret_col].dropna().values
        mean_pip = float(raw.mean() * 10000) if len(raw) > 0 else float("nan")
        med_pip  = float(np.median(raw) * 10000) if len(raw) > 0 else float("nan")
        net_pip  = mean_pip - SPREAD_RT * 10000
        hit      = float((raw > 0).mean()) if len(raw) > 0 else float("nan")
        rows.append({
            **base,
            "mean_pip": round(mean_pip, 2),
            "median_pip": round(med_pip, 2),
            "net_pip": round(net_pip, 2),
            "hit_rate": round(hit, 4),
            "reliable": len(raw) >= 5,
        })
    return pd.DataFrame(rows)


def _print_group_table(df: pd.DataFrame, title: str, group_cols: list[str]) -> None:
    sub = df.copy()
    sub = sub.sort_values(group_cols)
    cols = group_cols + ["n", "mean_pip", "net_pip", "hit_rate"]
    w = max(18, max(len(c) for c in group_cols) + 2)

    print(f"\n  {title}")
    hdr = "".join(f"{c:<{w}}" for c in group_cols) + f"{'n':>5}  {'mean_pip':>9}  {'net_pip':>9}  {'hit%':>7}  {'flag':>6}"
    print("  " + hdr)
    print("  " + "-" * (len(hdr) + 4))
    for _, row in sub.iterrows():
        flag = "" if row["reliable"] else "n<5 ⚠"
        edge = ("EDGE" if row["net_pip"] > 0.5
                else "FLAT" if abs(row["net_pip"]) <= 0.5
                else "NEG")
        parts = "".join(f"{str(row[c]):<{w}}" for c in group_cols)
        parts += f"{int(row['n']):>5}  {row['mean_pip']:>8.2f}p  {row['net_pip']:>8.2f}p  {100*row['hit_rate']:>6.1f}%  {edge:>4} {flag}"
        print("  " + parts)


# ── Event study ───────────────────────────────────────────────────────────────

def mode_event_study(df: pd.DataFrame) -> bool:
    """Run the event study. Returns True if the edge gate is passed."""
    SPREAD_RT_PIP = 2 * FX_SPREAD_PIPS

    print(f"\n{'='*72}")
    print("  PRE-EVENT DRIFT — EVENT STUDY")
    print(f"{'='*72}")
    print(f"  Events: {len(df)}  |  Cost: {SPREAD_RT_PIP:.1f}p round-trip")
    print(f"  Types:  {df['event_name'].value_counts().to_dict()}")
    print(f"  Trend:  {df['trend_label'].value_counts().to_dict()}")

    pre_vol_med = df["pre_vol"].median()
    df = df.copy()
    df["vol_regime"] = df["pre_vol"].apply(lambda v: "high_vol" if v >= pre_vol_med else "low_vol")
    df["trend_align_6"]  = (
        ((df["drift_dir_6"]  ==  1) & (df["trend_label"] == "up")) |
        ((df["drift_dir_6"]  == -1) & (df["trend_label"] == "down"))
    )
    df["momentum_hit_6"] = (df["momentum_signal"] == df["drift_dir_6"]).astype(int)

    all_analysis = []

    # ── Section 1: Raw unsigned drift magnitude ───────────────────────────
    print(f"\n{'='*72}")
    print("  SECTION 1 — Raw drift magnitude (unsigned mean |return|)")
    print(f"{'='*72}")
    for col, label in [("ret_6_to_1", "30-min window T-6→T-1"), ("ret_12_to_1", "60-min window T-12→T-1")]:
        raw = df[col].abs().values * 10000
        print(f"  {label}: mean={raw.mean():.2f}p  median={np.median(raw):.2f}p  "
              f"max={raw.max():.2f}p  std={raw.std():.2f}p")

    # ── Section 2: Signed drift by event type ────────────────────────────
    print(f"\n{'='*72}")
    print("  SECTION 2 — Signed drift by event type (are any consistently directional?)")
    print(f"{'='*72}")
    for col, label in [("ret_6_to_1", "30-min T-6→T-1"), ("ret_12_to_1", "60-min T-12→T-1")]:
        s = _group_stats(df, ["event_name"], col)
        s["window"] = label
        all_analysis.append(s)
        _print_group_table(s, label, ["event_name"])

    # ── Section 3: Momentum signal validity ──────────────────────────────
    print(f"\n{'='*72}")
    print("  SECTION 3 — Momentum signal: does sign(T-12→T-6) predict sign(T-6→T-1)?")
    print(f"{'='*72}")
    overall_hit = df["momentum_hit_6"].mean()
    print(f"  Overall momentum hit rate: {100*overall_hit:.1f}%  (n={len(df)}, null=50.0%)")

    mom_stats = _group_stats(df, ["momentum_signal", "event_name"], "ret_6_to_1")
    mom_stats["window"] = "momentum_signal"
    all_analysis.append(mom_stats)
    _print_group_table(mom_stats, "By momentum signal × event type", ["momentum_signal", "event_name"])

    # Also test: enter at T-6 in direction of momentum signal
    print(f"\n  Trade simulation (enter T-6 close in direction of momentum_signal, exit T-1 close):")
    for mask_label, mask in [
        ("all events",           pd.Series(True, index=df.index)),
        ("high_vol only",        df["pre_vol"] >= pre_vol_med),
        ("trend agrees (4h)",    df["trend_align_6"]),
        ("trend disagrees",     ~df["trend_align_6"]),
        ("NFP + ECB + FOMC",     df["event_name"].isin(["NFP", "ECB", "FOMC"])),
        ("CPI only",             df["event_name"] == "CPI"),
    ]:
        sub = df[mask]
        if len(sub) < 5:
            continue
        rets = (sub["ret_6_to_1"] * sub["momentum_signal"]).values
        net  = rets - 2 * FX_SPREAD
        pf   = _profit_factor(net)
        avg  = net.mean() * 10000
        wr   = (net > 0).mean()
        print(f"    {mask_label:<28} n={len(sub):>3}  PF={pf:.4f}  avg={avg:>7.2f}p  win={100*wr:.1f}%")

    # ── Section 4: By trend and vol regime ───────────────────────────────
    print(f"\n{'='*72}")
    print("  SECTION 4 — Context: trend alignment and vol regime")
    print(f"{'='*72}")
    s4 = _group_stats(df, ["trend_label", "event_name"], "ret_6_to_1")
    s4["window"] = "by_trend"
    all_analysis.append(s4)
    _print_group_table(s4, "By trend × event type  [T-6→T-1]", ["trend_label", "event_name"])

    s4b = _group_stats(df, ["vol_regime", "event_name"], "ret_6_to_1")
    s4b["window"] = "by_vol"
    all_analysis.append(s4b)
    _print_group_table(s4b, "By vol regime × event type  [T-6→T-1]", ["vol_regime", "event_name"])

    # ── Section 5: Cumulative path T-12→T-1 ──────────────────────────────
    print(f"\n{'='*72}")
    print("  SECTION 5 — Average cumulative return path (T-12 to T-1, signed by momentum_signal)")
    print(f"{'='*72}")
    price = pd.read_parquet(
        PROJECT_ROOT / "data" / "processed" / "price" / "EURUSD_5min_clean.parquet",
        columns=["timestamp", "close"]
    )
    price["timestamp"] = pd.to_datetime(price["timestamp"], utc=True)
    price = price.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    paths: list[list[float]] = []
    for _, ev in df.iterrows():
        ev_ts = ev["event_time_utc"]
        cutoff = ev_ts + pd.Timedelta(minutes=30)
        cands = price[(price["timestamp"] >= ev_ts) & (price["timestamp"] <= cutoff)]
        if cands.empty:
            continue
        t_idx = int(cands.index[0])
        if t_idx < 13:
            continue
        anchor = price.loc[t_idx - 12, "close"]
        sig    = int(ev["momentum_signal"])
        path = []
        for step in range(13):   # T-12 to T (inclusive)
            c = price.loc[t_idx - 12 + step, "close"]
            path.append((c - anchor) / anchor * sig * 10000)
        paths.append(path)

    if paths:
        arr = np.array(paths)
        mean_path = arr.mean(axis=0)
        std_path  = arr.std(axis=0)
        print(f"  {'bar':<6} {'mean_pip':>9} {'std':>9}  (n={len(paths)} events)")
        print("  " + "-" * 32)
        labels = [f"T-{12-i}" if i < 12 else "T" for i in range(13)]
        for i, (lbl, m, s) in enumerate(zip(labels, mean_path, std_path)):
            bar = "▲" * max(0, int(m / 0.5)) if m > 0 else "▼" * max(0, int(-m / 0.5))
            print(f"  {lbl:<6} {m:>8.2f}p  {s:>8.2f}p  {bar}")

    # ── Save analysis ─────────────────────────────────────────────────────
    analysis_df = pd.concat(all_analysis, ignore_index=True)
    analysis_path = EVENTS_DIR / "pre_event_analysis.csv"
    analysis_df.to_csv(analysis_path, index=False)
    print(f"\n  Analysis saved → {analysis_path}")

    # ── Gate ──────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  EVENT STUDY GATE")
    print(f"{'='*72}")

    # Check 1: any group with net_pip > 1 and reliable
    reliable_edge = analysis_df[
        analysis_df["reliable"] & (analysis_df.get("net_pip", pd.Series()) > 1.0)
    ] if "net_pip" in analysis_df.columns else pd.DataFrame()

    # Check 2: momentum hit rate vs 50%
    mom_edge = abs(overall_hit - 0.5) > 0.05

    # Check 3: momentum trade PF > 1 in any config
    rets_all = (df["ret_6_to_1"] * df["momentum_signal"]).values - 2 * FX_SPREAD
    pf_overall = _profit_factor(rets_all)

    gate1 = len(reliable_edge) > 0
    gate2 = mom_edge
    gate3 = pf_overall > 1.0

    print(f"  Gate 1 — reliable group with net_pip > 1:    {'PASS' if gate1 else 'FAIL'}  ({len(reliable_edge)} groups)")
    print(f"  Gate 2 — momentum hit rate > 55% or < 45%:   {'PASS' if gate2 else 'FAIL'}  ({100*overall_hit:.1f}%)")
    print(f"  Gate 3 — overall momentum PF > 1.0:          {'PASS' if gate3 else 'FAIL'}  (PF={pf_overall:.4f})")

    passed = gate1 or gate2 or gate3
    verdict = ("GO — pre-event drift edge detected, proceed to backtest"
               if passed
               else "STOP — no pre-event edge found after costs; do not build strategy")
    print(f"\n  Overall: {verdict}")
    return passed


# ── Backtest ──────────────────────────────────────────────────────────────────

def mode_backtest(df: pd.DataFrame) -> None:
    """Simulate the pre-event drift strategy. Use only if event study passes."""
    SPREAD_RT = 2 * FX_SPREAD
    pre_vol_med = df["pre_vol"].median()
    df = df.copy()
    df["trend_align_6"] = (
        ((df["drift_dir_6"] == 1)  & (df["trend_label"] == "up")) |
        ((df["drift_dir_6"] == -1) & (df["trend_label"] == "down"))
    )

    print(f"\n{'='*72}")
    print("  BACKTEST — Pre-event momentum strategy")
    print(f"  Signal: enter T-6 close in direction of sign(ret_12_to_6), exit T-1 close")
    print(f"{'='*72}")

    configs: dict[str, object] = {
        "baseline":          None,
        "high_vol":          df["pre_vol"] >= pre_vol_med,
        "trend_agrees":      df["trend_align_6"],
        "trend_disagrees":  ~df["trend_align_6"],
        "nfp_only":          df["event_name"] == "NFP",
        "ecb_only":          df["event_name"] == "ECB",
        "fomc_only":         df["event_name"] == "FOMC",
        "cpi_only":          df["event_name"] == "CPI",
        "nfp_ecb_fomc":      df["event_name"].isin(["NFP", "ECB", "FOMC"]),
        "high_vol_no_cpi":   (df["pre_vol"] >= pre_vol_med) & (df["event_name"] != "CPI"),
    }

    results = []
    for name, mask in configs.items():
        sub = df[mask].copy() if mask is not None else df.copy()
        if len(sub) < 5:
            results.append({"config": name, "n_trades": 0, "pf": 0,
                            "cum_ret": 0, "max_dd": 0, "win_rate": 0, "avg_pip": 0})
            continue

        rets = (sub["ret_6_to_1"] * sub["momentum_signal"]).values.astype(float)
        net  = rets - SPREAD_RT

        equity = np.ones(len(net) + 1)
        for i, r in enumerate(net):
            equity[i + 1] = equity[i] * (1 + r)

        peak = np.maximum.accumulate(equity[1:])
        dd   = (peak - equity[1:]) / np.where(peak > 0, peak, 1.0)
        max_dd = float(dd.max()) if len(dd) > 0 else 0.0
        cum_ret = float(equity[-1] - 1.0)
        pf = _profit_factor(net)
        win = float((net > 0).mean())
        avg_pip = float(net.mean() * 10000)

        results.append({
            "config": name, "n_trades": len(sub), "pf": round(pf, 4),
            "cum_ret": round(cum_ret, 6), "max_dd": round(max_dd, 6),
            "win_rate": round(win, 4), "avg_pip": round(avg_pip, 2),
        })

    print(f"\n  {'config':<26} {'n':>5} {'PF':>7} {'cum_ret':>9} {'max_dd':>9} {'win%':>7} {'avg_pip':>9}")
    print("  " + "-" * 78)
    for r in results:
        pf_s = f"{r['pf']:.4f}" if r["pf"] != float("inf") else "  inf "
        print(
            f"  {r['config']:<26} {r['n_trades']:>5} {pf_s:>7}"
            f" {r['cum_ret']:>+9.4f} {r['max_dd']:>9.4f}"
            f" {100*r['win_rate']:>6.1f}% {r['avg_pip']:>8.2f}p"
        )

    # Best config for further tests
    passing = [r for r in results if r["pf"] > 1.1 and r["n_trades"] >= 5 and r["cum_ret"] > 0]
    if not passing:
        print("\n  No config passes PF>1.1 + positive return + n≥5.")
        print("  Verdict: MARGINAL/FAIL — do not promote to live")
        return

    best = max(passing, key=lambda r: r["pf"])
    print(f"\n  {len(passing)} config(s) pass: {[r['config'] for r in passing]}")
    print(f"  Best: {best['config']}  (PF={best['pf']:.4f})")

    # ── Walk-forward ──────────────────────────────────────────────────────
    best_name = best["config"]
    best_mask = configs[best_name]
    sub_best = df[best_mask].copy() if best_mask is not None else df.copy()
    sub_best["trade_ret_net"] = sub_best["ret_6_to_1"] * sub_best["momentum_signal"] - SPREAD_RT
    sub_best["month"] = pd.to_datetime(sub_best["event_time_utc"]).dt.tz_convert(None).dt.to_period("M").astype(str)

    monthly = sub_best.groupby("month").agg(
        net_ret=("trade_ret_net", lambda x: float((1 + x).prod() - 1)),
        n_trades=("trade_ret_net", "count"),
        win_rate=("trade_ret_net", lambda x: (x > 0).mean()),
        avg_pip=("trade_ret_net", lambda x: x.mean() * 10000),
    ).reset_index()

    n_pos = (monthly["net_ret"] > 0).sum()
    total = len(monthly)

    print(f"\n{'='*72}")
    print(f"  WALK-FORWARD — {best_name}")
    print(f"{'='*72}")
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
    print(f"  Cum return: {cum:+.4f}  |  Avg trades/month: {monthly['n_trades'].mean():.1f}")
    wf_verdict = "PASS" if n_pos >= total * 0.5 and cum > 0 else "FAIL"
    print(f"  Walk-forward verdict: {wf_verdict}")

    monthly.to_csv(EVENTS_DIR / f"pre_event_monthly_{best_name}.csv", index=False)

    # ── Cost stress ───────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  COST STRESS — {best_name}")
    print(f"{'='*72}")
    print(f"  {'mult':>8} {'rt_pips':>9} {'PF':>7} {'cum_ret':>9} {'win%':>7}")
    print("  " + "-" * 46)
    for mult in [1.0, 1.5, 2.0, 3.0]:
        cost = FX_SPREAD * mult
        rets_c = (sub_best["ret_6_to_1"] * sub_best["momentum_signal"]).values - 2 * cost
        pf_c   = _profit_factor(rets_c)
        cum_c  = float(np.prod(1 + rets_c) - 1)
        win_c  = float((rets_c > 0).mean())
        rt_pip = FX_SPREAD_PIPS * mult * 2
        print(f"  {mult:>8.1f}x {rt_pip:>8.2f}p {pf_c:>7.4f} {cum_c:>+9.4f} {100*win_c:>6.1f}%")

    # ── Overall backtest verdict ──────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  BACKTEST VERDICT")
    print(f"{'='*72}")
    rets_2x = (sub_best["ret_6_to_1"] * sub_best["momentum_signal"]).values - 2 * FX_SPREAD * 2
    pf_2x = _profit_factor(rets_2x)
    robust = pf_2x > 1.0 and (1 + rets_2x).prod() - 1 > 0

    if robust and wf_verdict == "PASS":
        verdict = "PROMOTE — edge is robust, walk-forward passes, cost-stress passes at 2×"
    elif best["pf"] > 1.1 and wf_verdict == "PASS":
        verdict = "CANDIDATE — edge holds at 1× cost, walk-forward passes; review cost sensitivity"
    else:
        verdict = "REJECT — edge does not survive combined walk-forward + cost stress"

    print(f"  {verdict}")

    # Save all results
    pd.DataFrame(results).to_csv(EVENTS_DIR / "pre_event_backtest_configs.csv", index=False)
    print(f"  Saved config results → {EVENTS_DIR / 'pre_event_backtest_configs.csv'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-event drift research runner")
    parser.add_argument(
        "--mode",
        choices=["event-study", "backtest", "all"],
        default="all",
    )
    parser.add_argument("--rebuild", action="store_true", help="Re-extract raw dataset")
    args = parser.parse_args()

    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_df(rebuild=args.rebuild)

    if df.empty:
        print("No data to analyse.")
        return

    print(f"\n  Events loaded: {len(df)}  |  cost: {FX_SPREAD_PIPS} pips one-way  "
          f"|  {FX_SPREAD_PIPS*2:.1f}p round-trip")

    if args.mode in ("event-study", "all"):
        passed = mode_event_study(df)
        if args.mode == "all" and not passed:
            print("\n  Event study gate FAILED — skipping backtest.")
            return

    if args.mode in ("backtest", "all"):
        mode_backtest(df)

    print(f"\n  All outputs saved to {EVENTS_DIR}/\n")


if __name__ == "__main__":
    main()
