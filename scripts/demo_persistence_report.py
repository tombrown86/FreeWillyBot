"""
Demo persistence report — primary metrics for the 3–6 month clean demo phase.

Tracks real broker fills (mode=demo) for active regression strategies:
  - trade count (completed round-trips)
  - profit factor
  - max drawdown (equity curve from close events)
  - monthly returns

Deliberately omits win rate and “equity after N trades” snapshots.

Usage:
  python scripts/demo_persistence_report.py
  python scripts/demo_persistence_report.py --since 2026-06-10
  python scripts/demo_persistence_report.py --strategy regression_v1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXEC_DIR = PROJECT_ROOT / "data" / "logs" / "execution"
TRADES_CSV = EXEC_DIR / "trade_decisions.csv"

PERSISTENCE_TARGET_DAYS = (90, 180)  # 3 and 6 months


def _active_demo_strategies() -> list[str]:
    path = PROJECT_ROOT / "scripts" / "run_live_tick.py"
    spec = importlib.util.spec_from_file_location("run_live_tick", path)
    if spec is None or spec.loader is None:
        return ["regression_v1", "regression_v2_trendfilter"]
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    demo_ids = getattr(mod, "DEMO_BROKER_REAL_ORDER_STRATEGY_IDS", frozenset())
    return [s["id"] for s in getattr(mod, "STRATEGIES", []) if s["id"] in demo_ids]


def _load_demo_trades(strategy_id: str, since: pd.Timestamp | None) -> pd.DataFrame:
    if not TRADES_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(TRADES_CSV, low_memory=False)
    mask = df["strategy_id"] == strategy_id
    if "mode" in df.columns:
        mask &= df["mode"] == "demo"
    df = df[mask].copy()
    if df.empty:
        return df
    ts_col = "bar_close_utc" if "bar_close_utc" in df.columns else "timestamp"
    df["bar_close_utc"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=["bar_close_utc"]).sort_values("bar_close_utc")
    if since is not None:
        df = df[df["bar_close_utc"] >= since]
    return df.reset_index(drop=True)


def _pair_trades(trades_df: pd.DataFrame) -> list[dict]:
    pairs: list[dict] = []
    current: dict | None = None
    for _, row in trades_df.iterrows():
        action = str(row.get("action_taken", "")).upper()
        ts = row["bar_close_utc"]
        equity = float(row.get("sim_equity", 1.0) or 1.0)
        side = "long" if "LONG" in action else "short" if "SHORT" in action else None

        if "OPEN" in action and current is None:
            current = {"open_ts": ts, "side": side, "equity_open": equity}
        elif ("CLOSE" in action or "REVERSE" in action) and current is not None:
            current["close_ts"] = ts
            current["equity_close"] = equity
            eo = float(current["equity_open"])
            ec = float(current["equity_close"])
            if current["side"] == "short":
                trade_ret = (eo / ec - 1.0) if ec > 0 else 0.0
            else:
                trade_ret = (ec / eo - 1.0) if eo > 0 else 0.0
            current["trade_return"] = trade_ret
            pairs.append(current)
            if "REVERSE" in action:
                current = {"open_ts": ts, "side": side, "equity_open": equity}
            else:
                current = None
    return pairs


def _metrics(pairs: list[dict]) -> dict:
    if not pairs:
        return {
            "trade_count": 0,
            "profit_factor": None,
            "max_drawdown": None,
            "cumulative_return": None,
            "first_trade": None,
            "last_trade": None,
            "days_elapsed": 0,
        }

    rets = np.array([p["trade_return"] for p in pairs], dtype=float)
    gains = rets[rets > 0].sum()
    losses = np.abs(rets[rets < 0].sum())
    if losses > 0:
        pf = float(gains / losses)
    elif gains > 0:
        pf = float("inf")
    else:
        pf = 0.0

    equity = np.array([1.0] + [p["equity_close"] for p in pairs], dtype=float)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, 1.0)
    max_dd = float(np.max(dd))

    first = min(p["open_ts"] for p in pairs)
    last = max(p["close_ts"] for p in pairs)
    days = max(0, (last - first).days)

    return {
        "trade_count": len(pairs),
        "profit_factor": pf,
        "max_drawdown": max_dd,
        "cumulative_return": float(equity[-1] - 1.0),
        "first_trade": first,
        "last_trade": last,
        "days_elapsed": days,
    }


def _monthly_returns(pairs: list[dict]) -> pd.DataFrame:
    if not pairs:
        return pd.DataFrame(columns=["month", "trades", "return"])
    rows = []
    df = pd.DataFrame(pairs)
    df["month"] = df["close_ts"].dt.to_period("M")
    for month, grp in df.groupby("month"):
        rets = grp["trade_return"].values
        month_ret = float(np.prod(1.0 + rets) - 1.0) if len(rets) else 0.0
        rows.append({"month": str(month), "trades": len(grp), "return": month_ret})
    return pd.DataFrame(rows)


def _fmt_pf(pf: float | None) -> str:
    if pf is None:
        return "—"
    if pf == float("inf"):
        return "inf"
    return f"{pf:.3f}"


def report(strategy_id: str, since: pd.Timestamp | None) -> dict:
    trades = _load_demo_trades(strategy_id, since)
    pairs = _pair_trades(trades)
    m = _metrics(pairs)
    monthly = _monthly_returns(pairs)

    print(f"\n{'=' * 60}")
    print(f"  {strategy_id}  (mode=demo)")
    if since is not None:
        print(f"  since {since.date().isoformat()}")
    print(f"{'=' * 60}")
    print(f"  Trade count:        {m['trade_count']}")
    print(f"  Profit factor:      {_fmt_pf(m['profit_factor'])}")
    if m["max_drawdown"] is not None:
        print(f"  Max drawdown:       {m['max_drawdown']:.2%}")
    if m["cumulative_return"] is not None:
        print(f"  Cumulative return:  {m['cumulative_return']:+.2%}")
    if m["first_trade"] is not None:
        print(f"  First trade:        {m['first_trade'].strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"  Last trade:         {m['last_trade'].strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"  Days elapsed:       {m['days_elapsed']}")
        for target in PERSISTENCE_TARGET_DAYS:
            remaining = target - m["days_elapsed"]
            label = f"{target // 30}mo" if target == 90 else "6mo"
            if remaining > 0:
                print(f"  → {label} target:       {remaining} days remaining")
            else:
                print(f"  → {label} target:       reached")

    if not monthly.empty:
        print("\n  Monthly returns:")
        for _, row in monthly.iterrows():
            print(f"    {row['month']}  {row['trades']:3d} trades  {row['return']:+.2%}")
    elif m["trade_count"] == 0:
        print("\n  (no completed demo trades yet)")

    return {**m, "monthly": monthly.to_dict(orient="records")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo persistence metrics report")
    parser.add_argument("--since", default=None, help="ISO date YYYY-MM-DD (UTC)")
    parser.add_argument(
        "--strategy",
        action="append",
        dest="strategies",
        help="Strategy id (repeatable; default: all active demo strategies)",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON summary to stdout")
    args = parser.parse_args()

    since = pd.Timestamp(args.since, tz="UTC") if args.since else None
    strategies = args.strategies or _active_demo_strategies()
    if not strategies:
        print("No active demo strategies configured.", file=sys.stderr)
        return 1

    results = {}
    for sid in strategies:
        results[sid] = report(sid, since)

    print(f"\n{'=' * 60}")
    print("  Persistence phase: run ONLY regression_v1 + regression_v2_trendfilter")
    print("  Target: 3–6 months uninterrupted demo; track PF, max DD, monthly returns")
    print(f"{'=' * 60}\n")

    if args.json:
        def _serialize(obj):
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            if isinstance(obj, float) and (obj == float("inf") or obj != obj):
                return str(obj)
            raise TypeError(type(obj))

        print(json.dumps(results, indent=2, default=_serialize))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
