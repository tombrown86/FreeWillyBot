"""
Daily health report for regression_v2_trendfilter_portfolio_vol.

Reads live signal and execution logs, then prints a concise per-day breakdown
of the seven metrics worth tracking while the behaviour-stabilisation patch
(momentum filter + max-hold + fixed sizing) is being evaluated:

  1  Entries opened
  2  Entries blocked by short_term_momentum
  3  Closes triggered by max_hold
  4  Average hold (bars)
  5  Average adverse excursion per trade
  6  Worst intraday equity dip (per day)
  7  Lot size actually used

Usage:
  python scripts/v2_daily_report.py [--days N] [--strategy STRATEGY_ID]

Defaults:
  --days 7
  --strategy regression_v2_trendfilter_portfolio_vol
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXEC_DIR     = PROJECT_ROOT / "data" / "logs" / "execution"
PREDS_CSV    = PROJECT_ROOT / "data" / "predictions" / "predictions_live.csv"
TRADES_CSV   = EXEC_DIR / "trade_decisions.csv"
PAPER_CSV    = EXEC_DIR / "paper_simulation.csv"


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_predictions(strategy_id: str, days: int) -> pd.DataFrame:
    """Load predictions_live, dedup (keep latest livetick run per bar)."""
    if not PREDS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(PREDS_CSV, low_memory=False)
    df = df[df["strategy_id"] == strategy_id].copy()
    if df.empty:
        return df
    df["timestamp"]  = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["run_at"]     = pd.to_datetime(df["run_at"],    utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    # Keep latest livetick run per bar (multiple runs process the same bar)
    df = df.sort_values("run_at").drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    return df[df["timestamp"] >= cutoff].copy()


def _load_trades(strategy_id: str, days: int) -> pd.DataFrame:
    """Load trade_decisions (demo mode only) for the strategy."""
    frames = []
    for path in [TRADES_CSV, PAPER_CSV]:
        if not path.exists():
            continue
        t = pd.read_csv(path, low_memory=False)
        mask = (t["strategy_id"] == strategy_id)
        if "mode" in t.columns:
            mask &= (t["mode"] == "demo")
        frames.append(t[mask].copy())
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["bar_close_utc"] = pd.to_datetime(df.get("bar_close_utc", df.get("timestamp")),
                                          utc=True, errors="coerce")
    df = df.dropna(subset=["bar_close_utc"])
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    df = df[df["bar_close_utc"] >= cutoff].copy()
    df = df.sort_values("bar_close_utc").reset_index(drop=True)
    return df


def _extract_volume_units(broker_response_str: str) -> float | None:
    """Parse volume_units from broker_response JSON blob."""
    try:
        d = json.loads(broker_response_str)
        if "volume_units" in d:
            return float(d["volume_units"]) / 100_000  # convert to lots
    except Exception:
        pass
    return None


def _pair_trades(trades_df: pd.DataFrame) -> list[dict]:
    """
    Match OPEN / CLOSE rows from trade_decisions into trade pairs.
    Returns list of dicts with: open_ts, close_ts, side, equity_open,
    equity_close, lot_size.
    """
    pairs = []
    current: dict | None = None
    for _, row in trades_df.iterrows():
        action = str(row.get("action_taken", ""))
        ts     = row["bar_close_utc"]
        equity = float(row.get("sim_equity", 1.0) or 1.0)
        side   = "long" if "LONG" in action else "short" if "SHORT" in action else None
        br_str = str(row.get("broker_response", "") or "")
        lots   = _extract_volume_units(br_str)

        if "OPEN" in action and current is None:
            current = {
                "open_ts":    ts,
                "side":       side,
                "equity_open": equity,
                "lot_size":   lots,
            }
        elif ("CLOSE" in action or "REVERSE" in action) and current is not None:
            current["close_ts"]    = ts
            current["equity_close"] = equity
            if lots and not current["lot_size"]:
                current["lot_size"] = lots
            pairs.append(current)
            if "REVERSE" in action:
                current = {
                    "open_ts":    ts,
                    "side":       side,
                    "equity_open": equity,
                    "lot_size":   lots,
                }
            else:
                current = None
    return pairs


def _hold_bars(open_ts: pd.Timestamp, close_ts: pd.Timestamp, bar_min: int = 5) -> float:
    """Number of bars held (5-min bars by default)."""
    delta = (close_ts - open_ts).total_seconds() / 60.0
    return max(1.0, delta / bar_min)


def _adverse_excursion(
    open_ts: pd.Timestamp,
    close_ts: pd.Timestamp,
    side: str,
    preds: pd.DataFrame,
) -> float:
    """
    Maximum adverse excursion during the trade, approximated from per-bar
    bar_return values in predictions_live.  Returns the worst cumulative
    negative return (positive number = how much went against the trade,
    e.g. 0.0023 = 23 pips adverse).
    """
    if "bar_return" not in preds.columns:
        return float("nan")
    mask = (preds["timestamp"] > open_ts) & (preds["timestamp"] <= close_ts)
    window = preds[mask].copy()
    if window.empty:
        return 0.0
    rets = window["bar_return"].fillna(0).values.astype(float)
    signed = rets if side == "long" else -rets
    cum = np.cumsum(signed)
    mae = float(-np.min(np.minimum(0, cum)))
    return mae


def _day_equity_dip(day_preds: pd.DataFrame) -> float:
    """
    Worst intraday equity dip from the bar_return column, assuming we start
    at 1.0 at midnight (conservative: sum of all losing bars in the day).
    Uses the running minimum over cumulative bar returns.
    """
    if "bar_return" not in day_preds.columns or day_preds.empty:
        return 0.0
    rets = day_preds["bar_return"].fillna(0).values.astype(float)
    cum  = np.cumsum(rets)
    running_max = np.maximum.accumulate(cum)
    drawdown    = running_max - cum
    return float(np.max(drawdown))


# ── report ────────────────────────────────────────────────────────────────────

def report(strategy_id: str, days: int) -> None:
    preds  = _load_predictions(strategy_id, days)
    trades = _load_trades(strategy_id, days)

    if preds.empty and trades.empty:
        print(f"No data found for {strategy_id} in the last {days} days.")
        return

    pairs = _pair_trades(trades) if not trades.empty else []

    # Assign each prediction to a calendar date (UTC)
    if not preds.empty:
        preds["date"] = preds["timestamp"].dt.date

    # Assign each trade pair to the date of entry
    for p in pairs:
        p["date"] = p["open_ts"].date()

    # All dates to report
    dates_pred   = set(preds["date"].unique()) if not preds.empty else set()
    dates_trades = {p["date"] for p in pairs}
    all_dates    = sorted(dates_pred | dates_trades)

    # ── header ────────────────────────────────────────────────────────────────
    col_w = 12
    cols  = ["date", "opens", "mom_blk", "mxh_cls", "avg_hold", "avg_mae", "day_dip", "lots"]
    labels = {
        "date":     "Date",
        "opens":    "Entries",
        "mom_blk":  "Mom_blkd",
        "mxh_cls":  "MaxHold",
        "avg_hold": "AvgHold",
        "avg_mae":  "AvgMAE%",
        "day_dip":  "DayDip%",
        "lots":     "Lots",
    }
    hdr = "  ".join(f"{labels[c]:<{col_w}}" for c in cols)
    print(f"\n{'─' * len(hdr)}")
    print(f"  {strategy_id}")
    print(f"  Last {days} days  ·  {len(pairs)} completed trades found in logs")
    print(f"{'─' * len(hdr)}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for day in all_dates:
        day_preds  = preds[preds["date"] == day] if not preds.empty else pd.DataFrame()
        day_pairs  = [p for p in pairs if p["date"] == day]

        # 1. Entries opened
        opens = len(day_pairs)

        # 2. Blocked by short_term_momentum
        if not day_preds.empty and "reason" in day_preds.columns:
            mom_blk = int((day_preds["reason"].str.strip() == "short_term_momentum").sum())
        else:
            mom_blk = 0

        # 3. Closes by max_hold (from predictions_live reason on CLOSE bars)
        if not day_preds.empty and "reason" in day_preds.columns and "action" in day_preds.columns:
            mxh_cls = int(
                ((day_preds["reason"].str.strip() == "max_hold") &
                 (day_preds["action"].str.contains("CLOSE", na=False))).sum()
            )
        else:
            mxh_cls = 0

        # 4. Average hold bars
        if day_pairs:
            holds = [
                _hold_bars(p["open_ts"], p["close_ts"])
                for p in day_pairs if "close_ts" in p
            ]
            avg_hold = f"{np.mean(holds):.1f}" if holds else "—"
        else:
            avg_hold = "—"

        # 5. Average adverse excursion
        if day_pairs and not preds.empty:
            maes = [
                _adverse_excursion(p["open_ts"], p["close_ts"], p["side"] or "long", preds)
                for p in day_pairs if "close_ts" in p
            ]
            avg_mae = f"{np.mean(maes) * 100:.3f}" if maes else "—"
        else:
            avg_mae = "—"

        # 6. Worst intraday equity dip from bar returns (% of equity)
        if not day_preds.empty:
            day_dip = f"{_day_equity_dip(day_preds) * 100:.3f}"
        else:
            day_dip = "—"

        # 7. Lot sizes used
        lots_used = [p["lot_size"] for p in day_pairs if p.get("lot_size") is not None]
        lots_str  = f"{np.mean(lots_used):.2f}" if lots_used else "fixed"

        row = {
            "date":     str(day),
            "opens":    str(opens),
            "mom_blk":  str(mom_blk),
            "mxh_cls":  str(mxh_cls),
            "avg_hold": avg_hold,
            "avg_mae":  avg_mae,
            "day_dip":  day_dip,
            "lots":     lots_str,
        }
        print("  " + "  ".join(f"{row[c]:<{col_w}}" for c in cols))

    # ── totals row ────────────────────────────────────────────────────────────
    if all_dates:
        total_opens = len(pairs)
        total_mom   = int((preds["reason"].str.strip() == "short_term_momentum").sum()) \
                      if not preds.empty and "reason" in preds.columns else 0
        total_mxh   = int(
            ((preds["reason"].str.strip() == "max_hold") &
             (preds["action"].str.contains("CLOSE", na=False))).sum()
        ) if not preds.empty and "reason" in preds.columns and "action" in preds.columns else 0
        all_holds   = [
            _hold_bars(p["open_ts"], p["close_ts"])
            for p in pairs if "close_ts" in p
        ]
        all_maes    = [
            _adverse_excursion(p["open_ts"], p["close_ts"], p["side"] or "long", preds)
            for p in pairs if "close_ts" in p
        ] if not preds.empty else []
        lots_used   = [p["lot_size"] for p in pairs if p.get("lot_size") is not None]

        print("  " + "─" * (len(hdr) - 2))
        tot = {
            "date":     "TOTAL",
            "opens":    str(total_opens),
            "mom_blk":  str(total_mom),
            "mxh_cls":  str(total_mxh),
            "avg_hold": f"{np.mean(all_holds):.1f}" if all_holds else "—",
            "avg_mae":  f"{np.mean(all_maes) * 100:.3f}" if all_maes else "—",
            "day_dip":  "—",
            "lots":     f"{np.mean(lots_used):.2f}" if lots_used else "fixed",
        }
        print("  " + "  ".join(f"{tot[c]:<{col_w}}" for c in cols))

    print(f"{'─' * len(hdr)}")

    # ── warning signals ───────────────────────────────────────────────────────
    warnings = []
    if pairs:
        all_holds = [_hold_bars(p["open_ts"], p["close_ts"]) for p in pairs if "close_ts" in p]
        if all_holds:
            pct_at_max = sum(1 for h in all_holds if h >= 11) / len(all_holds)
            if pct_at_max >= 0.50:
                warnings.append(
                    f"  ⚠  {pct_at_max:.0%} of trades closed at/near max hold (12 bars) — "
                    "max_hold may be capping winners OR exits are too slow; review."
                )
    total_mom = int((preds["reason"].str.strip() == "short_term_momentum").sum()) \
                if not preds.empty and "reason" in preds.columns else 0
    if total_opens > 5 and total_mom == 0:
        warnings.append(
            "  ⚠  short_term_momentum has not fired in the window despite entries — "
            "threshold may be too loose (RV2_MOM_THRESHOLD=0.0005). "
            "Consider tightening slightly after 5+ more days."
        )

    if warnings:
        print()
        for w in warnings:
            print(w)
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days",     type=int, default=7,
                    help="Number of past days to include (default: 7)")
    ap.add_argument("--strategy", type=str,
                    default="regression_v2_trendfilter_portfolio_vol",
                    help="Strategy id to report on")
    args = ap.parse_args()
    report(args.strategy, args.days)


if __name__ == "__main__":
    main()
