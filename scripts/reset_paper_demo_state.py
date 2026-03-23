#!/usr/bin/env python3
"""
Reset paper_sim_state.json and optional per-strategy *state.json files to a clean baseline.

Use after confusion from multi-strategy demo bugs or when you want equity/positions to start at 1.0 / flat.

Usage (from repo root):
  python scripts/reset_paper_demo_state.py
  python scripts/reset_paper_demo_state.py --also-strategy-state
  python scripts/reset_paper_demo_state.py --signals
  python scripts/reset_paper_demo_state.py --demo-orders
  python scripts/reset_paper_demo_state.py --also-strategy-state --signals
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXEC = ROOT / "data" / "logs" / "execution"
PREDICTIONS_LIVE = ROOT / "data" / "predictions" / "predictions_live.csv"
TRADE_DECISIONS = EXEC / "trade_decisions.csv"
PAPER_SIM_CSV = EXEC / "paper_simulation.csv"


def strip_demo_rows_from_trade_csv(path: Path) -> None:
    """Remove rows where mode=demo; keep sim/paper rows. Rewrites file or removes if empty."""
    if not path.exists():
        print(f"(skip) {path.name} — not present")
        return
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            if not fieldnames:
                print(f"(skip) {path.name} — no header")
                return
            rows = list(reader)
    except OSError as e:
        print(f"Could not read {path}: {e}", file=sys.stderr)
        return

    kept = [r for r in rows if str(r.get("mode", "")).strip().lower() != "demo"]
    removed = len(rows) - len(kept)
    try:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(kept)
    except OSError as e:
        print(f"Could not write {path}: {e}", file=sys.stderr)
        return

    if not kept:
        path.unlink(missing_ok=True)
        print(f"Removed {path.name} (only demo rows; file deleted)")
    else:
        print(f"{path.name}: removed {removed} demo row(s), kept {len(kept)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--also-strategy-state",
        action="store_true",
        help="Also reset regression_v1_state.json and mean_reversion_v1_state.json to defaults (not classifier).",
    )
    p.add_argument(
        "--signals",
        action="store_true",
        help="Delete live signal + order log CSVs (predictions_live, trade_decisions, paper_simulation). Next tick recreates them.",
    )
    p.add_argument(
        "--demo-orders",
        action="store_true",
        help="Remove only demo broker rows (mode=demo) from trade_decisions.csv and paper_simulation.csv; keeps paper/sim rows.",
    )
    args = p.parse_args()

    paper = EXEC / "paper_sim_state.json"
    default_strategies = ["classifier_v1", "regression_v1", "mean_reversion_v1"]
    data = {
        sid: {"position": "flat", "equity": 1.0} for sid in default_strategies
    }
    for sid in default_strategies:
        data[f"{sid}_paper"] = {"position": "flat", "equity": 1.0}
    data["_demo_broker_pos"] = "flat"
    paper.parent.mkdir(parents=True, exist_ok=True)
    with open(paper, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {paper} (all strategies flat, equity 1.0, parallel *_paper keys reset, _demo_broker_pos=flat)")

    if args.also_strategy_state:
        reg = EXEC / "regression_v1_state.json"
        if reg.exists():
            s = {
                "n_trades": 0,
                "trade_rets": [],
                "peak_equity": 1.0,
                "current_equity": 1.0,
                "pause_remaining": 0,
                "paused": False,
                "current_position": 0,
                "trade_start_equity": 1.0,
            }
            with open(reg, "w") as f:
                json.dump(s, f, indent=2)
            print(f"Reset {reg}")
        mr = EXEC / "mean_reversion_v1_state.json"
        mr_default = {
            "position": 0,
            "bars_held": 0,
            "n_trades": 0,
            "trade_rets": [],
            "peak_equity": 1.0,
            "current_equity": 1.0,
            "trade_start_equity": 1.0,
            "day_start_equity": 1.0,
            "current_day": -1,
            "pause_remaining": 0,
            "paused": False,
        }
        with open(mr, "w") as f:
            json.dump(mr_default, f, indent=2)
        print(f"Wrote {mr}")

    if args.demo_orders:
        strip_demo_rows_from_trade_csv(TRADE_DECISIONS)
        strip_demo_rows_from_trade_csv(PAPER_SIM_CSV)

    if args.signals:
        for path, label in (
            (PREDICTIONS_LIVE, "predictions_live.csv"),
            (TRADE_DECISIONS, "trade_decisions.csv"),
            (PAPER_SIM_CSV, "paper_simulation.csv"),
        ):
            try:
                if path.exists():
                    path.unlink()
                    print(f"Removed {path} ({label})")
                else:
                    print(f"(skip) {label} — not present")
            except OSError as e:
                print(f"Could not remove {path}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
