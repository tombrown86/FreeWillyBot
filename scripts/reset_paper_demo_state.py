#!/usr/bin/env python3
"""
Reset paper_sim_state.json and optional per-strategy *state.json files to a clean baseline.

Use after confusion from multi-strategy demo bugs or when you want equity/positions to start at 1.0 / flat.

Usage (from repo root):
  python scripts/reset_paper_demo_state.py
  python scripts/reset_paper_demo_state.py --also-strategy-state
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXEC = ROOT / "data" / "logs" / "execution"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--also-strategy-state",
        action="store_true",
        help="Also reset regression_v1_state.json and mean_reversion_v1_state.json to defaults (not classifier).",
    )
    args = p.parse_args()

    paper = EXEC / "paper_sim_state.json"
    default_strategies = ["classifier_v1", "regression_v1", "mean_reversion_v1"]
    data = {
        sid: {"position": "flat", "equity": 1.0} for sid in default_strategies
    }
    data["_demo_broker_pos"] = "flat"
    paper.parent.mkdir(parents=True, exist_ok=True)
    with open(paper, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {paper} (all strategies flat, equity 1.0, _demo_broker_pos=flat)")

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
