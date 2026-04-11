#!/usr/bin/env python3
"""
Reset paper_sim_state.json and optional per-strategy *state.json files to a clean baseline.

Use after confusion from multi-strategy demo bugs or when you want equity/positions to start at 1.0 / flat.

Usage (from repo root):
  python scripts/reset_paper_demo_state.py
  python scripts/reset_paper_demo_state.py --also-strategy-state
  python scripts/reset_paper_demo_state.py --signals
  python scripts/reset_paper_demo_state.py --also-strategy-state --signals
  python scripts/reset_paper_demo_state.py --close-all-accounts   # close positions on all demo accounts first
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXEC = ROOT / "data" / "logs" / "execution"
PREDICTIONS_LIVE = ROOT / "data" / "predictions" / "predictions_live.csv"
TRADE_DECISIONS = EXEC / "trade_decisions.csv"
PAPER_SIM_CSV = EXEC / "paper_simulation.csv"


def _regression_strategy_state() -> dict:
    return {
        "n_trades": 0,
        "trade_rets": [],
        "peak_equity": 1.0,
        "current_equity": 1.0,
        "pause_remaining": 0,
        "paused": False,
        "current_position": 0,
        "trade_start_equity": 1.0,
    }


def _session_breakout_state() -> dict:
    return {
        "position": 0,
        "bars_held": 0,
        "last_session_id": "",
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
        help="Reset portfolio_state.json and all per-strategy *state.json files (regression v1/v2, mean reversion, session breakout).",
    )
    p.add_argument(
        "--signals",
        action="store_true",
        help="Delete live predictions + trade CSVs (including legacy backups) and livetick lock. Next tick recreates logs.",
    )
    p.add_argument(
        "--demo-orders",
        action="store_true",
        help="Remove only demo broker rows (mode=demo) from trade_decisions.csv and paper_simulation.csv; keeps paper/sim rows.",
    )
    p.add_argument(
        "--close-all-accounts",
        action="store_true",
        help=(
            "Close open positions on EVERY account listed in DEMO_CTRADER_ACCOUNT_BY_STRATEGY "
            "before resetting state. Requires PS_CTRADER_ACCESS_TOKEN etc. to be set."
        ),
    )
    args = p.parse_args()

    from src.strategy_registry import STRATEGIES
    from src.config_portfolio import DEMO_CTRADER_ACCOUNT_BY_STRATEGY as account_map

    if args.close_all_accounts:
        py = ROOT / ".venv" / "bin" / "python"
        if not py.exists():
            py = Path(sys.executable)

        def _close_one(acct_id: int | None, label: str) -> None:
            cmd = [str(py), "-m", "src.execution", "--close-all"]
            if acct_id is not None:
                cmd.extend(["--account-id", str(acct_id)])
            print(f"Closing all positions ({label}) …")
            try:
                r = subprocess.run(
                    cmd,
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                    timeout=90,
                    env={**os.environ},
                )
                tail = (r.stdout or "") + (r.stderr or "")
                if r.returncode != 0:
                    print(f"  → exit {r.returncode}: {tail[-500:]}", file=sys.stderr)
                else:
                    print(f"  → ok")
                    if tail.strip():
                        print(f"  {tail.strip()[-400:]}")
            except Exception as e:
                print(f"  → ERROR: {e}", file=sys.stderr)

        if not account_map:
            print("DEMO_CTRADER_ACCOUNT_BY_STRATEGY is empty — closing default account only")
            _close_one(None, "default PS_CTRADER_ACCOUNT_ID")
        else:
            closed_accounts = set()
            for sid, acct_id in account_map.items():
                if acct_id in closed_accounts:
                    continue
                _close_one(acct_id, f"account {acct_id} / {sid}")
                closed_accounts.add(acct_id)

    strat_ids = [s.id for s in STRATEGIES]
    data: dict = {}
    for sid in strat_ids:
        data[sid] = {"position": "flat", "equity": 1.0}
        data[f"{sid}_paper"] = {"position": "flat", "equity": 1.0}
    # Legacy shared key (backward compat)
    data["_demo_broker_pos"] = "flat"
    # Per-account keys for each entry in the account map
    for acct_id in set(account_map.values()):
        data[f"_demo_broker_pos_{acct_id}"] = "flat"

    paper = EXEC / "paper_sim_state.json"
    paper.parent.mkdir(parents=True, exist_ok=True)
    with open(paper, "w") as f:
        json.dump(data, f, indent=2)
    per_acct_note = f", per-account pos keys: {sorted(set(account_map.values()))}" if account_map else ""
    print(
        f"Wrote {paper} ({len(strat_ids)} strategies + *_paper, _demo_broker_pos=flat{per_acct_note})"
    )

    if args.also_strategy_state:
        from src.portfolio_engine import _default_state, save_portfolio_state

        save_portfolio_state(_default_state())
        print("Reset portfolio_state.json (defaults)")

        reg = _regression_strategy_state()
        for name in (
            "regression_v1_state.json",
            "regression_v2_trendfilter_state.json",
            "regression_v2_trendfilter_portfolio_vol_state.json",
        ):
            path = EXEC / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(reg, f, indent=2)
            print(f"Reset {path}")

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

        sb = EXEC / "session_breakout_v1_state.json"
        with open(sb, "w") as f:
            json.dump(_session_breakout_state(), f, indent=2)
        print(f"Wrote {sb}")

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

        for pattern in ("trade_decisions_legacy*.csv", "paper_simulation_legacy*.csv"):
            for path in sorted(EXEC.glob(pattern)):
                try:
                    path.unlink()
                    print(f"Removed {path.name}")
                except OSError as e:
                    print(f"Could not remove {path}: {e}", file=sys.stderr)

        lock = EXEC / "run_live_tick.lock"
        if lock.exists():
            try:
                lock.unlink()
                print(f"Removed {lock.name}")
            except OSError as e:
                print(f"Could not remove lock: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
