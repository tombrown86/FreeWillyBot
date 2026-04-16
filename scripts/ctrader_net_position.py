#!/usr/bin/env python3
"""
One-shot: print net EURUSD position for a cTrader login (fresh process = clean Twisted reactor).

Usage (from repo root, .env loaded):
  .venv/bin/python scripts/ctrader_net_position.py 4247810

Stdout: one JSON line {"pos": "flat"|"long"|"short"}
Exit 1 on error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_env = ROOT / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(_env)
    except ImportError:
        pass


def main() -> int:
    if len(sys.argv) < 2:
        print('{"pos": "flat", "error": "missing account login arg"}', file=sys.stderr)
        return 1
    try:
        acct = int(sys.argv[1])
    except ValueError:
        print('{"pos": "flat", "error": "invalid account"}', file=sys.stderr)
        return 1

    from src.execution import get_open_positions

    resp = get_open_positions(account_id_override=acct)
    if resp.get("simulated"):
        print(json.dumps({"pos": "flat", "note": "simulated"}))
        return 0
    positions = resp.get("positions") or []
    if not positions:
        print(json.dumps({"pos": "flat"}))
        return 0
    # Same rule as execution._current_position_from_broker for cTrader: first position side
    side = (positions[0].get("tradeSide") or "").upper()
    pos = "long" if side == "BUY" else "short" if side == "SELL" else "flat"
    out = {"pos": pos, "n_positions": len(positions)}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
