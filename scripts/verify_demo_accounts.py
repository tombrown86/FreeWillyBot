#!/usr/bin/env python3
"""
Verify each cTrader demo account in DEMO_CTRADER_ACCOUNT_BY_STRATEGY can authenticate
and return positions (Open API account list + account auth).

Run on the server from repo root (loads .env):
  .venv/bin/python scripts/verify_demo_accounts.py

Exit 0 if all accounts OK; exit 1 if any fail.
"""

from __future__ import annotations

import os
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
    os.chdir(ROOT)
    from src.config_portfolio import DEMO_CTRADER_ACCOUNT_BY_STRATEGY
    from src.execution import get_open_positions

    token = os.environ.get("PS_CTRADER_ACCESS_TOKEN", "").strip()
    if not token:
        print("ERROR: PS_CTRADER_ACCESS_TOKEN is not set (check .env on this machine).", file=sys.stderr)
        return 1

    # Unique account ids (same login may not appear twice, but dedupe anyway)
    by_acct: dict[int, list[str]] = {}
    for sid, acct in DEMO_CTRADER_ACCOUNT_BY_STRATEGY.items():
        by_acct.setdefault(int(acct), []).append(sid)

    if not by_acct:
        print("DEMO_CTRADER_ACCOUNT_BY_STRATEGY is empty — nothing to verify.")
        return 0

    failed = False
    for acct_id in sorted(by_acct.keys()):
        strategies = ", ".join(by_acct[acct_id])
        print(f"\n--- Account login/ctid {acct_id} ({strategies}) ---")
        try:
            out = get_open_positions(account_id_override=acct_id)
            if out.get("simulated"):
                print(f"FAIL: broker returned simulated (credentials missing or wrong account).")
                print(f"       Response: {out}")
                failed = True
                continue
            pos = out.get("positions", [])
            print(f"OK: authenticated; open positions: {len(pos)}")
            if pos:
                for p in pos[:5]:
                    print(f"     {p}")
        except Exception as e:
            print(f"FAIL: {type(e).__name__}: {e}")
            failed = True

    print()
    if failed:
        print(
            "One or more accounts failed. Typical fixes:\n"
            "  • Re-authorize the app at https://connect.spotware.com/playground and update PS_CTRADER_ACCESS_TOKEN in .env\n"
            "  • Ensure each login is under the same cTrader user as the OAuth app\n"
            "  • Confirm CTRADER_HOST_TYPE=demo in src/config.py for demo accounts"
        )
        return 1
    print("All listed demo accounts responded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
