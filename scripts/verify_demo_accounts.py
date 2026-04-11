#!/usr/bin/env python3
"""
Verify each cTrader demo account in DEMO_CTRADER_ACCOUNT_BY_STRATEGY can authenticate
and return positions (Open API account list + account auth).

Each account is checked in a **fresh Python process** so the Twisted reactor used by
the cTrader SDK does not hit ReactorNotRestartable when testing multiple logins.

Run on the server from repo root (loads .env):
  .venv/bin/python scripts/verify_demo_accounts.py

Exit 0 if all accounts OK; exit 1 if any fail.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    _env = ROOT / ".env"
    if _env.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(_env)
        except ImportError:
            pass

    if not os.environ.get("PS_CTRADER_ACCESS_TOKEN", "").strip():
        print("ERROR: PS_CTRADER_ACCESS_TOKEN is not set (check .env on this machine).", file=sys.stderr)
        return 1

    from src.config_portfolio import DEMO_CTRADER_ACCOUNT_BY_STRATEGY

    by_acct: dict[int, list[str]] = {}
    for sid, acct in DEMO_CTRADER_ACCOUNT_BY_STRATEGY.items():
        by_acct.setdefault(int(acct), []).append(sid)

    if not by_acct:
        print("DEMO_CTRADER_ACCOUNT_BY_STRATEGY is empty — nothing to verify.")
        return 0

    py = ROOT / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)

    failed = False
    for acct_id in sorted(by_acct.keys()):
        strategies = ", ".join(by_acct[acct_id])
        print(f"\n--- Account login/ctid {acct_id} ({strategies}) ---")
        try:
            r = subprocess.run(
                [
                    str(py),
                    "-m",
                    "src.execution",
                    "--positions",
                    "--account-id",
                    str(acct_id),
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=45,
                env={**os.environ},
            )
        except subprocess.TimeoutExpired:
            print("FAIL: subprocess timed out after 45s")
            failed = True
            continue

        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            print(f"FAIL: exit code {r.returncode}")
            for line in out.splitlines()[-8:]:
                print(f"  {line}")
            failed = True
            continue
        if "not authorized for this access token" in out:
            print("FAIL: OAuth token does not include this account (re-authorize in Playground).")
            for line in out.splitlines():
                if "Granted:" in line or "not authorized" in line:
                    print(f"  {line.strip()}")
            failed = True
            continue
        if "RuntimeError" in out or "cTrader error" in out:
            print("FAIL:")
            for line in out.splitlines()[-6:]:
                print(f"  {line}")
            failed = True
            continue
        if "'broker': 'ctrader'" in out.replace('"', "'") or '"broker": "ctrader"' in out:
            print("OK: authenticated (see stdout for positions).")
        elif "ctidTraderAccountId" in out:
            print("OK: authenticated.")
        else:
            print(f"FAIL: unexpected output (last lines):\n{out[-800:]!r}")
            failed = True

    print()
    if failed:
        print(
            "One or more accounts failed. Typical fixes:\n"
            "  • Re-authorize the app at https://connect.spotware.com/playground — generate a new access token\n"
            "    after logging into the broker; the Account List response must list every login you use.\n"
            "  • New demo accounts must belong to the same cTrader user as the OAuth flow; update .env with the new token.\n"
            "  • Confirm CTRADER_HOST_TYPE=demo in src/config.py for demo accounts."
        )
        return 1
    print("All listed demo accounts responded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
