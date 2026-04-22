#!/usr/bin/env python3
"""
One-shot helper: run process_signal in a fresh process so the cTrader Twisted reactor
is not reused (ReactorNotRestartable). Invoked only from execution._process_signal_ctrader_subprocess.
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


def main() -> None:
    data = json.loads(sys.stdin.read())

    # `src.execution` calls `logging.basicConfig(handlers=[StreamHandler(sys.stdout), ...])`
    # at import time, so any `logging.info(...)` it emits would corrupt the single JSON
    # line the parent expects on our stdout. Redirect stdout → stderr BEFORE the import
    # so the StreamHandler captures stderr, and only emit our JSON on the original stdout.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    from src.execution import process_signal

    at, br = process_signal(
        data["row"],
        data["current_position"],
        dry_run=bool(data["dry_run"]),
        account_id_override=data.get("account_id_override"),
    )

    sys.stdout = real_stdout
    json.dump({"action_taken": at, "broker_response": br}, sys.stdout)


if __name__ == "__main__":
    main()
