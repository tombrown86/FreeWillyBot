"""
Subprocess entry-point: refresh test_live_tail.parquet from cTrader bars.

Called by run_live_tick.py as a subprocess so the Twisted reactor runs in an
isolated process and does not conflict with the reactor used by execution.py
(get_open_positions) later in the same live-tick process.

Exit 0 on success, 1 on failure (caller logs and continues with existing tail).
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_env = PROJECT_ROOT / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env)
    except ImportError:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

if __name__ == "__main__":
    try:
        from src.build_features_regression_core import run_live_tail_ctrader
        run_live_tail_ctrader()
        sys.exit(0)
    except Exception as exc:
        logging.error("refresh_live_tail_ctrader failed: %s", exc)
        sys.exit(1)
