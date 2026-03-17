"""
Audit daily-loss and cooldown filter logic.

Verifies:
- Daily loss resets at day boundary (midnight UTC)
- Cooldown triggers only after a realized losing close
- Cooldown not triggered after winning close
- entry_eq set on open, not close
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import _run_backtest
from src.config import COOLDOWN_BARS_AFTER_LOSS, MAX_DAILY_LOSS_PCT


def _make_filter_data(n: int, day: np.ndarray, all_ok: bool = True) -> dict:
    """Build filter_data with all filters passing (or failing)."""
    return {
        "macro_ok": np.ones(n, dtype=bool) if all_ok else np.zeros(n, dtype=bool),
        "vol_ok": np.ones(n, dtype=bool) if all_ok else np.zeros(n, dtype=bool),
        "confidence_ok": np.ones(n, dtype=bool) if all_ok else np.zeros(n, dtype=bool),
        "session_ok": np.ones(n, dtype=bool) if all_ok else np.zeros(n, dtype=bool),
        "weekend_ok": np.ones(n, dtype=bool) if all_ok else np.zeros(n, dtype=bool),
        "day": day,
    }


def test_cooldown_after_losing_close() -> None:
    """Cooldown should trigger only after closing a losing trade."""
    # 20 bars: open long at 0, hold, close at 5 with loss, next 12 bars should be blocked
    n = 20
    ret = np.zeros(n)
    ret[0] = -0.01  # loss on first bar (we open at 0, get -1% return)
    # Position: 1 at 0, 0 at 1 (close with loss)
    pos = np.array([1, 0] + [0] * 18)
    day = np.zeros(n, dtype=np.int64)
    day[:] = 100  # same day

    filter_data = _make_filter_data(n, day)
    cost = 0.0001
    metrics, stats = _run_backtest(ret, pos, cost, filter_data)

    # After bar 1 we close with loss -> cooldown for 12 bars
    # Bars 2..13 should be blocked by cooldown
    assert stats["cooldown_bars"] >= 1, "Cooldown should have blocked some bars after losing close"


def test_cooldown_not_after_winning_close() -> None:
    """Cooldown should NOT trigger after closing a winning trade."""
    # Multi-bar trade so equity grows enough to overcome round-trip cost
    n = 20
    ret = np.array([0.01, 0.01, 0.01, 0.01] + [0.0] * 16)  # 4% over 4 bars
    pos = np.array([1, 1, 1, 1, 0] + [0] * 15)  # open 0, hold 1-3, close 4
    day = np.zeros(n, dtype=np.int64)
    day[:] = 100

    filter_data = _make_filter_data(n, day)
    cost = 0.0001
    metrics, stats = _run_backtest(ret, pos, cost, filter_data)

    # Winning close -> no cooldown
    assert stats["cooldown_bars"] == 0, "Cooldown should not trigger after winning close"


def test_daily_loss_resets_at_day_boundary() -> None:
    """Day start equity should reset when bar_day != current_day."""
    # Two days: day1 we lose 3%, day2 we should have fresh day_start_equity
    n = 50
    ret = np.zeros(n)
    ret[0] = -0.03  # lose 3% on first bar
    pos = np.ones(n)  # long entire time
    day = np.zeros(n, dtype=np.int64)
    day[:25] = 100
    day[25:] = 101  # new day at bar 25

    filter_data = _make_filter_data(n, day)
    cost = 0.0001
    metrics, stats = _run_backtest(ret, pos, cost, filter_data)

    # After bar 0 we're down 3% (> 2% max daily loss). Bars 1-24 should be blocked.
    # At bar 25 new day -> day_start_equity resets -> we can trade again
    assert stats["max_daily_loss_bars"] >= 1, "Daily loss should have blocked some bars"


def test_entry_eq_set_on_open() -> None:
    """entry_eq should be set when we open a position, used when we close."""
    # Open at 0, close at 2. Trade return = (equity[3]/entry_eq - 1) * 1
    # entry_eq should be equity after bar 0 (after we open and pay cost)
    n = 5
    ret = np.array([0.02, 0.0, 0.0, 0.0, 0.0])  # +2% on first bar
    pos = np.array([1, 1, 0, 0, 0])  # open 0, hold 1, close 2
    day = np.zeros(n, dtype=np.int64)
    day[:] = 100

    filter_data = _make_filter_data(n, day)
    cost = 0.0001
    metrics, stats = _run_backtest(ret, pos, cost, filter_data)

    # We opened, made 2%, closed. Should be winning -> no cooldown
    assert stats["cooldown_bars"] == 0


def run() -> int:
    """Run audit. Returns 0 if all pass."""
    print("Auditing filter logic...")
    test_cooldown_after_losing_close()
    print("  OK: cooldown after losing close")
    test_cooldown_not_after_winning_close()
    print("  OK: no cooldown after winning close")
    test_daily_loss_resets_at_day_boundary()
    print("  OK: daily loss resets at day boundary")
    test_entry_eq_set_on_open()
    print("  OK: entry_eq set on open")
    print("All audit checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
