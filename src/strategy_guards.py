"""
Shared safety guards for all live strategies.

Each function is stateless and returns (blocked: bool, reason: str).
When the guard passes, it returns (False, "").
When the guard fires, it returns (True, "<reason_label>").

The reason labels are intentionally short and machine-readable so they
can be stored directly in the strategy output dict's "reason" field and
aggregated for later analysis.

Usage:
    from src import strategy_guards as guards

    blocked, reason = guards.check_stale_bar(bar_ts, max_age_minutes=20)
    if not blocked:
        blocked, reason = guards.check_session(bar_ts, exclude_hours=[0,1,2])
    ...
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Stale bar guard
# ---------------------------------------------------------------------------

def check_stale_bar(bar_ts: pd.Timestamp, max_age_minutes: int) -> tuple[bool, str]:
    """Block if bar_ts is older than max_age_minutes relative to now (UTC).

    Prevents strategies from acting on data that missed a live-tick refresh
    (e.g. Mac was asleep, data pipeline lagged).
    """
    if bar_ts.tzinfo is None:
        bar_ts = bar_ts.tz_localize("UTC")
    age_min = (pd.Timestamp.now(tz="UTC") - bar_ts).total_seconds() / 60.0
    if age_min > max_age_minutes:
        return True, f"stale_bar ({age_min:.0f}min > {max_age_minutes}min)"
    return False, ""


# ---------------------------------------------------------------------------
# 2. Session hours guard
# ---------------------------------------------------------------------------

def check_session(bar_ts: pd.Timestamp, exclude_hours: list[int]) -> tuple[bool, str]:
    """Block if bar_ts.hour (UTC) is in exclude_hours.

    Useful for avoiding low-liquidity Asian hours on EURUSD, etc.
    Pass an empty list to disable.
    """
    if not exclude_hours:
        return False, ""
    if bar_ts.tzinfo is None:
        bar_ts = bar_ts.tz_localize("UTC")
    if bar_ts.hour in exclude_hours:
        return True, "session"
    return False, ""


# ---------------------------------------------------------------------------
# 3. Macro event blackout guard
# ---------------------------------------------------------------------------

def check_macro_event(
    bar_ts: pd.Timestamp,
    events_path: Path,
    blackout_min: int,
) -> tuple[bool, str]:
    """Block if bar_ts falls within ±blackout_min minutes of a high-importance macro event.

    Reads event_calendar.csv with columns: importance, event_timestamp_utc.
    Returns (False, "") if the calendar file does not exist (safe default: allow trade).
    """
    if not events_path.exists():
        return False, ""

    events = pd.read_csv(events_path)
    high = events[events["importance"] == "high"]
    if high.empty:
        return False, ""

    event_times = pd.to_datetime(high["event_timestamp_utc"], utc=True)
    if bar_ts.tzinfo is None:
        bar_ts = bar_ts.tz_localize("UTC")

    bar_min = int(bar_ts.value // 60_000_000_000)
    event_mins = (event_times.values.astype("datetime64[m]").astype(np.int64))
    min_dist = int(np.min(np.abs(event_mins - bar_min)))
    if min_dist <= blackout_min:
        return True, "macro_blackout"
    return False, ""


# ---------------------------------------------------------------------------
# 4. Volatility regime guard
# ---------------------------------------------------------------------------

def check_vol_regime(
    vol: float,
    vol_reference: np.ndarray,
    top_pct: int,
) -> tuple[bool, str]:
    """Block if vol is below the (100 - top_pct) percentile of vol_reference.

    Only trades during the most volatile top_pct% of bars, filtering out
    thin / drifting markets where mean-reversion / momentum edges fade.

    Args:
        vol: current bar's volatility value (e.g. vol_6).
        vol_reference: array of reference volatility values (e.g. from test set).
        top_pct: percentage of bars to allow (e.g. 20 = top 20% most volatile).
    """
    if top_pct <= 0 or len(vol_reference) == 0:
        return False, ""
    threshold = float(np.percentile(vol_reference, 100 - top_pct))
    if vol < threshold:
        return True, "vol_regime"
    return False, ""


# ---------------------------------------------------------------------------
# 5. Daily loss cap
# ---------------------------------------------------------------------------

def check_daily_loss(
    current_equity: float,
    day_start_equity: float,
    max_loss_pct: float,
) -> tuple[bool, str]:
    """Block if equity has fallen more than max_loss_pct from today's open equity.

    Stops trading for the rest of the day after an outsized intraday loss.
    """
    if max_loss_pct <= 0 or day_start_equity <= 0:
        return False, ""
    if current_equity < day_start_equity * (1.0 - max_loss_pct):
        return True, "daily_loss"
    return False, ""


# ---------------------------------------------------------------------------
# 6. Kill switch (rolling profit factor)
# ---------------------------------------------------------------------------

def check_kill_switch(
    trade_rets: list[float],
    n: int,
    min_pf: float,
) -> tuple[bool, str]:
    """Block if the rolling profit factor of the last n completed trades < min_pf.

    Profit factor = sum(wins) / abs(sum(losses)).
    Not triggered until at least n trades have been completed.
    """
    if n <= 0 or len(trade_rets) < n:
        return False, ""
    window = trade_rets[-n:]
    gains = sum(r for r in window if r > 0)
    losses = abs(sum(r for r in window if r < 0))
    pf = (gains / losses) if losses > 0 else float("inf")
    if pf < min_pf:
        return True, "kill_switch"
    return False, ""


# ---------------------------------------------------------------------------
# 7. Drawdown kill
# ---------------------------------------------------------------------------

def check_drawdown_kill(
    peak_equity: float,
    current_equity: float,
    dd_limit: float,
) -> tuple[bool, str]:
    """Block if drawdown from peak equity exceeds dd_limit (e.g. 0.02 = 2%).

    Complements the kill switch: fires on sustained drawdown even without
    a long string of completed trades.
    """
    if dd_limit <= 0 or peak_equity <= 0:
        return False, ""
    dd = (peak_equity - current_equity) / peak_equity
    if dd >= dd_limit:
        return True, "dd_kill"
    return False, ""
