"""
Portfolio engine — permission, sizing, and state management layer.

This module sits between the signal layer (live_signal_*.run()) and the
execution layer (run_live_tick._run_strategy()).  It is intentionally
stateless between calls: all persistent data lives in portfolio_state.json
so restarts never lose track of equity or kill-switch status.

Public API
----------
load_portfolio_state()  -> dict
save_portfolio_state(state)

enrich_signal_desired_position(signal) -> None
    In-place: set ``desired_position`` from ``action`` / ``signal`` when missing.

get_target_position(signal, state, recent_closes, cfg, strategy_id=...) -> dict
    Signed target size: ``target_units``, ``size_abs``, ``direction``, ``note``.

is_strategy_allowed(strategy_id, signal, state, cfg) -> (bool, reason_str)
    Permission layer.  Returns (True, "") when trading is allowed.
    Returns (False, reason) for any block condition.

compute_size(signal, state, recent_closes, cfg, strategy_id=...) -> float
    Sizing layer.  Returns the final position size in "units".
    Returns 0.0 when the result falls below the minimum executable size.

record_trade_result(state, trade_ret, strategy_id, cfg) -> dict
    Updates state after a trade closes.  Checks and fires portfolio-level
    kill switches.  Caller must save_portfolio_state(updated_state).

load_portfolio_config() -> dict
    Convenience: loads src.config_portfolio into a plain dict.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH   = PROJECT_ROOT / "data" / "logs" / "execution" / "portfolio_state.json"

log = logging.getLogger(__name__)


# ── State helpers ─────────────────────────────────────────────────────────────

def _default_state() -> dict:
    today = datetime.now(timezone.utc).date().isoformat()
    return {
        "equity":           1.0,
        "peak_equity":      1.0,
        "day_start_equity": 1.0,
        "last_day_date":    today,
        "trade_rets":       [],       # list[float], most recent last
        "active_strategy":  None,
        "active_direction": 0,
        "active_size":      0.0,
        "pause_until_utc":  None,     # ISO string or null
        "loss_streak":      0,
        "portfolio_paused": False,
        "pause_reason":     "",
    }


def load_portfolio_state() -> dict:
    """Load portfolio_state.json, creating it with defaults if absent."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH) as f:
                data = json.load(f)
            # Merge in any keys added since last write (forward-compat)
            defaults = _default_state()
            for k, v in defaults.items():
                data.setdefault(k, v)
            return data
        except Exception as exc:
            log.warning("portfolio_state.json unreadable (%s) — resetting to defaults", exc)
    state = _default_state()
    save_portfolio_state(state)
    return state


def save_portfolio_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def load_portfolio_config() -> dict:
    """Load src.config_portfolio into a plain dict."""
    from src import config_portfolio as cp
    return {
        k: getattr(cp, k)
        for k in dir(cp)
        if k.startswith("PORTFOLIO_")
    }


def enrich_signal_desired_position(signal: dict) -> None:
    """Set ``desired_position`` from ``action`` / ``signal`` when not already set.

    Mutates ``signal`` in place. Values: +1 long, -1 short, 0 flat.
    """
    if signal.get("desired_position") is not None:
        try:
            signal["desired_position"] = int(signal["desired_position"])
        except (TypeError, ValueError):
            signal["desired_position"] = 0
        return

    act = str(signal.get("action", "") or "").strip().upper()
    if act in (
        "OPEN_LONG",
        "REVERSE_LONG",
        "REVERSE_TO_LONG",
        "REVERSE_TO_LONG_SIMULATED",
    ):
        signal["desired_position"] = 1
    elif act in (
        "OPEN_SHORT",
        "REVERSE_SHORT",
        "REVERSE_TO_SHORT",
        "REVERSE_TO_SHORT_SIMULATED",
    ):
        signal["desired_position"] = -1
    elif act in ("CLOSE", "HOLD", "NONE", ""):
        signal["desired_position"] = 0
    else:
        sig = str(signal.get("signal", "") or "").lower()
        if sig == "long":
            signal["desired_position"] = 1
        elif sig == "short":
            signal["desired_position"] = -1
        else:
            signal["desired_position"] = 0


def desired_direction_from_signal(signal: dict) -> int:
    """Return +1 / -1 / 0 for long / short / flat intent."""
    enrich_signal_desired_position(signal)
    return int(signal.get("desired_position", 0))


def get_target_position(
    signal: dict,
    state: dict,
    recent_closes: pd.Series | None,
    cfg: dict,
    *,
    strategy_id: str | None = None,
) -> dict[str, Any]:
    """Execution-layer hint: signed size in account units.

    Returns
    -------
    dict with keys:
        target_units  — +size for long, -size for short, 0 flat
        size_abs      — magnitude after vol/trend/DD/streak scaling (may be 0)
        direction     — -1, 0, +1
        note          — empty or reason size is zero
    """
    d = desired_direction_from_signal(signal)
    sz = compute_size(signal, state, recent_closes, cfg, strategy_id=strategy_id)
    if d == 0 or sz <= 0:
        return {
            "target_units": 0.0,
            "size_abs":     0.0,
            "direction":    0,
            "note":         "flat_or_zero_size",
        }
    return {
        "target_units": float(d) * float(sz),
        "size_abs":     float(sz),
        "direction":    d,
        "note":         "",
    }


def event_strategy_should_flatten_others(
    incoming_strategy_id: str,
    cfg: dict,
) -> bool:
    """Batch 5 conflict policy hook: event-driven strategies take precedence.

    When ``incoming_strategy_id`` is listed in PORTFOLIO_EVENT_STRATEGY_IDS,
    the live tick runner should process non-event strategies first and flatten
    them before the event strategy sends orders. Empty tuple = disabled.
    """
    ev = cfg.get("PORTFOLIO_EVENT_STRATEGY_IDS") or ()
    return bool(ev) and incoming_strategy_id in ev


# ── Daily reset ───────────────────────────────────────────────────────────────

def _maybe_daily_reset(state: dict) -> dict:
    """Reset day_start_equity at the start of a new UTC calendar day."""
    today = datetime.now(timezone.utc).date().isoformat()
    if state.get("last_day_date") != today:
        state["day_start_equity"] = state["equity"]
        state["last_day_date"]    = today
    return state


# ── Permission layer ──────────────────────────────────────────────────────────

def is_strategy_allowed(
    strategy_id: str,
    signal: dict,
    state: dict,
    cfg: dict,
) -> tuple[bool, str]:
    """Return (True, "") when the strategy may trade, (False, reason) otherwise.

    Checks (in priority order):
    1. portfolio_paused hard flag
    2. pause_until_utc cooldown timer
    3. Portfolio DD kill threshold
    4. Portfolio rolling PF kill threshold
    5. Bar age > PORTFOLIO_MAX_BAR_AGE_HOURS
    6. Conflicting open position from another strategy
    """
    state = _maybe_daily_reset(state)
    now_utc = datetime.now(timezone.utc)

    # 1. Hard pause flag
    if state.get("portfolio_paused"):
        return False, f"portfolio_paused ({state.get('pause_reason', '')})"

    # 2. Cooldown timer
    pause_until = state.get("pause_until_utc")
    if pause_until:
        try:
            pu = datetime.fromisoformat(pause_until.replace("Z", "+00:00"))
            if pu.tzinfo is None:
                pu = pu.replace(tzinfo=timezone.utc)
            if now_utc < pu:
                remaining = int((pu - now_utc).total_seconds() / 60)
                return False, f"portfolio_cooldown ({remaining}min remaining)"
        except Exception:
            pass

    # 3. Portfolio DD kill
    peak   = float(state.get("peak_equity", 1.0))
    equity = float(state.get("equity", 1.0))
    if peak > 0:
        dd = (peak - equity) / peak
        kill_dd = float(cfg.get("PORTFOLIO_KILL_DD", 0.03))
        if dd >= kill_dd:
            pause_h = int(cfg.get("PORTFOLIO_PAUSE_HOURS", 6))
            state["pause_until_utc"] = (
                now_utc + timedelta(hours=pause_h)
            ).isoformat()
            state["pause_reason"] = f"portfolio_dd_kill (DD={dd:.2%})"
            save_portfolio_state(state)
            return False, state["pause_reason"]

    # 4. Portfolio rolling PF kill
    trade_rets = state.get("trade_rets", [])
    n_kill     = int(cfg.get("PORTFOLIO_KILL_PF_N", 20))
    min_pf     = float(cfg.get("PORTFOLIO_KILL_PF_MIN", 0.85))
    if len(trade_rets) >= n_kill:
        window = trade_rets[-n_kill:]
        gains  = sum(r for r in window if r > 0)
        losses = abs(sum(r for r in window if r < 0))
        pf     = gains / losses if losses > 0 else float("inf")
        if pf < min_pf:
            pause_h = int(cfg.get("PORTFOLIO_PAUSE_HOURS", 6))
            state["pause_until_utc"] = (
                now_utc + timedelta(hours=pause_h)
            ).isoformat()
            state["pause_reason"] = f"portfolio_pf_kill (PF={pf:.3f} < {min_pf})"
            save_portfolio_state(state)
            return False, state["pause_reason"]

    # 5. Bar age (portfolio-level stricter limit)
    bar_ts_raw = signal.get("bar_ts") or signal.get("bar_time") or signal.get("timestamp")
    if bar_ts_raw is not None:
        try:
            bar_ts = pd.Timestamp(bar_ts_raw)
            if bar_ts.tzinfo is None:
                bar_ts = bar_ts.tz_localize("UTC")
            now_ts = pd.Timestamp.now(tz="UTC")
            age_hours = (now_ts - bar_ts).total_seconds() / 3600.0
            max_age = float(cfg.get("PORTFOLIO_MAX_BAR_AGE_HOURS", 0.5))
            if age_hours > max_age:
                return False, f"portfolio_stale_bar ({age_hours:.1f}h > {max_age}h)"
        except Exception:
            pass

    # 6. Conflicting open position from another strategy
    active_strat = state.get("active_strategy")
    active_dir   = int(state.get("active_direction", 0))
    desired      = desired_direction_from_signal(signal)
    if (
        active_strat is not None
        and active_strat != strategy_id
        and active_dir != 0
        and desired != 0
        and desired != active_dir
    ):
        return False, (
            f"conflict: {active_strat} holds {'long' if active_dir>0 else 'short'}, "
            f"{strategy_id} wants {'long' if desired>0 else 'short'}"
        )

    # 7. Sibling conflict — same-family strategies (same underlying signal, different sizing)
    # block execution orders when a sibling already has an open position so the shared
    # demo / live account never carries doubled exposure.
    siblings = cfg.get("PORTFOLIO_STRATEGY_SIBLINGS") or {}
    my_siblings = siblings.get(strategy_id) or []
    if isinstance(my_siblings, str):
        my_siblings = [my_siblings]
    if active_strat in my_siblings and active_dir != 0 and desired != 0:
        return False, (
            f"sibling_conflict: sibling {active_strat} already holds "
            f"{'long' if active_dir>0 else 'short'}; "
            f"{strategy_id} defers (paper book still tracks)"
        )

    return True, ""


# ── Sizing layer ──────────────────────────────────────────────────────────────

def _resolve_sizing_mode(
    cfg: dict,
    signal: dict,
    strategy_id: str | None,
) -> str:
    """Return full | vol_only | fixed."""
    m = signal.get("portfolio_sizing_mode")
    if isinstance(m, str) and m.strip().lower() in ("full", "vol_only", "fixed"):
        return m.strip().lower()
    by = cfg.get("PORTFOLIO_SIZING_MODE_BY_STRATEGY") or {}
    if strategy_id and strategy_id in by:
        return str(by[strategy_id]).strip().lower()
    return str(cfg.get("PORTFOLIO_SIZING_MODE_DEFAULT", "full")).strip().lower()


def _realized_vol(recent_closes: pd.Series, lookback: int) -> float:
    """Std of log-returns over the last `lookback` bars. Returns NaN if insufficient data."""
    if len(recent_closes) < lookback + 1:
        return float("nan")
    tail = recent_closes.iloc[-(lookback + 1):]
    log_rets = np.log(tail.values[1:] / tail.values[:-1])
    return float(np.std(log_rets))


def compute_size(
    signal: dict,
    state: dict,
    recent_closes: pd.Series | None,
    cfg: dict,
    *,
    strategy_id: str | None = None,
) -> float:
    """Compute the final position size in units.

    Multiplier stack (all factors clip at configured bounds):
      base_size × vol_mult × trend_mult × dd_mult × streak_mult
    Mode ``vol_only`` keeps vol_mult only; ``fixed`` uses base size only.
    Returns 0.0 if the result is below PORTFOLIO_MIN_SIZE.

    Parameters
    ----------
    signal        : output dict from live_signal_*.run()
    state         : current portfolio_state dict
    recent_closes : recent close prices (pd.Series, most recent last)
                    May be None in backtests where vol is pre-computed.
    cfg           : portfolio config dict (from load_portfolio_config())
    strategy_id   : optional override for PORTFOLIO_SIZING_MODE_BY_STRATEGY

    """
    mode = _resolve_sizing_mode(cfg, signal, strategy_id)
    base    = float(cfg.get("PORTFOLIO_BASE_SIZE",  1.0))
    min_sz  = float(cfg.get("PORTFOLIO_MIN_SIZE",   0.5))
    max_sz  = float(cfg.get("PORTFOLIO_MAX_SIZE",   2.0))

    # ── Vol multiplier ────────────────────────────────────────────────────
    vol_mult = 1.0
    if mode in ("full", "vol_only") and recent_closes is not None and len(recent_closes) > 1:
        lookback   = int(cfg.get("PORTFOLIO_VOL_LOOKBACK_BARS", 48))
        vol_target = float(cfg.get("PORTFOLIO_VOL_TARGET",      0.0003))
        clip_lo    = float(cfg.get("PORTFOLIO_VOL_CLIP_LOW",    0.5))
        clip_hi    = float(cfg.get("PORTFOLIO_VOL_CLIP_HIGH",   2.0))
        rv = _realized_vol(recent_closes, lookback)
        if not np.isnan(rv) and rv > 0:
            vol_mult = float(np.clip(vol_target / rv, clip_lo, clip_hi))

    # ── Trend-strength multiplier ─────────────────────────────────────────
    trend_mult = 1.0
    if mode == "full":
        trend_str    = abs(float(signal.get("trend_strength", 0.0) or 0.0))
        strong_thresh = float(cfg.get("PORTFOLIO_TREND_STRONG_THRESH", 0.003))
        medium_thresh = float(cfg.get("PORTFOLIO_TREND_MEDIUM_THRESH", 0.001))
        if trend_str >= strong_thresh:
            trend_mult = 1.0
        elif trend_str >= medium_thresh:
            trend_mult = 0.75
        else:
            trend_mult = 0.5

    # ── Drawdown multiplier ───────────────────────────────────────────────
    dd_mult = 1.0
    if mode == "full":
        peak    = float(state.get("peak_equity", 1.0))
        equity  = float(state.get("equity",      1.0))
        if peak > 0:
            dd = (peak - equity) / peak
            tier1 = float(cfg.get("PORTFOLIO_DD_TIER1", 0.01))
            tier2 = float(cfg.get("PORTFOLIO_DD_TIER2", 0.02))
            if dd >= tier2:
                dd_mult = 0.5
            elif dd >= tier1:
                dd_mult = 0.75

    # ── Loss-streak multiplier ────────────────────────────────────────────
    streak_mult = 1.0
    if mode == "full":
        streak      = int(state.get("loss_streak", 0))
        streak_n    = int(cfg.get("PORTFOLIO_LOSS_STREAK_N",    3))
        streak_fac  = float(cfg.get("PORTFOLIO_LOSS_STREAK_MULT", 0.5))
        if streak >= streak_n:
            streak_mult = streak_fac

    # ── Combine ───────────────────────────────────────────────────────────
    raw   = base * vol_mult * trend_mult * dd_mult * streak_mult
    final = float(np.clip(raw, min_sz, max_sz))

    if final < min_sz:
        log.debug(
            "compute_size: raw=%.3f clipped to %.3f < min %.3f → 0.0",
            raw, final, min_sz,
        )
        return 0.0

    log.debug(
        "compute_size: mode=%s base=%.2f vol=%.3f trend=%.3f dd=%.3f streak=%.3f → %.3f",
        mode, base, vol_mult, trend_mult, dd_mult, streak_mult, final,
    )
    step = float(cfg.get("PORTFOLIO_LOT_STEP", 0.01))
    if step > 0:
        final = round(final / step) * step
        final = float(np.clip(final, min_sz, max_sz))
    if final < min_sz:
        return 0.0
    return round(final, 2)


# ── Trade result recorder ─────────────────────────────────────────────────────

def record_trade_result(
    state: dict,
    trade_ret: float,
    strategy_id: str,
    cfg: dict,
) -> dict:
    """Update portfolio state after a trade closes.

    Appends trade_ret to trade_rets, updates equity/peak, manages loss streak,
    and checks portfolio-level kill thresholds.

    Parameters
    ----------
    state       : current portfolio_state dict (mutated in place AND returned)
    trade_ret   : fractional net return of the closed trade (after costs)
    strategy_id : id of the strategy that closed the trade
    cfg         : portfolio config dict

    Returns
    -------
    Updated state dict (same object, also mutated).
    """
    # Keep only last 2× kill-switch window to avoid unbounded growth
    max_keep = int(cfg.get("PORTFOLIO_KILL_PF_N", 20)) * 2
    state["trade_rets"].append(float(trade_ret))
    if len(state["trade_rets"]) > max_keep:
        state["trade_rets"] = state["trade_rets"][-max_keep:]

    # Update equity (multiplicative)
    state["equity"] = float(state["equity"]) * (1.0 + float(trade_ret))
    state["equity"] = max(state["equity"], 1e-9)

    # Update peak
    if state["equity"] > state.get("peak_equity", 1.0):
        state["peak_equity"] = state["equity"]

    # Loss streak
    if trade_ret < 0:
        state["loss_streak"] = int(state.get("loss_streak", 0)) + 1
    else:
        state["loss_streak"] = 0

    # Clear active position if this strategy was holding it
    if state.get("active_strategy") == strategy_id:
        state["active_strategy"] = None
        state["active_direction"] = 0
        state["active_size"]      = 0.0

    # Check portfolio-level kill switches (DD and PF)
    # These are checked again here so the state file is always up to date
    # even if is_strategy_allowed() was not called this bar.
    now_utc  = datetime.now(timezone.utc)
    peak     = float(state["peak_equity"])
    equity   = float(state["equity"])
    kill_dd  = float(cfg.get("PORTFOLIO_KILL_DD",  0.03))
    pause_h  = int(cfg.get("PORTFOLIO_PAUSE_HOURS", 6))

    if peak > 0 and (peak - equity) / peak >= kill_dd:
        state["pause_until_utc"] = (now_utc + timedelta(hours=pause_h)).isoformat()
        state["pause_reason"]    = (
            f"portfolio_dd_kill (DD={(peak-equity)/peak:.2%} trade={trade_ret:+.4f})"
        )
        log.warning("Portfolio DD kill fired: %s", state["pause_reason"])

    return state


# ── Convenience: open-position tracker ───────────────────────────────────────

def record_position_open(
    state: dict,
    strategy_id: str,
    direction: int,
    size: float,
) -> dict:
    """Mark that a strategy has just opened a position.

    Called by run_live_tick after an OPEN_LONG / OPEN_SHORT action succeeds.
    """
    state["active_strategy"]  = strategy_id
    state["active_direction"] = int(direction)
    state["active_size"]      = float(size)
    return state


def record_position_close(state: dict, strategy_id: str) -> dict:
    """Clear the active position when a strategy flattens or reverses."""
    if state.get("active_strategy") == strategy_id:
        state["active_strategy"]  = None
        state["active_direction"] = 0
        state["active_size"]      = 0.0
    return state


# ── Diagnostic summary ────────────────────────────────────────────────────────

def portfolio_summary(state: dict) -> dict[str, Any]:
    """Return a human-readable snapshot of current portfolio state."""
    peak   = float(state.get("peak_equity", 1.0))
    equity = float(state.get("equity",      1.0))
    dd     = (peak - equity) / peak if peak > 0 else 0.0

    rets = state.get("trade_rets", [])
    if rets:
        gains  = sum(r for r in rets if r > 0)
        losses = abs(sum(r for r in rets if r < 0))
        pf     = gains / losses if losses > 0 else float("inf")
        win_rt = sum(1 for r in rets if r > 0) / len(rets)
    else:
        pf = float("nan")
        win_rt = float("nan")

    paused = bool(state.get("portfolio_paused")) or bool(state.get("pause_until_utc"))

    return {
        "equity":               round(equity,  6),
        "peak_equity":          round(peak,    6),
        "drawdown":             round(dd,      6),
        "current_drawdown":     round(dd,      6),
        "n_trades":             len(rets),
        "profit_factor":        round(pf,      4) if not np.isinf(pf) else "inf",
        "rolling_profit_factor": round(pf,     4) if not np.isinf(pf) else "inf",
        "win_rate":             round(win_rt,  4),
        "loss_streak":          state.get("loss_streak", 0),
        "active_strategy":      state.get("active_strategy"),
        "paused":               paused,
        "pause_reason":         state.get("pause_reason", ""),
    }
