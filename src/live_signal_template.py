"""
Strategy template — copy this file to src/live_signal_<name>.py to start a new strategy.

Steps to create a new strategy:
    1. Copy this file: cp src/live_signal_template.py src/live_signal_myname_v1.py
    2. Search for TODO and fill in each section.
    3. Add your strategy to STRATEGIES in scripts/run_live_tick.py:
           {"id": "myname_v1", "module": "src.live_signal_myname_v1", "fn": "run"}
    4. Add any new config constants to src/config.py (follow the MR_* or REGRESSION_* naming).

All safety guards are already wired up in the run() function below.
Delete the guards you do not need; do not skip them without a reason.

Guard execution order (recommended):
    stale_bar → session → macro_event → vol_regime
    → daily_loss → kill_switch → drawdown_kill
    → [strategy signal logic]

Output dict schema (all keys required by run_live_tick.py and run_dashboard.py):
    strategy_id      str    e.g. "myname_v1"
    timestamp        str    "%Y-%m-%d %H:%M:%S UTC"
    signal           str    "BUY" | "SELL" | "FLAT"
    action           str    "OPEN_LONG" | "OPEN_SHORT" | "REVERSE_LONG" |
                             "REVERSE_SHORT" | "CLOSE" | "NONE"
    pred             float  your primary signal value (e.g. z-score, model pred)
    vol_6            float  current bar volatility (pass-through for logging)
    confidence       float  0–1 signal strength (or abs(pred) for rule-based)
    blocked          int    1 if a guard fired, else 0
    reason           str    guard label e.g. "stale_bar", "vol_regime", ""
    P_buy            float  buy probability / buy-side signal strength
    P_sell           float  sell probability / sell-side signal strength
    bar_return       float  last bar's realised return (ret_1) — pass-through
    signal_source    str    e.g. "myname_features_tail"
    bar_lag_hours    float  hours between bar_ts and now()
    readiness_0_100  int    0–100 proximity to a new trade (100 = trade fired)
    trade_hint       str    ≤ 280 char human explanation for dashboard
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import strategy_guards as guards

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# TODO: set your strategy ID (must match the "id" key in run_live_tick.py STRATEGIES)
STRATEGY_ID = "myname_v1"

STATE_FILE = PROJECT_ROOT / "data" / "logs" / "execution" / f"{STRATEGY_ID}_state.json"
FEATURES_DIR = PROJECT_ROOT / "data" / "features_regression_core"
FEATURES_PARQUET = FEATURES_DIR / "test.parquet"
FEATURES_CSV = FEATURES_DIR / "test.csv"
FEATURES_LIVE_TAIL_PARQUET = FEATURES_DIR / "test_live_tail.parquet"
# Used by vol_regime guard (percentile reference distribution)
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions_regression" / "test_predictions.parquet"
# Used by macro_event guard
MACRO_EVENTS_PATH = PROJECT_ROOT / "data" / "raw" / "macro" / "event_calendar.csv"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Import constants from src/config.py.

    TODO: add your strategy-specific constants to config.py, import them here,
    and return them in this dict. Keep guard keys using the standard names below
    so the guard chain works without modification.
    """
    from src.config import (
        MACRO_EVENT_BLACKOUT_MIN,
        MAX_DAILY_LOSS_PCT,
        REGRESSION_DD_KILL,          # TODO: replace with YOUR_DD_KILL if you want a different value
        REGRESSION_KILL_SWITCH_N,    # TODO: replace with YOUR_KILL_SWITCH_N
        REGRESSION_KILL_SWITCH_PF,   # TODO: replace with YOUR_KILL_SWITCH_PF
        REGRESSION_PAUSE_BARS,       # TODO: replace with YOUR_PAUSE_BARS
        REGRESSION_VOL_PCT,          # TODO: replace with YOUR_VOL_PCT (0 = disabled)
        MR_MAX_BAR_AGE_MINUTES,      # TODO: replace with YOUR_MAX_BAR_AGE_MINUTES
        SESSION_EXCLUDE_HOURS,
    )
    return {
        # Universal guard settings — keep these key names
        "max_bar_age_minutes": MR_MAX_BAR_AGE_MINUTES,
        "session_exclude_hours": SESSION_EXCLUDE_HOURS,
        "macro_blackout_min": MACRO_EVENT_BLACKOUT_MIN,
        "vol_pct": REGRESSION_VOL_PCT,          # 0 = disabled
        "max_daily_loss_pct": MAX_DAILY_LOSS_PCT,
        "kill_switch_n": REGRESSION_KILL_SWITCH_N,
        "kill_switch_pf": REGRESSION_KILL_SWITCH_PF,
        "dd_kill": REGRESSION_DD_KILL,
        "pause_bars": REGRESSION_PAUSE_BARS,

        # TODO: add your strategy-specific config keys here, e.g.:
        # "my_threshold": MY_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def _default_state() -> dict:
    """Initialise persistent state.

    The standard keys (equity tracking, kill-switch, pause) are required by the
    guard chain. Add your own keys for position tracking, hold timers, etc.
    """
    return {
        # Position tracking
        "position": 0,           # 0 = flat, 1 = long, -1 = short
        # Equity / risk tracking (required by guard chain)
        "n_trades": 0,
        "trade_rets": [],
        "peak_equity": 1.0,
        "current_equity": 1.0,
        "trade_start_equity": 1.0,
        "day_start_equity": 1.0,
        "current_day": -1,
        "pause_remaining": 0,
        "paused": False,
        # TODO: add your own state keys here, e.g.:
        # "bars_held": 0,
    }


def _load_state() -> dict:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                raw = json.load(f)
            for k, v in _default_state().items():
                raw.setdefault(k, v)
            return raw
        except Exception:
            pass
    return _default_state()


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Feature loading (standard live-tail pattern)
# ---------------------------------------------------------------------------

def _load_features() -> pd.DataFrame | None:
    """Load features, preferring the live tail if it contains newer bars."""
    tail_path = None
    test_end_ts = None

    if FEATURES_PARQUET.exists():
        df_test = pd.read_parquet(FEATURES_PARQUET)
        df_test["timestamp"] = pd.to_datetime(df_test["timestamp"], utc=True)
        test_end_ts = df_test["timestamp"].max()

    if FEATURES_LIVE_TAIL_PARQUET.exists():
        df_tail = pd.read_parquet(FEATURES_LIVE_TAIL_PARQUET)
        df_tail["timestamp"] = pd.to_datetime(df_tail["timestamp"], utc=True)
        if not df_tail.empty:
            tail_end_ts = df_tail["timestamp"].max()
            if test_end_ts is None or tail_end_ts > test_end_ts:
                tail_path = FEATURES_LIVE_TAIL_PARQUET

    if tail_path is not None:
        df_test = pd.read_parquet(FEATURES_PARQUET) if FEATURES_PARQUET.exists() else pd.DataFrame()
        df_tail = pd.read_parquet(tail_path)
        df = pd.concat([df_test, df_tail], ignore_index=True) if not df_test.empty else df_tail
    elif FEATURES_PARQUET.exists():
        df = pd.read_parquet(FEATURES_PARQUET)
    elif FEATURES_CSV.exists():
        df = pd.read_csv(FEATURES_CSV)
    else:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return df


def _load_vol_reference() -> np.ndarray | None:
    """Reference vol distribution for the percentile vol_regime guard."""
    if not PREDICTIONS_PATH.exists():
        return None
    ref = pd.read_parquet(PREDICTIONS_PATH)
    if "vol_6" not in ref.columns:
        return None
    return ref["vol_6"].fillna(0).values.astype(float)


# ---------------------------------------------------------------------------
# Trade record helper
# ---------------------------------------------------------------------------

def _record_trade_close(state: dict) -> None:
    trade_ret = state["current_equity"] / state.get("trade_start_equity", state["current_equity"]) - 1
    state["trade_rets"].append(float(trade_ret))
    state["trade_rets"] = state["trade_rets"][-100:]
    state["n_trades"] = state.get("n_trades", 0) + 1


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(n_bars: int = 1) -> list[dict]:
    """Generate signals for the last n_bars.

    Returns a list of output dicts (one per bar) in the standard schema.
    """
    cfg = _load_config()

    df = _load_features()
    if df is None or df.empty:
        logging.error("[%s] No feature file found", STRATEGY_ID)
        return []

    # TODO: replace with your required columns
    needed: list[str] = ["ret_1"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        logging.error("[%s] Feature file missing columns: %s", STRATEGY_ID, missing)
        return []

    vol_ref = _load_vol_reference()
    state = _load_state()
    output: list[dict] = []

    for i in range(n_bars):
        row_idx = len(df) - n_bars + i
        if row_idx < 0:
            continue

        row = df.iloc[row_idx]
        bar_ts = row["timestamp"]
        if bar_ts.tzinfo is None:
            bar_ts = bar_ts.tz_localize("UTC")

        ret_1 = float(row["ret_1"]) if pd.notna(row.get("ret_1")) else 0.0
        vol = float(row.get("vol_6", 0) or 0)

        # ── Guard chain ────────────────────────────────────────────────────
        blocked = False
        block_reason = ""

        # 1. Stale bar — data too old to be trusted
        if not blocked:
            blocked, block_reason = guards.check_stale_bar(bar_ts, cfg["max_bar_age_minutes"])

        # 2. Session hours
        if not blocked:
            blocked, block_reason = guards.check_session(bar_ts, cfg["session_exclude_hours"])

        # 3. Macro event blackout
        if not blocked:
            blocked, block_reason = guards.check_macro_event(
                bar_ts, MACRO_EVENTS_PATH, cfg["macro_blackout_min"]
            )

        # 4. Vol regime (skip if vol_pct == 0 or no reference data)
        if not blocked and cfg.get("vol_pct", 0) > 0 and vol_ref is not None:
            blocked, block_reason = guards.check_vol_regime(vol, vol_ref, cfg["vol_pct"])

        # 5. Daily loss cap
        bar_day = int(bar_ts.normalize().value)
        if bar_day != state["current_day"]:
            state["current_day"] = bar_day
            state["day_start_equity"] = state["current_equity"]
        if not blocked:
            blocked, block_reason = guards.check_daily_loss(
                state["current_equity"], state["day_start_equity"], cfg["max_daily_loss_pct"]
            )

        # 6. Pause countdown (kill switch / dd kill use this)
        if state["paused"] and state["pause_remaining"] > 0:
            state["pause_remaining"] -= 1
            if state["pause_remaining"] <= 0:
                state["paused"] = False
                logging.info("[%s] Resuming after pause", STRATEGY_ID)
        if not blocked and state["paused"]:
            blocked = True
            block_reason = "paused"
        # ── End universal guard chain ──────────────────────────────────────

        # ── TODO: compute your signal here ────────────────────────────────
        # Example pattern:
        #   my_signal_value = ...
        #   desired = 1 if my_signal_value > threshold else (-1 if my_signal_value < -threshold else 0)
        desired = 0       # TODO: replace with your signal logic
        pred_value = 0.0  # TODO: replace with your primary signal value
        confidence_value = 0.0  # TODO: replace with |pred| or model probability
        # ── End signal logic ───────────────────────────────────────────────

        action = "NONE"
        signal = "FLAT"
        current_pos = int(state.get("position", 0))

        if not blocked:
            # 7. Kill switch and drawdown kill — only when about to change position
            if desired != current_pos and desired != 0:
                _ks_blocked, _ks_reason = guards.check_kill_switch(
                    state["trade_rets"], cfg["kill_switch_n"], cfg["kill_switch_pf"]
                )
                if _ks_blocked:
                    state["paused"] = True
                    state["pause_remaining"] = cfg["pause_bars"]
                    blocked = True
                    block_reason = _ks_reason
                    desired = 0
                if not blocked:
                    _dd_blocked, _dd_reason = guards.check_drawdown_kill(
                        state["peak_equity"], state["current_equity"], cfg["dd_kill"]
                    )
                    if _dd_blocked:
                        state["paused"] = True
                        state["pause_remaining"] = cfg["pause_bars"]
                        blocked = True
                        block_reason = _dd_reason
                        desired = 0

        if not blocked:
            # Apply position transitions
            if desired == 1 and current_pos != 1:
                signal = "BUY"
                action = "OPEN_LONG" if current_pos == 0 else "REVERSE_LONG"
                if current_pos != 0:
                    _record_trade_close(state)
                state["trade_start_equity"] = state["current_equity"]
                state["position"] = 1
            elif desired == -1 and current_pos != -1:
                signal = "SELL"
                action = "OPEN_SHORT" if current_pos == 0 else "REVERSE_SHORT"
                if current_pos != 0:
                    _record_trade_close(state)
                state["trade_start_equity"] = state["current_equity"]
                state["position"] = -1
            elif desired == 0 and current_pos != 0:
                signal = "FLAT"
                action = "CLOSE"
                _record_trade_close(state)
                state["position"] = 0
            else:
                signal = "BUY" if current_pos == 1 else ("SELL" if current_pos == -1 else "FLAT")
                action = "NONE"

        if blocked:
            signal = "BUY" if current_pos == 1 else ("SELL" if current_pos == -1 else "FLAT")
            action = "NONE"

        # Update equity simulation
        current_pos_after = int(state.get("position", 0))
        if current_pos_after != 0:
            state["current_equity"] = state.get("current_equity", 1.0) * (1 + current_pos_after * ret_1)
        state["peak_equity"] = max(state.get("peak_equity", 1.0), state.get("current_equity", 1.0))

        lag_h = (pd.Timestamp.now(tz="UTC") - bar_ts).total_seconds() / 3600.0

        # TODO: replace with your readiness / hint logic
        readiness = 100 if action != "NONE" else (15 if blocked else 50)
        trade_hint = f"Blocked: {block_reason}" if blocked else "Monitoring"

        output.append({
            "strategy_id": STRATEGY_ID,
            "timestamp": bar_ts.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "signal": signal,
            "action": action,
            "pred": round(pred_value, 6),
            "vol_6": round(vol, 8),
            "confidence": round(confidence_value, 6),
            "blocked": 1 if blocked else 0,
            "reason": block_reason,
            "P_buy": round(max(pred_value, 0.0), 6),
            "P_sell": round(max(-pred_value, 0.0), 6),
            "bar_return": float(ret_1),
            "signal_source": f"{STRATEGY_ID}_features_tail",
            "bar_lag_hours": round(lag_h, 2),
            "readiness_0_100": readiness,
            "trade_hint": trade_hint,
        })

    _save_state(state)
    return output


def reset_state() -> None:
    _save_state(_default_state())
    logging.info("[%s] State reset", STRATEGY_ID)
