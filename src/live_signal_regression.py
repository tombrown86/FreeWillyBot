"""
Regression strategy live signal generator.

Replays signals from data/predictions_regression/test_predictions.parquet in bar order,
applying the locked production config (selection filters + kill switch + DD kill).
Persists runtime state (kill switch, drawdown tracking) between ticks in a JSON state file.

STRATEGY_ID = "regression_v1"
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STRATEGY_ID = "regression_v1"
STATE_FILE = PROJECT_ROOT / "data" / "logs" / "execution" / "regression_v1_state.json"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions_regression" / "test_predictions.parquet"
CURSOR_FILE = PROJECT_ROOT / "data" / "logs" / "execution" / "regression_v1_cursor.json"


def _load_config() -> dict:
    """Load production config constants."""
    from src.config import (
        REGRESSION_DD_KILL,
        REGRESSION_KILL_SWITCH_N,
        REGRESSION_KILL_SWITCH_PF,
        REGRESSION_PAUSE_BARS,
        REGRESSION_PRED_THRESHOLD,
        REGRESSION_TOP_PCT,
        REGRESSION_VOL_PCT,
    )
    return {
        "top_pct": REGRESSION_TOP_PCT,
        "vol_pct": REGRESSION_VOL_PCT,
        "pred_threshold": REGRESSION_PRED_THRESHOLD,
        "kill_switch_n": REGRESSION_KILL_SWITCH_N,
        "kill_switch_pf": REGRESSION_KILL_SWITCH_PF,
        "dd_kill": REGRESSION_DD_KILL,
        "pause_bars": REGRESSION_PAUSE_BARS,
    }


def _default_state() -> dict:
    """Initial kill-switch / risk state."""
    return {
        "n_trades": 0,
        "trade_rets": [],
        "peak_equity": 1.0,
        "current_equity": 1.0,
        "pause_remaining": 0,
        "paused": False,
        "current_position": 0,       # -1 short, 0 flat, 1 long
        "trade_start_equity": 1.0,
    }


def _load_state() -> dict:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return _default_state()


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _load_cursor() -> int:
    """Return index of next bar to process."""
    CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    if CURSOR_FILE.exists():
        try:
            with open(CURSOR_FILE) as f:
                return int(json.load(f).get("cursor", 0))
        except Exception:
            pass
    return 0


def _save_cursor(cursor: int) -> None:
    CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CURSOR_FILE, "w") as f:
        json.dump({"cursor": cursor}, f)


def _position_allowed(pred: float, vol: float, all_pred: np.ndarray, all_vol: np.ndarray, cfg: dict) -> int:
    """Return desired position: 1 long, -1 short, 0 flat."""
    th_long = np.percentile(all_pred, 100 - cfg["top_pct"])
    th_short = np.percentile(all_pred, cfg["top_pct"])
    th_vol = np.percentile(all_vol, 100 - cfg["vol_pct"])

    if vol < th_vol:
        return 0
    if abs(pred) <= cfg["pred_threshold"]:
        return 0
    if pred >= th_long:
        return 1
    if pred <= th_short:
        return -1
    return 0


def _check_kill_switch(state: dict, cfg: dict) -> bool:
    """Return True if kill switch should fire (trade-based PF check)."""
    n = cfg["kill_switch_n"]
    if n <= 0 or state["n_trades"] < n:
        return False
    window = state["trade_rets"][-n:]
    gains = sum(r for r in window if r > 0)
    losses = abs(sum(r for r in window if r < 0))
    pf = (gains / losses) if losses > 0 else float("inf")
    return pf < cfg["kill_switch_pf"]


def _check_dd_kill(state: dict, cfg: dict) -> bool:
    """Return True if drawdown kill should fire."""
    if cfg["dd_kill"] <= 0 or state["peak_equity"] <= 0:
        return False
    dd = (state["peak_equity"] - state["current_equity"]) / state["peak_equity"]
    return dd >= cfg["dd_kill"]


def run(n_bars: int = 1) -> list[dict]:
    """
    Advance the cursor by n_bars through test_predictions.parquet,
    apply production filters and risk controls, return signal dicts.

    Output keys: strategy_id, timestamp, signal, action, pred, vol_6, blocked, reason, confidence
    """
    if not PREDICTIONS_PATH.exists():
        logging.error("[%s] test_predictions.parquet not found — run predict_regression_test first", STRATEGY_ID)
        return []

    df = pd.read_parquet(PREDICTIONS_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    cursor = _load_cursor()
    if cursor >= len(df):
        logging.warning("[%s] All bars consumed (cursor=%d, total=%d) — resetting", STRATEGY_ID, cursor, len(df))
        cursor = 0
        _save_cursor(0)

    cfg = _load_config()
    state = _load_state()

    # Use full dataset for percentile thresholds (consistent with backtests)
    all_pred = df["pred"].values.astype(float)
    all_vol = df["vol_6"].fillna(0).values.astype(float)

    output = []
    for offset in range(n_bars):
        idx = cursor + offset
        if idx >= len(df):
            break

        bar = df.iloc[idx]
        timestamp = str(bar["timestamp"])
        pred = float(bar["pred"])
        vol = float(bar.get("vol_6", 0) or 0)
        target_ret = float(bar.get("target_ret", 0) or 0)

        blocked = False
        reason = ""
        action = "NONE"
        signal = "FLAT"

        # --- resume from timed pause ---
        if state["paused"] and cfg["pause_bars"] > 0:
            state["pause_remaining"] -= 1
            if state["pause_remaining"] <= 0:
                state["paused"] = False
                state["pause_remaining"] = 0
                logging.info("[%s] Resuming after pause at bar %d", STRATEGY_ID, idx)

        if state["paused"]:
            blocked = True
            reason = "paused"
        else:
            # --- desired position from selection filters ---
            desired = _position_allowed(pred, vol, all_pred, all_vol, cfg)

            # --- kill switch / DD kill check (at position change) ---
            if desired != state["current_position"]:
                if _check_kill_switch(state, cfg):
                    state["paused"] = True
                    state["pause_remaining"] = cfg["pause_bars"]
                    blocked = True
                    reason = "kill_switch"
                    desired = 0
                elif _check_dd_kill(state, cfg):
                    state["paused"] = True
                    state["pause_remaining"] = cfg["pause_bars"]
                    blocked = True
                    reason = "dd_kill"
                    desired = 0

        if not blocked:
            prev = state["current_position"]
            if desired == 1 and prev != 1:
                signal = "BUY"
                action = "OPEN_LONG" if prev == 0 else "REVERSE_LONG"
                state["n_trades"] += 1
                if prev != 0:
                    _record_trade_close(state)
                state["trade_start_equity"] = state["current_equity"]
                state["current_position"] = 1
            elif desired == -1 and prev != -1:
                signal = "SELL"
                action = "OPEN_SHORT" if prev == 0 else "REVERSE_SHORT"
                state["n_trades"] += 1
                if prev != 0:
                    _record_trade_close(state)
                state["trade_start_equity"] = state["current_equity"]
                state["current_position"] = -1
            elif desired == 0 and prev != 0:
                signal = "FLAT"
                action = "CLOSE"
                _record_trade_close(state)
                state["current_position"] = 0
            else:
                signal = "BUY" if desired == 1 else ("SELL" if desired == -1 else "FLAT")
                action = "NONE"

        # --- update equity ---
        if state["current_position"] != 0:
            state["current_equity"] *= 1 + state["current_position"] * target_ret
        state["peak_equity"] = max(state["peak_equity"], state["current_equity"])

        output.append({
            "strategy_id": STRATEGY_ID,
            "timestamp": timestamp,
            "signal": signal,
            "action": action,
            "pred": round(pred, 8),
            "vol_6": round(vol, 8),
            "confidence": round(abs(pred), 8),
            "blocked": 1 if blocked else 0,
            "reason": reason,
            "P_buy": round(pred, 8) if pred > 0 else 0.0,
            "P_sell": round(abs(pred), 8) if pred < 0 else 0.0,
        })

    _save_cursor(cursor + n_bars)
    _save_state(state)
    return output


def _record_trade_close(state: dict) -> None:
    """Record the return of the just-closed trade."""
    trade_ret = state["current_equity"] / state["trade_start_equity"] - 1
    state["trade_rets"].append(float(trade_ret))
    # Keep only last 100 to avoid unbounded growth
    state["trade_rets"] = state["trade_rets"][-100:]


def reset_state() -> None:
    """Reset cursor and kill-switch state (call to restart paper trading from scratch)."""
    _save_cursor(0)
    _save_state(_default_state())
    logging.info("[%s] State and cursor reset", STRATEGY_ID)
