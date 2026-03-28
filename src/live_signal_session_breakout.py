"""
Session Breakout strategy v1 — rule-based, no ML.

Edge: capture momentum bursts at London and NY session opens.
When price breaks above/below the rolling N-bar high/low, follow the breakout.
Completely independent data source (raw OHLC) — no shared state with other strategies.

Signal logic:
    range_high = rolling N-bar high of the previous N bars (shift(1) — no leakage)
    range_low  = rolling N-bar low  of the previous N bars
    BUY  if close > range_high and in_session and range large enough
    SELL if close < range_low  and in_session and range large enough
    else FLAT

Position management:
    - Hold for exactly SB_HOLD_BARS bars, then force CLOSE.
    - Only one position at a time (no pyramiding).
    - Only one trade per session (london or ny) per calendar day.
    - Blocked outside London (07–10 UTC) and NY (12–15 UTC) sessions.
    - Blocked if range_size < SB_MIN_RANGE_SIZE (avoids flat/choppy bars).

Safety guards:
    - Stale bar: blocked if bar > SB_MAX_BAR_AGE_MINUTES old.
    - Session: blocked outside London/NY windows.
    - Macro event: blocked if near high-importance event.
    - Vol regime: disabled by default (SB_VOL_PCT = 0).
    - Daily loss cap: blocked if intraday equity < day_start * (1 - MAX_DAILY_LOSS_PCT).
    - Kill switch: paused when rolling PF of last SB_KILL_SWITCH_N trades < SB_KILL_SWITCH_PF.
    - Drawdown kill: paused when drawdown from equity peak >= SB_DD_KILL.

Data: data/processed/price/EURUSD_5min_clean.parquet — uses raw OHLC bars.
      Completely independent from features_regression_core used by other strategies.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import strategy_guards as guards

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STRATEGY_ID = "session_breakout_v1"
STATE_FILE = PROJECT_ROOT / "data" / "logs" / "execution" / "session_breakout_v1_state.json"
PRICE_PARQUET = PROJECT_ROOT / "data" / "processed" / "price" / "EURUSD_5min_clean.parquet"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions_regression" / "test_predictions.parquet"
MACRO_EVENTS_PATH = PROJECT_ROOT / "data" / "raw" / "macro" / "event_calendar.csv"


def _load_config() -> dict:
    from src.config import (
        MACRO_EVENT_BLACKOUT_MIN,
        MAX_DAILY_LOSS_PCT,
        SB_DD_KILL,
        SB_HOLD_BARS,
        SB_KILL_SWITCH_N,
        SB_KILL_SWITCH_PF,
        SB_MAX_BAR_AGE_MINUTES,
        SB_MIN_RANGE_SIZE,
        SB_N_LOOKBACK,
        SB_PAUSE_BARS,
        SB_VOL_PCT,
    )
    return {
        "n_lookback": SB_N_LOOKBACK,
        "hold_bars": SB_HOLD_BARS,
        "min_range_size": SB_MIN_RANGE_SIZE,
        "max_bar_age_minutes": SB_MAX_BAR_AGE_MINUTES,
        "vol_pct": SB_VOL_PCT,
        "max_daily_loss_pct": MAX_DAILY_LOSS_PCT,
        "kill_switch_n": SB_KILL_SWITCH_N,
        "kill_switch_pf": SB_KILL_SWITCH_PF,
        "dd_kill": SB_DD_KILL,
        "pause_bars": SB_PAUSE_BARS,
        "macro_blackout_min": MACRO_EVENT_BLACKOUT_MIN,
    }


def _default_state() -> dict:
    return {
        "position": 0,            # 0 = flat, 1 = long, -1 = short
        "bars_held": 0,
        "last_session_id": "",    # e.g. "2026-03-28_london" — one trade per session
        "n_trades": 0,
        "trade_rets": [],
        "peak_equity": 1.0,
        "current_equity": 1.0,
        "trade_start_equity": 1.0,
        "day_start_equity": 1.0,
        "current_day": -1,
        "pause_remaining": 0,
        "paused": False,
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


def _load_price() -> "pd.DataFrame | None":
    if not PRICE_PARQUET.exists():
        return None
    df = pd.read_parquet(PRICE_PARQUET)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return df


def _load_vol_reference() -> "np.ndarray | None":
    """Reference vol distribution for the percentile vol_regime guard."""
    if not PREDICTIONS_PATH.exists():
        return None
    ref = pd.read_parquet(PREDICTIONS_PATH)
    if "vol_6" not in ref.columns:
        return None
    return ref["vol_6"].fillna(0).values.astype(float)


def _session_id(bar_ts: pd.Timestamp) -> str:
    """Return a unique session identifier, e.g. '2026-03-28_london' or '2026-03-28_ny'."""
    h = bar_ts.hour
    if 7 <= h < 10:
        name = "london"
    elif 12 <= h < 15:
        name = "ny"
    else:
        name = "other"
    return f"{bar_ts.date()}_{name}"


def _proximity_hint(
    close: float,
    range_high: float,
    range_low: float,
    range_size: float,
    cfg: dict,
    blocked: bool,
    block_reason: str,
    in_session: bool,
    current_session_id: str,
    last_session_id: str,
) -> tuple[int, str]:
    parts: list[str] = []

    if blocked:
        parts.append(f"Blocked: {block_reason}")
        readiness = 15
    elif not in_session:
        parts.append("Outside session window")
        readiness = 10
    elif current_session_id == last_session_id:
        parts.append(f"Session {current_session_id} already traded")
        readiness = 20
    elif range_size < cfg["min_range_size"]:
        parts.append(f"Range {range_size:.5f} < min {cfg['min_range_size']:.5f}")
        readiness = 30
    else:
        dist_up = close - range_high
        dist_dn = range_low - close
        if dist_up > 0:
            parts.append(f"BUY breakout active: close {close:.5f} > range_high {range_high:.5f} (+{dist_up:.5f})")
            readiness = 100
        elif dist_dn > 0:
            parts.append(f"SELL breakout active: close {close:.5f} < range_low {range_low:.5f} (-{dist_dn:.5f})")
            readiness = 100
        else:
            pct_up = max(0.0, min(1.0, close / range_high)) if range_high > 0 else 0.0
            pct_dn = max(0.0, min(1.0, range_low / close)) if close > 0 else 0.0
            pct = max(pct_up, pct_dn)
            readiness = int(40 + 50 * pct)
            parts.append(
                f"In session · range {range_size:.5f} · "
                f"watching {range_high:.5f}/{range_low:.5f}"
            )

    hint = " · ".join(parts)[:280]
    return max(0, min(99, readiness)), hint


def run(n_bars: int = 1) -> list[dict]:
    cfg = _load_config()

    df = _load_price()
    if df is None or df.empty:
        logging.error("[%s] Price file not found: %s", STRATEGY_ID, PRICE_PARQUET)
        return []

    needed = ["timestamp", "high", "low", "close"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        logging.error("[%s] Price file missing columns: %s", STRATEGY_ID, missing)
        return []

    n = cfg["n_lookback"]
    total_needed = n + n_bars
    if len(df) < total_needed:
        logging.warning(
            "[%s] Not enough rows (%d) for lookback %d + n_bars %d",
            STRATEGY_ID, len(df), n, n_bars,
        )

    # Compute rolling range (shift(1) prevents lookahead leakage)
    df["range_high"] = df["high"].rolling(n).max().shift(1)
    df["range_low"] = df["low"].rolling(n).min().shift(1)
    df["range_size"] = df["range_high"] - df["range_low"]
    df["ret_1"] = df["close"].pct_change()
    df["vol_6"] = df["ret_1"].rolling(6).std()

    vol_ref = _load_vol_reference()
    state = _load_state()
    output: list[dict] = []

    for i in range(n_bars):
        row_idx = len(df) - n_bars + i
        if row_idx < n:
            continue

        row = df.iloc[row_idx]
        bar_ts = row["timestamp"]
        if bar_ts.tzinfo is None:
            bar_ts = bar_ts.tz_localize("UTC")

        bar_hour = bar_ts.hour
        in_session = (7 <= bar_hour < 10) or (12 <= bar_hour < 15)
        current_session_id = _session_id(bar_ts)

        close = float(row["close"])
        range_high = float(row["range_high"]) if pd.notna(row["range_high"]) else float("nan")
        range_low = float(row["range_low"]) if pd.notna(row["range_low"]) else float("nan")
        range_size = float(row["range_size"]) if pd.notna(row["range_size"]) else 0.0
        ret_1 = float(row["ret_1"]) if pd.notna(row.get("ret_1")) else 0.0
        vol = float(row.get("vol_6", 0) or 0)

        # ── Guard chain ────────────────────────────────────────────────────
        blocked = False
        block_reason = ""

        if not blocked:
            blocked, block_reason = guards.check_stale_bar(bar_ts, cfg["max_bar_age_minutes"])

        if not blocked and not in_session:
            blocked = True
            block_reason = "outside_session"

        if not blocked:
            blocked, block_reason = guards.check_macro_event(
                bar_ts, MACRO_EVENTS_PATH, cfg["macro_blackout_min"]
            )

        if not blocked and cfg.get("vol_pct", 0) > 0 and vol_ref is not None:
            blocked, block_reason = guards.check_vol_regime(vol, vol_ref, cfg["vol_pct"])

        bar_day = int(bar_ts.normalize().value)
        if bar_day != state["current_day"]:
            state["current_day"] = bar_day
            state["day_start_equity"] = state["current_equity"]
        if not blocked:
            blocked, block_reason = guards.check_daily_loss(
                state["current_equity"], state["day_start_equity"], cfg["max_daily_loss_pct"]
            )

        if state["paused"] and state["pause_remaining"] > 0:
            state["pause_remaining"] -= 1
            if state["pause_remaining"] <= 0:
                state["paused"] = False
                logging.info("[%s] Resuming after pause (bar %s)", STRATEGY_ID, bar_ts)

        if not blocked and state["paused"]:
            blocked = True
            block_reason = "paused"
        # ── End universal guard chain ──────────────────────────────────────

        action = "NONE"
        signal = "FLAT"
        current_pos = int(state.get("position", 0))
        bars_held = int(state.get("bars_held", 0))

        if not blocked:
            # Force close when hold timer expires
            if current_pos != 0 and bars_held >= cfg["hold_bars"]:
                signal = "FLAT"
                action = "CLOSE"
                _record_trade_close(state)
                state["position"] = 0
                state["bars_held"] = 0
                current_pos = 0
                bars_held = 0

            if current_pos == 0:
                # Determine desired position
                desired = 0
                valid_range = range_size >= cfg["min_range_size"]
                session_fresh = current_session_id != state["last_session_id"]
                range_valid = pd.notna(range_high) and pd.notna(range_low)

                if valid_range and session_fresh and range_valid:
                    if close > range_high:
                        desired = 1
                    elif close < range_low:
                        desired = -1

                # Kill switch and drawdown kill before opening
                if desired != 0:
                    _ks_blocked, _ks_reason = guards.check_kill_switch(
                        state["trade_rets"], cfg["kill_switch_n"], cfg["kill_switch_pf"]
                    )
                    if _ks_blocked:
                        state["paused"] = True
                        state["pause_remaining"] = cfg["pause_bars"]
                        blocked = True
                        block_reason = _ks_reason
                        desired = 0

                if not blocked and desired != 0:
                    _dd_blocked, _dd_reason = guards.check_drawdown_kill(
                        state["peak_equity"], state["current_equity"], cfg["dd_kill"]
                    )
                    if _dd_blocked:
                        state["paused"] = True
                        state["pause_remaining"] = cfg["pause_bars"]
                        blocked = True
                        block_reason = _dd_reason
                        desired = 0

                if not blocked and desired == 1:
                    signal = "BUY"
                    action = "OPEN_LONG"
                    state["trade_start_equity"] = state["current_equity"]
                    state["position"] = 1
                    state["bars_held"] = 1
                    state["last_session_id"] = current_session_id
                elif not blocked and desired == -1:
                    signal = "SELL"
                    action = "OPEN_SHORT"
                    state["trade_start_equity"] = state["current_equity"]
                    state["position"] = -1
                    state["bars_held"] = 1
                    state["last_session_id"] = current_session_id
            else:
                # Holding — age the timer
                signal = "BUY" if current_pos == 1 else "SELL"
                action = "NONE"
                state["bars_held"] = bars_held + 1

        if blocked:
            signal = "BUY" if current_pos == 1 else ("SELL" if current_pos == -1 else "FLAT")
            action = "NONE"
            if current_pos != 0:
                state["bars_held"] = bars_held + 1

        # Equity simulation
        current_pos_after = int(state.get("position", 0))
        if current_pos_after != 0:
            state["current_equity"] = state.get("current_equity", 1.0) * (1 + current_pos_after * ret_1)
        state["peak_equity"] = max(state.get("peak_equity", 1.0), state.get("current_equity", 1.0))

        lag_h = (pd.Timestamp.now(tz="UTC") - bar_ts).total_seconds() / 3600.0
        rdy, hint = _proximity_hint(
            close, range_high, range_low, range_size, cfg,
            blocked, block_reason, in_session, current_session_id,
            state.get("last_session_id", ""),
        )
        if action != "NONE":
            rdy = 100

        breakout_size = 0.0
        if pd.notna(range_high) and pd.notna(range_low):
            if close > range_high:
                breakout_size = round(close - range_high, 6)
            elif close < range_low:
                breakout_size = round(range_low - close, 6)

        output.append({
            "strategy_id": STRATEGY_ID,
            "timestamp": bar_ts.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "signal": signal,
            "action": action,
            "pred": round(breakout_size, 6),
            "vol_6": round(vol, 8),
            "confidence": round(breakout_size / range_size, 6) if range_size > 0 else 0.0,
            "blocked": 1 if blocked else 0,
            "reason": block_reason,
            "P_buy": round(max(breakout_size if close > range_high else 0.0, 0.0), 6) if pd.notna(range_high) else 0.0,
            "P_sell": round(max(breakout_size if close < range_low else 0.0, 0.0), 6) if pd.notna(range_low) else 0.0,
            "bar_return": float(ret_1),
            "signal_source": "session_breakout_price_tail",
            "bar_lag_hours": round(lag_h, 2),
            "readiness_0_100": rdy,
            "trade_hint": hint,
            "bars_held": int(state.get("bars_held", 0)),
        })

    _save_state(state)
    return output


def _record_trade_close(state: dict) -> None:
    trade_ret = state["current_equity"] / state.get("trade_start_equity", state["current_equity"]) - 1
    state["trade_rets"].append(float(trade_ret))
    state["trade_rets"] = state["trade_rets"][-100:]
    state["n_trades"] = state.get("n_trades", 0) + 1


def reset_state() -> None:
    _save_state(_default_state())
    logging.info("[%s] State reset", STRATEGY_ID)
