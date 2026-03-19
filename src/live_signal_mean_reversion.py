"""
Mean reversion strategy v1 — rule-based, no ML.

Edge: when price deviates strongly from its short-term mean (measured as the
z-score of ma_20_gap over the last MR_LOOKBACK_BARS bars), bet on reversion.

Signal logic:
    z = zscore of ma_20_gap (last MR_LOOKBACK_BARS bars)
    if z > MR_ZSCORE_THRESHOLD  → SELL (price far above MA, expect down)
    if z < -MR_ZSCORE_THRESHOLD → BUY  (price far below MA, expect up)
    else                         → FLAT

Position management:
    - Hold for exactly MR_HOLD_BARS bars, then force CLOSE regardless of signal.
    - Only one position at a time (no pyramiding).
    - Blocked during macro event windows (is_event_window == 1).
    - Blocked outside London and NY sessions.

Safety guards (all via strategy_guards):
    - Stale bar: blocked if bar > MR_MAX_BAR_AGE_MINUTES old.
    - Session: blocked outside London/NY (is_london_session / is_ny_session columns).
    - Macro event: blocked if is_event_window == 1.
    - Vol regime: blocked when vol_6 is below top MR_VOL_PCT percentile (0 = disabled).
    - Daily loss cap: blocked if intraday equity < day_start * (1 - MAX_DAILY_LOSS_PCT).
    - Kill switch: paused when rolling PF of last MR_KILL_SWITCH_N trades < MR_KILL_SWITCH_PF.
    - Drawdown kill: paused when drawdown from equity peak >= MR_DD_KILL.

Data: data/features_regression_core/test.parquet — uses ma_20_gap, ret_1,
      is_event_window, is_london_session, is_ny_session, vol_6 (all pre-computed).
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import strategy_guards as guards

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STRATEGY_ID = "mean_reversion_v1"
STATE_FILE = PROJECT_ROOT / "data" / "logs" / "execution" / "mean_reversion_v1_state.json"
FEATURES_DIR = PROJECT_ROOT / "data" / "features_regression_core"
FEATURES_PARQUET = FEATURES_DIR / "test.parquet"
FEATURES_CSV = FEATURES_DIR / "test.csv"
FEATURES_LIVE_TAIL_PARQUET = FEATURES_DIR / "test_live_tail.parquet"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions_regression" / "test_predictions.parquet"
MACRO_EVENTS_PATH = PROJECT_ROOT / "data" / "raw" / "macro" / "event_calendar.csv"


def _load_config() -> dict:
    from src.config import (
        MACRO_EVENT_BLACKOUT_MIN,
        MAX_DAILY_LOSS_PCT,
        MR_DD_KILL,
        MR_HOLD_BARS,
        MR_KILL_SWITCH_N,
        MR_KILL_SWITCH_PF,
        MR_LOOKBACK_BARS,
        MR_MAX_BAR_AGE_MINUTES,
        MR_PAUSE_BARS,
        MR_VOL_PCT,
        MR_ZSCORE_THRESHOLD,
    )

    return {
        "zscore_threshold": MR_ZSCORE_THRESHOLD,
        "hold_bars": MR_HOLD_BARS,
        "lookback": MR_LOOKBACK_BARS,
        "max_bar_age_minutes": MR_MAX_BAR_AGE_MINUTES,
        "vol_pct": MR_VOL_PCT,
        "max_daily_loss_pct": MAX_DAILY_LOSS_PCT,
        "kill_switch_n": MR_KILL_SWITCH_N,
        "kill_switch_pf": MR_KILL_SWITCH_PF,
        "dd_kill": MR_DD_KILL,
        "pause_bars": MR_PAUSE_BARS,
        "macro_blackout_min": MACRO_EVENT_BLACKOUT_MIN,
    }


def _default_state() -> dict:
    return {
        "position": 0,           # 0 = flat, 1 = long, -1 = short
        "bars_held": 0,          # bars since entry
        "n_trades": 0,
        "trade_rets": [],        # last 100 completed trade returns
        "peak_equity": 1.0,
        "current_equity": 1.0,
        "trade_start_equity": 1.0,
        "day_start_equity": 1.0,
        "current_day": -1,       # int64 day marker (pd.Timestamp.normalize)
        "pause_remaining": 0,
        "paused": False,
    }


def _load_state() -> dict:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                raw = json.load(f)
            # Migrate: fill any keys added since this state was last saved
            default = _default_state()
            for k, v in default.items():
                raw.setdefault(k, v)
            return raw
        except Exception:
            pass
    return _default_state()


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _load_features() -> "pd.DataFrame | None":
    """
    Load features for live inference. Prefers test + test_live_tail.parquet when the
    live tail has bars newer than test.parquet (same as regression_v1), so the
    bar timestamp reflects the actual latest bar, not the last data-refresh cutoff.
    """
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
                logging.info(
                    "[%s] Using live tail (latest bar %s UTC)",
                    STRATEGY_ID,
                    tail_end_ts.strftime("%Y-%m-%d %H:%M"),
                )

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


def _load_vol_reference() -> "np.ndarray | None":
    """Load vol_6 reference distribution from test_predictions.parquet for percentile guard."""
    if not PREDICTIONS_PATH.exists():
        return None
    ref = pd.read_parquet(PREDICTIONS_PATH)
    if "vol_6" not in ref.columns:
        return None
    return ref["vol_6"].fillna(0).values.astype(float)


def _zscore_of_last(series: "pd.Series", lookback: int) -> float:
    """z-score of the last value relative to the prior `lookback` values."""
    tail = series.tail(lookback)
    if len(tail) < 5:
        return 0.0
    std = float(tail.std())
    if std < 1e-12:
        return 0.0
    return float((tail.iloc[-1] - tail.mean()) / std)


def _proximity_hint(zscore: float, cfg: dict, blocked: bool, block_reason: str) -> tuple[int, str]:
    """readiness_0_100 and human-readable trade hint."""
    th = cfg["zscore_threshold"]
    parts: list[str] = []

    if blocked:
        parts.append(f"Blocked: {block_reason}")
        readiness = 15
    else:
        readiness = 40

    dist = abs(abs(zscore) - th)
    pct_to_threshold = max(0.0, min(1.0, abs(zscore) / max(th, 1e-9)))
    readiness = max(readiness, int(30 + 55 * pct_to_threshold))

    if abs(zscore) >= th:
        direction = "SHORT" if zscore > 0 else "LONG"
        parts.append(f"z={zscore:.2f} ≥ threshold {th} → {direction} entry")
        readiness = 100
    else:
        parts.append(
            f"z={zscore:.2f}, need |z| ≥ {th} ({dist:.2f} to threshold)"
        )
        readiness = min(readiness, 88)

    hint = " · ".join(parts)[:280]
    return max(0, min(99, readiness)), hint


def run(n_bars: int = 1) -> list[dict]:
    cfg = _load_config()

    df = _load_features()
    if df is None or df.empty:
        logging.error("[%s] No features_regression_core test file", STRATEGY_ID)
        return []

    needed = ["ma_20_gap", "ret_1"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        logging.error("[%s] Feature file missing columns: %s", STRATEGY_ID, missing)
        return []

    # Need enough rows for z-score lookback + the bars we want to emit
    lookback = cfg["lookback"]
    total_needed = lookback + n_bars
    if len(df) < total_needed:
        logging.warning(
            "[%s] Not enough rows (%d) for lookback %d + n_bars %d",
            STRATEGY_ID, len(df), lookback, n_bars,
        )

    # Load vol reference for percentile guard (None = guard disabled)
    vol_ref = _load_vol_reference()

    state = _load_state()
    output: list[dict] = []

    for i in range(n_bars):
        # Index into df: work from the tail
        row_idx = len(df) - n_bars + i
        if row_idx < lookback:
            continue

        row = df.iloc[row_idx]
        bar_ts = row["timestamp"]
        if bar_ts.tzinfo is None:
            bar_ts = bar_ts.tz_localize("UTC")

        # Compute z-score using the preceding `lookback` bars (inclusive of this bar)
        slice_end = row_idx + 1
        slice_start = max(0, slice_end - lookback)
        zscore = _zscore_of_last(df["ma_20_gap"].iloc[slice_start:slice_end], lookback)

        ret_1 = float(row["ret_1"]) if pd.notna(row.get("ret_1")) else 0.0
        vol = float(row.get("vol_6", 0) or 0)

        # ── Guard chain (universal checks first) ───────────────────────────
        blocked = False
        block_reason = ""

        if not blocked:
            blocked, block_reason = guards.check_stale_bar(bar_ts, cfg["max_bar_age_minutes"])

        # Session: use pre-computed feature columns (London + NY)
        if not blocked:
            in_session = bool(
                row.get("is_london_session", 1) or row.get("is_ny_session", 1)
            )
            if not in_session:
                blocked = True
                block_reason = "outside_session"

        # Macro event: use pre-computed column, fall back to calendar file
        if not blocked:
            in_event_window = bool(row.get("is_event_window", 0))
            if in_event_window:
                blocked = True
                block_reason = "macro_event_window"
            else:
                # Calendar fallback in case feature column is absent or stale
                _ev_blocked, _ev_reason = guards.check_macro_event(
                    bar_ts, MACRO_EVENTS_PATH, cfg["macro_blackout_min"]
                )
                if _ev_blocked:
                    blocked = True
                    block_reason = _ev_reason

        # Vol regime
        if not blocked and cfg.get("vol_pct", 0) > 0 and vol_ref is not None:
            blocked, block_reason = guards.check_vol_regime(vol, vol_ref, cfg["vol_pct"])

        # Daily loss cap — reset day bucket when date changes
        bar_day = int(bar_ts.normalize().value)
        if bar_day != state["current_day"]:
            state["current_day"] = bar_day
            state["day_start_equity"] = state["current_equity"]
        if not blocked:
            blocked, block_reason = guards.check_daily_loss(
                state["current_equity"], state["day_start_equity"], cfg["max_daily_loss_pct"]
            )

        # Pause countdown (kill switch / dd kill)
        if state["paused"] and state["pause_remaining"] > 0:
            state["pause_remaining"] -= 1
            if state["pause_remaining"] <= 0:
                state["paused"] = False
                logging.info("[%s] Resuming after pause (bar %s)", STRATEGY_ID, bar_ts)

        if not blocked and state["paused"]:
            blocked = True
            block_reason = "paused"
        # ── End guard chain ────────────────────────────────────────────────

        action = "NONE"
        signal = "FLAT"
        current_pos = int(state.get("position", 0))
        bars_held = int(state.get("bars_held", 0))

        if not blocked:
            # Check forced close first (hold timer expired)
            if current_pos != 0 and bars_held >= cfg["hold_bars"]:
                signal = "FLAT"
                action = "CLOSE"
                _record_trade_close(state)
                state["position"] = 0
                state["bars_held"] = 0
                current_pos = 0
                bars_held = 0
            else:
                # Determine desired position from z-score
                if zscore >= cfg["zscore_threshold"]:
                    desired = -1  # price above mean → bet on reversion down
                elif zscore <= -cfg["zscore_threshold"]:
                    desired = 1   # price below mean → bet on reversion up
                else:
                    desired = 0

                # Kill switch and drawdown kill — only when about to open/change position
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
                    if desired == 1 and current_pos != 1:
                        signal = "BUY"
                        action = "OPEN_LONG" if current_pos == 0 else "REVERSE_LONG"
                        if current_pos != 0:
                            _record_trade_close(state)
                        state["trade_start_equity"] = state["current_equity"]
                        state["position"] = 1
                        state["bars_held"] = 1
                    elif desired == -1 and current_pos != -1:
                        signal = "SELL"
                        action = "OPEN_SHORT" if current_pos == 0 else "REVERSE_SHORT"
                        if current_pos != 0:
                            _record_trade_close(state)
                        state["trade_start_equity"] = state["current_equity"]
                        state["position"] = -1
                        state["bars_held"] = 1
                    elif desired == 0 and current_pos != 0:
                        signal = "FLAT"
                        action = "CLOSE"
                        _record_trade_close(state)
                        state["position"] = 0
                        state["bars_held"] = 0
                    else:
                        # Holding or flat with no signal
                        signal = "BUY" if current_pos == 1 else ("SELL" if current_pos == -1 else "FLAT")
                        action = "NONE"
                        if current_pos != 0:
                            state["bars_held"] = bars_held + 1

        if blocked:
            # Blocked — still age out the hold timer so we close on next unblocked bar
            signal = "BUY" if current_pos == 1 else ("SELL" if current_pos == -1 else "FLAT")
            action = "NONE"
            if current_pos != 0:
                state["bars_held"] = bars_held + 1

        # Update equity simulation
        current_pos_after = int(state.get("position", 0))
        if current_pos_after != 0:
            state["current_equity"] = state.get("current_equity", 1.0) * (1 + current_pos_after * ret_1)
        state["peak_equity"] = max(state.get("peak_equity", 1.0), state.get("current_equity", 1.0))

        lag_h = (pd.Timestamp.now(tz="UTC") - bar_ts).total_seconds() / 3600.0
        rdy, hint = _proximity_hint(zscore, cfg, blocked, block_reason)
        if action != "NONE":
            rdy = 100

        output.append({
            "strategy_id": STRATEGY_ID,
            "timestamp": bar_ts.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "signal": signal,
            "action": action,
            "pred": round(zscore, 6),          # z-score as the "prediction" equivalent
            "vol_6": round(vol, 8),
            "confidence": round(abs(zscore), 6),
            "blocked": 1 if blocked else 0,
            "reason": block_reason,
            "P_buy": round(max(-zscore, 0.0), 6),   # higher when z is negative (BUY)
            "P_sell": round(max(zscore, 0.0), 6),    # higher when z is positive (SELL)
            "bar_return": float(ret_1),
            "signal_source": "mean_reversion_features_tail",
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
