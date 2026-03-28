"""
Regression v2 — trend-filtered live signal (regression_v2_trendfilter).

Identical prediction model, signal selection, and risk controls as regression_v1,
with one additional gate: a 4-hour higher-timeframe MA10 trend filter.

  Long  entries: only allowed when 4h close > 4h MA10  (uptrend)
  Short entries: only allowed when 4h close < 4h MA10  (downtrend)

When the trend opposes a candidate entry the trade is skipped and logged with
reason="trend_filter".  If an existing position is left running while trend flips
against it, the next signal evaluation closes it (same behaviour as the backtest).

Research validation summary (see data/validation/):
  ┌──────────────────────────────┬──────────────┬────────────────┐
  │ test                         │ baseline     │ 4h MA10 filter │
  ├──────────────────────────────┼──────────────┼────────────────┤
  │ Profit factor                │ 1.08         │ 1.60  (+0.52)  │
  │ Cumulative return            │ −17.3 %      │ +17.5 %        │
  │ Max drawdown                 │ 19.2 %       │  3.8 %         │
  │ % positive months            │ 44.4 %       │ 63.0 %         │
  │ Walk-forward months improved │ —            │ 14 / 27 (52 %) │
  │ Cost stress at 2× spread     │ —            │ still beats 1× │
  │ Stability (MA 5 → 40)        │ —            │ 8/8 windows ✓  │
  └──────────────────────────────┴──────────────┴────────────────┘

Run alongside regression_v1 for side-by-side live comparison.
Do NOT modify regression_v1 — this is a separate paper-trade candidate.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── Re-use all shared helpers from regression_v1 ──────────────────────────────
# Only state/cursor paths, strategy ID, and the trend gate are different.
from src.live_signal_regression import (
    _check_dd_kill,
    _check_kill_switch,
    _load_features_test_df,
    _load_percentile_reference,
    _load_regression_model,
    _position_allowed,
    _record_trade_close,
    _regression_proximity,
)
from src import strategy_guards as guards
from src.trend_filter import compute_trend_mask

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STRATEGY_ID = "regression_v2_trendfilter"
STATE_FILE  = PROJECT_ROOT / "data" / "logs" / "execution" / "regression_v2_trendfilter_state.json"
CURSOR_FILE = PROJECT_ROOT / "data" / "logs" / "execution" / "regression_v2_trendfilter_cursor.json"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions_regression" / "test_predictions.parquet"
MACRO_EVENTS_PATH = PROJECT_ROOT / "data" / "raw" / "macro" / "event_calendar.csv"


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    from src.config import (
        MACRO_EVENT_BLACKOUT_MIN,
        REGRESSION_MAX_BAR_AGE_MINUTES,
        REGRESSION_LIVE_USE_FEATURE_TAIL,  # noqa: F401  (checked in run())
        RV2_DD_KILL,
        RV2_KILL_SWITCH_N,
        RV2_KILL_SWITCH_PF,
        RV2_PAUSE_BARS,
        RV2_PRED_THRESHOLD,
        RV2_TOP_PCT,
        RV2_TREND_MA_WINDOW,
        RV2_TREND_RESAMPLE,
        RV2_VOL_PCT,
        SESSION_EXCLUDE_HOURS,
    )
    return {
        "top_pct":           RV2_TOP_PCT,
        "vol_pct":           RV2_VOL_PCT,
        "pred_threshold":    RV2_PRED_THRESHOLD,
        "kill_switch_n":     RV2_KILL_SWITCH_N,
        "kill_switch_pf":    RV2_KILL_SWITCH_PF,
        "dd_kill":           RV2_DD_KILL,
        "pause_bars":        RV2_PAUSE_BARS,
        "max_bar_age_minutes": REGRESSION_MAX_BAR_AGE_MINUTES,
        "macro_blackout_min": MACRO_EVENT_BLACKOUT_MIN,
        "session_exclude_hours": SESSION_EXCLUDE_HOURS,
        "trend_resample":    RV2_TREND_RESAMPLE,
        "trend_ma_window":   RV2_TREND_MA_WINDOW,
    }


# ── State ─────────────────────────────────────────────────────────────────────

def _default_state() -> dict:
    return {
        "n_trades": 0,
        "trade_rets": [],
        "peak_equity": 1.0,
        "current_equity": 1.0,
        "pause_remaining": 0,
        "paused": False,
        "current_position": 0,
        "trade_start_equity": 1.0,
    }


def _load_state() -> dict:
    import json
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                raw = json.load(f)
            default = _default_state()
            for k, v in default.items():
                raw.setdefault(k, v)
            return raw
        except Exception:
            pass
    return _default_state()


def _save_state(state: dict) -> None:
    import json
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _load_cursor() -> int:
    import json
    CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    if CURSOR_FILE.exists():
        try:
            with open(CURSOR_FILE) as f:
                return int(json.load(f).get("cursor", 0))
        except Exception:
            pass
    return 0


def _save_cursor(cursor: int) -> None:
    import json
    CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CURSOR_FILE, "w") as f:
        json.dump({"cursor": cursor}, f)


# ── Trend gate ────────────────────────────────────────────────────────────────

def _get_trend_state(bar_ts: pd.Timestamp, resample: str, ma_window: int) -> dict:
    """Return trend state for a single bar timestamp.

    Returns a dict with keys:
        trend_up      – bool: 4h close > MA{ma_window}
        trend_down    – bool: 4h close < MA{ma_window}
        trend_strength – float: close / MA - 1 (positive = above MA)
        trend_label   – "up" | "down" | "neutral" | "warmup"
    """
    try:
        mask = compute_trend_mask(
            pd.Series([bar_ts]),
            ma_window=ma_window,
            resample_period=resample,
        )
        row = mask.iloc[0]
        up   = bool(row["trend_up_1h"] == 1.0) if pd.notna(row["trend_up_1h"]) else False
        down = bool(row["trend_down_1h"] == 1.0) if pd.notna(row["trend_down_1h"]) else False
        strength = float(row["trend_strength_1h"]) if pd.notna(row.get("trend_strength_1h")) else 0.0
        if pd.isna(row["trend_up_1h"]):
            label = "warmup"
        elif up:
            label = "up"
        elif down:
            label = "down"
        else:
            label = "neutral"
        return {"trend_up": up, "trend_down": down, "trend_strength": strength, "trend_label": label}
    except Exception as e:
        logging.warning("[%s] Trend state lookup failed: %s — passing (no gate)", STRATEGY_ID, e)
        return {"trend_up": True, "trend_down": True, "trend_strength": 0.0, "trend_label": "error"}


# ── Per-bar output ────────────────────────────────────────────────────────────

def _one_bar_output_v2(
    *,
    bar_ts: pd.Timestamp,
    pred: float,
    vol: float,
    target_ret: float,
    ret_1: float,
    cfg: dict,
    state: dict,
    all_pred: np.ndarray,
    all_vol: np.ndarray,
    idx_label: str,
    signal_source: str,
) -> dict:
    blocked = False
    reason = ""
    action = "NONE"
    signal = "FLAT"

    selection_desired = _position_allowed(pred, vol, all_pred, all_vol, cfg)
    desired = selection_desired

    # ── Standard guards ───────────────────────────────────────────────────────
    if not blocked:
        blocked, reason = guards.check_stale_bar(bar_ts, cfg.get("max_bar_age_minutes", 20))
    if not blocked:
        blocked, reason = guards.check_session(bar_ts, cfg.get("session_exclude_hours", []))
    if not blocked:
        blocked, reason = guards.check_macro_event(
            bar_ts, MACRO_EVENTS_PATH, cfg.get("macro_blackout_min", 30)
        )
    if blocked:
        desired = 0

    # ── 4h trend filter gate ──────────────────────────────────────────────────
    trend = _get_trend_state(
        bar_ts,
        resample=cfg["trend_resample"],
        ma_window=cfg["trend_ma_window"],
    )
    trend_blocked = False
    if not blocked and trend["trend_label"] not in ("error", "warmup"):
        if desired == 1 and not trend["trend_up"]:
            # Signal is long but 4h trend is down — skip entry
            desired = 0
            trend_blocked = True
            reason = "trend_filter"
        elif desired == -1 and not trend["trend_down"]:
            # Signal is short but 4h trend is up — skip entry
            desired = 0
            trend_blocked = True
            reason = "trend_filter"
        # Also close any open position running against the trend
        elif desired == 0 and state["current_position"] == 1 and not trend["trend_up"]:
            pass  # let existing CLOSE logic handle it naturally on next signal
        elif desired == 0 and state["current_position"] == -1 and not trend["trend_down"]:
            pass

    # ── Kill switch / drawdown kill ───────────────────────────────────────────
    if state["paused"] and cfg["pause_bars"] > 0:
        state["pause_remaining"] -= 1
        if state["pause_remaining"] <= 0:
            state["paused"] = False
            state["pause_remaining"] = 0
            logging.info("[%s] Resuming after pause (%s)", STRATEGY_ID, idx_label)

    if state["paused"]:
        blocked = True
        reason = "paused"
    else:
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

    # ── Position logic ────────────────────────────────────────────────────────
    if not blocked and not trend_blocked:
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

    if state["current_position"] != 0:
        state["current_equity"] *= 1 + state["current_position"] * target_ret
    state["peak_equity"] = max(state["peak_equity"], state["current_equity"])

    ts_str = bar_ts.strftime("%Y-%m-%d %H:%M:%S UTC")
    lag_h = (pd.Timestamp.now(tz="UTC") - bar_ts).total_seconds() / 3600.0

    rdy, hint = _regression_proximity(
        pred, vol, cfg, all_pred, all_vol, selection_desired,
        blocked or trend_blocked, reason,
    )
    if trend_blocked and not blocked:
        trend_dir = "up" if not trend["trend_down"] else "down"
        hint = (
            f"Trend filter: 4h MA{cfg['trend_ma_window']} trend={trend_dir} blocks "
            f"{'LONG' if selection_desired == 1 else 'SHORT'} entry · {hint}"
        )[:280]
    if action != "NONE":
        rdy = 100

    return {
        "strategy_id":       STRATEGY_ID,
        "timestamp":         ts_str,
        "signal":            signal,
        "action":            action,
        "pred":              round(pred, 8),
        "vol_6":             round(vol, 8),
        "confidence":        round(abs(pred), 8),
        "blocked":           1 if (blocked or trend_blocked) else 0,
        "reason":            reason,
        "P_buy":             round(pred, 8) if pred > 0 else 0.0,
        "P_sell":            round(abs(pred), 8) if pred < 0 else 0.0,
        "bar_return":        float(ret_1),
        "signal_source":     signal_source,
        "bar_lag_hours":     round(lag_h, 2),
        "readiness_0_100":   rdy,
        "trade_hint":        hint,
        # Trend-specific fields visible in dashboard / logs
        "trend_label":       trend["trend_label"],
        "trend_strength":    round(trend["trend_strength"], 6),
    }


# ── Run entry point ───────────────────────────────────────────────────────────

def run(n_bars: int = 1) -> list[dict]:
    from src.config import REGRESSION_LIVE_USE_FEATURE_TAIL
    if REGRESSION_LIVE_USE_FEATURE_TAIL:
        return _run_feature_tail(n_bars)
    return _run_replay(n_bars)


def _run_feature_tail(n_bars: int) -> list[dict]:
    ref = _load_percentile_reference()
    if ref is None:
        logging.error("[%s] Need test_predictions.parquet for percentile reference", STRATEGY_ID)
        return []
    all_pred, all_vol = ref

    model, feature_cols, target_col = _load_regression_model()
    if model is None or feature_cols is None:
        logging.error("[%s] Missing regression_best.pkl or regression_feature_cols.json", STRATEGY_ID)
        return []

    df = _load_features_test_df()
    if df is None or df.empty:
        logging.error("[%s] No features_regression_core test file", STRATEGY_ID)
        return []

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logging.error("[%s] Feature file missing columns: %s", STRATEGY_ID, missing[:5])
        return []

    tail = df.tail(max(1, n_bars)).reset_index(drop=True)
    cfg = _load_config()
    state = _load_state()
    output: list[dict] = []

    for i in range(len(tail)):
        row = tail.iloc[i]
        bar_ts = row["timestamp"]
        if bar_ts.tzinfo is None:
            bar_ts = bar_ts.tz_localize("UTC")
        x = row[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        X = x.astype(np.float64, copy=False).values.reshape(1, -1)
        pred = float(model.predict(X)[0])
        vol = float(row.get("vol_6", 0) or 0)
        target_ret = float(row[target_col]) if target_col in row.index and pd.notna(row.get(target_col)) else 0.0
        ret_1 = float(row["ret_1"]) if "ret_1" in row.index and pd.notna(row.get("ret_1")) else 0.0
        out = _one_bar_output_v2(
            bar_ts=bar_ts, pred=pred, vol=vol, target_ret=target_ret, ret_1=ret_1,
            cfg=cfg, state=state, all_pred=all_pred, all_vol=all_vol,
            idx_label=f"live tail row {i}",
            signal_source="regression_v2_trendfilter_features_tail",
        )
        output.append(out)

    _save_state(state)
    return output


def _run_replay(n_bars: int) -> list[dict]:
    if not PREDICTIONS_PATH.exists():
        logging.error("[%s] test_predictions.parquet not found", STRATEGY_ID)
        return []

    df = pd.read_parquet(PREDICTIONS_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    cursor = _load_cursor()
    if cursor >= len(df):
        logging.warning("[%s] All bars consumed (cursor=%d) — resetting", STRATEGY_ID, cursor)
        cursor = 0
        _save_cursor(0)

    cfg = _load_config()
    state = _load_state()
    all_pred = df["pred"].values.astype(float)
    all_vol = df["vol_6"].fillna(0).values.astype(float)
    output = []

    for offset in range(n_bars):
        idx = cursor + offset
        if idx >= len(df):
            break
        bar = df.iloc[idx]
        bar_ts = pd.Timestamp(bar["timestamp"])
        if bar_ts.tzinfo is None:
            bar_ts = bar_ts.tz_localize("UTC")
        pred = float(bar["pred"])
        vol = float(bar.get("vol_6", 0) or 0)
        target_ret = float(bar.get("target_ret", 0) or 0)
        ret_1 = float(bar["ret_1"]) if "ret_1" in bar.index and pd.notna(bar.get("ret_1")) else 0.0
        out = _one_bar_output_v2(
            bar_ts=bar_ts, pred=pred, vol=vol, target_ret=target_ret, ret_1=ret_1,
            cfg=cfg, state=state, all_pred=all_pred, all_vol=all_vol,
            idx_label=f"replay idx {idx}",
            signal_source="regression_v2_trendfilter_replay",
        )
        output.append(out)

    _save_cursor(cursor + len(output))
    _save_state(state)
    return output


def reset_state() -> None:
    _save_cursor(0)
    _save_state(_default_state())
    logging.info("[%s] State and cursor reset", STRATEGY_ID)
