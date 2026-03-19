"""
Regression strategy live signal generator.

Default (REGRESSION_LIVE_USE_FEATURE_TAIL=True): last bar(s) from
data/features_regression_core/test.parquet — model prediction + production filters.
Timestamps match your feature pipeline (e.g. March 2026).

Optional replay: set REGRESSION_LIVE_USE_FEATURE_TAIL=False to walk
test_predictions.parquet with a cursor (historical replay / comparison).
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src import strategy_guards as guards

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STRATEGY_ID = "regression_v1"
STATE_FILE = PROJECT_ROOT / "data" / "logs" / "execution" / "regression_v1_state.json"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions_regression" / "test_predictions.parquet"
CURSOR_FILE = PROJECT_ROOT / "data" / "logs" / "execution" / "regression_v1_cursor.json"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
FEATURES_TEST_PARQUET = PROJECT_ROOT / "data" / "features_regression_core" / "test.parquet"
FEATURES_TEST_CSV = PROJECT_ROOT / "data" / "features_regression_core" / "test.csv"
FEATURES_LIVE_TAIL_PARQUET = PROJECT_ROOT / "data" / "features_regression_core" / "test_live_tail.parquet"
MACRO_EVENTS_PATH = PROJECT_ROOT / "data" / "raw" / "macro" / "event_calendar.csv"


def _load_config() -> dict:
    """Load production config constants."""
    from src.config import (
        MACRO_EVENT_BLACKOUT_MIN,
        REGRESSION_DD_KILL,
        REGRESSION_KILL_SWITCH_N,
        REGRESSION_KILL_SWITCH_PF,
        REGRESSION_MAX_BAR_AGE_MINUTES,
        REGRESSION_PAUSE_BARS,
        REGRESSION_PRED_THRESHOLD,
        REGRESSION_TOP_PCT,
        REGRESSION_VOL_PCT,
        SESSION_EXCLUDE_HOURS,
    )
    return {
        "top_pct": REGRESSION_TOP_PCT,
        "vol_pct": REGRESSION_VOL_PCT,
        "pred_threshold": REGRESSION_PRED_THRESHOLD,
        "kill_switch_n": REGRESSION_KILL_SWITCH_N,
        "kill_switch_pf": REGRESSION_KILL_SWITCH_PF,
        "dd_kill": REGRESSION_DD_KILL,
        "pause_bars": REGRESSION_PAUSE_BARS,
        "max_bar_age_minutes": REGRESSION_MAX_BAR_AGE_MINUTES,
        "macro_blackout_min": MACRO_EVENT_BLACKOUT_MIN,
        "session_exclude_hours": SESSION_EXCLUDE_HOURS,
    }


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


def _regression_proximity(
    pred: float,
    vol: float,
    cfg: dict,
    all_pred: np.ndarray,
    all_vol: np.ndarray,
    selection_desired: int,
    risk_blocked: bool,
    risk_reason: str,
) -> tuple[int, str]:
    """
    readiness_0_100: rough how close to a new trade (selection + vol + pred size).
    Caller bumps to 100 when action != NONE. trade_hint explains gaps.
    """
    n = len(all_pred)
    pred_rank_pct = float((all_pred < pred).sum() / max(n, 1) * 100.0)
    vol_rank_pct = float((all_vol < vol).sum() / max(n, 1) * 100.0)
    need_vol_rank = 100.0 - float(cfg["vol_pct"])
    th_long = float(np.percentile(all_pred, 100 - cfg["top_pct"]))
    th_short = float(np.percentile(all_pred, cfg["top_pct"]))
    th_vol = float(np.percentile(all_vol, 100 - cfg["vol_pct"]))
    pt = float(cfg["pred_threshold"])

    parts: list[str] = []
    readiness = 35

    if risk_blocked and risk_reason:
        parts.append(f"Risk: {risk_reason}")
        readiness = 12 if risk_reason == "paused" else 28

    if vol < th_vol:
        parts.append(f"Vol low vs gate (bar ≈{vol_rank_pct:.0f}th pct; need ~top {cfg['vol_pct']}% ≈{need_vol_rank:.0f}+)")
        readiness = min(readiness, 42)
    else:
        parts.append(f"Vol gate ok (≈{vol_rank_pct:.0f}th pctile)")
        readiness = max(readiness, 52)

    if abs(pred) <= pt:
        parts.append(f"|pred| too small for trade ({abs(pred):.6f} ≤ {pt})")
        readiness = min(readiness, 48)
    else:
        parts.append(f"|pred| ok ({abs(pred):.6f} > {pt})")
        readiness = max(readiness, 55)

    if selection_desired == 1:
        parts.append(f"Would go LONG (pred rank {pred_rank_pct:.1f}% vs ~{100 - cfg['top_pct']:.2f}%+ for long)")
        readiness = max(readiness, 88)
    elif selection_desired == -1:
        parts.append(f"Would go SHORT (pred rank {pred_rank_pct:.1f}% vs ~{cfg['top_pct']:.2f}%- for short)")
        readiness = max(readiness, 88)
    else:
        if vol >= th_vol and abs(pred) > pt:
            parts.append(f"Extreme not reached (pred between short {th_short:.6f} … long {th_long:.6f})")
        readiness = min(readiness, 72)

    hint = " · ".join(parts)[:280]
    return max(0, min(99, int(readiness))), hint


def _check_kill_switch(state: dict, cfg: dict) -> bool:
    blocked, _ = guards.check_kill_switch(
        state["trade_rets"], cfg["kill_switch_n"], cfg["kill_switch_pf"]
    )
    return blocked


def _check_dd_kill(state: dict, cfg: dict) -> bool:
    blocked, _ = guards.check_drawdown_kill(
        state["peak_equity"], state["current_equity"], cfg["dd_kill"]
    )
    return blocked


def _load_percentile_reference() -> tuple[np.ndarray, np.ndarray] | None:
    """Pred / vol distributions from offline test predictions (same as backtests)."""
    if not PREDICTIONS_PATH.exists():
        return None
    ref = pd.read_parquet(PREDICTIONS_PATH)
    return (
        ref["pred"].values.astype(float),
        ref["vol_6"].fillna(0).values.astype(float),
    )


def _load_regression_model():
    model_path = MODELS_DIR / "regression_best.pkl"
    cols_path = MODELS_DIR / "regression_feature_cols.json"
    cfg_path = MODELS_DIR / "regression_best_config.json"
    if not model_path.exists() or not cols_path.exists():
        return None, None, None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(cols_path) as f:
        feature_cols = json.load(f)
    target_col = "target_ret_6"
    if cfg_path.exists():
        with open(cfg_path) as f:
            meta = json.load(f)
            target_col = meta.get("target_col", target_col)
    return model, feature_cols, target_col


def _load_features_test_df() -> pd.DataFrame | None:
    """
    Load features for live inference. Prefers test_live_tail.parquet (the most recent bars
    that were beyond the target horizon when features were built) if it exists and is newer
    than test.parquet. Falls back to test.parquet otherwise.
    """
    # Determine which file to use for the tail
    tail_path: Path | None = None
    test_end_ts: pd.Timestamp | None = None

    if FEATURES_TEST_PARQUET.exists():
        df_test = pd.read_parquet(FEATURES_TEST_PARQUET)
        df_test["timestamp"] = pd.to_datetime(df_test["timestamp"], utc=True)
        test_end_ts = df_test["timestamp"].max()

    if FEATURES_LIVE_TAIL_PARQUET.exists():
        df_tail = pd.read_parquet(FEATURES_LIVE_TAIL_PARQUET)
        df_tail["timestamp"] = pd.to_datetime(df_tail["timestamp"], utc=True)
        if not df_tail.empty:
            tail_end_ts = df_tail["timestamp"].max()
            # Use live tail if it extends beyond test.parquet (it should always, by max_h bars)
            if test_end_ts is None or tail_end_ts > test_end_ts:
                tail_path = FEATURES_LIVE_TAIL_PARQUET
                logging.info(
                    "[%s] Using live tail: %s (vs test end %s)",
                    STRATEGY_ID,
                    tail_end_ts.strftime("%Y-%m-%d %H:%M UTC"),
                    test_end_ts.strftime("%Y-%m-%d %H:%M UTC") if test_end_ts is not None else "N/A",
                )

    if tail_path is not None:
        # Combine test + tail so percentile context and tail lookup both work
        df_test = pd.read_parquet(FEATURES_TEST_PARQUET) if FEATURES_TEST_PARQUET.exists() else pd.DataFrame()
        df_tail = pd.read_parquet(tail_path)
        df = pd.concat([df_test, df_tail], ignore_index=True) if not df_test.empty else df_tail
    elif FEATURES_TEST_PARQUET.exists():
        df = pd.read_parquet(FEATURES_TEST_PARQUET)
    elif FEATURES_TEST_CSV.exists():
        df = pd.read_csv(FEATURES_TEST_CSV)
    else:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return df


def _one_bar_output(
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

    # ── Universal guards (new) ──────────────────────────────────────────────
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
    # ── End universal guards ────────────────────────────────────────────────

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

    if state["current_position"] != 0:
        state["current_equity"] *= 1 + state["current_position"] * target_ret
    state["peak_equity"] = max(state["peak_equity"], state["current_equity"])

    ts_str = bar_ts.strftime("%Y-%m-%d %H:%M:%S UTC")
    lag_h = (pd.Timestamp.now(tz="UTC") - bar_ts).total_seconds() / 3600.0
    rdy, hint = _regression_proximity(
        pred,
        vol,
        cfg,
        all_pred,
        all_vol,
        selection_desired,
        blocked,
        reason,
    )
    if action != "NONE":
        rdy = 100
    return {
        "strategy_id": STRATEGY_ID,
        "timestamp": ts_str,
        "signal": signal,
        "action": action,
        "pred": round(pred, 8),
        "vol_6": round(vol, 8),
        "confidence": round(abs(pred), 8),
        "blocked": 1 if blocked else 0,
        "reason": reason,
        "P_buy": round(pred, 8) if pred > 0 else 0.0,
        "P_sell": round(abs(pred), 8) if pred < 0 else 0.0,
        "bar_return": float(ret_1),
        "signal_source": signal_source,
        "bar_lag_hours": round(lag_h, 2),
        "readiness_0_100": rdy,
        "trade_hint": hint,
    }


def _record_trade_close(state: dict) -> None:
    trade_ret = state["current_equity"] / state["trade_start_equity"] - 1
    state["trade_rets"].append(float(trade_ret))
    state["trade_rets"] = state["trade_rets"][-100:]


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
        out = _one_bar_output(
            bar_ts=bar_ts,
            pred=pred,
            vol=vol,
            target_ret=target_ret,
            ret_1=ret_1,
            cfg=cfg,
            state=state,
            all_pred=all_pred,
            all_vol=all_vol,
            idx_label=f"live tail row {i}",
            signal_source="regression_features_tail",
        )
        output.append(out)

    _save_state(state)
    return output


def _run_replay(n_bars: int) -> list[dict]:
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

        out = _one_bar_output(
            bar_ts=bar_ts,
            pred=pred,
            vol=vol,
            target_ret=target_ret,
            ret_1=ret_1,
            cfg=cfg,
            state=state,
            all_pred=all_pred,
            all_vol=all_vol,
            idx_label=f"replay idx {idx}",
            signal_source="replay_predictions",
        )
        output.append(out)

    _save_cursor(cursor + len(output))
    _save_state(state)
    return output


def reset_state() -> None:
    _save_cursor(0)
    _save_state(_default_state())
    logging.info("[%s] State and cursor reset", STRATEGY_ID)
