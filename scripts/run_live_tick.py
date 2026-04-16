"""
Phase 16 — Live tick script (every 5 min).

Runs all registered strategies, appends signals to predictions_live.csv.
By default also runs **paper execution**: tracks per-strategy simulated position
and equity (each bar's `bar_return` × `portfolio_size` while long/short), logs to
trade_decisions.csv and paper_simulation.csv. Use --no-execute for signals only.

**Demo broker**: With --demo-broker (or RUN_LIVETICK_DEMO_BROKER=1), strategies listed
in ``DEMO_BROKER_REAL_ORDER_STRATEGY_IDS`` send real orders (each to its own login via
``DEMO_CTRADER_ACCOUNT_BY_STRATEGY``); others stay simulated (signals + paper equity still update).
``regression_v2_trendfilter`` and ``mean_reversion_v1`` stay paper/sim on demo.
Still EXECUTION_PAPER_ONLY globally.
"""

import csv
import importlib
import inspect
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# So RUN_LIVETICK_DEMO_BROKER (and other vars) work when run by launchd/cron
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
    except ImportError:
        pass

from src.config import (
    LIVETICK_AUTO_REFRESH_COOLDOWN_MINUTES,
    LIVETICK_FEATURE_STALE_MINUTES,
    LIVETICK_HEARTBEAT_JSON,
    LIVETICK_STALE_MINUTES,
    PREDICTIONS_LIVE_CSV,
    TRADE_DECISIONS_CSV,
)
from src.portfolio_engine import (  # noqa: E402
    enrich_signal_desired_position,
    get_target_position,
    is_strategy_allowed,
    load_portfolio_config,
    load_portfolio_state,
    record_position_close,
    record_position_open,
    record_trade_result,
    save_portfolio_state,
)

# Portfolio config loaded once at import time; never mutated at runtime.
_PORTFOLIO_CFG: dict = load_portfolio_config()

# Price path for realized-vol calculation inside get_target_position / compute_size.
_PRICE_PARQUET = PROJECT_ROOT / "data" / "processed" / "price" / "EURUSD_5min_clean.parquet"

# Order matters for portfolio batch 5: if any future event-driven strategy is listed in
# PORTFOLIO_EVENT_STRATEGY_IDS, register it AFTER continuous strategies so the same tick
# can flatten regression before an event strategy sends orders (see portfolio_engine).
STRATEGIES = [
    {"id": "classifier_v1",              "module": "src.live_signal",                           "fn": "run"},
    {"id": "regression_v1",              "module": "src.live_signal_regression",                "fn": "run"},
    {"id": "mean_reversion_v1",          "module": "src.live_signal_mean_reversion",            "fn": "run"},
    {"id": "regression_v2_trendfilter",  "module": "src.live_signal_regression_v2_trendfilter", "fn": "run"},
    {
        "id": "regression_v2_trendfilter_portfolio_vol",
        "module": "src.live_signal_regression_v2_trendfilter",
        "fn": "run",
    },
    # session_breakout_v1 disabled 2026-03: 27/27 parameter combos losing, structural rejection
    # {"id": "session_breakout_v1", "module": "src.live_signal_session_breakout", "fn": "run"},
]

# With --demo-broker, real cTrader orders only for these; others stay simulated (paper books still update).
# Each strategy in this set should have its own account in DEMO_CTRADER_ACCOUNT_BY_STRATEGY.
DEMO_BROKER_REAL_ORDER_STRATEGY_IDS: frozenset[str] = frozenset(
    {
        "classifier_v1",
        "regression_v1",
        "regression_v2_trendfilter_portfolio_vol",
    }
)

# Per-strategy cTrader account mapping (login numbers). Imported directly from portfolio config.
# Strategies absent from this map fall back to PS_CTRADER_ACCOUNT_ID in .env.
try:
    from src.config_portfolio import DEMO_CTRADER_ACCOUNT_BY_STRATEGY as _DEMO_ACCOUNT_BY_STRATEGY
except ImportError:
    _DEMO_ACCOUNT_BY_STRATEGY = {}


def _resolve_demo_account_id(strategy_id: str) -> int | None:
    """Return the cTrader login/account_id for a strategy, or None to use the .env default."""
    v = _DEMO_ACCOUNT_BY_STRATEGY.get(strategy_id)
    return int(v) if v is not None else None

PAPER_STATE_PATH = PROJECT_ROOT / "data" / "logs" / "execution" / "paper_sim_state.json"
PAPER_SIM_CSV = PROJECT_ROOT / "data" / "logs" / "execution" / "paper_simulation.csv"
LIVE_TICK_LOCK = PROJECT_ROOT / "data" / "logs" / "execution" / "run_live_tick.lock"
HEARTBEAT_PATH = PROJECT_ROOT / LIVETICK_HEARTBEAT_JSON
FEATURES_LIVE_TAIL = PROJECT_ROOT / "data" / "features_regression_core" / "test_live_tail.parquet"
FEATURES_TEST = PROJECT_ROOT / "data" / "features_regression_core" / "test.parquet"


def _default_paper_state() -> dict:
    return {s["id"]: {"position": "flat", "equity": 1.0} for s in STRATEGIES}


def _load_recent_closes(n_bars: int = 64) -> "pd.Series | None":
    """Return the last n_bars close prices for portfolio vol calculation.

    Returns None if the price file is unavailable (engine degrades gracefully).
    """
    try:
        import pandas as _pd
        df = _pd.read_parquet(
            _PRICE_PARQUET,
            columns=["timestamp", "close"],
        )
        df["timestamp"] = _pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").tail(n_bars + 1)
        return df["close"].reset_index(drop=True)
    except Exception:
        return None


def _normalize_paper_entry(raw: dict) -> dict:
    pos = raw.get("position", "flat")
    if pos not in ("flat", "long", "short"):
        pos = "flat"
    eq = float(raw.get("equity", 1.0))
    return {"position": pos, "equity": max(eq, 1e-9)}


def _load_paper_state() -> dict:
    """Load paper_sim_state.json: per-strategy books, optional `{id}_paper` parallel sim, `_demo_broker_pos`."""
    PAPER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    base = _default_paper_state()
    if not PAPER_STATE_PATH.exists():
        return base
    try:
        with open(PAPER_STATE_PATH) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return base
        out: dict = dict(base)
        strat_ids = {s["id"] for s in STRATEGIES}
        for k, v in data.items():
            if k == "_demo_broker_pos":
                if v in ("flat", "long", "short"):
                    out[k] = v
                continue
            if k.startswith("_demo_broker_pos_"):
                # Per-account position key — preserve as-is (string value)
                if v in ("flat", "long", "short"):
                    out[k] = v
                continue
            if not isinstance(v, dict):
                continue
            if k in strat_ids or (k.endswith("_paper") and k[:-6] in strat_ids):
                out[k] = _normalize_paper_entry(v)
        return out
    except Exception:
        pass
    return base


def _read_heartbeat() -> dict:
    HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not HEARTBEAT_PATH.exists():
        return {}
    try:
        with open(HEARTBEAT_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _write_heartbeat(**kwargs: str) -> None:
    HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = _read_heartbeat()
    data.update({k: v for k, v in kwargs.items() if v})
    with open(HEARTBEAT_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _want_auto_refresh(allow_auto: bool) -> bool:
    """True if we should run data refresh before this tick (machine was likely off/asleep, or features are stale)."""
    if not allow_auto:
        return False
    now = datetime.now(timezone.utc)
    h = _read_heartbeat()
    last_ok = _parse_iso(h.get("last_full_success_utc"))
    last_ref = _parse_iso(h.get("last_data_refresh_utc"))
    stale_sec = LIVETICK_STALE_MINUTES * 60
    cool_sec = LIVETICK_AUTO_REFRESH_COOLDOWN_MINUTES * 60

    # Check 1: no recent successful tick (machine was off/asleep)
    if last_ok is None:
        return True
    heartbeat_stale = (now - last_ok).total_seconds() > stale_sec

    # Check 2: regression feature tail is too old (intraday staleness)
    feature_stale = _is_feature_tail_stale(now)

    if not heartbeat_stale and not feature_stale:
        return False

    # Respect cooldown to avoid hammering refresh if strategies keep failing
    if last_ref is None:
        return True
    return (now - last_ref).total_seconds() >= cool_sec


def _feature_tail_end_utc() -> datetime | None:
    """Return the timestamp of the last bar in the regression feature files."""
    import pandas as pd

    for path in (FEATURES_LIVE_TAIL, FEATURES_TEST):
        if path.exists():
            try:
                df = pd.read_parquet(path, columns=["timestamp"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                ts = df["timestamp"].max()
                if pd.notna(ts):
                    return ts.to_pydatetime().astimezone(timezone.utc)
            except Exception:
                continue
    return None


def _is_data_refresh_running() -> bool:
    """True if a run_daily_data_refresh process is already running (so we don't spawn a second one)."""
    try:
        r = subprocess.run(
            ["pgrep", "-f", "scripts.run_daily_data_refresh"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            timeout=5,
        )
        return r.returncode == 0
    except Exception:
        return False


def _is_feature_tail_stale(now: datetime) -> bool:
    """True if the most recent regression feature bar is older than LIVETICK_FEATURE_STALE_MINUTES."""
    tail_end = _feature_tail_end_utc()
    if tail_end is None:
        return True
    age_min = (now - tail_end).total_seconds() / 60.0
    if age_min > LIVETICK_FEATURE_STALE_MINUTES:
        logging.info(
            "Regression feature tail is %.0f min old (last bar %s) — triggering data refresh",
            age_min,
            tail_end.strftime("%Y-%m-%d %H:%M UTC"),
        )
        return True
    return False


def _save_paper_state(state: dict) -> None:
    PAPER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PAPER_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _reconcile_demo_broker_refs(refs: dict[int, dict]) -> None:
    """Set each ref['pos'] from live cTrader reconcile (subprocess = fresh reactor per account).

    Prevents repeated OPEN_* when persisted paper_sim_state desyncs from the broker
    (e.g. crash before save, old mapping, or missing ref falling back to sim_pos=flat).
    """
    if not refs:
        return
    py = PROJECT_ROOT / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)
    script = PROJECT_ROOT / "scripts" / "ctrader_net_position.py"
    for acct_id, ref in refs.items():
        try:
            r = subprocess.run(
                [str(py), str(script), str(acct_id)],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ},
            )
            line = (r.stdout or "").strip().splitlines()[-1] if (r.stdout or "").strip() else ""
            if r.returncode != 0 or not line:
                logging.warning(
                    "Broker reconcile acct=%s: rc=%s stderr=%s",
                    acct_id,
                    r.returncode,
                    (r.stderr or "")[:400],
                )
                continue
            j = json.loads(line)
            pos = j.get("pos", "flat")
            if pos not in ("flat", "long", "short"):
                pos = "flat"
            ref["pos"] = pos
            n = int(j.get("n_positions", 0) or 0)
            if n > 1:
                logging.warning(
                    "Broker acct=%s: %s separate positions (expected 0–1). "
                    "Close extras in cTrader or: python -m src.execution --close-all --account-id %s",
                    acct_id,
                    n,
                    acct_id,
                )
            logging.info("Broker reconcile acct=%s pos=%s (n_positions=%s)", acct_id, pos, n or "?")
        except Exception as e:
            logging.warning("Broker reconcile acct=%s failed: %s", acct_id, e)


def _demo_broker_pos_after_action(action_taken: str) -> str | None:
    """If this action changed the real demo account, return new net position; else None."""
    at = (action_taken or "").strip().upper()
    if at in (
        "OPEN_LONG",
        "OPEN_LONG_SIMULATED",
        "REVERSE_TO_LONG_SIMULATED",
        "REVERSE_TO_LONG",
    ):
        return "long"
    if at in (
        "OPEN_SHORT",
        "OPEN_SHORT_SIMULATED",
        "REVERSE_TO_SHORT_SIMULATED",
        "REVERSE_TO_SHORT",
    ):
        return "short"
    if at in ("CLOSE", "CLOSE_SIMULATED"):
        return "flat"
    return None


def _next_sim_position(current: str, action_taken: str) -> str:
    """Map execution outcome to new simulated position."""
    if action_taken in (
        "OPEN_LONG_SIMULATED",
        "REVERSE_TO_LONG_SIMULATED",
        "OPEN_LONG",
        "REVERSE_TO_LONG",
    ):
        return "long"
    if action_taken in (
        "OPEN_SHORT_SIMULATED",
        "REVERSE_TO_SHORT_SIMULATED",
        "OPEN_SHORT",
        "REVERSE_TO_SHORT",
    ):
        return "short"
    if action_taken in ("CLOSE_SIMULATED", "CLOSE"):
        return "flat"
    return current


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _append_predictions_row(row: dict, run_at: str, strategy_id: str) -> None:
    csv_path = PROJECT_ROOT / PREDICTIONS_LIVE_CSV
    _ensure_dir(csv_path)
    write_row = {
        "strategy_id": strategy_id,
        "timestamp": row.get("timestamp", ""),
        "signal": row.get("signal", ""),
        "confidence": row.get("confidence", ""),
        "blocked": row.get("blocked", ""),
        "reason": row.get("reason", ""),
        "action": row.get("action", ""),
        "P_buy": row.get("P_buy", ""),
        "P_sell": row.get("P_sell", ""),
        "pred": row.get("pred", ""),
        "readiness_0_100": row.get("readiness_0_100", ""),
        "trade_hint": row.get("trade_hint", ""),
        "signal_source": row.get("signal_source", ""),
        "bar_lag_hours": row.get("bar_lag_hours", ""),
        "run_at": run_at,
    }
    fieldnames = list(write_row.keys())
    file_exists = csv_path.exists()
    needs_new_header = not file_exists
    if file_exists:
        with open(csv_path) as f:
            first = f.readline()
        if first.strip() and "readiness_0_100" not in first:
            legacy = csv_path.with_name(
                csv_path.stem + "_legacy_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + ".csv"
            )
            csv_path.rename(legacy)
            logging.info("Renamed predictions log without bar_lag_hours to %s", legacy.name)
            needs_new_header = True
            file_exists = False
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if needs_new_header:
            w.writeheader()
        w.writerow(write_row)


TRADE_DECISION_FIELDS = [
    "strategy_id",
    "mode",
    "timestamp",
    "bar_close_utc",
    "signal",
    "action_from_model",
    "action_taken",
    "sim_equity",
    "sim_position_after",
    "bar_return",
    "account_position_before",
    "broker_response",
    "order_open_utc",
    "run_at",
]


def _parse_order_open_time_utc(broker_response: str) -> str:
    """Best-effort: cTrader protobuf text often embeds openTimestamp (ms since epoch)."""
    if not broker_response:
        return ""
    text = broker_response
    try:
        d = json.loads(broker_response)
        if isinstance(d, dict):
            inner = d.get("order_response") or d.get("order_response_text")
            if isinstance(inner, str):
                text = inner
            elif inner is not None:
                text = str(inner)
    except json.JSONDecodeError:
        pass
    m = re.search(r"openTimestamp:\s*(\d{10,})", text)
    if not m:
        return ""
    try:
        ms = int(m.group(1))
        dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, OSError):
        return ""


def _append_paper_rows(
    strategy_id: str,
    timestamp: str,
    signal: str,
    action_model: str,
    action_taken: str,
    broker_response: str,
    run_at: str,
    sim_equity: float,
    sim_position_after: str,
    bar_return: float,
    *,
    mode: str = "sim",
    account_position_before: str = "",
) -> None:
    bar_close = (timestamp or "").strip()
    order_open = _parse_order_open_time_utc(broker_response) if mode == "demo" else ""
    row = {
        "strategy_id": strategy_id,
        "mode": mode,
        "timestamp": timestamp,
        "bar_close_utc": bar_close,
        "signal": signal,
        "action_from_model": action_model,
        "action_taken": action_taken,
        "sim_equity": f"{sim_equity:.6f}",
        "sim_position_after": sim_position_after,
        "bar_return": f"{bar_return:.8f}",
        "account_position_before": account_position_before,
        "broker_response": broker_response,
        "order_open_utc": order_open,
        "run_at": run_at,
    }
    for path in (PROJECT_ROOT / TRADE_DECISIONS_CSV, PAPER_SIM_CSV):
        _ensure_dir(path)
        exists = path.exists()
        needs_new_header = not exists
        if exists:
            with open(path) as f:
                first = f.readline()
            if first.strip() and (
                "sim_equity" not in first
                or "account_position_before" not in first
            ):
                legacy = path.with_name(
                    path.stem + "_legacy_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + ".csv"
                )
                path.rename(legacy)
                logging.info("Renamed legacy trade log to %s", legacy.name)
                needs_new_header = True
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRADE_DECISION_FIELDS)
            if needs_new_header:
                w.writeheader()
            w.writerow(row)


def _run_strategy(
    strategy: dict,
    execute: bool,
    run_at: str,
    paper_state: dict,
    *,
    demo_broker: bool = False,
    broker_position_ref: dict | None = None,
    parallel_paper_sim: bool = False,
    account_id_override: int | None = None,
) -> bool:
    sid = strategy["id"]
    strategy_demo = bool(
        demo_broker and sid in DEMO_BROKER_REAL_ORDER_STRATEGY_IDS
    )
    if strategy_demo and (
        broker_position_ref is None or account_id_override is None
    ):
        logging.error(
            "[%s] demo broker disabled this tick: missing broker_position_ref or "
            "account_id_override — add sid to DEMO_CTRADER_ACCOUNT_BY_STRATEGY in "
            "src/config_portfolio.py",
            sid,
        )
        strategy_demo = False
    state_snapshot: str | None = None
    state_file_path = None
    try:
        mod = importlib.import_module(strategy["module"])
        fn = getattr(mod, strategy["fn"])

        # Snapshot the strategy's state file before running so we can roll back if
        # the broker call fails. Without this, _save_state() inside the strategy
        # records e.g. current_position=1 before cTrader is called, and a timeout
        # leaves state and reality permanently out of sync.
        _paths_fn = getattr(mod, "state_paths_for", None)
        if _paths_fn is not None:
            try:
                sp, _ = _paths_fn(sid)
                if sp.exists():
                    state_snapshot = sp.read_text()
                    state_file_path = sp
            except Exception:
                pass
        else:
            _sf = getattr(mod, "STATE_FILE", None)
            if _sf is not None and _sf.exists():
                try:
                    state_snapshot = _sf.read_text()
                    state_file_path = _sf
                except Exception:
                    pass

        _call = {"n_bars": 1}
        if "strategy_id" in inspect.signature(fn).parameters:
            _call["strategy_id"] = sid
        rows = fn(**_call)
        # State snapshot no longer needed for rollback once fn() has returned successfully.
        # Broker failures below should not undo what the strategy already committed.
        _signal_fn_succeeded = True
        if not rows:
            logging.warning("[%s] No signal produced", sid)
            return False

        # ── Portfolio engine: permission + sizing ────────────────────────
        _port_state = load_portfolio_state()
        _recent_closes = _load_recent_closes(
            int(_PORTFOLIO_CFG.get("PORTFOLIO_VOL_LOOKBACK_BARS", 64)) + 4
        )

        for i, row in enumerate(rows):
            row = dict(row)
            enrich_signal_desired_position(row)
            # 1. Permission check — may override row to HOLD
            _allowed, _block_reason = is_strategy_allowed(
                sid, row, _port_state, _PORTFOLIO_CFG
            )
            if not _allowed:
                row["blocked"]          = True
                row["reason"]           = _block_reason
                row["action"]           = "HOLD"
                row["signal"]           = "flat"
                row["desired_position"] = 0
                row["portfolio_target_units"] = 0.0
                logging.info("[%s] portfolio blocked: %s", sid, _block_reason)
            else:
                # 2. Sizing + signed target (execution layer hint)
                _tgt = get_target_position(
                    row, _port_state, _recent_closes, _PORTFOLIO_CFG, strategy_id=sid
                )
                row["portfolio_size"]       = _tgt["size_abs"]
                row["portfolio_target_units"] = _tgt["target_units"]
                row["portfolio_size_note"]  = _tgt.get("note", "")
            rows[i] = row
        # ── end portfolio permission + sizing ────────────────────────────

        for row in rows:
            _append_predictions_row(row, run_at, strategy_id=sid)
            logging.info(
                "[%s] signal=%s action=%s blocked=%s",
                sid,
                row.get("signal", "?"),
                row.get("action", "?"),
                row.get("blocked", "?"),
            )

            if execute:
                from src.execution import process_signal

                # Independent dry-run book on the same bar as demo (strategy_id = `{sid}_paper` in CSVs).
                if demo_broker and parallel_paper_sim:
                    pk = f"{sid}_paper"
                    st_p = paper_state.setdefault(pk, {"position": "flat", "equity": 1.0})
                    sim_p = st_p.get("position", "flat")
                    if sim_p not in ("flat", "long", "short"):
                        sim_p = "flat"
                    eq_before_p = float(st_p["equity"])
                    br = float(row.get("bar_return") or 0.0)
                    try:
                        _ps = float(row.get("portfolio_size") or 1.0)
                    except (TypeError, ValueError):
                        _ps = 1.0
                    _ps = max(0.0, _ps)
                    if sim_p == "long":
                        eq_after_bar_p = eq_before_p * (1.0 + br * _ps)
                    elif sim_p == "short":
                        eq_after_bar_p = eq_before_p * (1.0 - br * _ps)
                    else:
                        eq_after_bar_p = eq_before_p

                    action_p, resp_p = process_signal(row, sim_p, dry_run=True)
                    new_p = _next_sim_position(sim_p, action_p)
                    st_p["equity"] = eq_after_bar_p
                    st_p["position"] = new_p
                    if action_p and str(action_p).strip().upper() != "NONE":
                        _append_paper_rows(
                            pk,
                            row.get("timestamp", ""),
                            row.get("signal", ""),
                            row.get("action", ""),
                            action_p,
                            resp_p,
                            run_at,
                            eq_after_bar_p,
                            new_p,
                            br,
                            mode="sim",
                            account_position_before=sim_p,
                        )
                    logging.info(
                        "[%s] parallel paper sim: action_taken=%s equity=%.6f position=%s",
                        sid,
                        action_p,
                        eq_after_bar_p,
                        new_p,
                    )

                st = paper_state.setdefault(sid, {"position": "flat", "equity": 1.0})
                # Per-strategy simulated book (equity / sim position) vs one shared demo account.
                sim_pos = st.get("position", "flat")
                if sim_pos not in ("flat", "long", "short"):
                    sim_pos = "flat"
                eq_before = float(st["equity"])
                br = float(row.get("bar_return") or 0.0)
                try:
                    _ps = float(row.get("portfolio_size") or 1.0)
                except (TypeError, ValueError):
                    _ps = 1.0
                _ps = max(0.0, _ps)
                if sim_pos == "long":
                    eq_after_bar = eq_before * (1.0 + br * _ps)
                elif sim_pos == "short":
                    eq_after_bar = eq_before * (1.0 - br * _ps)
                else:
                    eq_after_bar = eq_before

                dry_run = not strategy_demo
                if strategy_demo and broker_position_ref is not None:
                    broker_pos = broker_position_ref["pos"]
                else:
                    broker_pos = sim_pos
                account_before = broker_pos
                action_taken, broker_response = process_signal(
                    row, broker_pos, dry_run=dry_run, account_id_override=account_id_override
                )
                new_pos = _next_sim_position(sim_pos, action_taken)
                st["equity"] = eq_after_bar
                st["position"] = new_pos
                if strategy_demo and broker_position_ref is not None:
                    bp_new = _demo_broker_pos_after_action(action_taken)
                    if bp_new is not None:
                        broker_position_ref["pos"] = bp_new

                # ── Portfolio engine: record open / close (real demo orders only) ──
                try:
                    if strategy_demo:
                        if sim_pos == "flat" and new_pos in ("long", "short"):
                            # Position opened — record entry equity and direction
                            _dir = 1 if new_pos == "long" else -1
                            _sz  = float(row.get("portfolio_size", 1.0))
                            record_position_open(_port_state, sid, _dir, _sz)
                            st["last_open_equity"] = eq_after_bar
                        elif sim_pos in ("long", "short") and new_pos == "flat":
                            # Position closed — compute trade return and update portfolio state
                            _open_eq = float(st.get("last_open_equity") or eq_before)
                            _trade_ret = float(eq_after_bar - _open_eq) / max(_open_eq, 1e-12)
                            record_trade_result(_port_state, _trade_ret, sid, _PORTFOLIO_CFG)
                            record_position_close(_port_state, sid)
                        save_portfolio_state(_port_state)
                except Exception as _pe:
                    logging.warning("[%s] portfolio state update error: %s", sid, _pe)
                # ── end portfolio engine record ──────────────────────────

                if demo_broker and str(action_taken or "").strip().upper() == "NONE":
                    ma = str(row.get("action", "") or "").strip().upper()
                    br = broker_response or ""
                    if ma and ma != "NONE" and "skipped" in br:
                        logging.info(
                            "[%s] demo: skipped (model_action=%s account_before=%s) — %s",
                            sid,
                            ma,
                            account_before,
                            br[:500],
                        )

                # Log CSV rows only when a simulated order would fire (skip noisy NONE ticks)
                if action_taken and str(action_taken).strip().upper() != "NONE":
                    _append_paper_rows(
                        sid,
                        row.get("timestamp", ""),
                        row.get("signal", ""),
                        row.get("action", ""),
                        action_taken,
                        broker_response,
                        run_at,
                        eq_after_bar,
                        new_pos,
                        br,
                        mode="demo" if strategy_demo else "sim",
                        account_position_before=account_before,
                    )
                logging.info(
                    "[%s] %s: action_taken=%s equity=%.6f position=%s",
                    sid,
                    "demo broker" if strategy_demo else "paper sim",
                    action_taken,
                    eq_after_bar,
                    new_pos,
                )
        return True
    except Exception as e:
        # Only restore the pre-run state snapshot if fn() had NOT yet returned
        # (i.e. the error is from inside the strategy itself before its _save_state call).
        # If _signal_fn_succeeded is set, the strategy already committed its new state;
        # rolling that back here would desync state from the paper/demo equity book.
        if not locals().get("_signal_fn_succeeded") and state_snapshot is not None and state_file_path is not None:
            try:
                state_file_path.write_text(state_snapshot)
                logging.warning("[%s] Rolled back strategy state after execution failure", sid)
            except Exception as rb_err:
                logging.warning("[%s] State rollback failed: %s", sid, rb_err)
        logging.exception("[%s] strategy failed: %s", sid, e)
        return False


def run(
    refresh: bool = False,
    execute: bool = True,
    *,
    allow_auto_refresh: bool = True,
    demo_broker: bool = False,
    parallel_paper_sim: bool = False,
) -> int:
    try:
        import fcntl as _fcntl
    except ImportError:
        _fcntl = None  # Windows: no file lock; avoid overlapping runs manually

    if _fcntl is not None:
        LIVE_TICK_LOCK.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = open(LIVE_TICK_LOCK, "a+")
        try:
            _fcntl.flock(lock_fd.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except BlockingIOError:
            logging.warning("Another run_live_tick is already running — skipping (no duplicate rows)")
            lock_fd.close()
            return 0
        try:
            return _run_locked(
                refresh=refresh,
                execute=execute,
                allow_auto_refresh=allow_auto_refresh,
                demo_broker=demo_broker,
                parallel_paper_sim=parallel_paper_sim,
            )
        finally:
            try:
                _fcntl.flock(lock_fd.fileno(), _fcntl.LOCK_UN)
            except OSError:
                pass
            lock_fd.close()
    return _run_locked(
        refresh=refresh,
        execute=execute,
        allow_auto_refresh=allow_auto_refresh,
        demo_broker=demo_broker,
        parallel_paper_sim=parallel_paper_sim,
    )


def _run_locked(
    refresh: bool = False,
    execute: bool = True,
    *,
    allow_auto_refresh: bool = True,
    demo_broker: bool = False,
    parallel_paper_sim: bool = False,
) -> int:
    try:
        if not refresh and _want_auto_refresh(allow_auto_refresh):
            logging.info(
                "Livetick heartbeat stale (>%d min since last full success, or first run) — "
                "running data refresh to catch up after sleep/off time",
                LIVETICK_STALE_MINUTES,
            )
            refresh = True

        if refresh:
            if _is_data_refresh_running():
                logging.info(
                    "Data refresh already in progress — running strategies with current data (no wait)"
                )
                refresh = False
            else:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "scripts.run_daily_data_refresh"],
                    cwd=PROJECT_ROOT,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                logging.info(
                    "Data refresh started in background (PID %s) — running strategies with current data",
                    proc.pid,
                )
                # Don't wait: refresh script writes last_data_refresh_utc when Phase A completes

        # Always refresh the live tail from cTrader before running strategies.
        # Run in a subprocess so the Twisted reactor (used by cTrader SDK) gets its
        # own fresh process — execution.py's reactor call for get_open_positions later
        # in this same process would hit ReactorNotRestartable if we ran it in-process.
        try:
            result = subprocess.run(
                [sys.executable, "-m", "scripts.refresh_live_tail_ctrader"],
                cwd=PROJECT_ROOT,
                timeout=45,
            )
            if result.returncode != 0:
                logging.warning("cTrader live tail refresh subprocess returned non-zero — using existing tail")
        except subprocess.TimeoutExpired:
            logging.warning("cTrader live tail refresh timed out — using existing tail")
        except Exception as _ct_err:
            logging.warning("cTrader live tail refresh failed (%s) — using existing tail", _ct_err)

        if parallel_paper_sim and not demo_broker:
            logging.warning("Parallel paper sim is only used with demo broker — ignoring")
            parallel_paper_sim = False

        run_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        paper_state = _load_paper_state() if execute else {}

        # Per-account broker position refs (one entry per distinct cTrader account in use).
        # This replaces the old single broker_position_ref — each real-demo strategy now has
        # its own account and therefore its own independent position tracker.
        broker_position_refs: dict[int, dict] = {}  # account_id → {"pos": "flat"|"long"|"short"}
        if execute and demo_broker:
            for _s in STRATEGIES:
                if _s["id"] not in DEMO_BROKER_REAL_ORDER_STRATEGY_IDS:
                    continue
                _acct = _resolve_demo_account_id(_s["id"])
                if _acct is None:
                    continue
                if _acct not in broker_position_refs:
                    _bp_key = f"_demo_broker_pos_{_acct}"
                    # Prefer per-account key; fall back to legacy shared key on first run
                    _bp = paper_state.get(_bp_key, paper_state.get("_demo_broker_pos", "flat"))
                    if _bp not in ("flat", "long", "short"):
                        _bp = "flat"
                    broker_position_refs[_acct] = {"pos": _bp}
                    logging.info("Demo broker mode: account %s position (persisted) = %s", _acct, _bp)
            if parallel_paper_sim:
                logging.info(
                    "Parallel paper sim: also persisting independent dry-run books under {id}_paper"
                )
            _reconcile_demo_broker_refs(broker_position_refs)
        results = [
            _run_strategy(
                s,
                execute,
                run_at,
                paper_state,
                demo_broker=demo_broker,
                broker_position_ref=broker_position_refs.get(_resolve_demo_account_id(s["id"]) or -1),
                parallel_paper_sim=parallel_paper_sim,
                account_id_override=_resolve_demo_account_id(s["id"]),
            )
            for s in STRATEGIES
        ]
        if execute:
            # Persist per-account positions; keep legacy key pointing at the first account's pos
            for _acct_id, _ref in broker_position_refs.items():
                paper_state[f"_demo_broker_pos_{_acct_id}"] = _ref["pos"]
            if broker_position_refs:
                # Legacy key: use the position of the first account for backward compat
                _first_pos = next(iter(broker_position_refs.values()))["pos"]
                paper_state["_demo_broker_pos"] = _first_pos
            _save_paper_state(paper_state)
        ok = all(results)
        if ok:
            _write_heartbeat(
                last_full_success_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                demo_broker_active="true" if demo_broker else "false",
                parallel_paper_sim_active="true" if (execute and demo_broker and parallel_paper_sim) else "false",
            )
        return 0 if ok else 1

    except Exception as e:
        logging.exception("Live tick failed: %s", e)
        return 1


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="Live tick: all strategies; paper execution on by default")
    parser.add_argument("--refresh", action="store_true", help="Run data refresh first")
    parser.add_argument(
        "--no-auto-refresh",
        action="store_true",
        help="Disable stale-heartbeat auto refresh (manual runs)",
    )
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Signals only — no paper trade log or equity tracking",
    )
    parser.add_argument(
        "--demo-broker",
        action="store_true",
        help="Send real orders to demo broker (cTrader/OANDA/Binance). Overrides RUN_LIVETICK_DEMO_BROKER.",
    )
    parser.add_argument(
        "--parallel-paper",
        action="store_true",
        help="With demo broker, also run independent dry-run execution on the same bar (RUN_LIVETICK_PARALLEL_PAPER_SIM).",
    )
    args = parser.parse_args()
    execute = not args.no_execute
    demo_broker = args.demo_broker or (
        os.environ.get("RUN_LIVETICK_DEMO_BROKER", "").strip().lower() in ("1", "true", "yes")
    )
    parallel_paper = args.parallel_paper or (
        os.environ.get("RUN_LIVETICK_PARALLEL_PAPER_SIM", "").strip().lower() in ("1", "true", "yes")
    )

    sys.exit(
        run(
            refresh=args.refresh,
            execute=execute,
            allow_auto_refresh=not args.no_auto_refresh,
            demo_broker=demo_broker,
            parallel_paper_sim=parallel_paper,
        )
    )
