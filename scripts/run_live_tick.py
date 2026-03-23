"""
Phase 16 — Live tick script (every 5 min).

Runs all registered strategies, appends signals to predictions_live.csv.
By default also runs **paper execution**: tracks per-strategy simulated position
and equity (using each bar's bar_return), logs to trade_decisions.csv and
paper_simulation.csv. Use --no-execute for signals only.

**Demo broker**: With --demo-broker (or RUN_LIVETICK_DEMO_BROKER=1), the script
sends real orders to the configured broker (cTrader/OANDA/Binance demo). Position
is read from and updated on the broker; still paper/demo only (EXECUTION_PAPER_ONLY).
Use after you are happy with paper results.
"""

import csv
import importlib
import json
import logging
import os
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

STRATEGIES = [
    {"id": "classifier_v1", "module": "src.live_signal", "fn": "run"},
    {"id": "regression_v1", "module": "src.live_signal_regression", "fn": "run"},
    {"id": "mean_reversion_v1", "module": "src.live_signal_mean_reversion", "fn": "run"},
]

PAPER_STATE_PATH = PROJECT_ROOT / "data" / "logs" / "execution" / "paper_sim_state.json"
PAPER_SIM_CSV = PROJECT_ROOT / "data" / "logs" / "execution" / "paper_simulation.csv"
LIVE_TICK_LOCK = PROJECT_ROOT / "data" / "logs" / "execution" / "run_live_tick.lock"
HEARTBEAT_PATH = PROJECT_ROOT / LIVETICK_HEARTBEAT_JSON
FEATURES_LIVE_TAIL = PROJECT_ROOT / "data" / "features_regression_core" / "test_live_tail.parquet"
FEATURES_TEST = PROJECT_ROOT / "data" / "features_regression_core" / "test.parquet"


def _default_paper_state() -> dict:
    return {s["id"]: {"position": "flat", "equity": 1.0} for s in STRATEGIES}


def _load_paper_state() -> dict:
    PAPER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PAPER_STATE_PATH.exists():
        try:
            with open(PAPER_STATE_PATH) as f:
                data = json.load(f)
            out = _default_paper_state()
            for sid in out:
                if sid in data and isinstance(data[sid], dict):
                    pos = data[sid].get("position", "flat")
                    if pos not in ("flat", "long", "short"):
                        pos = "flat"
                    eq = float(data[sid].get("equity", 1.0))
                    out[sid] = {"position": pos, "equity": max(eq, 1e-9)}
            # One demo account: last known net position (for cTrader skip logic)
            bp = data.get("_demo_broker_pos", "flat")
            if bp in ("flat", "long", "short"):
                out["_demo_broker_pos"] = bp
            return out
        except Exception:
            pass
    return _default_paper_state()


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
    "signal",
    "action_from_model",
    "action_taken",
    "sim_equity",
    "sim_position_after",
    "bar_return",
    "broker_response",
    "run_at",
]


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
) -> None:
    row = {
        "strategy_id": strategy_id,
        "mode": mode,
        "timestamp": timestamp,
        "signal": signal,
        "action_from_model": action_model,
        "action_taken": action_taken,
        "sim_equity": f"{sim_equity:.6f}",
        "sim_position_after": sim_position_after,
        "bar_return": f"{bar_return:.8f}",
        "broker_response": broker_response,
        "run_at": run_at,
    }
    for path in (PROJECT_ROOT / TRADE_DECISIONS_CSV, PAPER_SIM_CSV):
        _ensure_dir(path)
        exists = path.exists()
        needs_new_header = not exists
        if exists:
            with open(path) as f:
                first = f.readline()
            if first.strip() and "sim_equity" not in first:
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
) -> bool:
    sid = strategy["id"]
    state_snapshot: str | None = None
    state_file_path = None
    try:
        mod = importlib.import_module(strategy["module"])
        fn = getattr(mod, strategy["fn"])

        # Snapshot the strategy's state file before running so we can roll back if
        # the broker call fails. Without this, _save_state() inside the strategy
        # records e.g. current_position=1 before cTrader is called, and a timeout
        # leaves state and reality permanently out of sync.
        _sf = getattr(mod, "STATE_FILE", None)
        if _sf is not None and _sf.exists():
            try:
                state_snapshot = _sf.read_text()
                state_file_path = _sf
            except Exception:
                pass

        rows = fn(n_bars=1)
        if not rows:
            logging.warning("[%s] No signal produced", sid)
            return False

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

                st = paper_state.setdefault(sid, {"position": "flat", "equity": 1.0})
                # Per-strategy simulated book (equity / sim position) vs one shared demo account.
                sim_pos = st.get("position", "flat")
                if sim_pos not in ("flat", "long", "short"):
                    sim_pos = "flat"
                eq_before = float(st["equity"])
                br = float(row.get("bar_return") or 0.0)
                if sim_pos == "long":
                    eq_after_bar = eq_before * (1.0 + br)
                elif sim_pos == "short":
                    eq_after_bar = eq_before * (1.0 - br)
                else:
                    eq_after_bar = eq_before

                dry_run = not demo_broker
                if demo_broker and broker_position_ref is not None:
                    broker_pos = broker_position_ref["pos"]
                else:
                    broker_pos = sim_pos
                action_taken, broker_response = process_signal(row, broker_pos, dry_run=dry_run)
                new_pos = _next_sim_position(sim_pos, action_taken)
                st["equity"] = eq_after_bar
                st["position"] = new_pos
                if demo_broker and broker_position_ref is not None:
                    bp_new = _demo_broker_pos_after_action(action_taken)
                    if bp_new is not None:
                        broker_position_ref["pos"] = bp_new

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
                        mode="demo" if demo_broker else "sim",
                    )
                logging.info(
                    "[%s] %s: action_taken=%s equity=%.6f position=%s",
                    sid,
                    "demo broker" if demo_broker else "paper sim",
                    action_taken,
                    eq_after_bar,
                    new_pos,
                )
        return True
    except Exception as e:
        # The strategy's _save_state() may have already written an updated position
        # (e.g. current_position=1) even though the broker call that follows never
        # succeeded. Restore the pre-run snapshot so the next tick retries correctly.
        if state_snapshot is not None and state_file_path is not None:
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
    )


def _run_locked(
    refresh: bool = False,
    execute: bool = True,
    *,
    allow_auto_refresh: bool = True,
    demo_broker: bool = False,
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

        run_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        paper_state = _load_paper_state() if execute else {}
        broker_position_ref = None
        if execute and demo_broker:
            # Persisted net demo account position (one broker; not inferred from per-strategy sim).
            # Twisted reactor can only start once per process — cannot reconcile broker here
            # before place_market_order(). Updated only when a real order fills.
            bp = paper_state.get("_demo_broker_pos", "flat")
            if bp not in ("flat", "long", "short"):
                bp = "flat"
            broker_position_ref = {"pos": bp}
            logging.info("Demo broker mode: account position (persisted) = %s", bp)
        results = [
            _run_strategy(
                s,
                execute,
                run_at,
                paper_state,
                demo_broker=demo_broker,
                broker_position_ref=broker_position_ref,
            )
            for s in STRATEGIES
        ]
        if execute:
            if demo_broker and broker_position_ref is not None:
                paper_state["_demo_broker_pos"] = broker_position_ref["pos"]
            _save_paper_state(paper_state)
        ok = all(results)
        if ok:
            _write_heartbeat(
                last_full_success_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                demo_broker_active="true" if demo_broker else "false",
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
    args = parser.parse_args()
    execute = not args.no_execute
    demo_broker = args.demo_broker or (
        os.environ.get("RUN_LIVETICK_DEMO_BROKER", "").strip().lower() in ("1", "true", "yes")
    )

    sys.exit(
        run(
            refresh=args.refresh,
            execute=execute,
            allow_auto_refresh=not args.no_auto_refresh,
            demo_broker=demo_broker,
        )
    )
