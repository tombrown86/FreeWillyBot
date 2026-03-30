"""
FreeWillyBot dashboard webserver.

Serves a single-page dashboard with live signals, execution logs, backtest metrics,
data diagnostics, model config, pipeline status, and feature freshness.

Launch (use project venv):
  ./scripts/run_dashboard.sh
  .venv/bin/python scripts/run_dashboard.py

Optional: --port 8080
"""

import argparse
import csv
import html as html_module
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_csv(path: Path, limit: int | None = None) -> tuple[list[dict], str | None]:
    """Load CSV, return (rows, error). Rows limited to last `limit` if set."""
    if not path.exists():
        return [], f"File not found: {path}"
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if limit and len(rows) > limit:
            rows = rows[-limit:]
        return rows, None
    except Exception as e:
        return [], str(e)


def _load_json(path: Path) -> tuple[dict | list | None, str | None]:
    """Load JSON, return (data, error)."""
    if not path.exists():
        return None, f"File not found: {path}"
    try:
        with open(path) as f:
            return json.load(f), None
    except Exception as e:
        return None, str(e)


def _file_mtime(path: Path) -> str:
    """Return human-readable mtime or '-' if missing."""
    if not path.exists():
        return "-"
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "-"


def load_config_summary() -> dict:
    """Load config summary from src.config."""
    try:
        from src import config

        return {
            "symbol": getattr(config, "SYMBOL", "?"),
            "bar_interval": getattr(config, "BAR_INTERVAL", "?"),
            "no_trade_threshold": getattr(config, "NO_TRADE_THRESHOLD_PCT", "?"),
            "min_confidence_pct": getattr(config, "MIN_CONFIDENCE_PCT", "?"),
            "max_daily_loss_pct": getattr(config, "MAX_DAILY_LOSS_PCT", "?"),
            "cooldown_bars": getattr(config, "COOLDOWN_BARS_AFTER_LOSS", "?"),
            "execution_paper_only": getattr(config, "EXECUTION_PAPER_ONLY", "?"),
        }
    except Exception:
        return {"error": "Could not load config"}


def load_regression_production_config() -> dict:
    """Load the locked production config for the regression strategy."""
    try:
        from src import config

        return {
            "top_pct": getattr(config, "REGRESSION_TOP_PCT", "?"),
            "vol_pct": getattr(config, "REGRESSION_VOL_PCT", "?"),
            "pred_threshold": getattr(config, "REGRESSION_PRED_THRESHOLD", "?"),
            "kill_switch_n": getattr(config, "REGRESSION_KILL_SWITCH_N", "?"),
            "kill_switch_pf": getattr(config, "REGRESSION_KILL_SWITCH_PF", "?"),
            "dd_kill": getattr(config, "REGRESSION_DD_KILL", "?"),
            "pause_bars": getattr(config, "REGRESSION_PAUSE_BARS", "?"),
        }
    except Exception:
        return {"error": "Could not load config"}


# Dashboard shows more rows so client-side filters stay useful.
DASHBOARD_PREDICTIONS_LIMIT = 500
# Signal log: render at most this many rows; show first N visible by default (expand for the rest).
DASHBOARD_SIGNAL_LOG_CAP = 100
DASHBOARD_SIGNAL_LOG_VISIBLE = 10


def load_predictions_live(root: Path, *, limit: int = 50) -> tuple[list[dict], str | None]:
    """Latest `limit` rows from predictions_live.csv (newest at end of file)."""
    path = root / "data" / "predictions" / "predictions_live.csv"
    return _load_csv(path, limit=limit)


def load_trade_decisions(root: Path) -> tuple[list[dict], str | None]:
    """All rows from trade_decisions.csv. Missing file = no decisions yet (not an error)."""
    path = root / "data" / "logs" / "execution" / "trade_decisions.csv"
    if not path.exists():
        return [], None
    return _load_csv(path)


def load_paper_sim_state(root: Path) -> dict | None:
    """Per-strategy paper equity/position from live tick state file."""
    path = root / "data" / "logs" / "execution" / "paper_sim_state.json"
    if not path.exists():
        return None
    data, err = _load_json(path)
    if err or not isinstance(data, dict):
        return None
    return data


def load_livetick_heartbeat(root: Path) -> dict:
    """Read livetick heartbeat JSON. Returns {} if missing/invalid."""
    path = root / "data" / "logs" / "execution" / "livetick_heartbeat.json"
    data, err = _load_json(path)
    if err or not isinstance(data, dict):
        return {}
    return data


def _paper_strategy_ids(root: Path) -> list[str]:
    """Strategy ids from run_live_tick.STRATEGIES so dashboard stays in sync."""
    import importlib.util
    path = root / "scripts" / "run_live_tick.py"
    if not path.exists():
        return ["classifier_v1", "regression_v1", "mean_reversion_v1"]
    try:
        spec = importlib.util.spec_from_file_location("run_live_tick", path)
        if spec is None or spec.loader is None:
            return ["classifier_v1", "regression_v1", "mean_reversion_v1"]
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return [s["id"] for s in getattr(mod, "STRATEGIES", [])]
    except Exception:
        return ["classifier_v1", "regression_v1", "mean_reversion_v1"]


def load_execution_log(root: Path) -> tuple[list[dict], str | None]:
    """Latest execution_log_*.csv (most recent by name)."""
    dir_path = root / "data" / "logs" / "execution"
    if not dir_path.exists():
        return [], "Directory not found"
    files = sorted(dir_path.glob("execution_log_*.csv"), reverse=True)
    if not files:
        return [], "No execution logs found"
    return _load_csv(files[0], limit=50)


def load_latest_backtest(root: Path) -> tuple[dict | None, str | None]:
    """Latest backtest_report_*.json."""
    dir_path = root / "data" / "backtests"
    if not dir_path.exists():
        return None, "Directory not found"
    files = sorted(dir_path.glob("backtest_report_*.json"), reverse=True)
    if not files:
        return None, "No backtest reports found"
    return _load_json(files[0])


def load_validation_report(root: Path) -> tuple[list[dict], str | None]:
    """validation_report.csv."""
    path = root / "data" / "validation" / "validation_report.csv"
    return _load_csv(path)


def load_regime_report(root: Path) -> tuple[list[dict], str | None]:
    """regime_report.csv."""
    path = root / "data" / "validation" / "regime_report.csv"
    return _load_csv(path)


def load_diagnostics(root: Path) -> tuple[dict | None, str | None]:
    """diagnostics_report.json."""
    path = root / "data" / "validation" / "diagnostics_report.json"
    return _load_json(path)


def load_baseline_choice(root: Path) -> tuple[dict | None, str | None]:
    """baseline_choice.json."""
    path = root / "data" / "models" / "baseline_choice.json"
    return _load_json(path)


def load_meta_config(root: Path) -> tuple[dict | None, str | None]:
    """meta_config.json."""
    path = root / "data" / "models" / "meta_config.json"
    return _load_json(path)


def load_pipeline_status(root: Path) -> list[tuple[str, str]]:
    """Log file -> last modified time."""
    logs_dir = root / "data" / "logs"
    files = [
        "livetick_stdout.log",
        "livetick_stderr.log",
        "retrain_stdout.log",
        "retrain_stderr.log",
        "data_refresh_stdout.log",
        "data_refresh_stderr.log",
        "build_features.log",
        "build_price_bars.log",
        "live_signal.log",
        "execution.log",
    ]
    return [(f, _file_mtime(logs_dir / f)) for f in files]


def load_walk_forward_results(root: Path) -> tuple[list[dict], list[dict]]:
    """Load walk-forward CSVs (1m and 2m windows)."""
    bt_dir = root / "data" / "backtests_regression"
    rows_1m, _ = _load_csv(bt_dir / "walk_forward_1m.csv")
    rows_2m, _ = _load_csv(bt_dir / "walk_forward_2m.csv")
    return rows_1m, rows_2m


def load_cost_stress(root: Path) -> list[dict]:
    rows, _ = _load_csv(root / "data" / "backtests_regression" / "cost_stress.csv")
    return rows


def load_strategy_registry():
    """Load strategy entries from src.strategy_registry. Returns [] on error."""
    try:
        from src.strategy_registry import STRATEGIES
        return STRATEGIES
    except Exception:
        return []


def _parse_trade_decision_ts(row: dict) -> datetime | None:
    """Best-effort parse from trade_decisions.csv row (run_at preferred)."""
    for key in ("run_at", "timestamp", "bar_close_utc"):
        raw = row.get(key)
        if not raw:
            continue
        s = str(raw).strip()
        if not s:
            continue
        try:
            if s.endswith(" UTC"):
                s = s[:-4].strip()
            if "T" not in s and " " in s:
                # "2026-03-30 12:00:00" -> ISO
                d_part, t_part = s.split(None, 1)
                s = f"{d_part}T{t_part}"
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            continue
    return None


def compute_trade_frequency_from_trade_log(trade_rows: list[dict]) -> dict[str, dict[str, dict]]:
    """Per strategy_id -> mode (sim|demo|live) -> {n, span_days, per_day, first, last}.

    Counts only rows where action_taken is set and not NONE (actual order attempts).
    """
    from collections import defaultdict

    raw: dict[tuple[str, str], list[datetime]] = defaultdict(list)
    for r in trade_rows:
        at = str(r.get("action_taken", "")).strip().upper()
        if not at or at == "NONE":
            continue
        mode = str(r.get("mode", "sim")).strip().lower()
        if mode not in ("sim", "demo", "live"):
            mode = "sim"
        sid = str(r.get("strategy_id", "")).strip()
        if not sid:
            continue
        ts = _parse_trade_decision_ts(r)
        if ts is None:
            continue
        raw[(sid, mode)].append(ts)

    out: dict[str, dict[str, dict]] = {}
    for (sid, mode), times in raw.items():
        times.sort()
        n = len(times)
        if n == 0:
            continue
        if n == 1:
            span_days = 1.0
        else:
            delta = (times[-1] - times[0]).total_seconds() / 86400.0
            span_days = max(delta, 1.0 / 24.0)
        per_day = n / span_days
        out.setdefault(sid, {})[mode] = {
            "n": n,
            "span_days": span_days,
            "per_day": per_day,
            "per_month": per_day * 30.44,
            "first": times[0].strftime("%Y-%m-%d %H:%M UTC"),
            "last": times[-1].strftime("%Y-%m-%d %H:%M UTC"),
        }
    return out


def _render_trade_frequency_section(
    registry: list,
    live_by_sid: dict[str, dict[str, dict]],
    *,
    trade_err: str | None,
) -> str:
    """HTML: backtest expectation vs trade_decisions.csv by mode."""
    intro = (
        "<p class='desc'><strong>Backtest / validation</strong> — qualitative cadence from research "
        "(see each strategy card for detail). "
        "<strong>Paper (sim)</strong> — rows in <code>trade_decisions.csv</code> with <code>mode=sim</code> "
        "(per-strategy simulated book). "
        "<strong>Demo</strong> — <code>mode=demo</code> (cTrader demo fills). "
        "<strong>Live</strong> — <code>mode=live</code> when real execution is enabled. "
        "Rates use wall-clock span between first and last <em>recorded</em> order in the log (not bar timestamps). "
        "Parallel dry-run uses <code>strategy_id</code> ending in <code>_paper</code>.</p>"
    )
    if trade_err:
        intro += f"<p class='desc'>{_html_escape(trade_err)}</p>"

    def cell(sid: str, mode: str) -> str:
        d = (live_by_sid.get(sid) or {}).get(mode)
        if not d or d.get("n", 0) == 0:
            return "—"
        return (
            f"<strong>{d['per_day']:.2f}/day</strong>"
            f"<br><span class='desc'>{d['n']} orders · {d['span_days']:.1f}d span</span>"
        )

    rows_html = ""
    for entry in registry:
        bt = (entry.trade_frequency_backtest or "").strip() or "—"
        src = (entry.trade_frequency_source or "").strip()
        src_cell = f"{_html_escape(bt)}"
        if src:
            src_cell += f"<br><span class='desc'>{_html_escape(src)}</span>"
        rows_html += (
            f"<tr><td><code>{_html_escape(entry.id)}</code></td>"
            f"<td>{src_cell}</td>"
            f"<td>{cell(entry.id, 'sim')}</td>"
            f"<td>{cell(entry.id, 'demo')}</td>"
            f"<td>{cell(entry.id, 'live')}</td></tr>"
        )

    parallel_rows = ""
    for sid in sorted(live_by_sid.keys()):
        if not sid.endswith("_paper"):
            continue
        st = live_by_sid[sid]
        for mode in ("sim", "demo", "live"):
            d = st.get(mode)
            if not d:
                continue
            parallel_rows += (
                f"<tr><td><code>{_html_escape(sid)}</code></td><td>{_html_escape(mode)}</td>"
                f"<td><strong>{d['per_day']:.2f}/day</strong> "
                f"<span class='desc'>(n={d['n']})</span></td></tr>"
            )

    parallel_block = ""
    if parallel_rows:
        parallel_block = (
            "<h4>Parallel paper ids (<code>*_paper</code>)</h4>"
            "<table><thead><tr><th>strategy_id</th><th>mode</th><th>Rate</th></tr></thead><tbody>"
            + parallel_rows
            + "</tbody></table>"
        )

    return (
        '<section class="trade-freq-section">'
        "<h3>Trade frequency — backtest vs recorded</h3>"
        f"{intro}"
        "<table class='trade-freq-table'><thead><tr>"
        "<th>Strategy</th><th>Backtest / validation (expected)</th>"
        "<th>Paper sim</th><th>cTrader demo</th><th>Live</th>"
        "</tr></thead><tbody>"
        f"{rows_html}</tbody></table>"
        f"{parallel_block}"
        "</section>"
    )


def load_portfolio_state_display(root: Path) -> dict | None:
    """Load portfolio_state.json for the portfolio engine status widget."""
    path = root / "data" / "logs" / "execution" / "portfolio_state.json"
    data, err = _load_json(path)
    if err or not isinstance(data, dict):
        return None
    return data


def load_portfolio_config_display() -> dict:
    """Return a display-friendly summary of portfolio config."""
    try:
        from src import config_portfolio as cp
        return {
            k: getattr(cp, k)
            for k in dir(cp)
            if k.startswith("PORTFOLIO_")
        }
    except Exception:
        return {}


def _format_portfolio_state(state: dict | None, cfg: dict) -> str:
    """Render a compact portfolio engine status block for the dashboard."""
    if not state:
        return "<p class='desc'>No portfolio state file yet — run the live tick once to create it.</p>"

    equity  = float(state.get("equity", 1.0))
    peak    = float(state.get("peak_equity", 1.0))
    dd      = (peak - equity) / peak if peak > 0 else 0.0
    active  = state.get("active_strategy") or "—"
    a_dir   = int(state.get("active_direction", 0))
    a_sz    = float(state.get("active_size", 0.0))
    paused  = bool(state.get("portfolio_paused"))
    p_until = state.get("pause_until_utc") or ""
    p_reason = state.get("pause_reason") or ""
    streak  = int(state.get("loss_streak", 0))
    trade_n = len(state.get("trade_rets", []))

    dir_str = "long" if a_dir > 0 else ("short" if a_dir < 0 else "flat")
    dd_class = "color:#f88" if dd >= 0.02 else ("color:#fa8" if dd >= 0.01 else "color:#8c8")
    pause_html = ""
    if paused or p_until:
        pause_html = (
            f"<p style='color:#f88;font-weight:bold;margin:0.3rem 0'>⚠ PORTFOLIO PAUSED"
            + (f" until {_html_escape(p_until)}" if p_until else "")
            + (f" — {_html_escape(p_reason)}" if p_reason else "")
            + "</p>"
        )

    by_strat = cfg.get("PORTFOLIO_SIZING_MODE_BY_STRATEGY") or {}
    default_mode = str(cfg.get("PORTFOLIO_SIZING_MODE_DEFAULT", "full"))
    sizing_rows = ""
    try:
        from scripts.run_live_tick import STRATEGIES as _strats
        ids = [s["id"] for s in _strats]
    except Exception:
        ids = list(by_strat.keys())
    for sid in ids:
        mode = by_strat.get(sid, default_mode)
        mode_color = "#caca7a" if mode == "vol_only" else ("#7aacda" if mode == "full" else "#8af")
        sizing_rows += (
            f"<tr><td>{_html_escape(sid)}</td>"
            f"<td style='color:{mode_color};font-weight:bold'>{_html_escape(mode)}</td></tr>"
        )

    siblings = cfg.get("PORTFOLIO_STRATEGY_SIBLINGS") or {}
    sibling_rows = ""
    for sid, sibs in siblings.items():
        sibling_rows += f"<tr><td>{_html_escape(sid)}</td><td>{_html_escape(', '.join(sibs))}</td></tr>"

    return f"""
{pause_html}
<div class="stat-bar">
  <span>Equity: <strong>{equity:.6f}</strong></span>
  &nbsp;·&nbsp;<span>Peak: <strong>{peak:.6f}</strong></span>
  &nbsp;·&nbsp;<span>Drawdown: <strong><span style="{dd_class}">{dd:.3%}</span></strong></span>
  &nbsp;·&nbsp;<span>Active strategy: <strong>{_html_escape(active)}</strong>
    {"(" + dir_str + " " + str(a_sz) + " lots)" if a_dir != 0 else ""}</span>
  &nbsp;·&nbsp;<span>Loss streak: <strong>{streak}</strong></span>
  &nbsp;·&nbsp;<span>Closed trades tracked: <strong>{trade_n}</strong></span>
</div>
<h4>Sizing modes</h4>
<p class="desc"><strong>full</strong> = vol × trend × DD × streak &nbsp;|&nbsp; <strong>vol_only</strong> = vol multiplier only &nbsp;|&nbsp; <strong>fixed</strong> = base size only</p>
<table><thead><tr><th>Strategy id</th><th>Mode</th></tr></thead><tbody>{sizing_rows}</tbody></table>
<h4>Sibling groups <span class="badge badge-sim">same-signal pairs</span></h4>
<p class="desc">When a sibling already holds an open position, the other is blocked from sending broker orders (paper equity books still track independently).</p>
<table><thead><tr><th>Strategy</th><th>Siblings (defers when sibling is active)</th></tr></thead><tbody>{sibling_rows}</tbody></table>
"""



def load_strategy_live_stats(root: Path, pred_rows: list[dict], paper_state: dict | None) -> dict:
    """
    For each strategy id return a dict of live stats derived from in-memory data:
        last_bar_ts, last_run_at, recent_signals (BUY/SELL/FLAT counts),
        current_position, current_equity, last_bar_source
    """
    out: dict[str, dict] = {}
    # Aggregate signal counts from last 50 prediction rows
    for row in pred_rows:
        sid = row.get("strategy_id", "")
        if not sid:
            continue
        if sid not in out:
            out[sid] = {
                "last_bar_ts": "",
                "last_run_at": "",
                "signal_counts": {"BUY": 0, "SELL": 0, "FLAT": 0},
                "current_position": "?",
                "current_equity": None,
                "last_bar_source": "",
                "bar_lag_hours": "",
            }
        s = str(row.get("signal", "")).upper()
        if s in out[sid]["signal_counts"]:
            out[sid]["signal_counts"][s] += 1
        # Most recent row wins for timestamp fields
        out[sid]["last_bar_ts"] = row.get("timestamp", "")
        out[sid]["last_run_at"] = row.get("run_at", "")
        out[sid]["bar_lag_hours"] = row.get("bar_lag_hours", "")
        out[sid]["last_bar_source"] = row.get("signal_source", "")

    # Overlay equity / position from paper state (skip meta keys and parallel `{id}_paper` entries)
    if paper_state and isinstance(paper_state, dict):
        for sid, st in paper_state.items():
            if sid.startswith("_"):
                continue
            if sid.endswith("_paper"):
                continue
            if isinstance(st, dict):
                if sid not in out:
                    out[sid] = {
                        "last_bar_ts": "", "last_run_at": "",
                        "signal_counts": {"BUY": 0, "SELL": 0, "FLAT": 0},
                        "current_position": "?", "current_equity": None,
                        "last_bar_source": "", "bar_lag_hours": "",
                    }
                out[sid]["current_position"] = st.get("position", "?")
                out[sid]["current_equity"] = st.get("equity")
        for sid, st in paper_state.items():
            if not sid.endswith("_paper") or not isinstance(st, dict):
                continue
            base = sid[: -len("_paper")]
            if base not in out:
                out[base] = {
                    "last_bar_ts": "", "last_run_at": "",
                    "signal_counts": {"BUY": 0, "SELL": 0, "FLAT": 0},
                    "current_position": "?", "current_equity": None,
                    "last_bar_source": "", "bar_lag_hours": "",
                }
            out[base]["parallel_paper_position"] = st.get("position", "?")
            out[base]["parallel_paper_equity"] = st.get("equity")
    return out


def load_feature_freshness(root: Path) -> list[tuple[str, str]]:
    """Key data files -> last modified time."""
    symbol, bar = "EURUSD", "5min"
    try:
        from src import config
        symbol = getattr(config, "SYMBOL", "EURUSD")
        bar = getattr(config, "BAR_INTERVAL", "5min")
    except Exception:
        pass
    price_stem = f"{symbol}_{bar}_clean"
    entries = [
        (f"processed/price/{price_stem}.csv", root / "data" / "processed" / "price" / f"{price_stem}.csv"),
        (f"processed/price/{price_stem}.parquet", root / "data" / "processed" / "price" / f"{price_stem}.parquet"),
        ("processed/aligned/cross_asset_aligned.csv", root / "data" / "processed" / "aligned" / "cross_asset_aligned.csv"),
        ("processed/aligned/macro_aligned.csv", root / "data" / "processed" / "aligned" / "macro_aligned.csv"),
        ("features/train.csv", root / "data" / "features" / "train.csv"),
        ("features/test.csv", root / "data" / "features" / "test.csv"),
        ("features/forecaster_predictions.csv", root / "data" / "features" / "forecaster_predictions.csv"),
    ]
    return [(label, _file_mtime(p)) for label, p in entries]


def _html_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _render_table(rows: list[dict], keys: list[str] | None = None) -> str:
    """Render rows as HTML table."""
    if not rows:
        return "<p>No rows</p>"
    keys = keys or list(rows[0].keys())
    lines = [
        "<table><thead><tr>",
        "".join(f"<th>{_html_escape(k)}</th>" for k in keys),
        "</tr></thead><tbody>",
    ]
    for row in rows:
        lines.append("<tr>")
        for k in keys:
            v = row.get(k, "")
            lines.append(f"<td>{_html_escape(str(v))}</td>")
        lines.append("</tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


def _equity_table_from_state(
    state: dict | None,
    root: Path,
    *,
    key_suffix: str | None,
) -> str:
    """Per-strategy equity from paper_sim_state: key is sid, or sid + '_paper' for parallel sim."""
    if not state or not isinstance(state, dict):
        return ""
    strategy_ids = _paper_strategy_ids(root)
    suf = key_suffix or ""
    parts: list[str] = []
    for sid in sorted(strategy_ids):
        key = f"{sid}{suf}" if suf else sid
        st = state.get(key)
        if not isinstance(st, dict):
            st = {"position": "flat", "equity": 1.0}
        try:
            eq = float(st.get("equity", 1.0))
            pct = (eq - 1.0) * 100
            pos = str(st.get("position", "flat"))
        except (TypeError, ValueError):
            eq, pct, pos = 1.0, 0.0, "?"
        parts.append(
            f"<tr><td>{_html_escape(sid)}</td><td>{eq:.6f}</td>"
            f"<td>{pct:+.4f}%</td><td>{_html_escape(pos)}</td></tr>"
        )
    if not parts:
        return ""
    return (
        '<table class="equity-table"><thead><tr>'
        '<th>Strategy</th><th>Equity</th><th>Return</th><th>Position</th>'
        "</tr></thead><tbody>"
        + "".join(parts)
        + "</tbody></table>"
    )


def _split_sim_order_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Rows with strategy_id ending in _paper (parallel paper) vs other mode=sim rows."""
    parallel: list[dict] = []
    main: list[dict] = []
    for r in rows:
        if str(r.get("mode", "sim")).strip().lower() != "sim":
            continue
        sid = str(r.get("strategy_id", ""))
        (parallel if sid.endswith("_paper") else main).append(r)
    return parallel, main


def _data_attr(name: str, value: object) -> str:
    """Safe HTML data-* attribute fragment."""
    v = html_module.escape(str(value), quote=True)
    return f'data-{name}="{v}"'


def _filter_select_options(values: list[str], select_id: str, label: str) -> str:
    parts = [
        f'<label class="filter-label">{_html_escape(label)} '
        f'<select id="{_html_escape(select_id)}">',
        '<option value="">All</option>',
    ]
    for v in values:
        ev = html_module.escape(str(v), quote=True)
        parts.append(f'<option value="{ev}">{_html_escape(str(v))}</option>')
    parts.append("</select></label>")
    return "".join(parts)


def _render_signal_log_block(
    rows: list[dict],
    err: str | None,
    *,
    visible_rows: int = DASHBOARD_SIGNAL_LOG_VISIBLE,
    cap_rows: int = DASHBOARD_SIGNAL_LOG_CAP,
) -> str:
    """Signal log table with strategy / signal / blocked / text filters. Newest rows at end."""
    if err:
        return f"<p>{_html_escape(err)}</p>"
    if not rows:
        return (
            "<p>No signals yet — run <code>python scripts/run_live_tick.py</code> "
            "to generate the first bar.</p>"
        )
    if len(rows) > cap_rows:
        rows = rows[-cap_rows:]
    keys = list(rows[0].keys())
    strategies = sorted({str(r.get("strategy_id", "")) for r in rows if str(r.get("strategy_id", "")).strip()})
    signals = sorted({str(r.get("signal", "")).upper() for r in rows if str(r.get("signal", "")).strip()})
    blocked_vals = sorted({str(r.get("blocked", "")) for r in rows})

    filters = (
        '<div class="table-filters" role="region" aria-label="Signal log filters">'
        + _filter_select_options(strategies, "siglog-f-strategy", "Strategy")
        + _filter_select_options(signals, "siglog-f-signal", "Signal")
        + _filter_select_options(blocked_vals, "siglog-f-blocked", "Blocked")
        + '<label class="filter-label">Search <input type="search" id="siglog-q" '
        + 'placeholder="Reason, hint, source…" autocomplete="off"></label>'
        + '<span id="siglog-count" class="filter-count"></span></div>'
    )

    thead = "<thead><tr>" + "".join(f"<th>{_html_escape(k)}</th>" for k in keys) + "</tr></thead>"
    tbody_lines = ['<tbody id="siglog-tbody">']
    n = len(rows)
    for i, row in enumerate(rows):
        # Oldest rows are hidden first; newest `visible_rows` stay visible when collapsed.
        extra = " siglog-row-extra" if n > visible_rows and i < n - visible_rows else ""
        ds = _data_attr("strategy", row.get("strategy_id", ""))
        sig = _data_attr("signal", str(row.get("signal", "")).upper())
        blk = _data_attr("blocked", row.get("blocked", ""))
        tbody_lines.append(f"<tr class=\"siglog-tr{extra}\" {ds} {sig} {blk}>")
        for k in keys:
            v = row.get(k, "")
            tbody_lines.append(f"<td>{_html_escape(str(v))}</td>")
        tbody_lines.append("</tr>")
    tbody_lines.append("</tbody>")
    collapse_note = ""
    if n > visible_rows:
        collapse_note = (
            f'<p class="siglog-collapse-bar"><button type="button" class="siglog-expand-btn" '
            f'id="siglog-expand-btn" aria-expanded="false">Show older rows (1–{n - visible_rows})</button> '
            f'<span class="desc">Showing newest {visible_rows} of {n} loaded (up to {DASHBOARD_PREDICTIONS_LIMIT} in file).</span></p>'
        )
    table = (
        f'<div id="siglog-wrap" class="siglog-wrap-collapsed">'
        f'<table class="filterable-table" id="siglog-table">{thead}{"".join(tbody_lines)}</table>'
        f"{collapse_note}</div>"
    )
    return filters + table


def _render_orders_filter_block(rows: list[dict], table_id: str) -> str:
    """Orders table with strategy / action / mode filters (demo or paper)."""
    if not rows:
        return "<p>No rows</p>"
    keys = list(rows[0].keys())
    strategies = sorted({str(r.get("strategy_id", "")) for r in rows if str(r.get("strategy_id", "")).strip()})
    actions = sorted({str(r.get("action_taken", "")) for r in rows if str(r.get("action_taken", "")).strip()})
    modes = sorted({str(r.get("mode", "")) for r in rows if str(r.get("mode", "")).strip()})

    filters = (
        f'<div class="table-filters" role="region" aria-label="Order filters">'
        + _filter_select_options(strategies, f"{table_id}-f-strategy", "Strategy")
        + _filter_select_options(actions, f"{table_id}-f-action", "Action taken")
        + _filter_select_options(modes, f"{table_id}-f-mode", "Mode")
        + f'<label class="filter-label">Search <input type="search" id="{table_id}-q" '
        + 'placeholder="Timestamp, response…" autocomplete="off"></label>'
        + f'<span id="{table_id}-count" class="filter-count"></span></div>'
    )

    thead = "<thead><tr>" + "".join(f"<th>{_html_escape(k)}</th>" for k in keys) + "</tr></thead>"
    tbody_lines = [f'<tbody id="{table_id}-tbody">']
    for row in rows:
        ds = _data_attr("strategy", row.get("strategy_id", ""))
        da = _data_attr("action", row.get("action_taken", ""))
        dm = _data_attr("mode", row.get("mode", ""))
        tbody_lines.append(f"<tr {ds} {da} {dm}>")
        for k in keys:
            v = row.get(k, "")
            tbody_lines.append(f"<td>{_html_escape(str(v))}</td>")
        tbody_lines.append("</tr>")
    tbody_lines.append("</tbody>")
    table = f'<table class="filterable-table" id="{table_id}-table">{thead}{"".join(tbody_lines)}</table>'
    return filters + table


def _dashboard_table_filter_script() -> str:
    """Client-side filters for signal log and order tables."""
    return r"""
<script>
(function () {
  var siglogApplyRef = null;
  function debounce(fn, ms) {
    var t;
    return function () {
      clearTimeout(t);
      var args = arguments, self = this;
      t = setTimeout(function () { fn.apply(self, args); }, ms);
    };
  }
    function bindTable(cfg) {
    var id = cfg.id;
    var tbody = document.getElementById(id + "-tbody");
    if (!tbody) return;
    var countEl = document.getElementById(id + "-count");
    var fields = cfg.fields || [];

    function apply() {
      var qEl = document.getElementById(id + "-q");
      var q = (qEl && qEl.value || "").toLowerCase().trim();
      var n = 0, tot = 0;
      var wrap = id === "siglog" ? document.getElementById("siglog-wrap") : null;
      tbody.querySelectorAll("tr").forEach(function (tr) {
        tot++;
        var ok = true;
        fields.forEach(function (f) {
          var sel = document.getElementById(id + "-f-" + f);
          if (!sel || !sel.value) return;
          var dv = tr.getAttribute("data-" + f) || "";
          if (dv !== sel.value) ok = false;
        });
        if (ok && q) {
          var txt = (tr.textContent || "").toLowerCase();
          if (txt.indexOf(q) < 0) ok = false;
        }
        var show = ok;
        if (ok && wrap && wrap.classList.contains("siglog-wrap-collapsed") && tr.classList.contains("siglog-row-extra")) {
          show = false;
        }
        tr.style.display = show ? "" : "none";
        if (show) n++;
      });
      if (countEl) countEl.textContent = n + " / " + tot + " rows";
    }

    fields.forEach(function (f) {
      var sel = document.getElementById(id + "-f-" + f);
      if (sel) sel.addEventListener("change", apply);
    });
    var qEl = document.getElementById(id + "-q");
    if (qEl) qEl.addEventListener("input", debounce(apply, 200));
    apply();
    if (id === "siglog") siglogApplyRef = apply;
  }
  bindTable({ id: "siglog", fields: ["strategy", "signal", "blocked"] });
  bindTable({ id: "orddemo", fields: ["strategy", "action", "mode"] });
  bindTable({ id: "ordpaper", fields: ["strategy", "action", "mode"] });
  (function () {
    var wrap = document.getElementById("siglog-wrap");
    var btn = document.getElementById("siglog-expand-btn");
    if (!wrap || !btn) return;
    btn.addEventListener("click", function () {
      var collapsed = wrap.classList.toggle("siglog-wrap-collapsed");
      btn.setAttribute("aria-expanded", collapsed ? "false" : "true");
      btn.textContent = collapsed ? "Show older rows" : "Show only newest 10 rows";
      if (siglogApplyRef) siglogApplyRef();
    });
  })();
})();
</script>
"""


def _wf_format_cell(key: str, value: object) -> str:
    """Format walk-forward CSV cells: net_return and max_dd as %, profit_factor rounded."""
    if value is None or value == "":
        return "—"
    try:
        if key == "net_return":
            return f"{float(value):.2%}"
        if key == "max_dd":
            return f"{float(value):.2%}"
        if key == "profit_factor":
            return f"{float(value):.2f}"
        if key == "n_trades":
            return str(int(float(value)))
    except (ValueError, TypeError):
        pass
    return str(value)


def _render_walk_forward_table(rows: list[dict]) -> str:
    """Walk-forward table with human-readable percentages and stable column order."""
    if not rows:
        return "<p>No rows</p>"
    keys = list(rows[0].keys())
    preferred = ["window_start", "window_months", "net_return", "profit_factor", "max_dd", "n_trades"]
    keys = [k for k in preferred if k in keys] + [k for k in keys if k not in preferred]
    lines = [
        "<table class='wf-table'><thead><tr>",
        "".join(f"<th>{_html_escape(k)}</th>" for k in keys),
        "</tr></thead><tbody>",
    ]
    for row in rows:
        lines.append("<tr>")
        for k in keys:
            raw = row.get(k, "")
            lines.append(
                f"<td title='{_html_escape(str(raw))}'>{_html_escape(_wf_format_cell(k, raw))}</td>"
            )
        lines.append("</tr>")
    lines.append("</tbody></table>")
    return "\n".join(lines)


def _walk_forward_summary_html(rows: list[dict]) -> str:
    """Stats for walk-forward windows; explains sum vs full-period headline."""
    if not rows:
        return "<p>No data</p>"
    try:
        nets = [float(r["net_return"]) for r in rows]
    except (KeyError, ValueError, TypeError):
        return "<p>Could not parse walk-forward rows</p>"
    n_pos = sum(1 for n in nets if n > 0)
    worst = min(nets)
    best = max(nets)
    total = sum(nets)
    mean = total / len(nets)
    return f"""<div class="wf-summary">
<p><strong>{n_pos}/{len(nets)} windows positive</strong> (after costs, each window independent)</p>
<ul>
<li><strong>Sum of window returns</strong>: {total:.2%} — arithmetic sum of per-window returns; <em>not</em> the same as compound full-period ROE on one continuous run.</li>
<li><strong>Mean per window</strong>: {mean:.2%} · <strong>Best</strong>: {best:.2%} · <strong>Worst</strong>: {worst:.2%}</li>
</ul>
</div>"""


def _render_json_block(obj: dict | list) -> str:
    """Render JSON as preformatted block."""
    try:
        raw = json.dumps(obj, indent=2)
        return f"<pre>{_html_escape(raw)}</pre>"
    except Exception:
        return "<p>Could not serialize</p>"


def build_html(root: Path) -> str:
    """Build full dashboard HTML."""
    cfg = load_config_summary()
    reg_cfg = load_regression_production_config()
    pred_rows, pred_err = load_predictions_live(root, limit=DASHBOARD_PREDICTIONS_LIMIT)
    trade_rows, trade_err = load_trade_decisions(root)
    paper_state = load_paper_sim_state(root)
    heartbeat = load_livetick_heartbeat(root)
    is_demo = str(heartbeat.get("demo_broker_active", "false")).lower() in ("true", "1", "yes")
    exec_rows, exec_err = load_execution_log(root)
    backtest, bt_err = load_latest_backtest(root)
    val_rows, val_err = load_validation_report(root)
    regime_rows, regime_err = load_regime_report(root)
    diag, diag_err = load_diagnostics(root)
    baseline, base_err = load_baseline_choice(root)
    meta_cfg, meta_err = load_meta_config(root)
    pipeline = load_pipeline_status(root)
    freshness = load_feature_freshness(root)
    wf_1m, wf_2m = load_walk_forward_results(root)
    cost_stress = load_cost_stress(root)

    registry = load_strategy_registry()
    strategy_live = load_strategy_live_stats(root, pred_rows, paper_state)
    portfolio_state = load_portfolio_state_display(root)
    portfolio_cfg   = load_portfolio_config_display()
    tf_live = compute_trade_frequency_from_trade_log(trade_rows or [])
    trade_freq_html = _render_trade_frequency_section(registry, tf_live, trade_err=trade_err)

    # ── walk-forward (regression_v1) ───────────────────────────────────────────
    wf_intro_html = """<details class="inner-details wf-details">
<summary>How to read walk-forward vs full-period validation</summary>
<div class="inner-body wf-note">
<p><strong>Why the numbers here can look worse than the ~3–6% style headline in <code>src/config.py</code>:</strong></p>
<ul>
<li>The <strong>production comment</strong> in config is a <strong>single continuous out-of-sample backtest</strong> (e.g. Jan–Dec 2024) with the locked regression rules — one equity curve, one net return (e.g. ~+5.4% in that validation run).</li>
<li>This table is <strong>rolling independent windows</strong>: each row is its own mini test period. <strong>Summing</strong> monthly <code>net_return</code> values is <em>not</em> the same metric as that full-period number (different compounding, different trade sets, and a bad month drags the sum without invalidating the full-year story).</li>
<li><strong>Net sum</strong> here answers: &quot;If I add up each window’s return as reported, what do I get?&quot; It can be slightly negative while several windows are strongly positive — check <strong>mean</strong>, <strong>best/worst</strong>, and the per-window table.</li>
</ul>
<p class="desc">Hover a table cell to see the raw float stored in the CSV. Regenerate files with <code>scripts/run_walk_forward_regression.py</code> after changing strategy code.</p>
</div>
</details>"""

    wf_block_html = (
        wf_intro_html
        + "<h4>12 × 1-month windows</h4>"
        + _walk_forward_summary_html(wf_1m)
        + (_render_walk_forward_table(wf_1m) if wf_1m else "<p>No data — run scripts/run_walk_forward_regression.py</p>")
        + "<h4>6 × 2-month windows</h4>"
        + _walk_forward_summary_html(wf_2m)
        + (_render_walk_forward_table(wf_2m) if wf_2m else "<p>No data — run scripts/run_walk_forward_regression.py</p>")
    )
    cost_stress_html = _render_table(cost_stress) if cost_stress else "<p>No data — run scripts/run_cost_stress_regression.py</p>"

    # ── registry-driven strategy cards ───────────────────────────────────────
    def _render_strategy_card(entry, live: dict, mode_badge: str) -> str:
        cfg_vals = entry.live_config()
        # Config param table
        param_rows = ""
        for p in entry.params:
            val = cfg_vals.get(p.key, "?")
            display = f"{val} {p.unit}".strip() if p.unit else str(val)
            param_rows += (
                f"<tr><td><code>{_html_escape(p.key)}</code></td>"
                f"<td><strong>{_html_escape(display)}</strong></td>"
                f"<td>{_html_escape(p.label)}</td>"
                f"<td class='desc'>{_html_escape(p.description)}</td></tr>"
            )
        param_table = (
            "<table class='param-table'><thead><tr>"
            "<th>Config key</th><th>Value</th><th>Name</th><th>What it does</th>"
            "</tr></thead><tbody>" + param_rows + "</tbody></table>"
            if param_rows else "<p class='desc'>No parameters defined.</p>"
        )

        # Sizing mode badge
        sizing_by = portfolio_cfg.get("PORTFOLIO_SIZING_MODE_BY_STRATEGY") or {}
        sizing_default = str(portfolio_cfg.get("PORTFOLIO_SIZING_MODE_DEFAULT", "full"))
        sizing_mode = sizing_by.get(entry.id, sizing_default)
        sizing_colors = {"vol_only": "badge-technique", "full": "badge-active", "fixed": "badge-sim"}
        sizing_cls = sizing_colors.get(sizing_mode, "badge-future")
        sizing_badge = (
            f'<span class="badge {sizing_cls}">'
            f"SIZING: {_html_escape(sizing_mode.upper())}</span>"
        )

        # Live stats bar
        stats = live.get(entry.id, {})
        counts = stats.get("signal_counts", {})
        buy_n  = counts.get("BUY", 0)
        sell_n = counts.get("SELL", 0)
        flat_n = counts.get("FLAT", 0)
        total  = buy_n + sell_n + flat_n
        last_bar   = _html_escape(str(stats.get("last_bar_ts", "—")))
        last_run   = _html_escape(str(stats.get("last_run_at", "—")))
        lag        = stats.get("bar_lag_hours", "")
        try:
            lag_f = float(lag)
            lag_html = f" <span class='lag {'lag-warn' if lag_f > 0.5 else 'lag-ok'}'>(lag {lag}h)</span>"
        except (TypeError, ValueError):
            lag_html = ""
        pos        = _html_escape(str(stats.get("current_position", "?")))
        eq         = stats.get("current_equity")
        eq_html    = f"{float(eq):.6f}" if eq is not None else "—"
        pp_eq = stats.get("parallel_paper_equity")
        pp_pos = stats.get("parallel_paper_position")
        parallel_html = ""
        if pp_eq is not None or pp_pos is not None:
            try:
                pp_eq_html = f"{float(pp_eq):.6f}" if pp_eq is not None else "—"
            except (TypeError, ValueError):
                pp_eq_html = "—"
            pp_pos_html = _html_escape(str(pp_pos if pp_pos is not None else "?"))
            parallel_html = (
                f" &nbsp;·&nbsp; <span>Parallel paper: pos <strong>{pp_pos_html}</strong></span>"
                f" &nbsp;·&nbsp; <span>Equity: <strong>{pp_eq_html}</strong></span>"
            )
        src        = _html_escape(str(stats.get("last_bar_source", "—")))
        locked_badge = '<span class="badge badge-locked">LOCKED CONFIG</span>' if entry.config_locked else ""
        active_badge = '<span class="badge badge-active">ACTIVE</span>' if entry.active else '<span class="badge badge-future">INACTIVE</span>'

        stats_html = (
            f"<div class='stat-bar'>"
            f"<span>Last bar: <strong>{last_bar}</strong>{lag_html}</span>"
            f" &nbsp;·&nbsp; <span>Last run: <strong>{last_run}</strong></span>"
            f" &nbsp;·&nbsp; <span>Position: <strong>{pos}</strong></span>"
            f" &nbsp;·&nbsp; <span>Equity: <strong>{eq_html}</strong></span>"
            f"{parallel_html}"
            f" &nbsp;·&nbsp; <span>Source: <code>{src}</code></span>"
            f" &nbsp;·&nbsp; <span class='sig-dist'>BUY <strong>{buy_n}</strong> · SELL <strong>{sell_n}</strong> · FLAT <strong>{flat_n}</strong>"
            + (f" <span class='desc'>(last {total})</span>" if total else "")
            + "</span></div>"
        )

        return f"""<details class="strategy-card">
<summary>
  <span class="strat-name">{_html_escape(entry.name)}</span>
  {active_badge} {locked_badge}
  <span class="badge badge-technique">{_html_escape(entry.technique)}</span>
  {sizing_badge}
  {mode_badge}
</summary>
<div class="strat-body">
  {stats_html}
  <div class="strat-desc">
    <p>{entry.plain_description}</p>
  </div>
  <details class="inner-details">
    <summary>Technical detail</summary>
    <div class="inner-body">
      <p class="mono-desc">{entry.technical_description}</p>
      <p class="desc"><strong>Best conditions:</strong> {_html_escape(entry.best_conditions)}</p>
      <p class="desc"><strong>Known weaknesses:</strong> {_html_escape(entry.known_weaknesses)}</p>
    </div>
  </details>
  <details class="inner-details" open>
    <summary>Live parameters (from config.py)</summary>
    <div class="inner-body">{param_table}</div>
  </details>
</div>
</details>"""

    mode_badge = f'<span class="badge {"badge-demo" if is_demo else "badge-sim"}">{"DEMO" if is_demo else "SIMULATION"}</span>'
    strategies_html = "\n".join(
        _render_strategy_card(entry, strategy_live, mode_badge)
        for entry in registry
    ) if registry else "<p>Could not load strategy registry.</p>"

    # ── per-strategy signal counts (for signal log header) ────────────────────
    strategy_ids = sorted({r.get("strategy_id", "unknown") for r in pred_rows})
    sig_parts = []
    for sid in (strategy_ids or []):
        s_rows = [r for r in pred_rows if r.get("strategy_id", "") == sid]
        counts = {"BUY": 0, "SELL": 0, "FLAT": 0}
        for r in s_rows:
            s = str(r.get("signal", "")).upper()
            if s in counts:
                counts[s] += 1
        sig_parts.append(f"<strong>{_html_escape(sid)}</strong>: BUY {counts['BUY']} | SELL {counts['SELL']} | FLAT {counts['FLAT']}")
    signal_dist = " &nbsp;&nbsp;&nbsp; ".join(sig_parts) if sig_parts else "No signals yet — run scripts/run_live_tick.py"

    # ── simulated trade decisions (split from future real trades) ────────────
    paper_trade_rows = [r for r in trade_rows if r.get("mode", "sim") == "sim"]
    demo_trade_rows  = [r for r in trade_rows if r.get("mode") == "demo"]
    real_trade_rows  = [r for r in trade_rows if r.get("mode") == "live"]

    def _paper_equity_summary(
        state: dict | None,
        rows: list[dict],
        root: Path,
        *,
        skip_if_parallel: bool = False,
    ) -> str:
        """Single-table equity (paper-only mode or CSV fallback). Empty when skip_if_parallel (demo uses split tables)."""
        strategy_ids = _paper_strategy_ids(root)
        if skip_if_parallel:
            return ""
        lines = [
            '<table class="equity-table"><thead><tr>'
            '<th>Strategy</th><th>Simulated equity</th><th>Return</th><th>Position</th>'
            "</tr></thead><tbody>",
        ]
        # Show all strategies from run_live_tick so new strategies appear even before first tick
        if state and isinstance(state, dict):
            for sid in sorted(strategy_ids):
                st = state.get(sid) or {}
                if not isinstance(st, dict):
                    st = {"position": "flat", "equity": 1.0}
                try:
                    eq = float(st.get("equity", 1.0))
                    pct = (eq - 1.0) * 100
                    pos = str(st.get("position", "flat"))
                except (TypeError, ValueError):
                    eq, pct, pos = 1.0, 0.0, "?"
                lines.append(
                    f"<tr><td>{_html_escape(sid)}</td><td>{eq:.6f}</td>"
                    f"<td>{pct:+.4f}%</td><td>{_html_escape(pos)}</td></tr>"
                )
            lines.append("</tbody></table>")
            return "".join(lines)
        if trade_err:
            return ""
        strategy_ids = _paper_strategy_ids(root)
        if not any("sim_equity" in r for r in rows):
            for sid in sorted(strategy_ids):
                lines.append(
                    f"<tr><td>{_html_escape(sid)}</td><td>1.000000</td>"
                    "<td>+0.0000%</td><td>flat</td></tr>"
                )
            lines.append("</tbody></table>")
            return "".join(lines) + (
                "<p class=\"desc\">Run <code>python scripts/run_live_tick.py</code> once to create "
                "<code>paper_sim_state.json</code>.</p>"
            )
        last: dict[str, dict] = {}
        for r in reversed(rows):
            sid = r.get("strategy_id") or ""
            if sid and sid not in last:
                last[sid] = r
        for sid in sorted(strategy_ids):
            r = last.get(sid)
            if not r:
                lines.append(
                    f"<tr><td>{_html_escape(sid)}</td><td>1.000000</td>"
                    "<td>+0.0000%</td><td>flat</td></tr>"
                )
                continue
            try:
                eq = float(r.get("sim_equity", 1))
                pct = (eq - 1.0) * 100
                pos = r.get("sim_position_after", "")
            except (TypeError, ValueError):
                eq, pct, pos = 1.0, 0.0, "?"
            lines.append(
                f"<tr><td>{_html_escape(sid)}</td><td>{eq:.6f}</td>"
                f"<td>{pct:+.4f}%</td><td>{_html_escape(str(pos))}</td></tr>"
            )
        lines.append("</tbody></table>")
        return "".join(lines)

    _ids_for_parallel = _paper_strategy_ids(root)
    has_parallel_state = bool(
        paper_state and isinstance(paper_state, dict) and any(
            isinstance(paper_state.get(f"{sid}_paper"), dict) for sid in _ids_for_parallel
        )
    )
    demo_path_equity_html = (
        _equity_table_from_state(paper_state, root, key_suffix=None) if paper_state else ""
    )
    parallel_paper_equity_html = (
        _equity_table_from_state(paper_state, root, key_suffix="_paper") if has_parallel_state else ""
    )
    paper_sim_equity_html = _paper_equity_summary(
        paper_state, trade_rows, root, skip_if_parallel=has_parallel_state and is_demo
    )
    demo_equity_block = demo_path_equity_html or (
        paper_sim_equity_html if not has_parallel_state else ""
    )

    def _is_order_row(r: dict) -> bool:
        at = str(r.get("action_taken", "")).strip().upper()
        return bool(at) and at != "NONE"

    def _orders_html(rows: list[dict], empty_msg: str, table_id: str) -> str:
        if trade_err:
            return trade_err
        order_rows = [r for r in rows if _is_order_row(r)]
        if not order_rows:
            return f"<p>{empty_msg}</p>"
        return _render_orders_filter_block(order_rows[-200:], table_id)

    paper_parallel_rows, paper_main_sim_rows = _split_sim_order_rows(paper_trade_rows)
    paper_orders_html = _orders_html(
        paper_parallel_rows if is_demo else paper_trade_rows,
        "No parallel paper orders yet — independent dry-run rows use strategy_id ending in _paper when demo + parallel sim is enabled.",
        "ordpaper",
    )
    demo_orders_html = _orders_html(
        demo_trade_rows,
        "No demo broker orders yet — orders will appear here when the next tick places or changes a position.",
        "orddemo",
    )
    real_trade_html = (
        _render_table(real_trade_rows) if real_trade_rows else
        "<p>No real trades recorded yet. Real trades will appear here when execution_paper_only = False and a broker order is filled.</p>"
    )

    # ── execution / backtest / validation ────────────────────────────────────
    exec_html = exec_err if exec_err else (
        _render_table(exec_rows) if exec_rows else "<p>No execution log entries yet.</p>"
    )
    bt_html = bt_err if bt_err else ""
    if backtest:
        bt_html = "<h4>Strategy</h4>" + _render_table([backtest.get("strategy", {})])
        bt_html += "<h4>Filter stats</h4>" + _render_table([backtest.get("filter_stats", {})])
        bt_html += "<h4>Baselines</h4>" + _render_table([
            {"name": "strategy", **backtest.get("strategy", {})},
            {"name": "momentum", **backtest.get("baseline_momentum", {})},
            {"name": "flat", **backtest.get("baseline_flat", {})},
        ])
    val_html = val_err if val_err else _render_table(val_rows)
    regime_html = regime_err if regime_err else _render_table(regime_rows)
    diag_html = diag_err if diag_err else _render_json_block(diag) if diag else "<p>No data</p>"
    base_html = base_err if base_err else _render_json_block(baseline) if baseline else "<p>No data</p>"
    meta_html = meta_err if meta_err else _render_json_block(meta_cfg) if meta_cfg else "<p>No data</p>"

    # ── pred signal table (filterable) ───────────────────────────────────────
    pred_html = _render_signal_log_block(pred_rows, pred_err)

    # ── collapsible execution blocks ─────────────────────────────────────────
    _paper_open = "" if is_demo else " open"
    _demo_open  = " open" if is_demo else ""

    _paper_parallel_section = ""
    if is_demo:
        _paper_parallel_section = (
            "<h3>Parallel paper equity <span class=\"badge badge-sim\">SIMULATION</span></h3>"
            "<p class=\"desc\">Independent dry-run book per strategy (state keys <code>{id}_paper</code>). "
            "Same bar as demo; no broker orders. Compare to demo-path equity in the section above.</p>"
            + (
                parallel_paper_equity_html
                or '<p class="desc">No <code>*_paper</code> keys in state — set RUN_LIVETICK_PARALLEL_PAPER_SIM with demo broker.</p>'
            )
        )

    # Built outside f-string: nested quotes/backslashes inside {...} break Python's f-string parser.
    _paper_sim_only_equity_block = ""
    if not is_demo:
        _paper_sim_only_equity_block = (
            '<h3>Current equity <span class="badge badge-sim">SIMULATION</span></h3>'
            '<p class="desc">Each strategy keeps its own simulated account starting at 1.0. While flat, equity is unchanged. '
            "While long or short, each bar applies that bar's return. Reset via <code>paper_sim_state.json</code> or the reset script.</p>"
            + paper_sim_equity_html
        )

    paper_block_html = f"""<details{_paper_open}>
<summary><strong>Paper simulation</strong> <span class="badge badge-sim">SIMULATION</span> — parallel paper equity and orders</summary>
<section>
{_paper_sim_only_equity_block}
{_paper_parallel_section if is_demo else ""}
<h3>Orders <span class="badge badge-sim">SIMULATION</span></h3>
<p class="desc">Parallel paper only: <code>strategy_id</code> ends with <code>_paper</code> (mode=sim). Full signal history is in the log above.</p>
{paper_orders_html}
</section>
</details>"""

    _demo_count_note = (
        '<p class="desc">Why parallel paper rows can outnumber demo: strategies share <strong>one</strong> demo account. '
        "After the first strategy opens a position, <code>process_signal</code> often skips others "
        "(<code>already_long</code> / <code>already_short</code>). Parallel paper uses a separate book per strategy, "
        "so each can log.</p>"
    )

    demo_block_html = f"""<details{_demo_open}>
<summary><strong>Demo broker</strong> <span class="badge badge-demo">DEMO</span> — demo-path equity and orders</summary>
<section>
<h3>Demo-path equity <span class="badge badge-demo">DEMO</span></h3>
<p class="desc">Per-strategy book after each demo fill (shared account position for skip logic). State file: <code>paper_sim_state.json</code>.</p>
{demo_equity_block or '<p class="desc">No equity state yet — run livetick or check state file.</p>'}
<h3>Orders <span class="badge badge-demo">DEMO</span></h3>
{_demo_count_note}
<p class="desc">Real demo fills only (mode=demo). Latest 200 rows; use filters.</p>
{demo_orders_html}
</section>
</details>"""

    real_trades_block_html = f"""<details>
<summary><strong>Real trades</strong> <span class="badge badge-live">LIVE</span> <span class="badge badge-future">NOT YET ENABLED</span></summary>
<section>
<p class="desc">Real trades will be recorded here once <code>execution_paper_only = False</code> is set in config and
the system is connected to a live broker. Until then this section will always be empty.</p>
<h3>Executed orders</h3>
{real_trade_html}
<h3>Broker execution log — latest 50</h3>
<p class="desc">Low-level log from the execution module showing each order attempt, broker status codes,
fill prices, and any errors. Useful for diagnosing connectivity or slippage issues when live trading.</p>
{exec_html}
</section>
</details>"""

    pipeline_html = "<table><thead><tr><th>Log file</th><th>Last modified</th></tr></thead><tbody>"
    for name, mtime in pipeline:
        pipeline_html += f"<tr><td>{_html_escape(name)}</td><td>{_html_escape(mtime)}</td></tr>"
    pipeline_html += "</tbody></table>"

    freshness_html = "<table><thead><tr><th>File</th><th>Last modified</th></tr></thead><tbody>"
    for name, mtime in freshness:
        freshness_html += f"<tr><td>{_html_escape(name)}</td><td>{_html_escape(mtime)}</td></tr>"
    freshness_html += "</tbody></table>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="60">
<title>FreeWillyBot dashboard</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 1rem 2rem; background: #1a1a1a; color: #e0e0e0; }}
h1 {{ font-size: 1.5rem; margin-bottom: 0.25rem; }}
h2 {{ font-size: 1.1rem; margin-top: 2rem; margin-bottom: 0.3rem;
     border-bottom: 1px solid #444; padding-bottom: 0.25rem; }}
h3 {{ font-size: 1rem; margin-top: 1rem; margin-bottom: 0.25rem; color: #bbb; }}
h4 {{ font-size: 0.9rem; margin-top: 0.75rem; color: #aaa; }}
section {{ margin-bottom: 2rem; }}
table {{ border-collapse: collapse; font-size: 0.83rem; font-family: ui-monospace, monospace; }}
th, td {{ border: 1px solid #444; padding: 0.35rem 0.65rem; text-align: left; }}
th {{ background: #2a2a2a; }}
tr:nth-child(even) {{ background: #222; }}
pre {{ background: #222; padding: 0.75rem; overflow-x: auto; font-size: 0.8rem; border: 1px solid #444; }}
p {{ margin: 0.4rem 0; font-size: 0.9rem; }}
code {{ background: #2a2a2a; padding: 0.1rem 0.35rem; border-radius: 3px; font-size: 0.82rem; }}
.desc {{ color: #999; font-size: 0.83rem; margin: 0.2rem 0 0.6rem 0; line-height: 1.5; }}
.wf-note {{ border-left: 3px solid #4a6a8a; padding: 0.5rem 0.75rem; margin: 0.5rem 0 0.75rem 0; background: #1a1f24; font-size: 0.83rem; color: #ccc; line-height: 1.55; }}
.wf-note ul {{ margin: 0.35rem 0 0.35rem 1.1rem; }}
.wf-summary {{ margin: 0.35rem 0 0.75rem 0; font-size: 0.85rem; color: #ddd; }}
.wf-summary ul {{ margin: 0.25rem 0 0.35rem 1.1rem; padding: 0; }}
.wf-summary li {{ margin: 0.2rem 0; }}
.wf-table td {{ font-variant-numeric: tabular-nums; }}
.wf-details {{ margin-bottom: 1rem; }}
.signal-dist {{ font-family: ui-monospace; color: #8af; margin: 0.3rem 0 0.6rem 0; font-size: 0.85rem; }}
.badge {{ display: inline-block; padding: 0.15rem 0.55rem; border-radius: 3px;
          font-size: 0.72rem; font-weight: bold; vertical-align: middle;
          letter-spacing: 0.04em; margin-left: 0.4rem; }}
.badge-locked  {{ background: #1a4a1a; border: 1px solid #3a8a3a; color: #7dca7d; }}
.badge-sim     {{ background: #1a2a4a; border: 1px solid #3a5a9a; color: #7aacda; }}
.badge-demo    {{ background: #2a1a4a; border: 1px solid #6a3a9a; color: #b07ada; }}
.badge-live    {{ background: #4a1a1a; border: 1px solid #9a3a3a; color: #da7a7a; }}
.badge-future  {{ background: #333;    border: 1px solid #666;    color: #999; }}
.section-group {{ border-left: 3px solid #333; padding-left: 1rem; margin-bottom: 2rem; }}
details {{ margin-bottom: 1.5rem; border: 1px solid #333; border-radius: 4px; }}
details > summary {{
    padding: 0.6rem 0.9rem; cursor: pointer; font-size: 0.95rem;
    background: #222; border-radius: 4px; list-style: none; user-select: none;
}}
details > summary::-webkit-details-marker {{ display: none; }}
details > summary::before {{ content: "▶ "; font-size: 0.75rem; opacity: 0.6; }}
details[open] > summary::before {{ content: "▼ "; }}
details > summary:hover {{ background: #2a2a2a; }}
details[open] > summary {{ border-bottom: 1px solid #333; border-radius: 4px 4px 0 0; }}
details > section, details > .details-body {{ padding: 0.75rem 1rem; }}
.badge-active   {{ background: #1a3a1a; border: 1px solid #3a7a3a; color: #7aca7a; }}
.badge-technique {{ background: #2a2a1a; border: 1px solid #7a7a3a; color: #caca7a; font-weight: normal; }}
.strategy-card > summary {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem; }}
.strat-name {{ font-weight: bold; font-size: 0.95rem; margin-right: 0.3rem; }}
.strat-body {{ padding: 0.75rem 1rem; }}
.strat-desc {{ margin: 0.5rem 0 0.75rem 0; font-size: 0.9rem; line-height: 1.6; border-left: 3px solid #444; padding-left: 0.75rem; }}
.stat-bar {{ font-size: 0.82rem; font-family: ui-monospace, monospace; background: #1e1e1e; border: 1px solid #333; border-radius: 3px; padding: 0.4rem 0.7rem; margin-bottom: 0.75rem; flex-wrap: wrap; display: flex; gap: 0.1rem 0; line-height: 1.8; }}
.sig-dist {{ color: #8af; }}
.lag-warn {{ color: #f88; font-size: 0.78rem; }}
.lag-ok   {{ color: #8c8; font-size: 0.78rem; }}
.mono-desc {{ font-family: ui-monospace, monospace; font-size: 0.8rem; line-height: 1.7; color: #ccc; }}
.param-table {{ font-size: 0.8rem; }}
.param-table td:nth-child(2) {{ white-space: nowrap; }}
.inner-details {{ margin: 0.6rem 0; border: 1px solid #2a2a2a; border-radius: 3px; }}
.inner-details > summary {{ padding: 0.4rem 0.7rem; cursor: pointer; background: #1e1e1e; font-size: 0.85rem; color: #aaa; list-style: none; }}
.inner-details > summary::-webkit-details-marker {{ display: none; }}
.inner-details > summary::before {{ content: "▶ "; font-size: 0.7rem; opacity: 0.5; }}
.inner-details[open] > summary::before {{ content: "▼ "; }}
.inner-body {{ padding: 0.5rem 0.75rem; }}
.logo-header {{ text-align: center; margin-bottom: 0.5rem; }}
.dashboard-logo {{ max-width: 200px; height: auto; display: block; margin: 0 auto; }}
.table-filters {{ display: flex; flex-wrap: wrap; gap: 0.65rem 1rem; align-items: center;
                  margin: 0.5rem 0 0.85rem 0; padding: 0.5rem 0.65rem; background: #1e1e1e;
                  border: 1px solid #333; border-radius: 4px; }}
.filter-label {{ font-size: 0.82rem; color: #ccc; }}
.table-filters select, .table-filters input[type="search"] {{
    background: #2a2a2a; color: #e8e8e8; border: 1px solid #555; border-radius: 3px;
    padding: 0.28rem 0.45rem; font-size: 0.82rem; margin-left: 0.25rem; max-width: 14rem; }}
.filter-count {{ font-size: 0.78rem; color: #888; margin-left: 0.25rem; }}
.filterable-table {{ margin-top: 0.25rem; }}
.equity-table {{ margin: 0.4rem 0 0.85rem 0; width: 100%; max-width: 56rem; }}
.equity-table th, .equity-table td {{ font-variant-numeric: tabular-nums; }}
.siglog-collapse-bar {{ margin: 0.35rem 0 0.75rem 0; }}
.siglog-expand-btn {{ background: #2a2a2a; color: #e8e8e8; border: 1px solid #555; border-radius: 3px;
  padding: 0.35rem 0.75rem; cursor: pointer; font-size: 0.82rem; }}
.siglog-expand-btn:hover {{ background: #333; }}
</style>
</head>
<body>
<div class="logo-header">
  <img src="/static/logo.png" alt="Free Willy" class="dashboard-logo">
</div>
<h1>FreeWillyBot — trading dashboard</h1>
<p class="desc">{"<strong>Demo broker mode</strong> — orders are sent to the configured demo account (cTrader/OANDA/Binance); no real money at risk." if is_demo else "All strategies are currently running in <strong>simulation mode</strong> — no real money is at risk."}
Signals are generated every 2 minutes when <code>run_live_tick.py</code> runs.
{"" if is_demo else "Real-trade sections are shown below but will remain empty until live execution is enabled."}
<small style="opacity:0.5">(page auto-refreshes every 60 s)</small></p>

<!-- ═══════════════════════════════════════════════════════ STRATEGIES -->
<h2>Strategies</h2>
<p class="desc">{len(registry)} strategies registered · {sum(1 for s in registry if s.active)} active.
Each strategy runs independently on every tick; signals are tagged with a <em>strategy_id</em>.
Expand a card to see its technique, live parameters, and last-tick stats.</p>
{strategies_html}

{trade_freq_html}

<!-- ═══════════════════════════════════════════════════════ PORTFOLIO ENGINE -->
<h2>Portfolio engine</h2>
<p class="desc">Cross-strategy position permission and sizing layer.
Reads <code>portfolio_state.json</code> on every tick; gates each strategy before execution; adjusts lot size by mode.
State file: <code>data/logs/execution/portfolio_state.json</code>.</p>
<section>
{_format_portfolio_state(portfolio_state, portfolio_cfg)}
<details class="inner-details">
  <summary>Portfolio risk config (from config_portfolio.py)</summary>
  <div class="inner-body">
  <table><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>
  {''.join(f"<tr><td><code>{_html_escape(k)}</code></td><td>{_html_escape(str(v))}</td></tr>" for k,v in sorted(portfolio_cfg.items()) if not isinstance(v, dict))}
  </tbody></table>
  </div>
</details>
</section>

<!-- ═══════════════════════════════════════════════════════ EXECUTION -->
<h2>{"Demo broker" if is_demo else "Paper trading"} <span class="badge {"badge-demo" if is_demo else "badge-sim"}">{"DEMO" if is_demo else "SIMULATION"}</span></h2>
<p class="desc">Every time the live tick runs, each strategy evaluates the latest bar and records what it <em>would</em> do{"." if not is_demo else ", and sends an order to the demo broker account."}</p>

<section>
<h3>Signal log — newest {DASHBOARD_SIGNAL_LOG_CAP} of {DASHBOARD_PREDICTIONS_LIMIT} loaded</h3>
<p class="desc">Shows the latest {DASHBOARD_SIGNAL_LOG_CAP} rows (newest 10 visible by default; use the button for older rows in that window). One row per strategy per tick. <strong>timestamp</strong> is the <em>bar close time</em> in the data (not necessarily &quot;now&quot;).
<strong>bar_lag_hours</strong> = hours between that bar and when the tick ran (large = stale features — run data refresh).
<strong>signal_source</strong>: <code>test_csv_tail</code> = classifier last bar; <code>regression_features_tail</code> = regression last bar + live model prediction; <code>replay_predictions</code> only if <code>REGRESSION_LIVE_USE_FEATURE_TAIL = False</code> in config.
<strong>run_at</strong> = wall-clock UTC when the job ran.
<strong>readiness_0_100</strong> = rough how close to a trade (100 = this bar fired an open/close/reverse).
<strong>trade_hint</strong> = plain-English: what is missing (confidence, vol, extreme pred, risk pause, etc.).
Use the filters below to narrow by strategy, signal, blocked flag, or text search.</p>
<p class="signal-dist">{signal_dist}</p>
{pred_html}
</section>

{demo_block_html}
{paper_block_html}
{real_trades_block_html}

<!-- ═══════════════════════════════════════════════════════ BACKTEST VALIDATION -->
<h2>Backtest &amp; validation</h2>
<p class="desc">Historical results used to validate the strategies before paper trading.
These are fixed — they don't update unless you re-run the validation scripts.</p>

<section>
<h3>Walk-forward validation (regression_v1)</h3>
<p class="desc">Rolling windows the model never trained on — good for checking whether an edge holds across time.
Each row is one window; <strong>net_return</strong> is after costs for that window only. Use the expandable note below to compare this to the full-period headline in <code>config.py</code>.</p>
{wf_block_html}
</section>

<section>
<h3>Cost stress test (regression_v1)</h3>
<p class="desc">Shows how the strategy holds up when trading costs are artificially increased (1×, 1.5×, 2×).
A strategy that only works at unrealistically low costs is fragile. We want the edge to survive at 2× costs.</p>
{cost_stress_html}
</section>

<section>
<h3>Backtest report (classifier_v1, latest run)</h3>
<p class="desc">Summary metrics from the most recent full backtest run on the classifier strategy.
<strong>Profit factor</strong>: total gross profit divided by total gross loss — above 1.0 means net positive.
<strong>Sharpe</strong>: return per unit of risk. Baselines show how a simple momentum or flat strategy would have done.</p>
{bt_html if bt_html else "<p>No backtest report found — run the backtest script to generate one.</p>"}
</section>

<section>
<h3>Validation report (classifier_v1)</h3>
<p class="desc">Per-metric summary from the model validation run — shows whether the classifier signal is statistically
meaningful (e.g. directional accuracy, calibration, edge after costs).</p>
{val_html}
</section>

<section>
<h3>Regime report (classifier_v1)</h3>
<p class="desc">Breaks down performance by market regime (e.g. trending vs choppy, high vs low volatility).
Shows whether the strategy's edge is consistent across different market conditions.</p>
{regime_html}
</section>

<!-- ═══════════════════════════════════════════════════════ MODEL & DATA -->
<h2>Model &amp; data</h2>

<section>
<h3>Data diagnostics</h3>
<p class="desc">Checks run on the raw price and feature data — missing bars, outliers, alignment issues.
Any warnings here mean the input data may affect signal quality.</p>
{diag_html}
</section>

<section>
<h3>Model config</h3>
<p class="desc">Technical settings used when training the models — which algorithm was chosen and why,
and the hyperparameters from the best training run.</p>
<h4>Baseline choice</h4>
{base_html}
<h4>Meta config</h4>
{meta_html}
</section>

<section>
<h3>Pipeline status</h3>
<p class="desc">Shows when each automated job last ran, based on log file timestamps.
If a file is stale, the corresponding pipeline step may not be running correctly on its schedule.</p>
{pipeline_html}
</section>

<section>
<h3>Feature freshness</h3>
<p class="desc">Key data files and when they were last updated. The price data and features should refresh daily.
If these timestamps are old, signals may be based on stale data.</p>
{freshness_html}
</section>

{_dashboard_table_filter_script()}
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="FreeWillyBot dashboard webserver")
    parser.add_argument("--port", type=int, default=5050, help="Port to bind (default: 5050)")
    args = parser.parse_args()

    try:
        from flask import Flask
    except ModuleNotFoundError:
        sys.exit(
            "Flask not installed. Install dependencies and run with the project venv:\n"
            "  .venv/bin/pip install -r requirements.txt\n"
            "  .venv/bin/python scripts/run_dashboard.py\n"
            "Or: ./scripts/run_dashboard.sh"
        )

    app = Flask(
        __name__,
        static_folder=str(PROJECT_ROOT / "static"),
        static_url_path="/static",
    )

    @app.route("/")
    def index() -> str:
        return build_html(PROJECT_ROOT)

    port = args.port
    print(f"Dashboard at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
