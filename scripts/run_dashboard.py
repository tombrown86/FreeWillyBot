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
import json
import sys
from datetime import datetime
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


def load_predictions_live(root: Path) -> tuple[list[dict], str | None]:
    """Latest 50 rows from predictions_live.csv."""
    path = root / "data" / "predictions" / "predictions_live.csv"
    return _load_csv(path, limit=50)


def load_trade_decisions(root: Path) -> tuple[list[dict], str | None]:
    """All rows from trade_decisions.csv. Missing file = no decisions yet (not an error)."""
    path = root / "data" / "logs" / "execution" / "trade_decisions.csv"
    if not path.exists():
        return [], None
    return _load_csv(path)


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
    pred_rows, pred_err = load_predictions_live(root)
    trade_rows, trade_err = load_trade_decisions(root)
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

    # ── classifier config ────────────────────────────────────────────────────
    config_html = ""
    if "error" in cfg:
        config_html = f"<p>Error: {_html_escape(cfg['error'])}</p>"
    else:
        CLASSIFIER_LABELS = {
            "symbol":               "Trading instrument (Forex pair)",
            "bar_interval":         "Candle size — each bar covers this time window",
            "no_trade_threshold":   "Skip signals with very weak predicted moves (% of ATR)",
            "min_confidence_pct":   "Minimum model confidence required before entering a trade",
            "max_daily_loss_pct":   "Daily loss limit — bot stops trading after hitting this drawdown",
            "cooldown_bars":        "Number of bars to wait after a losing trade before re-entering",
            "execution_paper_only": "When True the bot never sends real orders — simulation only",
        }
        config_html = "<table><thead><tr><th>Setting</th><th>Value</th><th>What it means</th></tr></thead><tbody>"
        for k, v in cfg.items():
            label = CLASSIFIER_LABELS.get(k, k)
            config_html += f"<tr><td>{_html_escape(k)}</td><td><strong>{_html_escape(str(v))}</strong></td><td>{_html_escape(label)}</td></tr>"
        config_html += "</tbody></table>"

    # ── regression production config ─────────────────────────────────────────
    LABEL_MAP = {
        "top_pct":         ("Only trade when the prediction is in the top/bottom N% of all predictions — extreme confidence filter", "%"),
        "vol_pct":         ("Only trade when market volatility is in the top N% — avoids quiet/flat markets", "%"),
        "pred_threshold":  ("Ignore predictions smaller than this value — removes near-zero noise signals", ""),
        "kill_switch_n":   ("After this many completed trades, evaluate recent performance", "trades"),
        "kill_switch_pf":  ("If the profit factor of the last N trades drops below this, pause trading", ""),
        "dd_kill":         ("If the account drops more than this % from its recent peak, pause trading", "fraction"),
        "pause_bars":      ("How long to stay paused after a kill-switch fires (72 bars = 6 hours at 5-min)", "bars"),
    }
    if "error" in reg_cfg:
        reg_config_html = f"<p>Error: {_html_escape(reg_cfg['error'])}</p>"
    else:
        reg_config_html = "<table><thead><tr><th>Parameter</th><th>Value</th><th>What it does</th></tr></thead><tbody>"
        for k, v in reg_cfg.items():
            label, unit = LABEL_MAP.get(k, (k, ""))
            display = f"{v} {unit}".strip() if unit else str(v)
            reg_config_html += f"<tr><td>{_html_escape(k)}</td><td><strong>{_html_escape(display)}</strong></td><td>{_html_escape(label)}</td></tr>"
        reg_config_html += "</tbody></table>"

    # ── walk-forward summary ─────────────────────────────────────────────────
    def _wf_summary(rows: list[dict]) -> str:
        if not rows:
            return "<p>No data</p>"
        try:
            nets = [float(r["net_return"]) for r in rows]
            n_pos = sum(1 for n in nets if n > 0)
            worst = min(nets)
            total = sum(nets)
            return (
                f"<p><strong>{n_pos}/{len(nets)} windows positive</strong> &nbsp;|&nbsp; "
                f"Net sum: {total:.4f} &nbsp;|&nbsp; "
                f"Worst window: {worst:.4f}</p>"
            )
        except Exception:
            return ""

    wf_1m_html = _wf_summary(wf_1m) + (_render_table(wf_1m) if wf_1m else "")
    wf_2m_html = _wf_summary(wf_2m) + (_render_table(wf_2m) if wf_2m else "")
    cost_stress_html = _render_table(cost_stress) if cost_stress else "<p>No data — run scripts/run_cost_stress_regression.py</p>"

    # ── per-strategy signal counts ───────────────────────────────────────────
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
    sim_trade_rows = [r for r in trade_rows if str(r.get("strategy_id", "")).endswith("_v1") or r.get("mode", "sim") != "live"]
    real_trade_rows = [r for r in trade_rows if r.get("mode", "") == "live"]

    sim_trade_html = trade_err if trade_err else (
        _render_table(sim_trade_rows) if sim_trade_rows else
        "<p>No simulated trades yet — they will appear here once the live tick runs and a strategy fires a BUY/SELL signal.</p>"
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

    # ── pred signal table ─────────────────────────────────────────────────────
    pred_html = pred_err if pred_err else (
        _render_table(pred_rows) if pred_rows else
        "<p>No signals yet — run <code>python scripts/run_live_tick.py</code> to generate the first bar.</p>"
    )

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
.signal-dist {{ font-family: ui-monospace; color: #8af; margin: 0.3rem 0 0.6rem 0; font-size: 0.85rem; }}
.badge {{ display: inline-block; padding: 0.15rem 0.55rem; border-radius: 3px;
          font-size: 0.72rem; font-weight: bold; vertical-align: middle;
          letter-spacing: 0.04em; margin-left: 0.4rem; }}
.badge-locked  {{ background: #1a4a1a; border: 1px solid #3a8a3a; color: #7dca7d; }}
.badge-sim     {{ background: #1a2a4a; border: 1px solid #3a5a9a; color: #7aacda; }}
.badge-live    {{ background: #4a1a1a; border: 1px solid #9a3a3a; color: #da7a7a; }}
.badge-future  {{ background: #333;    border: 1px solid #666;    color: #999; }}
.section-group {{ border-left: 3px solid #333; padding-left: 1rem; margin-bottom: 2rem; }}
</style>
</head>
<body>
<h1>FreeWillyBot — trading dashboard</h1>
<p class="desc">All strategies are currently running in <strong>simulation mode</strong> — no real money is at risk.
Signals are generated every 5 minutes when <code>run_live_tick.py</code> runs.
Real-trade sections are shown below but will remain empty until live execution is enabled.</p>

<!-- ═══════════════════════════════════════════════════════ STRATEGIES -->
<h2>Strategies</h2>
<p class="desc">The bot currently has one validated strategy. More will be added over time for comparison.
Each strategy runs independently and logs its signals tagged with a <em>strategy_id</em>.</p>

<div class="section-group">
<section>
<h3>Regression strategy (regression_v1) <span class="badge badge-locked">LOCKED CONFIG</span> <span class="badge badge-sim">SIMULATION</span></h3>
<p class="desc">A machine-learning regression model trained on 5-minute EURUSD bars.
It predicts the return 6 bars (30 min) ahead and only trades when the signal is in the extreme top or bottom
of all predictions — so most bars produce no signal.
Two risk controls (kill switch + drawdown limit) pause the strategy during bad patches and resume automatically.
The parameters below were finalised after out-of-sample validation on Jan–Dec 2024 and must not be changed without a new validation run.</p>
{reg_config_html}
</section>

<section>
<h3>Classifier strategy (classifier_v1) <span class="badge badge-sim">SIMULATION</span></h3>
<p class="desc">A binary classification model that outputs a BUY or SELL probability for each bar.
Signals are filtered by minimum confidence, volatility regime, session hours, and a daily loss limit.
Config parameters below control those filters.</p>
{config_html}
</section>
</div>

<!-- ═══════════════════════════════════════════════════════ PAPER TRADING -->
<h2>Paper trading <span class="badge badge-sim">SIMULATION</span></h2>
<p class="desc">Every time the live tick runs, each strategy evaluates the latest bar and records what it <em>would</em> do.
Nothing is sent to a broker — this is purely to track signal quality and build a track record before going live.</p>

<section>
<h3>Signal log — latest 50 bars</h3>
<p class="desc">One row per strategy per tick. <strong>signal</strong>: BUY / SELL / FLAT (the raw model output).
<strong>action</strong>: what the strategy decided to do with its position (OPEN_LONG, OPEN_SHORT, CLOSE, NONE).
<strong>blocked</strong>: 1 if a risk filter stopped the trade. <strong>reason</strong>: which filter fired.
<strong>pred</strong>: the raw model prediction value. <strong>confidence</strong>: absolute prediction magnitude.</p>
<p class="signal-dist">{signal_dist}</p>
{pred_html}
</section>

<section>
<h3>Simulated trade decisions <span class="badge badge-sim">SIMULATION</span></h3>
<p class="desc">Records the trade action each strategy <em>would have executed</em> on a real broker.
These are generated when the tick script runs with <code>--execute</code>.
Columns: <strong>strategy_id</strong> — which strategy fired.
<strong>signal</strong> — BUY/SELL/FLAT from the model.
<strong>action_taken</strong> — the order action (e.g. OPEN_LONG).
<strong>broker_response</strong> — simulated broker reply (dry run).</p>
{sim_trade_html}
</section>

<!-- ═══════════════════════════════════════════════════════ REAL TRADES (future) -->
<h2>Real trades <span class="badge badge-live">LIVE</span> <span class="badge badge-future">NOT YET ENABLED</span></h2>
<p class="desc">Real trades will be recorded here once <code>execution_paper_only = False</code> is set in config and
the system is connected to a live broker. Until then this section will always be empty.
The table structure will be identical to simulated trades so they can be compared side by side.</p>

<section>
<h3>Executed orders</h3>
{real_trade_html}
</section>

<section>
<h3>Broker execution log — latest 50</h3>
<p class="desc">Low-level log from the execution module showing each order attempt, broker status codes,
fill prices, and any errors. Useful for diagnosing connectivity or slippage issues when live trading.</p>
{exec_html}
</section>

<!-- ═══════════════════════════════════════════════════════ BACKTEST VALIDATION -->
<h2>Backtest &amp; validation</h2>
<p class="desc">Historical results used to validate the strategies before paper trading.
These are fixed — they don't update unless you re-run the validation scripts.</p>

<section>
<h3>Walk-forward validation (regression_v1)</h3>
<p class="desc">The strategy was tested on rolling time windows it never trained on — the most honest way to measure robustness.
Each row is one independent test window. <strong>net_return</strong> is the return after costs.
A high fraction of positive windows means the edge is consistent across time, not just a lucky period.</p>
<h4>12 × 1-month windows</h4>
{wf_1m_html}
<h4>6 × 2-month windows</h4>
{wf_2m_html}
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

    app = Flask(__name__)

    @app.route("/")
    def index() -> str:
        return build_html(PROJECT_ROOT)

    port = args.port
    print(f"Dashboard at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
