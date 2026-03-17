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


def load_predictions_live(root: Path) -> tuple[list[dict], str | None]:
    """Latest 50 rows from predictions_live.csv."""
    path = root / "data" / "predictions" / "predictions_live.csv"
    return _load_csv(path, limit=50)


def load_trade_decisions(root: Path) -> tuple[list[dict], str | None]:
    """All rows from trade_decisions.csv."""
    path = root / "data" / "logs" / "execution" / "trade_decisions.csv"
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

    # Signal distribution from predictions
    signal_counts = {"BUY": 0, "SELL": 0, "FLAT": 0}
    for r in pred_rows:
        s = str(r.get("signal", "")).upper()
        if s in signal_counts:
            signal_counts[s] += 1

    config_html = ""
    if "error" in cfg:
        config_html = f"<p>Error: {_html_escape(cfg['error'])}</p>"
    else:
        config_html = "<table><tbody>"
        for k, v in cfg.items():
            config_html += f"<tr><td>{_html_escape(k)}</td><td>{_html_escape(str(v))}</td></tr>"
        config_html += "</tbody></table>"

    pred_html = pred_err if pred_err else _render_table(pred_rows)
    trade_html = trade_err if trade_err else _render_table(trade_rows)
    exec_html = exec_err if exec_err else _render_table(exec_rows)
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

    pipeline_html = "<table><thead><tr><th>Log file</th><th>Last modified</th></tr></thead><tbody>"
    for name, mtime in pipeline:
        pipeline_html += f"<tr><td>{_html_escape(name)}</td><td>{_html_escape(mtime)}</td></tr>"
    pipeline_html += "</tbody></table>"

    freshness_html = "<table><thead><tr><th>File</th><th>Last modified</th></tr></thead><tbody>"
    for name, mtime in freshness:
        freshness_html += f"<tr><td>{_html_escape(name)}</td><td>{_html_escape(mtime)}</td></tr>"
    freshness_html += "</tbody></table>"

    signal_dist = f"BUY: {signal_counts['BUY']} | SELL: {signal_counts['SELL']} | FLAT: {signal_counts['FLAT']}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FreeWillyBot dashboard</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 1rem 2rem; background: #1a1a1a; color: #e0e0e0; }}
h1 {{ font-size: 1.5rem; margin-bottom: 0.5rem; }}
h2 {{ font-size: 1.2rem; margin-top: 1.5rem; margin-bottom: 0.5rem; border-bottom: 1px solid #444; padding-bottom: 0.25rem; }}
h4 {{ font-size: 1rem; margin-top: 0.75rem; }}
section {{ margin-bottom: 1.5rem; }}
table {{ border-collapse: collapse; font-size: 0.85rem; font-family: ui-monospace, monospace; }}
th, td {{ border: 1px solid #444; padding: 0.35rem 0.6rem; text-align: left; }}
th {{ background: #2a2a2a; }}
tr:nth-child(even) {{ background: #222; }}
pre {{ background: #222; padding: 0.75rem; overflow-x: auto; font-size: 0.8rem; border: 1px solid #444; }}
p {{ margin: 0.5rem 0; }}
.signal-dist {{ font-family: ui-monospace; color: #8af; margin-bottom: 0.5rem; }}
</style>
</head>
<body>
<h1>FreeWillyBot dashboard</h1>

<section>
<h2>Config summary</h2>
{config_html}
</section>

<section>
<h2>Live signals (latest 50)</h2>
<p class="signal-dist">{signal_dist}</p>
{pred_html}
</section>

<section>
<h2>Trade decisions</h2>
{trade_html}
</section>

<section>
<h2>Execution log (latest 50)</h2>
{exec_html}
</section>

<section>
<h2>Backtest (latest)</h2>
{bt_html}
</section>

<section>
<h2>Validation report</h2>
{val_html}
</section>

<section>
<h2>Regime report</h2>
{regime_html}
</section>

<section>
<h2>Data diagnostics</h2>
{diag_html}
</section>

<section>
<h2>Model config</h2>
<h4>Baseline choice</h4>
{base_html}
<h4>Meta config</h4>
{meta_html}
</section>

<section>
<h2>Pipeline status</h2>
{pipeline_html}
</section>

<section>
<h2>Feature freshness</h2>
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
