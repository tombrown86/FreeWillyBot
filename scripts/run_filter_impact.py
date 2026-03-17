"""
Measure filter impact: bars blocked, trades prevented, PnL improvement per filter.

Runs backtest with each filter disabled one at a time and compares to full filtered run.
Output: data/validation/filter_impact_report.json
"""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"


def _run_with_overrides(overrides: dict[str, bool] | None) -> tuple[dict, dict]:
    """Run backtest with filter overrides. Returns (metrics, filter_stats)."""
    from src.backtest import run

    result = run(
        use_frozen=True,
        top_pct=5,
        filter_overrides=overrides,
        return_only=True,
    )
    if result is None:
        raise RuntimeError("Backtest failed")
    metrics, filter_stats, _ = result
    return metrics, filter_stats


def run() -> int:
    """Run filter impact analysis. Returns 0 on success."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    # Full filtered (baseline)
    logging.info("Running full filtered backtest...")
    metrics_full, stats_full = _run_with_overrides(None)

    # Unfiltered (all filters disabled)
    all_off = {
        "vol_ok": True,
        "macro_ok": True,
        "confidence_ok": True,
        "session_ok": True,
        "weekend_ok": True,
        "cooldown": True,
        "daily_loss": True,
    }
    logging.info("Running unfiltered backtest...")
    metrics_unfiltered, _ = _run_with_overrides(all_off)

    # Per-filter: disable one at a time
    filters = [
        ("vol_ok", "vol_regime"),
        ("macro_ok", "macro_blackout"),
        ("cooldown", "cooldown_after_loss"),
        ("daily_loss", "max_daily_loss"),
    ]
    # confidence_ok, session_ok, weekend_ok may have 0 blocks for EURUSD
    for key, _ in [("confidence_ok", "weak_confidence"), ("session_ok", "session"), ("weekend_ok", "weekend")]:
        filters.append((key, key))

    report = {
        "full_filtered": {
            "cum_return": metrics_full["cumulative_return"],
            "max_drawdown": metrics_full["max_drawdown"],
            "profit_factor": metrics_full["profit_factor"],
            "num_trades": metrics_full["num_trades"],
        },
        "unfiltered": {
            "cum_return": metrics_unfiltered["cumulative_return"],
            "max_drawdown": metrics_unfiltered["max_drawdown"],
            "profit_factor": metrics_unfiltered["profit_factor"],
            "num_trades": metrics_unfiltered["num_trades"],
        },
        "filter_stats_full": stats_full,
        "per_filter": {},
    }

    for override_key, filter_name in filters:
        overrides = {override_key: True}
        logging.info("Running with %s disabled...", filter_name)
        metrics_no_f, stats_no_f = _run_with_overrides(overrides)

        stat_key = {
            "vol_ok": "spread_proxy_bars",
            "macro_ok": "macro_blackout_bars",
            "confidence_ok": "weak_confidence_bars",
            "session_ok": "session_exclude_bars",
            "weekend_ok": "weekend_bars",
            "cooldown": "cooldown_bars",
            "daily_loss": "max_daily_loss_bars",
        }.get(override_key, "unknown")

        bars_blocked = stats_full.get(stat_key, 0)
        pnl_with = metrics_full["cumulative_return"]
        pnl_without = metrics_no_f["cumulative_return"]
        pnl_improvement = pnl_with - pnl_without  # positive = filter helps

        report["per_filter"][filter_name] = {
            "bars_blocked": bars_blocked,
            "trades_with_filter": metrics_full["num_trades"],
            "trades_without_filter": metrics_no_f["num_trades"],
            "trades_prevented": metrics_no_f["num_trades"] - metrics_full["num_trades"],
            "cum_return_without_filter": pnl_without,
            "pnl_improvement": pnl_improvement,
        }

    out_path = VALIDATION_DIR / "filter_impact_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    logging.info("Filter impact report saved to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(run())
