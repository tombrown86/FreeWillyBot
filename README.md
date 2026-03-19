# FreeWillyBot

Research and paper-trading stack for **5-minute EUR/USD** (configurable). It trains forecasting models, backtests strategies with costs and risk controls, and runs **multi-strategy paper signals** with a local web dashboard. **No live money** is used by default (`EXECUTION_PAPER_ONLY = True`).

---

## What’s in the box

| Piece | Description |
|-------|-------------|
| **Classifier strategy** (`classifier_v1`) | XGBoost-style directional model with session, volatility, and confidence filters. |
| **Regression strategy** (`regression_v1`) | Return forecast + extreme selection (top/bottom predictions, vol filter) + **kill switch** (rolling profit factor) + **drawdown pause**. Production parameters are locked in `src/config.py` under `REGRESSION_*`. |
| **Paper trading** | `scripts/run_live_tick.py` runs every strategy each tick: signals → `predictions_live.csv`. **Paper execution is on by default**: per-strategy simulated position + equity (bar-by-bar returns), logged to `trade_decisions.csv` and `paper_simulation.csv`. Use `--demo-broker` (or `RUN_LIVETICK_DEMO_BROKER=1`) to send real orders to the configured demo broker (cTrader/OANDA/Binance). Use `--no-execute` for signals only. |
| **Dashboard** | Flask app: metrics, walk-forward tables, cost stress, live signal log, simulated vs (future) real trade sections. |

Data pipeline: Dukascopy (and optional cross-asset / macro features). See `src/config.py` for symbol, bar size, train/validation/test dates, and execution flags.

---

## Requirements

- **Python 3.11+** recommended (some optional deps may differ on 3.12).
- Create a venv and install:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional: copy `.env.example` to `.env` if you use FRED or broker APIs (see project docs / `python-dotenv` usage).

---

## Quick start

```bash
source .venv/bin/activate

# One-off: refresh prices & features (see scripts/run_daily_data_refresh.py)
python -m scripts.run_daily_data_refresh

# Paper tick: signals + simulated trades & equity (default)
python scripts/run_live_tick.py

# If the Mac was asleep/off, the next run auto-runs `run_daily_data_refresh` when the heartbeat is stale
# (see `LIVETICK_STALE_MINUTES` in `src/config.py`). Use `--no-auto-refresh` to skip that.

# Signals only (no trade_decisions / equity file)
python scripts/run_live_tick.py --no-execute

# Demo broker: send real orders to cTrader/OANDA/Binance demo (after paper looks good)
python scripts/run_live_tick.py --demo-broker
# Or set RUN_LIVETICK_DEMO_BROKER=1 in .env so launchd uses demo broker automatically.

# Web dashboard (default http://localhost:5050)
python scripts/run_dashboard.py
# or: ./scripts/run_dashboard.sh
```

---

## Project layout (high level)

```
data/
  processed/          # Cleaned prices, aligned series
  features/           # Classifier feature tables
  features_regression_core/   # Regression features
  models/             # Saved models, feature lists, configs
  predictions/        # Live/paper signal CSV
  predictions_regression/   # e.g. test_predictions.parquet
  backtests/          # Classifier backtest JSON reports
  backtests_regression/     # Walk-forward, grid, cost stress CSVs
  logs/               # Execution, cron logs
scripts/              # Entry points (refresh, train, backtest, dashboard, …)
src/                  # Core logic (features, training, backtest, live_signal*, execution)
```

---

## Useful scripts

| Script | Purpose |
|--------|---------|
| `run_daily_data_refresh.py` | Update price data; rebuild classifier features + **regression core** features (so live regression `timestamp` stays current). |
| `run_daily_retrain.py` | Scheduled retrain path for classifier pipeline. |
| `run_train_regression.py` | Train regression model. |
| `predict_regression_test.py` | Build test-set predictions (e.g. for regression paper replay). |
| `run_backtest_regression.py` | Regression grid / single backtest. |
| `run_walk_forward_regression.py` | Rolling walk-forward on regression strategy. |
| `run_cost_stress_regression.py` | Cost multiplier stress test. |
| `run_validate.py` / `run_full_validate.py` | Classifier validation reports. |
| `run_data_diagnostics.py` | Data quality checks. |

Adding a new strategy: implement `run(n_bars=1)` returning signal dicts, then register it in `STRATEGIES` inside `scripts/run_live_tick.py`.

---

## Configuration

All shared settings live in **`src/config.py`**: symbol, bar interval, train/test windows, spread/cost assumptions, classifier filters, and **locked regression production parameters** (`REGRESSION_TOP_PCT`, `REGRESSION_VOL_PCT`, `REGRESSION_PRED_THRESHOLD`, kill switch and drawdown settings).

---

## Disclaimer

This is **research software**, not financial advice. Forex trading is risky. The default configuration does not place live orders; enabling real execution requires explicit config changes and your own broker compliance. Use at your own risk.

