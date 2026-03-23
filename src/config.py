"""
Configuration for FreeWillyBot price forecasting.

Symbol: EUR/USD (5 min bars)
Forecast horizon: 6 bars ahead
Data source: Dukascopy historical data for FX research
"""

from datetime import date

# Instrument: EURUSD or BTCUSDT
SYMBOL = "EURUSD"

# Bar interval (e.g. "5min", "1h", "1d")
BAR_INTERVAL = "5min"

# Forecast horizon in number of bars
FORECAST_HORIZON = 6

# Training period start date
TRAINING_START_DATE = date(2022, 1, 1)

# Validation period start date (between train and test)
VALIDATION_START_DATE = date(2023, 10, 1)

# Test period start date (validation/test data begins here)
TEST_START_DATE = date(2024, 1, 1)

# Label horizon (bars ahead for future_return_30m; 6 bars = 30 min at 5min)
LABEL_HORIZON_BARS = 6

# Buy/sell class thresholds (as decimal, e.g. 0.0003 = 0.03%)
BUY_THRESHOLD_PCT = 0.0003
SELL_THRESHOLD_PCT = -0.0003

# Use news + FinBERT sentiment when news files exist (temporarily disabled)
USE_NEWS = False

# Phase 17 — Ablation: False = price + time only (no cross-asset, no macro)
USE_EXOGENOUS = True

# Data source: Dukascopy (EUR/USD), Binance (BTC/USDT), or yfinance (fallback)
DATA_SOURCE = "dukascopy"  # "dukascopy" | "binance" | "yfinance"

# Phase 11 — Training
NO_TRADE_THRESHOLD_PCT = 0.60  # min prob to trade (model max ~0.54; 0.55+ = 0 trades)
FORECASTER_CONTEXT_BARS = 256
FORECASTER_HORIZON_BARS = 6  # same as LABEL_HORIZON_BARS

# Phase 12 — Backtest costs
FX_SPREAD_PIPS = 1.0  # 1 pip = 0.0001 for EURUSD
CRYPTO_FEE_PCT = 0.001  # 0.1% per side
CRYPTO_SLIPPAGE_PCT = 0.0005  # 0.05% per side

# Phase 13 — Safety filters
MACRO_EVENT_BLACKOUT_MIN = 30  # no trades within +/- N min of major events
SPREAD_PROXY_VOLATILITY_PCT = 0.003  # no trades when volatility_20 > this (legacy; unused when VOL_REGIME_TOP_PCT set)
VOL_REGIME_TOP_PCT: int | None = 30  # trade only when volatility_20 in top X%; None = use SPREAD_PROXY
MIN_CONFIDENCE_PCT = 0.60  # weak confidence filter (same as NO_TRADE_THRESHOLD)
CONFIDENCE_MARGIN_PCT = 0.10  # only trade when |P_buy - P_sell| > this (stronger no-trade zone)
CONFIDENCE_TOP_PCT: int | None = None  # when set (10, 20, 30), use percentile filter; when None, use threshold
MAX_CONCURRENT_POSITIONS = 1  # single-asset: always 1
MAX_DAILY_LOSS_PCT = 0.02  # stop trading for day if down 2%
MAX_POSITION_SIZE = 1.0  # units; 1 = full size
COOLDOWN_BARS_AFTER_LOSS = 12  # wait 12 bars (1h) after closing a losing trade
CRYPTO_SKIP_WEEKEND = True  # no crypto trades Sat/Sun
SESSION_EXCLUDE_HOURS = []  # e.g. [0,1,2,3,4,5,6,7,22,23] to exclude low-liquidity

# Phase 15 — Execution (paper/demo only)
EXECUTION_PAPER_ONLY = True  # enforce no live execution
OANDA_INSTRUMENT = "EUR_USD"  # OANDA format for EURUSD
BINANCE_SYMBOL = "BTCUSDT"  # Binance format
EXECUTION_TEST_UNITS = 100  # tiny size for test orders (OANDA: 100 units = 0.01 lot)

# cTrader (Pepperstone demo) execution
CTRADER_HOST_TYPE = "demo"           # "demo" or "live"
CTRADER_EURUSD_SYMBOL_ID = 1        # cTrader symbol ID for EURUSD; confirm via ProtoOASymbolsListReq
CTRADER_VOLUME_LOTS: float = 1.0    # lot size; 1.0 = 1 standard lot = 100 000 base currency units (minimum for this account)

# Phase 16 — Automation
PREDICTIONS_LIVE_CSV = "data/predictions/predictions_live.csv"
TRADE_DECISIONS_CSV = "data/logs/execution/trade_decisions.csv"
# Livetick: if Mac sleeps/ is off, launchd misses runs. Next successful tick writes this file.
LIVETICK_HEARTBEAT_JSON = "data/logs/execution/livetick_heartbeat.json"
# If no fully successful tick in this many minutes, next run adds data refresh (catch up bars/features).
LIVETICK_STALE_MINUTES: int = 45
# Do not auto-run another data refresh sooner than this (avoids hammering if strategies fail after refresh).
LIVETICK_AUTO_REFRESH_COOLDOWN_MINUTES: int = 25
# Trigger an intraday data refresh when the regression feature tail is older than this many minutes.
# Features are typically 60 min behind price (target horizon = 12 bars * 5 min) so 90 min is a
# reasonable staleness threshold — fresh enough for intraday signals without over-refreshing.
LIVETICK_FEATURE_STALE_MINUTES: int = 90

# Phase 18 — Data quality
MAX_BAR_RETURN_PCT = 0.05  # Flag bars with abs(return) > 5% as impossible jump

# Mean reversion strategy v1 — rule-based, no ML
MR_ZSCORE_THRESHOLD: float = 2.0   # |z| above this triggers entry
MR_HOLD_BARS: int = 6              # bars to hold before forced close (6 bars = 30 min)
MR_LOOKBACK_BARS: int = 50         # window for rolling mean/std of ma_20_gap
MR_MAX_BAR_AGE_MINUTES: int = 20   # refuse to trade if bar is older than this (stale data guard)
MR_VOL_PCT: int = 20               # trade only when vol_6 in top 20% (0 = disabled)
MR_KILL_SWITCH_N: int = 20         # evaluate rolling PF after every 20 completed trades
MR_KILL_SWITCH_PF: float = 0.9    # pause if rolling PF of last N trades < this
MR_DD_KILL: float = 0.02           # pause when drawdown from equity peak exceeds 2%
MR_PAUSE_BARS: int = 72            # resume after 72 bars (= 6 hours at 5-min bars)

# Stale bar guard for classifier_v1 and regression_v1
CLASSIFIER_MAX_BAR_AGE_MINUTES: int = 20
REGRESSION_MAX_BAR_AGE_MINUTES: int = 20

# Regression experiment (Batch 1) — multi-horizon return targets, no Chronos/TimesFM
REGRESSION_HORIZONS = [3, 6, 12]  # bars ahead for target_ret_h

# Batch 2 — Feature set: "core" (minimal) | "full" (Batch 1 classifier-style)
FEATURE_SET = "core"

# Live regression signal: use latest bar(s) from features_regression_core/test (+ model predict).
# False = replay test_predictions.parquet from cursor (historical bar times in logs).
REGRESSION_LIVE_USE_FEATURE_TAIL: bool = True

# ─────────────────────────────────────────────────────────────────────────────
# Production config — regression strategy (locked after final validation)
# Validated on out-of-sample test period Jan–Dec 2024.
# Full-period: net +5.4%, PF 1.34, max DD 3.9%, 721 trades.
# With KS+DD risk controls: net sum flips positive on walk-forward,
#   worst month -1.67%, max DD 2.01%.
# ─────────────────────────────────────────────────────────────────────────────

# Selection
REGRESSION_TOP_PCT: float = 0.25         # trade top/bottom 0.25% of predictions
REGRESSION_VOL_PCT: int = 20             # trade only when vol_6 in top 20%
REGRESSION_PRED_THRESHOLD: float = 0.00005  # |pred| must exceed this to trade

# Risk controls — kill switch (trade-based)
REGRESSION_KILL_SWITCH_N: int = 20       # evaluate rolling PF after every 20 completed trades
REGRESSION_KILL_SWITCH_PF: float = 0.9  # pause if rolling PF of last 20 trades < 0.9

# Risk controls — drawdown kill
REGRESSION_DD_KILL: float = 0.02        # pause when drawdown from equity peak exceeds 2%
REGRESSION_PAUSE_BARS: int = 72         # resume after 72 bars (= 6 hours at 5-min bars)
