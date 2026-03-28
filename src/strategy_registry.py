"""
Strategy registry — single source of truth for every strategy in FreeWillyBot.

Each entry in STRATEGIES describes one strategy completely:
  - identity, status, technique
  - the config keys it reads (pulled live from src.config)
  - human-readable labels and explanations for every parameter
  - a plain-English description for newcomers
  - a technical summary for practitioners

The dashboard reads this file at render time so it is always up to date.
To add a new strategy: append an entry here and register it in run_live_tick.STRATEGIES.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParamSpec:
    """One configurable parameter belonging to a strategy."""

    key: str                 # attribute name in src.config
    label: str               # short human name shown in the table header
    unit: str = ""           # optional unit suffix (e.g. "%" "bars" "pips")
    description: str = ""    # one-sentence plain-English explanation


@dataclass
class StrategyEntry:
    """Everything the dashboard (and any future tooling) needs to know about a strategy."""

    id: str                             # matches strategy_id in CSVs / run_live_tick.STRATEGIES
    name: str                           # display name
    active: bool                        # currently running in the live tick?
    module: str                         # Python module path for the live signal runner
    signal_source_key: str              # value of signal_source column in predictions_live.csv
    config_locked: bool                 # True = params frozen after final validation
    technique: str                      # one-liner: "ML regression", "ML classifier", "rule-based"

    # Descriptions — shown on dashboard
    plain_description: str              # < 3 sentences; assumes zero prior knowledge
    technical_description: str          # full explanation for practitioners

    # When is this strategy useful?
    best_conditions: str                # brief bullet-style string
    # What can hurt it?
    known_weaknesses: str

    # Config parameters to display
    params: list[ParamSpec] = field(default_factory=list)

    def live_config(self) -> dict[str, Any]:
        """Return {key: value} by reading the current src.config at call time."""
        try:
            from src import config
            return {p.key: getattr(config, p.key, "?") for p in self.params}
        except Exception:
            return {}


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

STRATEGIES: list[StrategyEntry] = [

    # ── 1. regression_v1 ─────────────────────────────────────────────────────
    StrategyEntry(
        id="regression_v1",
        name="ML regression (regression_v1)",
        active=True,
        module="src.live_signal_regression",
        signal_source_key="regression_features_tail",
        config_locked=True,
        technique="ML regression — gradient boosted trees (LightGBM)",

        plain_description=(
            "A machine-learning model trained on millions of 5-minute EUR/USD price bars. "
            "It predicts <em>how much the price will move over the next 30 minutes</em> and only "
            "trades when that prediction is in the extreme top or bottom of everything it has ever "
            "predicted — so most bars produce no signal at all. "
            "Two automatic safety switches pause trading during losing streaks and resume "
            "when the model recovers."
        ),
        technical_description=(
            "LightGBM multi-horizon regressor trained on core price/vol/macro features "
            "(train: 2022-01-01 – 2023-09-30). Targets are multi-horizon log-returns "
            "(3, 6, 12 bars). Live inference uses the 6-bar (30 min) horizon prediction. "
            "Signal filter stack: (1) prediction percentile ≤ top/bottom REGRESSION_TOP_PCT%; "
            "(2) |pred| ≥ REGRESSION_PRED_THRESHOLD (removes near-zero noise); "
            "(3) vol_6 in top REGRESSION_VOL_PCT% (avoids quiet/flat regimes); "
            "(4) stale-bar guard (REGRESSION_MAX_BAR_AGE_MINUTES); "
            "(5) macro event blackout (±MACRO_EVENT_BLACKOUT_MIN min around scheduled releases). "
            "Risk controls: trade-based kill-switch (rolling PF &lt; REGRESSION_KILL_SWITCH_PF "
            "over last REGRESSION_KILL_SWITCH_N trades) and drawdown kill "
            "(equity drop &gt; REGRESSION_DD_KILL from peak). "
            "Both pause for REGRESSION_PAUSE_BARS bars then auto-resume. "
            "Live bar data sourced from cTrader (execution broker) — Dukascopy used only for training."
        ),
        best_conditions=(
            "High-volatility London/NY session overlap · "
            "Clear short-term directional momentum · "
            "No scheduled macro events within 30 min"
        ),
        known_weaknesses=(
            "Degrades when market microstructure changes sharply (e.g. NFP, central bank surprises) · "
            "Low-volatility Asian session produces very few signals · "
            "Relies on stationarity of vol/return features — regime changes can temporarily hurt accuracy"
        ),
        params=[
            ParamSpec("REGRESSION_TOP_PCT",        "Percentile gate",    "%",     "Only act when prediction is in the extreme top/bottom N% of all-time predictions"),
            ParamSpec("REGRESSION_VOL_PCT",         "Vol gate",           "%",     "Only trade when the 6-bar volatility is in the top N% — filters out quiet/flat markets"),
            ParamSpec("REGRESSION_PRED_THRESHOLD",  "Min |pred|",         "",      "Ignore predicted moves smaller than this — removes near-zero noise signals"),
            ParamSpec("REGRESSION_KILL_SWITCH_N",   "Kill-switch window", "trades","Evaluate rolling profit factor after every N completed trades"),
            ParamSpec("REGRESSION_KILL_SWITCH_PF",  "Kill-switch PF min", "",      "Pause trading if rolling profit factor of last N trades drops below this"),
            ParamSpec("REGRESSION_DD_KILL",         "Drawdown kill",      "frac",  "Pause when equity drops more than this fraction from its peak"),
            ParamSpec("REGRESSION_PAUSE_BARS",      "Pause duration",     "bars",  "Bars to stay paused after a kill-switch fires (72 bars = 6 h at 5-min bars)"),
            ParamSpec("REGRESSION_MAX_BAR_AGE_MINUTES", "Max bar age",    "min",   "Block trade if the latest bar is older than this — stale data guard"),
            ParamSpec("MACRO_EVENT_BLACKOUT_MIN",   "Event blackout",     "min",   "No trades within ± N min of a scheduled macro release"),
        ],
    ),

    # ── 2. classifier_v1 ─────────────────────────────────────────────────────
    StrategyEntry(
        id="classifier_v1",
        name="ML classifier (classifier_v1)",
        active=True,
        module="src.live_signal",
        signal_source_key="test_csv_tail",
        config_locked=False,
        technique="ML classifier — gradient boosted trees + Chronos forecaster ensemble",

        plain_description=(
            "An ML model that reads the latest 5-minute EUR/USD bar and asks: "
            "<em>will the price be higher (BUY) or lower (SELL) in 30 minutes?</em> "
            "It outputs a probability between 0 and 1. "
            "A trade is only placed when that probability is high enough <em>and</em> "
            "several other safety checks (market volatility, time of day, recent losses) all pass."
        ),
        technical_description=(
            "LightGBM binary classifier trained on classifier feature set "
            "(labels: BUY if future_return_30m &gt; BUY_THRESHOLD_PCT, SELL if &lt; SELL_THRESHOLD_PCT, else FLAT). "
            "Confidence filter: max(P_buy, P_sell) ≥ MIN_CONFIDENCE_PCT (same as NO_TRADE_THRESHOLD_PCT). "
            "Additional filter: |P_buy − P_sell| ≥ CONFIDENCE_MARGIN_PCT to enforce a no-trade zone. "
            "Optional: when CONFIDENCE_TOP_PCT is set, uses percentile-based filter instead of threshold. "
            "Regime gate: vol_20 in top VOL_REGIME_TOP_PCT% (or SPREAD_PROXY_VOLATILITY_PCT legacy). "
            "Session: SESSION_EXCLUDE_HOURS controls which UTC hours are suppressed. "
            "Daily loss kill: position closed and no new trades if day's P&L &lt; −MAX_DAILY_LOSS_PCT. "
            "Cooldown: COOLDOWN_BARS_AFTER_LOSS bars of silence after a losing close. "
            "Signal source is the last row of features/test.csv (classic ML pipeline — "
            "bar timestamp typically 60 min behind live due to target horizon chop)."
        ),
        best_conditions=(
            "Trending sessions with clear directional conviction · "
            "High probability outputs (>0.65) with large |P_buy − P_sell| margin · "
            "Low scheduled-event risk"
        ),
        known_weaknesses=(
            "Bar timestamp is ~60 min behind price (target horizon chop from training pipeline) · "
            "Probability calibration may drift between retrains · "
            "MIN_CONFIDENCE_PCT is close to the model's practical max (~0.54) — very few signals fire"
        ),
        params=[
            ParamSpec("NO_TRADE_THRESHOLD_PCT",  "Confidence gate",    "",      "Only trade when max(P_buy, P_sell) exceeds this probability"),
            ParamSpec("MIN_CONFIDENCE_PCT",       "Min confidence",     "",      "Alias for NO_TRADE_THRESHOLD_PCT used by the live signal guard"),
            ParamSpec("CONFIDENCE_MARGIN_PCT",    "Margin gate",        "",      "Also require |P_buy − P_sell| to exceed this — creates a strong no-trade zone around 0.5"),
            ParamSpec("VOL_REGIME_TOP_PCT",       "Vol gate",           "%",     "Trade only when 20-bar volatility is in the top N%"),
            ParamSpec("MAX_DAILY_LOSS_PCT",       "Daily loss limit",   "frac",  "Stop trading for the rest of the day after this drawdown"),
            ParamSpec("COOLDOWN_BARS_AFTER_LOSS", "Cooldown",           "bars",  "Bars to wait after a losing trade before taking a new position"),
            ParamSpec("CLASSIFIER_MAX_BAR_AGE_MINUTES", "Max bar age",  "min",   "Block trade if the latest bar is older than this"),
            ParamSpec("MACRO_EVENT_BLACKOUT_MIN", "Event blackout",     "min",   "No trades within ± N min of a scheduled macro release"),
        ],
    ),

    # ── 3. mean_reversion_v1 ─────────────────────────────────────────────────
    StrategyEntry(
        id="mean_reversion_v1",
        name="Rule-based mean reversion (mean_reversion_v1)",
        active=True,
        module="src.live_signal_mean_reversion",
        signal_source_key="mean_reversion_features_tail",
        config_locked=False,
        technique="Rule-based — z-score of 20-bar MA gap",

        plain_description=(
            "No machine learning — pure statistics. "
            "This strategy watches how far the EUR/USD price has stretched from its "
            "short-term average. When it stretches <em>too far</em> (measured by a z-score), "
            "it bets that the price will snap back to the middle. "
            "The position is held for up to 30 minutes then automatically closed. "
            "Think of it as a rubber-band trade: the further the stretch, the stronger the pull back."
        ),
        technical_description=(
            "Uses the <code>ma_20_gap</code> feature = (close − EMA_20) / ATR_20. "
            "Entry signal: z-score of ma_20_gap over the last MR_LOOKBACK_BARS bars. "
            "BUY when z ≤ −MR_ZSCORE_THRESHOLD (price too far below its mean), "
            "SELL when z ≥ +MR_ZSCORE_THRESHOLD (too far above). "
            "Exit: forced close after MR_HOLD_BARS bars. "
            "Filters: (1) stale-bar guard; "
            "(2) session gate — London or NY session only (is_london_session / is_ny_session); "
            "(3) macro event blackout; "
            "(4) vol gate — vol_6 in top MR_VOL_PCT% (sufficient liquidity to revert). "
            "Risk controls: trade-based kill-switch (rolling PF over last MR_KILL_SWITCH_N trades) "
            "and drawdown kill (MR_DD_KILL). Both pause for MR_PAUSE_BARS bars. "
            "No ML model — fully deterministic and interpretable."
        ),
        best_conditions=(
            "Range-bound / mean-reverting regimes · "
            "Intraday overextensions during liquid London or NY sessions · "
            "After a sharp spike with no follow-through (e.g. news reaction that fades)"
        ),
        known_weaknesses=(
            "Fails badly during strong trending moves — mean never comes · "
            "z-score threshold tuned on 2022-2024 data; may need recalibration as regime shifts · "
            "30-min forced exit can leave money on the table in slow-reverting environments"
        ),
        params=[
            ParamSpec("MR_ZSCORE_THRESHOLD",    "Z-score threshold",  "",      "Enter when the gap from the 20-bar MA is this many standard deviations away from its own recent mean"),
            ParamSpec("MR_HOLD_BARS",           "Hold bars",          "bars",  "Force-close the position after this many bars regardless of P&L (30 min at 5-min bars)"),
            ParamSpec("MR_LOOKBACK_BARS",       "Z-score lookback",   "bars",  "Rolling window for computing the z-score of the MA gap"),
            ParamSpec("MR_VOL_PCT",             "Vol gate",           "%",     "Trade only when short-term volatility is in the top N% — needs liquidity for the reversion"),
            ParamSpec("MR_KILL_SWITCH_N",       "Kill-switch window", "trades","Evaluate rolling profit factor after every N completed trades"),
            ParamSpec("MR_KILL_SWITCH_PF",      "Kill-switch PF min", "",      "Pause trading if rolling profit factor drops below this"),
            ParamSpec("MR_DD_KILL",             "Drawdown kill",      "frac",  "Pause when equity drops more than this fraction from its peak"),
            ParamSpec("MR_PAUSE_BARS",          "Pause duration",     "bars",  "Bars to stay paused after a kill-switch fires"),
            ParamSpec("MR_MAX_BAR_AGE_MINUTES", "Max bar age",        "min",   "Block trade if the latest bar is older than this — stale data guard"),
            ParamSpec("MACRO_EVENT_BLACKOUT_MIN","Event blackout",    "min",   "No trades within ± N min of a scheduled macro release"),
        ],
    ),

    # ── 4. session_breakout_v1 ────────────────────────────────────────────────
    StrategyEntry(
        id="session_breakout_v1",
        name="Rule-based session breakout (session_breakout_v1)",
        active=False,   # DISABLED — structural failure confirmed in research (2026-03)
        module="src.live_signal_session_breakout",
        signal_source_key="session_breakout_price_tail",
        config_locked=False,
        technique="Rule-based — rolling N-bar high/low breakout at session opens",

        plain_description=(
            "No machine learning — pure price logic. "
            "At the London open (07:00–10:00 UTC) and NY open (12:00–15:00 UTC), "
            "this strategy watches whether EUR/USD price breaks above or below the range "
            "it has traded over the last hour. A break upward triggers a BUY; "
            "a break downward triggers a SELL. "
            "The position is held for up to 60 minutes then automatically closed. "
            "At most one trade per session per day."
        ),
        technical_description=(
            "Data source: EURUSD_5min_clean.parquet (raw OHLC — independent from features_regression_core). "
            "Signal: range_high = rolling(SB_N_LOOKBACK).max().shift(1); range_low analogous. "
            "BUY when close &gt; range_high; SELL when close &lt; range_low. "
            "Filters: (1) in-session — London 07–10 UTC or NY 12–15 UTC; "
            "(2) range_size &gt; SB_MIN_RANGE_SIZE (avoids flat/noise bars); "
            "(3) once-per-session guard (last_session_id in state.json prevents re-entry). "
            "Exit: forced close after SB_HOLD_BARS bars. "
            "Guards: stale-bar, macro event blackout, daily loss cap, kill switch, DD kill. "
            "No ML — fully deterministic and independently auditable. "
            "Completely separate edge from mean_reversion_v1 (momentum vs reversion)."
        ),
        best_conditions=(
            "London or NY session opens with clear directional momentum · "
            "Range large enough to be meaningful (> SB_MIN_RANGE_SIZE) · "
            "Trending intraday environment — breakouts follow through"
        ),
        known_weaknesses=(
            "Choppy / range-bound markets generate false breakouts · "
            "Only 1–2 trades per day by design — low frequency · "
            "Momentum reversal immediately after session open can cause quick losses"
        ),
        params=[
            ParamSpec("SB_N_LOOKBACK",           "Range window",       "bars",  "Rolling bars used to define the breakout range (N × 5min = lookback window in minutes)"),
            ParamSpec("SB_HOLD_BARS",             "Hold bars",          "bars",  "Force-close the position after this many bars regardless of P&L"),
            ParamSpec("SB_MIN_RANGE_SIZE",        "Min range",          "price", "Skip bars where the rolling range is smaller than this — avoids flat/choppy markets"),
            ParamSpec("SB_VOL_PCT",               "Vol gate",           "%",     "Trade only when vol_6 is in the top N% (0 = disabled)"),
            ParamSpec("SB_KILL_SWITCH_N",         "Kill-switch window", "trades","Evaluate rolling profit factor after every N completed trades"),
            ParamSpec("SB_KILL_SWITCH_PF",        "Kill-switch PF min", "",      "Pause trading if rolling profit factor drops below this"),
            ParamSpec("SB_DD_KILL",               "Drawdown kill",      "frac",  "Pause when equity drops more than this fraction from its peak"),
            ParamSpec("SB_PAUSE_BARS",            "Pause duration",     "bars",  "Bars to stay paused after a kill-switch fires"),
            ParamSpec("SB_MAX_BAR_AGE_MINUTES",   "Max bar age",        "min",   "Block trade if the latest bar is older than this — stale data guard"),
            ParamSpec("MACRO_EVENT_BLACKOUT_MIN", "Event blackout",     "min",   "No trades within ± N min of a scheduled macro release"),
        ],
    ),
    # ── 5. regression_v2_trendfilter ─────────────────────────────────────────
    StrategyEntry(
        id="regression_v2_trendfilter",
        name="ML regression + 4h trend filter (regression_v2_trendfilter)",
        active=True,
        module="src.live_signal_regression_v2_trendfilter",
        signal_source_key="regression_v2_trendfilter_features_tail",
        config_locked=False,
        technique="ML regression (LightGBM) + rule-based 4-hour MA10 trend filter",

        plain_description=(
            "The same machine-learning prediction model as regression_v1, but with an extra layer: "
            "a 4-hour trend filter that only allows <em>long trades when the 4-hour chart is in an "
            "uptrend</em> and <em>short trades when it is in a downtrend</em> (measured by whether "
            "the 4-hour closing price is above or below a 10-bar moving average). "
            "This prevents the model from fighting the higher-timeframe momentum — the most common "
            "source of losses in regression_v1. "
            "Running in paper alongside regression_v1 to measure live improvement."
        ),
        technical_description=(
            "Same LightGBM regressor as regression_v1 (trained 2022-01-01 – 2023-09-30, "
            "target: 6-bar log-return). Same signal selection stack: prediction percentile "
            "≤ top/bottom RV2_TOP_PCT%; |pred| ≥ RV2_PRED_THRESHOLD; vol_6 in top RV2_VOL_PCT%. "
            "<br><br>"
            "<strong>Additional 4h trend gate</strong>: before any position change, "
            "compute the 4-hour close MA(RV2_TREND_MA_WINDOW) from EURUSD_5min_clean.parquet "
            "resampled to 4h bars (last close of each 4h candle). "
            "Gate rule: long entries pass only when close_4h &gt; MA_4h; "
            "short entries pass only when close_4h &lt; MA_4h. "
            "Blocked bars are logged with reason='trend_filter'. "
            "Live output includes trend_label (up/down/neutral/warmup) and trend_strength "
            "(close_4h / MA_4h − 1) for every bar. "
            "<br><br>"
            "<strong>Research validation</strong> (data/validation/trend_filter_*.csv):<br>"
            "• Sweep: 4h MA10 PF 1.60 vs baseline 1.08; cum return +17.5% vs −17.3%<br>"
            "• Stability: all MA windows 5–40 beat baseline (confirmed plateau, not spike)<br>"
            "• Walk-forward: 14/27 months improved; positive months 44% → 63%<br>"
            "• Cost stress: filtered at 2× spread still beats unfiltered at 1×<br>"
            "• Max drawdown: 3.8% vs 19.2% baseline — filter cuts worst drawdown by 80%<br>"
            "<br>"
            "Risk controls identical to regression_v1: kill-switch (rolling PF of last "
            "RV2_KILL_SWITCH_N trades &lt; RV2_KILL_SWITCH_PF), drawdown kill "
            "(equity drop &gt; RV2_DD_KILL from peak), both pause for RV2_PAUSE_BARS bars. "
            "<br><br>"
            "Note: RV2_TOP_PCT=1.0 is wider than v1's 0.25% to give the trend filter enough "
            "candidate trades — the filter itself provides the primary selectivity. "
            "Validated on the same test period (Jan 2024 – Mar 2026). "
            "Promoted from research to paper-trade after passing all 4 validation gates: "
            "sweep, walk-forward, cost-stress, stability."
        ),
        best_conditions=(
            "Clear 4h directional trend (price well above/below 4h MA10) · "
            "High-volatility London/NY session · "
            "Trending days — the filter deliberately avoids choppy/ranging environments · "
            "No scheduled macro events within 30 min"
        ),
        known_weaknesses=(
            "Trend reversals: when the 4h trend flips, in-flight positions can be closed "
            "early while the underlying model signal may still be valid · "
            "Fewer trades than v1 (~50% of v1 trade count) — lower throughput means "
            "equity curve builds slowly and kill-switch needs more time to gather evidence · "
            "4h MA10 computed from offline price file — same warmup limitation as any MA "
            "(first ~40 hours of data have no trend signal, treated as pass-through) · "
            "If 4h trend is neutral (close == MA exactly) entry is allowed — rare but possible"
        ),
        params=[
            ParamSpec("RV2_TREND_RESAMPLE",   "HTF period",         "",      "Resample period for higher-timeframe bars used in the trend filter (4h = four 1-hour candles)"),
            ParamSpec("RV2_TREND_MA_WINDOW",  "Trend MA window",    "bars",  "Moving average window on 4-hour bars — MA10 on 4h ≈ 40-hour trend signal"),
            ParamSpec("RV2_TOP_PCT",          "Percentile gate",    "%",     "Trade top/bottom N% of predictions — wider than v1 (1% vs 0.25%) because trend filter provides primary selectivity"),
            ParamSpec("RV2_VOL_PCT",          "Vol gate",           "%",     "Trade only when vol_6 in top N% — same purpose as v1 vol gate"),
            ParamSpec("RV2_PRED_THRESHOLD",   "Min |pred|",         "",      "Minimum absolute prediction to trade (0 = disabled — trend gate is the primary filter)"),
            ParamSpec("RV2_KILL_SWITCH_N",    "Kill-switch window", "trades","Evaluate rolling profit factor after every N completed trades"),
            ParamSpec("RV2_KILL_SWITCH_PF",   "Kill-switch PF min", "",      "Pause trading if rolling PF of last N trades drops below this"),
            ParamSpec("RV2_DD_KILL",          "Drawdown kill",      "frac",  "Pause when equity drops more than this fraction from its peak"),
            ParamSpec("RV2_PAUSE_BARS",       "Pause duration",     "bars",  "Bars to stay paused after a kill-switch fires (72 bars = 6 h at 5-min bars)"),
            ParamSpec("MACRO_EVENT_BLACKOUT_MIN", "Event blackout", "min",   "No trades within ± N min of a scheduled macro release (shared with all strategies)"),
        ],
    ),

]

# Fast lookup by id
STRATEGY_MAP: dict[str, StrategyEntry] = {s.id: s for s in STRATEGIES}
