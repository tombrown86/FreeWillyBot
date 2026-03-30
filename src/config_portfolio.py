"""
Portfolio engine configuration — frozen after initial paper-trade validation.

Kept separate from src/config.py so it can be locked independently of strategy
signal parameters. Every constant here is consumed only by src/portfolio_engine.py
and scripts/run_portfolio_backtest.py — live signal modules never import this file.

Frozen signal assumptions (regression_v2_trendfilter — read from src.config, not duplicated)
----------------------------------------------------------------------------------------
The promoted strategy’s thresholds, trend filter, and per-strategy risk controls remain
on src.config (RV2_* and shared REGRESSION_* where applicable). This file documents the
linkage so the portfolio layer stays aligned with the same frozen research:

  RV2_TREND_RESAMPLE, RV2_TREND_MA_WINDOW   — 4h MA10 trend gate
  RV2_TOP_PCT, RV2_VOL_PCT, RV2_PRED_THRESHOLD
  RV2_KILL_SWITCH_N, RV2_KILL_SWITCH_PF, RV2_DD_KILL, RV2_PAUSE_BARS

Change RV2_* only after a new validation pass; change PORTFOLIO_* for risk / sizing only.

Primary strategy for this portfolio build (single active edge for now): see
PORTFOLIO_PRIMARY_STRATEGY_ID below.

Research provenance
-------------------
regression_v2_trendfilter validation (data/validation/):
  4h MA10 trend filter PF 1.60 vs baseline 1.08
  positive months 63% vs 44% baseline
  cost stress: filtered at 2x still beats unfiltered at 1x
  walk-forward: 14/27 months improved, worst month -0.97%
Starting config is deliberately conservative:
  base size 1.0 unit, vol targeting clipped 0.5–2x, min size 0.5 unit
  portfolio DD kill 3% (above the per-strategy 2% kill) acts as a hard backstop
"""

# ─────────────────────────────────────────────────────────────────────────────
# Strategy identity (batch 1 — single portfolio owner for now)
# ─────────────────────────────────────────────────────────────────────────────

PORTFOLIO_PRIMARY_STRATEGY_ID: str = "regression_v2_trendfilter"

# If a future event-driven strategy id is registered, list it here so conflict rules
# can prefer flattening non-event strategies first. Empty list = no event strategies yet.
PORTFOLIO_EVENT_STRATEGY_IDS: tuple[str, ...] = ()

# Paper evaluation window (batch 8 — manual; no code enforces this)
PORTFOLIO_PAPER_EVAL_WEEKS: int = 4

# Broker lot step (cTrader micro-lot increments). Used when rounding final size.
PORTFOLIO_LOT_STEP: float = 0.01

# Sizing stack per strategy: "full" (vol×trend×DD×streak), "vol_only", "fixed" (base only)
PORTFOLIO_SIZING_MODE_DEFAULT: str = "full"

PORTFOLIO_SIZING_MODE_BY_STRATEGY: dict[str, str] = {
    # Same signal as regression_v2_trendfilter; portfolio uses vol-targeting only (backtest winner)
    "regression_v2_trendfilter_portfolio_vol": "vol_only",
}

PORTFOLIO_STRATEGY_SIBLINGS: dict[str, list[str]] = {
    # regression_v2_trendfilter and its portfolio-vol variant share the same signal.
    # Only one should hold an active position on the demo account at any time.
    # Paper books track independently (is_strategy_allowed only gates execution orders).
    "regression_v2_trendfilter":              ["regression_v2_trendfilter_portfolio_vol"],
    "regression_v2_trendfilter_portfolio_vol": ["regression_v2_trendfilter"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Risk budget
# ─────────────────────────────────────────────────────────────────────────────

PORTFOLIO_TARGET_DAILY_VOL:   float = 0.0025   # 0.25% equity per day (annualised ≈ 4%)
PORTFOLIO_MAX_SINGLE_TRADE:   float = 0.0010   # 0.10% equity max risk per trade (reference)
PORTFOLIO_MAX_GROSS_EXPOSURE: float = 1.0      # never exceed 1x gross; single asset

# ─────────────────────────────────────────────────────────────────────────────
# Trade size bounds
# Sizes are "units" that map to CTRADER_VOLUME_LOTS at execution time.
# 1 unit = 1 standard lot = 100 000 base currency units.
# Minimum enforced BEFORE sending any order; trades below floor are silently skipped.
# ─────────────────────────────────────────────────────────────────────────────

PORTFOLIO_BASE_SIZE: float = 1.0   # default lot size when no scaling is applied
PORTFOLIO_MIN_SIZE:  float = 0.5   # floor — do not trade smaller than this
PORTFOLIO_MAX_SIZE:  float = 2.0   # cap — never send an order larger than this

# ─────────────────────────────────────────────────────────────────────────────
# Volatility targeting
# Realized vol is computed as std of last LOOKBACK_BARS bar-level log returns.
# VOL_TARGET sets the desired per-bar volatility level.
# When actual vol > target, size is reduced; when vol < target, size increases
# up to VOL_CLIP_HIGH.
# ─────────────────────────────────────────────────────────────────────────────

PORTFOLIO_VOL_LOOKBACK_BARS: int   = 48      # 4 hours of 5-min bars
PORTFOLIO_VOL_TARGET:        float = 0.0003  # ~3 pips per bar at 1.10 EURUSD
PORTFOLIO_VOL_CLIP_LOW:      float = 0.5     # floor: never reduce by more than half
PORTFOLIO_VOL_CLIP_HIGH:     float = 2.0     # cap: never more than double base

# ─────────────────────────────────────────────────────────────────────────────
# Trend-strength size buckets
# Uses the trend_strength field from regression_v2_trendfilter signal output
# (close/MA−1 on 4h bars). Bucketed rather than continuous to reduce sensitivity.
# ─────────────────────────────────────────────────────────────────────────────

PORTFOLIO_TREND_STRONG_THRESH: float = 0.003   # |strength| >= this → 1.0x
PORTFOLIO_TREND_MEDIUM_THRESH: float = 0.001   # |strength| >= this → 0.75x
# |strength| below medium threshold → 0.5x

# ─────────────────────────────────────────────────────────────────────────────
# Portfolio-level drawdown scaling
# Applied on top of vol and trend multipliers. The per-strategy DD kill (2%)
# is still active; these buckets add a softer graduated response before that.
# ─────────────────────────────────────────────────────────────────────────────

PORTFOLIO_DD_TIER1: float = 0.01   # DD < 1%  → full size (1.0x)
PORTFOLIO_DD_TIER2: float = 0.02   # DD 1–2%  → 0.75x
# DD >= 2% → 0.5x; above this the per-strategy DD kill fires anyway

# ─────────────────────────────────────────────────────────────────────────────
# Portfolio-level kill switch (all strategies combined)
# Fires AFTER per-strategy kills — acts as a final backstop.
# ─────────────────────────────────────────────────────────────────────────────

PORTFOLIO_KILL_DD:     float = 0.03   # pause if portfolio equity DD from peak > 3%
PORTFOLIO_KILL_PF_N:   int   = 20     # evaluate rolling PF after N combined closed trades
PORTFOLIO_KILL_PF_MIN: float = 0.85   # pause if combined PF of last N trades < 0.85
PORTFOLIO_PAUSE_HOURS: int   = 6      # cooldown duration after portfolio kill fires

# ─────────────────────────────────────────────────────────────────────────────
# Stale data hard limit (portfolio level, stricter than per-strategy guards)
# ─────────────────────────────────────────────────────────────────────────────

PORTFOLIO_MAX_BAR_AGE_HOURS: float = 0.5   # reject signal if bar_ts is > 30 min old

# ─────────────────────────────────────────────────────────────────────────────
# Loss-streak throttle
# After N consecutive losing trades, reduce size until a winning trade resets the streak.
# ─────────────────────────────────────────────────────────────────────────────

PORTFOLIO_LOSS_STREAK_N:    int   = 3     # consecutive losses before throttle fires
PORTFOLIO_LOSS_STREAK_MULT: float = 0.5   # size multiplier while streak is active
