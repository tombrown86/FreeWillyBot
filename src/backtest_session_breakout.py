"""
Session Breakout strategy v1 — backtest engine.

Mirrors the structure of src/backtest_regression.py.

Public API:
    run_single(df, n, min_range, hold_bars, cost_per_leg, cost_mult) -> dict
    run_grid(df, ...)                                                 -> pd.DataFrame
    run_walk_forward(df, n, min_range, hold_bars)                     -> list[dict]

The backtest is fully vectorised for the signal computation, then uses a
forward-pass loop for the hold timer and once-per-session blocking — the two
stateful behaviours that cannot be expressed as simple vector operations.

Cost model: FX_SPREAD per leg (same as other backtests in this repo).
"""

from __future__ import annotations

import itertools
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FX_SPREAD_PIPS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRICE_PARQUET = PROJECT_ROOT / "data" / "processed" / "price" / "EURUSD_5min_clean.parquet"
BACKTESTS_DIR = PROJECT_ROOT / "data" / "backtests_session_breakout"

FX_SPREAD = FX_SPREAD_PIPS * 0.0001

# ── default grid ─────────────────────────────────────────────────────────────
N_VALUES = [8, 12, 16]
MIN_RANGE_VALUES = [0.0002, 0.0003, 0.0004]
HOLD_VALUES = [4, 6, 12]
BARS_PER_YEAR = 252 * 288  # 5-min bars in a trading year


# ── data helpers ─────────────────────────────────────────────────────────────

def load_price(start_date: str | None = None) -> pd.DataFrame:
    """Load EURUSD_5min_clean.parquet. Optionally filter to start_date (YYYY-MM-DD)."""
    if not PRICE_PARQUET.exists():
        raise FileNotFoundError(f"Price file not found: {PRICE_PARQUET}")
    df = pd.read_parquet(PRICE_PARQUET)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    if start_date:
        df = df[df["timestamp"] >= pd.Timestamp(start_date, tz="UTC")].reset_index(drop=True)
    return df


# ── core signal computation ───────────────────────────────────────────────────

def _compute_signals(df: pd.DataFrame, n: int, min_range: float) -> pd.DataFrame:
    """
    Add range_high, range_low, range_size, in_session, raw_signal to df (in-place copy).
    Uses shift(1) to prevent lookahead leakage.
    """
    df = df.copy()
    df["range_high"] = df["high"].rolling(n).max().shift(1)
    df["range_low"] = df["low"].rolling(n).min().shift(1)
    df["range_size"] = df["range_high"] - df["range_low"]
    df["ret_1"] = df["close"].pct_change()

    hour = df["timestamp"].dt.hour
    df["in_session"] = ((hour >= 7) & (hour < 10)) | ((hour >= 12) & (hour < 15))
    df["valid_range"] = df["range_size"] >= min_range

    df["raw_signal"] = 0
    mask_buy = df["close"] > df["range_high"]
    mask_sell = df["close"] < df["range_low"]
    df.loc[mask_buy, "raw_signal"] = 1
    df.loc[mask_sell, "raw_signal"] = -1

    # Apply session + range filter (no hold / session-dedup yet)
    df["filtered_signal"] = df["raw_signal"].where(df["in_session"] & df["valid_range"], 0)
    return df


def _session_label(ts: pd.Timestamp) -> str:
    h = ts.hour
    if 7 <= h < 10:
        return "london"
    if 12 <= h < 15:
        return "ny"
    return "other"


# ── single backtest run ────────────────────────────────────────────────────────

def _profit_factor(ret: np.ndarray, positions: np.ndarray) -> float:
    trade_ret = ret * positions
    gains = trade_ret[trade_ret > 0].sum()
    losses = abs(trade_ret[trade_ret < 0].sum())
    return float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)


def run_single(
    df: pd.DataFrame,
    n: int = 12,
    min_range: float = 0.0003,
    hold_bars: int = 12,
    cost_per_leg: float = FX_SPREAD,
    cost_mult: float = 1.0,
) -> dict:
    """
    Run a single backtest on the given price DataFrame.

    Returns dict: net_return, profit_factor, max_dd, n_trades, n, min_range, hold_bars.
    """
    cost = cost_per_leg * cost_mult
    df = _compute_signals(df, n, min_range)

    filtered_signal = df["filtered_signal"].values.astype(int)
    ret = df["ret_1"].fillna(0).values.astype(float)
    timestamps = df["timestamp"].values

    n_rows = len(df)
    equity = np.ones(n_rows + 1)
    position = 0
    bars_held = 0
    last_session_id = ""
    n_trades = 0

    for i in range(n_rows):
        sig = filtered_signal[i]
        ts = pd.Timestamp(timestamps[i])

        # Force close on hold expiry (before evaluating new entry)
        if position != 0 and bars_held >= hold_bars:
            legs = 1
            equity[i + 1] = equity[i] * (1 - legs * cost)
            n_trades += 1
            position = 0
            bars_held = 0
        else:
            equity[i + 1] = equity[i]

        if position == 0 and sig != 0:
            session_id = f"{ts.date()}_{_session_label(ts)}"
            if session_id != last_session_id and "other" not in session_id:
                equity[i + 1] *= (1 - cost)
                n_trades += 1
                position = sig
                bars_held = 1
                last_session_id = session_id
        elif position != 0:
            equity[i + 1] *= 1 + position * ret[i]
            bars_held += 1

    cum_ret = float(equity[-1] - 1.0)

    eq = equity[1:]
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, 1)
    max_dd = float(np.max(dd))

    period_ret = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1)
    period_ret = period_ret[np.isfinite(period_ret)]
    if len(period_ret) > 1 and np.std(period_ret) > 0:
        sharpe = float(np.sqrt(BARS_PER_YEAR) * np.mean(period_ret) / np.std(period_ret))
    else:
        sharpe = 0.0

    gains = period_ret[period_ret > 0].sum()
    losses = abs(period_ret[period_ret < 0].sum())
    pf = float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)

    return {
        "n": n,
        "min_range": min_range,
        "hold_bars": hold_bars,
        "net_return": cum_ret,
        "profit_factor": pf,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "n_trades": n_trades,
        "cost_mult": cost_mult,
    }


# ── grid search ───────────────────────────────────────────────────────────────

def run_grid(
    df: pd.DataFrame,
    n_values: list[int] | None = None,
    min_range_values: list[float] | None = None,
    hold_values: list[int] | None = None,
    cost_mult: float = 1.0,
    save: bool = True,
) -> pd.DataFrame:
    """
    Run all combinations of (n, min_range, hold_bars).

    Returns a DataFrame sorted by profit_factor descending.
    Saves to data/backtests_session_breakout/grid_results.csv when save=True.
    """
    n_values = n_values or N_VALUES
    min_range_values = min_range_values or MIN_RANGE_VALUES
    hold_values = hold_values or HOLD_VALUES

    combos = list(itertools.product(n_values, min_range_values, hold_values))
    logging.info("[session_breakout] Grid: %d combinations", len(combos))

    results = []
    for n, mr, hb in combos:
        try:
            r = run_single(df, n=n, min_range=mr, hold_bars=hb, cost_mult=cost_mult)
            results.append(r)
            logging.info(
                "[session_breakout] N=%d mr=%.4f hold=%d → net=%.4f PF=%.2f trades=%d",
                n, mr, hb, r["net_return"], r["profit_factor"], r["n_trades"],
            )
        except Exception as exc:
            logging.warning("[session_breakout] combo N=%d mr=%.4f hold=%d failed: %s", n, mr, hb, exc)

    grid_df = pd.DataFrame(results).sort_values("profit_factor", ascending=False).reset_index(drop=True)

    if save:
        BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
        out = BACKTESTS_DIR / "grid_results.csv"
        grid_df.to_csv(out, index=False)
        logging.info("[session_breakout] Grid results saved to %s", out)

    return grid_df


# ── walk-forward ──────────────────────────────────────────────────────────────

def run_walk_forward(
    df: pd.DataFrame,
    n: int = 12,
    min_range: float = 0.0003,
    hold_bars: int = 12,
    window_months: int = 1,
    save: bool = True,
    label: str = "",
) -> list[dict]:
    """
    Rolling walk-forward: split df into monthly windows, run backtest on each.

    window_months=1 → 12×1m windows (matches walk_forward_regression.py convention).
    Saves to data/backtests_session_breakout/walk_forward_{label}.csv when save=True.
    """
    df = df.copy()
    df["_month"] = df["timestamp"].dt.to_period("M")
    months = sorted(df["_month"].unique())

    results: list[dict] = []
    step = window_months

    for start_idx in range(0, len(months) - step + 1, step):
        window = months[start_idx: start_idx + step]
        w_df = df[df["_month"].isin(window)].copy().reset_index(drop=True)
        if len(w_df) < 50:
            continue
        try:
            r = run_single(w_df, n=n, min_range=min_range, hold_bars=hold_bars)
            r["window_start"] = str(window[0])
            r["window_months"] = window_months
            results.append(r)
            logging.info(
                "[session_breakout] WF %s (%dm): net=%.4f PF=%.2f trades=%d",
                window[0], window_months, r["net_return"], r["profit_factor"], r["n_trades"],
            )
        except Exception as exc:
            logging.warning("[session_breakout] WF window %s failed: %s", window[0], exc)

    if save and results:
        BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
        suffix = f"_{label}" if label else f"_n{n}_mr{str(min_range).replace('.', '')}_h{hold_bars}"
        out = BACKTESTS_DIR / f"walk_forward{suffix}.csv"
        pd.DataFrame(results).to_csv(out, index=False)
        logging.info("[session_breakout] Walk-forward results saved to %s", out)

    return results


# ── stability sweep ───────────────────────────────────────────────────────────

def run_stability(
    df: pd.DataFrame,
    center_n: int = 12,
    center_min_range: float = 0.0003,
    center_hold: int = 12,
    save: bool = True,
) -> pd.DataFrame:
    """
    Perturb each parameter ±1 step around the chosen config.
    Used to detect cliff-edge overfitting: if adjacent params collapse, the config is fragile.
    """
    n_candidates = sorted({max(4, center_n - 4), center_n, center_n + 4})
    mr_step = 0.00005
    mr_candidates = sorted({max(0.0001, round(center_min_range - mr_step, 5)), center_min_range, round(center_min_range + mr_step, 5)})
    hold_candidates = sorted({max(2, center_hold - 2), center_hold, min(24, center_hold + 4)})

    results = []
    for n, mr, hb in itertools.product(n_candidates, mr_candidates, hold_candidates):
        try:
            r = run_single(df, n=n, min_range=mr, hold_bars=hb)
            r["is_center"] = (n == center_n and mr == center_min_range and hb == center_hold)
            results.append(r)
        except Exception as exc:
            logging.warning("[session_breakout] stability N=%d mr=%.5f hold=%d failed: %s", n, mr, hb, exc)

    stab_df = pd.DataFrame(results).sort_values("profit_factor", ascending=False).reset_index(drop=True)

    if save and not stab_df.empty:
        BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
        out = BACKTESTS_DIR / "stability_results.csv"
        stab_df.to_csv(out, index=False)
        logging.info("[session_breakout] Stability results saved to %s", out)

    return stab_df


# ── cost stress ───────────────────────────────────────────────────────────────

def run_cost_stress(
    df: pd.DataFrame,
    n: int = 12,
    min_range: float = 0.0003,
    hold_bars: int = 12,
    cost_mults: list[float] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Run the same config at different cost multipliers (1×, 1.5×, 2×)."""
    cost_mults = cost_mults or [1.0, 1.5, 2.0]
    results = []
    for mult in cost_mults:
        r = run_single(df, n=n, min_range=min_range, hold_bars=hold_bars, cost_mult=mult)
        results.append(r)
        logging.info(
            "[session_breakout] Cost stress %sx: net=%.4f PF=%.2f trades=%d",
            mult, r["net_return"], r["profit_factor"], r["n_trades"],
        )

    stress_df = pd.DataFrame(results)

    if save and not stress_df.empty:
        BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
        out = BACKTESTS_DIR / "cost_stress_results.csv"
        stress_df.to_csv(out, index=False)
        logging.info("[session_breakout] Cost stress results saved to %s", out)

    return stress_df


# ═════════════════════════════════════════════════════════════════════════════
# v2 — Opening Range Breakout (ORB)
# ═════════════════════════════════════════════════════════════════════════════
#
# Core difference from v1:
#   v1: rolling N-bar max/min updated every bar → entry fires at the exhaustion
#       of a recent move (buying the local high/low).
#   v2: range is frozen at [session_start, session_start + or_minutes) for each
#       calendar day. After the opening window closes, the first bar that closes
#       beyond the frozen range triggers entry. This turns "buying exhaustion"
#       into "trading expansion out of an established opening balance."
# ═════════════════════════════════════════════════════════════════════════════

# v2 grid defaults (18 combos: 3 × 2 × 3 — intentionally small)
OR_MINUTES_VALUES = [15, 30, 60]
ENTRY_BUFFER_VALUES = [0.0, 0.0001]   # 0 = no buffer; 0.0001 = 1 pip
HOLD_VALUES_V2 = [6, 12, 18]

# Session definitions (UTC — same convention as v1)
SESSIONS_V2 = {
    "london": 7,   # session starts at 07:00 UTC
    "ny": 12,      # session starts at 12:00 UTC
}


# ── ORB signal computation ────────────────────────────────────────────────────

def _compute_signals_orb(
    df: pd.DataFrame,
    or_minutes: int = 30,
    entry_buffer: float = 0.0,
    entry_cutoff_bars: int = 18,
    sessions: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute per-row ORB signals.

    For each (date, session):
      1. Collect bars in [session_start, session_start + or_minutes).
         These are the "opening range" bars — no entry allowed here.
      2. Compute or_high = max(high) and or_low = min(low) over that window.
      3. In the entry window [or_end, or_end + entry_cutoff_bars * 5min):
         - First bar where close > or_high + entry_buffer → signal = +1 (BUY)
         - First bar where close < or_low  - entry_buffer → signal = -1 (SELL)
         - At most one signal per (date, session) — first one wins.

    Adds columns: or_high, or_low, or_range_size, or_signal, in_or_window,
                  in_entry_window, session_label.
    ret_1 is also added for the equity simulation.
    """
    sessions = sessions or list(SESSIONS_V2.keys())
    df = df.copy()
    df["ret_1"] = df["close"].pct_change()
    df["_date"] = df["timestamp"].dt.date
    df["_hour"] = df["timestamp"].dt.hour
    df["_minute"] = df["timestamp"].dt.minute
    df["_ts_minutes"] = df["_hour"] * 60 + df["_minute"]  # minutes since midnight UTC

    # Initialise output columns
    df["or_high"] = np.nan
    df["or_low"] = np.nan
    df["or_range_size"] = np.nan
    df["or_signal"] = 0
    df["in_or_window"] = False
    df["in_entry_window"] = False
    df["session_label"] = ""

    for sess_name, sess_start_hour in SESSIONS_V2.items():
        if sess_name not in sessions:
            continue

        sess_start_min = sess_start_hour * 60          # e.g. 420 for London (7h)
        or_end_min = sess_start_min + or_minutes       # e.g. 450 for 30m OR
        entry_end_min = or_end_min + entry_cutoff_bars * 5  # 5-min bars

        # Mark bars in the opening range window
        in_or = (
            (df["_ts_minutes"] >= sess_start_min)
            & (df["_ts_minutes"] < or_end_min)
        )
        # Mark bars in the entry window
        in_entry = (
            (df["_ts_minutes"] >= or_end_min)
            & (df["_ts_minutes"] < entry_end_min)
        )

        df.loc[in_or, "in_or_window"] = True
        df.loc[in_or | in_entry, "session_label"] = sess_name
        df.loc[in_entry, "in_entry_window"] = True

        # Per-day OR high/low: groupby date over opening-range bars
        or_bars = df[in_or][["_date", "high", "low"]].copy()
        if or_bars.empty:
            continue

        daily_or = or_bars.groupby("_date").agg(
            or_high_=("high", "max"),
            or_low_=("low", "min"),
        ).reset_index()

        # Broadcast OR high/low to ALL bars of that date (entry window will filter)
        df_with_or = df.merge(daily_or, on="_date", how="left")
        # Only fill for this session's rows (OR + entry window)
        sess_mask = in_or | in_entry
        df.loc[sess_mask, "or_high"] = df_with_or.loc[sess_mask, "or_high_"].values
        df.loc[sess_mask, "or_low"] = df_with_or.loc[sess_mask, "or_low_"].values
        df.loc[sess_mask, "or_range_size"] = (
            df_with_or.loc[sess_mask, "or_high_"] - df_with_or.loc[sess_mask, "or_low_"]
        ).values

        # Generate breakout signals in the entry window only
        # We need at most one signal per (date, session) — use groupby with idxfirst
        entry_df = df[in_entry].copy()
        if entry_df.empty:
            continue

        entry_df["_raw_sig"] = 0
        buy_mask = entry_df["close"] > (entry_df["or_high"] + entry_buffer)
        sell_mask = entry_df["close"] < (entry_df["or_low"] - entry_buffer)
        # Long signal is priority over short (both can't happen on first entry bar)
        entry_df.loc[buy_mask, "_raw_sig"] = 1
        entry_df.loc[sell_mask & ~buy_mask, "_raw_sig"] = -1

        # Keep only the FIRST non-zero signal per date for this session
        sig_rows = entry_df[entry_df["_raw_sig"] != 0].copy()
        if sig_rows.empty:
            continue
        first_sig = sig_rows.groupby("_date")["_raw_sig"].first().reset_index()
        first_sig.columns = ["_date", "_first_sig"]

        # We need the original index of the first-signal bar
        sig_rows2 = sig_rows.reset_index().merge(first_sig, on="_date")
        # Filter to rows where the signal matches the first signal for that date
        sig_rows2 = sig_rows2[sig_rows2["_raw_sig"] == sig_rows2["_first_sig"]]
        # Take only the first bar per date
        first_idx = sig_rows2.groupby("_date")["index"].first()

        df.loc[first_idx.values, "or_signal"] = sig_rows2.set_index("index").loc[
            first_idx.values, "_first_sig"
        ].values

    df.drop(columns=["_date", "_hour", "_minute", "_ts_minutes"], inplace=True)
    return df


# ── single v2 backtest run ────────────────────────────────────────────────────

def run_single_v2(
    df: pd.DataFrame,
    or_minutes: int = 30,
    entry_buffer: float = 0.0,
    entry_cutoff_bars: int = 18,
    hold_bars: int = 12,
    cost_per_leg: float = FX_SPREAD,
    cost_mult: float = 1.0,
    sessions: list[str] | None = None,
) -> dict:
    """
    Run a single ORB backtest on the given price DataFrame.

    Returns dict with standard metrics plus or_minutes, entry_buffer, hold_bars.
    """
    cost = cost_per_leg * cost_mult
    df = _compute_signals_orb(
        df, or_minutes=or_minutes, entry_buffer=entry_buffer,
        entry_cutoff_bars=entry_cutoff_bars, sessions=sessions,
    )

    or_signal = df["or_signal"].values.astype(int)
    ret = df["ret_1"].fillna(0).values.astype(float)
    session_labels = df["session_label"].values
    timestamps = df["timestamp"].values

    n_rows = len(df)
    equity = np.ones(n_rows + 1)
    position = 0
    bars_held = 0
    last_session_id = ""
    n_trades = 0

    for i in range(n_rows):
        sig = or_signal[i]
        ts = pd.Timestamp(timestamps[i])
        sess = str(session_labels[i])

        # Force close on hold expiry
        if position != 0 and bars_held >= hold_bars:
            equity[i + 1] = equity[i] * (1 - cost)
            n_trades += 1
            position = 0
            bars_held = 0
        else:
            equity[i + 1] = equity[i]

        if position == 0 and sig != 0 and sess not in ("", "other"):
            session_id = f"{ts.date()}_{sess}"
            if session_id != last_session_id:
                equity[i + 1] *= (1 - cost)
                n_trades += 1
                position = sig
                bars_held = 1
                last_session_id = session_id
        elif position != 0:
            equity[i + 1] *= 1 + position * ret[i]
            bars_held += 1

    cum_ret = float(equity[-1] - 1.0)

    eq = equity[1:]
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, 1)
    max_dd = float(np.max(dd))

    period_ret = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1)
    period_ret = period_ret[np.isfinite(period_ret)]
    if len(period_ret) > 1 and np.std(period_ret) > 0:
        sharpe = float(np.sqrt(BARS_PER_YEAR) * np.mean(period_ret) / np.std(period_ret))
    else:
        sharpe = 0.0

    gains = period_ret[period_ret > 0].sum()
    losses = abs(period_ret[period_ret < 0].sum())
    pf = float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)

    return {
        "or_minutes": or_minutes,
        "entry_buffer": entry_buffer,
        "hold_bars": hold_bars,
        "entry_cutoff_bars": entry_cutoff_bars,
        "net_return": cum_ret,
        "profit_factor": pf,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "n_trades": n_trades,
        "cost_mult": cost_mult,
    }


# ── v2 grid ───────────────────────────────────────────────────────────────────

def run_grid_v2(
    df: pd.DataFrame,
    or_minutes_values: list[int] | None = None,
    entry_buffer_values: list[float] | None = None,
    hold_values: list[int] | None = None,
    entry_cutoff_bars: int = 18,
    cost_mult: float = 1.0,
    save: bool = True,
) -> pd.DataFrame:
    """
    Run all combinations of (or_minutes, entry_buffer, hold_bars) — 18 combos by default.

    Returns DataFrame sorted by profit_factor descending.
    Saves to data/backtests_session_breakout/grid_results_v2.csv.
    """
    or_minutes_values = or_minutes_values or OR_MINUTES_VALUES
    entry_buffer_values = entry_buffer_values or ENTRY_BUFFER_VALUES
    hold_values = hold_values or HOLD_VALUES_V2

    combos = list(itertools.product(or_minutes_values, entry_buffer_values, hold_values))
    logging.info("[session_breakout_v2] Grid: %d combinations", len(combos))

    results = []
    for orm, eb, hb in combos:
        try:
            r = run_single_v2(
                df, or_minutes=orm, entry_buffer=eb, hold_bars=hb,
                entry_cutoff_bars=entry_cutoff_bars, cost_mult=cost_mult,
            )
            results.append(r)
            logging.info(
                "[session_breakout_v2] OR=%dm buf=%.4f hold=%d → net=%.4f PF=%.2f trades=%d",
                orm, eb, hb, r["net_return"], r["profit_factor"], r["n_trades"],
            )
        except Exception as exc:
            logging.warning(
                "[session_breakout_v2] combo OR=%dm buf=%.4f hold=%d failed: %s",
                orm, eb, hb, exc,
            )

    grid_df = pd.DataFrame(results).sort_values("profit_factor", ascending=False).reset_index(drop=True)

    if save:
        BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
        out = BACKTESTS_DIR / "grid_results_v2.csv"
        grid_df.to_csv(out, index=False)
        logging.info("[session_breakout_v2] Grid results saved to %s", out)

    return grid_df


# ── v2 walk-forward ───────────────────────────────────────────────────────────

def run_walk_forward_v2(
    df: pd.DataFrame,
    or_minutes: int = 30,
    entry_buffer: float = 0.0,
    hold_bars: int = 12,
    entry_cutoff_bars: int = 18,
    window_months: int = 1,
    save: bool = True,
    label: str = "",
) -> list[dict]:
    """
    Rolling walk-forward for ORB strategy: 1-month windows by default.

    Saves to data/backtests_session_breakout/walk_forward_v2_{label}.csv.
    """
    df = df.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df["_month"] = df["timestamp"].dt.to_period("M")
    months = sorted(df["_month"].unique())

    results: list[dict] = []
    step = window_months

    for start_idx in range(0, len(months) - step + 1, step):
        window = months[start_idx: start_idx + step]
        w_df = df[df["_month"].isin(window)].copy().reset_index(drop=True)
        if len(w_df) < 50:
            continue
        try:
            r = run_single_v2(
                w_df, or_minutes=or_minutes, entry_buffer=entry_buffer,
                hold_bars=hold_bars, entry_cutoff_bars=entry_cutoff_bars,
            )
            r["window_start"] = str(window[0])
            r["window_months"] = window_months
            results.append(r)
            logging.info(
                "[session_breakout_v2] WF %s (%dm): net=%.4f PF=%.2f trades=%d",
                window[0], window_months, r["net_return"], r["profit_factor"], r["n_trades"],
            )
        except Exception as exc:
            logging.warning("[session_breakout_v2] WF window %s failed: %s", window[0], exc)

    if save and results:
        BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
        suffix = label if label else f"_or{or_minutes}_buf{str(entry_buffer).replace('.','')}_h{hold_bars}"
        out = BACKTESTS_DIR / f"walk_forward_v2{suffix}.csv"
        pd.DataFrame(results).to_csv(out, index=False)
        logging.info("[session_breakout_v2] Walk-forward results saved to %s", out)

    return results


# ── v2 stability ──────────────────────────────────────────────────────────────

def run_stability_v2(
    df: pd.DataFrame,
    center_or_minutes: int = 30,
    center_entry_buffer: float = 0.0,
    center_hold: int = 12,
    entry_cutoff_bars: int = 18,
    save: bool = True,
) -> pd.DataFrame:
    """Perturb ±1 step around the chosen ORB config to detect cliff-edge overfitting."""
    or_candidates = sorted({max(5, center_or_minutes - 15), center_or_minutes, center_or_minutes + 15})
    buf_candidates = sorted({0.0, center_entry_buffer, round(center_entry_buffer + 0.0001, 4)})
    hold_candidates = sorted({max(2, center_hold - 6), center_hold, center_hold + 6})

    results = []
    for orm, eb, hb in itertools.product(or_candidates, buf_candidates, hold_candidates):
        try:
            r = run_single_v2(df, or_minutes=orm, entry_buffer=eb, hold_bars=hb,
                              entry_cutoff_bars=entry_cutoff_bars)
            r["is_center"] = (
                orm == center_or_minutes
                and eb == center_entry_buffer
                and hb == center_hold
            )
            results.append(r)
        except Exception as exc:
            logging.warning(
                "[session_breakout_v2] stability OR=%dm buf=%.4f hold=%d failed: %s",
                orm, eb, hb, exc,
            )

    stab_df = pd.DataFrame(results).sort_values("profit_factor", ascending=False).reset_index(drop=True)

    if save and not stab_df.empty:
        BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
        out = BACKTESTS_DIR / "stability_results_v2.csv"
        stab_df.to_csv(out, index=False)
        logging.info("[session_breakout_v2] Stability results saved to %s", out)

    return stab_df


# ── v2 cost stress ────────────────────────────────────────────────────────────

def run_cost_stress_v2(
    df: pd.DataFrame,
    or_minutes: int = 30,
    entry_buffer: float = 0.0,
    hold_bars: int = 12,
    entry_cutoff_bars: int = 18,
    cost_mults: list[float] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Run the same ORB config at different cost multipliers (1×, 1.5×, 2×)."""
    cost_mults = cost_mults or [1.0, 1.5, 2.0]
    results = []
    for mult in cost_mults:
        r = run_single_v2(
            df, or_minutes=or_minutes, entry_buffer=entry_buffer,
            hold_bars=hold_bars, entry_cutoff_bars=entry_cutoff_bars,
            cost_mult=mult,
        )
        results.append(r)
        logging.info(
            "[session_breakout_v2] Cost stress %sx: net=%.4f PF=%.2f trades=%d",
            mult, r["net_return"], r["profit_factor"], r["n_trades"],
        )

    stress_df = pd.DataFrame(results)

    if save and not stress_df.empty:
        BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
        out = BACKTESTS_DIR / "cost_stress_results_v2.csv"
        stress_df.to_csv(out, index=False)
        logging.info("[session_breakout_v2] Cost stress results saved to %s", out)

    return stress_df


# ── event study ───────────────────────────────────────────────────────────────

def run_event_study(
    df: pd.DataFrame,
    or_minutes_list: list[int] | None = None,
    entry_buffer: float = 0.0,
    fwd_bars: list[int] | None = None,
    entry_cutoff_bars: int = 18,
    save: bool = True,
) -> pd.DataFrame:
    """
    Event study: measure forward-return distribution at each ORB signal.

    For every candidate ORB signal event, record the realised forward returns
    at fwd_bars horizons.  Groups results by (or_minutes, session, direction)
    so you can directly compare ORB vs v1 rolling-breakout forward returns.

    Also computes equivalent stats for v1 rolling-breakout signals (N=8, best
    from grid) to give a side-by-side comparison — the primary go/no-go gate.

    Output columns per group:
        or_minutes, session, direction, horizon_bars,
        mean_fwd_ret, median_fwd_ret, hit_rate, n_events,
        mean_fwd_ret_net  (after 1× spread deducted)

    Saves to data/backtests_session_breakout/event_study.csv.
    """
    or_minutes_list = or_minutes_list or [15, 30, 60]
    fwd_bars = fwd_bars or [3, 6, 12]

    df = df.copy()
    df["ret_1"] = df["close"].pct_change()
    # Pre-compute forward returns for all horizons
    for h in fwd_bars:
        df[f"fwd_{h}"] = df["close"].pct_change(h).shift(-h)

    spread = FX_SPREAD  # cost per leg (one-way)

    records = []

    # ── ORB events ────────────────────────────────────────────────────────────
    for orm in or_minutes_list:
        df_orb = _compute_signals_orb(
            df, or_minutes=orm, entry_buffer=entry_buffer,
            entry_cutoff_bars=entry_cutoff_bars,
        )
        event_rows = df_orb[df_orb["or_signal"] != 0].copy()
        if event_rows.empty:
            logging.warning("[event_study] No ORB events found for OR=%dm", orm)
            continue

        for sess in ("london", "ny"):
            for direction, dir_label in [(1, "long"), (-1, "short")]:
                subset = event_rows[
                    (event_rows["session_label"] == sess)
                    & (event_rows["or_signal"] == direction)
                ]
                if subset.empty:
                    continue
                for h in fwd_bars:
                    fwd = subset[f"fwd_{h}"].dropna() * direction  # sign-adjusted
                    records.append({
                        "strategy": "v2_orb",
                        "or_minutes": orm,
                        "session": sess,
                        "direction": dir_label,
                        "horizon_bars": h,
                        "n_events": len(fwd),
                        "mean_fwd_ret": float(fwd.mean()) if len(fwd) else np.nan,
                        "median_fwd_ret": float(fwd.median()) if len(fwd) else np.nan,
                        "hit_rate": float((fwd > 0).mean()) if len(fwd) else np.nan,
                        "mean_fwd_ret_net": (float(fwd.mean()) - spread) if len(fwd) else np.nan,
                    })

    # ── v1 rolling-breakout events (comparison baseline) ─────────────────────
    df_v1 = _compute_signals(df, n=8, min_range=0.0002)
    v1_events = df_v1[df_v1["filtered_signal"] != 0].copy()
    hour = v1_events["timestamp"].dt.hour
    for sess, (h_lo, h_hi) in [("london", (7, 10)), ("ny", (12, 15))]:
        for direction, dir_label in [(1, "long"), (-1, "short")]:
            subset = v1_events[
                (hour >= h_lo) & (hour < h_hi)
                & (v1_events["filtered_signal"] == direction)
            ]
            if subset.empty:
                continue
            for h in fwd_bars:
                fwd = subset[f"fwd_{h}"].dropna() * direction
                records.append({
                    "strategy": "v1_rolling",
                    "or_minutes": 8,    # N=8 rolling window (best from grid)
                    "session": sess,
                    "direction": dir_label,
                    "horizon_bars": h,
                    "n_events": len(fwd),
                    "mean_fwd_ret": float(fwd.mean()) if len(fwd) else np.nan,
                    "median_fwd_ret": float(fwd.median()) if len(fwd) else np.nan,
                    "hit_rate": float((fwd > 0).mean()) if len(fwd) else np.nan,
                    "mean_fwd_ret_net": (float(fwd.mean()) - spread) if len(fwd) else np.nan,
                })

    event_df = pd.DataFrame(records)

    if save and not event_df.empty:
        BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
        out = BACKTESTS_DIR / "event_study.csv"
        event_df.to_csv(out, index=False)
        logging.info("[event_study] Results saved to %s", out)

    return event_df
