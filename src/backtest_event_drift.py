"""
Event drift backtest engine.

Simulates entering at T+1 (first bar after the macro event) in the direction
of the initial move, holding for a fixed number of bars, then exiting.

Configurations tested:
  baseline      : all events, enter in direction of initial move
  no_cpi_up     : exclude CPI up-moves (study showed they strongly revert)
  high_vol_only : only events where pre-event volatility is above median
  trend_agrees  : only when 4h MA10 trend aligns with initial move direction
  nfp_ecb_fomc  : only NFP / ECB / FOMC (highest-quality drift groups)

Public API
----------
run_single(df, hold_bars, cost_per_leg, filter_fn) -> (cum_ret, n_trades, max_dd, pf)
run_all(df, hold_bars, cost_per_leg) -> list[dict]

df must be the output of build_event_dataset() with columns:
  initial_move, ret_N (N = hold_bars), pre_vol, trend_agrees, event_name
"""

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _profit_factor(trade_rets: np.ndarray) -> float:
    gains  = trade_rets[trade_rets > 0].sum()
    losses = np.abs(trade_rets[trade_rets < 0].sum())
    return float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)


def run_single(
    df: pd.DataFrame,
    hold_bars: int = 6,
    cost_per_leg: float = 0.0001,
    filter_fn: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> dict:
    """Simulate the event drift strategy for one configuration.

    Parameters
    ----------
    df          : event dataset (from build_event_dataset)
    hold_bars   : exit after this many bars from T (default 6 = 30 min)
    cost_per_leg: round-trip = 2 × this (default 1 pip)
    filter_fn   : callable(df) -> bool Series — rows that pass the filter

    Returns
    -------
    dict with cum_ret, n_trades, max_dd, profit_factor, win_rate
    plus per-event rows for monthly breakdown
    """
    ret_col = f"ret_{hold_bars}"
    if ret_col not in df.columns:
        raise ValueError(f"Column {ret_col} not in dataset — rebuild with hold_bars={hold_bars}")

    sub = df.copy()
    if filter_fn is not None:
        mask = filter_fn(sub)
        sub = sub[mask].reset_index(drop=True)

    if sub.empty:
        return {"n_trades": 0, "cum_ret": 0.0, "max_dd": 0.0, "profit_factor": 0.0, "win_rate": 0.0}

    # Trade P&L = initial_move × ret_N − 2 × cost_per_leg
    sub["trade_ret_gross"] = sub["initial_move"] * sub[ret_col]
    sub["trade_ret_net"]   = sub["trade_ret_gross"] - 2 * cost_per_leg

    trade_rets = sub["trade_ret_net"].values.astype(float)
    n_trades = len(trade_rets)

    # Equity curve (multiplicative)
    equity = np.ones(n_trades + 1)
    for i, r in enumerate(trade_rets):
        equity[i + 1] = equity[i] * (1 + r)

    cum_ret = float(equity[-1] - 1.0)
    peak = np.maximum.accumulate(equity[1:])
    dd = (peak - equity[1:]) / np.where(peak > 0, peak, 1.0)
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
    pf = _profit_factor(trade_rets)
    win_rate = float((trade_rets > 0).mean())

    # Monthly breakdown
    if "event_time_utc" in sub.columns:
        sub["month"] = pd.to_datetime(sub["event_time_utc"]).dt.tz_convert(None).dt.to_period("M").astype(str)
        monthly = sub.groupby("month").agg(
            net_ret=("trade_ret_net", lambda x: float((1 + x).prod() - 1)),
            n_trades=("trade_ret_net", "count"),
        ).reset_index()
    else:
        monthly = pd.DataFrame()

    return {
        "n_trades": n_trades,
        "cum_ret": round(cum_ret, 6),
        "max_dd": round(max_dd, 6),
        "profit_factor": round(pf, 4),
        "win_rate": round(win_rate, 4),
        "avg_trade_pips": round(float(trade_rets.mean()) * 10000, 2),
        "monthly": monthly,
    }


def run_all(
    df: pd.DataFrame,
    hold_bars: int = 6,
    cost_per_leg: float = 0.0001,
) -> list[dict]:
    """Run all strategy configurations and return a list of result dicts."""
    pre_vol_median = df["pre_vol"].median()

    configs = {
        "baseline": None,
        "no_cpi_up": lambda d: ~((d["event_name"] == "CPI") & (d["initial_move"] == 1)),
        "high_vol": lambda d: d["pre_vol"] >= pre_vol_median,
        "trend_agrees": lambda d: d["trend_agrees"],
        "trend_disagrees": lambda d: ~d["trend_agrees"],
        "nfp_only": lambda d: d["event_name"] == "NFP",
        "cpi_only": lambda d: d["event_name"] == "CPI",
        "fomc_ecb": lambda d: d["event_name"].isin(["FOMC", "ECB"]),
        "high_vol_no_cpi_up": lambda d: (d["pre_vol"] >= pre_vol_median)
                                         & ~((d["event_name"] == "CPI") & (d["initial_move"] == 1)),
    }

    results = []
    for name, fn in configs.items():
        res = run_single(df, hold_bars=hold_bars, cost_per_leg=cost_per_leg, filter_fn=fn)
        res["config"] = name
        res["hold_bars"] = hold_bars
        res.pop("monthly", None)
        results.append(res)

    return results


def run_hold_sweep(
    df: pd.DataFrame,
    hold_bars_list: list[int] | None = None,
    cost_per_leg: float = 0.0001,
) -> list[dict]:
    """Sweep hold_bars for the best configurations."""
    if hold_bars_list is None:
        hold_bars_list = [1, 3, 6, 12]

    pre_vol_median = df["pre_vol"].median()
    configs = {
        "baseline":   None,
        "no_cpi_up":  lambda d: ~((d["event_name"] == "CPI") & (d["initial_move"] == 1)),
        "high_vol":   lambda d: d["pre_vol"] >= pre_vol_median,
    }

    results = []
    for h in hold_bars_list:
        for name, fn in configs.items():
            if f"ret_{h}" not in df.columns:
                continue
            res = run_single(df, hold_bars=h, cost_per_leg=cost_per_leg, filter_fn=fn)
            res["config"] = name
            res["hold_bars"] = h
            res.pop("monthly", None)
            results.append(res)

    return results
