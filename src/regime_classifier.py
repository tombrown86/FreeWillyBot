"""
Regime classifier — rule-based regime gate for regression strategy.

Three signals:
  A. Vol level:       vol_6 >= median(vol_6)       → high vol environment
  B. Vol change:      vol_change > 0               → rising vol
  C. Trend strength:  abs(ret_12) >= threshold     → strong directional move

Combine with mode "and" (all must pass) or "or" (any must pass).
Returns a boolean array: True = trade allowed this bar.
"""

import numpy as np


def build_regime_mask(
    vol: np.ndarray,
    ret_12: np.ndarray,
    vol_change: np.ndarray,
    vol_above_median: bool = True,
    vol_rising: bool | None = None,
    ret_12_threshold: float = 0.0,
    mode: str = "and",
) -> np.ndarray:
    """
    Rule-based regime classifier.

    Parameters
    ----------
    vol : vol_6 array (rolling 6-bar std of returns)
    ret_12 : 12-bar past return (trend strength proxy)
    vol_change : vol_6 - vol_6.shift(1) (positive = rising vol)
    vol_above_median : Signal A — trade only when vol >= median(vol)
    vol_rising : Signal B — None = ignore; True = require rising vol; False = require falling vol
    ret_12_threshold : Signal C — trade only when abs(ret_12) >= threshold (0.0 = disabled)
    mode : "and" = all active signals must pass; "or" = any active signal must pass

    Returns
    -------
    np.ndarray of bool, same length as vol
    """
    signals = []

    if vol_above_median:
        signals.append(vol >= np.nanmedian(vol))

    if vol_rising is True:
        signals.append(vol_change > 0)
    elif vol_rising is False:
        signals.append(vol_change < 0)

    if ret_12_threshold > 0:
        signals.append(np.abs(ret_12) >= ret_12_threshold)

    if not signals:
        return np.ones(len(vol), dtype=bool)

    if mode == "and":
        mask = signals[0].copy()
        for s in signals[1:]:
            mask &= s
    else:
        mask = signals[0].copy()
        for s in signals[1:]:
            mask |= s

    return mask


def regime_stats(mask: np.ndarray) -> dict:
    """Return summary stats for a regime mask."""
    pct_active = float(mask.mean() * 100)
    return {
        "pct_active": pct_active,
        "n_active": int(mask.sum()),
        "n_total": len(mask),
    }
