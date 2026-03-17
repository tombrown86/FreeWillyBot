# Spread proxy stress test

**Use frozen:** True

| Regime | Bars | Trades | Cum return | Profit factor | Max DD | Sharpe |
|--------|------|--------|------------|---------------|--------|--------|
| normal-vol (bottom 50%, vol <= 0.000181) | 82281 | 1780 | -0.1852 | 0.85 | 0.2360 | -4.80 |
| high-vol (top 50%, vol > 0.000181) | 82281 | 460 | -0.0733 | 0.96 | 0.0988 | -1.20 |

**Note:** SPREAD_PROXY_VOLATILITY_PCT (0.003) never blocks bars in this dataset (vol_20 max ~0.0026). Used median split for comparison.

**Interpretation:** High-vol (top 50%) had better cum_ret and PF than normal-vol here. The fixed threshold may be too high for EURUSD; consider lowering or using percentile-based rule.
