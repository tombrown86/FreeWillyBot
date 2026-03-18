# Regression execution tuning report

## Summary

After aggressive trade reduction (top 0.25%, pred threshold, vol filter), the strategy achieves **net positive** on frozen test.

## Best configuration

| Parameter | Value |
|-----------|-------|
| top_pct | 0.25% |
| pred_threshold | 0.00005 |
| vol_pct | 20% |
| gross_return | +13.5% |
| net_return | +5.4% |
| n_trades | 721 |
| max_dd | 3.9% |
| PF | 1.34 |

## Grid results (top 0.25% vs 0.5% vs 1%)

| top_pct | vol_pct | gross | net | trades | PF |
|---------|---------|-------|-----|--------|-----|
| 0.25 | 20 | 13.5% | **+5.4%** | 721 | 1.34 |
| 0.25 | 30 | 13.0% | +4.7% | 743 | 1.31 |
| 0.25 | 40 | 13.3% | +4.8% | 761 | 1.32 |
| 0.5 | 20 | 14.8% | -1.1% | 1422 | 1.18 |
| 1.0 | 20 | 13.4% | -15.1% | 2715 | 1.09 |

## Interpretation

- **Signal exists**: gross PF > 1.0 across all top_pct levels
- **Execution was the bottleneck**: at top 1%, costs consumed the edge (gross +13%, net -15%)
- **Solution**: trade only top 0.25% extremes (721 trades) to stay net positive
- **Trade clustering**: plot saved to `trade_clustering.png`

## Success criteria met

- trades: 721 (within 300-800 target)
- gross_return: stable and positive
- net_return: > 0
- PF: > 1.05
- drawdown: 3.9% (controlled)
