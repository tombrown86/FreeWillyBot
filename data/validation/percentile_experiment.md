# Percentile filter experiment (frozen test)

**Run date:** 2026-03-17

## Comparison

| Filter | Trades | Profit factor | Full return |
|--------|--------|---------------|-------------|
| **Threshold 0.35** (baseline) | 4,609 | 0.98 | -13.24% |
| **Top 10%** | 2,070 | 0.96 | -15.51% |
| **Top 20%** | 5,860 | 0.92 | -44.97% |
| **Top 30%** | 7,006 | 0.94 | -42.19% |

## Findings

- **Top 10%:** Fewest trades (2,070), highest profit factor (0.96), least negative return (-15.5%). Best trade quality.
- **Top 20% / 30%:** More trades, worse returns. Adding lower-confidence signals degrades performance.
- **Baseline (0.35):** More trades than top 10% but better return (-13.2%). The threshold filter may be interacting differently with macro/vol/session filters than the percentile filter.

## Recommendation

Top 10% gives the best trade quality (fewest trades, highest PF). For further tuning: cost stress test, ablation, and filter effectiveness (checklist D).
