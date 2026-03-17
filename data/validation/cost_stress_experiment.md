# Cost stress experiment (frozen test, top 10%)

**Run date:** 2026-03-17

## Comparison

| Cost mult | Trades | Profit factor | Full return |
|-----------|--------|---------------|-------------|
| 1.0x | 2,070 | 0.96 | -15.51% |
| 1.5x | 2,082 | 0.93 | -24.23% |
| 2.0x | 2,098 | 0.90 | -31.55% |

## Findings

- Higher costs degrade returns: -15.5% at 1x to -31.6% at 2x.
- Profit factor drops from 0.96 to 0.90.
- Edge does not survive 1.5x or 2.0x costs; strategy is cost-sensitive.
