# Confidence threshold comparison (frozen test only)

**Model max confidence:** ~0.54 (no bar has max(P_buy, P_sell) ≥ 0.55)

| Threshold | Trades | Profit factor | Full return |
|-----------|--------|---------------|-------------|
| **0.35** (baseline) | 4,609 | 0.98 | -13.24% |
| **0.60** | 0 | 0.00 | 0% |

**Note:** 0.60 is above the model's maximum output. The meta-model never predicts P_buy or P_sell ≥ 0.55, so thresholds 0.55–0.65 yield zero trades.

**Implication:** To reduce overtrading via threshold alone, you need a model that outputs higher confidence. Options:
1. Retrain with calibration (e.g. temperature scaling) to sharpen probabilities
2. Use a different model that produces more confident predictions
3. Use 0.45–0.50 as a practical upper bound with this model (~1,248 bars ≥ 0.50)
