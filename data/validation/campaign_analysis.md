# Validation campaign analysis

**Run date:** 2026-03-17

## Campaign steps executed

1. `freeze_test_set.py` — Snapshot test set to `data/frozen_test/`
2. `train_price_model.py --baselines-only` — Walk-forward validation (9 folds × 2 models)
3. `run_validate.py --mode monthly` — Rolling 1-month windows
4. `run_validate.py --months 1 6 12` — Default windows (1m, 6m, 12m)
5. `run_trade_consistency_report.py` — Trade count analysis
6. `backtest` — Full test-period backtest

---

## 1. Walk-forward (6–12 windows)

**Result:** 9 folds per model ✓ (within 6–12 target)

| Model   | Mean acc | Mean net_return | Mean n_trades | Std net_return | Folds |
|---------|----------|-----------------|---------------|----------------|-------|
| XGBoost | 52.1%    | -11.2%          | 984           | —              | 9     |
| LogReg  | 52.9%    | -3.9%           | 164           | —              | 9     |

**Chosen baseline:** LogReg (tie-break; LogReg has better OOS net return despite lower precision)

**Stability:** LogReg net return ranges from -9.1% (fold 5) to +1.9% (fold 8). One positive fold out of 9 — do not rely on one good month.

---

## 2. Monthly stability

**Result:** 10 positive / 27 months (37.0%)

| Metric           | Value |
|------------------|-------|
| Best month       | 2024-07 (+2.18%, sharpe 5.44) |
| Worst month      | 2025-10 (-6.20%, sharpe -11.22) |
| Positive months  | 10    |
| Total months     | 27    |

**Exceptional period check:** No single month contributes >50% of positive return — results are not driven by one exceptional month.

**Interpretation:** Positive months are spread across 2024-07, 2024-08, 2024-09, 2024-10, 2024-12, 2025-05, 2025-07, 2026-01, 2026-02, 2026-03. Performance is inconsistent; long periods of drawdown (e.g. 2024-01 to 2024-06, 2025-10 to 2025-12).

---

## 3. Trade count consistency

### Monthly windows

| Metric   | Value |
|----------|-------|
| Mean     | 158.7 |
| Std      | 32.2  |
| Min      | 64 (2026-03) |
| Max      | 221 (2025-07) |
| Max/Min  | 3.45 ⚠️ |

**Flag:** Unstable — max/min > 3. High variance may indicate weak thresholds or regime dependence.

**Why 64 vs 221?** 2026-03 is a partial month (run through Mar 16); 2025-07 is a full month. Normalize by trading days or bars when comparing. Also consider: fewer bars in volatile regimes (e.g. macro blackouts) may reduce trades.

### Walk-forward

| Model   | Mean  | Std   | Min   | Max   | Max/Min | Stable? |
|---------|-------|-------|-------|-------|---------|---------|
| XGBoost | 984   | 27.8  | 935   | 1031  | 1.10    | ✓       |
| LogReg  | 164   | 70.9  | 12    | 254   | 21.17   | ⚠️      |

**LogReg flag:** 12 trades (fold 1) vs 254 (fold 6) — extreme variance. Fold 1 (Oct 2023) may have different regime or LogReg is underconfident early in validation; fold 6 trades more aggressively. Investigate weak thresholds or regime dependence.

---

## 4. Rolling windows (1m, 6m, 12m)

| Window | Cum return | Max DD | Sharpe | Trades |
|--------|------------|--------|--------|-------|
| 1m     | +4.60%     | 5.64%  | 4.51   | 112   |
| 6m     | -3.14%     | 15.64% | -0.67  | 924   |
| 12m    | -13.25%    | 24.75% | -1.49  | 1985  |

**Interpretation:** Recent 1m window is positive; 6m and 12m are negative. Do not rely on 1m alone — longer windows show strategy underperforms over time.

---

## 5. Full backtest

| Metric   | Strategy | Momentum | Flat |
|----------|----------|----------|------|
| Cum ret  | -34.15%  | -99.98%  | 0%   |
| Max DD   | 41.32%   | 99.98%   | —    |
| Sharpe   | -2.23    | -21.16   | —    |
| Trades   | 4264     | —        | —    |

**Interpretation:** Strategy loses money on test; momentum baseline is worse. Flat baseline is best.

---

## 6. Frozen test

**Status:** `--use-frozen` failed with "No predictions overlap with frozen test timestamps."

**Cause:** `test_predictions.csv` was built before this freeze run. The frozen parquet has timestamps from the snapshot; predictions may be keyed to a different test set (e.g. extended range). Re-run `train_meta_model` after `freeze_test_set` to generate predictions on the frozen block, then run `--use-frozen`.

---

## Summary and recommendations

1. **Do not rely on one good month** — 1m window is positive but 6m/12m are negative. Monthly stability is 37% positive.
2. **LogReg trade frequency is unstable** — 12 vs 254 trades across folds. Investigate threshold calibration and regime differences.
3. **Monthly trade count variance** — 64 vs 221 trades; partial months and regime effects explain some. Consider `n_trades_per_bar` for normalization.
4. **Frozen test workflow** — Freeze before tuning; generate predictions after freeze; use `--use-frozen` for final evaluation to ensure untouched block.
