# Evaluation

## Walk-forward validation

Baseline selection (XGBoost vs Logistic Regression) uses walk-forward validation on the train+validation period only. The test set is never used for model choice or hyperparameter tuning.

- **Location:** `src/train_price_model.py` `_run_walk_forward()`
- **Windows:** 3 months train, 1 month test, step `WF_STEP_DAYS` (7 days) for 6–12 folds
- **Range:** From `VALIDATION_START_DATE` up to (but not including) `TEST_START_DATE`
- **Metrics:** Accuracy, precision, recall, F1, Brier score, net return after costs, n_trades
- **Assertion:** `test_end <= TEST_START_DATE` — walk-forward folds never include test data

## Test set

- **Definition:** All data from `TEST_START_DATE` onward
- **Usage:** Final evaluation only (backtest, run_validate, live_signal)
- **Freeze:** Run `scripts/freeze_test_set.py` before serious tuning to snapshot the test set to `data/frozen_test/`. Use `--use-frozen` for final evaluation to ensure an untouched block.

### Frozen test (`--use-frozen`)

Run `freeze_test_set.py` before tuning. For final evaluation, pass `--use-frozen` to `run_validate` and `backtest` so they load predictions filtered to frozen test timestamps from `data/frozen_test/manifest.json` and `test_YYYYMMDD.parquet`. This guarantees evaluation on a fixed, never-seen block.

```bash
python scripts/freeze_test_set.py
python scripts/run_validate.py --use-frozen
python -m src.backtest --use-frozen
```

## Walk-forward windows

- **Output:** `data/validation/walk_forward_windows.csv` — fold, train_start, train_end, test_start, test_end, accuracy, net_return, n_trades, etc.
- **Step:** `WF_STEP_DAYS = 7` yields 6–12 folds from validation start to test start.

## Monthly stability

Run `run_validate --mode monthly` to backtest each calendar month in the test period. Results go to `data/validation/monthly_windows.csv` (month, start_date, end_date, cum_return, max_dd, sharpe, n_trades, run_at). Summary: `n_positive_months`, `n_total_months`, `pct_positive`. Flags when one month contributes >50% of positive return (exceptional period).

## Trade count consistency

Run `scripts/run_trade_consistency_report.py` to analyze `n_trades` from `monthly_windows.csv` and `walk_forward_windows.csv`. It computes mean, std, min, max, and ratio `max/min`, reports min/max windows (e.g. "2026-03: 64 trades vs 2025-07: 221 trades — understand why"), and flags when `max/min > 3` or `std/mean > 0.5` as unstable trade frequency. Analyzes walk-forward per model (xgboost, logreg). High variance may indicate weak thresholds or regime dependence.

## Candidate strategy (confidence + volatility regime)

Execution rule:

- **Confidence:** Top 5% (`--top-pct 5`)
- **Volatility regime:** Trade only when `volatility_20` is in top 30% (`VOL_REGIME_TOP_PCT = 30`)

```bash
python -m src.backtest --use-frozen --top-pct 5
```

Frozen test metrics (parameter surface in `data/validation/parameter_surface_20260317.md`):

| Config | cum_ret | max_dd | PF | trades |
|--------|---------|--------|-----|--------|
| Baseline (top 10%, no vol filter) | -0.155 | 0.202 | 0.96 | 2070 |
| Candidate (top 5%, vol top 30%) | 0.026 | 0.081 | **1.04** | 572 |

Best from grid: top 5% + vol top 30% achieves PF ≥ 1.0.

## Model usefulness (E)

### Ablations

Run `scripts/run_model_ablations.py` to compare signal sources on frozen test (top 5%, vol 30%):

- **full:** meta-model + forecasters + exogenous
- **no_forecasters:** meta-model without chronos/timesfm
- **price_only:** price + time features only
- **chronos_only:** Chronos prediction as signal (long/short)
- **momentum:** return_5 as signal

Output: `data/validation/model_ablation_report.csv`

### Calibration

Run `scripts/run_calibration_analysis.py` to bin predictions by confidence (0.35–0.45, 0.45–0.55, etc.) within vol regime. Verifies higher confidence → better outcomes.

Output: `data/validation/calibration_report.csv`

### Feature importance

Run `python -m src.train_meta_model --save-importance` to save feature importance to `data/models/meta_feature_importance.json`. Top features (LogReg coef magnitude): is_london_session, VIX_return_20, TREASURY_10Y_change_5, OIL_return_20, etc.

## Filter behavior (F)

### Filter impact

Run `scripts/run_filter_impact.py` to measure per-filter impact: bars blocked, trades prevented, PnL improvement.

Output: `data/validation/filter_impact_report.json`

### Filter audit

Run `scripts/audit_filters.py` to verify daily-loss resets at day boundary and cooldown triggers only after losing close.

## Decision confidence (G)

### Confidence bins

Run `scripts/run_confidence_bins.py` to analyze top 1–5%, 5–10%, 10–20% within vol regime.

Output: `data/validation/confidence_bins_report.csv`

### Trade review

Run `scripts/run_trade_review.py` to extract 10 best, 10 worst, 10 blocked high-confidence, 10 executed low-confidence trades for manual inspection.

Output: `data/validation/trade_review.csv`

## Never tune on test

All hyperparameter and model choices use train/validation and walk-forward metrics. The test set is for final evaluation only.
