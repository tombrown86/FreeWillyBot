# Feature changelog

When modifying `src/build_features.py` (new columns, merge logic, thresholds), add an entry here.

## 2026-03-16

- Phase 18: Added Parquet output alongside CSV for train/validation/test.
- Added `USE_EXOGENOUS` and `USE_NEWS` overrides for ablation; when `USE_EXOGENOUS=False`, skip cross-asset and macro features.
- Added `--diagnose` flag to log NaN counts per column before save.

## Previous

- Price features: return_1, return_5, volatility_20, rsi_14, macd, atr_14, ma_gap_20, ma_gap_50.
- Cross-asset: SP500, VIX, GOLD, OIL, TREASURY_10Y returns and changes.
- Macro/event: is_cpi_window, is_fomc_day, minutes_to_event, policy_rate_level.
- Time: hour, weekday, is_london_session, is_ny_session.
- News: sentiment_score from GDELT GKG (merge_asof backward).
