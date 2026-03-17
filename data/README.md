# FreeWillyBot data directory

## Raw vs processed

**Raw data** (`data/raw/`) is append-only. Download scripts write here. Build and train modules never write to raw.

**Processed data** (`data/processed/`, `data/features/`) is overwritten on each run. Versioned copies are saved to `versions/` subdirectories.

## Missing-bar policy

**Policy: drop.** Missing 5min bars (e.g. weekends for FX, holidays, gaps) are excluded from the dataset. No forward-fill or interpolation. This avoids lookahead and preserves the actual trading schedule. Diagnostics report missing-bar counts in `data/validation/diagnostics_report.json`.

## Timezone assumptions

All timestamps are **UTC**. Price bars, cross-asset, macro, event calendar, and news use `utc=True` when parsing. Diagnostics verify timezone consistency across tables.

| Path | Contents |
|------|----------|
| `raw/price/` | Chunked price CSVs from Dukascopy/Binance/yfinance |
| `raw/cross_asset/` | SP500, VIX, GOLD, OIL |
| `raw/macro/` | CPI, FED_FUNDS, TREASURY_10Y, event_calendar |
| `raw/news/` | GDELT raw files |
| `processed/price/` | Clean 5min bars (CSV + Parquet) |
| `processed/price/versions/` | Versioned Parquet by date |
| `processed/aligned/` | Cross-asset and macro aligned to price index |
| `processed/aligned/versions/` | Versioned aligned tables |
| `features/` | train, validation, test (CSV + Parquet) |
| `features/versions/` | Versioned feature sets |
| `frozen_test/` | One-time test set snapshot for tuning |
| `validation/` | Diagnostics and validation reports |
