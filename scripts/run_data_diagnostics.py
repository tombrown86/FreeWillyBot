"""
Phase 17 — Data quality diagnostics.

Run after run_daily_data_refresh (or on existing data).
Reports: timestamp consistency, missing values, duplicated bars.
Output: data/validation/diagnostics_report.json
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BAR_INTERVAL, CRYPTO_SKIP_WEEKEND, MAX_BAR_RETURN_PCT, SYMBOL, TEST_START_DATE

VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"
RAW_PRICE_DIR = PROJECT_ROOT / "data" / "raw" / "price"
PROCESSED_PRICE = PROJECT_ROOT / "data" / "processed" / "price"
ALIGNED_DIR = PROJECT_ROOT / "data" / "processed" / "aligned"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
PROCESSED_NEWS_DIR = PROJECT_ROOT / "data" / "processed" / "news"
RAW_MACRO_DIR = PROJECT_ROOT / "data" / "raw" / "macro"


def _load_processed_price():
    """Load clean price bars; None if not found. Uses shared utils for corrupt-parquet handling."""
    try:
        from src.utils import load_processed_price

        return load_processed_price(PROCESSED_PRICE, SYMBOL, BAR_INTERVAL)
    except FileNotFoundError:
        return None


def _check_raw_price_duplicates() -> dict:
    """Check raw price files for duplicate timestamps."""
    import pandas as pd

    report = {"stage": "raw_price", "duplicates": 0, "total_rows": 0, "unique_timestamps": 0}
    files = list(RAW_PRICE_DIR.glob("*.csv"))
    if not files:
        report["error"] = "No raw price files found"
        return report

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.lower().strip() for c in df.columns]
        col = "timestamp" if "timestamp" in df.columns else "date"
        if col not in df.columns:
            continue
        df[col] = pd.to_datetime(df[col], utc=True)
        dfs.append(df)

    if not dfs:
        report["error"] = "No valid price data"
        return report

    combined = pd.concat(dfs, ignore_index=True)
    report["total_rows"] = len(combined)
    report["unique_timestamps"] = combined["timestamp"].nunique()
    report["duplicates"] = report["total_rows"] - report["unique_timestamps"]
    return report


def _check_processed_bars() -> dict:
    """Check clean bars for duplicates and timestamp consistency."""
    import pandas as pd

    report = {"stage": "processed_bars", "duplicates": 0, "total_rows": 0, "has_tz": True}
    path = PROCESSED_PRICE / f"{SYMBOL}_{BAR_INTERVAL}_clean.csv"
    if not path.exists():
        report["error"] = f"File not found: {path}"
        return report

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    report["total_rows"] = len(df)
    report["duplicates"] = len(df) - df["timestamp"].nunique()
    report["has_tz"] = df["timestamp"].dt.tz is not None
    return report


def _check_features_missing_values() -> dict:
    """Check feature files for NaN rates per column."""
    import pandas as pd

    report = {"stage": "features", "files": {}, "critical_nan_columns": []}
    for name in ["train", "validation", "test"]:
        path = FEATURES_DIR / f"{name}.csv"
        if not path.exists():
            report["files"][name] = {"error": "not found"}
            continue
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        nan_pct = (df.isna().sum() / len(df) * 100).to_dict()
        report["files"][name] = {
            "rows": len(df),
            "nan_pct": {k: round(v, 2) for k, v in nan_pct.items() if v > 0},
        }
        critical = [c for c, p in nan_pct.items() if p > 50 and c not in ("timestamp",)]
        if critical:
            report["critical_nan_columns"].extend(critical)
    report["critical_nan_columns"] = list(set(report["critical_nan_columns"]))
    return report


def _check_forecaster_merge() -> dict:
    """Check forecaster predictions vs test for timestamp alignment."""
    import pandas as pd

    report = {"stage": "forecaster_merge", "test_rows": 0, "forecaster_rows": 0, "merged_rows": 0, "nan_forecaster_pct": 0}
    test_path = FEATURES_DIR / "test.csv"
    fc_path = FEATURES_DIR / "forecaster_predictions.csv"
    if not test_path.exists():
        report["error"] = "test.csv not found"
        return report
    if not fc_path.exists():
        report["error"] = "forecaster_predictions.csv not found"
        return report

    test = pd.read_csv(test_path)
    test["timestamp"] = pd.to_datetime(test["timestamp"], utc=True)
    fc = pd.read_csv(fc_path)
    fc["timestamp"] = pd.to_datetime(fc["timestamp"], utc=True)

    merged = test.merge(fc[["timestamp", "chronos_pred_return", "timesfm_pred_return"]], on="timestamp", how="left")
    report["test_rows"] = len(test)
    report["forecaster_rows"] = len(fc)
    report["merged_rows"] = len(merged)
    chronos_nan = merged["chronos_pred_return"].isna().sum() / len(merged) * 100 if "chronos_pred_return" in merged.columns else 0
    report["nan_forecaster_pct"] = round(chronos_nan, 2)
    return report


def _check_missing_bars() -> dict:
    """Check for missing 5min bars vs expected schedule. FX: exclude weekends; crypto: 24/7."""
    import pandas as pd

    report = {"stage": "missing_bars", "expected": 0, "actual": 0, "missing": 0, "missing_pct": 0, "gap_samples": []}
    df = _load_processed_price()
    if df is None:
        report["error"] = f"File not found: {PROCESSED_PRICE / f'{SYMBOL}_{BAR_INTERVAL}_clean.csv'}"
        return report
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    first = df["timestamp"].min()
    last = df["timestamp"].max()
    expected_index = pd.date_range(start=first, end=last, freq="5min", tz="UTC")
    if "USDT" not in SYMBOL:
        expected_index = expected_index[expected_index.dayofweek < 5]
    actual_set = set(df["timestamp"].values)
    expected_set = set(expected_index.values)
    missing = expected_set - actual_set
    report["expected"] = len(expected_set)
    report["actual"] = len(actual_set)
    report["missing"] = len(missing)
    report["missing_pct"] = round(100 * len(missing) / len(expected_set), 2) if expected_set else 0

    if missing:
        missing_sorted = sorted(missing)
        gaps = []
        gap_start = missing_sorted[0]
        gap_count = 1
        for i in range(1, len(missing_sorted)):
            if (pd.Timestamp(missing_sorted[i]) - pd.Timestamp(missing_sorted[i - 1])).total_seconds() > 300:
                gaps.append({"start": str(pd.Timestamp(gap_start)), "count": gap_count})
                gap_start = missing_sorted[i]
                gap_count = 1
            else:
                gap_count += 1
        gaps.append({"start": str(pd.Timestamp(gap_start)), "count": gap_count})
        report["gap_samples"] = gaps[:5]
    return report


def _check_zero_prices() -> dict:
    """Check raw and processed price for zero or negative OHLC."""
    import pandas as pd

    report = {"stage": "zero_prices", "raw": {}, "processed": {}}
    ohlc = ["open", "high", "low", "close"]

    files = list(RAW_PRICE_DIR.glob("*.csv"))
    if files:
        dfs = []
        for f in files[:10]:
            df = pd.read_csv(f)
            df.columns = [c.lower().strip() for c in df.columns]
            dfs.append(df)
        raw = pd.concat(dfs, ignore_index=True)
        for col in ohlc:
            if col not in raw.columns:
                continue
            zero_count = (raw[col] == 0).sum()
            neg_count = (raw[col] < 0).sum()
            if zero_count > 0 or neg_count > 0:
                report["raw"][col] = {"zero": int(zero_count), "negative": int(neg_count)}
        report["raw"]["total_rows"] = len(raw)
    else:
        report["raw"]["error"] = "No raw files"

    proc = _load_processed_price()
    if proc is not None:
        for col in ohlc:
            if col not in proc.columns:
                continue
            zero_count = (proc[col] == 0).sum()
            neg_count = (proc[col] < 0).sum()
            if zero_count > 0 or neg_count > 0:
                report["processed"][col] = {"zero": int(zero_count), "negative": int(neg_count)}
        report["processed"]["total_rows"] = len(proc)
    else:
        report["processed"]["error"] = "Not found"
    return report


def _check_impossible_jumps() -> dict:
    """Flag bars where abs(close.pct_change()) > MAX_BAR_RETURN_PCT."""
    import pandas as pd

    report = {"stage": "impossible_jumps", "count": 0, "max_return_pct": 0, "sample_timestamps": []}
    df = _load_processed_price()
    if df is None:
        report["error"] = f"File not found: {PROCESSED_PRICE / f'{SYMBOL}_{BAR_INTERVAL}_clean.csv'}"
        return report
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "close" not in df.columns:
        report["error"] = "No close column"
        return report

    ret = df["close"].pct_change()
    threshold = MAX_BAR_RETURN_PCT
    flags = (ret.abs() > threshold) & ret.notna()
    report["count"] = int(flags.sum())
    if flags.any():
        report["max_return_pct"] = round(100 * ret.abs().max(), 2)
        report["sample_timestamps"] = df.loc[flags, "timestamp"].head(5).astype(str).tolist()
    return report


def _check_aligned_duplicates() -> dict:
    """Check aligned tables for duplicate timestamps."""
    import pandas as pd

    report = {"stage": "aligned", "cross_asset_duplicates": 0, "macro_duplicates": 0}
    for name, key in [("cross_asset", "cross_asset_aligned.csv"), ("macro", "macro_aligned.csv")]:
        path = ALIGNED_DIR / key
        if not path.exists():
            report[f"{name}_duplicates"] = -1
            report[f"{name}_error"] = "not found"
            continue
        df = pd.read_csv(path)
        if "timestamp" not in df.columns and "index" in df.columns:
            df = df.rename(columns={"index": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        dupes = len(df) - df["timestamp"].nunique()
        report[f"{name}_duplicates"] = dupes
    return report


def _check_bar_spacing() -> dict:
    """Verify consecutive bars are exactly 5 minutes apart; gaps (multiples of 5min) allowed."""
    import pandas as pd

    report = {"stage": "bar_spacing", "valid": True, "invalid_diffs": 0, "sample_invalid": []}
    df = _load_processed_price()
    if df is None:
        report["error"] = "Price file not found"
        return report
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    diffs = df["timestamp"].diff().dropna()
    # Valid: exactly 5min or gap (multiple of 5min). Invalid: 0, or between 0 and 5min, or not multiple of 5min.
    secs = diffs.dt.total_seconds()
    invalid = (secs < 300) | ((secs >= 300) & (secs % 300 != 0))
    report["invalid_diffs"] = int(invalid.sum())
    report["valid"] = not invalid.any()
    if invalid.any():
        idx = invalid.idxmax()
        report["sample_invalid"] = [
            str(df.loc[idx - 1, "timestamp"]),
            str(df.loc[idx, "timestamp"]),
            str(diffs.loc[idx]),
        ]
    return report


def _check_timezone_consistency() -> dict:
    """Assert all timestamps have tz=UTC across price, aligned, events, news, features."""
    import pandas as pd

    report = {"stage": "timezone_consistency", "valid": True, "issues": []}
    checks = []

    # Price
    df = _load_processed_price()
    if df is not None:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if df["timestamp"].dt.tz is None:
            report["issues"].append("price: timestamps have no timezone")
        checks.append(("price", len(df)))

    # Aligned
    for name, key in [("cross_asset", "cross_asset_aligned.csv"), ("macro", "macro_aligned.csv")]:
        path = ALIGNED_DIR / key
        if path.exists():
            df = pd.read_csv(path)
            if "timestamp" not in df.columns and "index" in df.columns:
                df = df.rename(columns={"index": "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            if df["timestamp"].dt.tz is None:
                report["issues"].append(f"{name}: timestamps have no timezone")
            checks.append((name, len(df)))

    # Event calendar
    events_path = RAW_MACRO_DIR / "event_calendar.csv"
    if events_path.exists():
        df = pd.read_csv(events_path)
        if "event_timestamp_utc" in df.columns:
            ts = pd.to_datetime(df["event_timestamp_utc"], utc=True)
            if ts.dt.tz is None:
                report["issues"].append("event_calendar: event_timestamp_utc has no timezone")

    # News
    if PROCESSED_NEWS_DIR.exists():
        for f in list(PROCESSED_NEWS_DIR.glob("*.csv"))[:5]:
            df = pd.read_csv(f)
            if "timestamp" in df.columns:
                ts = pd.to_datetime(df["timestamp"], utc=True)
                if ts.dt.tz is None:
                    report["issues"].append(f"news {f.name}: timestamps have no timezone")
                    break

    # Features
    for name in ["train", "validation", "test"]:
        path = FEATURES_DIR / f"{name}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            if df["timestamp"].dt.tz is None:
                report["issues"].append(f"features {name}: timestamps have no timezone")
            break

    report["valid"] = len(report["issues"]) == 0
    return report


def _check_news_duplicates() -> dict:
    """Check news files for duplicate timestamps."""
    import pandas as pd

    report = {"stage": "news_duplicates", "files": {}, "total_duplicates": 0}
    if not PROCESSED_NEWS_DIR.exists():
        report["error"] = "News dir not found"
        return report
    files = list(PROCESSED_NEWS_DIR.glob("*.csv"))
    if not files:
        report["error"] = "No news CSV files"
        return report
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.lower().strip() for c in df.columns]
        if "timestamp" not in df.columns:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        dupes = len(df) - df["timestamp"].nunique()
        report["files"][f.name] = {"rows": len(df), "duplicates": int(dupes)}
        report["total_duplicates"] += dupes
    return report


def _check_no_label_leakage() -> dict:
    """Assert no feature uses shift(-k) for k>0; only labels may look forward."""
    report = {"stage": "no_label_leakage", "valid": True, "violations": []}
    build_features_path = PROJECT_ROOT / "src" / "build_features.py"
    if not build_features_path.exists():
        report["error"] = "build_features.py not found"
        return report
    content = build_features_path.read_text()
    lines = content.splitlines()
    for i, line in enumerate(lines, 1):
        if "shift(-" in line or "shift( -" in line:
            if "future_return_30m" in line or "LABEL_HORIZON" in line:
                continue
            report["violations"].append({"line": i, "content": line.strip()[:80]})
    report["valid"] = len(report["violations"]) == 0
    return report


def _check_merged_features_duplicates() -> dict:
    """Check train/validation/test for duplicate timestamps."""
    import pandas as pd

    report = {"stage": "merged_features_duplicates", "files": {}, "total_duplicates": 0}
    for name in ["train", "validation", "test"]:
        path = FEATURES_DIR / f"{name}.csv"
        if not path.exists():
            report["files"][name] = {"error": "not found"}
            continue
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        dupes = len(df) - df["timestamp"].nunique()
        report["files"][name] = {"rows": len(df), "duplicates": int(dupes)}
        report["total_duplicates"] += dupes
    return report


def run() -> dict:
    """Run all diagnostics. Return report dict."""
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "run_at": datetime.now().isoformat(),
        "checks": [],
    }

    for check_fn in [
        _check_raw_price_duplicates,
        _check_processed_bars,
        _check_aligned_duplicates,
        _check_missing_bars,
        _check_bar_spacing,
        _check_zero_prices,
        _check_impossible_jumps,
        _check_no_label_leakage,
        _check_timezone_consistency,
        _check_news_duplicates,
        _check_merged_features_duplicates,
        _check_features_missing_values,
        _check_forecaster_merge,
    ]:
        try:
            result = check_fn()
            report["checks"].append(result)
        except Exception as e:
            report["checks"].append({"stage": check_fn.__name__, "error": str(e)})

    out_path = VALIDATION_DIR / "diagnostics_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info("Diagnostics report saved to %s", out_path)
    return report


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run()
