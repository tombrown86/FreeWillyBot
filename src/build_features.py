"""
Phase 10 — Build features and labels.

Creates price, cross-asset, macro/event, time, and optional news (FinBERT) features.
Builds labels (future_return_30m) and classes (buy/sell/no-trade).
Splits by time and saves train/validation/test to data/features/.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    BAR_INTERVAL,
    BUY_THRESHOLD_PCT,
    LABEL_HORIZON_BARS,
    SELL_THRESHOLD_PCT,
    SYMBOL,
    TEST_START_DATE,
    TRAINING_START_DATE,
    USE_EXOGENOUS,
    USE_NEWS,
    VALIDATION_START_DATE,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PRICE = PROJECT_ROOT / "data" / "processed" / "price"
ALIGNED_DIR = PROJECT_ROOT / "data" / "processed" / "aligned"
RAW_MACRO_DIR = PROJECT_ROOT / "data" / "raw" / "macro"
PROCESSED_NEWS_DIR = PROJECT_ROOT / "data" / "processed" / "news"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
LOG_FILE = PROJECT_ROOT / "data" / "logs" / "build_features.log"


def _setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal_line
    return macd_line, macd_signal_line, macd_hist


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _load_processed(path_stem: Path, name: str) -> pd.DataFrame | None:
    """Load from Parquet if exists, else CSV. path_stem is dir, name is base like 'cross_asset_aligned'."""
    parquet_path = path_stem / f"{name}.parquet"
    csv_path = path_stem / f"{name}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, list[Path]]:
    """Load price, cross-asset, macro, events, news. Return None for missing. Prefers Parquet over CSV."""
    from src.utils import load_processed_price

    price = load_processed_price(PROCESSED_PRICE, SYMBOL, BAR_INTERVAL)
    price["timestamp"] = pd.to_datetime(price["timestamp"], utc=True)
    price = price.sort_values("timestamp").reset_index(drop=True)

    cross = _load_processed(ALIGNED_DIR, "cross_asset_aligned")
    if cross is not None:
        cross["timestamp"] = pd.to_datetime(cross["timestamp"], utc=True)
        if cross.empty:
            cross = None
            logging.warning("Cross-asset file empty")
    else:
        logging.warning("Cross-asset file not found: %s", ALIGNED_DIR / "cross_asset_aligned")

    macro = _load_processed(ALIGNED_DIR, "macro_aligned")
    if macro is not None:
        macro["timestamp"] = pd.to_datetime(macro["timestamp"], utc=True)
        if macro.empty:
            macro = None
            logging.warning("Macro file empty")
    else:
        logging.warning("Macro file not found: %s", ALIGNED_DIR / "macro_aligned")

    events = None
    events_path = RAW_MACRO_DIR / "event_calendar.csv"
    if events_path.exists():
        events = pd.read_csv(events_path)
        events["event_timestamp_utc"] = pd.to_datetime(events["event_timestamp_utc"], utc=True)
        if events.empty:
            events = None
    else:
        logging.warning("Event calendar not found: %s", events_path)

    news_files = list(PROCESSED_NEWS_DIR.glob("*.csv")) if PROCESSED_NEWS_DIR.exists() else []
    return price, cross, macro, events, news_files


def build_price_features(price: pd.DataFrame) -> pd.DataFrame:
    """Price features: returns, volatility, RSI, MACD, ATR, MA gaps."""
    df = price.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    df["return_1"] = close.pct_change(1)
    df["return_5"] = close.pct_change(5)
    df["volatility_20"] = df["return_1"].rolling(20).std()

    df["rsi_14"] = _rsi(close, 14)
    macd_line, macd_signal_line, macd_hist = _macd(close)
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal_line
    df["macd_hist"] = macd_hist
    df["atr_14"] = _atr(high, low, close, 14)

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    df["ma_gap_20"] = (close - ma20) / close.replace(0, np.nan)
    df["ma_gap_50"] = (close - ma50) / close.replace(0, np.nan)

    return df


def build_cross_asset_features(
    price: pd.DataFrame, cross: pd.DataFrame | None, macro: pd.DataFrame | None
) -> pd.DataFrame:
    """Cross-asset features: 1-bar, 5-bar, 20-bar returns for SP500, VIX, GOLD, OIL, TREASURY_10Y."""
    df = price[["timestamp"]].copy()
    if cross is None and macro is None:
        return df

    merged = price[["timestamp"]].copy()
    if cross is not None:
        merged = merged.merge(cross, on="timestamp", how="left")
    if macro is not None:
        for col in ["TREASURY_10Y"]:
            if col in macro.columns:
                m = macro[["timestamp", col]].copy()
                merged = merged.merge(m, on="timestamp", how="left")

    assets = ["SP500", "VIX", "GOLD", "OIL", "TREASURY_10Y"]
    for asset in assets:
        if asset not in merged.columns:
            continue
        s = merged[asset].ffill()
        if s.isna().all():
            continue
        s = s.fillna(0)
        df[f"{asset}_return_1"] = s.pct_change(1)
        df[f"{asset}_return_5"] = s.pct_change(5)
        df[f"{asset}_return_20"] = s.pct_change(20)
        if asset == "TREASURY_10Y":
            df[f"{asset}_change_1"] = s.diff(1)
            df[f"{asset}_change_5"] = s.diff(5)

    return df


def build_macro_event_features(
    price: pd.DataFrame, macro: pd.DataFrame | None, events: pd.DataFrame | None
) -> pd.DataFrame:
    """Macro/event features: is_cpi_window, is_fomc_day, minutes_to_event, policy_rate_level."""
    df = price[["timestamp"]].copy()
    df["is_cpi_window"] = 0
    df["is_fomc_day"] = 0
    df["minutes_to_event"] = np.nan
    df["policy_rate_level"] = np.nan

    if macro is not None and "FED_FUNDS" in macro.columns:
        m = macro[["timestamp", "FED_FUNDS"]].copy()
        m = m.rename(columns={"FED_FUNDS": "policy_rate_level"})
        df = df.drop(columns=["policy_rate_level"]).merge(m, on="timestamp", how="left")

    if events is None or events.empty:
        return df

    ts = pd.DatetimeIndex(df["timestamp"])
    # Ensure UTC
    ev = events.copy()
    ev["event_timestamp_utc"] = pd.to_datetime(ev["event_timestamp_utc"], utc=True)

    cpi_events = ev[ev["event_name"] == "CPI"]["event_timestamp_utc"].dropna()
    for e in cpi_events:
        diff_sec = np.abs((ts - e).total_seconds())
        df.loc[diff_sec <= 30 * 60, "is_cpi_window"] = 1

    fomc_dates = set(ev[ev["event_name"] == "FOMC"]["event_timestamp_utc"].dropna().dt.date)
    df["is_fomc_day"] = pd.Series(ts.date).isin(fomc_dates).astype(int).values

    all_events = ev["event_timestamp_utc"].dropna().sort_values()
    if len(all_events) > 0:
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        event_times = np.array([(pd.Timestamp(e) - epoch).total_seconds() for e in all_events]).astype(np.int64)
        bar_times = ((ts - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds()).astype(np.int64)
        idx = np.searchsorted(event_times, bar_times, side="right")
        mins_to_next = np.full(len(bar_times), np.nan, dtype=float)
        valid = idx < len(event_times)
        mins_to_next[valid] = np.minimum((event_times[idx[valid]] - bar_times[valid]) / 60, 1440)
        df["minutes_to_event"] = mins_to_next

    return df


def build_time_features(price: pd.DataFrame) -> pd.DataFrame:
    """Time features: hour, weekday, London session, NY session."""
    df = price[["timestamp"]].copy()
    ts = pd.DatetimeIndex(df["timestamp"])
    df["hour"] = ts.hour
    df["weekday"] = ts.weekday
    df["is_london_session"] = ((ts.hour >= 8) & (ts.hour < 17)).astype(int)
    df["is_ny_session"] = ((ts.hour >= 13) & (ts.hour < 22)).astype(int)
    return df


def build_news_features(price: pd.DataFrame, news_files: list[Path]) -> pd.DataFrame:
    """
    News sentiment by 5-min bucket. NaN where no news.
    Supports two formats:
    - GKG: timestamp (daily), sentiment_score, fx_theme_count — use as-is, merge_asof
    - DOC: timestamp, title — run FinBERT on titles, aggregate by 5min
    """
    df = price[["timestamp"]].copy()
    df["sentiment_score"] = np.nan

    if not USE_NEWS or not news_files:
        return df

    all_news = []
    for f in news_files:
        n = pd.read_csv(f)
        n.columns = [c.lower().strip() for c in n.columns]
        n["timestamp"] = pd.to_datetime(n["timestamp"], utc=True)
        all_news.append(n)

    if not all_news:
        return df

    news_df = pd.concat(all_news, ignore_index=True)

    # GKG path: pre-computed sentiment_score, daily resolution.
    # GKG for date D is published D+1; shift timestamps so bar at D+1 gets D's news.
    if "sentiment_score" in news_df.columns and news_df["sentiment_score"].notna().any():
        gkg = news_df[["timestamp", "sentiment_score"]].dropna(subset=["sentiment_score"])
        if not gkg.empty:
            gkg = gkg.groupby("timestamp")["sentiment_score"].mean().reset_index()
            gkg["timestamp"] = gkg["timestamp"] + pd.Timedelta(days=1)  # Release date: D's GKG available D+1
            gkg = gkg.sort_values("timestamp")
            price_sorted = df[["timestamp"]].sort_values("timestamp")
            merged = pd.merge_asof(
                price_sorted,
                gkg,
                on="timestamp",
                direction="backward",
            )
            df = df.drop(columns=["sentiment_score"], errors="ignore").merge(
                merged[["timestamp", "sentiment_score"]], on="timestamp", how="left"
            )
            return df

    # DOC path: titles for FinBERT.
    # Aggregate by 5min bucket; use max(timestamp) as release time so merge_asof only uses headlines known at bar time.
    if "title" not in news_df.columns:
        return df

    news_df = news_df.dropna(subset=["title"])
    news_df = news_df[news_df["title"].astype(str).str.len() > 3]
    if news_df.empty:
        return df

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        titles = news_df["title"].astype(str).tolist()
        batch_size = 32
        scores = []
        for i in range(0, len(titles), batch_size):
            batch = titles[i : i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=128)
            with torch.no_grad():
                out = model(**inputs)
            probs = torch.softmax(out.logits, dim=1)
            pos = probs[:, 0].numpy()
            neg = probs[:, 1].numpy()
            s = pos - neg
            scores.extend(s.tolist())

        news_df["sentiment_score"] = scores
        news_df["bucket"] = news_df["timestamp"].dt.floor("5min")
        agg = (
            news_df.groupby("bucket")
            .agg(sentiment_score=("sentiment_score", "mean"), timestamp=("timestamp", "max"))
            .reset_index(drop=True)
        )
        agg = agg.drop_duplicates(subset=["timestamp"], keep="first")
        agg = agg.sort_values("timestamp")
        price_sorted = df[["timestamp"]].sort_values("timestamp")
        merged = pd.merge_asof(price_sorted, agg[["timestamp", "sentiment_score"]], on="timestamp", direction="backward")
        df = df.drop(columns=["sentiment_score"], errors="ignore").merge(
            merged[["timestamp", "sentiment_score"]], on="timestamp", how="left"
        )
    except Exception as e:
        logging.warning("FinBERT skipped: %s", e)

    return df


def build_labels_and_classes(price: pd.DataFrame) -> pd.DataFrame:
    """future_return_30m and label_class (1=buy, -1=sell, 0=no_trade)."""
    df = price[["timestamp", "close"]].copy()
    df["future_return_30m"] = (df["close"].shift(-LABEL_HORIZON_BARS) - df["close"]) / df["close"]
    df = df.drop(columns=["close"])
    df = df.iloc[:-LABEL_HORIZON_BARS]

    def classify(r):
        if pd.isna(r):
            return np.nan
        if r > BUY_THRESHOLD_PCT:
            return 1
        if r < SELL_THRESHOLD_PCT:
            return -1
        return 0

    df["label_class"] = df["future_return_30m"].apply(classify)
    return df


def run(diagnose: bool = False, use_exogenous: bool | None = None, use_news: bool | None = None) -> None:
    """Build features, labels, split by time, save.
    diagnose: log NaN counts per column.
    use_exogenous/use_news: override config when provided (for ablation).
    """
    _setup_logging()
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    use_exog = use_exogenous if use_exogenous is not None else USE_EXOGENOUS
    use_news_val = use_news if use_news is not None else USE_NEWS

    price, cross, macro, events, news_files = load_inputs()
    if not use_exog:
        cross = None
        macro = None
        events = None
        logging.info("Exogenous disabled: skipping cross-asset, macro, events")
    if not use_news_val:
        news_files = []
    logging.info("Loaded price: %d rows", len(price))

    # Build feature blocks
    df = build_price_features(price)
    cross_feat = build_cross_asset_features(price, cross, macro) if use_exog else price[["timestamp"]].copy()
    macro_feat = build_macro_event_features(price, macro, events) if use_exog else price[["timestamp"]].copy()
    time_feat = build_time_features(price)
    news_feat = build_news_features(price, news_files)
    labels = build_labels_and_classes(price)

    # Merge: keep timestamp, OHLCV from price, add feature cols
    feature_cols = [c for c in df.columns if c not in ("timestamp", "open", "high", "low", "close", "volume")]
    df = df[["timestamp"] + feature_cols]

    for name, feat_df in [("cross", cross_feat), ("macro", macro_feat), ("time", time_feat), ("news", news_feat)]:
        drop = [c for c in feat_df.columns if c != "timestamp"]
        if drop:
            right = feat_df[["timestamp"] + drop].drop_duplicates(subset=["timestamp"], keep="first")
            before_len = len(df)
            df = df.merge(right, on="timestamp", how="left")
            if len(df) != before_len:
                logging.warning("Merge %s produced row count change %d -> %d; deduped right", name, before_len, len(df))
                df = df.drop_duplicates(subset=["timestamp"], keep="first")

    df = df.merge(labels, on="timestamp", how="inner")
    df = df.dropna(subset=["future_return_30m", "label_class"], how="all")

    if diagnose:
        nan_pct = (df.isna().sum() / len(df) * 100)
        for col in df.columns:
            pct = nan_pct.get(col, 0)
            if pct > 0:
                logging.info("Diagnose: %s NaN %.2f%%", col, pct)

    # Time split
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    train = df[df["timestamp"] < pd.Timestamp(VALIDATION_START_DATE, tz="UTC")]
    val = df[(df["timestamp"] >= pd.Timestamp(VALIDATION_START_DATE, tz="UTC")) & (df["timestamp"] < pd.Timestamp(TEST_START_DATE, tz="UTC"))]
    test = df[df["timestamp"] >= pd.Timestamp(TEST_START_DATE, tz="UTC")]

    train.to_csv(FEATURES_DIR / "train.csv", index=False)
    val.to_csv(FEATURES_DIR / "validation.csv", index=False)
    test.to_csv(FEATURES_DIR / "test.csv", index=False)
    train.to_parquet(FEATURES_DIR / "train.parquet", index=False)
    val.to_parquet(FEATURES_DIR / "validation.parquet", index=False)
    test.to_parquet(FEATURES_DIR / "test.parquet", index=False)

    # Phase 18: Versioned copies
    from datetime import datetime, timezone
    version = datetime.now(timezone.utc).strftime("%Y%m%d")
    versions_dir = FEATURES_DIR / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    for name, df_save in [("train", train), ("validation", val), ("test", test)]:
        df_save.to_parquet(versions_dir / f"{name}_{version}.parquet", index=False)
    logging.info("Saved versioned features")

    logging.info("Saved train: %d | validation: %d | test: %d", len(train), len(val), len(test))
    logging.info("Features: %s", [c for c in df.columns if c not in ("timestamp", "future_return_30m", "label_class")])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose", action="store_true", help="Log NaN counts per column")
    args = parser.parse_args()
    run(diagnose=args.diagnose)
