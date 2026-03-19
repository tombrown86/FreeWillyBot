"""
Live bar fetcher — cTrader (Pepperstone demo/live).

Fetches recent closed 5-min OHLCV bars for EURUSD via ProtoOAGetTrendbarsReq.
Used only for the live tail (the bars the live tick trades on).
Training and backtesting remain on Dukascopy data — this file is never called
from the training pipeline.

Public API:
    fetch_ctrader_bars(n_bars)        -> DataFrame[timestamp,open,high,low,close,volume,bar_source]
    merge_ctrader_live_bars(n_bars)   -> same, merged with Dukascopy history

bar_source column values:
    "ctrader"    bars fetched from this module
    "dukascopy"  bars from the Dukascopy batch pipeline (carry-through from clean parquet)
"""

import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── config imports ────────────────────────────────────────────────────────────
from src.config import (
    BAR_INTERVAL,
    CTRADER_EURUSD_SYMBOL_ID,
    CTRADER_HOST_TYPE,
    SYMBOL,
)

PROCESSED_PRICE_PARQUET = (
    PROJECT_ROOT / "data" / "processed" / "price" / f"{SYMBOL}_{BAR_INTERVAL}_clean.parquet"
)

# How many extra bars to request beyond n_bars to guard against partial/open bar
_FETCH_BUFFER = 5
# Seconds per bar (5-min)
_BAR_SECONDS = 5 * 60
# cTrader price pipette divisor for EURUSD (5 decimal places → 1e-5)
_EURUSD_PIPETTE = 1e-5
# Reactor call timeout (seconds) — reuse execution.py constant
_CT_TIMEOUT_S = 30


def _ctrader_access_token() -> str:
    return os.environ.get("PS_CTRADER_ACCESS_TOKEN", "").strip()


def _ctrader_account_id_str() -> str | None:
    for key in ("PS_CTRADER_ACCOUNT_ID", "PS_CTRADEER_ACCOUNT_ID", "PS_CTRADEER_LOGIN"):
        v = os.environ.get(key, "").strip()
        if v:
            return v
    return None


def _ctrader_credentials_available() -> bool:
    return bool(_ctrader_access_token() and _ctrader_account_id_str())


def fetch_ctrader_bars(n_bars: int = 100) -> pd.DataFrame:
    """
    Fetch the last `n_bars` closed 5-min EURUSD OHLCV bars from cTrader.

    Returns a DataFrame with columns:
        timestamp (UTC, tz-aware), open, high, low, close, volume, bar_source

    Raises RuntimeError if cTrader credentials are missing or the call fails.
    """
    if not _ctrader_credentials_available():
        raise RuntimeError(
            "cTrader credentials not set — need PS_CTRADER_ACCESS_TOKEN and "
            "PS_CTRADER_ACCOUNT_ID in .env"
        )

    from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoHeartbeatEvent
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOAAccountAuthReq,
        ProtoOAAccountAuthRes,
        ProtoOAApplicationAuthReq,
        ProtoOAApplicationAuthRes,
        ProtoOAErrorRes,
        ProtoOAGetAccountListByAccessTokenReq,
        ProtoOAGetAccountListByAccessTokenRes,
        ProtoOAGetTrendbarsReq,
        ProtoOAGetTrendbarsRes,
    )
    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATrendbarPeriod
    from twisted.internet import reactor

    access_token = _ctrader_access_token()
    account_id = int(_ctrader_account_id_str() or "0")
    client_id = os.environ.get("PS_CTRADER_CLIENTID", "").strip()
    client_secret = os.environ.get("PS_CTRADER_SECRET", "").strip()

    want_live = CTRADER_HOST_TYPE.lower() == "live"
    host = EndPoints.PROTOBUF_LIVE_HOST if want_live else EndPoints.PROTOBUF_DEMO_HOST
    ct_client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)

    done_event = threading.Event()
    result: dict = {"data": None, "error": None, "done": done_event}
    state = {"phase": 0}

    def fail(msg: str) -> None:
        if not result["done"].is_set():
            result["error"] = msg
            result["done"].set()
        _safe_stop_reactor()

    def _safe_stop_reactor() -> None:
        from twisted.internet.error import ReactorNotRunning

        def _stop() -> None:
            try:
                if reactor.running:
                    reactor.stop()
            except ReactorNotRunning:
                pass

        try:
            reactor.callFromThread(_stop)
        except Exception:
            pass

    def on_connected(c) -> None:
        req = ProtoOAApplicationAuthReq()
        req.clientId = client_id
        req.clientSecret = client_secret
        d = c.send(req)
        d.addErrback(lambda f: fail(str(f)))

    def on_message(c, message) -> None:
        pt = message.payloadType
        if pt == ProtoHeartbeatEvent().payloadType:
            return
        if pt == ProtoOAErrorRes().payloadType:
            err = Protobuf.extract(message)
            fail(getattr(err, "description", None) or str(err))
            return

        # Phase 0 → app auth OK → get account list
        if pt == ProtoOAApplicationAuthRes().payloadType and state["phase"] == 0:
            state["phase"] = 1
            req = ProtoOAGetAccountListByAccessTokenReq()
            req.accessToken = access_token
            d = c.send(req)
            d.addErrback(lambda f: fail(str(f)))
            return

        # Phase 1 → resolve ctidTraderAccountId → account auth
        if pt == ProtoOAGetAccountListByAccessTokenRes().payloadType and state["phase"] == 1:
            res = Protobuf.extract(message)
            allowed = list(res.ctidTraderAccount)
            match = next((a for a in allowed if int(a.ctidTraderAccountId) == account_id), None)
            if not match:
                match = next((a for a in allowed if int(a.traderLogin) == account_id), None)
            if not match:
                details = [
                    f"ctid={int(a.ctidTraderAccountId)} login={int(a.traderLogin)}"
                    for a in allowed
                ]
                fail(
                    f"cTrader account {account_id} not in access token grant. "
                    f"Granted: {details or '(none)'}"
                )
                return
            effective_ctid = int(match.ctidTraderAccountId)
            state["phase"] = 2
            state["effective_ctid"] = effective_ctid
            req = ProtoOAAccountAuthReq()
            req.ctidTraderAccountId = effective_ctid
            req.accessToken = access_token
            d = c.send(req)
            d.addErrback(lambda f: fail(str(f)))
            return

        # Phase 2 → account auth OK → send trend bars request
        if pt == ProtoOAAccountAuthRes().payloadType and state["phase"] == 2:
            state["phase"] = 3
            effective_ctid = state.get("effective_ctid", account_id)

            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            fetch_count = n_bars + _FETCH_BUFFER
            from_ms = now_ms - int(fetch_count * _BAR_SECONDS * 1000)

            req = ProtoOAGetTrendbarsReq()
            req.ctidTraderAccountId = effective_ctid
            req.symbolId = CTRADER_EURUSD_SYMBOL_ID
            req.period = ProtoOATrendbarPeriod.Value("M5")
            req.fromTimestamp = from_ms
            req.toTimestamp = now_ms
            req.count = fetch_count
            d = c.send(req)

            def on_bars(resp):
                extracted = Protobuf.extract(resp) if hasattr(resp, "payloadType") else resp
                result["data"] = extracted
                result["done"].set()
                _safe_stop_reactor()

            d.addCallback(on_bars)
            d.addErrback(lambda f: fail(str(f)))
            return

    def on_error(failure) -> None:
        fail(str(failure))

    def on_disconnected(c, reason) -> None:
        if not result["done"].is_set():
            result["error"] = f"Disconnected: {reason}"
            result["done"].set()

    ct_client.setConnectedCallback(on_connected)
    ct_client.setDisconnectedCallback(on_disconnected)
    ct_client.setMessageReceivedCallback(on_message)

    def run_reactor():
        ct_client.startService()
        reactor.run(installSignalHandlers=False)

    t = threading.Thread(target=run_reactor, daemon=True)
    t.start()

    signalled = done_event.wait(timeout=_CT_TIMEOUT_S)
    if not signalled:
        result["error"] = f"cTrader bars fetch timed out after {_CT_TIMEOUT_S}s"

    t.join(timeout=8)

    if result["error"]:
        raise RuntimeError(f"cTrader fetch_ctrader_bars failed: {result['error']}")

    raw = result["data"]
    if raw is None or not hasattr(raw, "trendbar") or not raw.trendbar:
        raise RuntimeError("cTrader returned no bars (empty trendbar list)")

    return _decode_trendbars(raw.trendbar, n_bars)


def _decode_trendbars(trendbars, n_bars: int) -> pd.DataFrame:
    """
    Convert ProtoOATrendbar repeated field to a clean DataFrame.

    cTrader delta encoding:
        low   = raw low / _EURUSD_PIPETTE   (absolute)
        open  = low + deltaOpen  * _EURUSD_PIPETTE
        close = low + deltaClose * _EURUSD_PIPETTE
        high  = low + deltaHigh  * _EURUSD_PIPETTE

    utcTimestampInMinutes is the bar OPEN time in minutes since epoch.
    We use bar close = open time + 5 min so timestamps match our convention.
    """
    rows = []
    for bar in trendbars:
        low_price = bar.low * _EURUSD_PIPETTE
        open_price = low_price + bar.deltaOpen * _EURUSD_PIPETTE
        close_price = low_price + bar.deltaClose * _EURUSD_PIPETTE
        high_price = low_price + bar.deltaHigh * _EURUSD_PIPETTE

        # bar open time in UTC seconds
        bar_open_s = int(bar.utcTimestampInMinutes) * 60
        # Use bar CLOSE time as timestamp (consistent with Dukascopy convention)
        bar_close_s = bar_open_s + _BAR_SECONDS
        ts = datetime.fromtimestamp(bar_close_s, tz=timezone.utc)

        rows.append({
            "timestamp": ts,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": int(bar.volume),
            "bar_source": "ctrader",
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Drop the current (open, potentially incomplete) bar — keep only fully closed bars.
    # A bar is considered closed if its close time is > 1 bar period in the past.
    now_utc = datetime.now(timezone.utc)
    cutoff = pd.Timestamp(now_utc - timedelta(seconds=_BAR_SECONDS))
    df = df[df["timestamp"] <= cutoff].reset_index(drop=True)

    # Return only the last n_bars
    return df.tail(n_bars).reset_index(drop=True)


def merge_ctrader_live_bars(n_bars: int = 100) -> pd.DataFrame:
    """
    Merge recent cTrader bars into the Dukascopy history for live tail feature building.

    Strategy:
    - Load EURUSD_5min_clean.parquet (Dukascopy, bar_source="dukascopy").
    - Fetch the last n_bars from cTrader (bar_source="ctrader").
    - For any timestamp present in both, prefer the cTrader version.
    - Return the merged DataFrame sorted by timestamp.

    The clean parquet is NEVER written back — training data is untouched.
    """
    # Load Dukascopy history
    if not PROCESSED_PRICE_PARQUET.exists():
        raise FileNotFoundError(
            f"Processed price file not found: {PROCESSED_PRICE_PARQUET}. "
            "Run build_price_bars first."
        )

    duka_df = pd.read_parquet(PROCESSED_PRICE_PARQUET)
    duka_df["timestamp"] = pd.to_datetime(duka_df["timestamp"], utc=True)

    # Stamp Dukascopy rows if bar_source column is missing (pre-existing file)
    if "bar_source" not in duka_df.columns:
        duka_df["bar_source"] = "dukascopy"

    # Fetch cTrader bars
    ct_df = fetch_ctrader_bars(n_bars)
    if ct_df.empty:
        logging.warning("cTrader returned 0 bars — using Dukascopy tail only")
        return duka_df

    ct_timestamps = set(ct_df["timestamp"].tolist())

    # Remove Dukascopy rows that overlap with cTrader (prefer cTrader for recent bars)
    duka_trimmed = duka_df[~duka_df["timestamp"].isin(ct_timestamps)]

    # Align columns
    all_cols = list(duka_df.columns)
    for col in all_cols:
        if col not in ct_df.columns:
            ct_df[col] = None
    ct_df = ct_df[all_cols]

    merged = pd.concat([duka_trimmed, ct_df], ignore_index=True)
    merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    merged = merged.reset_index(drop=True)

    logging.info(
        "merge_ctrader_live_bars: %d Dukascopy rows + %d cTrader rows → %d total (last bar: %s, source=%s)",
        len(duka_trimmed),
        len(ct_df),
        len(merged),
        merged["timestamp"].max().strftime("%Y-%m-%d %H:%M UTC"),
        merged.iloc[-1]["bar_source"],
    )
    return merged
