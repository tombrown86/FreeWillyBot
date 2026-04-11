"""
Phase 15 — Demo execution plumbing.

Connect to OANDA (FX) or Binance (crypto) demo/testnet only.
place_market_order, get_open_positions, close_position, cancel_all_orders.
Log every broker response. Never hard-code secrets.

cTrader (Pepperstone demo) is used when SYMBOL is EURUSD and
PS_CTRADER_ACCESS_TOKEN plus PS_CTRADER_ACCOUNT_ID (or PS_CTRADEER_ACCOUNT_ID / PS_CTRADEER_LOGIN) are set.
"""

import json
import logging
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    BINANCE_SYMBOL,
    CTRADER_EURUSD_SYMBOL_ID,
    CTRADER_HOST_TYPE,
    CTRADER_VOLUME_LOTS,
    EXECUTION_PAPER_ONLY,
    EXECUTION_TEST_UNITS,
    OANDA_INSTRUMENT,
    SYMBOL,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXECUTION_LOG_DIR = PROJECT_ROOT / "data" / "logs" / "execution"
LOG_FILE = PROJECT_ROOT / "data" / "logs" / "execution.log"

# Load .env if present
_env = PROJECT_ROOT / ".env"
if _env.exists():
    from dotenv import load_dotenv
    load_dotenv(_env)


def _ctrader_access_token() -> str:
    return os.environ.get("PS_CTRADER_ACCESS_TOKEN", "").strip()


def _ctrader_account_id_str() -> str | None:
    """Numeric cTrader account id (ctidTraderAccountId)."""
    for key in (
        "PS_CTRADER_ACCOUNT_ID",
        "PS_CTRADEER_ACCOUNT_ID",
        "PS_CTRADEER_LOGIN",
    ):
        v = os.environ.get(key, "").strip()
        if v:
            return v
    return None


def _setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    EXECUTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _get_broker() -> str:
    """Return 'ctrader', 'oanda', or 'binance' based on SYMBOL and available credentials."""
    if "USDT" in SYMBOL:
        return "binance"
    # Prefer cTrader when its access token and account ID are set
    if _ctrader_access_token() and _ctrader_account_id_str():
        return "ctrader"
    return "oanda"


def _log_broker_response(operation: str, response: dict | str) -> None:
    """Log full broker response."""
    if isinstance(response, dict):
        body = json.dumps(response, default=str)
    else:
        body = str(response)
    logging.info("Broker response [%s]: %s", operation, body)


# ─── cTrader helpers ──────────────────────────────────────────────────────────
# The cTrader Open API SDK is Twisted-based (async). We run a short-lived reactor
# in a background thread so execution.py can stay synchronous.  Each call:
#   1. Starts a thread with the reactor.
#   2. Uses reactor.callFromThread to schedule work.
#   3. Signals a threading.Event when done; main thread blocks until then.
#   4. Stops the reactor so the thread exits cleanly.

_CT_TIMEOUT_S = 30  # seconds (app auth → account list → account auth → request)


def _ctrader_safe_stop_reactor() -> None:
    """Avoid ReactorNotRunning when stop is scheduled twice (work_fn + main thread)."""
    from twisted.internet import reactor
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


def _ctrader_call(work_fn, *, account_id_override: int | None = None) -> dict:
    """
    Run `work_fn(client, access_token, account_id, result_holder)` inside
    a Twisted reactor running in a thread.  Returns result_holder["result"].

    Open API order (per cTrader docs): application auth → get account list by access
    token → account auth → then trading requests. Skipping the list step causes
    \"Trading account is not authorized\" on reconcile/orders.

    work_fn signature:
        work_fn(client, access_token: str, account_id: int, result: dict) -> None
    It must set result["data"] (dict) and/or result["error"] (str) then call
    result["done"].set() to unblock the caller.

    account_id_override: if provided, use this login/ctid instead of PS_CTRADER_ACCOUNT_ID.
    """
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
    )
    from twisted.internet import reactor

    access_token = _ctrader_access_token()
    account_id = account_id_override if account_id_override is not None else int(_ctrader_account_id_str() or "0")
    client_id = os.environ.get("PS_CTRADER_CLIENTID", "").strip()
    client_secret = os.environ.get("PS_CTRADER_SECRET", "").strip()

    want_live = CTRADER_HOST_TYPE.lower() == "live"
    host = EndPoints.PROTOBUF_LIVE_HOST if want_live else EndPoints.PROTOBUF_DEMO_HOST
    ct_client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)

    done_event = threading.Event()
    result: dict = {"data": None, "error": None, "done": done_event}
    # Phases: 0 wait app auth res → 1 wait account list → 2 wait account auth → 3 run work
    state = {"phase": 0}

    def fail(msg: str) -> None:
        if not result["done"].is_set():
            result["error"] = msg
            result["done"].set()
        _ctrader_safe_stop_reactor()

    def on_connected(c):
        req = ProtoOAApplicationAuthReq()
        req.clientId = client_id
        req.clientSecret = client_secret
        d = c.send(req)
        d.addErrback(on_error)

    def on_message(c, message):
        pt = message.payloadType
        if pt == ProtoHeartbeatEvent().payloadType:
            return
        if pt == ProtoOAErrorRes().payloadType:
            err = Protobuf.extract(message)
            fail(getattr(err, "description", None) or str(err))
            return
        if pt == ProtoOAApplicationAuthRes().payloadType and state["phase"] == 0:
            state["phase"] = 1
            req = ProtoOAGetAccountListByAccessTokenReq()
            req.accessToken = access_token
            d = c.send(req)
            d.addErrback(on_error)
            return
        if pt == ProtoOAGetAccountListByAccessTokenRes().payloadType and state["phase"] == 1:
            res = Protobuf.extract(message)
            allowed = list(res.ctidTraderAccount)
            # ctidTraderAccountId is the Open API id; traderLogin is the number you see as "login" in cTrader.
            match = next((a for a in allowed if int(a.ctidTraderAccountId) == account_id), None)
            if not match:
                match = next((a for a in allowed if int(a.traderLogin) == account_id), None)
            if not match:
                details = [
                    f"ctid={int(a.ctidTraderAccountId)} login={int(a.traderLogin)}"
                    for a in allowed
                ]
                fail(
                    f"cTrader account {account_id} not authorized for this access token. "
                    f"Granted: {details or '(none)'}. "
                    "Set PS_CTRADER_ACCOUNT_ID to a listed ctid or login, or re-authorize in Playground."
                )
                return
            effective_ctid = int(match.ctidTraderAccountId)
            if effective_ctid != account_id:
                import logging

                logging.info(
                    "Resolved configured id %s to ctidTraderAccountId %s (login vs internal id)",
                    account_id,
                    effective_ctid,
                )
            acc_is_live = bool(match.isLive)
            if acc_is_live != want_live:
                fail(
                    f"Account ctid {effective_ctid} is {'live' if acc_is_live else 'demo'} but "
                    f"CTRADER_HOST_TYPE is {'live' if want_live else 'demo'} in config.py — "
                    "they must match."
                )
                return
            state["phase"] = 2
            state["effective_ctid"] = effective_ctid
            req = ProtoOAAccountAuthReq()
            req.ctidTraderAccountId = effective_ctid
            req.accessToken = access_token
            d = c.send(req)
            d.addErrback(on_error)
            return
        if pt == ProtoOAAccountAuthRes().payloadType and state["phase"] == 2:
            state["phase"] = 3
            effective = state.get("effective_ctid", account_id)
            reactor.callFromThread(work_fn, c, access_token, effective, result)
            return
        # Responses with clientMsgId are handled by send() deferreds; do not duplicate work_fn here.

    def on_error(failure):
        fail(str(failure))

    def on_disconnected(c, reason):
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
        result["error"] = f"cTrader call timed out after {_CT_TIMEOUT_S}s"

    _ctrader_safe_stop_reactor()
    t.join(timeout=8)

    if result["error"]:
        raise RuntimeError(f"cTrader error: {result['error']}")
    return result["data"] or {}


def _ctrader_place_order(side: str, *, account_id_override: int | None = None) -> dict:
    """Place a minimal MARKET order on the demo account."""
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOANewOrderReq,
        ProtoOAExecutionEvent,
    )
    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
        ProtoOAOrderType,
        ProtoOATradeSide,
    )

    volume_units = int(CTRADER_VOLUME_LOTS * 100_000)  # lots → units (1 lot = 100 000)
    trade_side = "BUY" if side.lower() == "buy" else "SELL"
    _result_ref: list[dict] = [{}]

    def work(c, access_token, account_id, result):
        req = ProtoOANewOrderReq()
        req.ctidTraderAccountId = account_id
        req.symbolId = CTRADER_EURUSD_SYMBOL_ID
        req.orderType = ProtoOAOrderType.Value("MARKET")
        req.tradeSide = ProtoOATradeSide.Value(trade_side)
        req.volume = volume_units
        d = c.send(req)

        def on_resp(resp):
            from ctrader_open_api import Protobuf
            extracted = Protobuf.extract(resp) if hasattr(resp, "payloadType") else resp
            result["data"] = {"order_response": str(extracted), "side": side, "volume_units": volume_units}
            result["done"].set()
            _ctrader_safe_stop_reactor()

        def on_err(f):
            result["error"] = str(f.value) if hasattr(f, "value") else str(f)
            result["done"].set()
            _ctrader_safe_stop_reactor()

        d.addCallback(on_resp)
        d.addErrback(on_err)

    return _ctrader_call(work, account_id_override=account_id_override)


def _ctrader_get_positions(*, account_id_override: int | None = None) -> dict:
    """Query open positions via ProtoOAReconcileReq. Returns parsed positions/orders and eurusd_symbol_id."""
    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATradeSide
    from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAReconcileReq

    def work(c, access_token, account_id, result):
        req = ProtoOAReconcileReq()
        req.ctidTraderAccountId = account_id
        d = c.send(req)

        def on_resp(resp):
            from ctrader_open_api import Protobuf
            extracted = Protobuf.extract(resp) if hasattr(resp, "payloadType") else resp
            # Keep raw string for logging; add structured data and EURUSD symbol id we use for orders
            positions_list = []
            for pos in extracted.position:
                td = pos.tradeData
                trade_side = ProtoOATradeSide.Name(td.tradeSide) if td.tradeSide else ""
                positions_list.append({
                    "positionId": int(pos.positionId),
                    "symbolId": int(td.symbolId),
                    "volume": int(td.volume),
                    "tradeSide": trade_side,
                })
            orders_list = []
            for o in extracted.order:
                td = getattr(o, "tradeData", None)
                orders_list.append({
                    "orderId": int(o.orderId),
                    "symbolId": int(td.symbolId) if td else 0,
                    "volume": int(td.volume) if td else 0,
                    "tradeSide": ProtoOATradeSide.Name(td.tradeSide) if td and td.tradeSide else "",
                })
            result["data"] = {
                "ctidTraderAccountId": int(extracted.ctidTraderAccountId),
                "positions": positions_list,
                "orders": orders_list,
                "eurusd_symbol_id": CTRADER_EURUSD_SYMBOL_ID,
                "reconcile": str(extracted),
            }
            result["done"].set()
            _ctrader_safe_stop_reactor()

        def on_err(f):
            result["error"] = str(f.value) if hasattr(f, "value") else str(f)
            result["done"].set()
            _ctrader_safe_stop_reactor()

        d.addCallback(on_resp)
        d.addErrback(on_err)

    return _ctrader_call(work, account_id_override=account_id_override)


def _ctrader_close_position(position_id: int, volume_units: int) -> dict:
    """Close a specific position by ID.

    NOTE: only call this if you are already inside a _ctrader_call work_fn and therefore
    already have an authenticated client + running reactor.  Do NOT use this directly from
    application code; use _ctrader_close_all() which does reconcile+close in one reactor
    session to avoid ReactorNotRestartable.
    """
    from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAClosePositionReq

    def work(c, access_token, account_id, result):
        req = ProtoOAClosePositionReq()
        req.ctidTraderAccountId = account_id
        req.positionId = position_id
        req.volume = volume_units
        d = c.send(req)

        def on_resp(resp):
            from ctrader_open_api import Protobuf
            extracted = Protobuf.extract(resp) if hasattr(resp, "payloadType") else resp
            result["data"] = {"close_response": str(extracted), "positionId": position_id}
            result["done"].set()
            _ctrader_safe_stop_reactor()

        def on_err(f):
            result["error"] = str(f.value) if hasattr(f, "value") else str(f)
            result["done"].set()
            _ctrader_safe_stop_reactor()

        d.addCallback(on_resp)
        d.addErrback(on_err)

    return _ctrader_call(work)


def _ctrader_close_all(*, account_id_override: int | None = None) -> dict:
    """
    Reconcile open positions AND close them all in a SINGLE reactor session.

    The Twisted reactor is a process-level singleton — once reactor.run() has been
    called and stopped, it cannot be restarted (ReactorNotRestartable).  The old
    close_position() code called _ctrader_get_positions() then _ctrader_close_position()
    as two separate _ctrader_call() invocations, which caused the second call to hang
    for the full 30-second timeout every single time after the first cTrader call in
    that process.

    This function does reconcile → close-all in one work_fn so only one reactor session
    is needed.
    """
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOAClosePositionReq,
        ProtoOAReconcileReq,
    )
    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATradeSide
    from ctrader_open_api import Protobuf

    def work(c, access_token, account_id, result):
        # Step 1: reconcile to get open positions
        rec_req = ProtoOAReconcileReq()
        rec_req.ctidTraderAccountId = account_id
        d = c.send(rec_req)

        def on_reconcile(resp):
            extracted = Protobuf.extract(resp) if hasattr(resp, "payloadType") else resp
            positions = []
            for pos in extracted.position:
                td = pos.tradeData
                positions.append({
                    "positionId": int(pos.positionId),
                    "volume": int(td.volume),
                    "tradeSide": ProtoOATradeSide.Name(td.tradeSide) if td.tradeSide else "",
                })

            if not positions:
                result["data"] = {"closed": [], "message": "no open positions"}
                result["done"].set()
                _ctrader_safe_stop_reactor()
                return

            # Step 2: close all positions; wait for each response before signalling done
            closed_results: list[dict] = []
            pending = {"count": len(positions)}

            def on_close_resp(resp, pos_id):
                ext = Protobuf.extract(resp) if hasattr(resp, "payloadType") else resp
                closed_results.append({"close_response": str(ext), "positionId": pos_id})
                pending["count"] -= 1
                if pending["count"] <= 0:
                    result["data"] = {"closed": closed_results}
                    result["done"].set()
                    _ctrader_safe_stop_reactor()

            def on_close_err(f, pos_id):
                err_msg = str(f.value) if hasattr(f, "value") else str(f)
                logging.warning("cTrader close error for pos %s: %s", pos_id, err_msg)
                closed_results.append({"error": err_msg, "positionId": pos_id})
                pending["count"] -= 1
                if pending["count"] <= 0:
                    # Partial success — return what we got
                    result["data"] = {"closed": closed_results}
                    result["done"].set()
                    _ctrader_safe_stop_reactor()

            for pos in positions:
                close_req = ProtoOAClosePositionReq()
                close_req.ctidTraderAccountId = account_id
                close_req.positionId = pos["positionId"]
                close_req.volume = pos["volume"]
                dc = c.send(close_req)
                dc.addCallback(on_close_resp, pos["positionId"])
                dc.addErrback(on_close_err, pos["positionId"])

        def on_reconcile_err(f):
            result["error"] = str(f.value) if hasattr(f, "value") else str(f)
            result["done"].set()
            _ctrader_safe_stop_reactor()

        d.addCallback(on_reconcile)
        d.addErrback(on_reconcile_err)

    return _ctrader_call(work, account_id_override=account_id_override)
# ─────────────────────────────────────────────────────────────────────────────


def place_market_order(
    side: str,
    units: float | None = None,
    *,
    account_id_override: int | None = None,
) -> dict:
    """
    Place a small test market order. Paper/demo only.
    side: 'buy' or 'sell'
    account_id_override: cTrader login/ctid to use instead of PS_CTRADER_ACCOUNT_ID.
    """
    if not EXECUTION_PAPER_ONLY:
        raise RuntimeError("EXECUTION_PAPER_ONLY must be True; no live execution")

    _setup_logging()
    broker = _get_broker()
    units = units or EXECUTION_TEST_UNITS

    if broker == "ctrader":
        access_token = _ctrader_access_token()
        account_id = _ctrader_account_id_str()
        if not access_token or not account_id:
            logging.warning(
                "cTrader: set PS_CTRADER_ACCESS_TOKEN and PS_CTRADER_ACCOUNT_ID "
                "(or PS_CTRADEER_ACCOUNT_ID / PS_CTRADEER_LOGIN); simulating"
            )
            out = {"simulated": True, "side": side, "broker": "ctrader"}
            _log_broker_response("place_market_order", out)
            return out
        resp = _ctrader_place_order(side, account_id_override=account_id_override)
        _log_broker_response("place_market_order", resp)
        return resp

    elif broker == "oanda":
        token = os.environ.get("OANDA_ACCESS_TOKEN")
        account_id = os.environ.get("OANDA_ACCOUNT_ID")
        if not token or not account_id:
            logging.warning("OANDA_ACCESS_TOKEN or OANDA_ACCOUNT_ID not set; simulating")
            out = {"simulated": True, "side": side, "units": units, "broker": "oanda"}
            _log_broker_response("place_market_order", out)
            return out

        from oandapyV20 import API
        from oandapyV20.endpoints import orders
        from oandapyV20.contrib.requests import MarketOrderRequest

        api = API(access_token=token, environment="practice")
        direction = 1 if side.lower() == "buy" else -1
        req = MarketOrderRequest(
            instrument=OANDA_INSTRUMENT,
            units=int(direction * units),
        ).data
        r = orders.OrderCreate(account_id, data={"order": req})
        resp = api.request(r)
        _log_broker_response("place_market_order", resp)
        return resp

    else:
        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            logging.warning("BINANCE_API_KEY or BINANCE_API_SECRET not set; simulating")
            out = {"simulated": True, "side": side, "broker": "binance"}
            _log_broker_response("place_market_order", out)
            return out

        from binance.client import Client

        client = Client(api_key, api_secret, testnet=True)
        order_side = Client.SIDE_BUY if side.lower() == "buy" else Client.SIDE_SELL
        quantity = "0.0001"  # minimal BTC for test
        resp = client.create_order(
            symbol=BINANCE_SYMBOL,
            side=order_side,
            type=Client.ORDER_TYPE_MARKET,
            quantity=quantity,
        )
        _log_broker_response("place_market_order", resp)
        return resp


def get_open_positions(*, account_id_override: int | None = None) -> dict:
    """Query open positions. Log response."""
    _setup_logging()
    broker = _get_broker()

    if broker == "ctrader":
        access_token = _ctrader_access_token()
        account_id = _ctrader_account_id_str()
        if not access_token or not account_id:
            logging.warning(
                "cTrader: set PS_CTRADER_ACCESS_TOKEN and PS_CTRADER_ACCOUNT_ID "
                "(or PS_CTRADEER_ACCOUNT_ID / PS_CTRADEER_LOGIN); simulating"
            )
            out = {"simulated": True, "positions": [], "broker": "ctrader"}
            _log_broker_response("get_open_positions", out)
            return out
        resp = _ctrader_get_positions(account_id_override=account_id_override)
        resp["broker"] = "ctrader"
        _log_broker_response("get_open_positions", resp)
        return resp

    elif broker == "oanda":
        token = os.environ.get("OANDA_ACCESS_TOKEN")
        account_id = os.environ.get("OANDA_ACCOUNT_ID")
        if not token or not account_id:
            logging.warning("OANDA credentials not set; simulating")
            out = {"simulated": True, "positions": [], "broker": "oanda"}
            _log_broker_response("get_open_positions", out)
            return out

        from oandapyV20 import API
        from oandapyV20.endpoints import positions

        api = API(access_token=token, environment="practice")
        r = positions.OpenPositions(account_id)
        resp = api.request(r)
        _log_broker_response("get_open_positions", resp)
        return resp

    else:
        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            logging.warning("BINANCE credentials not set; simulating")
            out = {"simulated": True, "positions": [], "broker": "binance"}
            _log_broker_response("get_open_positions", out)
            return out

        from binance.client import Client

        client = Client(api_key, api_secret, testnet=True)
        acc = client.get_account()
        positions = [
            b for b in acc.get("balances", [])
            if float(b.get("free", 0)) != 0 or float(b.get("locked", 0)) != 0
        ]
        out = {"balances": positions, "broker": "binance"}
        _log_broker_response("get_open_positions", out)
        return out


def close_position(position_id: str | None = None, *, account_id_override: int | None = None) -> dict:
    """Close a position. For OANDA pass position_id; for Binance close BTC position.

    For cTrader: always closes ALL open positions in a single reactor session via
    _ctrader_close_all(), regardless of position_id argument.  The old approach
    (separate reconcile call + close call) caused ReactorNotRestartable on every
    close attempt after the first cTrader call in the same process, resulting in
    30-second timeouts on every tick.

    account_id_override: cTrader login/ctid to close positions on instead of PS_CTRADER_ACCOUNT_ID.
    """
    if not EXECUTION_PAPER_ONLY:
        raise RuntimeError("EXECUTION_PAPER_ONLY must be True")

    _setup_logging()
    broker = _get_broker()

    if broker == "ctrader":
        access_token = _ctrader_access_token()
        account_id = _ctrader_account_id_str()
        if not access_token or not account_id:
            out = {"simulated": True, "action": "close", "broker": "ctrader"}
            _log_broker_response("close_position", out)
            return out
        # Single reactor session: reconcile + close all in one _ctrader_call
        resp = _ctrader_close_all(account_id_override=account_id_override)
        resp["broker"] = "ctrader"
        _log_broker_response("close_position", resp)
        return resp

    elif broker == "oanda":
        token = os.environ.get("OANDA_ACCESS_TOKEN")
        account_id = os.environ.get("OANDA_ACCOUNT_ID")
        if not token or not account_id:
            out = {"simulated": True, "action": "close", "broker": "oanda"}
            _log_broker_response("close_position", out)
            return out

        from oandapyV20 import API
        from oandapyV20.endpoints import positions

        api = API(access_token=token, environment="practice")
        r = positions.OpenPositions(account_id)
        resp = api.request(r)
        pos_list = resp.get("positions", [])
        results = []
        for p in pos_list:
            pid = p.get("instrument")
            if position_id and pid != position_id:
                continue
            close_r = positions.PositionClose(account_id, instrument=pid, data={"longUnits": "ALL", "shortUnits": "ALL"})
            cr = api.request(close_r)
            results.append(cr)
            _log_broker_response("close_position", cr)
        return {"closed": results} if results else {"closed": [], "message": "no positions"}

    else:
        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            out = {"simulated": True, "action": "close", "broker": "binance"}
            _log_broker_response("close_position", out)
            return out

        from binance.client import Client

        client = Client(api_key, api_secret, testnet=True)
        acc = client.get_account()
        btc = next((b for b in acc.get("balances", []) if b["asset"] == "BTC"), None)
        if not btc or float(btc.get("free", 0)) <= 0:
            out = {"message": "no BTC to close", "broker": "binance"}
            _log_broker_response("close_position", out)
            return out
        qty = btc["free"]
        resp = client.create_order(
            symbol=BINANCE_SYMBOL,
            side=Client.SIDE_SELL,
            type=Client.ORDER_TYPE_MARKET,
            quantity=qty,
        )
        _log_broker_response("close_position", resp)
        return resp


def close_all_positions(*, account_id_override: int | None = None) -> dict:
    """Close all open positions on the given account (or default PS_CTRADER_ACCOUNT_ID)."""
    return close_position(position_id=None, account_id_override=account_id_override)


def cancel_all_orders() -> dict:
    """Cancel all open orders."""
    if not EXECUTION_PAPER_ONLY:
        raise RuntimeError("EXECUTION_PAPER_ONLY must be True")

    _setup_logging()
    broker = _get_broker()

    if broker == "ctrader":
        # cTrader market orders fill immediately; no pending orders to cancel
        out = {"cancelled": [], "broker": "ctrader", "note": "market orders only"}
        _log_broker_response("cancel_all_orders", out)
        return out

    elif broker == "oanda":
        token = os.environ.get("OANDA_ACCESS_TOKEN")
        account_id = os.environ.get("OANDA_ACCOUNT_ID")
        if not token or not account_id:
            out = {"simulated": True, "cancelled": [], "broker": "oanda"}
            _log_broker_response("cancel_all_orders", out)
            return out

        from oandapyV20 import API
        from oandapyV20.endpoints import orders

        api = API(access_token=token, environment="practice")
        r = orders.OrderList(account_id, params={"state": "PENDING"})
        resp = api.request(r)
        orders_list = resp.get("orders", [])
        cancelled = []
        for o in orders_list:
            oid = o["id"]
            cancel_r = orders.OrderCancel(account_id, orderID=oid)
            cr = api.request(cancel_r)
            cancelled.append(cr)
            _log_broker_response("cancel_all_orders", cr)
        return {"cancelled": cancelled}

    else:
        api_key = os.environ.get("BINANCE_API_KEY")
        api_secret = os.environ.get("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            out = {"simulated": True, "cancelled": [], "broker": "binance"}
            _log_broker_response("cancel_all_orders", out)
            return out

        from binance.client import Client

        client = Client(api_key, api_secret, testnet=True)
        open_orders = client.get_open_orders(symbol=BINANCE_SYMBOL)
        cancelled = []
        for o in open_orders:
            cr = client.cancel_order(symbol=BINANCE_SYMBOL, orderId=o["orderId"])
            cancelled.append(cr)
            _log_broker_response("cancel_all_orders", cr)
        return {"cancelled": cancelled}


def _current_position_from_broker() -> str:
    """Return 'long', 'short', or 'flat' based on broker state."""
    resp = get_open_positions()
    if resp.get("simulated"):
        return "flat"
    broker = _get_broker()
    if broker == "ctrader":
        positions = resp.get("positions", [])
        if not positions:
            return "flat"
        # Use first position's tradeSide (we only allow MAX_CONCURRENT_POSITIONS = 1)
        side = (positions[0].get("tradeSide") or "").upper()
        return "long" if side == "BUY" else "short" if side == "SELL" else "flat"
    elif broker == "oanda":
        positions = resp.get("positions", [])
        for p in positions:
            long_units = int(p.get("long", {}).get("units", 0))
            short_units = int(p.get("short", {}).get("units", 0))
            if long_units > 0:
                return "long"
            if short_units > 0:
                return "short"
        return "flat"
    else:
        balances = resp.get("balances", [])
        btc = next((b for b in balances if b.get("asset") == "BTC"), None)
        if not btc:
            return "flat"
        free = float(btc.get("free", 0))
        if free > 0.0001:
            return "long"
        return "flat"


def process_signal(
    row: dict,
    current_position: str,
    dry_run: bool = True,
    *,
    account_id_override: int | None = None,
) -> tuple[str, str]:
    """
    Process one signal row. Apply safeguards.
    Returns (action_taken, broker_response_json).
    Safeguards:
    - if blocked=1: do nothing, log NONE
    - if already in position and action is OPEN_*: allow only if reversal
    - one position max, no pyramiding

    account_id_override: cTrader login/ctid to route orders to (strategy's own account).
    """
    _setup_logging()
    action = row.get("action", "NONE")
    blocked = row.get("blocked", 0)
    timestamp = row.get("timestamp", "")
    signal = row.get("signal", "")
    confidence = row.get("confidence", 0)
    reason = row.get("reason", "")

    if blocked:
        logging.info("Signal blocked: action=NONE reason=%s", reason)
        return "NONE", json.dumps({"skipped": "blocked", "reason": reason})

    if action == "NONE":
        return "NONE", json.dumps({"skipped": "no_action"})

    if action == "CLOSE":
        if current_position == "flat":
            return "NONE", json.dumps({"skipped": "already_flat"})
        if dry_run:
            return "CLOSE_SIMULATED", json.dumps({"simulated": True, "action": "close"})
        resp = close_all_positions(account_id_override=account_id_override)
        return "CLOSE", json.dumps(resp, default=str)

    if action == "OPEN_LONG":
        if current_position == "long":
            return "NONE", json.dumps({"skipped": "already_long"})
        if current_position == "short":
            if dry_run:
                return "REVERSE_TO_LONG_SIMULATED", json.dumps({"simulated": True, "from": "short", "to": "long"})
            close_all_positions(account_id_override=account_id_override)
        if dry_run:
            return "OPEN_LONG_SIMULATED", json.dumps({"simulated": True, "side": "buy"})
        resp = place_market_order(side="buy", account_id_override=account_id_override)
        return "OPEN_LONG", json.dumps(resp, default=str)

    if action == "OPEN_SHORT":
        if current_position == "short":
            return "NONE", json.dumps({"skipped": "already_short"})
        if current_position == "long":
            if dry_run:
                return "REVERSE_TO_SHORT_SIMULATED", json.dumps({"simulated": True, "from": "long", "to": "short"})
            close_all_positions(account_id_override=account_id_override)
        if dry_run:
            return "OPEN_SHORT_SIMULATED", json.dumps({"simulated": True, "side": "sell"})
        resp = place_market_order(side="sell", account_id_override=account_id_override)
        return "OPEN_SHORT", json.dumps(resp, default=str)

    # Regression strategy: flip direction in one bar
    if action == "REVERSE_LONG":
        if current_position == "long":
            return "NONE", json.dumps({"skipped": "already_long"})
        if current_position == "short":
            if dry_run:
                return "REVERSE_TO_LONG_SIMULATED", json.dumps({"simulated": True, "from": "short", "to": "long"})
            close_all_positions(account_id_override=account_id_override)
        if dry_run:
            return "OPEN_LONG_SIMULATED", json.dumps({"simulated": True, "side": "buy"})
        resp = place_market_order(side="buy", account_id_override=account_id_override)
        return "OPEN_LONG", json.dumps(resp, default=str)

    if action == "REVERSE_SHORT":
        if current_position == "short":
            return "NONE", json.dumps({"skipped": "already_short"})
        if current_position == "long":
            if dry_run:
                return "REVERSE_TO_SHORT_SIMULATED", json.dumps({"simulated": True, "from": "long", "to": "short"})
            close_all_positions(account_id_override=account_id_override)
        if dry_run:
            return "OPEN_SHORT_SIMULATED", json.dumps({"simulated": True, "side": "sell"})
        resp = place_market_order(side="sell", account_id_override=account_id_override)
        return "OPEN_SHORT", json.dumps(resp, default=str)

    return "NONE", json.dumps({"skipped": "unknown_action"})


def run_with_signals(
    signal_rows: list[dict],
    dry_run: bool = True,
) -> None:
    """
    Process signal rows with execution. Log to CSV.
    dry_run=True: no broker calls, only simulate and log.
    """
    _setup_logging()
    EXECUTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = EXECUTION_LOG_DIR / f"execution_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    current_position = "flat"
    if not dry_run:
        current_position = _current_position_from_broker()

    rows = []
    for row in signal_rows:
        action_taken, broker_response = process_signal(row, current_position, dry_run=dry_run)

        if action_taken == "OPEN_LONG" or action_taken == "OPEN_SHORT":
            current_position = "long" if "LONG" in action_taken else "short"
        elif action_taken == "CLOSE" or "CLOSE" in action_taken:
            current_position = "flat"

        log_row = {
            "timestamp": row.get("timestamp", ""),
            "signal": row.get("signal", ""),
            "confidence": row.get("confidence", ""),
            "blocked": row.get("blocked", ""),
            "reason": row.get("reason", ""),
            "action_taken": action_taken,
            "broker_response": broker_response[:500],
        }
        rows.append(log_row)

    import csv
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        logging.info("Execution log saved to %s", csv_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demo execution: place test order, query positions, etc.")
    parser.add_argument("--test-order", action="store_true", help="Place a small test market order")
    parser.add_argument("--positions", action="store_true", help="Query open positions")
    parser.add_argument("--close-all", action="store_true", help="Close all positions")
    parser.add_argument("--cancel-orders", action="store_true", help="Cancel all orders")
    parser.add_argument("--run-signals", action="store_true", help="Run live_signal and process with execution (dry-run)")
    parser.add_argument("--live", action="store_true", help="Actually call broker (still demo/testnet)")
    parser.add_argument(
        "--account-id",
        type=int,
        default=None,
        help="Override cTrader account login/ctid (e.g. 4247810). Defaults to PS_CTRADER_ACCOUNT_ID.",
    )
    args = parser.parse_args()

    _acct = args.account_id
    if args.test_order:
        place_market_order(side="buy", account_id_override=_acct)
    elif args.positions:
        print(get_open_positions(account_id_override=_acct))
    elif args.close_all:
        close_all_positions(account_id_override=_acct)
    elif args.cancel_orders:
        cancel_all_orders()
    elif args.run_signals:
        from src.live_signal import run as run_live_signal
        rows = run_live_signal(n_bars=50)
        run_with_signals(rows, dry_run=not args.live)
    else:
        parser.print_help()
