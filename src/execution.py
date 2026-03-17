"""
Phase 15 — Demo execution plumbing.

Connect to OANDA (FX) or Binance (crypto) demo/testnet only.
place_market_order, get_open_positions, close_position, cancel_all_orders.
Log every broker response. Never hard-code secrets.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    BINANCE_SYMBOL,
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
    """Return 'oanda' or 'binance' based on SYMBOL."""
    return "binance" if "USDT" in SYMBOL else "oanda"


def _log_broker_response(operation: str, response: dict | str) -> None:
    """Log full broker response."""
    if isinstance(response, dict):
        body = json.dumps(response, default=str)
    else:
        body = str(response)
    logging.info("Broker response [%s]: %s", operation, body)


def place_market_order(
    side: str,
    units: float | None = None,
) -> dict:
    """
    Place a small test market order. Paper/demo only.
    side: 'buy' or 'sell'
    """
    if not EXECUTION_PAPER_ONLY:
        raise RuntimeError("EXECUTION_PAPER_ONLY must be True; no live execution")

    _setup_logging()
    broker = _get_broker()
    units = units or EXECUTION_TEST_UNITS

    if broker == "oanda":
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


def get_open_positions() -> dict:
    """Query open positions. Log response."""
    _setup_logging()
    broker = _get_broker()

    if broker == "oanda":
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


def close_position(position_id: str | None = None) -> dict:
    """Close a position. For OANDA pass position_id; for Binance close BTC position."""
    if not EXECUTION_PAPER_ONLY:
        raise RuntimeError("EXECUTION_PAPER_ONLY must be True")

    _setup_logging()
    broker = _get_broker()

    if broker == "oanda":
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


def close_all_positions() -> dict:
    """Close all open positions."""
    return close_position(position_id=None)


def cancel_all_orders() -> dict:
    """Cancel all open orders."""
    if not EXECUTION_PAPER_ONLY:
        raise RuntimeError("EXECUTION_PAPER_ONLY must be True")

    _setup_logging()
    broker = _get_broker()

    if broker == "oanda":
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
    if broker == "oanda":
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
) -> tuple[str, str]:
    """
    Process one signal row. Apply safeguards.
    Returns (action_taken, broker_response_json).
    Safeguards:
    - if blocked=1: do nothing, log NONE
    - if already in position and action is OPEN_*: allow only if reversal
    - one position max, no pyramiding
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
        resp = close_all_positions()
        return "CLOSE", json.dumps(resp, default=str)

    if action == "OPEN_LONG":
        if current_position == "long":
            return "NONE", json.dumps({"skipped": "already_long"})
        if current_position == "short":
            if dry_run:
                return "CLOSE_THEN_OPEN_SIMULATED", json.dumps({"simulated": True})
            close_all_positions()
        if dry_run:
            return "OPEN_LONG_SIMULATED", json.dumps({"simulated": True, "side": "buy"})
        resp = place_market_order(side="buy")
        return "OPEN_LONG", json.dumps(resp, default=str)

    if action == "OPEN_SHORT":
        if current_position == "short":
            return "NONE", json.dumps({"skipped": "already_short"})
        if current_position == "long":
            if dry_run:
                return "CLOSE_THEN_OPEN_SIMULATED", json.dumps({"simulated": True})
            close_all_positions()
        if dry_run:
            return "OPEN_SHORT_SIMULATED", json.dumps({"simulated": True, "side": "sell"})
        resp = place_market_order(side="sell")
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
    args = parser.parse_args()

    if args.test_order:
        place_market_order(side="buy")
    elif args.positions:
        print(get_open_positions())
    elif args.close_all:
        close_all_positions()
    elif args.cancel_orders:
        cancel_all_orders()
    elif args.run_signals:
        from src.live_signal import run as run_live_signal
        rows = run_live_signal(n_bars=50)
        run_with_signals(rows, dry_run=not args.live)
    else:
        parser.print_help()
