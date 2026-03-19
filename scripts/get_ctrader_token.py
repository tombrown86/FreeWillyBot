#!/usr/bin/env python
"""
One-time OAuth2 helper — get a cTrader access token.

Run once:
    python scripts/get_ctrader_token.py

It will:
  1. Open the cTrader auth page in your browser.
  2. Ask you to paste the ?code= value from the redirect URL.
  3. Exchange it for an access + refresh token.
  4. Print the values so you can paste them into .env.

Required env vars (already in .env):
    PS_CTRADER_CLIENTID
    PS_CTRADER_SECRET

Redirect URI must match your app in the cTrader Open API portal exactly
(same scheme, host, path, trailing slash). Set PS_CTRADER_REDIRECT_URI in .env.
Examples: http://localhost:8080/callback  or  https://github.com/youruser
If you used a GitHub profile URL, the browser may show GitHub (or 404) after login —
that is fine; copy the ?code=... value from the address bar.
If you see "does not contain provided URI", the .env value does not match the portal.

This script opens the official grant page on id.ctrader.com (not openapi.ctrader.com/apps/auth).
Codes from the wrong flow can yield ACCESS_DENIED on token exchange.

Add to .env after running:
    PS_CTRADER_ACCESS_TOKEN=<printed value>
    PS_CTRADER_REFRESH_TOKEN=<printed value>
    PS_CTRADER_ACCOUNT_ID=<your numeric demo account ID, e.g. PS_CTRADEER_LOGIN>
"""

import os
import sys
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse, urlencode

import requests

# Load .env from repo root
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    from dotenv import load_dotenv
    load_dotenv(_env)

# Official OAuth grant page (see help.ctrader.com open-api account-authentication)
CTRADER_GRANT_PAGE = "https://id.ctrader.com/my/settings/openapi/grantingaccess/"
CTRADER_TOKEN_URL = "https://openapi.ctrader.com/apps/token"


def _normalize_auth_code(raw: str) -> str:
    """Accept raw code, full redirect URL, or 'code=...' fragment."""
    raw = raw.strip().strip('"').strip("'")
    if not raw:
        return raw
    if raw.startswith("code="):
        raw = raw[5:].split("&")[0]
    if "code=" in raw:
        parsed = urlparse(raw if "://" in raw else f"https://dummy.invalid/?{raw}")
        codes = parse_qs(parsed.query).get("code")
        if codes:
            return unquote(codes[0].strip())
    return unquote(raw)


def _exchange_authorization_code(
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    code: str,
) -> dict:
    """GET apps/token per cTrader docs (same as ctrader_open_api.Auth.getToken)."""
    r = requests.get(
        CTRADER_TOKEN_URL,
        params={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "client_secret": client_secret,
        },
        headers={"Accept": "application/json"},
        timeout=30,
    )
    try:
        return r.json()
    except Exception:
        return {
            "errorCode": "INVALID_RESPONSE",
            "description": f"HTTP {r.status_code}: {r.text[:300]}",
        }


def main() -> None:
    raw_id = os.environ.get("PS_CTRADER_CLIENTID", "")
    raw_secret = os.environ.get("PS_CTRADER_SECRET", "")
    client_id = raw_id.strip()
    client_secret = raw_secret.strip()
    redirect_uri = os.environ.get("PS_CTRADER_REDIRECT_URI", "").strip()

    if raw_secret != client_secret or raw_id != client_id:
        print("NOTE: Trimmed leading/trailing spaces from client id and/or secret in .env.")
        print()

    if not client_id or not client_secret:
        print("ERROR: PS_CTRADER_CLIENTID and PS_CTRADER_SECRET must be set in .env")
        sys.exit(1)
    if not redirect_uri:
        print("ERROR: PS_CTRADER_REDIRECT_URI must be set in .env to the exact redirect")
        print("       URL registered on your Open API application (e.g. https://github.com/tombrown86).")
        sys.exit(1)

    print(f"App client ID : {client_id[:12]}…")
    print(f"Redirect URI  : {redirect_uri}")
    print("  (must match one URI in your cTrader Open API app settings)")
    print()

    # Official grant URL (includes product=web per docs)
    auth_uri = CTRADER_GRANT_PAGE + "?" + urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": "trading",
            "product": "web",
        }
    )

    print("Opening official cTrader grant page in your browser…")
    print(f"  {auth_uri}")
    webbrowser.open(auth_uri)

    print()
    print("After you approve, your browser will redirect to your registered URI with:")
    print(f"  {redirect_uri}?code=XXXXXXXX   (or &code= if the URL already has query params)")
    print("You can paste the full address bar URL or only the code.")
    print("The authorization code expires in about 1 minute — paste it immediately.")
    print()
    code = _normalize_auth_code(input("Paste the code or full redirect URL here: "))
    if not code:
        print("No code entered, exiting.")
        sys.exit(1)

    token = _exchange_authorization_code(client_id, client_secret, redirect_uri, code)
    if "accessToken" not in token:
        print("Token exchange failed:", token)
        desc = str(token.get("description", token))
        err = str(token.get("errorCode", ""))
        if "URI" in desc or "uri" in desc.lower():
            print()
            print("Hint: set PS_CTRADER_REDIRECT_URI in .env to the exact redirect URL")
            print("      registered for this app (Open API application page).")
        if err == "ACCESS_DENIED" or "ACCESS_DENIED" in desc:
            print()
            print("ACCESS_DENIED — check:")
            print("  • You used THIS run’s browser link (id.ctrader.com grant page). Codes from")
            print("    openapi.ctrader.com/apps/auth or the Playground may not exchange here.")
            print("  • Authorization code expires in ~1 minute — paste immediately after redirect.")
            print("  • Each code is one-time; don’t re-paste an old code.")
            print("  • PS_CTRADER_SECRET / PS_CTRADER_REDIRECT_URI must match the Open API app exactly.")
            print("  • Or skip code flow: Open API portal → your app → Playground → copy tokens to .env.")
        sys.exit(1)

    print()
    print("─" * 60)
    print("Add these to your .env file:")
    print()
    print(f"PS_CTRADER_ACCESS_TOKEN={token['accessToken']}")
    print(f"PS_CTRADER_REFRESH_TOKEN={token.get('refreshToken', '')}")
    print()
    print(f"Token expires in: {token.get('expiresIn', '?')} seconds (~{int(token.get('expiresIn', 0)) // 86400} days)")
    print("─" * 60)
    print()
    print("Your numeric account ID is PS_CTRADEER_LOGIN in .env.")
    print("Add: PS_CTRADER_ACCOUNT_ID=<that number>")


if __name__ == "__main__":
    main()
