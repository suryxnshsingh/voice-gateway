"""API keys for servers; short-lived signed tokens for browsers (which can't set WebSocket headers)."""

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time

from fastapi import Depends, HTTPException, Request

from .config import settings
from .db import Customer

log = logging.getLogger(__name__)

_secret = settings.secret_key.encode()
if not _secret:
    _secret = secrets.token_bytes(32)
    log.warning("VG_SECRET_KEY is not set; session tokens will not survive a restart")


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def mint_session_token(customer_id: str, ttl_seconds: int = 120, key_id: int | None = None) -> tuple[str, int]:
    exp = int(time.time()) + ttl_seconds
    claims = {"c": customer_id, "e": exp} | ({"k": key_id} if key_id is not None else {})
    body = _b64(json.dumps(claims).encode())
    sig = _b64(hmac.new(_secret, body.encode(), hashlib.sha256).digest())
    return f"vgs_{body}.{sig}", exp


def read_session_claims(token: str) -> dict | None:
    """{"c": customer id, "e": expiry, "k": API key id when minted from a Supabase key}, or None."""
    if not token.startswith("vgs_") or "." not in token:
        return None
    body, sig = token[4:].split(".", 1)
    expected = _b64(hmac.new(_secret, body.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, expected):
        return None
    payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    return payload if payload["e"] >= time.time() else None


def read_session_token(token: str) -> str | None:
    claims = read_session_claims(token)
    return claims["c"] if claims else None


def bearer(header: str | None) -> str | None:
    if header and header.lower().startswith("bearer "):
        return header[7:].strip()
    return None


def looks_like_session(token: str) -> bool:
    """A Supabase access token (a JWT) rather than an API key."""
    return token.count(".") == 2 and not token.startswith(("vg_", "vgs_"))


async def customer_from(request: Request) -> Customer:
    """FastAPI dependency for the public API: Authorization: Bearer <api key>, or a signed in dashboard
    session for the playground, checked by app.state.accounts (Supabase in production). Raises 401 for an
    unknown key and 403 for an account that isn't active."""
    key = bearer(request.headers.get("authorization"))
    accounts = request.app.state.accounts
    if not key:
        customer = None
    elif looks_like_session(key):
        customer = await accounts.for_session(key)
    else:
        customer = await accounts.for_key(key)
    if customer is None:
        raise HTTPException(401, "Invalid or missing API key.")
    return customer


def local_customer_from(request: Request) -> Customer:
    """For internal routes (voice agent, self-serve cloning): keys in the gateway's own database only."""
    key = bearer(request.headers.get("authorization"))
    customer = request.app.state.db.customer_for_key(key) if key else None
    if customer is None:
        raise HTTPException(401, "invalid or missing API key")
    return customer


CurrentCustomer = Depends(customer_from)
LocalCustomer = Depends(local_customer_from)
