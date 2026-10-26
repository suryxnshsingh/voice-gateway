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


def mint_session_token(customer_id: str, ttl_seconds: int = 120) -> tuple[str, int]:
    exp = int(time.time()) + ttl_seconds
    body = _b64(json.dumps({"c": customer_id, "e": exp}).encode())
    sig = _b64(hmac.new(_secret, body.encode(), hashlib.sha256).digest())
    return f"vgs_{body}.{sig}", exp


def read_session_token(token: str) -> str | None:
    if not token.startswith("vgs_") or "." not in token:
        return None
    body, sig = token[4:].split(".", 1)
    expected = _b64(hmac.new(_secret, body.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, expected):
        return None
    payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    return payload["c"] if payload["e"] >= time.time() else None


def bearer(header: str | None) -> str | None:
    if header and header.lower().startswith("bearer "):
        return header[7:].strip()
    return None


def customer_from(request: Request) -> Customer:
    """FastAPI dependency: Authorization: Bearer <api key>."""
    key = bearer(request.headers.get("authorization"))
    customer = request.app.state.db.customer_for_key(key) if key else None
    if customer is None:
        raise HTTPException(401, "invalid or missing API key")
    return customer


CurrentCustomer = Depends(customer_from)
