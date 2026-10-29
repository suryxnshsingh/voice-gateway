"""Accounts: who an API key belongs to, which voices it may use, and where usage goes.

Two backends with the same interface:

- LocalAccounts: the gateway's own SQLite (development, tests, the internal voice agent).
- SupabaseAccounts: the Dhvni dashboard's Supabase project, used when VG_SUPABASE_URL is set.
  Keys, accounts, voices and the usage log live there (dhvni-web README, "What the voice gateway
  needs to do"). Lookups go through PostgREST with the secret key and are cached briefly; when
  Supabase can't be reached, the last answer is used for up to ten minutes. A signed in dashboard
  session (its Supabase access token) is accepted like a key, for the dashboard's playground; its
  requests carry no key id.

Every public request is written to the local `requests` table first. With Supabase, UsageSync
copies those rows to `usage_events` in batches, and the database prices them.
"""

import asyncio
import hashlib
import logging
import time
from datetime import datetime

import httpx
from fastapi import HTTPException

from .db import DB, Customer

log = logging.getLogger(__name__)

STALE_S = 600
NOT_ACTIVE = {
    "pending": "This account is waiting for approval.",
    "suspended": "This account is suspended. Please contact support.",
}


def public_voice(v: dict) -> dict:
    """A local voice row as the API shows it."""
    return {
        "id": v["id"],
        "name": v["name"],
        "speaker_name": v["speaker_name"],
        "status": v["status"],
        "shared": v["customer_id"] is None,
        "reference_seconds": v["reference_seconds"],
        "fine_tuned": False,
        "created_at": v["created_at"],
        "activated_at": v["activated_at"],
    }


def without_adapter(v: dict) -> dict:
    """What the API shows of a voice: whether it is fine tuned, never which adapter it runs on."""
    return {k: x for k, x in v.items() if k != "adapter"}


def _unix(ts: str | None) -> float | None:
    return round(datetime.fromisoformat(ts).timestamp(), 2) if ts else None


def _require_active(status: str) -> None:
    if status != "active":
        raise HTTPException(403, NOT_ACTIVE.get(status, "This account is not active. Please contact support."))


class LocalAccounts:
    remote = False

    def __init__(self, db: DB):
        self.db = db

    async def for_key(self, key: str) -> Customer | None:
        return self.db.customer_for_key(key)

    async def for_token(self, claims: dict) -> Customer | None:
        return self.db.customer(claims["c"])

    async def for_session(self, access_token: str, key_id: str | None = None) -> Customer | None:
        return None

    async def voices(self, customer: Customer) -> list[dict]:
        return [public_voice(v) for v in self.db.voices_for(customer.id) if v["status"] == "active"]

    async def voice(self, customer: Customer, voice_id: str) -> dict | None:
        v = self.db.voice_for(customer.id, voice_id)
        return public_voice(v) if v and v["status"] == "active" else None

    async def balance_paise(self, customer: Customer) -> float | None:
        return None

    async def close(self) -> None:
        pass


class SupabaseError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"supabase {status}: {body[:300]}")
        self.status = status


class SupabaseAccounts:
    remote = True

    def __init__(self, url: str, secret_key: str, cache_s: float = 30, transport: httpx.AsyncBaseTransport | None = None):
        # supabase-js sends the key in both headers; PostgREST then runs as service_role.
        self.url = url.rstrip("/")
        self.http = httpx.AsyncClient(
            base_url=self.url + "/rest/v1",
            headers={"apikey": secret_key, "Authorization": f"Bearer {secret_key}"},
            timeout=httpx.Timeout(10, connect=5),
            transport=transport,
        )
        self.cache_s = cache_s
        self._cache: dict[tuple, tuple[float, object]] = {}

    async def close(self) -> None:
        await self.http.aclose()

    async def _call(self, method: str, path: str, **kw):
        r = await self.http.request(method, path, **kw)
        if r.status_code >= 300:
            raise SupabaseError(r.status_code, r.text)
        return r.json() if r.content else None

    async def _cached(self, key: tuple, fetch, ttl: float | None = None):
        now = time.monotonic()
        hit = self._cache.get(key)
        if hit and hit[0] > now:
            return hit[1]
        try:
            value = await fetch()
        except (httpx.HTTPError, SupabaseError) as e:
            if hit and now - hit[0] < STALE_S:
                log.warning("supabase unreachable, using a cached answer for %s: %r", key[0], e)
                return hit[1]
            log.error("supabase lookup %s failed: %r", key[0], e)
            raise HTTPException(503, "The service is briefly unavailable. Please retry.",
                                headers={"Retry-After": "5"}) from e
        if len(self._cache) > 20000:  # unknown keys add entries too; drop the expired ones
            self._cache = {k: v for k, v in self._cache.items() if v[0] > now}
        self._cache[key] = (now + (self.cache_s if ttl is None else ttl), value)
        return value

    async def for_key(self, key: str) -> Customer | None:
        digest = hashlib.sha256(key.encode()).hexdigest()

        async def fetch():
            rows = await self._call("POST", "/rpc/verify_api_key", json={"p_key": key})
            return rows[0] if rows else None

        row = await self._cached(("key", digest), fetch)
        if row is None:
            return None
        _require_active(row["status"])
        return Customer(id=row["user_id"], name="", key_id=row["api_key_id"])

    async def session_user(self, access_token: str) -> tuple[str, str, str | None] | None:
        """Whose dashboard session this is: (user id, account status, role), or None if it isn't one."""
        digest = hashlib.sha256(access_token.encode()).hexdigest()

        async def user():
            r = await self.http.get(self.url + "/auth/v1/user", headers={"Authorization": f"Bearer {access_token}"})
            if r.status_code in (401, 403):
                return None  # expired or not a session of this project
            if r.status_code >= 300:
                raise SupabaseError(r.status_code, r.text)
            return r.json()["id"]

        user_id = await self._cached(("session", digest), user)
        if user_id is None:
            return None

        async def profile():
            rows = await self._call("GET", "/profiles", params={"select": "status,role", "id": f"eq.{user_id}"})
            return (rows[0]["status"], rows[0].get("role")) if rows else None

        found = await self._cached(("profile", user_id), profile)
        return None if found is None else (user_id, found[0], found[1])

    async def for_session(self, access_token: str, key_id: str | None = None) -> Customer | None:
        """A dashboard session: Supabase Auth says whose it is, the profile says whether it may be used.

        A customer's playground names one of their live API keys (`key_id`), and its requests count
        against that key like any other. An admin may leave it out: those requests skip the credit check
        and Supabase prices them at 0, since they carry no key."""
        found = await self.session_user(access_token)
        if found is None:
            return None
        user_id, status, role = found
        _require_active(status)

        if not key_id:
            if role == "admin":
                return Customer(id=user_id, name="", free=True, playground=True)
            raise HTTPException(400, "Choose one of your API keys to use the playground.")
        if not key_id.isdigit():
            raise HTTPException(400, "X-Api-Key-Id must be the id of one of your API keys.")

        async def key():
            return await self._call("GET", "/api_keys", params={
                "select": "revoked_at", "id": f"eq.{key_id}", "user_id": f"eq.{user_id}"})

        rows = await self._cached(("session_key", key_id, user_id), key)
        if not rows or rows[0]["revoked_at"] is not None:
            raise HTTPException(403, "That API key is revoked or not on this account. Choose another one.")
        return Customer(id=user_id, name="", key_id=int(key_id), playground=True)

    async def for_token(self, claims: dict) -> Customer | None:
        """A browser token minted from a key: the key must still be live and the account active."""
        user_id, key_id = claims["c"], claims.get("k")
        if key_id is None:
            return None

        async def fetch():
            return await self._call("GET", "/api_keys", params={
                "select": "revoked_at,profiles(status)", "id": f"eq.{key_id}", "user_id": f"eq.{user_id}"})

        rows = await self._cached(("token_key", key_id, user_id), fetch)
        if not rows or rows[0]["revoked_at"] is not None:
            return None
        _require_active((rows[0].get("profiles") or {}).get("status", ""))
        return Customer(id=user_id, name="", key_id=key_id)

    async def voices(self, customer: Customer) -> list[dict]:
        async def fetch():
            return await self._call("GET", "/voices", params={
                "select": "id,user_id,name,speaker_name,status,reference_seconds,adapter,created_at,activated_at",
                "status": "eq.active", "or": f"(user_id.eq.{customer.id},user_id.is.null)",
                "order": "created_at.desc"})

        rows = await self._cached(("voices", customer.id), fetch)
        return [{
            "id": v["id"], "name": v["name"], "speaker_name": v["speaker_name"], "status": v["status"],
            "shared": v["user_id"] is None, "reference_seconds": float(v["reference_seconds"]),
            "fine_tuned": bool(v.get("adapter")), "adapter": v.get("adapter"),
            "created_at": _unix(v["created_at"]), "activated_at": _unix(v["activated_at"]),
        } for v in rows]

    async def voice(self, customer: Customer, voice_id: str) -> dict | None:
        return next((v for v in await self.voices(customer) if v["id"] == voice_id), None)

    # Voice requests, for automatic cloned voices from the dashboard (dashboard_clone.py). -------

    async def clone_request(self, request_id: int, user_id: str) -> dict | None:
        rows = await self._call("GET", "/voice_requests", params={
            "select": "id,user_id,kind,consent,status,consent_words,auto_attempts",
            "id": f"eq.{request_id}", "user_id": f"eq.{user_id}"})
        return rows[0] if rows else None

    async def allowance(self, user_id: str) -> dict:
        rows = await self._call("POST", "/rpc/voice_allowance", json={"p_user": user_id})
        return rows[0] if rows else {"clone_limit": 0, "clones_used": 0, "fine_tune_limit": 0, "fine_tunes_used": 0}

    async def request_files(self, user_id: str, request_id: int, folder: str) -> list[dict]:
        """A folder of a request in the voice-requests bucket, oldest first: [{"name", "path", "size"}]."""
        prefix = f"{user_id}/{request_id}/{folder}"
        r = await self.http.post(self.url + "/storage/v1/object/list/voice-requests", json={
            "prefix": prefix, "limit": 100, "offset": 0, "sortBy": {"column": "created_at", "order": "asc"}})
        if r.status_code >= 300:
            raise SupabaseError(r.status_code, r.text)
        return [{"name": o["name"], "path": f"{prefix}/{o['name']}", "size": (o.get("metadata") or {}).get("size", 0)}
                for o in r.json() if o.get("id")]

    async def download(self, path: str) -> bytes:
        r = await self.http.get(self.url + f"/storage/v1/object/voice-requests/{path}")
        if r.status_code >= 300:
            raise SupabaseError(r.status_code, r.text)
        return r.content

    async def record_clone_check(self, request_id: int, check: dict) -> int:
        return int(await self._call("POST", "/rpc/record_clone_check", json={"p_request": request_id, "p_check": check}))

    async def complete_clone_request(self, request_id: int, voice_id: str, seconds: float, check: dict) -> None:
        await self._call("POST", "/rpc/complete_clone_request", json={
            "p_request": request_id, "p_voice_id": voice_id, "p_reference_seconds": round(seconds, 1), "p_check": check})

    def forget_voices(self, user_id: str) -> None:
        """A new voice is usable at once, not after the voices cache expires."""
        self._cache.pop(("voices", user_id), None)

    async def balance_paise(self, customer: Customer) -> float | None:
        async def fetch():
            rows = await self._call("POST", "/rpc/credit_balance", json={"p_user": customer.id})
            return float(rows[0]["balance_paise"]) if rows else 0.0

        return await self._cached(("balance", customer.id), fetch)


class UsageSync:
    """Copies pending rows of the local request log to Supabase's usage_events.

    Several rows per statement, since the rollup trigger runs once per statement. Inserts use
    `on conflict (request_id) do nothing`, so a batch that is sent twice is counted once. A row
    Supabase refuses (say, its account was deleted) is parked as failed so the rest still flow.
    """

    def __init__(self, db: DB, http: httpx.AsyncClient, interval_s: float = 2.0, batch: int = 200):
        self.db, self.http, self.interval_s, self.batch = db, http, interval_s, batch

    @staticmethod
    def _row(r: dict) -> dict:
        return {
            "request_id": r["request_id"], "user_id": r["customer_id"], "api_key_id": r["api_key_id"],
            "endpoint": r["endpoint"], "voice_id": (r["voice_id"] or None) and r["voice_id"][:64],
            "characters": r["characters"], "audio_seconds": round(r["audio_seconds"], 2),
            "status_code": r["status_code"], "latency_ms": r["latency_ms"],
            "created_at": datetime.fromtimestamp(r["created_at"]).astimezone().isoformat(),
            "source": r.get("source") or "api",
        }

    async def _post(self, rows: list[dict]) -> httpx.Response:
        return await self.http.post("/usage_events", params={"on_conflict": "request_id"},
                                    json=[self._row(r) for r in rows],
                                    headers={"Prefer": "resolution=ignore-duplicates,return=minimal"})

    async def flush(self) -> int:
        """Sends one batch. Returns how many rows left the queue; raises when Supabase is unreachable."""
        rows = self.db.pending_requests(self.batch)
        if not rows:
            return 0
        r = await self._post(rows)
        if r.status_code < 300:
            self.db.mark_requests([x["request_id"] for x in rows], "sent")
            return len(rows)
        if r.status_code in (401, 403, 404, 429) or r.status_code >= 500:
            raise SupabaseError(r.status_code, r.text)
        for row in rows:  # the batch has a row Supabase won't take: find it
            one = await self._post([row])
            if one.status_code < 300:
                self.db.mark_requests([row["request_id"]], "sent")
            elif one.status_code in (401, 403, 404, 429) or one.status_code >= 500:
                raise SupabaseError(one.status_code, one.text)
            else:
                log.error("usage row %s refused: %s", row["request_id"], one.text[:300])
                self.db.mark_requests([row["request_id"]], f"failed: {one.status_code} {one.text[:200]}")
        return len(rows)

    async def run(self) -> None:
        backoff = self.interval_s
        while True:
            try:
                sent = await self.flush()
                backoff = self.interval_s
                if sent == self.batch:
                    continue
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("usage sync failed, retrying in %.0f s: %r", backoff, e)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            await asyncio.sleep(self.interval_s)

    async def drain(self, timeout_s: float = 5) -> None:
        """On shutdown: send what's queued, briefly. Anything left goes out on the next start."""
        deadline = time.monotonic() + timeout_s
        try:
            while time.monotonic() < deadline and await asyncio.wait_for(self.flush(), timeout_s):
                pass
        except Exception as e:
            log.warning("usage left queued at shutdown: %r", e)
