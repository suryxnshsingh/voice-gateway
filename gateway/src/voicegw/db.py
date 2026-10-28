"""SQLite store for customers, API keys, voices, consent records, usage and the public request log.

One file, stdlib only. Fine for a single gateway box; the interface is small enough to move to
Postgres when there is more than one.
"""

import hashlib
import json
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS api_keys (
    hash TEXT PRIMARY KEY,
    prefix TEXT NOT NULL,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    label TEXT,
    created_at REAL NOT NULL,
    revoked_at REAL
);
CREATE TABLE IF NOT EXISTS voices (
    id TEXT PRIMARY KEY,
    customer_id TEXT REFERENCES customers(id),   -- NULL = shared with every customer
    name TEXT NOT NULL,
    speaker_name TEXT NOT NULL,
    status TEXT NOT NULL,                        -- pending_consent | active | revoked
    reference_seconds REAL NOT NULL,
    consent_phrase TEXT,
    consent_code TEXT,
    consent_expires_at REAL,
    consent_attempts INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    activated_at REAL,
    revoked_at REAL
);
CREATE TABLE IF NOT EXISTS consents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    voice_id TEXT NOT NULL REFERENCES voices(id),
    method TEXT NOT NULL,                        -- recorded_statement | offline_attestation
    passed INTEGER NOT NULL,
    detail TEXT NOT NULL,                        -- JSON: transcript, scores, attestation, operator
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS usage_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    session_id TEXT,
    kind TEXT NOT NULL,                          -- stt_seconds | llm_input_tokens | llm_output_tokens | tts_characters | tts_seconds | tone_calls
    quantity REAL NOT NULL,
    provider TEXT NOT NULL,
    model TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_by_customer ON usage_events(customer_id, created_at);
-- One row per public API request (TTS, streaming TTS, STT), successful or not. With Supabase
-- accounts, accounts.UsageSync copies pending rows to the dashboard's usage_events.
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    api_key_id INTEGER,
    endpoint TEXT NOT NULL,                      -- tts | tts_stream | stt
    voice_id TEXT,
    characters INTEGER NOT NULL,
    audio_seconds REAL NOT NULL,
    status_code INTEGER NOT NULL,
    latency_ms INTEGER,                          -- TTS: to the first audio; STT: the whole request
    created_at REAL NOT NULL,
    sync TEXT NOT NULL                           -- none (local accounts) | pending | sent | failed: <why>
);
CREATE INDEX IF NOT EXISTS requests_by_customer ON requests(customer_id, created_at);
CREATE INDEX IF NOT EXISTS requests_pending ON requests(created_at) WHERE sync = 'pending';
"""

# Columns added after the first release; applied to existing databases on startup.
MIGRATIONS = [
    "ALTER TABLE voices ADD COLUMN persona TEXT",    # default LLM instructions when a session sets none
    "ALTER TABLE voices ADD COLUMN greeting TEXT",   # spoken first when a session asks to be greeted
]


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


@dataclass
class Customer:
    id: str
    name: str
    key_id: int | None = None   # the Supabase api_keys row the request came in with


class DB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        for sql in MIGRATIONS:
            try:
                self._conn.execute(sql)
            except sqlite3.OperationalError:
                pass  # already applied
        self._lock = threading.Lock()

    def _exec(self, sql: str, args=()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, args)

    def _one(self, sql: str, args=()) -> dict | None:
        row = self._exec(sql, args).fetchone()
        return dict(row) if row else None

    def _all(self, sql: str, args=()) -> list[dict]:
        return [dict(r) for r in self._exec(sql, args).fetchall()]

    # customers and keys

    def create_customer(self, name: str) -> Customer:
        c = Customer(new_id("cust"), name)
        self._exec("INSERT INTO customers VALUES (?, ?, ?)", (c.id, c.name, time.time()))
        return c

    def customers(self) -> list[dict]:
        return self._all("SELECT * FROM customers ORDER BY created_at")

    def customer(self, customer_id: str) -> Customer | None:
        row = self._one("SELECT id, name FROM customers WHERE id = ?", (customer_id,))
        return Customer(**row) if row else None

    def create_api_key(self, customer_id: str, label: str = "") -> str:
        key = "vg_" + secrets.token_urlsafe(32)
        self._exec(
            "INSERT INTO api_keys VALUES (?, ?, ?, ?, ?, NULL)",
            (_hash_key(key), key[:10], customer_id, label, time.time()),
        )
        return key

    def customer_for_key(self, key: str) -> Customer | None:
        row = self._one(
            "SELECT c.id, c.name FROM api_keys k JOIN customers c ON c.id = k.customer_id "
            "WHERE k.hash = ? AND k.revoked_at IS NULL",
            (_hash_key(key),),
        )
        return Customer(**row) if row else None

    # voices

    def create_voice(self, *, customer_id: str | None, name: str, speaker_name: str, reference_seconds: float,
                     consent_phrase: str | None, consent_code: str | None, consent_expires_at: float | None,
                     status: str = "pending_consent") -> dict:
        vid = new_id("voice")
        self._exec(
            "INSERT INTO voices (id, customer_id, name, speaker_name, status, reference_seconds, consent_phrase, "
            "consent_code, consent_expires_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (vid, customer_id, name, speaker_name, status, reference_seconds, consent_phrase, consent_code,
             consent_expires_at, time.time()),
        )
        return self.voice(vid)

    def voice(self, voice_id: str) -> dict | None:
        return self._one("SELECT * FROM voices WHERE id = ?", (voice_id,))

    def voice_for(self, customer_id: str, voice_id: str) -> dict | None:
        """A voice the customer may use or manage: its own, or a shared one."""
        return self._one(
            "SELECT * FROM voices WHERE id = ? AND (customer_id = ? OR customer_id IS NULL) AND status != 'revoked'",
            (voice_id, customer_id),
        )

    def voices_for(self, customer_id: str) -> list[dict]:
        return self._all(
            "SELECT * FROM voices WHERE (customer_id = ? OR customer_id IS NULL) AND status != 'revoked' "
            "ORDER BY created_at DESC",
            (customer_id,),
        )

    def set_persona(self, voice_id: str, persona: str | None, greeting: str | None) -> None:
        self._exec("UPDATE voices SET persona = ?, greeting = ? WHERE id = ?", (persona, greeting, voice_id))

    def set_reference_seconds(self, voice_id: str, seconds: float) -> None:
        self._exec("UPDATE voices SET reference_seconds = ? WHERE id = ?", (seconds, voice_id))

    def bump_consent_attempts(self, voice_id: str) -> None:
        self._exec("UPDATE voices SET consent_attempts = consent_attempts + 1 WHERE id = ?", (voice_id,))

    def activate_voice(self, voice_id: str) -> None:
        self._exec("UPDATE voices SET status = 'active', activated_at = ? WHERE id = ?", (time.time(), voice_id))

    def revoke_voice(self, voice_id: str) -> None:
        self._exec("UPDATE voices SET status = 'revoked', revoked_at = ? WHERE id = ?", (time.time(), voice_id))

    def add_consent(self, voice_id: str, method: str, passed: bool, detail: dict) -> None:
        self._exec(
            "INSERT INTO consents (voice_id, method, passed, detail, created_at) VALUES (?, ?, ?, ?, ?)",
            (voice_id, method, int(passed), json.dumps(detail, ensure_ascii=False), time.time()),
        )

    def consents(self, voice_id: str) -> list[dict]:
        rows = self._all("SELECT * FROM consents WHERE voice_id = ? ORDER BY created_at", (voice_id,))
        for r in rows:
            r["detail"] = json.loads(r["detail"])
        return rows

    # usage

    def add_usage(self, rows: list[tuple]) -> None:
        """rows: (customer_id, session_id, kind, quantity, provider, model)."""
        now = time.time()
        with self._lock:
            self._conn.executemany(
                "INSERT INTO usage_events (customer_id, session_id, kind, quantity, provider, model, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(*r, now) for r in rows if r[3]],
            )

    def usage_summary(self, customer_id: str, since: float, until: float) -> list[dict]:
        return self._all(
            "SELECT kind, provider, model, SUM(quantity) AS quantity, COUNT(DISTINCT session_id) AS sessions "
            "FROM usage_events WHERE customer_id = ? AND created_at >= ? AND created_at < ? "
            "GROUP BY kind, provider, model ORDER BY kind",
            (customer_id, since, until),
        )

    # public request log

    def add_request(self, row: dict, sync: str) -> None:
        self._exec(
            "INSERT OR IGNORE INTO requests (request_id, customer_id, api_key_id, endpoint, voice_id, characters, "
            "audio_seconds, status_code, latency_ms, created_at, sync) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (row["request_id"], row["customer_id"], row["api_key_id"], row["endpoint"], row["voice_id"],
             row["characters"], row["audio_seconds"], row["status_code"], row["latency_ms"], row["created_at"], sync),
        )

    def pending_requests(self, limit: int) -> list[dict]:
        return self._all("SELECT * FROM requests WHERE sync = 'pending' ORDER BY created_at LIMIT ?", (limit,))

    def mark_requests(self, request_ids: list[str], sync: str) -> None:
        with self._lock:
            self._conn.executemany("UPDATE requests SET sync = ? WHERE request_id = ?", [(sync, r) for r in request_ids])

    def request_counts(self) -> dict[str, int]:
        rows = self._all("SELECT CASE WHEN sync LIKE 'failed%' THEN 'failed' ELSE sync END AS s, COUNT(*) AS n "
                         "FROM requests GROUP BY s")
        return {r["s"]: r["n"] for r in rows}

    def request_totals(self, customer_id: str, since: float, until: float) -> dict[str, float]:
        """Billable totals: successful requests only, as the dashboard prices them."""
        row = self._one(
            "SELECT COALESCE(SUM(CASE WHEN endpoint != 'stt' THEN characters END), 0) AS tts_characters, "
            "COALESCE(SUM(CASE WHEN endpoint != 'stt' THEN audio_seconds END), 0) AS tts_seconds, "
            "COALESCE(SUM(CASE WHEN endpoint = 'stt' THEN audio_seconds END), 0) AS stt_seconds "
            "FROM requests WHERE customer_id = ? AND created_at >= ? AND created_at < ? AND status_code < 400",
            (customer_id, since, until),
        )
        return {k: float(v) for k, v in row.items()}
