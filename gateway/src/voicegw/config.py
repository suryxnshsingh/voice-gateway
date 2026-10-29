from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VG_", env_file=".env", extra="ignore")

    # Public API branding. Customers only ever see these names, never the engines behind them.
    brand: str = "Dhvni"
    brand_tts_model: str = "dhvni-voice-1"
    brand_stt_model: str = "dhvni-transcribe-1"
    # Accounts, API keys, voices and the usage log from the Dhvni dashboard's Supabase project
    # (secret key, which skips row level security). Empty: the gateway's own SQLite (dev and tests).
    supabase_url: str = ""
    supabase_secret_key: str = ""
    # Seconds to reuse an answer from Supabase: a revoked key or a new voice takes up to this long.
    accounts_cache_s: float = 30
    # Refuse TTS and STT with 402 once an account's prepaid credit balance is used up.
    require_credit: bool = False
    # v1: operators clone voices for customers (voicegw import-voice); customers only list them.
    self_serve_voices: bool = False
    # The internal test page and the voice-agent WebSocket. Turn off on the public deployment.
    playground_enabled: bool = True
    # Engine behind the public STT endpoint (relayed as-is).
    stt_relay_model: str = "gpt-transcribe"
    # Per-customer limits (in-process; move to Redis when there is more than one gateway).
    tts_concurrency: int = 4
    stt_concurrency: int = 8
    requests_per_minute: int = 120
    tts_max_chars: int = 5000
    stt_max_bytes: int = 25 * 1024 * 1024

    # The dashboard's playground: a signed in dashboard session in place of an API key, for hearing a
    # voice on a few short lines, not for production use. Every playground limit is here (playground.py).
    # A session only reaches POST /v1/tts, POST /v1/stt and GET /v1/playground, never a WebSocket.
    playground_origins: list[str] = ["https://www.dhvni.in", "https://dhvni.in"]
    playground_tts_max_chars: int = 300           # per request
    playground_tts_chars_per_day: int = 3000      # per account and India day; admins have no daily limit
    playground_stt_max_seconds: float = 120       # per recording
    playground_stt_max_bytes: int = 10 * 1024 * 1024
    playground_stt_seconds_per_day: float = 600   # per account and India day
    playground_requests_per_minute: int = 10      # per account
    playground_concurrency: int = 1               # per account, for each of TTS and STT

    host: str = "127.0.0.1"
    port: int = 8000
    data_dir: Path = Path("./data")
    # Signs short-lived browser session tokens. Must be set (and stable) outside local dev.
    secret_key: str = ""

    worker_url: str = "http://127.0.0.1:8100"
    worker_token: str = ""

    # Speech-to-text providers in order of preference; a failing one is skipped for a while.
    stt_providers: list[str] = ["soniox", "openai"]
    openai_stt_model: str = "gpt-live-transcribe"
    openai_stt_delay: str = "minimal"
    # Words the transcriber should expect (names, brands, syllabus terms).
    stt_keywords: list[str] = [
        "Arivihan", "Ritesh Sir", "NCERT", "MP Board", "NEET", "JEE", "Unnati Batch",
        "mitochondria", "ATP", "DNA", "RNA", "photosynthesis", "chlorophyll", "enzyme", "Calvin cycle",
        "cell", "osmosis", "diffusion", "chromosome", "protein", "glucose", "oxygen", "carbon dioxide",
        "Newton", "velocity", "acceleration", "current", "voltage", "electron", "molecule", "equation",
        "revision", "syllabus", "chapter", "board exam",
    ]
    # For OpenAI speech-to-text. Defaults to the LLM key when the LLM is OpenAI itself.
    openai_api_key: str = ""

    soniox_api_key: str = ""
    soniox_ws_url: str = "wss://stt-rt.soniox.com/transcribe-websocket"
    # stt-rt-v3 was retired in Feb 2026 (per Soniox's model docs); v5 is current. Unverified here: account unfunded.
    soniox_model: str = "stt-rt-v5"

    # Any OpenAI-compatible chat endpoint (OpenAI, OpenRouter, Gemini's compat endpoint).
    llm_api_key: str = ""
    llm_base_url: str | None = None
    llm_model: str = "gpt-5.4-mini"
    # Classifies tone from transcript + acoustic scores. Small and fast; runs beside the reply.
    # gpt-4.1-mini answered this ~0.6 s faster than gpt-5.4-mini in testing (Oct 2026).
    tone_model: str = "gpt-4.1-mini"
    # Reasoning models (gpt-5.x/6.x) think before the first token unless told not to; for voice
    # that costs ~1 s (gpt-6-luna: 1.9 s -> 0.9 s first token with "none", Oct 2026). Empty = model default.
    llm_reasoning_effort: str = ""
    # OpenAI service tier for replies; "priority" measured ~200 ms faster to the first sentence at 2x price.
    llm_service_tier: str = ""
    llm_max_tokens: int = 400
    history_turns: int = 12

    # How long a reply may wait for this turn's acoustic tone before the LLM starts. 0 (default): don't
    # wait; the tone still reaches the transcript events and every later turn.
    tone_wait_ms: int = 0
    # Assistant audio queued on the client. Smaller reacts faster to pause/stop; larger rides out jitter.
    playout_lead_ms: int = 200
    # Audio to render before playback starts. 0 on a GPU (TTS is several times real time); ~500 on
    # a Mac, where VoxCPM2 runs at about real time and would otherwise stutter.
    playout_prebuffer_ms: int = 0

    # Consent check: ECAPA cosine between the reference and the consent recording.
    consent_speaker_threshold: float = 0.40
    consent_ttl_hours: int = 24
    consent_max_attempts: int = 5
    voice_min_seconds: float = 5.0
    voice_max_seconds: float = 120.0

    @property
    def openai_key(self) -> str:
        return self.openai_api_key or (self.llm_api_key if not self.llm_base_url else "")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "voicegw.sqlite3"

    @property
    def voices_dir(self) -> Path:
        return self.data_dir / "voices"


settings = Settings()
