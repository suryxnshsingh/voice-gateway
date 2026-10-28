"""voicegw: run the gateway and manage customers, keys and voices."""

import argparse
import asyncio
import json
import logging
import sys
import time


def _db():
    from .config import settings
    from .db import DB

    return DB(settings.db_path)


def cmd_serve(args) -> None:
    import uvicorn

    from .config import settings

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run("voicegw.app:app", host=settings.host, port=settings.port, ws_max_size=4 * 1024 * 1024)


def cmd_create_customer(args) -> None:
    db = _db()
    c = db.create_customer(args.name)
    key = db.create_api_key(c.id, "default")
    print(json.dumps({"customer_id": c.id, "name": c.name, "api_key": key}, indent=2))
    print("Store the key now; only its hash is kept.", file=sys.stderr)


def cmd_create_key(args) -> None:
    db = _db()
    if db.customer(args.customer) is None:
        sys.exit(f"no customer {args.customer}")
    print(db.create_api_key(args.customer, args.label or ""))


def cmd_customers(args) -> None:
    for c in _db().customers():
        print(f"{c['id']}  {c['name']}")


def cmd_import_voice(args) -> None:
    """For voices whose consent was collected outside the API, e.g. a teacher's signed release form."""
    from .audio import ffmpeg_convert
    from .config import settings
    from .providers.worker import WorkerClient

    db = _db()
    if not args.shared and db.customer(args.customer) is None:
        sys.exit(f"no customer {args.customer}")
    raw = open(args.reference, "rb").read()
    wav = asyncio.run(ffmpeg_convert(raw, 48000))
    seconds = (len(wav) - 44) / (2 * 48000)
    v = db.create_voice(
        customer_id=None if args.shared else args.customer, name=args.name, speaker_name=args.speaker,
        reference_seconds=round(seconds, 2), consent_phrase=None, consent_code=None, consent_expires_at=None,
        status="active",
    )
    vdir = settings.voices_dir / v["id"]
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "reference.wav").write_bytes(wav)
    db.activate_voice(v["id"])
    db.add_consent(v["id"], "offline_attestation", True, {
        "attestation": args.attestation, "operator": args.operator, "recorded_at": time.time(),
    })

    async def push():
        w = WorkerClient(settings.worker_url, settings.worker_token)
        try:
            await w.put_voice(v["id"], wav)
        finally:
            await w.close()

    try:
        asyncio.run(push())
    except Exception as e:
        print(f"worker not reachable ({e}); the voice loads on first use", file=sys.stderr)
    print(json.dumps({"voice_id": v["id"], "seconds": round(seconds, 1), "shared": args.shared}, indent=2))


def cmd_replace_reference(args) -> None:
    """Swap in a better recording of the same person. Refused if it isn't the same speaker,
    because the voice's consent covers that person only."""
    from .audio import ffmpeg_convert
    from .config import settings
    from .providers.worker import WorkerClient

    db = _db()
    v = db.voice(args.voice)
    if v is None or v["status"] != "active":
        sys.exit(f"no active voice {args.voice}")
    vdir = settings.voices_dir / v["id"]
    old = (vdir / "reference.wav").read_bytes()
    trim = [*(["-ss", str(args.start)] if args.start is not None else []), *(["-to", str(args.end)] if args.end is not None else [])]
    raw = open(args.reference, "rb").read()
    if trim:
        import subprocess
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", args.reference, *trim, "-f", "wav", "-"],
                             capture_output=True, check=True).stdout
    new = asyncio.run(ffmpeg_convert(raw, 48000))
    seconds = round((len(new) - 44) / (2 * 48000), 2)
    if not settings.voice_min_seconds <= seconds <= settings.voice_max_seconds:
        sys.exit(f"reference must be {settings.voice_min_seconds:.0f}-{settings.voice_max_seconds:.0f} s (got {seconds} s)")

    async def run():
        w = WorkerClient(settings.worker_url, settings.worker_token)
        try:
            score = await w.speaker_similarity(old, new)
            if score < settings.consent_speaker_threshold and not args.force:
                return score, False
            backup = vdir / f"reference_prev_{int(time.time())}.wav"
            backup.write_bytes(old)
            (vdir / "reference.wav").write_bytes(new)
            db.set_reference_seconds(v["id"], seconds)
            db.add_consent(v["id"], "reference_replaced", True, {
                "operator": args.operator, "source": args.reference, "speaker_score_vs_previous": round(score, 4),
                "forced": bool(args.force and score < settings.consent_speaker_threshold), "previous": backup.name,
                "seconds": seconds,
            })
            await w.put_voice(v["id"], new)
            return score, True
        finally:
            await w.close()

    score, done = asyncio.run(run())
    if not done:
        sys.exit(f"refused: the new recording doesn't match the current speaker (similarity {score:.2f} < "
                 f"{settings.consent_speaker_threshold}). A different person needs their own consent.")
    print(json.dumps({"voice_id": v["id"], "seconds": seconds, "speaker_score_vs_previous": round(score, 3)}, indent=2))


def cmd_set_persona(args) -> None:
    db = _db()
    if db.voice(args.voice) is None:
        sys.exit(f"no voice {args.voice}")
    persona = open(args.persona_file, encoding="utf-8").read().strip() if args.persona_file else None
    db.set_persona(args.voice, persona, args.greeting)
    print(json.dumps({"voice_id": args.voice, "persona_chars": len(persona or ""), "greeting": args.greeting},
                     ensure_ascii=False, indent=2))


def cmd_consents(args) -> None:
    print(json.dumps(_db().consents(args.voice), indent=2, ensure_ascii=False))


def cmd_usage(args) -> None:
    now = time.time()
    db, since = _db(), now - args.days * 86400
    print(json.dumps({"api": db.request_totals(args.customer, since, now),
                      "agent": db.usage_summary(args.customer, since, now)}, indent=2))


def cmd_check(args) -> None:
    """Supabase reachable with this key? Every live voice's audio on this gateway? Usage flowing?"""
    from .accounts import SupabaseAccounts
    from .config import settings

    if not settings.supabase_url:
        sys.exit("VG_SUPABASE_URL is not set: this gateway uses its local database for accounts.")

    async def run():
        acc = SupabaseAccounts(settings.supabase_url, settings.supabase_secret_key)
        try:
            await acc._call("POST", "/rpc/verify_api_key", json={"p_key": "vg_check"})
            return await acc._call("GET", "/voices", params={"select": "id,name,status,user_id", "order": "created_at"})
        finally:
            await acc.close()

    try:
        voices = asyncio.run(run())
    except Exception as e:
        sys.exit(f"Supabase check failed: {e}")
    print(f"Supabase: ok ({settings.supabase_url})")
    missing = 0
    for v in voices:
        here = (settings.voices_dir / v["id"] / "reference.wav").exists()
        missing += v["status"] == "active" and not here
        print(f"  {v['id']}  {v['status']:<8}  {'shared' if v['user_id'] is None else 'account ' + v['user_id'][:8]}"
              f"  {'audio here' if here else 'NO AUDIO HERE'}  {v['name']}")
    print(f"Request log: {_db().request_counts()}")
    if missing:
        sys.exit(f"{missing} live voice(s) have no reference audio on this gateway; requests for them get 503.")


def cmd_talk(args) -> None:
    from .devclient import talk

    asyncio.run(talk(args.url, args.api_key, args.voice, args.files, args.out, args.instructions, not args.no_draft))


def main() -> None:
    p = argparse.ArgumentParser(prog="voicegw")
    sub = p.add_subparsers(required=True)

    sub.add_parser("serve", help="run the gateway").set_defaults(fn=cmd_serve)

    s = sub.add_parser("create-customer", help="create a customer and its first API key")
    s.add_argument("name")
    s.set_defaults(fn=cmd_create_customer)

    s = sub.add_parser("create-key", help="add an API key to a customer")
    s.add_argument("customer")
    s.add_argument("--label")
    s.set_defaults(fn=cmd_create_key)

    sub.add_parser("customers", help="list customers").set_defaults(fn=cmd_customers)

    s = sub.add_parser("import-voice", help="add a voice whose consent was collected offline")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--customer")
    g.add_argument("--shared", action="store_true", help="usable by every customer")
    s.add_argument("--name", required=True)
    s.add_argument("--speaker", required=True, help="the person whose voice this is")
    s.add_argument("--reference", required=True, help="audio file, 10-60 s of clean speech")
    s.add_argument("--attestation", required=True, help="where the signed consent lives, e.g. 'Release form, HR drive, signed 2026-10-20'")
    s.add_argument("--operator", required=True, help="who is importing it")
    s.set_defaults(fn=cmd_import_voice)

    s = sub.add_parser("replace-reference", help="swap in a better recording of the same speaker")
    s.add_argument("voice")
    s.add_argument("reference", help="audio or video file")
    s.add_argument("--start", type=float, help="trim: start second")
    s.add_argument("--end", type=float, help="trim: end second")
    s.add_argument("--operator", required=True)
    s.add_argument("--force", action="store_true", help="accept a low speaker match (still recorded)")
    s.set_defaults(fn=cmd_replace_reference)

    s = sub.add_parser("set-persona", help="default instructions and greeting for a voice")
    s.add_argument("voice")
    s.add_argument("--persona-file", help="text file with the persona instructions")
    s.add_argument("--greeting", help="what the voice says first when a session asks to be greeted")
    s.set_defaults(fn=cmd_set_persona)

    s = sub.add_parser("consents", help="show consent records for a voice")
    s.add_argument("voice")
    s.set_defaults(fn=cmd_consents)

    s = sub.add_parser("usage", help="usage totals for a customer")
    s.add_argument("customer")
    s.add_argument("--days", type=float, default=30)
    s.set_defaults(fn=cmd_usage)

    sub.add_parser("check", help="check the Supabase connection and that every live voice has its audio here"
                   ).set_defaults(fn=cmd_check)

    s = sub.add_parser("talk", help="dev client: send audio files as user turns, save the spoken replies")
    s.add_argument("files", nargs="+", help="one audio file per user turn")
    s.add_argument("--api-key", required=True)
    s.add_argument("--voice", required=True)
    s.add_argument("--url", default="ws://127.0.0.1:8000/v1/realtime")
    s.add_argument("--out", default="reply", help="prefix for reply wav files")
    s.add_argument("--instructions", default="")
    s.add_argument("--no-draft", action="store_true", help="turn off early (preemptive) replies, for comparison")
    s.set_defaults(fn=cmd_talk)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
