from voicegw.audio import PcmRing
from voicegw.auth import mint_session_token, read_session_token
from voicegw.consent import code_matches, new_challenge
from voicegw.text import SentenceSplitter, speakable


def feed_all(text, step=3):
    sp = SentenceSplitter()
    out = []
    for i in range(0, len(text), step):
        out += sp.feed(text[i : i + step])
    return out + sp.flush()


def test_short_opening_stays_whole_then_sentences_merge():
    chunks = feed_all(
        "Photosynthesis, in simple words, is how plants make food. It uses sunlight, water and carbon dioxide. "
        "The oxygen we breathe is a by-product. Isn't that neat?"
    )
    assert chunks[0] == "Photosynthesis, in simple words, is how plants make food."
    assert chunks[1] == "It uses sunlight, water and carbon dioxide. The oxygen we breathe is a by-product."
    assert all(len(c) >= 40 for c in chunks[:-1])


def test_long_opening_cuts_at_a_clause():
    chunks = feed_all(
        "When light hits the chlorophyll in the thylakoid membranes of the chloroplast, electrons get excited "
        "and start a chain of reactions that ends in ATP. Next we look at the Calvin cycle."
    )
    assert chunks[0] == "When light hits the chlorophyll in the thylakoid membranes of the chloroplast,"


def test_no_split_inside_numbers_or_abbreviations():
    chunks = feed_all("The value of pi is about 3.14 and e.g. radius 2 gives area 12.56 square units. Done here now.")
    assert chunks[0].startswith("The value of pi is about 3.14 and e.g. radius 2 gives area 12.56")


def test_hindi_danda_ends_a_sentence():
    chunks = feed_all("प्रकाश संश्लेषण पौधों में भोजन बनाने की प्रक्रिया है। इसमें सूर्य का प्रकाश लगता है। और पानी भी।")
    assert chunks[0].endswith("है।")


def test_speakable_strips_markdown():
    assert speakable("**Step 1:** use `x`\n- bullet one 🎉") == "Step 1: use x bullet one"


def test_code_matches_tolerates_small_slips():
    ok, missing = code_matches("rocket mango violin", "my verification code is rockets, mango, violins.")
    assert ok and not missing
    ok, missing = code_matches("rocket mango violin", "my code is rocket mango")
    assert not ok and missing == ["violin"]


def test_challenge_has_three_distinct_words_in_phrase():
    phrase, code = new_challenge("Asha Verma", "Arivihan", "hi")
    words = code.split()
    assert len(set(words)) == 3 and all(w in phrase for w in words) and "Asha Verma" in phrase


def test_pcm_ring_slices_by_stream_time():
    ring = PcmRing(seconds=1)
    ms = 32  # bytes per ms at 16 kHz PCM16
    for i in range(30):  # 1.5 s in 50 ms frames, each frame filled with its index
        ring.append(bytes([i]) * 50 * ms)
    assert ring.end_ms == 1500
    assert ring.slice(0, 400) == b""  # aged out
    assert ring.slice(1000, 1050) == bytes([20]) * 50 * ms


def test_session_token_roundtrip_and_tamper():
    token, _ = mint_session_token("cust_x")
    assert read_session_token(token) == "cust_x"
    assert read_session_token(token[:-2] + "xx") is None
    expired, _ = mint_session_token("cust_x", ttl_seconds=-5)
    assert read_session_token(expired) is None


from voicegw.bargein import is_backchannel, looks_like_echo, real_words  # noqa: E402
from voicegw.frontend import AudioFrontend, VadParams  # noqa: E402


def test_backchannels_and_real_words():
    assert is_backchannel("Haan, okay.") and is_backchannel("हाँ जी") and is_backchannel("mm-hmm")
    assert not is_backchannel("wait") and not is_backchannel("okay but why")
    assert real_words("okay but why") == ["but", "why"]


def test_echo_detection_needs_overlap_and_length():
    said = "Plants use sunlight to make sugar from water and carbon dioxide."
    assert looks_like_echo("use sunlight to make sugar", said)
    assert looks_like_echo("sunlight", said)  # too short to tell: held back as possible echo
    assert not looks_like_echo("wait", said)
    assert not looks_like_echo("why do plants need water at night", said)


def test_vad_ignores_silence_and_quiet_noise():
    import numpy as np

    fe = AudioFrontend(VadParams())
    rng = np.random.default_rng(1)
    noise = (rng.normal(0, 0.003, 16000 * 2) * 32767).astype("<i2").tobytes()
    events = []
    for i in range(0, len(noise), 1280):
        events += fe.process(noise[i : i + 1280]).events
    fe.close()
    assert events == []


def test_vad_raises_pause_before_stop_and_resume_between():
    import numpy as np

    from voicegw.frontend import float_to_pcm16

    # 1.5 s of the synthetic test voice (it has a short internal pause), then 1 s of silence.
    speech = np.fromfile("/tmp/vg-scratch/audio/ref16.pcm", dtype="<i2")[: 16000 * 3 // 2].astype(np.float32) / 32768
    audio = np.concatenate([speech, np.zeros(16000, dtype=np.float32)])
    fe = AudioFrontend(VadParams(), noise_suppression=False)
    pcm = float_to_pcm16(audio)
    kinds = []
    for i in range(0, len(pcm), 1280):
        kinds += [e.kind for e in fe.process(pcm[i : i + 1280]).events]
    assert kinds[0] == "start" and kinds[-2:] == ["pause", "stop"]
    # every pause in the middle is followed by a resume
    middle = kinds[1:-2]
    assert all(b == "resume" for a, b in zip(middle, middle[1:]) if a == "pause")
