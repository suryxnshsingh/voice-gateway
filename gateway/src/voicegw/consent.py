"""Voice-cloning consent: the speaker reads a statement with a random code, live, in their own voice.

A clone goes active only when (1) every code word shows up in the transcript of the consent
recording, so it was recorded for this request and not replayed, and (2) the consent recording
and the reference audio are the same speaker.
"""

import difflib
import re
import secrets

# Short, distinct, easy to say in an Indian-English accent, and hard for STT to confuse.
CODE_WORDS = [
    "apple", "banana", "candle", "dolphin", "elephant", "falcon", "garden", "harbor", "island", "jungle",
    "kettle", "lemon", "mango", "needle", "orange", "pepper", "rabbit", "river", "silver", "tiger",
    "umbrella", "violin", "window", "yellow", "zebra", "anchor", "basket", "camera", "desert", "engine",
    "forest", "guitar", "hammer", "jacket", "ladder", "magnet", "pencil", "pillow", "planet", "rocket",
    "saddle", "tomato", "tunnel", "valley", "wallet", "button", "carpet", "cherry", "copper", "donkey",
    "feather", "ginger", "honey", "kitten", "marble", "monkey", "parrot", "pocket", "puzzle", "spider",
]

TEMPLATES = {
    "en": (
        "I, {speaker}, am the owner of this voice. I consent to {customer} creating and using "
        "a synthetic copy of my voice. My verification code is {code}."
    ),
    # Worded so no verb agrees with the speaker's gender.
    "hi": (
        "Mera naam {speaker} hai, aur yeh awaaz meri hai. Meri anumati hai ki {customer} meri awaaz ki "
        "synthetic copy banaye aur use kare. Mera verification code hai {code}."
    ),
}


def new_challenge(speaker: str, customer: str, language: str = "en", n_words: int = 3) -> tuple[str, str]:
    """Returns (phrase to read, code words space-separated)."""
    words = []
    while len(words) < n_words:
        w = secrets.choice(CODE_WORDS)
        if w not in words:
            words.append(w)
    code = " ".join(words)
    template = TEMPLATES.get(language, TEMPLATES["en"])
    return template.format(speaker=speaker, customer=customer, code=", ".join(words)), code


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def code_matches(code: str, transcript: str, cutoff: float = 0.8) -> tuple[bool, list[str]]:
    """Every code word must appear (allowing small STT slips like "rocket" -> "rockets")."""
    heard = _words(transcript)
    missing = [w for w in code.split() if not difflib.get_close_matches(w, heard, n=1, cutoff=cutoff)]
    return not missing, missing
