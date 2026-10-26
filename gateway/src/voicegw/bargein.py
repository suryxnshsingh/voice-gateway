"""Rules for telling a real interruption from a cough, a backchannel, or the assistant's own echo."""

import re

# Acknowledgements a listener makes while someone else talks. Said during a reply, they mean
# "go on", not "stop". "no", "wait" and "stop" are deliberately absent.
BACKCHANNELS = {
    "hmm", "hm", "hmmm", "mm", "mmm", "mhm", "mmhmm", "uh", "huh", "uhhuh", "um", "umm", "ah", "oh", "ooh",
    "ok", "okay", "yeah", "yes", "yep", "yup", "right", "sure", "cool", "nice", "got", "it", "i", "see",
    "haan", "han", "haa", "ha", "haanji", "hanji", "ji", "achha", "acha", "accha", "theek", "thik", "hai",
    "हाँ", "हां", "हम्म", "अच्छा", "ठीक", "है", "जी", "हाँजी", "ओके",
}

# Latin plus Devanagari minus the dandas (sentence marks). Python's \w skips the vowel signs (matras) that most
# Hindi words contain, which would split "हाँ" into pieces.
_WORD = re.compile(r"[0-9A-Za-z\u00c0-\u024f\u0900-\u0963\u0966-\u097f]+(?:'[A-Za-z]+)?")


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def real_words(text: str, ignore_backchannels: bool = True) -> list[str]:
    ws = words(text)
    return [w for w in ws if w not in BACKCHANNELS] if ignore_backchannels else ws


def is_backchannel(text: str) -> bool:
    ws = words(text)
    return bool(ws) and all(w in BACKCHANNELS for w in ws)


def looks_like_echo(heard: str, assistant_text: str, overlap: float = 0.8) -> bool:
    """The mic picked up the assistant itself (speakers, no echo cancellation): the "user" is
    saying almost exactly the words the assistant is saying right now.

    One or two words can't be told apart from a user repeating a term, so while every word heard
    is one the assistant just said, it counts as echo until there is more to go on.
    """
    hw = words(heard)
    if not hw:
        return False
    said = set(words(assistant_text))
    hits = sum(w in said for w in hw)
    if len(hw) < 3:
        return hits == len(hw)
    return hits / len(hw) >= overlap
