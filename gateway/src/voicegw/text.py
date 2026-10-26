"""Turning streamed LLM text into speakable chunks."""

import re

# Sentence ends: . ! ? and the Devanagari danda, then whitespace (so "3.14" never splits).
_SENTENCE_END = re.compile(r"[.!?।]+[\"')\]]*\s")
# A period after these is an abbreviation, not a sentence end.
_ABBREV = re.compile(r"(?:\b(?:e\.g|i\.e|etc|vs|approx|fig|dr|mr|mrs|ms|prof|st|sr|jr|ca)|\b[A-Za-z])\.$", re.IGNORECASE)
_CLAUSE_END = re.compile(r"[,;:—]\s")
_MARKDOWN = re.compile(r"[*_`#>|~]+")
_BULLET = re.compile(r"^\s*(?:[-•]|\d+[.)])\s+", re.MULTILINE)
_EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿️]")


def speakable(text: str) -> str:
    text = _BULLET.sub("", text)
    text = _MARKDOWN.sub("", text)
    text = _EMOJI.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class SentenceSplitter:
    """Feed LLM deltas, get back chunks ready for TTS.

    The first chunk may end at a clause break so audio starts sooner; later chunks are whole
    sentences, merged up to min_chars so the voice doesn't restart its prosody every few words.
    """

    def __init__(self, first_min_chars: int = 40, min_chars: int = 80, max_chars: int = 300):
        self._buf = ""
        self._first = True
        self._first_min = first_min_chars
        self._min = min_chars
        self._max = max_chars

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        out = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            chunk, self._buf = self._buf[:cut], self._buf[cut:]
            chunk = speakable(chunk)
            if chunk:
                out.append(chunk)
                self._first = False
        return out

    def flush(self) -> list[str]:
        chunk, self._buf = speakable(self._buf), ""
        return [chunk] if chunk else []

    def _find_cut(self) -> int | None:
        floor = self._first_min if self._first else self._min
        if self._buf.find("\n\n") >= floor // 2:
            return self._buf.find("\n\n") + 2
        for m in _SENTENCE_END.finditer(self._buf):
            if m.end() >= floor and not _ABBREV.search(self._buf[: m.start() + 1]):
                return m.end()
        if self._first and len(self._buf) >= 2 * self._first_min:
            # A long opening sentence: cut at a clause rather than keep the listener waiting. Short
            # openings stay whole; every cut restarts the voice's intonation.
            for m in _CLAUSE_END.finditer(self._buf):
                if m.end() >= self._first_min:
                    return m.end()
        if len(self._buf) > self._max:
            space = self._buf.rfind(" ", 0, self._max)
            return space + 1 if space > 0 else self._max
        return None
