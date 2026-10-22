import re

# A sentence ends at . ! ? (optionally followed by a closing quote or bracket)
# when whitespace follows, or at a line break. "2.5 mg" and "U.S." mid-word
# do not split because no whitespace follows the dot.
_BOUNDARY = re.compile(r"[.!?][\"')\]]*(?=\s)|\n+")


class SentenceChunker:
    """Turn a token stream into speakable pieces as early as possible.

    Text-to-speech sounds natural per sentence, and the first sentence is
    what the listener waits for, so pieces are released the moment a
    boundary arrives. Runaway sentences are cut at a comma or space.
    """

    def __init__(
        self, *, min_chars: int = 12, max_chars: int = 220, first_clause_chars: int = 40
    ) -> None:
        self._buffer = ""
        self._min = min_chars
        self._max = max_chars
        # The first piece may end at a comma once it is this long: speech
        # starts sooner and a clause-sized opening still sounds natural.
        self._first_clause = first_clause_chars
        self._released = False

    def push(self, text: str) -> list[str]:
        self._buffer += text
        pieces: list[str] = []
        while True:
            piece = self._take_sentence() or self._take_first_clause() or self._take_overflow()
            if not piece:
                return pieces
            self._released = True
            pieces.append(piece)

    def _take_first_clause(self) -> str | None:
        if self._released or not self._first_clause:
            return None
        cut = self._buffer.find(", ", self._first_clause - 1)
        if cut < 0:
            return None
        piece = self._buffer[: cut + 1].strip()
        self._buffer = self._buffer[cut + 2 :]
        return piece

    def flush(self) -> str | None:
        rest = self._buffer.strip()
        self._buffer = ""
        return rest or None

    def _take_sentence(self) -> str | None:
        for match in _BOUNDARY.finditer(self._buffer):
            end = match.end()
            if len(self._buffer[:end].strip()) >= self._min:
                piece = self._buffer[:end].strip()
                self._buffer = self._buffer[end:].lstrip()
                return piece
        return None

    def _take_overflow(self) -> str | None:
        if len(self._buffer) <= self._max:
            return None
        head = self._buffer[: self._max]
        cut = head.rfind(", ")
        cut = cut + 1 if cut > self._min else head.rfind(" ")
        if cut <= self._min:
            cut = self._max
        piece = self._buffer[:cut].strip()
        self._buffer = self._buffer[cut:].lstrip()
        return piece
