import re

from backend.domain.interfaces.speech import SpeechClient

_CITATION = re.compile(r"\s*\[[^\[\]]{1,200}\]")
_MARKDOWN = re.compile(r"[*_#`]+")


def spoken_text(text: str) -> str:
    """Citations and markdown are for the screen; drop them before speaking."""
    cleaned = _MARKDOWN.sub("", _CITATION.sub("", text or ""))
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


class CompositeSpeechClient(SpeechClient):
    """Deepgram listens, ElevenLabs speaks. One interface for the API layer."""

    def __init__(self, *, stt: SpeechClient, tts: SpeechClient) -> None:
        self._stt = stt
        self._tts = tts

    async def transcribe(self, audio: bytes, content_type: str) -> str:
        return await self._stt.transcribe(audio, content_type)

    async def synthesize(self, text: str) -> bytes:
        return await self._tts.synthesize(spoken_text(text))

    async def synthesize_stream(self, text: str, *, voice: str | None = None):
        async for chunk in self._tts.synthesize_stream(spoken_text(text), voice=voice):
            yield chunk

    async def synthesize_pcm_stream(
        self, text: str, *, sample_rate: int = 24000, voice: str | None = None
    ):
        async for chunk in self._tts.synthesize_pcm_stream(
            spoken_text(text), sample_rate=sample_rate, voice=voice
        ):
            yield chunk
