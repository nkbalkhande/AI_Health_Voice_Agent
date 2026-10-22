from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Literal


class SpeechClient(ABC):
    """Speech-to-text and text-to-speech for one voice provider."""

    @abstractmethod
    async def transcribe(self, audio: bytes, content_type: str) -> str:
        pass

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        pass

    async def synthesize_stream(self, text: str, *, voice: str | None = None):
        """Yield audio as it is produced. Default returns one finished clip.

        ``voice`` is a provider voice id; providers that cannot switch voices
        (or do not know the id) use their configured voice.
        """
        audio = await self.synthesize(text)
        if audio:
            yield audio

    async def synthesize_pcm_stream(
        self, text: str, *, sample_rate: int = 24000, voice: str | None = None
    ) -> AsyncIterator[bytes]:
        """Yield raw 16-bit little-endian mono PCM as it is produced.

        Live voice plays these bytes as they arrive, so the provider must
        stream; there is no sensible fallback from a finished MP3.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot stream PCM")
        yield b""  # pragma: no cover


@dataclass
class TranscriptEvent:
    """One message from a live transcription stream.

    ``partial``: interim words for the segment being spoken (may change).
    ``final``: a segment whose words will not change.
    ``speech_started``: the provider heard voice activity.
    ``utterance_end``: the speaker has finished; ``text`` is unused.
    ``speech_final`` on a final segment means it also ends the utterance.
    """

    kind: Literal["partial", "final", "speech_started", "utterance_end"]
    text: str = ""
    speech_final: bool = False


class LiveTranscription(ABC):
    """An open streaming speech-to-text connection."""

    @abstractmethod
    async def send_audio(self, pcm: bytes) -> None:
        pass

    @abstractmethod
    def events(self) -> AsyncIterator[TranscriptEvent]:
        pass

    @abstractmethod
    async def finish(self) -> None:
        """Flush buffered audio and close the stream."""


class LiveTranscriber(ABC):
    """Opens live transcription connections for raw PCM microphone audio."""

    @abstractmethod
    def connect(self, *, sample_rate: int) -> AbstractAsyncContextManager[LiveTranscription]:
        pass
