import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from urllib.parse import urlencode

import httpx

from backend.core.exceptions import ProviderNotConfiguredError
from backend.core.logger import logger
from backend.domain.interfaces.speech import (
    LiveTranscriber,
    LiveTranscription,
    SpeechClient,
    TranscriptEvent,
)

log = logger.getChild("voice.deepgram")


def parse_transcript(payload: dict) -> str:
    channels = (payload.get("results") or {}).get("channels") or []
    if not channels:
        return ""
    alternatives = channels[0].get("alternatives") or []
    if not alternatives:
        return ""
    return str(alternatives[0].get("transcript") or "").strip()


class DeepgramSpeechClient(SpeechClient):
    """Streaming speech-to-text. Speech output is ElevenLabs."""

    def __init__(
        self,
        api_key: str,
        *,
        stt_model: str = "nova-3",
        timeout_seconds: float = 60,
    ) -> None:
        if not (api_key or "").strip():
            raise ProviderNotConfiguredError("Deepgram API key is not configured")
        self._api_key = api_key.strip()
        self._stt_model = stt_model
        self._timeout = timeout_seconds

    async def transcribe(self, audio: bytes, content_type: str) -> str:
        if not audio:
            return ""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                "https://api.deepgram.com/v1/listen",
                params={
                    "model": self._stt_model,
                    "smart_format": "true",
                    "punctuate": "true",
                },
                headers={
                    "Authorization": f"Token {self._api_key}",
                    "Content-Type": content_type or "application/octet-stream",
                },
                content=audio,
            )
            self._raise_for_status(response)
            return parse_transcript(response.json())

    async def synthesize(self, text: str) -> bytes:
        raise ProviderNotConfiguredError(
            "Deepgram is configured for speech input only; use ElevenLabs for TTS"
        )

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return
        log.warning(
            "Deepgram request failed status=%s detail=%s",
            response.status_code,
            response.text[:300],
        )
        response.raise_for_status()


class DeepgramTTSClient(SpeechClient):
    """Streaming text-to-speech with Deepgram Aura."""

    # Aura rejects requests longer than this.
    _MAX_CHARS = 2000

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "aura-2-thalia-en",
        voices: tuple[str, ...] = (),
        timeout_seconds: float = 60,
    ) -> None:
        if not (api_key or "").strip():
            raise ProviderNotConfiguredError("Deepgram API key is not configured")
        self._api_key = api_key.strip()
        self._model = model
        self._voices = frozenset(voices) | {model}
        self._timeout = timeout_seconds
        # One pooled client: a live answer is several requests in a row, and
        # a fresh TLS handshake per sentence adds audible gaps.
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def transcribe(self, audio: bytes, content_type: str) -> str:
        raise ProviderNotConfiguredError("DeepgramTTSClient handles speech output only")

    async def synthesize(self, text: str) -> bytes:
        parts = [chunk async for chunk in self.synthesize_stream(text)]
        return b"".join(parts)

    async def synthesize_stream(self, text: str, *, voice: str | None = None):
        async for chunk in self._stream(text, {"encoding": "mp3"}, voice):
            yield chunk

    async def synthesize_pcm_stream(
        self, text: str, *, sample_rate: int = 24000, voice: str | None = None
    ):
        params = {"encoding": "linear16", "sample_rate": sample_rate, "container": "none"}
        async for chunk in self._stream(text, params, voice):
            yield chunk

    def model_for(self, voice: str | None) -> str:
        return voice if voice in self._voices else self._model

    async def _stream(self, text: str, audio_params: dict, voice: str | None = None):
        spoken = (text or "").strip()
        if not spoken:
            return
        async with self._http().stream(
            "POST",
            "https://api.deepgram.com/v1/speak",
            params={"model": self.model_for(voice), **audio_params},
            headers={
                "Authorization": f"Token {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"text": _truncate(spoken, self._MAX_CHARS)},
        ) as response:
            if not response.is_success:
                detail = (await response.aread()).decode("utf-8", "replace")[:300]
                log.warning(
                    "Deepgram TTS request failed status=%s detail=%s",
                    response.status_code,
                    detail,
                )
                response.raise_for_status()
            async for chunk in response.aiter_bytes():
                if chunk:
                    yield chunk


class DeepgramLiveTranscriber(LiveTranscriber):
    """Streaming speech-to-text over Deepgram's live websocket.

    Deepgram decides when the speaker has finished (endpointing plus
    utterance_end), so the client streams the microphone continuously.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "nova-3",
        endpointing_ms: int = 400,
        utterance_end_ms: int = 1000,
    ) -> None:
        if not (api_key or "").strip():
            raise ProviderNotConfiguredError("Deepgram API key is not configured")
        self._api_key = api_key.strip()
        self._model = model
        self._endpointing_ms = endpointing_ms
        self._utterance_end_ms = utterance_end_ms

    @asynccontextmanager
    async def connect(self, *, sample_rate: int) -> AsyncIterator[LiveTranscription]:
        from websockets.asyncio.client import connect

        params = urlencode(
            {
                "model": self._model,
                "encoding": "linear16",
                "sample_rate": sample_rate,
                "channels": 1,
                "interim_results": "true",
                "smart_format": "true",
                "punctuate": "true",
                "vad_events": "true",
                "endpointing": self._endpointing_ms,
                "utterance_end_ms": self._utterance_end_ms,
            }
        )
        async with connect(
            f"wss://api.deepgram.com/v1/listen?{params}",
            additional_headers={"Authorization": f"Token {self._api_key}"},
            max_size=None,
            open_timeout=10,
        ) as socket:
            stream = _DeepgramLiveStream(socket)
            keepalive = asyncio.create_task(stream.keepalive())
            try:
                yield stream
            finally:
                keepalive.cancel()
                await stream.finish()


class _DeepgramLiveStream(LiveTranscription):
    def __init__(self, socket) -> None:
        self._socket = socket
        self._closed = False

    async def send_audio(self, pcm: bytes) -> None:
        if not self._closed and pcm:
            await self._socket.send(pcm)

    async def keepalive(self) -> None:
        # Deepgram closes a stream after ~10 s without audio or keepalive.
        while not self._closed:
            await asyncio.sleep(5)
            with suppress(Exception):
                await self._socket.send(json.dumps({"type": "KeepAlive"}))

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        from websockets.exceptions import ConnectionClosed

        try:
            async for raw in self._socket:
                if isinstance(raw, bytes):
                    continue
                event = parse_live_message(json.loads(raw))
                if event is not None:
                    yield event
        except ConnectionClosed:
            return

    async def finish(self) -> None:
        if self._closed:
            return
        self._closed = True
        with suppress(Exception):
            await self._socket.send(json.dumps({"type": "CloseStream"}))
        with suppress(Exception):
            await self._socket.close()


def parse_live_message(message: dict) -> TranscriptEvent | None:
    kind = message.get("type")
    if kind == "SpeechStarted":
        return TranscriptEvent(kind="speech_started")
    if kind == "UtteranceEnd":
        return TranscriptEvent(kind="utterance_end")
    if kind != "Results":
        return None
    alternatives = (message.get("channel") or {}).get("alternatives") or []
    text = str(alternatives[0].get("transcript") or "").strip() if alternatives else ""
    if message.get("is_final"):
        return TranscriptEvent(
            kind="final", text=text, speech_final=bool(message.get("speech_final"))
        )
    if not text:
        return None
    return TranscriptEvent(kind="partial", text=text)


def _truncate(text: str, limit: int) -> str:
    """Cut at the last sentence end before the limit so speech stops cleanly."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    cut = max(head.rfind(". "), head.rfind("! "), head.rfind("? "), head.rfind("\n"))
    return head[: cut + 1] if cut > limit // 2 else head
