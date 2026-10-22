import httpx

from backend.core.exceptions import ProviderNotConfiguredError
from backend.core.logger import logger
from backend.domain.interfaces.speech import SpeechClient

log = logger.getChild("voice.elevenlabs")

_PCM_RATES = {16000, 22050, 24000, 44100}


class ElevenLabsSpeechClient(SpeechClient):
    """Streaming text-to-speech. Speech-to-text stays with Deepgram."""

    def __init__(
        self,
        api_key: str,
        *,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model: str = "eleven_flash_v2_5",
        timeout_seconds: float = 60,
    ) -> None:
        if not (api_key or "").strip():
            raise ProviderNotConfiguredError("ElevenLabs API key is not configured")
        self._api_key = api_key.strip()
        self._voice_id = voice_id
        self._model = model
        self._timeout = timeout_seconds

    async def transcribe(self, audio: bytes, content_type: str) -> str:
        raise ProviderNotConfiguredError(
            "ElevenLabs is configured for speech output only; use Deepgram for STT"
        )

    async def synthesize(self, text: str) -> bytes:
        parts = [chunk async for chunk in self.synthesize_stream(text)]
        return b"".join(parts)

    async def synthesize_stream(self, text: str, *, voice: str | None = None):
        async for chunk in self._stream(text, "mp3_44100_128", "audio/mpeg"):
            yield chunk

    async def synthesize_pcm_stream(
        self, text: str, *, sample_rate: int = 24000, voice: str | None = None
    ):
        if sample_rate not in _PCM_RATES:
            raise ValueError(f"ElevenLabs PCM supports {sorted(_PCM_RATES)} Hz")
        async for chunk in self._stream(text, f"pcm_{sample_rate}", "audio/pcm"):
            yield chunk

    async def _stream(self, text: str, output_format: str, accept: str):
        spoken = (text or "").strip()
        if not spoken:
            return
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST",
                f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}/stream",
                params={"output_format": output_format},
                headers={
                    "xi-api-key": self._api_key,
                    "Content-Type": "application/json",
                    "Accept": accept,
                },
                json={
                    "text": spoken[:2500],
                    "model_id": self._model,
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                },
            ) as response:
                if not response.is_success:
                    detail = (await response.aread()).decode("utf-8", "replace")[:300]
                    log.warning(
                        "ElevenLabs request failed status=%s detail=%s",
                        response.status_code,
                        detail,
                    )
                    response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
