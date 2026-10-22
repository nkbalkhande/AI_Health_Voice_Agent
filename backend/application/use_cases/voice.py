import base64
from dataclasses import dataclass, field

from backend.application.use_cases.chat import ChatResult, ChatUseCase
from backend.core.logger import logger
from backend.domain.interfaces.speech import SpeechClient

log = logger.getChild("voice.turn")


@dataclass
class VoiceTurnResult:
    transcript: str
    chat: ChatResult
    audio_base64: str = ""
    speech_error: str | None = None
    citations: list[dict] = field(default_factory=list)


class VoiceUseCase:
    """Spoken input uses the same session, retrieval, and answer as typed chat."""

    def __init__(self, *, speech: SpeechClient, chat: ChatUseCase) -> None:
        self._speech = speech
        self._chat = chat

    async def transcribe(self, audio: bytes, content_type: str) -> str:
        return await self._speech.transcribe(audio, content_type)

    async def synthesize(self, text: str) -> bytes:
        return await self._speech.synthesize(text)

    async def turn(
        self,
        audio: bytes,
        content_type: str,
        *,
        thread_id: str | None = None,
        filters: dict | None = None,
    ) -> VoiceTurnResult:
        transcript = (await self.transcribe(audio, content_type)).strip()
        if not transcript:
            raise ValueError("No speech detected")
        chat = await self._chat.execute(
            transcript,
            thread_id=thread_id,
            filters=filters,
        )
        audio_base64 = ""
        speech_error = None
        try:
            spoken = await self.synthesize(chat.answer)
            if spoken:
                audio_base64 = base64.b64encode(spoken).decode("ascii")
        except Exception:
            log.warning("Text-to-speech failed after the answer was saved", exc_info=True)
            speech_error = "The answer was saved, but speech audio could not be generated."
        return VoiceTurnResult(
            transcript=transcript,
            chat=chat,
            audio_base64=audio_base64,
            speech_error=speech_error,
            citations=chat.citations,
        )
