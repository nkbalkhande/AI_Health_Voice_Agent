import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from backend.api.deps import get_speech_client, get_voice_use_case
from backend.api.schemas import (
    SpeakRequest,
    TranscriptResponse,
    VoiceOption,
    VoiceOptionsResponse,
    VoiceTurnResponse,
)
from backend.application.use_cases.voice import VoiceUseCase
from backend.core.config import settings
from backend.core.exceptions import ChatThreadNotFoundError, ProviderNotConfiguredError
from backend.domain.interfaces.speech import SpeechClient

router = APIRouter(prefix="/voice", tags=["voice"])


@router.post("/transcribe", response_model=TranscriptResponse)
async def transcribe(
    file: UploadFile = File(...),
    speech: SpeechClient = Depends(get_speech_client),
) -> TranscriptResponse:
    try:
        audio = await file.read()
        transcript = await speech.transcribe(
            audio,
            file.content_type or "application/octet-stream",
        )
    except ProviderNotConfiguredError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail="Speech transcription failed") from e
    return TranscriptResponse(transcript=transcript)


@router.get("/voices", response_model=VoiceOptionsResponse)
async def voices() -> VoiceOptionsResponse:
    """Voices the client may request. Empty when the provider has one voice."""
    provider = settings.voice_tts_provider.strip().lower()
    if provider != "deepgram":
        return VoiceOptionsResponse(provider=provider, default=None, voices=[])
    options = [
        VoiceOption(
            id=str(v["id"]),
            label=str(v.get("label") or v["id"]),
            description=str(v.get("description") or ""),
        )
        for v in settings.deepgram_tts_voices
        if v.get("id")
    ]
    return VoiceOptionsResponse(
        provider=provider, default=settings.deepgram_tts_model, voices=options
    )


@router.post("/speak")
async def speak(
    body: SpeakRequest,
    speech: SpeechClient = Depends(get_speech_client),
):
    from fastapi.responses import StreamingResponse

    # Pull the first chunk before answering so provider errors (quota, auth)
    # become an HTTP error instead of a crash mid-stream after a 200.
    stream = speech.synthesize_stream(body.text, voice=body.voice)
    try:
        first = await anext(stream)
    except StopAsyncIteration:
        raise HTTPException(status_code=422, detail="Nothing to speak") from None
    except ProviderNotConfiguredError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        detail = (
            "Speech provider quota or plan limit reached"
            if code in (401, 402, 429)
            else "Speech synthesis failed"
        )
        raise HTTPException(status_code=502, detail=detail) from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail="Speech synthesis failed") from e

    async def audio():
        yield first
        async for chunk in stream:
            yield chunk

    return StreamingResponse(audio(), media_type="audio/mpeg")


@router.post("/turn", response_model=VoiceTurnResponse)
async def voice_turn(
    file: UploadFile = File(...),
    thread_id: str | None = Form(default=None),
    use_case: VoiceUseCase = Depends(get_voice_use_case),
) -> VoiceTurnResponse:
    try:
        audio = await file.read()
        result = await use_case.turn(
            audio,
            file.content_type or "application/octet-stream",
            thread_id=thread_id or None,
        )
    except ProviderNotConfiguredError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ChatThreadNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail="Speech transcription failed") from e
    chat = result.chat
    return VoiceTurnResponse(
        thread_id=chat.thread_id,
        transcript=result.transcript,
        answer=chat.answer,
        citations=result.citations,
        audio_base64=result.audio_base64,
        speech_error=result.speech_error,
        summary=chat.summary,
        facts=chat.facts,
        message_count=chat.message_count,
        model=chat.model,
    )
