import asyncio
import base64
import json
from contextlib import suppress

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from backend.api.deps import (
    get_chat_use_case,
    get_live_transcriber,
    get_speech_client,
    get_voice_use_case,
)
from backend.application.use_cases.chat import ChatUseCase
from backend.application.use_cases.voice import VoiceUseCase
from backend.application.voice.live_session import LiveVoiceSession
from backend.core.config import settings
from backend.core.exceptions import ChatThreadNotFoundError, ProviderNotConfiguredError
from backend.core.logger import logger

log = logger.getChild("api.ws")
router = APIRouter(tags=["websocket"])


@router.websocket("/ws/chat")
async def chat_websocket(
    websocket: WebSocket,
    chat: ChatUseCase = Depends(get_chat_use_case),
    voice: VoiceUseCase = Depends(get_voice_use_case),
) -> None:
    """One socket for a whole conversation.

    Client sends:
      {"type": "message", "message": "...", "thread_id": "..." | null}
      {"type": "voice", "audio_base64": "...", "content_type": "audio/webm",
       "thread_id": "..." | null}

    Server streams status/token events, then a done event. Voice turns send
    one done event that also carries the transcript and spoken audio.
    """
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            kind = payload.get("type", "message")
            thread_id = payload.get("thread_id") or None
            try:
                if kind == "voice":
                    audio = base64.b64decode(payload.get("audio_base64") or "")
                    result = await voice.turn(
                        audio,
                        payload.get("content_type") or "application/octet-stream",
                        thread_id=thread_id,
                    )
                    await websocket.send_json(
                        {
                            "type": "done",
                            "thread_id": result.chat.thread_id,
                            "transcript": result.transcript,
                            "answer": result.chat.answer,
                            "citations": result.citations,
                            "audio_base64": result.audio_base64,
                            "speech_error": result.speech_error,
                            "summary": result.chat.summary,
                            "facts": result.chat.facts,
                        }
                    )
                    continue

                message = (payload.get("message") or "").strip()
                if not message:
                    await websocket.send_json(
                        {"type": "error", "detail": "message is required"}
                    )
                    continue
                async for event in chat.stream(message, thread_id=thread_id):
                    await websocket.send_json(event)
            except ChatThreadNotFoundError as exc:
                await websocket.send_json({"type": "error", "detail": str(exc)})
            except ValueError as exc:
                await websocket.send_json({"type": "error", "detail": str(exc)})
            except Exception:
                log.warning("WebSocket turn failed", exc_info=True)
                await websocket.send_json(
                    {"type": "error", "detail": "The turn failed. Please retry."}
                )
    except WebSocketDisconnect:
        return


@router.websocket("/ws/voice")
async def live_voice_websocket(
    websocket: WebSocket,
    chat: ChatUseCase = Depends(get_chat_use_case),
) -> None:
    """Hands-free voice conversation.

    Client sends first:
      {"type": "start", "sample_rate": 48000, "thread_id": "..." | null,
       "voice": "aura-2-..." | null}
    then raw 16-bit mono PCM microphone audio as binary frames, and may send
      {"type": "interrupt"}                        stop the agent now
      {"type": "playback_end", "turn_id": "..."}   all audio of a turn played
      {"type": "stop"}                             end the session

    Server sends: ready, transcript (live captions), turn_start, status,
    token, done, audio (base64 PCM), audio_end, metrics, interrupt, error.
    """
    await websocket.accept()
    send_lock = asyncio.Lock()

    async def send(payload: dict) -> None:
        async with send_lock:
            with suppress(Exception):
                await websocket.send_json(payload)

    try:
        start = await websocket.receive_json()
    except (WebSocketDisconnect, ValueError):
        return
    sample_rate = min(max(int(start.get("sample_rate") or 16000), 8000), 48000)

    try:
        speech = get_speech_client()
        transcriber = get_live_transcriber()
    except ProviderNotConfiguredError as exc:
        await send({"type": "error", "detail": str(exc), "fatal": True})
        await websocket.close()
        return

    session = LiveVoiceSession(
        chat=chat,
        speech=speech,
        send=send,
        thread_id=start.get("thread_id") or None,
        tts_sample_rate=settings.voice_tts_sample_rate,
        voice=str(start.get("voice") or "")[:64] or None,
    )
    closing = False
    warm_up = asyncio.create_task(chat.warm_up())
    warm_up.add_done_callback(lambda t: t.cancelled() or t.exception())
    try:
        async with transcriber.connect(sample_rate=sample_rate) as live:

            async def pump_transcripts() -> None:
                async for event in live.events():
                    await session.on_transcript(event)
                if not closing:
                    await send(
                        {"type": "error", "detail": "Live transcription ended", "fatal": True}
                    )

            reader = asyncio.create_task(pump_transcripts())
            await send({"type": "ready", "thread_id": session.thread_id})
            try:
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        break
                    if message.get("bytes"):
                        await live.send_audio(message["bytes"])
                        continue
                    data = json.loads(message.get("text") or "{}")
                    kind = data.get("type")
                    if kind == "interrupt":
                        await session.interrupt()
                    elif kind == "playback_end":
                        await session.playback_finished(str(data.get("turn_id")))
                    elif kind == "stop":
                        break
            finally:
                closing = True
                await session.close()
                reader.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await reader
    except WebSocketDisconnect:
        pass
    except Exception:
        log.warning("Live voice session failed", exc_info=True)
        await send(
            {"type": "error", "detail": "Could not start live transcription", "fatal": True}
        )
    with suppress(Exception):
        await websocket.close()
