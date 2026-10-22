"""Hands-free voice conversation over one socket.

    microphone ─> live STT ─> utterance ─> chat.stream (RAG + web) ─┐
                                                                    │ tokens
    speaker <─ PCM <─ TTS per sentence <─ SentenceChunker <─────────┘

The listener hears the first sentence while the rest is still being
written. Speaking over the agent cancels the turn (barge-in).
"""

import asyncio
import base64
import re
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from backend.application.use_cases.chat import ChatUseCase
from backend.application.voice.sentences import SentenceChunker
from backend.core.exceptions import ChatThreadNotFoundError
from backend.core.logger import logger
from backend.domain.interfaces.speech import SpeechClient, TranscriptEvent

log = logger.getChild("voice.live")

Send = Callable[[dict[str, Any]], Awaitable[None]]

_WORD = re.compile(r"[a-z0-9']+")
# Words needed before speech counts as the user cutting in. One word is
# too often a cough, "uh", or a stray echo.
_BARGE_IN_WORDS = 2
# Share of heard words that must appear in the agent's own answer for the
# transcript to be treated as the speaker leaking into the microphone.
_ECHO_OVERLAP = 0.7
# Echo can still be transcribed shortly after playback ends.
_ECHO_TAIL_SECONDS = 1.5


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass
class _Turn:
    id: str
    text: str
    started: float
    task: asyncio.Task | None = None
    answer: str = ""
    audio_started: bool = False
    metrics: dict[str, float] = field(default_factory=dict)


class LiveVoiceSession:
    def __init__(
        self,
        *,
        chat: ChatUseCase,
        speech: SpeechClient,
        send: Send,
        thread_id: str | None = None,
        tts_sample_rate: int = 24000,
        voice: str | None = None,
    ) -> None:
        self._chat = chat
        self._speech = speech
        self._send = send
        self.thread_id = thread_id
        self._rate = tts_sample_rate
        self._voice = voice
        self._segments: list[str] = []
        self._turn: _Turn | None = None
        self._speaking: str | None = None
        self._echo_text = ""
        self._echo_until = 0.0
        self._counter = 0
        self._lock = asyncio.Lock()

    # ---------- inputs ----------

    async def on_transcript(self, event: TranscriptEvent) -> None:
        async with self._lock:
            if event.kind == "partial":
                await self._on_words(event.text, final=False)
            elif event.kind == "final":
                if event.text:
                    await self._on_words(event.text, final=True)
                if event.speech_final:
                    await self._end_utterance()
            elif event.kind == "utterance_end":
                await self._end_utterance()

    async def interrupt(self) -> None:
        """The user tapped to stop the agent."""
        async with self._lock:
            await self._cancel_turn(notify=True)

    async def playback_finished(self, turn_id: str) -> None:
        """The client has played every audio chunk of this turn."""
        async with self._lock:
            if self._speaking == turn_id:
                self._speaking = None
                self._echo_until = time.monotonic() + _ECHO_TAIL_SECONDS

    async def close(self) -> None:
        async with self._lock:
            await self._cancel_turn(notify=False)

    # ---------- listening ----------

    def _agent_busy(self) -> bool:
        return self._turn is not None or self._speaking is not None

    def _is_echo(self, text: str) -> bool:
        if not (self._agent_busy() or time.monotonic() < self._echo_until):
            return False
        heard = _words(text)
        if not heard:
            return True
        spoken = set(_words(self._echo_text))
        overlap = sum(word in spoken for word in heard) / len(heard)
        return overlap >= _ECHO_OVERLAP

    async def _on_words(self, text: str, *, final: bool) -> None:
        if self._is_echo(text):
            return
        if self._agent_busy() and len(_words(text)) >= _BARGE_IN_WORDS:
            await self._barge_in()
        if final:
            self._segments.append(text)
        caption = " ".join([*self._segments, *([] if final else [text])])
        await self._send({"type": "transcript", "text": caption, "final": final})

    async def _barge_in(self) -> None:
        turn = self._turn
        # Nothing was said yet, so the user was only pausing mid-thought:
        # fold the cancelled question into what they are saying now.
        merged = turn is not None and not turn.audio_started
        await self._cancel_turn(notify=True, merged=merged)
        if merged:
            self._segments.insert(0, turn.text)

    async def _end_utterance(self) -> None:
        text = " ".join(self._segments).strip()
        self._segments = []
        if not text:
            return
        await self._cancel_turn(notify=True)
        self._counter += 1
        turn = _Turn(id=str(self._counter), text=text, started=time.perf_counter())
        self._turn = turn
        self._echo_text = ""
        turn.task = asyncio.create_task(self._run_turn(turn))

    async def _cancel_turn(self, *, notify: bool, merged: bool = False) -> None:
        turn, speaking = self._turn, self._speaking
        self._turn = None
        self._speaking = None
        running = turn.task if turn and turn.task and not turn.task.done() else None
        # Cancel before any await so the turn cannot send more audio meanwhile.
        if running:
            running.cancel()
        target = turn.id if turn else speaking
        if notify and target:
            await self._send({"type": "interrupt", "turn_id": target, "merged": merged})
        if running:
            with suppress(asyncio.CancelledError, Exception):
                await running

    # ---------- answering ----------

    def _elapsed_ms(self, turn: _Turn) -> float:
        return round((time.perf_counter() - turn.started) * 1000)

    async def _run_turn(self, turn: _Turn) -> None:
        await self._send({"type": "turn_start", "turn_id": turn.id, "text": turn.text})
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        speaker = asyncio.create_task(self._speak(turn, sentences))
        chunker = SentenceChunker()
        stream = self._chat.stream(turn.text, thread_id=self.thread_id, voice=True)
        try:
            async for event in stream:
                kind = event.get("type")
                if kind == "status":
                    self.thread_id = event.get("thread_id") or self.thread_id
                    turn.metrics["retrieval_ms"] = self._elapsed_ms(turn)
                    await self._send({**event, "turn_id": turn.id})
                elif kind == "token":
                    text = event.get("text") or ""
                    if "first_token_ms" not in turn.metrics:
                        turn.metrics["first_token_ms"] = self._elapsed_ms(turn)
                    turn.answer += text
                    self._echo_text = turn.answer
                    await self._send({"type": "token", "turn_id": turn.id, "text": text})
                    for sentence in chunker.push(text):
                        sentences.put_nowait(sentence)
                elif kind == "done":
                    tail = chunker.flush()
                    if tail:
                        sentences.put_nowait(tail)
                    sentences.put_nowait(None)
                    await self._send(
                        {
                            "type": "done",
                            "turn_id": turn.id,
                            "thread_id": event.get("thread_id"),
                            "answer": event.get("answer"),
                            "citations": event.get("citations") or [],
                            "summary": event.get("summary"),
                            "facts": event.get("facts") or [],
                        }
                    )
            await speaker
            turn.metrics["total_ms"] = self._elapsed_ms(turn)
            await self._send({"type": "audio_end", "turn_id": turn.id})
            await self._send({"type": "metrics", "turn_id": turn.id, **turn.metrics})
        except asyncio.CancelledError:
            raise
        except ChatThreadNotFoundError as exc:
            await self._send({"type": "error", "turn_id": turn.id, "detail": str(exc)})
        except Exception:
            log.warning("Live voice turn failed", exc_info=True)
            await self._send(
                {"type": "error", "turn_id": turn.id, "detail": "That turn failed. Please say it again."}
            )
        finally:
            await stream.aclose()
            if not speaker.done():
                speaker.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await speaker
            if self._turn is turn:
                self._turn = None

    async def _speak(self, turn: _Turn, sentences: asyncio.Queue[str | None]) -> None:
        failed = False
        while (sentence := await sentences.get()) is not None:
            if failed:
                await self._send(
                    {"type": "say", "turn_id": turn.id, "text": sentence, "audio": False}
                )
                continue
            # The client shows this text when the audio that follows starts playing.
            said = False
            remainder = b""
            try:
                async for chunk in self._speech.synthesize_pcm_stream(
                    sentence, sample_rate=self._rate, voice=self._voice
                ):
                    data = remainder + chunk
                    # 16-bit samples: never split one across two messages.
                    cut = len(data) - (len(data) % 2)
                    data, remainder = data[:cut], data[cut:]
                    if not data:
                        continue
                    if not said:
                        said = True
                        await self._send(
                            {"type": "say", "turn_id": turn.id, "text": sentence, "audio": True}
                        )
                    if not turn.audio_started:
                        turn.audio_started = True
                        turn.metrics["first_audio_ms"] = self._elapsed_ms(turn)
                        self._speaking = turn.id
                    await self._send(
                        {
                            "type": "audio",
                            "turn_id": turn.id,
                            "sample_rate": self._rate,
                            "data": base64.b64encode(data).decode("ascii"),
                        }
                    )
            except Exception:
                log.warning("Live text-to-speech failed", exc_info=True)
                failed = True
                await self._send(
                    {
                        "type": "error",
                        "turn_id": turn.id,
                        "detail": "Speech output failed; the answer is shown as text.",
                    }
                )
                if not said:
                    await self._send(
                        {"type": "say", "turn_id": turn.id, "text": sentence, "audio": False}
                    )
