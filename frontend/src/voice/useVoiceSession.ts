import { useCallback, useEffect, useRef, useState } from "react";

import { API } from "../api";
import type { Citation, Fact } from "../types";
import { PcmPlayer, startMicrophone, type Microphone } from "./audio";
import { CONNECTION_LOST, microphoneError, serviceError, voiceSupportProblem, type VoiceError } from "./errors";

const VOICE_WS_URL = API.replace(/^http/, "ws") + "/api/v1/ws/voice";

export type VoiceStatus =
  | "idle"
  | "requesting_permission"
  | "connecting"
  | "listening"
  /** Words are arriving from speech-to-text; the turn has not been submitted yet. */
  | "transcribing"
  /** Understanding the question, searching, and writing the answer. */
  | "thinking"
  | "speaking"
  /** The user talked over the agent; shown briefly before listening again. */
  | "interrupted"
  | "error";

export type StartOptions = { voice?: string | null; micId?: string | null };

/** No new words for this long while transcribing means the user paused. */
const HEARING_IDLE_MS = 1500;
const INTERRUPTED_MS = 700;

export type VoiceMetrics = {
  retrieval_ms?: number;
  first_token_ms?: number;
  first_audio_ms?: number;
  total_ms?: number;
};

export type VoiceDone = {
  thread_id?: string;
  answer?: string;
  citations?: Citation[];
  summary?: string;
  facts?: Fact[];
};

const CHARS_PER_SECOND = 15;

type Reveal = { turnId: string; shown: string; words: string[]; timer?: number };

export type VoiceHandlers = {
  onTurnStart: (turnId: string, text: string) => void;
  /** The part of the answer spoken so far; text follows the voice. */
  onReveal: (turnId: string, text: string) => void;
  onDone: (turnId: string, data: VoiceDone) => void;
  /** Playback finished; `text` is the full answer. */
  onSpoken: (turnId: string, text: string) => void;
  onInterrupt: (turnId: string, merged: boolean) => void;
  onThread: (threadId: string) => void;
  onError: (message: string) => void;
};

export function useVoiceSession(handlers: VoiceHandlers) {
  const [status, setStatus] = useState<VoiceStatus>("idle");
  const [userCaption, setUserCaption] = useState("");
  /** Words heard since the last submitted turn; empty once a turn starts. */
  const [draft, setDraft] = useState("");
  const [agentCaption, setAgentCaption] = useState("");
  const [metrics, setMetrics] = useState<VoiceMetrics | null>(null);
  const [muted, setMuted] = useState(false);
  const [error, setError] = useState<VoiceError | null>(null);

  const handlersRef = useRef(handlers);
  handlersRef.current = handlers;
  const socketRef = useRef<WebSocket | null>(null);
  const micRef = useRef<Microphone | null>(null);
  const playerRef = useRef<PcmPlayer | null>(null);
  const currentTurnRef = useRef<string | null>(null);
  const cancelledRef = useRef(new Set<string>());
  const statusRef = useRef<VoiceStatus>("idle");
  const revealRef = useRef<Reveal | null>(null);
  const answersRef = useRef(new Map<string, string>());
  const sessionRef = useRef(0);
  const stateTimerRef = useRef<number | undefined>(undefined);

  const update = (next: VoiceStatus) => {
    window.clearTimeout(stateTimerRef.current);
    statusRef.current = next;
    setStatus(next);
  };

  /** Enter `next`, then fall back to listening after `ms` unless something else happened. */
  const updateFor = (next: VoiceStatus, ms: number) => {
    update(next);
    stateTimerRef.current = window.setTimeout(() => {
      if (statusRef.current === next) update("listening");
    }, ms);
  };

  const show = useCallback((turnId: string, text: string) => {
    setAgentCaption(text);
    handlersRef.current.onReveal(turnId, text);
  }, []);

  /** Stops the typing animation; with `flush`, shows the rest of the current sentence. */
  const settleReveal = useCallback(
    (flush: boolean) => {
      const reveal = revealRef.current;
      if (!reveal) return;
      window.clearTimeout(reveal.timer);
      reveal.timer = undefined;
      if (flush && reveal.words.length) {
        reveal.shown += reveal.words.join("");
        reveal.words = [];
        show(reveal.turnId, reveal.shown);
      }
    },
    [show],
  );

  /** Types a sentence out at speaking pace, starting when its audio starts. */
  const revealSentence = useCallback(
    (turnId: string, sentence: string, instant: boolean) => {
      settleReveal(true);
      const previous = revealRef.current?.turnId === turnId ? revealRef.current.shown : "";
      const reveal: Reveal = {
        turnId,
        shown: previous ? `${previous} ` : "",
        words: sentence.match(/\S+\s*/g) ?? [],
      };
      revealRef.current = reveal;
      if (instant) return settleReveal(true);
      const step = () => {
        const word = reveal.words.shift();
        if (word === undefined) return;
        reveal.shown += word;
        show(turnId, reveal.shown);
        reveal.timer = window.setTimeout(step, (word.length / CHARS_PER_SECOND) * 1000);
      };
      step();
    },
    [settleReveal, show],
  );

  const teardown = useCallback(() => {
    sessionRef.current++;
    window.clearTimeout(stateTimerRef.current);
    settleReveal(false);
    revealRef.current = null;
    answersRef.current.clear();
    setDraft("");
    const socket = socketRef.current;
    socketRef.current = null;
    if (socket) {
      socket.onmessage = socket.onclose = socket.onerror = null;
      if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "stop" }));
      socket.close();
    }
    micRef.current?.stop();
    micRef.current = null;
    playerRef.current?.close();
    playerRef.current = null;
    currentTurnRef.current = null;
    cancelledRef.current.clear();
  }, [settleReveal]);

  const fail = useCallback(
    (problem: VoiceError) => {
      teardown();
      setError(problem);
      update("error");
    },
    [teardown],
  );

  const handleMessage = useCallback(
    (data: Record<string, unknown>) => {
      const h = handlersRef.current;
      const turnId = data.turn_id as string | undefined;
      const stale = !!turnId && cancelledRef.current.has(turnId);

      switch (data.type) {
        case "ready":
          update("listening");
          break;
        case "transcript": {
          const text = String(data.text ?? "");
          setUserCaption(text);
          setDraft(text);
          const now = statusRef.current;
          if (now === "speaking" || now === "thinking" || now === "interrupted") break;
          if (text.trim()) updateFor("transcribing", HEARING_IDLE_MS);
          else update("listening");
          break;
        }
        case "turn_start":
          currentTurnRef.current = turnId!;
          setUserCaption(String(data.text ?? ""));
          setDraft("");
          setAgentCaption("");
          setMetrics(null);
          update("thinking");
          h.onTurnStart(turnId!, String(data.text ?? ""));
          break;
        case "status":
          if (data.thread_id) h.onThread(String(data.thread_id));
          break;
        case "say": {
          if (stale) break;
          const text = String(data.text ?? "");
          if (data.audio && playerRef.current) {
            playerRef.current.cue(turnId!, () => revealSentence(turnId!, text, false));
          } else {
            revealSentence(turnId!, text, true);
          }
          break;
        }
        case "audio":
          if (stale || !playerRef.current) break;
          playerRef.current.enqueue(turnId!, String(data.data), Number(data.sample_rate) || 24000);
          if (statusRef.current !== "speaking") update("speaking");
          break;
        case "audio_end":
          if (!stale) playerRef.current?.markEnd(turnId!);
          break;
        case "done":
          if (stale) break;
          answersRef.current.set(turnId!, String(data.answer ?? ""));
          h.onDone(turnId!, data as VoiceDone);
          break;
        case "metrics":
          if (!stale) setMetrics(data as VoiceMetrics);
          break;
        case "interrupt":
          cancelledRef.current.add(turnId!);
          playerRef.current?.stop();
          settleReveal(false);
          answersRef.current.delete(turnId!);
          h.onInterrupt(turnId!, Boolean(data.merged));
          if (currentTurnRef.current === turnId) currentTurnRef.current = null;
          setAgentCaption("");
          updateFor("interrupted", INTERRUPTED_MS);
          break;
        case "error":
          if (data.fatal) fail(serviceError(String(data.detail ?? "The voice session failed.")));
          else h.onError(String(data.detail ?? "Something went wrong"));
          break;
      }
    },
    [fail, revealSentence, settleReveal],
  );

  const finishSpeaking = useCallback(
    (turnId: string) => {
      if (cancelledRef.current.has(turnId)) return;
      settleReveal(true);
      const spoken = revealRef.current?.turnId === turnId ? revealRef.current.shown : "";
      const answer = answersRef.current.get(turnId) || spoken;
      answersRef.current.delete(turnId);
      if (revealRef.current?.turnId === turnId) revealRef.current = null;
      handlersRef.current.onSpoken(turnId, answer);
    },
    [settleReveal],
  );

  const start = useCallback(
    async (threadId: string | null, options: StartOptions = {}) => {
      teardown();
      const session = sessionRef.current;
      setError(null);
      setUserCaption("");
      setAgentCaption("");
      setMetrics(null);
      setMuted(false);

      const unsupported = voiceSupportProblem();
      if (unsupported) return fail(unsupported);
      update("requesting_permission");

      const player = new PcmPlayer((turnId) => {
        finishSpeaking(turnId);
        const socket = socketRef.current;
        if (socket?.readyState === WebSocket.OPEN) {
          socket.send(JSON.stringify({ type: "playback_end", turn_id: turnId }));
        }
        const current = currentTurnRef.current;
        if (current !== null && current !== turnId) return;
        currentTurnRef.current = null;
        if (statusRef.current === "speaking" || statusRef.current === "thinking") update("listening");
      });
      playerRef.current = player;
      try {
        await player.resume();
      } catch {
        return fail({
          title: "Audio playback is blocked",
          detail: "The browser didn't allow sound. Click anywhere on the page, then try again.",
          retryable: true,
        });
      }
      if (sessionRef.current !== session) return;

      let mic: Microphone;
      try {
        mic = await startMicrophone((pcm) => {
          const socket = socketRef.current;
          if (socket?.readyState === WebSocket.OPEN && statusRef.current !== "connecting") {
            socket.send(pcm);
          }
        }, options.micId);
      } catch (err) {
        if (sessionRef.current === session) fail(microphoneError(err));
        return;
      }
      if (sessionRef.current !== session) {
        mic.stop();
        return;
      }
      micRef.current = mic;
      update("connecting");

      const socket = new WebSocket(VOICE_WS_URL);
      socketRef.current = socket;
      socket.onopen = () =>
        socket.send(
          JSON.stringify({
            type: "start",
            sample_rate: mic.sampleRate,
            thread_id: threadId,
            voice: options.voice || null,
          }),
        );
      socket.onmessage = (event) => handleMessage(JSON.parse(event.data as string));
      socket.onerror = () => fail(CONNECTION_LOST);
      socket.onclose = () => {
        if (socketRef.current === socket) fail(CONNECTION_LOST);
      };
    },
    [fail, finishSpeaking, handleMessage, teardown],
  );

  const stop = useCallback(() => {
    teardown();
    update("idle");
  }, [teardown]);

  const interrupt = useCallback(() => {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ type: "interrupt" }));
  }, []);

  const toggleMute = useCallback(() => {
    setMuted((m) => {
      micRef.current?.setMuted(!m);
      return !m;
    });
  }, []);

  const level = useCallback(() => {
    if (statusRef.current === "speaking") return playerRef.current?.level() ?? 0;
    if (["listening", "transcribing", "interrupted"].includes(statusRef.current)) {
      return micRef.current?.level() ?? 0;
    }
    return 0;
  }, []);

  useEffect(() => teardown, [teardown]);

  return {
    status,
    userCaption,
    draft,
    agentCaption,
    metrics,
    muted,
    error,
    start,
    stop,
    interrupt,
    toggleMute,
    level,
  };
}
