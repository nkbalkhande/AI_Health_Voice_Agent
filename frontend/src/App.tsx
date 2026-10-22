import { ArrowDown, Brain, Keyboard, Mic, PanelLeftOpen, Settings, Sparkles, SquarePen, Volume2, VolumeX } from "lucide-react";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

import {
  WS_URL,
  blobToBase64,
  deleteThread,
  getThread,
  listThreads,
  newId,
  renameThread,
  synthesize,
} from "./api";
import ChatMessage from "./components/ChatMessage";
import Logo from "./components/Logo";
import Composer from "./components/Composer";
import VoiceDock from "./components/VoiceDock";
import MemoryPanel from "./components/MemoryPanel";
import PipelinePanel from "./components/PipelinePanel";
import SettingsPanel from "./components/SettingsPanel";
import Sidebar from "./components/Sidebar";
import VoiceMode from "./components/VoiceMode";
import VoiceStage from "./components/VoiceStage";
import { STATUS_LABELS, canInterrupt } from "./components/voiceText";
import type { ConnectionState, Fact, Message, PipelineState, ThreadSummary } from "./types";
import { PIPELINE_INITIAL } from "./types";
import { useVoiceSettings } from "./voice/settings";
import { StreamingSpeaker } from "./voice/streamingSpeaker";
import { useVoiceSession } from "./voice/useVoiceSession";

const THREAD_KEY = "voice-agent.threadId";
const AUTO_SPEAK_KEY = "voice-agent.autoSpeak";
const INPUT_MODE_KEY = "voice-agent.inputMode";

type InputMode = "voice" | "text";

type Turn = {
  viewKey: number;
  threadId: string | null;
  title: string;
  voice: boolean;
  assistantId: string;
};

function threadFromHash(): string | null {
  const match = window.location.hash.match(/^#\/c\/([\w-]+)/);
  return match ? match[1] : null;
}

function setHash(id: string | null) {
  const target = id ? `#/c/${id}` : "";
  if (window.location.hash !== target) {
    history.pushState(null, "", id ? target : window.location.pathname + window.location.search);
  }
  if (id) localStorage.setItem(THREAD_KEY, id);
  else localStorage.removeItem(THREAD_KEY);
}

function speakableText(markdown: string) {
  return markdown
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/\[\d+\]/g, "")
    .replace(/[*_#`>|]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function useIsDesktop() {
  const query = "(min-width: 768px)";
  const [desktop, setDesktop] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const mql = window.matchMedia(query);
    const onChange = () => setDesktop(mql.matches);
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, []);
  return desktop;
}

export default function App() {
  const isDesktop = useIsDesktop();
  const [sidebarOpen, setSidebarOpen] = useState(isDesktop);
  const [memoryOpen, setMemoryOpen] = useState(false);
  const [pipelineOpen, setPipelineOpen] = useState(false);

  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [threadsLoading, setThreadsLoading] = useState(true);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [summary, setSummary] = useState("");
  const [facts, setFacts] = useState<Fact[]>([]);
  const [loadingThread, setLoadingThread] = useState(false);

  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [busy, setBusy] = useState(false);
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [speakingId, setSpeakingId] = useState<string | null>(null);
  const [speechLoadingId, setSpeechLoadingId] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [atBottom, setAtBottom] = useState(true);
  const [autoSpeak, setAutoSpeak] = useState(() => localStorage.getItem(AUTO_SPEAK_KEY) !== "off");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const { settings: prefs, update: updatePrefs } = useVoiceSettings();
  const voicePrefRef = useRef(prefs.voice);
  voicePrefRef.current = prefs.voice;

  // Pipeline state tracking
  const [pipeline, setPipeline] = useState<PipelineState>(PIPELINE_INITIAL);
  const firstTokenRef = useRef(false);

  const socketRef = useRef<WebSocket | null>(null);
  const retryRef = useRef(0);
  const viewKeyRef = useRef(0);
  const turnRef = useRef<Turn | null>(null);
  const activeIdRef = useRef<string | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const speakerRef = useRef<StreamingSpeaker | null>(null);
  const speakerIdRef = useRef<string | null>(null);
  const endCallRef = useRef<() => void>(() => {});
  const scrollRef = useRef<HTMLDivElement | null>(null);

  activeIdRef.current = activeId;
  const autoSpeakRef = useRef(autoSpeak);
  autoSpeakRef.current = autoSpeak;

  const notify = useCallback((text: string) => {
    setToast(text);
    window.setTimeout(() => setToast((t) => (t === text ? null : t)), 4000);
  }, []);

  const refreshThreads = useCallback(async () => {
    try {
      setThreads(await listThreads());
    } catch {
      /* sidebar keeps its last known list while the backend is offline */
    } finally {
      setThreadsLoading(false);
    }
  }, []);

  // ---------- audio ----------
  const setRevealed = useCallback((messageId: string, revealed: number | undefined) => {
    setMessages((prev) => prev.map((m) => (m.id === messageId ? { ...m, revealed } : m)));
  }, []);

  const stopAudio = useCallback(() => {
    const spoken = speakerIdRef.current;
    if (speakerRef.current && spoken) setRevealed(spoken, undefined);
    speakerRef.current?.stop();
    speakerRef.current = null;
    speakerIdRef.current = null;
    const audio = audioRef.current;
    if (audio) {
      audio.pause();
      if (audio.src.startsWith("blob:")) URL.revokeObjectURL(audio.src);
    }
    audioRef.current = null;
    setSpeakingId(null);
  }, [setRevealed]);

  const playSrc = useCallback(
    (src: string, messageId: string) => {
      stopAudio();
      const audio = new Audio(src);
      audioRef.current = audio;
      setSpeakingId(messageId);
      audio.onended = () => {
        if (audioRef.current === audio) stopAudio();
      };
      void audio.play().catch(() => {
        setSpeakingId(null);
        notify("Your browser blocked autoplay. Press the speaker icon to listen.");
      });
    },
    [notify, stopAudio],
  );

  const startStreamingSpeech = useCallback(
    (messageId: string) => {
      stopAudio();
      const speaker = new StreamingSpeaker(
        speakableText,
        (chars) => {
          if (speakerRef.current === speaker) setRevealed(messageId, chars);
        },
        () => {
          if (speakerRef.current !== speaker) return;
          speakerRef.current = null;
          speakerIdRef.current = null;
          setRevealed(messageId, undefined);
          setSpeakingId(null);
        },
        notify,
        voicePrefRef.current,
      );
      speakerRef.current = speaker;
      speakerIdRef.current = messageId;
      setSpeakingId(messageId);
    },
    [notify, setRevealed, stopAudio],
  );

  function toggleAutoSpeak(next = !autoSpeak) {
    setAutoSpeak(next);
    localStorage.setItem(AUTO_SPEAK_KEY, next ? "on" : "off");
    if (!next && speakerRef.current) stopAudio();
  }

  async function speak(message: Message) {
    if (message.audioBase64) {
      playSrc(`data:audio/mpeg;base64,${message.audioBase64}`, message.id);
      return;
    }
    setSpeechLoadingId(message.id);
    try {
      const blob = await synthesize(speakableText(message.content), undefined, prefs.voice);
      playSrc(URL.createObjectURL(blob), message.id);
    } catch (err) {
      notify((err as Error).message);
    } finally {
      setSpeechLoadingId(null);
    }
  }

  // ---------- socket ----------
  const updateLastAssistant = (fn: (m: Message) => Message) =>
    setMessages((prev) => {
      const idx = prev.findLastIndex((m) => m.role === "assistant");
      if (idx < 0) return prev;
      const copy = [...prev];
      copy[idx] = fn(copy[idx]);
      return copy;
    });

  const failTurn = useCallback(
    (detail: string) => {
      const turn = turnRef.current;
      turnRef.current = null;
      setBusy(false);
      setTranscribing(false);
      stopAudio();
      setPipeline((p) => ({ ...p, rag: "error", web: "error", synthesis: "idle" }));
      if (turn && turn.viewKey === viewKeyRef.current) {
        setMessages((prev) => {
          const copy = prev.filter((m) => !(m.role === "user" && m.viaVoice && !m.content));
          const idx = copy.findLastIndex((m) => m.role === "assistant");
          if (idx >= 0 && copy[idx].streaming) {
            copy[idx] = { ...copy[idx], content: detail, streaming: false, error: true };
          }
          return copy;
        });
      }
      notify(detail);
    },
    [notify, stopAudio],
  );

  const handleEvent = useCallback(
    (data: Record<string, unknown>) => {
      const turn = turnRef.current;
      const visible = !!turn && turn.viewKey === viewKeyRef.current;

      if (data.type === "status") {
        const id = data.thread_id as string | undefined;
        const needsRetrieval = data.needs_retrieval as boolean | undefined;
        const intent = data.intent as string | undefined;
        const query = data.standalone_query as string | undefined;

        // Pipeline: status means understanding is done, retrieval starts
        setPipeline((p) => ({
          ...p,
          rag: needsRetrieval !== false ? "searching" : "skipped",
          web: needsRetrieval !== false ? "searching" : "skipped",
          synthesis: "idle",
          needsRetrieval: needsRetrieval !== false,
          intent: intent ?? "",
          query: query ?? "",
        }));
        firstTokenRef.current = false;

        if (turn && id && !turn.threadId) {
          turn.threadId = id;
          setThreads((prev) =>
            prev.some((t) => t.thread_id === id)
              ? prev
              : [
                  {
                    thread_id: id,
                    title: turn.title,
                    created_at: new Date().toISOString(),
                    updated_at: new Date().toISOString(),
                    message_count: 1,
                  },
                  ...prev,
                ],
          );
          if (visible) {
            setActiveId(id);
            setHash(id);
          }
        }
      } else if (data.type === "token") {
        // Pipeline: first token means retrieval is done, synthesis starts
        if (!firstTokenRef.current) {
          firstTokenRef.current = true;
          setPipeline((p) => ({
            ...p,
            rag: p.rag === "searching" ? "complete" : p.rag,
            web: p.web === "searching" ? "complete" : p.web,
            synthesis: "active",
          }));
        }

        if (visible) {
          const text = String(data.text ?? "");
          updateLastAssistant((m) => ({ ...m, content: m.content + text }));
          speakerRef.current?.push(text);
        }
      } else if (data.type === "done") {
        turnRef.current = null;
        setBusy(false);
        setTranscribing(false);

        // Pipeline: everything complete with sources
        const docResults = (data.document_results as PipelineState["documentResults"]) ?? [];
        const webResults = (data.web_results as PipelineState["webResults"]) ?? [];
        const citations = (data.citations as PipelineState["citations"]) ?? [];
        setPipeline((p) => ({
          ...p,
          rag: p.rag !== "skipped" ? "complete" : "skipped",
          web: p.web !== "skipped" ? "complete" : "skipped",
          synthesis: "complete",
          documentResults: docResults,
          webResults: webResults,
          citations: citations,
        }));

        const id = data.thread_id as string | undefined;
        if (visible) {
          if (id && activeIdRef.current !== id) {
            setActiveId(id);
            setHash(id);
          }
          const transcript = data.transcript as string | undefined;
          const assistantId = turn!.assistantId;
          speakerRef.current?.finish();
          setMessages((prev) => {
            const copy = [...prev];
            if (transcript !== undefined) {
              const u = copy.findLastIndex((m) => m.role === "user" && m.viaVoice);
              if (u >= 0) copy[u] = { ...copy[u], content: transcript || "(no speech detected)" };
            }
            const a = copy.findLastIndex((m) => m.role === "assistant");
            if (a >= 0) {
              copy[a] = {
                ...copy[a],
                id: assistantId,
                content: (data.answer as string) || copy[a].content,
                citations: data.citations as Message["citations"],
                audioBase64: (data.audio_base64 as string) || undefined,
                streaming: false,
              };
            }
            return copy;
          });
          if (data.summary !== undefined) setSummary((data.summary as string) ?? "");
          if (data.facts) setFacts(data.facts as Fact[]);
          if (data.audio_base64 && autoSpeakRef.current) playSrc(`data:audio/mpeg;base64,${data.audio_base64}`, assistantId);
          if (data.speech_error) notify(String(data.speech_error));
        }
        void refreshThreads();
      } else if (data.type === "error") {
        failTurn(String(data.detail ?? "Something went wrong"));
      }
    },
    [failTurn, notify, playSrc, refreshThreads],
  );

  useEffect(() => {
    let disposed = false;
    let timer: number | undefined;

    function connect() {
      setConnection("connecting");
      const socket = new WebSocket(WS_URL);
      socketRef.current = socket;
      socket.onopen = () => {
        retryRef.current = 0;
        setConnection("open");
      };
      socket.onmessage = (event) => handleEvent(JSON.parse(event.data as string));
      socket.onclose = () => {
        if (disposed) return;
        setConnection("closed");
        if (turnRef.current) failTurn("Connection lost. Please resend your message.");
        const delay = Math.min(1000 * 2 ** retryRef.current, 15000);
        retryRef.current += 1;
        timer = window.setTimeout(connect, delay);
      };
    }

    connect();
    return () => {
      disposed = true;
      window.clearTimeout(timer);
      socketRef.current?.close();
    };
  }, [handleEvent, failTurn]);

  // ---------- threads ----------
  const openThread = useCallback(
    async (id: string, { push = true } = {}) => {
      endCallRef.current();
      stopAudio();
      const key = ++viewKeyRef.current;
      setActiveId(id);
      if (push) setHash(id);
      setMessages([]);
      setSummary("");
      setFacts([]);
      setLoadingThread(true);
      setPipeline(PIPELINE_INITIAL);
      if (!isDesktop) setSidebarOpen(false);
      try {
        const data = await getThread(id);
        if (key !== viewKeyRef.current) return;
        if (!data) {
          notify("That conversation no longer exists.");
          setActiveId(null);
          setHash(null);
          return;
        }
        setMessages(data.messages);
        setSummary(data.summary);
        setFacts(data.facts);
      } catch (err) {
        if (key === viewKeyRef.current) notify((err as Error).message);
      } finally {
        if (key === viewKeyRef.current) setLoadingThread(false);
      }
    },
    [isDesktop, notify, stopAudio],
  );

  const newChat = useCallback(() => {
    endCallRef.current();
    stopAudio();
    viewKeyRef.current++;
    setActiveId(null);
    setHash(null);
    setMessages([]);
    setSummary("");
    setFacts([]);
    setLoadingThread(false);
    setPipeline(PIPELINE_INITIAL);
    if (!isDesktop) setSidebarOpen(false);
  }, [isDesktop, stopAudio]);

  useEffect(() => {
    void refreshThreads();
    const initial = threadFromHash() ?? localStorage.getItem(THREAD_KEY);
    if (initial) void openThread(initial);
    const onPop = () => {
      const id = threadFromHash();
      if (id === activeIdRef.current) return;
      if (id) void openThread(id, { push: false });
      else newChat();
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    setSidebarOpen(isDesktop);
  }, [isDesktop]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "o") {
        e.preventDefault();
        newChat();
      }
      if (e.key === "Escape" && !isDesktop) setSidebarOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isDesktop, newChat]);

  async function handleRename(id: string, title: string) {
    setThreads((prev) => prev.map((t) => (t.thread_id === id ? { ...t, title } : t)));
    try {
      await renameThread(id, title);
    } catch (err) {
      notify((err as Error).message);
      void refreshThreads();
    }
  }

  async function handleDelete(id: string) {
    setThreads((prev) => prev.filter((t) => t.thread_id !== id));
    if (activeIdRef.current === id) newChat();
    try {
      await deleteThread(id);
    } catch (err) {
      notify((err as Error).message);
      void refreshThreads();
    }
  }

  // ---------- sending ----------
  function socketReady() {
    if (socketRef.current?.readyState === WebSocket.OPEN) return true;
    notify("Not connected to the agent yet. Retrying…");
    return false;
  }

  function sendText(text: string) {
    if (busy || !socketReady()) return;
    const assistantId = newId();
    if (autoSpeak) startStreamingSpeech(assistantId);
    else stopAudio();

    // Reset pipeline for new turn
    setPipeline({
      ...PIPELINE_INITIAL,
      rag: "understanding",
      web: "understanding",
      query: text,
    });
    firstTokenRef.current = false;

    turnRef.current = {
      viewKey: viewKeyRef.current,
      threadId: activeId,
      title: text.length > 60 ? text.slice(0, 57) + "…" : text,
      voice: false,
      assistantId,
    };
    setMessages((prev) => [
      ...prev,
      { id: newId(), role: "user", content: text },
      {
        id: assistantId,
        role: "assistant",
        content: "",
        streaming: true,
        revealed: autoSpeak ? 0 : undefined,
      },
    ]);
    setBusy(true);
    setAtBottom(true);
    socketRef.current!.send(JSON.stringify({ type: "message", message: text, thread_id: activeId }));
  }

  async function toggleRecording() {
    if (recording) {
      recorderRef.current?.stop();
      return;
    }
    if (!socketReady()) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      recorderRef.current = recorder;
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size) chunksRef.current.push(e.data);
      };
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        setRecording(false);
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
        if (!blob.size || !socketReady()) return;
        stopAudio();
        const threadId = activeIdRef.current;
        const assistantId = newId();

        setPipeline({ ...PIPELINE_INITIAL, rag: "understanding", web: "understanding" });
        firstTokenRef.current = false;

        turnRef.current = {
          viewKey: viewKeyRef.current,
          threadId,
          title: "Voice conversation",
          voice: true,
          assistantId,
        };
        setMessages((prev) => [
          ...prev,
          { id: newId(), role: "user", content: "", viaVoice: true },
          { id: assistantId, role: "assistant", content: "", streaming: true },
        ]);
        setBusy(true);
        setTranscribing(true);
        setAtBottom(true);
        const audio_base64 = await blobToBase64(blob);
        socketRef.current!.send(
          JSON.stringify({ type: "voice", audio_base64, content_type: blob.type, thread_id: threadId }),
        );
      };
      recorder.start();
      setRecording(true);
    } catch {
      notify("Microphone permission was denied.");
    }
  }

  // ---------- live voice mode ----------
  const [focusOpen, setFocusOpen] = useState(false);
  const [inputMode, setInputModeState] = useState<InputMode>(() =>
    localStorage.getItem(INPUT_MODE_KEY) === "text" ? "text" : "voice",
  );
  const voiceKeyRef = useRef("");
  const voiceTitleRef = useRef("");
  const voiceId = (turnId: string, role: "u" | "a") => `v${voiceKeyRef.current}-${turnId}-${role}`;

  const voice = useVoiceSession({
    onTurnStart: (turnId, text) => {
      stopAudio();
      if (!voiceTitleRef.current) voiceTitleRef.current = text;
      setAtBottom(true);

      // Pipeline: voice turn started → searching
      setPipeline({
        ...PIPELINE_INITIAL,
        rag: "searching",
        web: "searching",
        query: text,
      });

      setMessages((prev) => [
        ...prev,
        { id: voiceId(turnId, "u"), role: "user", content: text, viaVoice: true },
        { id: voiceId(turnId, "a"), role: "assistant", content: "", streaming: true },
      ]);
    },
    onReveal: (turnId, text) => {
      const id = voiceId(turnId, "a");
      setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, content: text } : m)));

      // Pipeline: first reveal means retrieval done, generating
      setPipeline((p) =>
        p.synthesis !== "active" && p.synthesis !== "complete"
          ? { ...p, rag: "complete", web: "complete", synthesis: "active" }
          : p,
      );
    },
    onDone: (turnId, data) => {
      const id = voiceId(turnId, "a");
      setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, citations: data.citations } : m)));

      // Pipeline: complete with sources
      setPipeline((p) => ({
        ...p,
        synthesis: "complete",
        citations: data.citations ?? [],
      }));

      if (data.summary !== undefined) setSummary(data.summary ?? "");
      if (data.facts) setFacts(data.facts);
      void refreshThreads();
    },
    onSpoken: (turnId, text) => {
      const id = voiceId(turnId, "a");
      setMessages((prev) =>
        prev.map((m) => (m.id === id ? { ...m, content: text || m.content, streaming: false } : m)),
      );
    },
    onInterrupt: (turnId, merged) => {
      const userId = voiceId(turnId, "u");
      const agentId = voiceId(turnId, "a");
      setMessages((prev) =>
        merged
          ? prev.filter((m) => m.id !== userId && m.id !== agentId)
          : prev.flatMap((m) => {
              if (m.id !== agentId || !m.streaming) return [m];
              return m.content.trim()
                ? [{ ...m, content: `${m.content.trimEnd()} …`, streaming: false, interrupted: true }]
                : [];
            }),
      );
    },
    onThread: (id) => {
      if (activeIdRef.current === id) return;
      setActiveId(id);
      setHash(id);
      setThreads((prev) =>
        prev.some((t) => t.thread_id === id)
          ? prev
          : [
              {
                thread_id: id,
                title: voiceTitleRef.current || "Voice conversation",
                created_at: new Date().toISOString(),
                updated_at: new Date().toISOString(),
                message_count: 1,
              },
              ...prev,
            ],
      );
    },
    onError: notify,
  });

  const inCall = voice.status !== "idle";

  function startCall() {
    if (busy) return notify("Wait for the current answer to finish first.");
    stopAudio();
    setInputMode("voice");
    voiceKeyRef.current = newId();
    voiceTitleRef.current = "";
    setAtBottom(true);
    void voice.start(activeIdRef.current, { voice: prefs.voice, micId: prefs.micId });
  }

  function endCall() {
    if (voiceStatusRef.current === "idle") return;
    voice.stop();
    setFocusOpen(false);
    setMessages((prev) => prev.map((m) => (m.streaming && m.id.startsWith("v") ? { ...m, streaming: false } : m)));
    void refreshThreads();
  }
  const voiceStatusRef = useRef(voice.status);
  voiceStatusRef.current = voice.status;
  endCallRef.current = endCall;

  function setInputMode(mode: InputMode) {
    setInputModeState(mode);
    localStorage.setItem(INPUT_MODE_KEY, mode);
  }

  function typeInstead() {
    endCall();
    setInputMode("text");
  }

  const spaceRef = useRef<() => void>(() => {});
  spaceRef.current = () => {
    if (settingsOpen) return;
    if (voice.status === "idle" || voice.status === "error") {
      if (connection === "open" && !busy && !loadingThread) startCall();
    } else if (canInterrupt(voice.status)) {
      voice.interrupt();
    }
  };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== " " || e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
      const target = e.target;
      if (
        target instanceof Element &&
        target.closest("input, textarea, select, button, a, [contenteditable='true'], [role]:not([role='dialog'])")
      ) {
        return;
      }
      e.preventDefault();
      spaceRef.current();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // ---------- scrolling ----------
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (el && atBottom) el.scrollTop = el.scrollHeight;
  }, [messages, atBottom, voice.draft]);

  function onScroll() {
    const el = scrollRef.current;
    if (!el) return;
    setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 80);
  }

  const callProps = {
    status: voice.status,
    userCaption: voice.userCaption,
    agentCaption: voice.agentCaption,
    muted: voice.muted,
    error: voice.error,
    level: voice.level,
    onStart: startCall,
    onInterrupt: voice.interrupt,
    onToggleMute: voice.toggleMute,
    onEnd: endCall,
    onExpand: () => setFocusOpen(true),
    onType: typeInstead,
  };
  const showDraft = !!voice.draft.trim() && inCall && voice.status !== "error";
  const announcement = !inCall
    ? ""
    : voice.status === "error" && voice.error
      ? `${voice.error.title}. ${voice.error.detail}`
      : STATUS_LABELS[voice.status];

  const activeTitle = threads.find((t) => t.thread_id === activeId)?.title;
  const showEmpty = !activeId && messages.length === 0;

  return (
    <div className="flex h-dvh overflow-hidden bg-main text-slate-800">
      {/* Sidebar */}
      {isDesktop ? (
        <div
          className={`shrink-0 overflow-hidden transition-[width] duration-200 ease-out ${
            sidebarOpen ? "w-[272px]" : "w-0"
          }`}
        >
          <Sidebar
            threads={threads}
            loading={threadsLoading}
            activeId={activeId}
            connection={connection}
            onSelect={(id) => id !== activeId && void openThread(id)}
            onNew={newChat}
            onRename={handleRename}
            onDelete={handleDelete}
            onClose={() => setSidebarOpen(false)}
            onSettings={() => setSettingsOpen(true)}
          />
        </div>
      ) : (
        <>
          <div
            onClick={() => setSidebarOpen(false)}
            className={`fixed inset-0 z-30 bg-black/40 transition-opacity ${
              sidebarOpen ? "opacity-100" : "pointer-events-none opacity-0"
            }`}
          />
          <div
            className={`fixed inset-y-0 left-0 z-40 transition-transform duration-200 ${
              sidebarOpen ? "translate-x-0" : "-translate-x-full"
            }`}
          >
            <Sidebar
              threads={threads}
              loading={threadsLoading}
              activeId={activeId}
              connection={connection}
              onSelect={(id) => (id !== activeId ? void openThread(id) : setSidebarOpen(false))}
              onNew={newChat}
              onRename={handleRename}
              onDelete={handleDelete}
              onClose={() => setSidebarOpen(false)}
              onSettings={() => setSettingsOpen(true)}
            />
          </div>
        </>
      )}

      {/* Main column */}
      <main className="voice-surface relative flex min-w-0 flex-1 flex-col">
        {/* Top navigation bar */}
        <header className="flex h-14 shrink-0 items-center gap-1 border-b border-card-border bg-white/80 px-2 backdrop-blur md:px-3">
          {!sidebarOpen && (
            <>
              <button onClick={() => setSidebarOpen(true)} className="icon-btn" title="Open sidebar">
                <PanelLeftOpen size={18} />
              </button>
              <button onClick={newChat} className="icon-btn" title="New chat">
                <SquarePen size={18} />
              </button>
            </>
          )}
          <div className="flex min-w-0 flex-1 items-center gap-2 px-2">
            {!sidebarOpen && <Logo size={26} />}
            <h2 className="truncate text-[15px] font-medium text-slate-700">
              {activeTitle ?? (activeId ? "Conversation" : "HealthMate AI")}
            </h2>
          </div>
          {inCall && voice.status !== "error" && (
            <span className="mr-1 flex items-center gap-1.5 rounded-full bg-emerald-50 px-2.5 py-1 text-xs font-medium text-emerald-600">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-500" />
              Live
            </span>
          )}
          {connection !== "open" && (
            <span className="mr-2 hidden items-center gap-1.5 rounded-full bg-amber-50 px-2.5 py-1 text-xs text-amber-600 sm:flex">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-500" />
              {connection === "connecting" ? "Connecting" : "Reconnecting"}
            </span>
          )}
          <button
            onClick={() => toggleAutoSpeak()}
            className={`icon-btn ${autoSpeak ? "text-primary" : ""}`}
            title={autoSpeak ? "Read typed answers aloud: on" : "Read typed answers aloud: off"}
            aria-label="Read typed answers aloud"
            aria-pressed={autoSpeak}
          >
            {autoSpeak ? <Volume2 size={18} /> : <VolumeX size={18} />}
          </button>
          <button
            onClick={() => setMemoryOpen(!memoryOpen)}
            className={`icon-btn ${memoryOpen ? "bg-primary-light text-primary" : ""}`}
            title="Session memory"
            aria-label="Session memory"
            aria-pressed={memoryOpen}
          >
            <Brain size={18} />
            {facts.length > 0 && (
              <span className="absolute top-1 right-1 h-2 w-2 rounded-full bg-primary" />
            )}
          </button>
          <button
            onClick={() => setPipelineOpen(!pipelineOpen)}
            className={`icon-btn ${pipelineOpen ? "bg-primary-light text-primary" : ""}`}
            title="AI Research panel"
            aria-label="AI Research panel"
            aria-pressed={pipelineOpen}
          >
            <Sparkles size={18} />
          </button>
          <button
            onClick={() => setSettingsOpen(true)}
            className="icon-btn"
            title="Settings"
            aria-label="Settings"
            aria-haspopup="dialog"
          >
            <Settings size={18} />
          </button>
        </header>
        <div className="sr-only" role="status" aria-live="polite">
          {announcement}
        </div>

        {showEmpty ? (
          <>
            <VoiceStage
              {...callProps}
              metrics={voice.metrics}
              disabled={connection !== "open" || busy}
              onAsk={sendText}
            />
            {inputMode === "text" ? (
              <Composer
                disabled={connection !== "open"}
                busy={busy}
                recording={recording}
                transcribing={transcribing}
                onSend={sendText}
                onToggleRecording={toggleRecording}
                onVoiceMode={startCall}
                autoFocusKey="new"
              />
            ) : (
              !inCall && (
                <button
                  onClick={typeInstead}
                  className="mx-auto mb-6 flex items-center gap-2 rounded-full px-4 py-2 text-sm text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600"
                >
                  <Keyboard size={16} /> Prefer typing?
                </button>
              )
            )}
          </>
        ) : (
          <>
            <div ref={scrollRef} onScroll={onScroll} className="scrollbar-thin flex-1 overflow-y-auto">
              <div className="mx-auto w-full max-w-3xl space-y-8 px-4 pt-4 pb-10">
                {loadingThread &&
                  [0, 1].map((i) => (
                    <div key={i} className="space-y-6">
                      <div className="ml-auto h-10 w-1/2 animate-pulse rounded-3xl bg-slate-100" />
                      <div className="space-y-2">
                        <div className="h-4 w-11/12 animate-pulse rounded bg-slate-100" />
                        <div className="h-4 w-4/5 animate-pulse rounded bg-slate-100" />
                        <div className="h-4 w-2/3 animate-pulse rounded bg-slate-100" />
                      </div>
                    </div>
                  ))}
                {messages.map((m) =>
                  m.role === "user" && m.viaVoice && !m.content ? (
                    <div key={m.id} className="flex justify-end">
                      <div className="flex items-center gap-2 rounded-3xl border border-card-border bg-white px-5 py-2.5 text-sm text-slate-400 italic shadow-sm">
                        <span className="typing-dot" /> Transcribing your voice…
                      </div>
                    </div>
                  ) : (
                    <ChatMessage
                      key={m.id}
                      message={m}
                      speaking={speakingId === m.id}
                      liveSpeaking={voice.status === "speaking" && !!m.streaming && m.id.startsWith("v")}
                      loadingSpeech={speechLoadingId === m.id}
                      onSpeak={speak}
                      onStopSpeak={stopAudio}
                    />
                  ),
                )}
                {showDraft && (
                  <div className="flex justify-end">
                    <div className="draft-bubble max-w-[85%] rounded-3xl px-5 py-2.5 text-[15px] leading-relaxed text-slate-600 italic md:max-w-[70%]">
                      <span className="mb-1 flex items-center gap-1 text-[11px] font-medium tracking-wide text-primary/70 not-italic uppercase">
                        <Mic size={11} /> Hearing you…
                      </span>
                      {voice.draft}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {!atBottom && (
              <button
                onClick={() => setAtBottom(true)}
                className="absolute bottom-36 left-1/2 z-10 grid h-9 w-9 -translate-x-1/2 place-items-center rounded-full border border-card-border bg-white text-slate-500 shadow-lg hover:text-slate-700"
                title="Scroll to bottom"
              >
                <ArrowDown size={16} />
              </button>
            )}

            {inputMode === "text" && !inCall ? (
              <Composer
                disabled={connection !== "open" || loadingThread}
                busy={busy}
                recording={recording}
                transcribing={transcribing}
                onSend={sendText}
                onToggleRecording={toggleRecording}
                onVoiceMode={startCall}
                autoFocusKey={activeId ?? "new"}
              />
            ) : (
              <VoiceDock
                {...callProps}
                disabled={connection !== "open" || loadingThread || busy}
                onType={typeInstead}
              />
            )}
          </>
        )}

        {focusOpen && inCall && (
          <VoiceMode
            {...callProps}
            metrics={voice.metrics}
            onRetry={startCall}
            onMinimize={() => setFocusOpen(false)}
          />
        )}

        {settingsOpen && (
          <SettingsPanel
            settings={prefs}
            onChange={updatePrefs}
            autoSpeak={autoSpeak}
            onAutoSpeak={toggleAutoSpeak}
            inputMode={inputMode}
            onInputMode={setInputMode}
            inCall={inCall}
            onClose={() => setSettingsOpen(false)}
          />
        )}

        {toast && (
          <div
            role="status"
            aria-live="polite"
            className="toast-in absolute top-16 left-1/2 z-[60] -translate-x-1/2 rounded-xl border border-card-border bg-white px-4 py-2.5 text-sm text-slate-700 shadow-lg"
          >
            {toast}
          </div>
        )}
      </main>

      {/* Pipeline / Research panel - desktop */}
      {pipelineOpen &&
        (isDesktop ? (
          <PipelinePanel
            pipeline={pipeline}
            onClose={() => setPipelineOpen(false)}
          />
        ) : (
          <>
            <div onClick={() => setPipelineOpen(false)} className="fixed inset-0 z-30 bg-black/40" />
            <div className="fixed inset-y-0 right-0 z-40">
              <PipelinePanel
                pipeline={pipeline}
                onClose={() => setPipelineOpen(false)}
              />
            </div>
          </>
        ))}

      {/* Memory panel */}
      {memoryOpen && !pipelineOpen &&
        (isDesktop ? (
          <MemoryPanel
            threadId={activeId}
            summary={summary}
            facts={facts}
            onClose={() => setMemoryOpen(false)}
          />
        ) : (
          <>
            <div onClick={() => setMemoryOpen(false)} className="fixed inset-0 z-30 bg-black/40" />
            <div className="fixed inset-y-0 right-0 z-40">
              <MemoryPanel
                threadId={activeId}
                summary={summary}
                facts={facts}
                onClose={() => setMemoryOpen(false)}
              />
            </div>
          </>
        ))}
    </div>
  );
}
