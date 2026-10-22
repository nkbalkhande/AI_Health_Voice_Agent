import { Headphones, Keyboard, Mic, MicOff, Minimize2, PhoneOff, RotateCcw } from "lucide-react";
import { useEffect, useRef } from "react";

import type { VoiceError } from "../voice/errors";
import type { VoiceMetrics, VoiceStatus } from "../voice/useVoiceSession";
import Orb from "./Orb";
import { STATUS_HINTS, STATUS_LABELS, canInterrupt, liveCaption, seconds } from "./voiceText";

type Props = {
  status: VoiceStatus;
  userCaption: string;
  agentCaption: string;
  metrics: VoiceMetrics | null;
  muted: boolean;
  error: VoiceError | null;
  level: () => number;
  onInterrupt: () => void;
  onToggleMute: () => void;
  onRetry: () => void;
  onEnd: () => void;
  onMinimize: () => void;
  onType: () => void;
};

/** Distraction-free fullscreen voice overlay (stays dark for immersion). */
export default function VoiceMode({
  status,
  userCaption,
  agentCaption,
  metrics,
  muted,
  error,
  level,
  onInterrupt,
  onToggleMute,
  onRetry,
  onEnd,
  onMinimize,
  onType,
}: Props) {
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    rootRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onMinimize();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      previous?.focus?.();
    };
  }, [onMinimize]);

  const caption = liveCaption(status, userCaption, agentCaption);
  const interruptible = canInterrupt(status);
  const isError = status === "error";
  const starting = status === "connecting" || status === "requesting_permission";

  return (
    <div
      ref={rootRef}
      role="dialog"
      aria-modal="true"
      aria-label="Voice conversation"
      tabIndex={-1}
      className="voice-backdrop safe-top fixed inset-0 z-50 flex flex-col items-center overflow-hidden text-zinc-100 outline-none"
    >
      <header className="flex w-full items-center justify-between px-5 py-4">
        <div className="flex items-center gap-2 text-sm text-zinc-400">
          <span
            className={`h-2 w-2 rounded-full ${
              isError ? "bg-red-500" : status === "connecting" ? "animate-pulse bg-amber-400" : "bg-emerald-400"
            }`}
          />
          HealthMate AI · live
        </div>
        <span className="hidden items-center gap-1.5 text-xs text-zinc-500 sm:flex">
          <Headphones size={14} /> Headphones give the cleanest interruptions · Space interrupts · Esc goes back
        </span>
      </header>

      <div className="flex w-full max-w-2xl flex-1 flex-col items-center justify-center px-6">
        <Orb
          status={status}
          level={level}
          label={interruptible ? "Interrupt" : STATUS_LABELS[status]}
          onClick={interruptible ? onInterrupt : undefined}
        />

        <div className="mt-6 text-center">
          <p className="text-lg font-medium tracking-tight text-zinc-100">
            {isError && error ? error.title : STATUS_LABELS[status]}
          </p>
          <p className={`mx-auto mt-1 max-w-md text-sm ${isError ? "text-red-300/90" : "text-zinc-500"}`}>
            {isError ? error?.detail : STATUS_HINTS[status]}
          </p>
        </div>

        <div className="mt-8 flex min-h-[7.5rem] w-full items-start justify-center">
          {caption.text && !isError && (
            <p
              key={caption.agent ? "agent" : "user"}
              className={`caption-in line-clamp-4 max-w-xl text-center text-xl leading-relaxed md:text-2xl ${
                caption.agent ? "text-zinc-100" : "text-zinc-300"
              }`}
            >
              {caption.agent ? caption.text : `"${caption.text}"`}
            </p>
          )}
        </div>

        {metrics && (
          <div className="flex flex-wrap justify-center gap-2 text-xs">
            {(
              [
                ["Understood + searched", metrics.retrieval_ms],
                ["First word", metrics.first_token_ms],
                ["First audio", metrics.first_audio_ms],
              ] as const
            ).map(([label, ms]) => (
              <span key={label} className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-zinc-400">
                {label} <span className="font-mono text-zinc-200">{seconds(ms)}</span>
              </span>
            ))}
          </div>
        )}
      </div>

      <footer className="safe-bottom pb-6">
        <div className="flex items-center gap-3">
          {isError ? (
            error?.retryable !== false ? (
              <button onClick={onRetry} className="flex h-12 items-center gap-2 rounded-full bg-white px-4 text-sm font-medium text-zinc-900 hover:opacity-90">
                <RotateCcw size={16} /> Try again
              </button>
            ) : (
              <button onClick={onType} className="flex h-12 items-center gap-2 rounded-full bg-white px-4 text-sm font-medium text-zinc-900 hover:opacity-90">
                <Keyboard size={16} /> Type instead
              </button>
            )
          ) : (
            <button
              onClick={onToggleMute}
              disabled={starting}
              aria-pressed={muted}
              className={`grid h-12 w-12 place-items-center rounded-full transition-colors disabled:opacity-40 ${
                muted ? "bg-red-500/90 text-white" : "bg-white/10 text-zinc-100 hover:bg-white/15"
              }`}
              title={muted ? "Unmute microphone" : "Mute microphone"}
              aria-label={muted ? "Unmute microphone" : "Mute microphone"}
            >
              {muted ? <MicOff size={20} /> : <Mic size={20} />}
            </button>
          )}
          {!isError && (
            <button
              onClick={onMinimize}
              className="hidden sm:grid h-12 w-12 place-items-center rounded-full bg-white/10 text-zinc-100 hover:bg-white/15"
              title="Back to transcript"
              aria-label="Back to transcript"
            >
              <Minimize2 size={18} />
            </button>
          )}
          <button
            onClick={onEnd}
            className="grid h-12 w-12 place-items-center rounded-full bg-red-500/90 text-white hover:bg-red-500"
            title={isError ? "Close voice" : "End conversation"}
            aria-label={isError ? "Close voice" : "End conversation"}
          >
            <PhoneOff size={20} />
          </button>
        </div>
      </footer>
    </div>
  );
}
