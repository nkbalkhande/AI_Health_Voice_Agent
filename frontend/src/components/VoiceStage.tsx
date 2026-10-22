import {
  BookOpenCheck,
  Hand,
  Keyboard,
  Maximize2,
  Mic,
  MicOff,
  Minimize2,
  PhoneOff,
  RotateCcw,
  Zap,
} from "lucide-react";

import type { VoiceError } from "../voice/errors";
import type { VoiceMetrics, VoiceStatus } from "../voice/useVoiceSession";
import Orb from "./Orb";
import { STATUS_HINTS, STATUS_LABELS, canInterrupt, liveCaption, seconds } from "./voiceText";

const SUGGESTIONS = [
  "What causes fever in children?",
  "When is a fever dangerous?",
  "Symptoms of vitamin D deficiency",
  "How is dengue treated?",
];

const FEATURES = [
  { icon: Zap, text: "Real-time, sentence by sentence" },
  { icon: Hand, text: "Interrupt anytime" },
  { icon: BookOpenCheck, text: "Cited medical sources" },
];

type Props = {
  status: VoiceStatus;
  userCaption: string;
  agentCaption: string;
  metrics: VoiceMetrics | null;
  muted: boolean;
  error: VoiceError | null;
  level: () => number;
  disabled: boolean;
  onStart: () => void;
  onInterrupt: () => void;
  onToggleMute: () => void;
  onEnd: () => void;
  onExpand: () => void;
  onType: () => void;
  onAsk: (prompt: string) => void;
};

export default function VoiceStage(props: Props) {
  const { status, metrics, error, level, disabled } = props;
  const idle = status === "idle";
  const caption = liveCaption(status, props.userCaption, props.agentCaption);
  const interruptible = canInterrupt(status);

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col items-center justify-center px-4 pb-4">
      <Orb
        status={status}
        level={level}
        label={idle ? "Start talking" : interruptible ? "Interrupt" : STATUS_LABELS[status]}
        onClick={idle ? (disabled ? undefined : props.onStart) : interruptible ? props.onInterrupt : undefined}
      >
        {idle && <Mic size={40} strokeWidth={1.75} className="text-white/80" />}
      </Orb>

      <div className="-mt-2 text-center">
        <h1 className="text-2xl font-semibold tracking-tight text-slate-800 md:text-3xl">
          {idle ? "Hi, I'm your health assistant" : status === "error" && error ? error.title : STATUS_LABELS[status]}
        </h1>
        <p className={`mx-auto mt-2 max-w-md text-sm ${status === "error" ? "text-red-500" : "text-slate-500"}`}>
          {status === "error"
            ? error?.detail
            : idle
              ? "Tap the orb or press Space, then ask your question out loud."
              : STATUS_HINTS[status]}
        </p>
      </div>

      {idle ? (
        <>
          <div className="mt-6 flex flex-wrap justify-center gap-2">
            {FEATURES.map(({ icon: Icon, text }) => (
              <span
                key={text}
                className="flex items-center gap-1.5 rounded-full border border-card-border bg-white px-3 py-1 text-xs text-slate-500 shadow-sm"
              >
                <Icon size={13} className="text-primary" /> {text}
              </span>
            ))}
          </div>
          <div className="mt-8 w-full max-w-xl">
            <p className="mb-2 text-center text-[11px] font-medium tracking-[0.18em] uppercase text-slate-400">
              Try asking
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {SUGGESTIONS.map((prompt) => (
                <button
                  key={prompt}
                  onClick={() => props.onAsk(prompt)}
                  disabled={disabled}
                  className="rounded-full border border-card-border bg-white px-3.5 py-1.5 text-sm text-slate-600 shadow-sm transition-colors hover:border-primary/30 hover:bg-primary-light hover:text-primary disabled:opacity-40"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        </>
      ) : (
        <>
          <div className="mt-6 flex min-h-[6.5rem] w-full items-start justify-center">
            {caption.text && status !== "error" && (
              <p
                key={caption.agent ? "agent" : "user"}
                className={`caption-in line-clamp-4 max-w-xl text-center text-xl leading-relaxed md:text-2xl ${
                  caption.agent ? "text-slate-800" : "text-slate-500"
                }`}
              >
                {caption.agent ? caption.text : `"${caption.text}"`}
              </p>
            )}
          </div>
          {metrics && <MetricChips metrics={metrics} />}
          <CallControls {...props} />
        </>
      )}
    </div>
  );
}

export function MetricChips({ metrics }: { metrics: VoiceMetrics }) {
  return (
    <div className="flex flex-wrap justify-center gap-2 text-xs">
      {(
        [
          ["Understood + searched", metrics.retrieval_ms],
          ["First word", metrics.first_token_ms],
          ["First audio", metrics.first_audio_ms],
        ] as const
      ).map(([label, ms]) => (
        <span key={label} className="rounded-full border border-card-border bg-white px-3 py-1 text-slate-500 shadow-sm">
          {label} <span className="font-mono text-slate-700">{seconds(ms)}</span>
        </span>
      ))}
    </div>
  );
}

export function CallControls({
  status,
  muted,
  error,
  onToggleMute,
  onEnd,
  onExpand,
  onStart,
  onType,
  compact = false,
  expanded = false,
}: Pick<Props, "status" | "muted" | "error" | "onToggleMute" | "onEnd" | "onExpand" | "onStart"> & {
  onType?: () => void;
  compact?: boolean;
  expanded?: boolean;
}) {
  const size = compact ? "h-11 w-11" : "h-12 w-12";
  const icon = compact ? 18 : 20;
  const pill = `flex items-center gap-2 rounded-full px-4 text-sm font-medium ${compact ? "h-11" : "h-12"}`;
  const starting = status === "connecting" || status === "requesting_permission";
  return (
    <div className={`flex items-center ${compact ? "gap-1.5" : "mt-8 gap-3"}`}>
      {status === "error" ? (
        error?.retryable !== false ? (
          <button onClick={onStart} className={`${pill} bg-primary text-white shadow-sm hover:opacity-90`}>
            <RotateCcw size={16} /> Try again
          </button>
        ) : (
          onType && (
            <button onClick={onType} className={`${pill} bg-primary text-white shadow-sm hover:opacity-90`}>
              <Keyboard size={16} /> Type instead
            </button>
          )
        )
      ) : (
        <button
          onClick={onToggleMute}
          disabled={starting}
          aria-pressed={muted}
          className={`grid place-items-center rounded-full transition-colors disabled:opacity-40 ${size} ${
            muted ? "bg-red-500 text-white" : "bg-slate-100 text-slate-700 hover:bg-slate-200"
          }`}
          title={muted ? "Unmute microphone" : "Mute microphone"}
          aria-label={muted ? "Unmute microphone" : "Mute microphone"}
        >
          {muted ? <MicOff size={icon} /> : <Mic size={icon} />}
        </button>
      )}
      {status !== "error" && (
        <button
          onClick={onExpand}
          className={`hidden place-items-center rounded-full bg-slate-100 text-slate-700 hover:bg-slate-200 sm:grid ${size}`}
          title={expanded ? "Back to transcript" : "Focus view"}
          aria-label={expanded ? "Back to transcript" : "Focus view"}
        >
          {expanded ? <Minimize2 size={icon - 2} /> : <Maximize2 size={icon - 2} />}
        </button>
      )}
      <button
        onClick={onEnd}
        className={`grid place-items-center rounded-full bg-red-500 text-white hover:bg-red-600 shadow-sm ${size}`}
        title={status === "error" ? "Close voice" : "End conversation"}
        aria-label={status === "error" ? "Close voice" : "End conversation"}
      >
        <PhoneOff size={icon} />
      </button>
    </div>
  );
}
