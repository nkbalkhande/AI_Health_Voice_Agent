import { AudioLines, Keyboard } from "lucide-react";

import type { VoiceError } from "../voice/errors";
import type { VoiceStatus } from "../voice/useVoiceSession";
import Orb from "./Orb";
import { CallControls } from "./VoiceStage";
import { STATUS_HINTS, STATUS_LABELS, canInterrupt, liveCaption } from "./voiceText";

type Props = {
  status: VoiceStatus;
  userCaption: string;
  agentCaption: string;
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
};

export default function VoiceDock(props: Props) {
  const { status, level, disabled, error } = props;
  const caption = liveCaption(status, props.userCaption, props.agentCaption);
  const interruptible = canInterrupt(status);

  if (status === "idle") {
    return (
      <div className="safe-bottom mx-auto flex w-full max-w-3xl items-center justify-center gap-3 px-4 pt-2">
        <button
          onClick={props.onType}
          className="icon-btn h-12 w-12 rounded-full bg-slate-100"
          title="Type instead"
          aria-label="Type instead"
        >
          <Keyboard size={20} />
        </button>
        <button
          onClick={props.onStart}
          disabled={disabled}
          className="flex h-14 items-center gap-3 rounded-full bg-primary pr-6 pl-2 font-medium text-white shadow-lg transition-colors hover:bg-primary/90 disabled:opacity-40"
        >
          <span className="grid h-10 w-10 place-items-center rounded-full bg-white/20">
            <AudioLines size={20} strokeWidth={2.25} />
          </span>
          Tap to talk
          <kbd className="hidden rounded border border-white/30 px-1.5 text-[11px] font-medium text-white/70 md:inline">
            Space
          </kbd>
        </button>
      </div>
    );
  }

  const isError = status === "error";
  return (
    <div className="safe-bottom mx-auto w-full max-w-3xl px-3 pt-2 sm:px-4">
      <div
        className={`flex items-center gap-3 rounded-[28px] border bg-white py-2 pr-2 pl-3 shadow-lg ${
          isError ? "border-red-200" : "border-card-border"
        }`}
      >
        <Orb
          status={status}
          level={level}
          size="sm"
          label={interruptible ? "Interrupt" : STATUS_LABELS[status]}
          onClick={interruptible ? props.onInterrupt : undefined}
        />
        <div className="min-w-0 flex-1">
          <p className="flex items-center gap-2 text-sm font-medium text-slate-800">
            {isError && error ? error.title : STATUS_LABELS[status]}
            {(status === "listening" || status === "transcribing") && (
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-primary" />
            )}
          </p>
          <p
            className={`text-xs ${isError ? "line-clamp-2 text-red-500" : "truncate"} ${
              !isError && caption.agent ? "text-slate-600" : !isError ? "text-slate-400" : ""
            }`}
          >
            {isError
              ? error?.detail
              : caption.text
                ? caption.agent
                  ? caption.text
                  : `"${caption.text}"`
                : STATUS_HINTS[status]}
          </p>
        </div>
        <CallControls {...props} compact />
      </div>
    </div>
  );
}
