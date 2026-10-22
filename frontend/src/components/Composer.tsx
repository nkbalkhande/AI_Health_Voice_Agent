import { ArrowUp, AudioLines, Loader2, Mic, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";

type Props = {
  disabled: boolean;
  busy: boolean;
  recording: boolean;
  transcribing: boolean;
  onSend: (text: string) => void;
  onToggleRecording: () => void;
  onVoiceMode: () => void;
  autoFocusKey: string;
};

export default function Composer({
  disabled,
  busy,
  recording,
  transcribing,
  onSend,
  onToggleRecording,
  onVoiceMode,
  autoFocusKey,
}: Props) {
  const [text, setText] = useState("");
  const [seconds, setSeconds] = useState(0);
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    ref.current?.focus();
  }, [autoFocusKey]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }, [text]);

  useEffect(() => {
    if (!recording) {
      setSeconds(0);
      return;
    }
    const timer = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(timer);
  }, [recording]);

  const canSend = !!text.trim() && !busy && !disabled && !recording;

  function submit() {
    if (!canSend) return;
    onSend(text.trim());
    setText("");
  }

  return (
    <div className="safe-bottom mx-auto w-full max-w-3xl px-4">
      <div
        className={`rounded-[28px] border bg-white p-2 shadow-lg shadow-slate-200/50 transition-colors ${
          recording ? "border-red-300" : "border-card-border focus-within:border-primary/40 focus-within:ring-2 focus-within:ring-primary/10"
        }`}
      >
        {recording ? (
          <div className="flex h-[52px] items-center gap-3 px-3">
            <span className="relative flex h-3 w-3">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-500 opacity-60" />
              <span className="relative inline-flex h-3 w-3 rounded-full bg-red-500" />
            </span>
            <div className="flex flex-1 items-center gap-[3px]">
              {Array.from({ length: 32 }).map((_, i) => (
                <span
                  key={i}
                  className="wave-bar w-[3px] rounded-full bg-slate-400"
                  style={{ animationDelay: `${(i * 67) % 900}ms` }}
                />
              ))}
            </div>
            <span className="font-mono text-sm text-slate-600 tabular-nums">
              {Math.floor(seconds / 60)}:{String(seconds % 60).padStart(2, "0")}
            </span>
          </div>
        ) : (
          <textarea
            ref={ref}
            rows={1}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder={transcribing ? "Transcribing your voice…" : "Ask anything, or tap the mic to talk"}
            disabled={disabled || transcribing}
            className="scrollbar-thin block max-h-[220px] w-full resize-none bg-transparent px-3 pt-3 pb-1 text-[15px] leading-relaxed text-slate-800 outline-none placeholder:text-slate-400 disabled:opacity-60"
          />
        )}

        <div className="flex items-center justify-end gap-2 px-1 pt-1">
          <button
            onClick={onToggleRecording}
            disabled={disabled || (busy && !recording) || transcribing}
            title={recording ? "Stop and send" : "Dictate a message"}
            className={`grid h-9 w-9 place-items-center rounded-full transition-colors disabled:opacity-40 ${
              recording
                ? "bg-red-500 text-white hover:bg-red-400"
                : "text-slate-500 hover:bg-slate-100"
            }`}
          >
            {transcribing ? (
              <Loader2 size={18} className="animate-spin" />
            ) : recording ? (
              <Square size={14} fill="currentColor" />
            ) : (
              <Mic size={18} />
            )}
          </button>
          {!text.trim() && !busy && !recording ? (
            <button
              onClick={onVoiceMode}
              disabled={disabled || transcribing}
              title="Start voice conversation"
              className="flex h-9 items-center gap-1.5 rounded-full bg-primary px-3 text-sm font-medium text-white shadow-sm transition-opacity hover:opacity-90 disabled:bg-slate-300 disabled:text-slate-500"
            >
              <AudioLines size={17} strokeWidth={2.25} />
              <span className="hidden sm:inline">Voice</span>
            </button>
          ) : (
            <button
              onClick={submit}
              disabled={!canSend}
              title="Send (Enter)"
              className="grid h-9 w-9 place-items-center rounded-full bg-primary text-white shadow-sm transition-opacity hover:opacity-90 disabled:bg-slate-300 disabled:text-slate-500"
            >
              {busy ? <Loader2 size={18} className="animate-spin" /> : <ArrowUp size={18} strokeWidth={2.5} />}
            </button>
          )}
        </div>
      </div>
      <p className="mt-2 text-center text-[11px] text-slate-400">
        Answers are grounded in your documents and the web. Verify important information.
      </p>
    </div>
  );
}
