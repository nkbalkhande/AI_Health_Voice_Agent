import { Loader2, Play, Square, X } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";

import { listVoices, synthesize, type VoiceOption } from "../api";
import type { MotionPreference, VoiceSettings } from "../voice/settings";

type InputMode = "voice" | "text";

type Props = {
  settings: VoiceSettings;
  onChange: (patch: Partial<VoiceSettings>) => void;
  autoSpeak: boolean;
  onAutoSpeak: (on: boolean) => void;
  inputMode: InputMode;
  onInputMode: (mode: InputMode) => void;
  inCall: boolean;
  onClose: () => void;
};

const PREVIEW_TEXT = "Hi, this is how I'll sound when I answer your health questions.";

export default function SettingsPanel(props: Props) {
  const { settings, onChange, onClose } = props;
  const panelRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
      }
      if (e.key !== "Tab" || !panelRef.current) return;
      const focusable = panelRef.current.querySelectorAll<HTMLElement>(
        "button:not(:disabled), select:not(:disabled), input:not(:disabled), a[href]",
      );
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => {
      window.removeEventListener("keydown", onKey, true);
      previous?.focus?.();
    };
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-[70] flex items-end justify-center sm:items-center">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} aria-hidden="true" />
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-title"
        className="safe-bottom relative flex max-h-[90dvh] w-full max-w-lg flex-col rounded-t-3xl border border-card-border bg-white shadow-2xl sm:rounded-3xl sm:pb-0"
      >
        <div className="flex items-center justify-between px-5 pt-4 pb-2">
          <h2 id="settings-title" className="text-base font-semibold text-slate-800">
            Settings
          </h2>
          <button ref={closeRef} onClick={onClose} className="icon-btn" aria-label="Close settings">
            <X size={18} />
          </button>
        </div>

        <div className="scrollbar-thin space-y-6 overflow-y-auto px-5 pt-2 pb-6">
          {props.inCall && (
            <p className="rounded-xl bg-amber-50 px-3 py-2 text-xs text-amber-700">
              Voice and microphone changes apply from your next conversation.
            </p>
          )}
          <VoicePicker value={settings.voice} onChange={(voice) => onChange({ voice })} />
          <MicPicker value={settings.micId} onChange={(micId) => onChange({ micId })} />

          <Section title="Answers">
            <Toggle
              label="Read typed answers aloud"
              hint="When you type, the answer is spoken as it appears."
              checked={props.autoSpeak}
              onChange={props.onAutoSpeak}
            />
            <Field label="Start in" htmlFor="setting-input-mode">
              <Segmented
                id="setting-input-mode"
                value={props.inputMode}
                options={[
                  ["voice", "Voice"],
                  ["text", "Typing"],
                ]}
                onChange={(v) => props.onInputMode(v as InputMode)}
              />
            </Field>
          </Section>

          <Section title="Motion">
            <Field label="Animations" htmlFor="setting-motion">
              <Segmented
                id="setting-motion"
                value={settings.motion}
                options={[
                  ["system", "Match system"],
                  ["reduce", "Reduced"],
                  ["full", "Full"],
                ]}
                onChange={(v) => onChange({ motion: v as MotionPreference })}
              />
            </Field>
          </Section>

          <Section title="Privacy">
            <ul className="list-disc space-y-1.5 pl-4 text-xs leading-relaxed text-slate-500">
              <li>The microphone is only open during a voice conversation. Ending it, or muting, stops sending audio.</li>
              <li>
                Your audio goes to this app's server, which uses Deepgram to turn speech into text and answers into
                speech. API keys stay on the server.
              </li>
              <li>
                Conversations are saved on the server so you can reopen them from the sidebar. Delete one there to
                remove it.
              </li>
              <li>These settings are stored only in this browser.</li>
            </ul>
          </Section>

          <Section title="Keyboard">
            <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-xs text-slate-500">
              <Kbd keys="Space" />
              <dd>Start talking, or interrupt the agent</dd>
              <Kbd keys="Esc" />
              <dd>Leave the focus view or close a panel</dd>
              <Kbd keys="Ctrl Shift O" />
              <dd>New conversation</dd>
            </dl>
          </Section>
        </div>
      </div>
    </div>
  );
}

function VoicePicker({ value, onChange }: { value: string | null; onChange: (id: string | null) => void }) {
  const [voices, setVoices] = useState<VoiceOption[] | null>(null);
  const [fallback, setFallback] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  const [preview, setPreview] = useState<"idle" | "loading" | "playing">("idle");
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    listVoices()
      .then((data) => {
        setVoices(data.voices);
        setFallback(data.default);
      })
      .catch(() => setFailed(true));
  }, []);

  const stopPreview = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    const audio = audioRef.current;
    if (audio) {
      audio.pause();
      URL.revokeObjectURL(audio.src);
    }
    audioRef.current = null;
    setPreview("idle");
  };
  useEffect(() => stopPreview, []);

  async function playPreview() {
    if (preview !== "idle") return stopPreview();
    const controller = new AbortController();
    abortRef.current = controller;
    setPreview("loading");
    try {
      const blob = await synthesize(PREVIEW_TEXT, controller.signal, value ?? fallback);
      const audio = new Audio(URL.createObjectURL(blob));
      audioRef.current = audio;
      audio.onended = stopPreview;
      await audio.play();
      setPreview("playing");
    } catch (err) {
      if ((err as Error).name !== "AbortError") setFailed(true);
      stopPreview();
    }
  }

  const selected = value ?? fallback ?? "";
  return (
    <Section title="Voice">
      {voices === null && !failed && (
        <p className="flex items-center gap-2 text-xs text-slate-400">
          <Loader2 size={14} className="animate-spin" /> Loading voices…
        </p>
      )}
      {failed && <p className="text-xs text-red-500">Couldn't reach the voice service to list or preview voices.</p>}
      {voices && voices.length === 0 && (
        <p className="text-xs text-slate-400">The server's speech provider has a single configured voice.</p>
      )}
      {voices && voices.length > 0 && (
        <Field label="Agent voice" htmlFor="setting-voice">
          <div className="flex gap-2">
            <select
              id="setting-voice"
              value={selected}
              onChange={(e) => {
                stopPreview();
                onChange(e.target.value === fallback ? null : e.target.value);
              }}
              className="select min-w-0 flex-1"
            >
              {voices.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.label}
                  {v.description ? ` · ${v.description}` : ""}
                  {v.id === fallback ? " · default" : ""}
                </option>
              ))}
            </select>
            <button
              onClick={playPreview}
              className="flex h-11 shrink-0 items-center gap-1.5 rounded-xl border border-card-border bg-slate-50 px-3 text-sm text-slate-700 hover:bg-slate-100"
              aria-label={preview === "idle" ? "Preview voice" : "Stop preview"}
            >
              {preview === "loading" ? (
                <Loader2 size={15} className="animate-spin" />
              ) : preview === "playing" ? (
                <Square size={13} fill="currentColor" />
              ) : (
                <Play size={15} />
              )}
              <span className="hidden sm:inline">{preview === "idle" ? "Preview" : "Stop"}</span>
            </button>
          </div>
        </Field>
      )}
    </Section>
  );
}

function MicPicker({ value, onChange }: { value: string | null; onChange: (id: string | null) => void }) {
  const [mics, setMics] = useState<MediaDeviceInfo[]>([]);
  const supported = !!navigator.mediaDevices?.enumerateDevices;

  useEffect(() => {
    if (!supported) return;
    const load = () =>
      navigator.mediaDevices
        .enumerateDevices()
        .then((all) => setMics(all.filter((d) => d.kind === "audioinput" && d.deviceId !== "default")))
        .catch(() => setMics([]));
    void load();
    navigator.mediaDevices.addEventListener("devicechange", load);
    return () => navigator.mediaDevices.removeEventListener("devicechange", load);
  }, [supported]);

  const unnamed = mics.length > 0 && mics.every((m) => !m.label);
  const known = !value || mics.some((m) => m.deviceId === value);
  return (
    <Section title="Microphone">
      {!supported ? (
        <p className="text-xs text-slate-400">This browser doesn't let pages choose a microphone.</p>
      ) : (
        <>
          <Field label="Input device" htmlFor="setting-mic">
            <select
              id="setting-mic"
              value={known ? (value ?? "") : ""}
              onChange={(e) => onChange(e.target.value || null)}
              className="select w-full"
            >
              <option value="">System default</option>
              {mics.map((m, i) => (
                <option key={m.deviceId || i} value={m.deviceId}>
                  {m.label || `Microphone ${i + 1}`}
                </option>
              ))}
            </select>
          </Field>
          {unnamed && (
            <p className="text-xs text-slate-400">Device names appear after you allow microphone access once.</p>
          )}
          {!known && <p className="text-xs text-amber-600">Your saved microphone isn't connected; the default is used.</p>}
        </>
      )}
    </Section>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="space-y-3">
      <h3 className="text-[11px] font-semibold tracking-[0.14em] uppercase text-slate-400">{title}</h3>
      {children}
    </section>
  );
}

function Field({ label, htmlFor, children }: { label: string; htmlFor: string; children: ReactNode }) {
  return (
    <div className="space-y-1.5">
      <label id={`${htmlFor}-label`} htmlFor={htmlFor} className="block text-sm text-slate-700">
        {label}
      </label>
      {children}
    </div>
  );
}

function Toggle({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string;
  hint: string;
  checked: boolean;
  onChange: (on: boolean) => void;
}) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="flex w-full items-center justify-between gap-4 rounded-xl text-left"
    >
      <span>
        <span className="block text-sm text-slate-700">{label}</span>
        <span className="block text-xs text-slate-400">{hint}</span>
      </span>
      <span
        aria-hidden="true"
        className={`relative h-6 w-11 shrink-0 rounded-full transition-colors ${checked ? "bg-primary" : "bg-slate-200"}`}
      >
        <span
          className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow-sm transition-transform ${
            checked ? "translate-x-[22px]" : "translate-x-0.5"
          }`}
        />
      </span>
    </button>
  );
}

function Segmented({
  id,
  value,
  options,
  onChange,
}: {
  id: string;
  value: string;
  options: [string, string][];
  onChange: (value: string) => void;
}) {
  return (
    <div id={id} role="radiogroup" aria-labelledby={`${id}-label`} className="grid auto-cols-fr grid-flow-col gap-1 rounded-xl bg-slate-100 p-1">
      {options.map(([v, label]) => (
        <button
          key={v}
          role="radio"
          aria-checked={value === v}
          onClick={() => onChange(v)}
          className={`h-9 rounded-lg px-2 text-sm transition-colors ${
            value === v ? "bg-white text-slate-800 shadow-sm" : "text-slate-500 hover:text-slate-700"
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

function Kbd({ keys }: { keys: string }) {
  return (
    <dt>
      <kbd className="rounded border border-card-border bg-slate-50 px-1.5 py-0.5 font-sans text-[11px] text-slate-600">
        {keys}
      </kbd>
    </dt>
  );
}
