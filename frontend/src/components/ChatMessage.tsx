import {
  AlertCircle,
  Check,
  ChevronDown,
  Copy,
  FileText,
  Globe,
  Loader2,
  Mic,
  Square,
  Volume2,
} from "lucide-react";
import { Children, memo, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import type { Citation, Message } from "../types";
import Logo from "./Logo";

type Props = {
  message: Message;
  speaking: boolean;
  liveSpeaking?: boolean;
  loadingSpeech: boolean;
  onSpeak: (message: Message) => void;
  onStopSpeak: () => void;
};

function basename(path: string) {
  return path.split(/[\\/]/).pop() || path;
}

function hostname(url: string) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

const CITATION = /\[([^[\]]{1,160})\]/g;

function sourceLabel(raw: string) {
  const name = (raw.split(/[\\/]/).pop() || raw).trim().replace(/\.pdf$/i, "");
  if (name.length <= 32) return name;
  return `${name.slice(0, 24)}…`;
}

function withCitationChips(children: ReactNode): ReactNode {
  return Children.map(children, (child) => {
    if (typeof child !== "string" || !child.includes("[")) return child;
    const parts: ReactNode[] = [];
    let last = 0;
    for (const match of child.matchAll(CITATION)) {
      const index = match.index ?? 0;
      if (index > last) parts.push(child.slice(last, index));
      const full = match[1];
      const tight = /^[.,;:)]/.test(child.slice(index + match[0].length));
      parts.push(
        <span
          key={`${index}-${full}`}
          title={full}
          className={`inline-flex max-w-[16rem] items-center truncate rounded-md bg-slate-100 px-1.5 py-0.5 align-baseline text-[11px] font-medium text-slate-500 ${tight ? "ml-1" : "mx-1"}`}
        >
          {sourceLabel(full)}
        </span>,
      );
      last = index + match[0].length;
    }
    if (!parts.length) return child;
    if (last < child.length) parts.push(child.slice(last));
    return parts;
  });
}

function renderRichText(children: ReactNode) {
  return withCitationChips(children);
}

function dedupe(citations: Citation[]) {
  const seen = new Set<string>();
  return citations.filter((c) => {
    const key = c.source_type === "web" ? (c.url ?? c.source ?? "") : `${c.source}|${c.page_number}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function ChatMessage({
  message,
  speaking,
  liveSpeaking = false,
  loadingSpeech,
  onSpeak,
  onStopSpeak,
}: Props) {
  const [copied, setCopied] = useState(false);
  const voiceActive = speaking || liveSpeaking;

  if (message.role === "user") {
    return (
      <div className="group flex justify-end">
        <div className="max-w-[85%] rounded-2xl bg-primary px-5 py-2.5 text-[15px] leading-relaxed whitespace-pre-wrap text-white shadow-sm md:max-w-[70%]">
          {message.viaVoice && (
            <span className="mb-1 flex items-center gap-1 text-[11px] font-medium tracking-wide text-white/70 uppercase">
              <Mic size={11} /> You said
            </span>
          )}
          {message.content}
        </div>
      </div>
    );
  }

  const revealing = message.revealed !== undefined;
  const live = message.streaming || revealing;
  const shown = revealing ? message.content.slice(0, message.revealed) : message.content;
  const empty = !shown && live;
  const cutOff = !live && (message.interrupted || message.content.endsWith(" …"));

  async function copy() {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div className="group flex gap-4">
      <div className={`mt-0.5 shrink-0 ${voiceActive ? "orb-breathe" : ""}`} aria-hidden="true">
        <Logo size={32} />
      </div>
      <div className="min-w-0 flex-1">
        <div className="mb-1 flex h-5 items-center gap-2 text-xs font-medium text-slate-400">
          HealthMate AI
          {cutOff && (
            <span className="rounded-full bg-amber-50 px-2 py-px text-[11px] font-medium text-amber-600">
              Interrupted
            </span>
          )}
          {voiceActive && (
            <span className="flex items-center gap-1.5 text-primary">
              <span className="eq" aria-hidden="true">
                <span />
                <span />
                <span />
              </span>
              Speaking
            </span>
          )}
        </div>
        {empty ? (
          <div className="flex h-8 items-center gap-1.5">
            <span className="typing-dot" />
            <span className="typing-dot [animation-delay:150ms]" />
            <span className="typing-dot [animation-delay:300ms]" />
          </div>
        ) : message.error ? (
          <div className="flex items-start gap-2 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
            <AlertCircle size={16} className="mt-0.5 shrink-0" />
            {message.content}
          </div>
        ) : (
          <div className={`markdown ${live ? "streaming" : ""}`}>
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                a: (props) => <a {...props} target="_blank" rel="noreferrer" />,
                p: ({ children }) => <p>{renderRichText(children)}</p>,
                li: ({ children }) => <li>{renderRichText(children)}</li>,
              }}
            >
              {shown}
            </ReactMarkdown>
          </div>
        )}

        {live && speaking && (
          <button
            onClick={onStopSpeak}
            className="mt-2 flex items-center gap-1.5 rounded-full bg-primary-light px-3 py-1 text-xs text-primary hover:bg-primary/10"
            title="Stop speaking"
          >
            <Square size={10} fill="currentColor" /> Speaking — stop
          </button>
        )}

        {!live && !message.error && message.content && (
          <>
            {message.citations && message.citations.length > 0 && (
              <Sources citations={dedupe(message.citations)} />
            )}
            <div
              className={`mt-2 flex items-center gap-0.5 text-slate-400 transition-opacity ${
                speaking ? "" : "md:opacity-0 md:group-hover:opacity-100 md:focus-within:opacity-100"
              }`}
            >
              <button onClick={copy} className="icon-btn" title="Copy">
                {copied ? <Check size={16} className="text-emerald-500" /> : <Copy size={16} />}
              </button>
              {speaking ? (
                <button onClick={onStopSpeak} className="icon-btn text-primary" title="Stop">
                  <Square size={15} fill="currentColor" />
                </button>
              ) : (
                <button
                  onClick={() => onSpeak(message)}
                  className="icon-btn"
                  title="Read aloud"
                  disabled={loadingSpeech}
                >
                  {loadingSpeech ? (
                    <Loader2 size={16} className="animate-spin" />
                  ) : (
                    <Volume2 size={16} />
                  )}
                </button>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function Sources({ citations }: { citations: Citation[] }) {
  const [open, setOpen] = useState(false);
  const docs = citations.filter((c) => c.source_type !== "web");
  const web = citations.filter((c) => c.source_type === "web");

  return (
    <div className="mt-3">
      <button
        onClick={() => setOpen(!open)}
        className="inline-flex items-center gap-2 rounded-full border border-card-border bg-white px-3 py-1.5 text-xs text-slate-600 shadow-sm hover:bg-slate-50"
      >
        <span className="flex -space-x-1">
          {docs.length > 0 && (
            <span className="grid h-5 w-5 place-items-center rounded-full bg-rag-light ring-2 ring-white">
              <FileText size={11} className="text-sky-600" />
            </span>
          )}
          {web.length > 0 && (
            <span className="grid h-5 w-5 place-items-center rounded-full bg-websearch-light ring-2 ring-white">
              <Globe size={11} className="text-violet-600" />
            </span>
          )}
        </span>
        {citations.length} source{citations.length === 1 ? "" : "s"}
        <ChevronDown size={14} className={`transition-transform ${open ? "rotate-180" : ""}`} />
      </button>

      {open && (
        <div className="mt-2 grid gap-2 sm:grid-cols-2">
          {citations.map((c, i) => {
            const isWeb = c.source_type === "web";
            const label = isWeb ? (c.source ?? c.url ?? "Web page") : basename(c.source ?? "Document");
            const meta = isWeb
              ? c.url && hostname(c.url)
              : [c.sheet_name, c.page_number && c.page_number !== "N/A" ? `p. ${c.page_number}` : null]
                  .filter(Boolean)
                  .join(" · ");
            const body = (
              <>
                <div className="flex items-center gap-2">
                  {isWeb ? (
                    <Globe size={13} className="shrink-0 text-violet-500" />
                  ) : (
                    <FileText size={13} className="shrink-0 text-sky-500" />
                  )}
                  <span className="truncate text-[13px] font-medium text-slate-700">{label}</span>
                </div>
                {meta && <p className="mt-0.5 truncate text-[11px] text-slate-400">{meta}</p>}
                {c.snippet && (
                  <p className="mt-1.5 line-clamp-2 text-xs leading-relaxed text-slate-500">
                    {c.snippet}
                  </p>
                )}
              </>
            );
            const cls =
              "block rounded-xl border border-card-border bg-white p-3 text-left transition-colors hover:bg-slate-50 shadow-sm";
            return isWeb && c.url ? (
              <a key={i} href={c.url} target="_blank" rel="noreferrer" className={cls}>
                {body}
              </a>
            ) : (
              <div key={i} className={cls}>
                {body}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default memo(ChatMessage);
