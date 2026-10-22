import { Brain, X } from "lucide-react";

import type { Fact } from "../types";

type Props = {
  threadId: string | null;
  summary: string;
  facts: Fact[];
  onClose: () => void;
};

export default function MemoryPanel({ threadId, summary, facts, onClose }: Props) {
  const empty = !summary && facts.length === 0;
  return (
    <aside className="flex h-full w-[320px] flex-col border-l border-card-border bg-white">
      <div className="flex items-center justify-between border-b border-card-border px-4 py-3">
        <div className="flex items-center gap-2 text-sm font-semibold text-slate-800">
          <Brain size={16} className="text-primary" />
          Session Memory
        </div>
        <button onClick={onClose} className="icon-btn" title="Close">
          <X size={16} />
        </button>
      </div>

      <div className="scrollbar-thin flex-1 space-y-5 overflow-y-auto px-4 pb-4 pt-4">
        {empty ? (
          <div className="rounded-xl border border-dashed border-card-border p-4 text-sm leading-relaxed text-slate-400">
            {threadId
              ? "Nothing remembered yet. Facts and a running summary appear here as the conversation goes on."
              : "Start a conversation. Each chat keeps its own summary and facts, separate from the others."}
          </div>
        ) : (
          <>
            {summary && (
              <section>
                <h3 className="mb-2 text-[11px] font-semibold tracking-[0.14em] uppercase text-slate-400">
                  Summary
                </h3>
                <p className="text-sm leading-relaxed text-slate-600">{summary}</p>
              </section>
            )}
            {facts.length > 0 && (
              <section>
                <h3 className="mb-2 text-[11px] font-semibold tracking-[0.14em] uppercase text-slate-400">
                  Facts ({facts.length})
                </h3>
                <ul className="space-y-1.5">
                  {facts.map((f) => (
                    <li
                      key={f.key}
                      className="rounded-lg border border-card-border bg-slate-50 px-3 py-2 text-sm"
                    >
                      <span className="block text-[11px] text-slate-400">{f.key}</span>
                      <span className="text-slate-700">{f.value}</span>
                    </li>
                  ))}
                </ul>
              </section>
            )}
          </>
        )}

        <section className="rounded-xl bg-slate-50 p-3 text-[11px] leading-relaxed text-slate-400">
          Deepgram transcribes speech · LangGraph resolves follow-ups · documents and web search run
          in parallel · Deepgram Aura speaks the answer · PostgreSQL keeps every session.
          {threadId && <span className="mt-2 block font-mono break-all text-slate-300">{threadId}</span>}
        </section>
      </div>
    </aside>
  );
}
