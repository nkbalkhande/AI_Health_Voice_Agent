import {
  BookOpen,
  CheckCircle2,
  ChevronDown,
  Database,
  FileText,
  Globe,
  Loader2,
  Search,
  Sparkles,
  X,
  XCircle,
} from "lucide-react";
import { memo, useMemo, useState } from "react";

import type { Citation, PipelinePhase, PipelineState } from "../types";

type Props = {
  pipeline: PipelineState;
  onClose: () => void;
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

const PHASE_LABELS: Record<PipelinePhase, string> = {
  idle: "Waiting",
  understanding: "Understanding query",
  searching: "Searching…",
  complete: "Complete",
  skipped: "Skipped",
  error: "Error",
};

type SourceFilter = "all" | "rag" | "web";

function PipelinePanel({ pipeline, onClose }: Props) {
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>("all");
  const [sourcesOpen, setSourcesOpen] = useState(true);

  const isActive =
    pipeline.rag !== "idle" || pipeline.web !== "idle" || pipeline.synthesis !== "idle";

  const ragDone = pipeline.rag === "complete";
  const webDone = pipeline.web === "complete";
  const synthDone = pipeline.synthesis === "complete";

  const docCitations = useMemo(
    () => pipeline.citations.filter((c) => c.source_type !== "web"),
    [pipeline.citations],
  );
  const webCitations = useMemo(
    () => pipeline.citations.filter((c) => c.source_type === "web"),
    [pipeline.citations],
  );

  const filteredCitations = useMemo(() => {
    if (sourceFilter === "rag") return docCitations;
    if (sourceFilter === "web") return webCitations;
    return pipeline.citations;
  }, [sourceFilter, pipeline.citations, docCitations, webCitations]);

  const highlights = useMemo(() => {
    const tags: { label: string; color: string }[] = [];
    if (docCitations.length > 0)
      tags.push({ label: "Knowledge Base Insights", color: "bg-rag-light text-sky-700" });
    if (webCitations.length > 0)
      tags.push({ label: "Recent Web Information", color: "bg-websearch-light text-violet-700" });
    if (pipeline.citations.length > 2)
      tags.push({ label: "Multiple Sources", color: "bg-emerald-50 text-emerald-700" });
    if (pipeline.intent && pipeline.intent !== "greeting" && pipeline.intent !== "general")
      tags.push({ label: "Research Query", color: "bg-amber-50 text-amber-700" });
    return tags;
  }, [pipeline.citations, docCitations, webCitations, pipeline.intent]);

  return (
    <aside className="flex h-full w-[340px] flex-col border-l border-card-border bg-white">
      <div className="flex items-center justify-between border-b border-card-border px-4 py-3">
        <div className="flex items-center gap-2 text-sm font-semibold text-slate-800">
          <Sparkles size={16} className="text-primary" />
          AI Research
        </div>
        <button onClick={onClose} className="icon-btn" title="Close panel" aria-label="Close research panel">
          <X size={16} />
        </button>
      </div>

      <div className="scrollbar-thin flex-1 overflow-y-auto">
        {/* Section 1: Pipeline Processing */}
        <div className="border-b border-card-border p-4">
          <h3 className="mb-3 text-[11px] font-semibold tracking-[0.14em] uppercase text-slate-400">
            Agent Processing
          </h3>
          <div className="space-y-3">
            {/* RAG Pipeline Card */}
            <PipelineCard
              label="Knowledge Base"
              icon={<Database size={14} />}
              phase={pipeline.rag}
              accent="rag"
              resultCount={pipeline.documentResults.length}
              resultLabel="documents"
            />

            {/* Web Search Pipeline Card */}
            <PipelineCard
              label="Web Search"
              icon={<Globe size={14} />}
              phase={pipeline.web}
              accent="web"
              resultCount={pipeline.webResults.length}
              resultLabel="sources"
            />

            {/* Synthesis Indicator */}
            {isActive && (
              <SynthesisIndicator
                ragDone={ragDone}
                webDone={webDone}
                synthDone={synthDone}
                synthesizing={pipeline.synthesis === "active"}
              />
            )}
          </div>
        </div>

        {/* Section 2: Sources */}
        {pipeline.citations.length > 0 && (
          <div className="border-b border-card-border p-4">
            <button
              onClick={() => setSourcesOpen(!sourcesOpen)}
              className="mb-3 flex w-full items-center justify-between"
            >
              <h3 className="text-[11px] font-semibold tracking-[0.14em] uppercase text-slate-400">
                Sources Used ({pipeline.citations.length})
              </h3>
              <ChevronDown
                size={14}
                className={`text-slate-400 transition-transform ${sourcesOpen ? "rotate-180" : ""}`}
              />
            </button>

            {sourcesOpen && (
              <>
                {/* Filters */}
                <div className="mb-3 flex gap-1 rounded-lg bg-slate-100 p-1">
                  {(
                    [
                      ["all", `All (${pipeline.citations.length})`],
                      ["rag", `KB (${docCitations.length})`],
                      ["web", `Web (${webCitations.length})`],
                    ] as const
                  ).map(([key, label]) => (
                    <button
                      key={key}
                      onClick={() => setSourceFilter(key)}
                      className={`flex-1 rounded-md px-2 py-1.5 text-xs font-medium transition-colors ${
                        sourceFilter === key
                          ? "bg-white text-slate-800 shadow-sm"
                          : "text-slate-500 hover:text-slate-700"
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>

                {/* Source items */}
                <div className="space-y-2">
                  {filteredCitations.map((c, i) => (
                    <SourceItem key={i} citation={c} />
                  ))}
                </div>
              </>
            )}
          </div>
        )}

        {/* Section 3: Answer Highlights */}
        {highlights.length > 0 && synthDone && (
          <div className="p-4">
            <h3 className="mb-3 text-[11px] font-semibold tracking-[0.14em] uppercase text-slate-400">
              Answer Highlights
            </h3>
            <div className="flex flex-wrap gap-2">
              {highlights.map((h) => (
                <span
                  key={h.label}
                  className={`rounded-full px-3 py-1 text-xs font-medium ${h.color}`}
                >
                  {h.label}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Empty state */}
        {!isActive && pipeline.citations.length === 0 && (
          <div className="p-6 text-center">
            <div className="mx-auto mb-3 grid h-12 w-12 place-items-center rounded-full bg-primary-light">
              <Search size={20} className="text-primary" />
            </div>
            <p className="text-sm font-medium text-slate-700">Research Panel</p>
            <p className="mt-1 text-xs leading-relaxed text-slate-400">
              When you ask a question, this panel shows the AI's research process — searching your
              knowledge base and the web in parallel, then synthesizing sources.
            </p>
          </div>
        )}
      </div>
    </aside>
  );
}

/* ─── Sub-components ─── */

function PipelineCard({
  label,
  icon,
  phase,
  accent,
  resultCount,
  resultLabel,
}: {
  label: string;
  icon: React.ReactNode;
  phase: PipelinePhase;
  accent: "rag" | "web";
  resultCount: number;
  resultLabel: string;
}) {
  const isRag = accent === "rag";
  const accentBg = isRag ? "bg-rag-light" : "bg-websearch-light";
  const accentText = isRag ? "text-sky-600" : "text-violet-600";
  const accentBorder = isRag ? "border-sky-200" : "border-violet-200";
  const accentDot = isRag ? "bg-rag" : "bg-websearch";

  const isActive = phase === "understanding" || phase === "searching";
  const isDone = phase === "complete";
  const isSkipped = phase === "skipped";
  const isError = phase === "error";

  return (
    <div
      className={`rounded-xl border p-3 transition-colors ${
        isActive
          ? `${accentBorder} ${accentBg}`
          : isDone
            ? "border-emerald-200 bg-emerald-50/50"
            : isError
              ? "border-red-200 bg-red-50/50"
              : "border-card-border bg-slate-50/50"
      }`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`${accentText}`}>{icon}</span>
          <span className="text-sm font-medium text-slate-700">{label}</span>
        </div>
        <PhaseIndicator phase={phase} accentDot={accentDot} />
      </div>
      {phase !== "idle" && (
        <p className="mt-1.5 text-xs text-slate-500">
          {isActive && PHASE_LABELS[phase]}
          {isDone && `${resultCount} ${resultLabel} found`}
          {isSkipped && "Not needed for this query"}
          {isError && "Failed — other sources may still be available"}
        </p>
      )}
    </div>
  );
}

function PhaseIndicator({ phase, accentDot }: { phase: PipelinePhase; accentDot: string }) {
  if (phase === "idle") return null;
  if (phase === "understanding" || phase === "searching")
    return <Loader2 size={14} className={`animate-spin ${accentDot === "bg-rag" ? "text-rag" : "text-websearch"}`} />;
  if (phase === "complete")
    return <CheckCircle2 size={14} className="text-emerald-500" />;
  if (phase === "skipped")
    return <span className="text-xs text-slate-400">—</span>;
  if (phase === "error")
    return <XCircle size={14} className="text-red-400" />;
  return null;
}

function SynthesisIndicator({
  ragDone,
  webDone,
  synthDone,
  synthesizing,
}: {
  ragDone: boolean;
  webDone: boolean;
  synthDone: boolean;
  synthesizing: boolean;
}) {
  const bothDone = ragDone && webDone;
  return (
    <div
      className={`flex items-center gap-2 rounded-xl border p-3 transition-colors ${
        synthDone
          ? "border-emerald-200 bg-emerald-50/50"
          : synthesizing
            ? "border-primary/20 bg-primary-light"
            : "border-card-border bg-slate-50/50"
      }`}
    >
      <Sparkles
        size={14}
        className={synthDone ? "text-emerald-500" : synthesizing ? "text-primary" : "text-slate-400"}
      />
      <span className="flex-1 text-sm text-slate-600">
        {synthDone
          ? "Answer synthesized"
          : synthesizing
            ? "Combining research…"
            : bothDone
              ? "Ready to synthesize"
              : "Waiting for pipelines…"}
      </span>
      {synthesizing && <Loader2 size={14} className="animate-spin text-primary" />}
      {synthDone && <CheckCircle2 size={14} className="text-emerald-500" />}
    </div>
  );
}

function SourceItem({ citation }: { citation: Citation }) {
  const isWeb = citation.source_type === "web";
  const label = isWeb
    ? citation.title || citation.source || citation.url || "Web page"
    : basename(citation.source ?? "Document");
  const meta = isWeb
    ? citation.url && hostname(citation.url)
    : [
        citation.sheet_name,
        citation.page_number && citation.page_number !== "N/A" ? `p. ${citation.page_number}` : null,
      ]
        .filter(Boolean)
        .join(" · ");

  const content = (
    <>
      <div className="flex items-center gap-2">
        <span
          className={`grid h-6 w-6 shrink-0 place-items-center rounded-md ${
            isWeb ? "bg-websearch-light" : "bg-rag-light"
          }`}
        >
          {isWeb ? (
            <Globe size={12} className="text-violet-600" />
          ) : (
            <FileText size={12} className="text-sky-600" />
          )}
        </span>
        <div className="min-w-0 flex-1">
          <p className="truncate text-[13px] font-medium text-slate-700">{label}</p>
          {meta && <p className="truncate text-[11px] text-slate-400">{meta}</p>}
        </div>
        <span
          className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${
            isWeb ? "bg-websearch-light text-violet-600" : "bg-rag-light text-sky-600"
          }`}
        >
          {isWeb ? "Web" : "KB"}
        </span>
      </div>
      {citation.snippet && (
        <p className="mt-2 line-clamp-2 text-xs leading-relaxed text-slate-500">{citation.snippet}</p>
      )}
    </>
  );

  const cls =
    "block rounded-xl border border-card-border bg-white p-3 text-left transition-colors hover:bg-slate-50";
  return isWeb && citation.url ? (
    <a href={citation.url} target="_blank" rel="noreferrer" className={cls}>
      {content}
    </a>
  ) : (
    <div className={cls}>{content}</div>
  );
}

export default memo(PipelinePanel);
