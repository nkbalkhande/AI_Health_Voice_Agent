export type Citation = {
  source?: string;
  page_number?: string | number;
  url?: string;
  title?: string;
  snippet?: string;
  sheet_name?: string;
  source_type?: string;
};

export type Fact = { key: string; value: string };

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  createdAt?: string;
  viaVoice?: boolean;
  streaming?: boolean;
  /** While the answer is being spoken, only this many characters are shown. */
  revealed?: number;
  audioBase64?: string;
  error?: boolean;
  /** The user talked over this answer, so it stops mid-way. */
  interrupted?: boolean;
};

export type ThreadSummary = {
  thread_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
};

export type ConnectionState = "connecting" | "open" | "closed";

export type PipelinePhase = "idle" | "understanding" | "searching" | "complete" | "skipped" | "error";

export type PipelineState = {
  rag: PipelinePhase;
  web: PipelinePhase;
  synthesis: "idle" | "active" | "complete";
  needsRetrieval: boolean;
  intent: string;
  query: string;
  documentResults: Array<{ content: string; score: number; source?: string; page_number?: string | number }>;
  webResults: Array<{ title: string; url: string; snippet: string }>;
  citations: Citation[];
};

export const PIPELINE_INITIAL: PipelineState = {
  rag: "idle",
  web: "idle",
  synthesis: "idle",
  needsRetrieval: true,
  intent: "",
  query: "",
  documentResults: [],
  webResults: [],
  citations: [],
};
