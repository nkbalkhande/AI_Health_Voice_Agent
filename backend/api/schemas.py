from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    filters: dict | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    retrieved_chunks: list[dict]
    model: str
    latency_ms: float
    # R-34: QueryResult already computes these; dropping them left API clients
    # unable to tell a refusal from an error from a truncated answer.
    generation_status: str | None = None
    finish_reason: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    thread_id: str | None = Field(
        default=None,
        description="Existing conversation thread. Omit to start a new thread.",
    )
    filters: dict | None = None


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    citations: list[dict]
    retrieved_chunks: list[dict]
    model: str
    latency_ms: float
    message_count: int
    standalone_query: str = ""
    intent: str = ""
    summary: str = ""
    facts: list[dict] = Field(default_factory=list)
    # Branch A (documents) and Branch B (web), returned separately.
    document_results: list[dict] = Field(default_factory=list)
    web_results: list[dict] = Field(default_factory=list)


class CreateThreadResponse(BaseModel):
    thread_id: str


class ThreadSummaryResponse(BaseModel):
    thread_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class ThreadListResponse(BaseModel):
    threads: list[ThreadSummaryResponse]


class RenameThreadRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)


class ChatMessageResponse(BaseModel):
    role: str
    content: str
    created_at: str


class ThreadHistoryResponse(BaseModel):
    thread_id: str
    messages: list[ChatMessageResponse]
    summary: str = ""
    facts: list[dict] = Field(default_factory=list)


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str | None = Field(default=None, max_length=64)


class VoiceOption(BaseModel):
    id: str
    label: str
    description: str = ""


class VoiceOptionsResponse(BaseModel):
    provider: str
    default: str | None
    voices: list[VoiceOption]


class TranscriptResponse(BaseModel):
    transcript: str


class VoiceTurnResponse(BaseModel):
    thread_id: str
    transcript: str
    answer: str
    citations: list[dict]
    audio_base64: str = ""
    speech_error: str | None = None
    summary: str = ""
    facts: list[dict] = Field(default_factory=list)
    message_count: int
    model: str


class IngestDocumentItem(BaseModel):
    content: str
    source: str | None = None
    metadata: dict | None = None


class IngestRequest(BaseModel):
    documents: list[IngestDocumentItem]


class IngestResponse(BaseModel):
    document_count: int
    chunk_count: int


class IngestPdfResponse(BaseModel):
    document_count: int
    chunk_count: int
    # R-84: this used to echo the ENTIRE ParsedDocument - full text and
    # every page - back to the client on every upload, so a 200-page PDF
    # returned megabytes of markdown nobody read. Summarise instead.
    filename: str
    page_count: int
    image_count: int
    has_content: bool


# R-83: EvaluationSample / EvaluationRequest / EvaluationResponse were
# referenced by nothing (the live eval API uses the RagasEval* models below)
# and have been removed.


class RagasEvalSample(BaseModel):
    question: str = Field(..., min_length=1)
    ground_truth: str | None = Field(
        default=None,
        description="Reference answer. Required for context_recall/context_precision.",
    )


class RagasEvalRequest(BaseModel):
    cases: list[RagasEvalSample] = Field(..., min_length=1)


class RagasEvalResponse(BaseModel):
    runner: str
    status: str
    samples: int
    metrics: dict
    per_sample: list[dict]
    # R-88: the runner no longer echoes the whole dataset at the top level -
    # each per_sample entry carries its own "contexts". Kept as an optional
    # field so existing clients do not break on a missing key.
    contexts: list[dict] = Field(default_factory=list)
    # R-10: provenance the background-job path already returns. Without these,
    # a result saved off this endpoint carries no manifest at all.
    manifest: dict = Field(default_factory=dict)
    metrics_coverage: dict = Field(default_factory=dict)
    headline_metrics: dict = Field(default_factory=dict)
    diagnostic_metrics: dict = Field(default_factory=dict)
    evaluated_samples: int | None = None
    skipped_samples: int | None = None
    judge_model: str | None = None
    embedding_model: str | None = None


class EvaluationJobCreatedResponse(BaseModel):
    job_id: str
    status: str


class EvaluationJobStatusResponse(BaseModel):
    job_id: str
    status: str


class EvaluationJobNotFoundResponse(BaseModel):
    error: str


class EvaluationJobResultResponse(BaseModel):
    status: str
    result: dict | None = None
    error: str | None = None
