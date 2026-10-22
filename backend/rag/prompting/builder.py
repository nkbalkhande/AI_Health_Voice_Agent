from xml.sax.saxutils import quoteattr

from backend.domain.entities.chat import ChatMessage
from backend.domain.entities.chunk import Chunk
from backend.rag.agent_prompt.rag import SYSTEM_PROMPT

# R-01: evaluation contexts must be formatted EXACTLY the way generation embeds
# chunks in the prompt, or citation-bearing sentences look unsupported to the
# judge. format_chunk/format_contexts are the single source of that markup;
# EvaluateRagPipelineUseCase rebuilds Chunks from citations and reuses them.

# R-79: the pseudo-XML context wrapper interpolated document text verbatim, so
# a chunk containing a literal closing tag ended the context block early (and a
# filename containing a double quote broke the attribute). Attributes now go
# through quoteattr; closing tags in content are neutralised here.
_CLOSING_TAGS = ("</chunk>", "</retrieved_context>")


def _neutralize_tags(content: str) -> str:
    text = content or ""
    for tag in _CLOSING_TAGS:
        text = text.replace(tag, tag.replace("<", "&lt;"))
    return text


class PromptBuilder:
    SYSTEM = SYSTEM_PROMPT

    def build(self, query: str, chunks: list[Chunk]) -> str:
        return self.build_turn_user_message(query, chunks) + "\n\nAnswer:"

    def build_turn_user_message(self, query: str, chunks: list[Chunk]) -> str:
        context = self.format_context(chunks)
        return f"Context:\n{context}\n\nQuestion: {query}"

    def build_chat_messages(
        self,
        *,
        history: list[ChatMessage],
        query: str,
        chunks: list[Chunk] | None = None,
        max_history_messages: int,
        summary: str = "",
        facts: list[dict] | None = None,
        resolved_query: str | None = None,
        document_chunks: list[Chunk] | None = None,
        web_pages: list | None = None,
    ) -> list[dict[str, str]]:
        """
        Multi-turn LiteLLM payload: system + prior turns (raw text) + current
        turn. Document evidence and web evidence are rendered as two
        separate sections; `chunks` is kept as an alias for document chunks.
        """
        documents = document_chunks if document_chunks is not None else (chunks or [])
        context_block = self.format_conversation_context(documents, web_pages or [])
        system_content = self.SYSTEM
        if summary:
            system_content += f"\n\nSession summary:\n{summary}"
        if facts:
            rendered = "\n".join(
                f"- {item.get('key')}: {item.get('value')}"
                for item in facts
                if isinstance(item, dict) and item.get("key") and item.get("value")
            )
            if rendered:
                system_content += f"\n\nKnown facts:\n{rendered}"
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
        ]
        trimmed = history[-max_history_messages:] if max_history_messages > 0 else []
        for msg in trimmed:
            if msg.role in ("user", "assistant"):
                messages.append({"role": msg.role, "content": msg.content})
        question = f"Context:\n{context_block}\n\nQuestion: {query}"
        if resolved_query and resolved_query.strip() and resolved_query.strip() != query.strip():
            question += f"\n\nResolved search query: {resolved_query.strip()}"
        messages.append({"role": "user", "content": question})
        return messages

    def format_chunk(self, chunk: Chunk) -> str:
        """Format one chunk exactly the way generation embeds it in the prompt.

        Keeps the R-79 hardening: attributes go through ``quoteattr`` and
        literal closing tags in content are neutralised.
        """
        source = chunk.metadata.get("source", chunk.document_id)
        page = chunk.metadata.get("page_number", "N/A")
        return (
            f"<chunk\n    source={quoteattr(str(source))}\n"
            f"    page={quoteattr(str(page))}>\n\n"
            f"{_neutralize_tags(chunk.content)}\n\n"
            f"</chunk>"
        )

    def format_contexts(self, chunks: list[Chunk]) -> list[str]:
        """Per-chunk formatted strings for evaluation (same markup as generation)."""
        return [self.format_chunk(chunk) for chunk in chunks]

    def format_context(self, chunks: list[Chunk]) -> str:
        """Full context block inserted into the generation prompt."""
        if not chunks:
            return "<retrieved_context>\n\n(no context)\n\n</retrieved_context>"

        return (
            "<retrieved_context>\n\n"
            + "\n\n".join(self.format_contexts(chunks))
            + "\n\n</retrieved_context>"
        )

    def format_conversation_context(
        self, document_chunks: list[Chunk], web_pages: list
    ) -> str:
        """Two labeled sections. The branches are never blended into one list."""
        document_section = (
            "\n\n".join(self.format_contexts(document_chunks))
            if document_chunks
            else "(no document evidence)"
        )
        web_section = (
            "\n\n".join(self.format_web_page(page) for page in web_pages)
            if web_pages
            else "(no web evidence)"
        )
        return (
            "Document evidence (trusted knowledge base):\n"
            f"<document_evidence>\n\n{document_section}\n\n</document_evidence>\n\n"
            "Web evidence (internet search):\n"
            f"<web_evidence>\n\n{web_section}\n\n</web_evidence>"
        )

    def format_web_page(self, page) -> str:
        title = getattr(page, "title", "") or getattr(page, "url", "")
        url = getattr(page, "url", "")
        snippet = getattr(page, "snippet", "")
        return (
            f"<page\n    title={quoteattr(str(title))}\n"
            f"    url={quoteattr(str(url))}>\n\n"
            f"{_neutralize_tags(snippet)}\n\n</page>"
        )

    # Back-compat alias for callers/tests still using the private name.
    _format_context = format_context

    @staticmethod
    def chunks_from_citations(citations: list[dict]) -> list[Chunk]:
        """Rebuild Chunk objects from citation dicts for shared formatting.

        Omits a null ``source`` key so ``format_chunk`` falls back to
        ``document_id``, matching generation-time Chunk metadata behavior.
        """
        chunks: list[Chunk] = []
        for i, citation in enumerate(citations):
            metadata: dict = {
                "page_number": citation.get("page_number", "N/A"),
            }
            source = citation.get("source")
            if source is not None:
                metadata["source"] = source
            chunks.append(
                Chunk(
                    id=str(citation.get("index", i + 1)),
                    document_id=str(citation.get("document_id") or ""),
                    content=str(citation.get("content") or ""),
                    score=citation.get("score"),
                    metadata=metadata,
                )
            )
        return chunks
