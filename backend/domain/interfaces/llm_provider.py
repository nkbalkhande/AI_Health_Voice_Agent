from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

GenerationStatus = Literal[
    "answered",
    "abstained",
    "empty_generation",
    "incomplete_length",
    "refused",
    "generation_error",
    # Distinguish infrastructure failures from blank model replies so refusal
    # scorecards and eval eligibility can report them separately (R-09).
    "timeout",
    "infrastructure_error",
]


@dataclass(frozen=True)
class GenerationResult:
    """Structured outcome of a single LLM generation.

    Every generation ends in exactly one ``status`` so callers never have to
    guess whether an empty string means "the model said nothing", "the model
    refused", or "the request failed".
    """

    text: str

    status: GenerationStatus

    finish_reason: str | None = None
    refusal: str | None = None

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None

    request_id: str | None = None

    # R-85: cost was plumbed through QueryResult and the audit record, but
    # nothing ever produced a value - AuditRecord hardcoded None and
    # GenerationResult had no field at all.
    cost: float | None = None

    # Free-form detail for diagnostics (e.g. the exception message on error).
    error: str | None = None


class LLMProvider(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        pass

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Multi-turn completion. Each item:
        {"role": "user"|"assistant"|"system", "content": "..."}."""
        pass

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
    ):
        """Yield answer text as it is produced.

        The default waits for the full reply and yields it once. Providers
        that support token streaming override this.
        """
        result = await self.chat(messages, max_tokens=max_tokens)
        if result.text:
            yield result.text
