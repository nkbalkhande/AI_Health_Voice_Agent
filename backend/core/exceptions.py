class PlatformError(Exception):
    """Base exception for the GenAI platform."""


class ProviderNotConfiguredError(PlatformError):
    pass


class ProviderNotFoundError(PlatformError):
    pass


class RetrievalError(PlatformError):
    pass


class IngestionError(PlatformError):
    pass


class UnsupportedFileType(IngestionError):
    """Raised when ingestion is asked to parse an unsupported file extension."""

    pass


class ChatThreadNotFoundError(PlatformError):
    def __init__(self, thread_id: str) -> None:
        self.thread_id = thread_id
        super().__init__(f"Chat thread not found: {thread_id}")
