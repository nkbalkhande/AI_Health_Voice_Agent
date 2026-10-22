from backend.rag.ingestion.models import DocumentImageBlock, ParsedDocument

__all__ = [
    "DoclingDocumentParser",
    "DoclingParser",
    "ParsedDocument",
    "DocumentImageBlock",
    "IngestionFacade",
    "ParserFactory",
]


def __getattr__(name: str):
    if name in {"DoclingDocumentParser", "DoclingParser"}:
        from backend.rag.ingestion.parsers.docling_parser import DoclingParser

        if name == "DoclingDocumentParser":
            return DoclingParser
        return DoclingParser
    if name == "IngestionFacade":
        from backend.rag.ingestion.facade.ingestion_facade import IngestionFacade

        return IngestionFacade
    if name == "ParserFactory":
        from backend.rag.ingestion.factory.parser_factory import ParserFactory

        return ParserFactory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
