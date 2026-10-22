from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentImageBlock(BaseModel):
    page_no: int | None = None
    caption: str | None = None
    analysis: str | None = None
    #: Format-specific provenance merged onto the image chunk's metadata
    #: (XLSX: ``sheet_name``, ``anchor_range``, ``anchor_label`` …).
    metadata: dict[str, Any] = Field(default_factory=dict)


class PageMetadata(BaseModel):
    """Per-page provenance aligned with ``ParsedDocument.pages`` by index.

    ``page_number`` is the logical page identity (PDF page, PPTX slide, XLSX
    sheet index). Format-specific keys live in ``metadata`` so a new parser
    can attach ``sheet_name``, ``slide_title``, etc. without changing this model.
    """

    page_number: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """Structured output from document parsing — input to document building and chunking."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    filename: str = "document.pdf"
    text: str = ""
    pages: list[str] = Field(default_factory=list)
    #: Parallel to ``pages`` when the parser supplies per-page provenance (XLSX).
    page_metadata: list[PageMetadata] = Field(default_factory=list)
    tables: list[str] | None = None
    images: list[DocumentImageBlock] = Field(default_factory=list)
    docling_document: Any = None

    @property
    def has_content(self) -> bool:
        return (
            bool(self.text.strip())
            or any(page.strip() for page in self.pages)
            or any(
                (img.caption and img.caption.strip())
                or (img.analysis and img.analysis.strip())
                for img in self.images
            )
        )
