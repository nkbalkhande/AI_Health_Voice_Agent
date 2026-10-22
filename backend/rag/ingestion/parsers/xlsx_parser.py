"""XLSX parser: each worksheet becomes one page with sheet metadata (R-52)."""

from __future__ import annotations

import base64
import io
from typing import Callable

from backend.core.config import settings
from backend.core.exceptions import IngestionError
from backend.core.logger import logger
from backend.rag.ingestion.models import DocumentImageBlock, ParsedDocument, PageMetadata
from backend.rag.ingestion.parsers.base_parser import BaseDocumentParser
from backend.rag.ingestion.parsers.xlsx_drawings import XlsxImage
from backend.rag.ingestion.parsers.xlsx_sheets import XlsxSheet, extract_xlsx_sheets

log = logger.getChild("rag.ingestion.parsers.xlsx_parser")

#: ``(image_b64_png, page_content, page_no, caption) -> transcription``.
ImageAnalyzer = Callable[[str, str, int, str], str]

# Vision models read raster formats; Office metafiles are skipped.
_TRANSCRIBABLE_CONTENT_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/gif", "image/bmp", "image/tiff"}
)


def _default_analyzer(image_b64: str, page_content: str, page_no: int, caption: str) -> str:
    # Lazy import: litellm + settings wiring are only needed when transcribing.
    from backend.rag.ingestion.image_analyzer import analyze_image

    return analyze_image(
        page_content=page_content,
        image_b64=image_b64,
        page_no=page_no,
        caption=caption,
    )


def _png_base64(image: XlsxImage) -> str | None:
    """Re-encode any raster media as PNG for the vision request."""
    if not image.data or image.content_type not in _TRANSCRIBABLE_CONTENT_TYPES:
        return None
    if image.content_type == "image/png":
        return base64.b64encode(image.data).decode("ascii")
    try:
        from PIL import Image as PILImage
    except ImportError:  # pragma: no cover - pillow ships with docling
        return None
    with PILImage.open(io.BytesIO(image.data)) as pil:
        buffer = io.BytesIO()
        pil.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def image_caption(image: XlsxImage) -> str:
    """One deterministic line that pins a picture to its sheet, cells, and label.

    This is prepended to every transcription so the label ("Active Clients")
    and the transcribed pixels can never be separated by chunking — the cell
    context Rutvik's segmented-image approach would lose.
    """
    return (
        f'Embedded image "{image.shape_name}" on sheet "{image.sheet_name}" '
        f"(sheet {image.sheet_index}), cells {image.anchor_range}, "
        f"label: {image.label_text}."
    )


class XlsxParser(BaseDocumentParser):
    """Parse Excel workbooks with sheet-as-page metadata.

    Unlike Docling's whole-workbook convert (which collapses most sheets onto
    ``page_number=1``), this parser enumerates worksheets in workbook order and
    assigns a stable 1-based sheet index plus ``sheet_name`` to every page.

    Drawing layer: text boxes are rendered into the page text; pictures get an
    anchored placeholder in the page text and, when ``analyze_images`` is on
    (``ingestion.xlsx_analyze_images``), a vision transcription stored as a
    separate image chunk whose content starts with ``image_caption``.
    """

    def __init__(
        self,
        *,
        analyze_images: bool | None = None,
        analyzer: ImageAnalyzer | None = None,
    ) -> None:
        self._analyze_images = (
            settings.ingestion_xlsx_analyze_images
            if analyze_images is None
            else analyze_images
        )
        self._analyzer = analyzer or _default_analyzer

    def parse(
        self,
        document_bytes: bytes,
        *,
        filename: str = "workbook.xlsx",
    ) -> ParsedDocument:
        try:
            sheets = extract_xlsx_sheets(document_bytes)
        except Exception as exc:
            raise IngestionError(
                f"XLSX workbook parsing failed for {filename!r}: {exc}"
            ) from exc

        if not sheets:
            raise IngestionError(
                f"XLSX workbook {filename!r} contains no worksheets"
            )

        pages = [sheet.text for sheet in sheets]
        page_metadata = [
            PageMetadata(
                page_number=sheet.page_number,
                metadata=self._page_metadata(sheet),
            )
            for sheet in sheets
        ]
        images = self._transcribe_images(sheets, filename=filename)
        # Combined text keeps sheet order for fallbacks that ignore ``pages``.
        text = "\n\n".join(
            f"# Sheet: {sheet.sheet_name}\n{sheet.text}".strip()
            for sheet in sheets
            if sheet.text.strip()
        )
        log.info(
            "xlsx.parsed filename=%s sheet_count=%d non_empty_sheets=%d "
            "embedded_images=%d text_boxes=%d transcribed_images=%d",
            filename,
            len(sheets),
            sum(1 for page in pages if page.strip()),
            sum(len(sheet.images) for sheet in sheets),
            sum(len(sheet.text_boxes) for sheet in sheets),
            len(images),
        )
        return ParsedDocument(
            filename=filename,
            text=text,
            pages=pages,
            page_metadata=page_metadata,
            images=images,
            docling_document=[],
        )

    @staticmethod
    def _page_metadata(sheet: XlsxSheet) -> dict:
        metadata: dict = {"sheet_name": sheet.sheet_name}
        # Only sheets that actually carry drawings get the counts, so every
        # chunk of a screenshot-bearing sheet says so in its metadata and
        # citations, and plain sheets stay unchanged.
        if sheet.images:
            metadata["embedded_images"] = len(sheet.images)
        if sheet.text_boxes:
            metadata["embedded_text_boxes"] = len(sheet.text_boxes)
        return metadata

    def _transcribe_images(
        self, sheets: list[XlsxSheet], *, filename: str
    ) -> list[DocumentImageBlock]:
        if not self._analyze_images:
            return []
        blocks: list[DocumentImageBlock] = []
        # The same media file is often placed twice (Planning / Final columns);
        # transcribe it once and reuse the text for every placement.
        transcriptions: dict[str, str | None] = {}
        for sheet in sheets:
            for image in sheet.images:
                key = image.media_sha256 or f"{sheet.page_number}:{image.ordinal}"
                if key not in transcriptions:
                    transcriptions[key] = self._transcribe(image, page_content=sheet.text)
                transcription = transcriptions[key]
                if not transcription:
                    continue
                caption = image_caption(image)
                blocks.append(
                    DocumentImageBlock(
                        page_no=sheet.page_number,
                        caption=caption,
                        analysis=f"{caption}\n\n{transcription}",
                        metadata={
                            "sheet_name": sheet.sheet_name,
                            "shape_name": image.shape_name,
                            "anchor_range": image.anchor_range,
                            "anchor_label": image.label,
                            "anchor_label_cell": image.label_cell,
                            "media_sha256": image.media_sha256,
                        },
                    )
                )
        log.info(
            "xlsx.images filename=%s placements=%d unique_media=%d transcribed=%d",
            filename,
            sum(len(sheet.images) for sheet in sheets),
            len(transcriptions),
            sum(1 for value in transcriptions.values() if value),
        )
        return blocks

    def _transcribe(self, image: XlsxImage, *, page_content: str) -> str | None:
        image_b64 = _png_base64(image)
        if image_b64 is None:
            log.warning(
                "Skipping image transcription for %s on sheet %d (%s): unsupported media",
                image.shape_name,
                image.sheet_index,
                image.content_type,
            )
            return None
        try:
            text = self._analyzer(
                image_b64, page_content, image.sheet_index, image_caption(image)
            )
        except Exception as exc:  # noqa: BLE001 - keep ingest alive
            log.warning(
                "Image transcription failed for %s on sheet %d: %s",
                image.shape_name,
                image.sheet_index,
                exc,
            )
            return None
        return (text or "").strip() or None
