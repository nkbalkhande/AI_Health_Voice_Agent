import io

from docling.datamodel.accelerator_options import (
    AcceleratorDevice,
    AcceleratorOptions,
)
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from PyPDF2 import PdfReader

from backend.core.exceptions import IngestionError
from backend.core.logger import logger
from backend.rag.chunking.constants import PAGE_BREAK_PLACEHOLDER
from backend.rag.ingestion.models import ParsedPdfDocument, PdfImageBlock

log = logger.getChild("rag.ingestion.pdf_parser")


class DoclingPdfParser:
    """Extract text, tables, and image captions from PDF bytes via Docling."""

    def __init__(self, converter: DocumentConverter | None = None) -> None:
        self._fallback_converter = converter
        try:
            self._converter = self.gpu_docling_converter()
        except Exception as exc:
            log.warning(
                "GPU Docling converter unavailable (%s); "
                "falling back to default converter",
                exc,
            )
            try:
                self._converter =  self.cpu_docling_converter()
            except Exception as err:
                log.warning(
                "CPU OCR Docling converter unavailable (%s); "
                "falling back to default converter",
                err,
            )
                self._converter = DocumentConverter()

    def cpu_docling_converter(self) -> DocumentConverter:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.do_formula_enrichment = True
        # R-30: page images are never consumed downstream, but were rendered
        # at 4.2x for every page and held for the whole parse. Keep off until
        # an image consumer exists.
        pipeline_options.generate_page_images = False

        # Use Tesseract CLI for OCR
        ocr_options = TesseractCliOcrOptions(
            tesseract_cmd="tesseract",
        )
        pipeline_options.ocr_options = ocr_options
        converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=pipeline_options
                        ),
                    }
                )
        return converter

    def _convert(self, pdf_bytes: bytes, filename: str, start_page: int, end_page: int):
        """Convert a batch, falling back to a default converter on pipeline failure.

        Docling initializes its pipeline lazily on the first convert() call, so
        environment-specific failures (missing OCR binary, unavailable GPU, etc.)
        surface here rather than in __init__. Retry once with a plain converter.
        Docling consumes the stream on each attempt, so build a fresh one per try.
        """
        try:
            stream = DocumentStream(name=filename, stream=io.BytesIO(pdf_bytes))
            return self._converter.convert(stream, page_range=(start_page, end_page))
        except Exception as exc:
            if self._converter is self._fallback_converter:
                raise
            log.warning(
                "Docling converter failed (%s); retrying with default converter",
                exc,
            )
            fallback = self._fallback_converter or DocumentConverter()
            self._fallback_converter = fallback
            # R-46: do NOT replace self._converter. deps.get_pdf_parser is
            # lru_cached, so one transient batch failure used to permanently
            # downgrade OCR (Tesseract psm 6 -> docling default) and drop
            # formula enrichment for every later batch and document.
            stream = DocumentStream(name=filename, stream=io.BytesIO(pdf_bytes))
            return fallback.convert(stream, page_range=(start_page, end_page))

    def gpu_docling_converter(self) -> DocumentConverter:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True

        # Determine formula enrichment setting (independent of do_ocr)
        pipeline_options.do_formula_enrichment = True

        pipeline_options.accelerator_options = AcceleratorOptions(
            device=AcceleratorDevice.AUTO, num_threads=16
        )

        # R-30: page images are never consumed downstream, but were rendered
        # at 4.2x for every page and held for the whole parse. Keep off until
        # an image consumer exists.
        pipeline_options.generate_page_images = False

        # Use Tesseract CLI for OCR
        ocr_options = TesseractCliOcrOptions(
            tesseract_cmd="tesseract",
        )
        ocr_options.psm = 6
        pipeline_options.ocr_options = ocr_options
        converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=pipeline_options
                        ),
                    }
                )
        return converter
    def parse(
        self, pdf_bytes: bytes, *, filename: str = "input.pdf"
    ) -> ParsedPdfDocument:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            num_pages = len(reader.pages)
            total_markdown = ""
            pages: list[str] = []
            images: list[PdfImageBlock] = []
            docling_docs = []
            batch_size = 20
            for num in range(0, num_pages, batch_size):
                start_page = num + 1
                end_page = min(num + batch_size, num_pages)
                # Docling consumes/closes the stream on each convert() call,
                # so a fresh DocumentStream is built per batch inside _convert.
                result = self._convert(pdf_bytes, filename, start_page, end_page)
                markdown = result.document.export_to_markdown(
                    page_break_placeholder=PAGE_BREAK_PLACEHOLDER
                )
                total_markdown += markdown
                pages.extend(
                    page.strip()
                    for page in markdown.split(PAGE_BREAK_PLACEHOLDER)
                    if page.strip()
                )
                doc = result.document
                docling_docs.append(doc)

                if hasattr(doc, "pictures"):
                    for img in doc.pictures:
                        # R-17: docling PictureItem has no `caption` attribute
                        # (it exposes `captions: list[RefItem]` plus
                        # caption_text(doc)), so the old getattr always
                        # returned None and no image-caption document was ever
                        # created.
                        caption = None
                        caption_text = getattr(img, "caption_text", None)
                        if callable(caption_text):
                            caption = caption_text(doc) or None
                        images.append(PdfImageBlock(caption=caption))

            return ParsedPdfDocument(
                filename=filename,
                text=total_markdown,
                pages=pages,
                images=images,
                docling_document=[],
            )
        except Exception as e:
            raise IngestionError(f"Docling PDF processing failed: {e}") from e
