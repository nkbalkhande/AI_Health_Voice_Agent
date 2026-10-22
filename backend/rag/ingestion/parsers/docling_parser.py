from __future__ import annotations

import io

from docling.datamodel.accelerator_options import (
    AcceleratorDevice,
    AcceleratorOptions,
)
from docling.datamodel.base_models import ConversionStatus, DocumentStream, InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from backend.core.config import settings
from backend.core.exceptions import IngestionError
from backend.core.logger import logger
from backend.evaluation.provider_credentials import model_id
from backend.rag.chunking.constants import PAGE_BREAK_PLACEHOLDER
from backend.rag.ingestion.image_analyzer import (
    analyze_image,
    picture_page_no,
    pil_to_base64_png,
)
from backend.rag.ingestion.factory.page_counter_factory import PageCounterFactory
from backend.rag.ingestion.models import DocumentImageBlock, ParsedDocument
from backend.rag.ingestion.parsers.base_parser import BaseDocumentParser
from backend.rag.ingestion.vlm_prompt import DEFAULT_VLM_PROMPT

log = logger.getChild("rag.ingestion.parsers.docling_parser")

# Crops are sent to the VLM as-is; under-sampling here is unrecoverable for
# dense table cells.
_PICTURE_IMAGES_SCALE = 3.0


class DoclingParser(BaseDocumentParser):
    """Extract text, tables, and image captions/analysis from documents via Docling."""

    def __init__(
        self,
        converter: DocumentConverter | None = None,
        *,
        use_vlm: bool | None = None,
        analyze_images: bool | None = None,
        page_counter_factory: type[PageCounterFactory] = PageCounterFactory,
    ) -> None:
        self._use_vlm = settings.ingestion_use_vlm if use_vlm is None else use_vlm
        self._analyze_images = (
            settings.ingestion_analyze_images
            if analyze_images is None
            else analyze_images
        )
        self._page_counter_factory = page_counter_factory
        self._fallback_converter = converter
        # Built on first use by the VLM path only (see _picture_layout_converter).
        self._picture_converter: DocumentConverter | None = None
        if converter is not None:
            self._converter = converter
            return

        if self._use_vlm:
            log.info("VLM docling is enabled!")
            self._converter = self.vlm_docling_converter()
            return

        try:
            self._converter = self.gpu_docling_converter()
        except Exception as exc:
            log.warning(
                "GPU Docling converter unavailable (%s); "
                "falling back to CPU OCR converter",
                exc,
            )
            try:
                self._converter = self.cpu_docling_converter()
            except Exception as err:
                log.warning(
                    "CPU OCR Docling converter unavailable (%s); "
                    "falling back to default converter",
                    err,
                )
                self._converter = DocumentConverter()

    @staticmethod
    def _require_azure_vlm_env() -> tuple[str, str, str, str]:
        endpoint = (settings.azure_openai_endpoint or "").strip().rstrip("/")
        api_key = (settings.azure_openai_api_key or "").strip()
        api_version = (
            settings.azure_openai_api_version or "2025-03-01-preview"
        ).strip()
        deployment = model_id(settings.llm_model)

        missing: list[str] = []
        if not endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not api_key:
            missing.append("AZURE_OPENAI_API_KEY")
        if not deployment:
            missing.append("AZURE_OPENAI_DEPLOYMENT")
        if missing:
            raise IngestionError(
                "VLM ingestion enabled but missing required Azure config: "
                + ", ".join(missing)
            )
        return endpoint, api_key, deployment, api_version

    @classmethod
    def _vlm_endpoint(cls) -> tuple[str, dict[str, str], dict[str, str]]:
        """(url, headers, extra params) for the page-reading model in llm.model.

        `openai/<model>` calls OpenAI directly; anything else is treated as an
        Azure deployment name.
        """
        model = (settings.llm_model or "").strip()
        if model.lower().startswith("openai/"):
            api_key = (settings.openai_api_key or "").strip()
            if not api_key:
                raise IngestionError(
                    "VLM ingestion enabled but missing required OpenAI config: "
                    "OPENAI_API_KEY"
                )
            return (
                "https://api.openai.com/v1/chat/completions",
                {"Authorization": f"Bearer {api_key}"},
                {"model": model_id(model)},
            )
        endpoint, api_key, deployment, api_version = cls._require_azure_vlm_env()
        return (
            f"{endpoint}/openai/deployments/{deployment}"
            f"/chat/completions?api-version={api_version}",
            {"api-key": api_key},
            {},
        )

    def _apply_picture_image_options(self, pipeline_options: PdfPipelineOptions) -> None:
        """Enable picture crops when per-image analysis is on (R-30 consumer)."""
        if not self._analyze_images:
            pipeline_options.generate_page_images = False
            pipeline_options.generate_picture_images = False
            return
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = _PICTURE_IMAGES_SCALE

    def vlm_docling_converter(self) -> DocumentConverter:
        # Lazy-import so OCR-only processes do not depend on VLM modules.
        from docling.datamodel.pipeline_options import VlmPipelineOptions
        from docling.datamodel.pipeline_options_vlm_model import (
            ApiVlmOptions,
            ResponseFormat,
        )
        from docling.datamodel.settings import settings as docling_settings
        from docling.document_converter import ImageFormatOption
        from docling.pipeline.vlm_pipeline import VlmPipeline
        from pydantic import AnyUrl

        url, headers, params = self._vlm_endpoint()
        prompt = (settings.ingestion_vlm_prompt or DEFAULT_VLM_PROMPT).strip()
        if not prompt:
            raise IngestionError(
                "VLM ingestion enabled but ingestion.vlm_prompt is empty"
            )
        vlm_options = ApiVlmOptions(
            url=AnyUrl(url),
            headers=headers,
            # gpt-4o's output ceiling; dense table pages overran 8192.
            params={"max_completion_tokens": 16384, **params},
            prompt=prompt,
            response_format=ResponseFormat.MARKDOWN,
            temperature=0.0,
            scale=2.0,
            timeout=180,
            concurrency=settings.ingestion_vlm_concurrency,
        )
        # Docling only parallelises pages inside one page batch (default 4).
        docling_settings.perf.page_batch_size = max(
            docling_settings.perf.page_batch_size,
            settings.ingestion_vlm_concurrency,
        )
        log.info(
            "VLM page requests in parallel: %d", settings.ingestion_vlm_concurrency
        )
        pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_options,
            enable_remote_services=True,
        )
        # Picture crops cannot come from this pipeline: a MARKDOWN response is
        # re-parsed by Docling's markdown backend, which never emits PictureItem
        # with provenance. Image analysis runs on a separate layout pass instead
        # (see _picture_layout_converter).
        # Do not restrict allowed_formats: Docling still converts PPTX/DOCX/etc
        # via its default pipelines. Only PDF + IMAGE are routed through VLM.
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                ),
                InputFormat.IMAGE: ImageFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                ),
            },
        )

    def _picture_layout_converter(self) -> DocumentConverter:
        """Layout-only converter used to crop pictures alongside the VLM pass.

        OCR, table structure, and formula enrichment are off: the VLM already
        produced the page text, so this pass only needs Docling's layout model
        to locate pictures and render their crops.
        """
        if self._picture_converter is None:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False
            pipeline_options.do_table_structure = False
            pipeline_options.generate_page_images = True
            pipeline_options.generate_picture_images = True
            pipeline_options.images_scale = _PICTURE_IMAGES_SCALE
            self._picture_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    ),
                }
            )
        return self._picture_converter

    def _extract_images_via_layout_pass(
        self, document_bytes: bytes, *, filename: str, pages: list[str]
    ) -> list[DocumentImageBlock]:
        try:
            converter = self._picture_layout_converter()
            stream = DocumentStream(name=filename, stream=io.BytesIO(document_bytes))
            result = converter.convert(stream)
            self._assert_convert_ok(result, context="picture layout convert")
        except Exception as exc:  # noqa: BLE001 - keep ingest alive
            log.warning(
                "Picture layout pass failed for %s (%s); "
                "continuing without image analysis",
                filename,
                exc,
            )
            return []
        return self._extract_images(result.document, pages=pages)

    def cpu_docling_converter(self) -> DocumentConverter:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.do_formula_enrichment = True
        self._apply_picture_image_options(pipeline_options)

        ocr_options = TesseractCliOcrOptions(
            tesseract_cmd="tesseract",
        )
        pipeline_options.ocr_options = ocr_options
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                ),
            }
        )

    def _convert(
        self,
        document_bytes: bytes,
        filename: str,
        start_page: int | None = None,
        end_page: int | None = None,
    ):
        """Convert a batch or whole document, with default-converter fallback.

        Docling initializes its pipeline lazily on the first convert() call, so
        environment-specific failures (missing OCR binary, unavailable GPU, etc.)
        surface here rather than in __init__. Retry once with a plain converter.
        Docling consumes the stream on each attempt, so build a fresh one per try.
        """
        page_range = (
            (start_page, end_page)
            if start_page is not None and end_page is not None
            else None
        )
        try:
            stream = DocumentStream(name=filename, stream=io.BytesIO(document_bytes))
            if page_range is None:
                return self._converter.convert(stream)
            return self._converter.convert(stream, page_range=page_range)
        except Exception as exc:
            if self._converter is self._fallback_converter:
                raise
            log.warning(
                "Docling converter failed (%s); retrying with default converter",
                exc,
            )
            fallback = self._fallback_converter or DocumentConverter()
            self._fallback_converter = fallback
            # R-46: do NOT replace self._converter. The shared DoclingParser
            # instance is reused across requests, so one transient batch
            # failure used to permanently downgrade OCR (Tesseract psm 6 ->
            # docling default) and drop formula enrichment for every later
            # batch and document.
            stream = DocumentStream(name=filename, stream=io.BytesIO(document_bytes))
            if page_range is None:
                return fallback.convert(stream)
            return fallback.convert(stream, page_range=page_range)

    def gpu_docling_converter(self) -> DocumentConverter:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.do_formula_enrichment = True
        pipeline_options.accelerator_options = AcceleratorOptions(
            device=AcceleratorDevice.CUDA, num_threads=10
        )
        self._apply_picture_image_options(pipeline_options)

        ocr_options = TesseractCliOcrOptions(
            tesseract_cmd="tesseract",
        )
        ocr_options.psm = 6
        pipeline_options.ocr_options = ocr_options
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                ),
            }
        )

    @staticmethod
    def _assert_convert_ok(result, *, context: str) -> None:
        status = getattr(result, "status", None)
        if status is not None and status != ConversionStatus.SUCCESS:
            errors = getattr(result, "errors", None) or []
            detail = "; ".join(str(e) for e in errors) if errors else str(status)
            raise IngestionError(
                f"Docling {context} failed with status={status}: {detail}"
            )

    @staticmethod
    def _page_content_for(
        page_no: int | None,
        *,
        page_texts: dict[int, str] | None = None,
        pages: list[str] | None = None,
    ) -> str:
        if page_no is None:
            return ""
        if page_texts is not None:
            return page_texts.get(page_no, "")
        if pages is not None and 1 <= page_no <= len(pages):
            return pages[page_no - 1]
        return ""

    def _extract_images(
        self,
        doc,
        *,
        page_texts: dict[int, str] | None = None,
        pages: list[str] | None = None,
    ) -> list[DocumentImageBlock]:
        images: list[DocumentImageBlock] = []
        if not hasattr(doc, "pictures"):
            return images
        for img in doc.pictures:
            caption: str | None = None
            caption_text = getattr(img, "caption_text", None)
            if callable(caption_text):
                raw = caption_text(doc)
                caption = str(raw).strip() if raw else None

            page_no = picture_page_no(img)
            analysis: str | None = None

            if self._analyze_images:
                pil = None
                get_image = getattr(img, "get_image", None)
                if callable(get_image):
                    try:
                        pil = get_image(doc)
                    except Exception as exc:  # noqa: BLE001 - keep ingest alive
                        log.warning(
                            "Failed to rasterize picture on page %s: %s",
                            page_no,
                            exc,
                        )
                if pil is not None:
                    page_content = self._page_content_for(
                        page_no, page_texts=page_texts, pages=pages
                    )
                    try:
                        analysis = analyze_image(
                            page_content=page_content,
                            image_b64=pil_to_base64_png(pil),
                            page_no=page_no,
                            caption=caption,
                        ) or None
                    except Exception as exc:  # noqa: BLE001 - keep ingest alive
                        log.warning(
                            "Image analysis failed on page %s: %s",
                            page_no,
                            exc,
                        )
                else:
                    log.warning(
                        "Skipping image analysis on page %s: no raster image",
                        page_no,
                    )

            images.append(
                DocumentImageBlock(
                    page_no=page_no,
                    caption=caption,
                    analysis=analysis,
                )
            )
        return images

    def _parse_with_ocr(
        self, document_bytes: bytes, *, filename: str
    ) -> ParsedDocument:
        page_counter = self._page_counter_factory.create(filename)
        num_pages = page_counter.total_pages(document_bytes)
        if num_pages is None:
            return self._parse_without_batching(
                document_bytes,
                filename=filename,
            )

        total_markdown = ""
        pages: list[str] = []
        page_texts: dict[int, str] = {}
        images: list[DocumentImageBlock] = []
        batch_size = 20
        for num in range(0, num_pages, batch_size):
            start_page = num + 1
            end_page = min(num + batch_size, num_pages)
            # Docling consumes/closes the stream on each convert() call,
            # so a fresh DocumentStream is built per batch inside _convert.
            result = self._convert(document_bytes, filename, start_page, end_page)
            self._assert_convert_ok(result, context="OCR convert")
            markdown = result.document.export_to_markdown(
                page_break_placeholder=PAGE_BREAK_PLACEHOLDER
            )
            total_markdown += markdown
            # Keep empty slots so page_no stays aligned with Docling provenance.
            batch_pages = [
                page.strip() for page in markdown.split(PAGE_BREAK_PLACEHOLDER)
            ]
            for offset, page_text in enumerate(batch_pages):
                page_texts[start_page + offset] = page_text
            pages.extend(page for page in batch_pages if page)
            images.extend(
                self._extract_images(result.document, page_texts=page_texts)
            )

        return ParsedDocument(
            filename=filename,
            text=total_markdown,
            pages=pages,
            images=images,
            docling_document=[],
        )

    def _parse_without_batching(
        self,
        document_bytes: bytes,
        *,
        filename: str,
    ) -> ParsedDocument:
        """Convert a whole document when no reliable page count is available."""
        log.info(
            "Page count unavailable; using whole-document conversion: filename=%s",
            filename,
        )
        result = self._convert(document_bytes, filename)
        self._assert_convert_ok(result, context="document convert")
        markdown = result.document.export_to_markdown(
            page_break_placeholder=PAGE_BREAK_PLACEHOLDER
        )
        aligned_pages = [
            page.strip() for page in markdown.split(PAGE_BREAK_PLACEHOLDER)
        ]
        pages = [page for page in aligned_pages if page]
        images = self._extract_images(result.document, pages=aligned_pages)

        return ParsedDocument(
            filename=filename,
            text=markdown,
            pages=pages,
            images=images,
            docling_document=[],
        )

    @staticmethod
    def _vlm_failed_pages(result) -> set[int] | None:
        """Page numbers the VLM could not finish, or None if the whole convert failed.

        A partial success (e.g. one page hit the output-token limit) must not
        discard every other page of the document.
        """
        status = getattr(result, "status", None)
        if status is None or status == ConversionStatus.SUCCESS:
            return set()
        if status != ConversionStatus.PARTIAL_SUCCESS:
            return None
        pages = {
            page_no
            for error in (getattr(result, "errors", None) or [])
            if isinstance(page_no := getattr(error, "page_no", None), int)
        }
        return pages or None

    @staticmethod
    def _replace_with_text_layer(
        document_bytes: bytes,
        aligned_pages: list[str],
        failed_pages: set[int],
        *,
        filename: str,
    ) -> list[str]:
        import pypdfium2

        pages = list(aligned_pages)
        pdf = pypdfium2.PdfDocument(document_bytes)
        try:
            while len(pages) < len(pdf):
                pages.append("")
            for page_no in sorted(failed_pages):
                if 1 <= page_no <= len(pdf):
                    text = pdf[page_no - 1].get_textpage().get_text_range()
                    pages[page_no - 1] = text.strip()
        finally:
            pdf.close()
        log.warning(
            "VLM could not finish page(s) %s of %s; used the PDF text layer instead",
            sorted(failed_pages),
            filename,
        )
        return pages

    def _parse_with_vlm(
        self, document_bytes: bytes, *, filename: str
    ) -> ParsedDocument:
        # Prefer whole-document convert under VLM (page-range batching is OCR-path).
        stream = DocumentStream(name=filename, stream=io.BytesIO(document_bytes))
        result = self._converter.convert(stream)
        failed_pages = self._vlm_failed_pages(result)
        if failed_pages is None:
            self._assert_convert_ok(result, context="VLM convert")
        markdown = result.document.export_to_markdown(
            page_break_placeholder=PAGE_BREAK_PLACEHOLDER
        )
        # Preserve empty pages for 1-based page_no alignment with pictures.
        aligned_pages = [
            page.strip() for page in markdown.split(PAGE_BREAK_PLACEHOLDER)
        ]
        if failed_pages:
            aligned_pages = self._replace_with_text_layer(
                document_bytes, aligned_pages, failed_pages, filename=filename
            )
            markdown = f"\n\n{PAGE_BREAK_PLACEHOLDER}\n\n".join(aligned_pages)
        if not markdown.strip():
            raise IngestionError(
                "Docling VLM convert returned empty markdown "
                "(possible Azure API failure or truncated response)"
            )
        pages = [page for page in aligned_pages if page]
        images: list[DocumentImageBlock] = []
        if self._analyze_images:
            images = self._extract_images_via_layout_pass(
                document_bytes, filename=filename, pages=aligned_pages
            )
        return ParsedDocument(
            filename=filename,
            text=markdown,
            pages=pages,
            images=images,
            docling_document=[],
        )

    def parse(
        self,
        document_bytes: bytes,
        *,
        filename: str = "input.pdf",
    ) -> ParsedDocument:
        try:
            if self._use_vlm:
                return self._parse_with_vlm(document_bytes, filename=filename)
            return self._parse_with_ocr(document_bytes, filename=filename)
        except IngestionError:
            raise
        except Exception as e:
            raise IngestionError(f"Docling document processing failed: {e}") from e
