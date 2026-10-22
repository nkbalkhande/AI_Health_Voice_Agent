from backend.domain.entities.document import Document
from backend.rag.chunking.ids import deterministic_chunk_id
from backend.rag.ingestion.models import ParsedDocument


def base_metadata(
    parsed: ParsedDocument,
    *,
    metadata: dict | None,
) -> dict:
    return {**(metadata or {}), "filename": parsed.filename}


def append_image_documents(
    documents: list[Document],
    parsed: ParsedDocument,
    *,
    base_meta: dict,
    resolved_source: str,
) -> None:
    for index, image in enumerate(parsed.images):
        analysis = (image.analysis or "").strip()
        caption = (image.caption or "").strip()
        content = analysis or caption
        if not content:
            continue
        documents.append(
            Document(
                id=deterministic_chunk_id(
                    resolved_source,
                    page=image.page_no,
                    chunk_index=index,
                    content=content,
                ),
                content=content,
                metadata={
                    **base_meta,
                    # Parser-supplied provenance (XLSX sheet/anchor) first, so
                    # the fixed keys below always win.
                    **{k: v for k, v in image.metadata.items() if v not in (None, "")},
                    "content_type": "image",
                    "image_index": index,
                    "chunk_index": index,
                    "page_number": image.page_no,
                },
                source=resolved_source,
            )
        )
