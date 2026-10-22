from fastapi import APIRouter, Depends, File, Form, UploadFile

from backend.api.deps import get_ingestion_facade
from backend.api.schemas import IngestPdfResponse
from backend.core.exceptions import IngestionError
from backend.core.logger import logger
from backend.rag.ingestion.facade.ingestion_facade import IngestionFacade

log = logger.getChild("api.v1.ingestion")

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post(
    "/pdf",
    response_model=IngestPdfResponse,
    summary="Index a document into the vector store",
)
async def ingest_document(
    file: UploadFile = File(
        ...,
        description=(
            "Document to parse, chunk, embed, and index. "
            "Supported: PDF, DOCX, PPTX, XLSX, TXT, HTML/HTM."
        ),
    ),
    source: str | None = Form(
        default=None,
        description=(
            "Optional source label stored on each chunk. "
            "Defaults to the uploaded filename when omitted."
        ),
    ),
    facade: IngestionFacade = Depends(get_ingestion_facade),
) -> IngestPdfResponse:
    """Parse a document and index its chunks into the configured vector database.

    Pipeline:
    1. Select a format-specific parser via the ingestion facade.
    2. Extract text (and images when applicable) into ParsedDocument.
    3. Chunk content with the configured strategy (semantic or hybrid).
    4. Append ``File name : <filename>`` to every chunk before embedding.
    5. Embed chunks and upsert them into the vector store.

    Returns document/chunk counts plus a parsed-document summary. Returns ``422``
    when the document has no extractable content or the file type is unsupported.
    """
    document_bytes = await file.read()
    filename = file.filename or source or "input.pdf"
    log.info("Ingesting file: %s", filename)
    try:
        result = await facade.ingest(
            document_bytes,
            filename=filename,
            source=source or filename,
            metadata={"mime_type": file.content_type},
        )
        log.info("Ingested %s successfully", result.filename)
    except IngestionError as e:
        from fastapi import HTTPException

        raise HTTPException(status_code=422, detail=str(e)) from e

    return IngestPdfResponse(
        document_count=result.document_count,
        chunk_count=result.chunk_count,
        filename=result.filename,
        page_count=result.page_count,
        image_count=result.image_count,
        has_content=result.has_content,
    )
