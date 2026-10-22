import asyncio
import sys
from contextlib import asynccontextmanager

import litellm
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from backend.api.v1.router import api_router
from backend.core.config import settings
from backend.core.logger import logger

# R-69: litellm globals belong to the process entry point, not to whichever
# provider module happens to be imported first. The provider modules set the
# same flags defensively for direct/script use.
litellm.suppress_debug_info = True
litellm.set_verbose = False
litellm.drop_params = True

# Psycopg async (LangGraph Postgres checkpointer) cannot run on the
# Windows Proactor loop, so the API uses the selector loop there.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the cached providers before serving traffic.

    R-82: lifespan did nothing, so the first request paid the Qdrant
    connection, the BM25 sparse-model download and every provider
    construction — and a misconfiguration (missing collection, bad
    credentials, unknown provider) surfaced as a 500 on a user's query
    instead of a failed boot.
    """
    logger.info("Starting %s", settings.app_name)

    # Imported here so importing backend.main stays cheap for tooling.
    from backend.api.deps import (
        get_evaluate_pipeline_use_case,
        get_ingest_use_case,
        get_query_use_case,
    )

    def _warm() -> None:
        get_query_use_case()
        get_ingest_use_case()
        get_evaluate_pipeline_use_case()

    # Provider construction is blocking (network + model loads).
    await run_in_threadpool(_warm)
    from backend.api.deps import get_chat_use_case

    # The LangGraph checkpointer is shared process-wide; opening it here
    # keeps its multi-second setup off the first conversation turn.
    await get_chat_use_case().warm_up()
    logger.info(
        "Providers warm: collection=%s generator=%s judge=%s",
        settings.vectordb_collection,
        settings.llm_model,
        settings.evaluation_model,
    )

    yield
    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix="/api")
    return app


app = create_app()
