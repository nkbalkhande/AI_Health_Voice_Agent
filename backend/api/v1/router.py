from fastapi import APIRouter

from backend.api.v1 import chat, evaluation, ingestion, query, voice, ws

api_router = APIRouter(prefix="/v1")
api_router.include_router(query.router)
api_router.include_router(chat.router)
api_router.include_router(voice.router)
api_router.include_router(ws.router)
api_router.include_router(ingestion.router)
api_router.include_router(evaluation.router)

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
