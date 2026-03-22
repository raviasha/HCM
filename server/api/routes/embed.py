"""
Embedding proxy route.

Client sends PII-scrubbed text; server calls OpenAI embedding API
using the server-side API key and returns vectors.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from shared.contracts import EmbedRequest, EmbedResponse
from server.api.services import openai_service

router = APIRouter()


@router.post("/api/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    """Generate embeddings for PII-scrubbed texts."""
    if not req.texts:
        return EmbedResponse(embeddings=[])
    if len(req.texts) > 500:
        raise HTTPException(status_code=400, detail="Maximum 500 texts per request")
    try:
        vectors = openai_service.generate_embeddings(
            texts=req.texts,
            model=req.model,
        )
        return EmbedResponse(embeddings=vectors)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
