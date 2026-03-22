"""
Client-side feedback routes.

Feedback JSON is ingested into local ChromaDB.
Embeddings are fetched via the provider's /api/embed proxy.
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from shared.contracts import UploadResponse
from client.services import chroma_service
from client.services.audit_service import log_event

router = APIRouter(prefix="/api/data", tags=["Qualitative Data"])


@router.post("/upload-json", response_model=UploadResponse)
async def upload_feedback(
    file: UploadFile = File(...),
    company_name: str = Form(...),
):
    """Upload feedback JSON — data stays local, embeddings via provider proxy."""
    if not file.filename or not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="File must be a .json")

    content = await file.read()
    try:
        feedback_list = json.loads(content)
        if not isinstance(feedback_list, list):
            raise ValueError("JSON must be an array of feedback objects")

        count = chroma_service.ingest_feedback(company_name, feedback_list)
        log_event("data_upload_feedback", company_name, {"entry_count": count})
        return UploadResponse(
            success=True,
            message=f"Ingested {count} feedback entries for {company_name}",
            row_count=count,
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error ingesting feedback: {str(e)}")


@router.get("/search")
async def search_feedback(
    company_name: str,
    q: str,
    department: Optional[str] = None,
    n_results: int = 20,
):
    """Semantic search over local feedback."""
    results = chroma_service.query_feedback(
        company_name=company_name,
        query_text=q,
        department=department,
        n_results=n_results,
    )
    return {"results": results, "count": len(results)}


@router.get("/by-department")
async def get_feedback_by_department(company_name: str):
    """Get all feedback grouped by department from local ChromaDB."""
    data = chroma_service.get_feedback_by_department(company_name)
    summary = {dept: len(texts) for dept, texts in data.items()}
    return {"departments": summary, "total": sum(summary.values())}


@router.get("/count")
async def get_feedback_count(company_name: str):
    """Get total feedback entry count."""
    count = chroma_service.get_feedback_count(company_name)
    return {"company_name": company_name, "count": count}
