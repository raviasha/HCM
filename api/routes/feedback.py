"""
Voice of Employee feedback API routes.
Handles JSON upload, ChromaDB ingestion, and semantic search.
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from api.models.schemas import UploadResponse
from api.services import chroma_service

router = APIRouter(prefix="/api/feedback", tags=["Voice of Employee"])


@router.post("/upload", response_model=UploadResponse)
async def upload_feedback(
    file: UploadFile = File(...),
    company_name: str = Form(...),
):
    """Upload a JSON file containing employee feedback and ingest into ChromaDB."""
    if not file.filename or not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="File must be a .json")

    content = await file.read()
    try:
        feedback_list = json.loads(content)
        if not isinstance(feedback_list, list):
            raise ValueError("JSON must be an array of feedback objects")

        count = chroma_service.ingest_feedback(company_name, feedback_list)
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
    """Semantic search over employee feedback."""
    results = chroma_service.query_feedback(
        company_name=company_name,
        query_text=q,
        department=department,
        n_results=n_results,
    )
    return {"results": results, "count": len(results)}


@router.get("/by-department")
async def get_feedback_by_department(company_name: str):
    """Get all feedback grouped by department."""
    data = chroma_service.get_feedback_by_department(company_name)
    summary = {dept: len(texts) for dept, texts in data.items()}
    return {"departments": summary, "total": sum(summary.values())}


@router.get("/count")
async def get_feedback_count(company_name: str):
    """Get total feedback entry count for a company."""
    count = chroma_service.get_feedback_count(company_name)
    return {"company_name": company_name, "count": count}
