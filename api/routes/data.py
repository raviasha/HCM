"""
Structured data API routes.
Handles CSV upload and generic data queries.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from api.models.schemas import UploadResponse
from api.services.data_service import get_backend
from api.services import chroma_service
from api.services.audit_service import log_event

router = APIRouter(prefix="/api/data", tags=["Structured Data"])


@router.post("/upload-csv", response_model=UploadResponse)
async def upload_csv(
    file: UploadFile = File(...),
    company_name: str = Form(...),
):
    """Upload a CSV file containing employee/workforce data."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv")

    content = await file.read()
    try:
        backend = get_backend()
        row_count = backend.load(content, company_name)
        log_event("data_upload_csv", company_name, {"row_count": row_count})
        return UploadResponse(
            success=True,
            message=f"Loaded {row_count} records for {company_name}",
            row_count=row_count,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")


@router.delete("/{company_name}")
async def delete_company_data(company_name: str):
    """
    Delete all data for a company (GDPR Article 17 — Right to Erasure).
    Removes structured data, schema mappings, cached insights, and feedback.
    """
    from api.routes.insights import _insights_cache

    backend = get_backend()
    data_deleted = backend.delete(company_name)
    feedback_deleted = chroma_service.delete_feedback(company_name)
    _insights_cache.pop(company_name, None)

    if not data_deleted and not feedback_deleted:
        raise HTTPException(status_code=404, detail=f"No data found for '{company_name}'")

    log_event("data_deletion", company_name, {
        "structured_data_deleted": data_deleted,
        "feedback_deleted": feedback_deleted,
    })

    return {
        "success": True,
        "message": f"All data for '{company_name}' has been deleted.",
        "structured_data_deleted": data_deleted,
        "feedback_deleted": feedback_deleted,
    }
