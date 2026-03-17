"""
Structured data API routes.
Handles CSV upload and generic data queries.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from api.models.schemas import UploadResponse
from api.services.data_service import get_backend

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
        return UploadResponse(
            success=True,
            message=f"Loaded {row_count} records for {company_name}",
            row_count=row_count,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")
