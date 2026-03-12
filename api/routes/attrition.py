"""
Attrition data API routes.
Handles CSV upload and structured data queries.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from api.models.schemas import UploadResponse, DepartmentAttrition, SummaryKPIs
from api.services.data_service import get_backend

router = APIRouter(prefix="/api/attrition", tags=["Attrition Data"])


@router.post("/upload", response_model=UploadResponse)
async def upload_attrition_data(
    file: UploadFile = File(...),
    company_name: str = Form(...),
):
    """Upload a CSV file containing employee attrition data."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv")

    content = await file.read()
    try:
        backend = get_backend()
        row_count = backend.load(content, company_name)
        return UploadResponse(
            success=True,
            message=f"Loaded {row_count} employee records for {company_name}",
            row_count=row_count,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")


@router.get("/by-department", response_model=list[DepartmentAttrition])
async def get_attrition_by_department(company_name: str):
    """Get attrition statistics grouped by department."""
    backend = get_backend()
    if not backend.is_loaded(company_name):
        raise HTTPException(status_code=404, detail=f"No data for {company_name}")
    return backend.get_attrition_by_department(company_name)


@router.get("/risk-factors")
async def get_risk_factors(company_name: str):
    """Get top attrition risk factors ranked by correlation."""
    backend = get_backend()
    if not backend.is_loaded(company_name):
        raise HTTPException(status_code=404, detail=f"No data for {company_name}")
    return backend.get_risk_factors(company_name)


@router.get("/kpis", response_model=SummaryKPIs)
async def get_kpis(company_name: str):
    """Get summary KPIs for the loaded dataset."""
    backend = get_backend()
    if not backend.is_loaded(company_name):
        raise HTTPException(status_code=404, detail=f"No data for {company_name}")
    return backend.get_summary_kpis(company_name)


@router.get("/department-stats")
async def get_department_stats(company_name: str):
    """Get detailed statistics per department."""
    backend = get_backend()
    if not backend.is_loaded(company_name):
        raise HTTPException(status_code=404, detail=f"No data for {company_name}")
    return backend.get_department_stats(company_name)


@router.get("/overtime")
async def get_overtime_analysis(company_name: str):
    """Get overtime vs attrition analysis."""
    backend = get_backend()
    if not backend.is_loaded(company_name):
        raise HTTPException(status_code=404, detail=f"No data for {company_name}")
    return backend.get_overtime_analysis(company_name)
