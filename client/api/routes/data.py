"""
Client-side structured data routes.

All data stays local. PII classification and schema review happen here.
AI validation goes through the provider's /api/validate-pii endpoint.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from shared.contracts import (
    UploadResponse,
    SchemaReviewResponse,
    ColumnClassificationOut,
    ApproveSchemaRequest,
    FeedbackSampleOut,
    PiiDetectionOut,
)
from client.services.data_service import get_backend
from client.services import chroma_service
from client.services.audit_service import log_event
from client.services import provider_client
from client.services.pii_classifier import (
    ColumnClassification,
    classify_columns,
    classify_feedback_keys,
    apply_pii_policy,
    scrub_text_pii,
)

router = APIRouter(prefix="/api/data", tags=["Structured Data"])


@router.post("/upload-csv", response_model=UploadResponse)
async def upload_csv(
    file: UploadFile = File(...),
    company_name: str = Form(...),
):
    """Upload a CSV file — data stays in the local container."""
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
    """Delete all local data for a company (GDPR Article 17)."""
    from client.api.routes.insights import _insights_cache

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


# ── Schema Review & PII Consent ──────────────────────────────────────

@router.post("/analyze-schema", response_model=SchemaReviewResponse)
async def analyze_schema(company_name: str = Form(...)):
    """
    Step 2: Classify columns for PII locally, then validate via provider AI.
    Returns schema with classifications for user review.
    """
    backend = get_backend()
    if not backend.is_loaded(company_name):
        raise HTTPException(status_code=400, detail=f"No data loaded for '{company_name}'")

    df = backend._df(company_name)

    # 1. Deterministic Python PII classification (runs locally)
    classifications = classify_columns(df)

    # 2. AI validation via provider (sends only column names + dtypes)
    class_dicts = [c.to_dict() for c in classifications]
    try:
        validated = provider_client.validate_pii(class_dicts)
    except Exception:
        validated = class_dicts

    # 3. Apply PII policy to sample rows
    sample_rows = df.head(5).fillna("").to_dict(orient="records")
    cc_list = [ColumnClassification(**v) for v in validated]
    safe_sample = apply_pii_policy(sample_rows, cc_list)

    # 4. PII summary
    pii_summary: dict[str, int] = {}
    for c in validated:
        cat = c["pii_category"]
        pii_summary[cat] = pii_summary.get(cat, 0) + 1

    # 5. Feedback PII detection
    feedback_cols: list[ColumnClassificationOut] = []
    feedback_samples: list[FeedbackSampleOut] = []
    fb_count = chroma_service.get_feedback_count(company_name)
    fb_text_pii_count = 0
    if fb_count > 0:
        all_fb = chroma_service.get_all_feedback(company_name)
        if all_fb:
            sample_meta = all_fb[0].get("metadata", {})
            fb_classifications = classify_feedback_keys(sample_meta)
            feedback_cols = [
                ColumnClassificationOut(**c.to_dict()) for c in fb_classifications
            ]
            for entry in all_fb[:5]:
                original_text = entry.get("document", "")
                scrubbed, detections = scrub_text_pii(original_text)
                fb_text_pii_count += len(detections)
                feedback_samples.append(FeedbackSampleOut(
                    original_text=original_text,
                    scrubbed_text=scrubbed,
                    pii_detections=[PiiDetectionOut(**d.to_dict()) for d in detections],
                    metadata=entry.get("metadata", {}),
                ))

    log_event("schema_analysis", company_name, {
        "column_count": len(validated),
        "pii_summary": pii_summary,
    })

    return SchemaReviewResponse(
        company_name=company_name,
        row_count=len(df),
        column_count=len(df.columns),
        columns=[ColumnClassificationOut(**c) for c in validated],
        pii_summary=pii_summary,
        sample_rows=safe_sample,
        feedback_columns=feedback_cols,
        feedback_samples=feedback_samples,
        feedback_entry_count=fb_count,
        feedback_text_pii_count=fb_text_pii_count,
    )


_STRICTNESS = {"safe": 0, "quasi_identifier": 1, "direct_pii": 2, "identifier": 3}


@router.post("/approve-schema")
async def approve_schema(request: ApproveSchemaRequest):
    """Step 3: User approves PII classification (can only tighten)."""
    backend = get_backend()
    if not backend.is_loaded(request.company_name):
        raise HTTPException(status_code=400, detail=f"No data loaded for '{request.company_name}'")

    existing = backend.get_pii_classification(request.company_name)
    if existing:
        orig_map = {c.column_name: c for c in existing}
        for col in request.columns:
            orig = orig_map.get(col.column_name)
            if orig:
                orig_level = _STRICTNESS.get(orig.pii_category, 0)
                new_level = _STRICTNESS.get(col.pii_category, 0)
                if new_level < orig_level:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Cannot downgrade '{col.column_name}' from "
                               f"'{orig.pii_category}' to '{col.pii_category}'",
                    )

    approved = [
        ColumnClassification(
            column_name=c.column_name,
            dtype=c.dtype,
            pii_category=c.pii_category,
            handling=c.handling,
            confidence=c.confidence,
            reason=c.reason,
        )
        for c in request.columns
    ]
    backend.set_pii_classification(request.company_name, approved)

    if request.feedback_columns:
        fb_approved = [
            ColumnClassification(
                column_name=c.column_name,
                dtype=c.dtype,
                pii_category=c.pii_category,
                handling=c.handling,
                confidence=c.confidence,
                reason=c.reason,
            )
            for c in request.feedback_columns
        ]
        backend.set_feedback_pii_classification(request.company_name, fb_approved)

    log_event("schema_approval", request.company_name, {
        "columns_approved": len(approved),
    })

    return {
        "success": True,
        "message": f"Schema approved for '{request.company_name}' — {len(approved)} columns classified.",
    }
