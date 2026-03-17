"""
FastAPI application entry point.
Configures CORS, includes all route modules, and provides health check.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import data, feedback, insights

app = FastAPI(
    title="HCM AI Insights API",
    description=(
        "Backend API for the Human Capital Management AI Insights prototype. "
        "Provides endpoints for data ingestion (structured CSV + qualitative JSON), "
        "data querying, and AI-powered insight generation via GPT-4o."
    ),
    version="0.2.0",
)

# CORS — allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For prototype; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(data.router)
app.include_router(feedback.router)
app.include_router(insights.router)


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "HCM AI Insights API",
        "version": "0.2.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}
