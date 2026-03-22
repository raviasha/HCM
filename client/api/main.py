"""
Client container — FastAPI entry point.

Handles all data operations locally. Only anonymized/aggregated data
is sent to the provider backend for GPT-4o insight generation.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from client.config.settings import API_HOST, API_PORT
from client.api.routes import data, feedback, insights

app = FastAPI(
    title="HCM AI Client",
    description="Client container — data stays local, insights via provider",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data.router, tags=["data"])
app.include_router(feedback.router, tags=["feedback"])
app.include_router(insights.router, tags=["insights"])


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "client"}


@app.get("/")
async def root():
    return {
        "service": "HCM AI Client Container",
        "version": "2.0.0",
        "endpoints": [
            "/api/data/upload-csv",
            "/api/data/upload-json",
            "/api/data/analyze-schema",
            "/api/data/approve-schema",
            "/api/insights/generate",
            "/api/insights/ask",
            "/health",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
