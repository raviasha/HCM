"""
Provider backend — FastAPI entry point.

Exposes ONLY GPT-4o orchestration and embedding proxy endpoints.
No raw data storage, no dataframe handling, no ChromaDB.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.config.settings import API_HOST, API_PORT
from server.api.routes import insights, embed

app = FastAPI(
    title="HCM AI Provider",
    description="Provider backend for HCM AI Insights — GPT-4o orchestration only",
    version="2.0.0",
)

# CORS — allow client containers to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(insights.router, tags=["insights"])
app.include_router(embed.router, tags=["embed"])


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "provider"}


@app.get("/")
async def root():
    return {
        "service": "HCM AI Provider Backend",
        "version": "2.0.0",
        "endpoints": [
            "/api/analyze-schema",
            "/api/generate-plan",
            "/api/generate-insights",
            "/api/ask",
            "/api/validate-pii",
            "/api/embed",
            "/health",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
