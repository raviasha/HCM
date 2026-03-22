"""
Client-side configuration.

All data storage paths and provider backend URL.
The OpenAI API key is NOT needed here — all LLM calls go through the provider.
"""

import os

# ── Provider backend URL ─────────────────────────────────────────────
PROVIDER_URL = os.getenv("PROVIDER_URL", "http://localhost:8000")

# ── Local ChromaDB ───────────────────────────────────────────────────
CHROMA_DB_PATH = os.getenv(
    "CHROMA_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db"),
)

# ── Embedding model (sent to provider embed proxy) ──────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ── Client API ───────────────────────────────────────────────────────
API_HOST = os.getenv("CLIENT_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("CLIENT_API_PORT", "8001"))

# ── Streamlit ────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", f"http://localhost:{API_PORT}")
