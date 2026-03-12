"""
Centralized configuration for the HCM AI Insights application.
Loads environment variables and exposes typed settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

# --- OpenAI ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL: str = "text-embedding-3-small"
CHAT_MODEL: str = "gpt-4o"

# --- ChromaDB ---
CHROMA_DB_PATH: str = str(PROJECT_ROOT / "data" / "chroma_db")

# --- Data ---
SYNTHETIC_DATA_DIR: str = str(PROJECT_ROOT / "data" / "synthetic")
DATA_BACKEND: str = "pandas"  # Future: "duckdb"

# --- API ---
API_HOST: str = "0.0.0.0"
API_PORT: int = int(os.getenv("PORT", 8000))
API_BASE_URL: str = os.getenv("API_BASE_URL", f"http://localhost:{API_PORT}")

# --- Streamlit ---
STREAMLIT_PORT: int = 8501
