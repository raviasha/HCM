"""
Server-side configuration for the HCM AI Insights provider backend.
Only needs OpenAI credentials — no data storage configuration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

SERVER_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(SERVER_ROOT / ".env", override=True)

# --- OpenAI ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL: str = "text-embedding-3-small"
CHAT_MODEL: str = "gpt-4o"

# --- Server ---
API_HOST: str = "0.0.0.0"
API_PORT: int = int(os.getenv("PORT", 8000))
