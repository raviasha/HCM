"""
ChromaDB service for storing and querying Voice of Employee feedback embeddings.

Uses ChromaDB PersistentClient. Embeddings are fetched via the provider's
/api/embed proxy — the client never holds an OpenAI API key.

Each company gets its own collection for data isolation.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional

import chromadb
from chromadb.errors import NotFoundError
from chromadb.utils.embedding_functions import EmbeddingFunction

from client.config.settings import CHROMA_DB_PATH, PROVIDER_URL, EMBEDDING_MODEL


class _ProviderEmbeddingFunction(EmbeddingFunction):
    """Embedding function that calls the provider's /api/embed proxy."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        import httpx
        resp = httpx.post(
            f"{PROVIDER_URL}/api/embed",
            json={"texts": input, "model": EMBEDDING_MODEL},
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


# Module-level client (singleton)
_client: Optional[chromadb.PersistentClient] = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _client


def _get_embedding_fn() -> _ProviderEmbeddingFunction:
    return _ProviderEmbeddingFunction()


def _collection_name(company_name: str) -> str:
    """Sanitize company name into a valid ChromaDB collection name."""
    slug = re.sub(r"[^a-zA-Z0-9]", "_", company_name.lower()).strip("_")
    # ChromaDB requires 3-63 chars, starts/ends with alphanum
    slug = slug[:60]
    return f"fb_{slug}"


def ingest_feedback(company_name: str, feedback_list: list[dict]) -> int:
    """
    Embed and store all feedback entries for a company.
    Clears any existing data for this company first.

    Dynamically detects JSON keys — supports varying schemas across companies.
    The document text, ID, and department/unit fields are auto-detected from
    the first entry's keys.

    Returns:
        Number of documents ingested.
    """
    if not feedback_list:
        return 0

    # ── Auto-detect field names from the first entry ─────────────────
    sample = feedback_list[0]
    keys = set(sample.keys())

    # Text field: the employee's response / comment / answer
    _TEXT_CANDIDATES = ["response_text", "answer", "comment", "text", "feedback_text", "response"]
    text_key = next((k for k in _TEXT_CANDIDATES if k in keys), None)
    if text_key is None:
        # Fallback: pick the longest string value
        text_key = max(
            (k for k, v in sample.items() if isinstance(v, str) and len(v) > 20),
            key=lambda k: len(str(sample[k])),
            default=None,
        )
    if text_key is None:
        raise ValueError("Cannot detect text/response field in feedback JSON")

    # ID field
    _ID_CANDIDATES = ["feedback_id", "id", "entry_id"]
    id_key = next((k for k in _ID_CANDIDATES if k in keys), None)

    # Department / organizational unit field
    _DEPT_CANDIDATES = ["department", "dept", "unit", "team", "group", "store_region"]
    dept_key = next((k for k in _DEPT_CANDIDATES if k in keys), None)

    # Employee ID field
    _EMP_CANDIDATES = ["employee_id", "staff_id", "staff_number", "emp_id", "empid"]
    emp_key = next((k for k in _EMP_CANDIDATES if k in keys), None)

    # Feedback type / channel field
    _TYPE_CANDIDATES = ["feedback_type", "survey_type", "channel", "type", "source"]
    type_key = next((k for k in _TYPE_CANDIDATES if k in keys), None)

    # Date field
    _DATE_CANDIDATES = ["date", "submitted_at", "timestamp", "created_at"]
    date_key = next((k for k in _DATE_CANDIDATES if k in keys), None)

    # Question / prompt field
    _PROMPT_CANDIDATES = ["question_prompt", "question", "prompt"]
    prompt_key = next((k for k in _PROMPT_CANDIDATES if k in keys), None)

    # ── Ingest ───────────────────────────────────────────────────────
    client = _get_client()
    col_name = _collection_name(company_name)

    # Delete existing collection if present (fresh ingest)
    try:
        client.delete_collection(col_name)
    except (ValueError, NotFoundError, Exception):
        pass

    collection = client.get_or_create_collection(
        name=col_name,
        embedding_function=_get_embedding_fn(),
    )

    ids = []
    documents = []
    metadatas = []

    for idx, entry in enumerate(feedback_list):
        doc_id = str(entry[id_key]) if id_key else str(idx + 1)
        text = str(entry.get(text_key, ""))
        if not text.strip():
            continue

        # Build metadata with normalized keys so downstream code works
        meta: dict = {}
        if emp_key:
            # Hash employee_id for GDPR — original PII never stored
            raw_id = str(entry.get(emp_key, ""))
            meta["employee_id"] = hashlib.sha256(raw_id.encode()).hexdigest()[:16] if raw_id else ""
        if dept_key:
            meta["department"] = str(entry.get(dept_key, ""))
        if type_key:
            meta["feedback_type"] = str(entry.get(type_key, ""))
        if date_key:
            meta["date"] = str(entry.get(date_key, ""))
        if prompt_key:
            meta["question_prompt"] = str(entry.get(prompt_key, ""))

        ids.append(doc_id)
        documents.append(text)
        metadatas.append(meta)

    # ChromaDB supports batch upsert (max recommended ~5000 per call)
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    return len(ids)


def query_feedback(
    company_name: str,
    query_text: str,
    department: Optional[str] = None,
    n_results: int = 20,
) -> list[dict]:
    """
    Semantic search over feedback for a company.

    Returns list of dicts with keys: id, document, metadata, distance.
    """
    client = _get_client()
    col_name = _collection_name(company_name)

    try:
        collection = client.get_collection(
            name=col_name,
            embedding_function=_get_embedding_fn(),
        )
    except (ValueError, NotFoundError):
        return []

    where_filter = None
    if department:
        where_filter = {"department": department}

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where_filter,
    )

    # Flatten ChromaDB response structure
    items = []
    if results and results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            items.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })

    return items


def get_all_feedback(company_name: str) -> list[dict]:
    """
    Retrieve all stored feedback for a company.
    Returns list of dicts with keys: id, document, metadata.
    """
    client = _get_client()
    col_name = _collection_name(company_name)

    try:
        collection = client.get_collection(
            name=col_name,
            embedding_function=_get_embedding_fn(),
        )
    except (ValueError, NotFoundError):
        return []

    results = collection.get(include=["documents", "metadatas"])

    items = []
    if results and results["ids"]:
        for i in range(len(results["ids"])):
            items.append({
                "id": results["ids"][i],
                "document": results["documents"][i],
                "metadata": results["metadatas"][i],
            })

    return items


def get_feedback_by_department(company_name: str) -> Dict[str, List[str]]:
    """
    Get all feedback text grouped by department.
    Returns: {"Engineering": ["text1", "text2", ...], ...}
    """
    all_fb = get_all_feedback(company_name)
    by_dept: Dict[str, List[str]] = {}
    for item in all_fb:
        dept = item["metadata"]["department"]
        if dept not in by_dept:
            by_dept[dept] = []
        by_dept[dept].append(item["document"])
    return by_dept


def get_feedback_count(company_name: str) -> int:
    """Return total number of feedback entries for a company."""
    client = _get_client()
    col_name = _collection_name(company_name)
    try:
        collection = client.get_collection(
            name=col_name,
            embedding_function=_get_embedding_fn(),
        )
        return collection.count()
    except (ValueError, NotFoundError):
        return 0


def delete_feedback(company_name: str) -> bool:
    """Delete all feedback data for a company. Returns True if collection existed."""
    client = _get_client()
    col_name = _collection_name(company_name)
    try:
        client.delete_collection(col_name)
        return True
    except (ValueError, NotFoundError, Exception):
        return False
