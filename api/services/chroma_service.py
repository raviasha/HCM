"""
ChromaDB service for storing and querying Voice of Employee feedback embeddings.

Uses ChromaDB PersistentClient with OpenAI embeddings (text-embedding-3-small).
Each company gets its own collection for data isolation.

Production upgrade: Replace local PersistentClient with ChromaDB cloud or
a managed vector database for multi-tenant, high-volume workloads.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import chromadb
from chromadb.errors import NotFoundError
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from config.settings import CHROMA_DB_PATH, OPENAI_API_KEY, EMBEDDING_MODEL


# Module-level client (singleton)
_client: Optional[chromadb.PersistentClient] = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _client


def _get_embedding_fn() -> OpenAIEmbeddingFunction:
    return OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL,
    )


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

    Args:
        company_name: Name of the company (used to scope the collection).
        feedback_list: List of feedback dicts with keys:
            feedback_id, employee_id, feedback_type, date,
            question_prompt, response_text, department

    Returns:
        Number of documents ingested.
    """
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

    # Prepare batch data
    ids = []
    documents = []
    metadatas = []

    for entry in feedback_list:
        ids.append(str(entry["feedback_id"]))
        documents.append(entry["response_text"])
        metadatas.append({
            "employee_id": str(entry["employee_id"]),
            "department": entry["department"],
            "feedback_type": entry["feedback_type"],
            "date": entry["date"],
            "question_prompt": entry.get("question_prompt", ""),
        })

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
