from __future__ import annotations

import io
import json
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Iterable, Mapping, Sequence

import google.generativeai as genai
import psycopg2
from psycopg2 import Error as PsycopgError
from psycopg2.extras import RealDictCursor

EMBEDDING_MODEL = "text-embedding-004"


class RagSearchError(RuntimeError):
    """Domain-specific error raised when RAG search fails."""


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RagSearchError(f"Environment variable {name} is required for RAG search")
    return value


def get_db_url() -> str:
    """Return the Postgres connection URL for the RAG database."""

    return _require_env("SUPABASE_DB_URL")


def configure_gemini() -> str:
    """Configure Gemini client using the GOOGLE_API_KEY environment variable."""

    api_key = _require_env("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    return api_key


def _coerce_embedding_values(raw: Any) -> Sequence[float]:
    candidate = raw
    if isinstance(candidate, Mapping) and "values" in candidate:
        candidate = candidate.get("values")
    if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
        # Gemini may return a list with a single dict that stores values
        if candidate and isinstance(candidate[0], Mapping) and "values" in candidate[0]:
            candidate = candidate[0].get("values")
    if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes)):
        raise RagSearchError("Embedding payload has an unexpected format")
    return [float(value) for value in candidate]


def _extract_embedding(response: Any) -> Sequence[float]:
    embedding = None
    if hasattr(response, "embedding"):
        embedding = getattr(response, "embedding")
    if embedding is None and isinstance(response, Mapping):
        embedding = response.get("embedding")
    if embedding is None and hasattr(response, "embeddings"):
        embedding = getattr(response, "embeddings")
    if embedding is None and isinstance(response, Mapping):
        embedding = response.get("embeddings")
    if embedding is None:
        raise RagSearchError("Embedding response does not contain 'embedding' data")
    return _coerce_embedding_values(embedding)


def embed_query(query_text: str, model: str = EMBEDDING_MODEL) -> Sequence[float]:
    """Return the embedding for the provided query text using Gemini."""

    if not query_text.strip():
        raise RagSearchError("Search query must be non-empty")
    embedding_response = genai.embed_content(model=model, content=query_text)
    return _extract_embedding(embedding_response)


def search_raw_chunks(
    embedding: Iterable[float],
    match_threshold: float = 0.5,
    match_count: int = 5,
) -> list[Mapping[str, Any]]:
    """Search for raw chunks using the match_chunks stored procedure."""

    db_url = get_db_url()
    embedding_vector = list(float(value) for value in embedding)
    try:
        with psycopg2.connect(db_url, cursor_factory=RealDictCursor) as conn:
            with conn.cursor() as cursor:
                cursor.callproc("match_chunks", (embedding_vector, match_threshold, match_count))
                rows = cursor.fetchall()
                return [dict(row) for row in rows] if rows else []
    except PsycopgError as exc:  # pragma: no cover - depends on external DB
        logging.error("Failed to call match_chunks via callproc: %s", exc)
        raise RagSearchError("Failed to search chunks") from exc


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return str(value)


def build_raw_answer_file(payload: Mapping[str, Any]) -> tuple[str, bytes]:
    """Build a timestamped JSON file containing the raw RAG payload."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"rag_raw_answer_{timestamp}.json"
    buffer = io.BytesIO()
    buffer.write(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default).encode("utf-8"))
    buffer.seek(0)
    return filename, buffer.read()


def run_rag_search(query_text: str, match_count: int = 5) -> dict[str, Any]:
    """Execute the full RAG search flow and return metadata alongside matches."""

    configure_gemini()
    embedding = embed_query(query_text)
    results = search_raw_chunks(embedding, match_count=match_count)
    executed_at = datetime.now(timezone.utc).isoformat()
    return {
        "query": {
            "text": query_text,
            "embedding_model": EMBEDDING_MODEL,
            "match_count": match_count,
            "executed_at": executed_at,
        },
        "results": results,
        "metadata": {
            "result_count": len(results),
            "embedding_dimensions": len(embedding),
        },
    }


def build_raw_answer_document(payload: Mapping[str, Any]) -> tuple[str, bytes]:
    """Backward compatible alias for build_raw_answer_file."""

    return build_raw_answer_file(payload)

