from __future__ import annotations

import io
import json
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Mapping, Sequence

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


def _extract_embedding(response: Any) -> Sequence[float]:
    embedding = None
    if hasattr(response, "embedding"):
        embedding = getattr(response, "embedding")
    if embedding is None and isinstance(response, Mapping):
        embedding = response.get("embedding")
    if embedding is None:
        raise RagSearchError("Embedding response does not contain 'embedding' data")
    if isinstance(embedding, (str, bytes)) or not isinstance(embedding, Sequence):
        raise RagSearchError("Embedding payload has an unexpected format")
    return embedding


def _format_vector_literal(embedding: Sequence[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in embedding) + "]"


def _execute_match_chunks(cursor, embedding_literal: str, match_count: int):
    attempts = [
        ("SELECT * FROM match_chunks(%s::vector, %s)", (embedding_literal, match_count)),
        ("SELECT * FROM match_chunks(%s::vector)", (embedding_literal,)),
    ]
    last_error: PsycopgError | None = None
    for sql, params in attempts:
        try:
            cursor.execute(sql, params)
            records = cursor.fetchall()
            return records, sql
        except PsycopgError as exc:  # pragma: no cover - depends on external DB
            logging.warning("Failed to call match_chunks using query '%s': %s", sql, exc)
            if cursor.connection:  # pragma: no cover - defensive
                cursor.connection.rollback()
            last_error = exc
    if last_error:
        raise last_error
    raise RagSearchError("Unable to execute match_chunks query")


def run_rag_search(query_text: str, match_count: int = 10) -> dict[str, Any]:
    api_key = _require_env("GOOGLE_API_KEY")
    supabase_db_url = _require_env("SUPABASE_DB_URL")

    if not query_text.strip():
        raise RagSearchError("Search query must be non-empty")

    genai.configure(api_key=api_key)
    embedding_response = genai.embed_content(model=EMBEDDING_MODEL, content=query_text)
    embedding = _extract_embedding(embedding_response)
    embedding_literal = _format_vector_literal(embedding)

    with psycopg2.connect(supabase_db_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cursor:
            rows, sql_used = _execute_match_chunks(cursor, embedding_literal, match_count)

    results = [dict(row) for row in rows] if rows else []
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
            "sql_used": sql_used,
            "embedding_dimensions": len(embedding),
        },
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return str(value)


def build_raw_answer_document(payload: Mapping[str, Any]) -> tuple[str, bytes]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"rag_raw_answer_{timestamp}.json"
    buffer = io.BytesIO()
    buffer.write(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default).encode("utf-8"))
    buffer.seek(0)
    return filename, buffer.read()
