from __future__ import annotations

import io
import json
import logging
import os
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, TypedDict

import google.generativeai as genai
import psycopg2
from psycopg2 import Error as PsycopgError
from psycopg2.extras import RealDictCursor

EMBEDDING_MODEL = "text-embedding-004"
GROUNDED_ANSWER_MODEL_ID = "gemini-1.5-flash"


class RagSearchError(RuntimeError):
    """Domain-specific error raised when RAG search fails."""


class MatchChunkRow(TypedDict, total=False):
    """Typed representation of a row returned by the ``match_chunks`` function."""

    id: int | None
    book_id: int | str | None
    chunk_id: str | None
    content: str | None
    chunk: str | None
    similarity: float | None
    chunk_type: str | None
    lang: str | None
    relevance_score: float | None
    score: float | None
    match_score: float | None
    topic: str | None
    keywords: list[str] | None
    persons: list[str] | None
    locations: list[str] | None
    orthography: str | None
    chunk_year_start: int | None
    chunk_year_end: int | None
    media_ids: list[int] | None
    media_internal_ids: list[str] | None
    tg_msg_id: str | int | None
    scan_page_ids: list[int] | None
    scan_tg_msg_ids: list[int] | None
    ocr_tg_msg_ids: list[int] | None
    book_page: int | None
    book_title: str | None
    book_authors: list[str] | str | None
    book_year: int | None
    book_isbn: str | None
    source_link: str | None
    source_links: list[str] | None
    paragraph: str | None
    url: str | None


class RagSearchQuery(TypedDict):
    """Search metadata for the executed query."""

    text: str
    embedding_model: str
    match_count: int
    executed_at: str


class RagSearchMetadata(TypedDict, total=False):
    """Metadata describing the search results."""

    result_count: int
    embedding_dimensions: int
    threshold: float


class RagCitation(TypedDict):
    quote: str
    chunk_id: str


class RagAnswer(TypedDict, total=False):
    answer_text: str
    citations: list[RagCitation]


class RagSearchPayload(TypedDict):
    """Structured payload containing raw RAG search output."""

    query: RagSearchQuery
    results: list[MatchChunkRow]
    raw_chunks: list[MatchChunkRow]
    answer: RagAnswer
    metadata: RagSearchMetadata


def _require_env(name: str, *, hint: str | None = None) -> str:
    value = os.environ.get(name)
    if not value:
        message = hint or f"Environment variable {name} is required for RAG search"
        raise RagSearchError(message)
    return value


def get_db_url() -> str:
    """Return the Postgres connection URL for the RAG database."""

    rag_url = os.environ.get("SUPABASE_RAG_DB_URL")
    if rag_url:
        return rag_url

    fallback = os.environ.get("SUPABASE_DB_URL")
    if fallback:
        return fallback

    raise RagSearchError(
        "Environment variable SUPABASE_RAG_DB_URL is required for RAG search; "
        "legacy SUPABASE_DB_URL is only used as a fallback when present"
    )


def configure_gemini() -> str:
    """Configure Gemini client using the GOOGLE_API_KEY environment variable."""

    api_key = _require_env(
        "GOOGLE_API_KEY",
        hint="Environment variable GOOGLE_API_KEY is required for RAG search (/ask) embeddings",
    )
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
        embedding = response.embedding
    if embedding is None and isinstance(response, Mapping):
        embedding = response.get("embedding")
    if embedding is None and hasattr(response, "embeddings"):
        embedding = response.embeddings
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
) -> list[MatchChunkRow]:
    """Search for raw chunks using the match_chunks stored procedure."""

    db_url = get_db_url()
    embedding_vector = list(float(value) for value in embedding)
    embedding_text = "[" + ",".join(str(value) for value in embedding_vector) + "]"
    try:
        with psycopg2.connect(db_url, cursor_factory=RealDictCursor) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM match_chunks(%s::vector, %s, %s)",
                    (embedding_text, match_threshold, match_count),
                )
                rows = cursor.fetchall()
                if not rows:
                    return []

                results: list[MatchChunkRow] = []
                for row in rows:
                    item: MatchChunkRow = dict(row)

                    if not item.get("source_link") and item.get("source_links"):
                        source_links = item.get("source_links")
                        if isinstance(source_links, list) and source_links:
                            item["source_link"] = source_links[0]

                    for list_field in [
                        "scan_page_ids",
                        "scan_tg_msg_ids",
                        "ocr_tg_msg_ids",
                        "media_ids",
                        "media_internal_ids",
                        "source_links",
                    ]:
                        if item.get(list_field) is None:
                            item[list_field] = []

                    if item.get("scan_page_ids"):
                        item["scan_page_ids"] = [
                            int(value) for value in item["scan_page_ids"] if value is not None
                        ]

                    if item.get("scan_tg_msg_ids"):
                        item["scan_tg_msg_ids"] = [
                            int(value) for value in item["scan_tg_msg_ids"] if value is not None
                        ]

                    if item.get("ocr_tg_msg_ids"):
                        item["ocr_tg_msg_ids"] = [
                            int(value) for value in item["ocr_tg_msg_ids"] if value is not None
                        ]

                    if item.get("media_ids"):
                        item["media_ids"] = [
                            int(value) for value in item["media_ids"] if value is not None
                        ]

                    if item.get("media_internal_ids"):
                        item["media_internal_ids"] = [
                            str(value) for value in item["media_internal_ids"] if value is not None
                        ]

                    if item.get("source_links"):
                        item["source_links"] = [
                            str(value) for value in item["source_links"] if value is not None
                        ]

                    results.append(item)

                return results
    except PsycopgError as exc:  # pragma: no cover - depends on external DB
        logging.error("Failed to call match_chunks via callproc: %s", exc)
        raise RagSearchError("Failed to search chunks") from exc


def _normalize_chunk_id(value: str | int | None, index: int) -> str:
    base_id = str(value) if value not in (None, "") else f"chunk-{index + 1}"
    return base_id


def generate_grounded_answer(
    query_text: str, chunks: Sequence[MatchChunkRow], llm_client: Any, *, chunk_limit: int = 5
) -> RagAnswer:
    """Generate an answer grounded in the provided chunks using Gemini."""

    if llm_client is None or not hasattr(llm_client, "generate_content"):
        raise RagSearchError("A valid llm_client with generate_content is required")

    chunk_payload: OrderedDict[str, Mapping[str, Any]] = OrderedDict()
    seen_ids: dict[str, int] = {}
    for idx, chunk in enumerate(chunks[:chunk_limit]):
        if not isinstance(chunk, Mapping):
            continue

        chunk_id = _normalize_chunk_id(chunk.get("chunk_id") or chunk.get("id"), idx)
        collision_count = seen_ids.get(chunk_id, 0)
        seen_ids[chunk_id] = collision_count + 1
        unique_id = chunk_id if collision_count == 0 else f"{chunk_id}-{collision_count}"

        raw_content = chunk.get("chunk") or chunk.get("content") or ""
        content = raw_content if isinstance(raw_content, str) else str(raw_content)
        chunk_payload[unique_id] = {
            "content": content,
            "title": chunk.get("book_title"),
            "authors": chunk.get("book_authors"),
            "year": chunk.get("book_year") or chunk.get("chunk_year_start"),
            "source_link": chunk.get("source_link"),
            "chunk_type": chunk.get("chunk_type"),
            "topic": chunk.get("topic"),
        }

    if not chunk_payload:
        return {"answer_text": "", "citations": []}

    prompt = """
You are a question-answering assistant. Use ONLY the provided chunks to answer the user's query.

Return STRICTLY JSON with the following structure:
{
  "answer_text": "Grounded answer using the chunks",
  "citations": [
    {"quote": "direct quote from a chunk", "chunk_id": "<chunk id from the input>"}
  ]
}

Do not add any additional keys or text outside of JSON.
"""

    context_json = json.dumps(
        {"query": query_text, "chunks": chunk_payload}, ensure_ascii=False, indent=2
    )

    response = llm_client.generate_content([prompt, context_json])
    response_text = getattr(response, "text", None) or ""
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as exc:  # pragma: no cover - relies on external service
        logging.error("Failed to parse grounded answer JSON: %s", exc)
        raise RagSearchError("LLM response was not valid JSON") from exc

    answer_text = parsed.get("answer_text") if isinstance(parsed, Mapping) else None
    if not isinstance(answer_text, str):
        answer_text = ""

    raw_citations = parsed.get("citations") if isinstance(parsed, Mapping) else []
    citations: list[RagCitation] = []
    if isinstance(raw_citations, Sequence):
        for item in raw_citations:
            if not isinstance(item, Mapping):
                continue
            quote = item.get("quote")
            chunk_id = item.get("chunk_id")
            if isinstance(quote, str) and isinstance(chunk_id, str):
                citations.append({"quote": quote, "chunk_id": chunk_id})

    return {"answer_text": answer_text, "citations": citations}


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
    buffer.write(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default).encode("utf-8")
    )
    buffer.seek(0)
    return filename, buffer.read()


def run_rag_search(query_text: str, match_count: int = 5) -> RagSearchPayload:
    """Execute the full RAG search flow and return metadata alongside matches."""

    configure_gemini()
    embedding = embed_query(query_text)
    results = search_raw_chunks(embedding, match_count=match_count)
    llm_client = genai.GenerativeModel(
        GROUNDED_ANSWER_MODEL_ID, generation_config={"response_mime_type": "application/json"}
    )
    grounded_answer = generate_grounded_answer(
        query_text, results, llm_client, chunk_limit=match_count
    )
    executed_at = datetime.now(timezone.utc).isoformat()
    return {
        "query": {
            "text": query_text,
            "embedding_model": EMBEDDING_MODEL,
            "match_count": match_count,
            "executed_at": executed_at,
        },
        "results": results,
        "raw_chunks": results,
        "answer": grounded_answer,
        "metadata": {
            "result_count": len(results),
            "embedding_dimensions": len(embedding),
        },
    }


def build_raw_answer_document(payload: Mapping[str, Any]) -> tuple[str, bytes]:
    """Backward compatible alias for build_raw_answer_file."""

    return build_raw_answer_file(payload)
