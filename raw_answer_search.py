from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterable, Mapping

from rag_search import RagSearchError, build_raw_answer_document, run_rag_search


class RawSearchError(RuntimeError):
    """Domain-specific error raised when raw search fails."""


def search_raw_chunks(query_text: str, threshold: float = 0.5, match_count: int = 5) -> dict[str, Any]:
    try:
        payload = run_rag_search(query_text, match_count=match_count)
    except RagSearchError as exc:
        raise RawSearchError(str(exc)) from exc

    results = payload.get("results") or []
    filtered_results: list[Mapping[str, Any]] = []
    for row in results:
        score = None
        if isinstance(row, Mapping):
            if row.get("relevance_score") is not None:
                score = row.get("relevance_score")
            elif row.get("similarity") is not None:
                score = row.get("similarity")
            elif row.get("score") is not None:
                score = row.get("score")
            elif row.get("match_score") is not None:
                score = row.get("match_score")
        if score is None or score >= threshold:
            filtered_results.append(row)

    metadata = {**(payload.get("metadata") or {}), "threshold": threshold, "result_count": len(filtered_results)}
    return {**payload, "results": filtered_results, "metadata": metadata}


def build_raw_answer_file(payload: Mapping[str, Any]) -> tuple[str, bytes]:
    return build_raw_answer_document(payload)


RAW_ANSWER_MAX_SCAN_PAGES = 5


def deduplicate_pages(
    match_rows: Iterable[Mapping[str, Any]], max_pages: int = RAW_ANSWER_MAX_SCAN_PAGES
) -> list[Mapping[str, Any]]:
    pages: OrderedDict[str | int, Mapping[str, Any]] = OrderedDict()
    for row in match_rows:
        tg_msg_id = row.get("tg_msg_id") if isinstance(row, Mapping) else None
        if tg_msg_id in (None, ""):
            continue
        if tg_msg_id not in pages:
            pages[tg_msg_id] = row
        if len(pages) >= max_pages:
            break
    return list(pages.values())


def format_scan_caption(row: Mapping[str, Any]) -> str:
    parts: list[str] = []
    title = row.get("book_title")
    page = row.get("book_page")
    chunk = row.get("chunk") or row.get("content")
    if title:
        parts.append(f"Источник: {title}")
    if page:
        parts.append(f"Страница: {page}")
    if chunk:
        snippet = str(chunk)
        if len(snippet) > 240:
            snippet = snippet[:237].rstrip() + "…"
        parts.append(snippet)
    return "\n".join(parts)
