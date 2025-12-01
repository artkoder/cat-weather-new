from __future__ import annotations

from typing import Any, Mapping

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
            score = row.get("similarity") or row.get("score") or row.get("match_score")
        if score is None or score >= threshold:
            filtered_results.append(row)

    metadata = {**(payload.get("metadata") or {}), "threshold": threshold, "result_count": len(filtered_results)}
    return {**payload, "results": filtered_results, "metadata": metadata}


def build_raw_answer_file(payload: Mapping[str, Any]) -> tuple[str, bytes]:
    return build_raw_answer_document(payload)
