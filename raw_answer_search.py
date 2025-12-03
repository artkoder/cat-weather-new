from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Iterable, Mapping, Sequence

from rag_search import RagSearchError, build_raw_answer_document, run_rag_search


class RawSearchError(RuntimeError):
    """Domain-specific error raised when raw search fails."""


logger = logging.getLogger(__name__)


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
    pages: OrderedDict[str, Mapping[str, Any]] = OrderedDict()

    def _as_sequence(value: Any) -> Sequence[Any] | None:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return value
        return None

    def _normalize_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    for row in match_rows:
        if not isinstance(row, Mapping):
            continue

        scan_msg_ids = _as_sequence(row.get("scan_tg_msg_ids")) or []
        ocr_msg_ids_raw = _as_sequence(row.get("ocr_tg_msg_ids")) or []
        ocr_msg_ids = [_normalize_int(item) for item in ocr_msg_ids_raw]
        scan_page_ids_raw = _as_sequence(row.get("scan_page_ids")) or []
        scan_page_ids = [_normalize_int(item) for item in scan_page_ids_raw]

        if scan_msg_ids:
            if len({len(scan_msg_ids), len(scan_page_ids), len(ocr_msg_ids)}) != 1:
                logger.warning(
                    "Length mismatch between scan/page/ocr ids: scan=%s pages=%s ocr=%s",
                    len(scan_msg_ids),
                    len(scan_page_ids),
                    len(ocr_msg_ids),
                )

        candidate_pairs: list[tuple[Any, int | None, int | None]] = []
        if scan_msg_ids:
            for idx, msg_id in enumerate(scan_msg_ids):
                page_number = scan_page_ids[idx] if idx < len(scan_page_ids) else None
                if idx >= len(ocr_msg_ids):
                    logger.warning(
                        "Skipping scan msg_id %s page %s: missing OCR id at index %s",
                        msg_id,
                        page_number,
                        idx,
                    )
                    continue

                ocr_msg_id = ocr_msg_ids[idx]
                if ocr_msg_id is None:
                    logger.warning(
                        "Skipping scan msg_id %s page %s: invalid OCR id at index %s",
                        msg_id,
                        page_number,
                        idx,
                    )
                    continue
                candidate_pairs.append((msg_id, page_number, ocr_msg_id))
        else:
            tg_msg_id = row.get("tg_msg_id")
            if tg_msg_id not in (None, ""):
                page_number = scan_page_ids[0] if scan_page_ids else None
                if not ocr_msg_ids:
                    logger.warning(
                        "Skipping scan msg_id %s page %s: missing OCR id",
                        tg_msg_id,
                        page_number,
                    )
                else:
                    ocr_msg_id = ocr_msg_ids[0]
                    if ocr_msg_id is None:
                        logger.warning(
                            "Skipping scan msg_id %s page %s: invalid OCR id",
                            tg_msg_id,
                            page_number,
                        )
                    else:
                        candidate_pairs.append((tg_msg_id, page_number, ocr_msg_id))

        for msg_id, page_number, ocr_msg_id in candidate_pairs:
            if msg_id in (None, ""):
                continue
            dedup_key = str(msg_id)
            if dedup_key in pages:
                continue

            enriched_row = dict(row)
            enriched_row["tg_msg_id"] = msg_id
            if ocr_msg_id is not None:
                enriched_row["ocr_tg_msg_id"] = ocr_msg_id
            if page_number is not None and enriched_row.get("book_page") in (None, ""):
                enriched_row["book_page"] = page_number

            pages[dedup_key] = enriched_row
            if len(pages) >= max_pages:
                return list(pages.values())

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
