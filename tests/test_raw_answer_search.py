import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from raw_answer_search import deduplicate_pages


def test_deduplicate_pages_ocr_alignment(caplog):
    caplog.set_level(logging.WARNING)

    rows = [
        {
            "scan_tg_msg_ids": [11, 12],
            "ocr_tg_msg_ids": [21, 22],
            "scan_page_ids": [1, 2],
            "book_title": "Aligned",
        },
        {
            "scan_tg_msg_ids": [31, 32],
            "ocr_tg_msg_ids": [41],
            "scan_page_ids": [3, 4],
            "book_title": "Truncated OCR",
        },
        {
            "scan_tg_msg_ids": [51],
            "ocr_tg_msg_ids": [61, 62],
            "scan_page_ids": [5],
            "book_title": "Extra OCR",
        },
    ]

    results = deduplicate_pages(rows, max_pages=10)

    assert [result["tg_msg_id"] for result in results] == [11, 12, 31, 51]
    assert [result["ocr_tg_msg_id"] for result in results] == [21, 22, 41, 61]
    assert [result.get("book_page") for result in results] == [1, 2, 3, 5]

    warning_messages = [record.message for record in caplog.records if record.levelno == logging.WARNING]
    assert any("Length mismatch" in message for message in warning_messages)
    assert any("missing OCR id at index 1" in message for message in warning_messages)
    assert any("missing OCR id" in message for message in warning_messages)
