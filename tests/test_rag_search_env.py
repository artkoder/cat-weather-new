import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rag_search import RagSearchError, get_db_url


def test_prefers_rag_specific_url(monkeypatch):
    monkeypatch.setenv("SUPABASE_RAG_DB_URL", "postgres://rag")
    monkeypatch.setenv("SUPABASE_DB_URL", "postgres://legacy")

    assert get_db_url() == "postgres://rag"


def test_falls_back_to_legacy_url(monkeypatch):
    monkeypatch.delenv("SUPABASE_RAG_DB_URL", raising=False)
    monkeypatch.setenv("SUPABASE_DB_URL", "postgres://legacy")

    assert get_db_url() == "postgres://legacy"


def test_missing_urls_raise_error(monkeypatch):
    monkeypatch.delenv("SUPABASE_RAG_DB_URL", raising=False)
    monkeypatch.delenv("SUPABASE_DB_URL", raising=False)

    with pytest.raises(RagSearchError) as excinfo:
        get_db_url()

    assert "SUPABASE_RAG_DB_URL" in str(excinfo.value)
