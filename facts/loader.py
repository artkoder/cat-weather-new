from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Fact:
    id: str
    text: str


def _iter_fact_lines(content: Iterable[str]) -> Iterable[str]:
    for line in content:
        stripped = line.strip()
        if stripped.startswith("- "):
            text = stripped[2:].strip()
            if text:
                yield text


def load_baltic_facts(path: str | Path = "facts/baltic_facts.md") -> list[Fact]:
    file_path = Path(path)
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        logging.error("SEA_RUBRIC facts load_failed path=%s", file_path)
        return []

    facts: list[Fact] = []
    for text in _iter_fact_lines(lines):
        fact_id = hashlib.sha1(text.encode("utf-8")).hexdigest()
        facts.append(Fact(id=fact_id, text=text))

    logging.info("SEA_RUBRIC facts load path=%s count=%s", file_path, len(facts))
    return facts


def load_new_year_traditions(path: str | Path = "facts/new_year_traditions.md") -> list[Fact]:
    file_path = Path(path)
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        logging.error("NEW_YEAR_RUBRIC facts load_failed path=%s", file_path)
        return []

    facts: list[Fact] = []
    for text in _iter_fact_lines(lines):
        fact_id = hashlib.sha1(text.encode("utf-8")).hexdigest()
        facts.append(Fact(id=fact_id, text=text))

    logging.info("NEW_YEAR_RUBRIC facts load path=%s count=%s", file_path, len(facts))
    return facts
