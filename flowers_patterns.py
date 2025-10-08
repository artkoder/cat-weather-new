from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback parser
    yaml = None  # type: ignore[assignment]
    import ast

    def _parse_scalar(text: str) -> Any:
        value = text.strip()
        if not value:
            return ""
        lowered = value.lower()
        if lowered in {"null", "none"}:
            return None
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        try:
            return ast.literal_eval(value)
        except Exception:
            return value

    def _fallback_yaml_load(text: str) -> Any:
        lines = [line.rstrip("\n") for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
        index = 0

        def parse_block(indent: int) -> Any:
            nonlocal index
            result_dict: dict[str, Any] = {}
            result_list: list[Any] | None = None
            while index < len(lines):
                line = lines[index]
                stripped = line.lstrip(" ")
                current_indent = len(line) - len(stripped)
                if current_indent < indent:
                    break
                if current_indent > indent:
                    # nested block for previous key handled by recursion
                    return parse_block(current_indent)
                if stripped.startswith("- "):
                    if result_list is None:
                        result_list = []
                    value_text = stripped[2:].strip()
                    index += 1
                    if not value_text:
                        result_list.append(parse_block(current_indent + 2))
                        continue
                    if value_text.endswith(":"):
                        key = value_text[:-1].strip()
                        nested = parse_block(current_indent + 2)
                        result_list.append({key: nested})
                        continue
                    if ":" in value_text:
                        key, remainder = value_text.split(":", 1)
                        base_value = _parse_scalar(remainder.strip())
                        if index < len(lines):
                            next_indent = len(lines[index]) - len(lines[index].lstrip(" "))
                            if next_indent > current_indent:
                                nested = parse_block(current_indent + 2)
                                combined = {key.strip(): base_value}
                                if isinstance(nested, dict):
                                    combined.update(nested)
                                    result_list.append(combined)
                                else:
                                    result_list.append({key.strip(): nested})
                                continue
                        result_list.append({key.strip(): base_value})
                        continue
                    result_list.append(_parse_scalar(value_text))
                    continue
                if ":" not in stripped:
                    index += 1
                    continue
                key, remainder = stripped.split(":", 1)
                key = key.strip()
                remainder = remainder.strip()
                index += 1
                if remainder:
                    value = _parse_scalar(remainder)
                else:
                    value = parse_block(current_indent + 2)
                if result_list is not None:
                    result_list.append({key: value})
                else:
                    result_dict[key] = value
            if result_list is not None:
                return result_list
            return result_dict

        result = parse_block(0)
        return result


@dataclass(frozen=True)
class FlowerPattern:
    id: str
    kind: str
    template: str
    weight: float = 1.0
    requires_photo: bool = False
    requires_flowers: bool = False
    requires_weather: Iterable[str] | None = None
    requires_season: Iterable[str] | None = None
    always_include: bool = False
    photo_dependent: bool = False

    def matches_context(self, context: Mapping[str, Any]) -> bool:
        if self.requires_photo and not context.get("has_photo"):
            return False
        if self.requires_flowers and not context.get("flowers"):
            return False
        if self.requires_weather:
            weather = context.get("weather")
            if not weather:
                return False
            allowed = {str(item) for item in self.requires_weather}
            if weather not in allowed:
                return False
        if self.requires_season:
            season = context.get("season")
            if not season:
                return False
            allowed = {str(item) for item in self.requires_season}
            if season not in allowed:
                return False
        return True


class FlowerKnowledgeBase:
    def __init__(
        self,
        *,
        colors: dict[str, Any],
        flowers: dict[str, Any],
        traditions: dict[str, Any],
        weather: dict[str, Any],
        seasons: dict[str, Any],
        wisdom: list[dict[str, Any]],
        holidays: dict[str, Any],
        micro_engagement: list[dict[str, Any]],
        patterns: list[FlowerPattern],
        banned_words: set[str],
    ) -> None:
        self.colors = colors
        self.flowers = flowers
        self.traditions = traditions
        self.weather = weather
        self.seasons = seasons
        self.wisdom = wisdom
        self.holidays = holidays
        self.micro_engagement = micro_engagement
        self.patterns = patterns
        self.banned_words = banned_words
        self._flower_lookup: dict[str, str] = {}
        for slug, payload in flowers.items():
            for alias in payload.get("varieties", []):
                key = str(alias).strip().lower()
                if not key:
                    continue
                self._flower_lookup[key] = slug

    def resolve_flower(self, name: str) -> str | None:
        key = str(name or "").strip().lower()
        if not key:
            return None
        if key in self.flowers:
            return key
        return self._flower_lookup.get(key)

    def palette_for_flower(self, flower_id: str) -> dict[str, Any] | None:
        flower = self.flowers.get(flower_id) or {}
        palette_id = flower.get("color")
        if not palette_id:
            return None
        return self.colors.get(str(palette_id))


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        text = fh.read()
    if yaml is not None:
        data = yaml.safe_load(text)  # type: ignore[attr-defined]
    else:
        data = _fallback_yaml_load(text)
    return data or {}


def _load_patterns(path: Path) -> tuple[list[FlowerPattern], set[str]]:
    raw = _load_yaml(path)
    banned_words = {str(word).strip().lower() for word in raw.get("banned_words", []) if str(word).strip()}
    patterns: list[FlowerPattern] = []
    for entry in raw.get("patterns", []):
        if not isinstance(entry, Mapping):
            continue
        pattern = FlowerPattern(
            id=str(entry.get("id") or "").strip(),
            kind=str(entry.get("kind") or "").strip(),
            template=str(entry.get("template") or "").strip(),
            weight=float(entry.get("weight") or 1.0),
            requires_photo=bool(entry.get("requires_photo")),
            requires_flowers=bool(entry.get("requires_flowers")),
            requires_weather=entry.get("requires_weather"),
            requires_season=entry.get("requires_season"),
            always_include=bool(entry.get("always_include")),
            photo_dependent=bool(entry.get("photo_dependent")),
        )
        if not pattern.id or not pattern.kind or not pattern.template:
            continue
        patterns.append(pattern)
    return patterns, banned_words


@lru_cache(maxsize=1)
def load_flowers_knowledge(base_path: str | Path | None = None) -> FlowerKnowledgeBase:
    root = Path(base_path) if base_path else Path(__file__).resolve().parent / "data" / "flowers"
    colors = _load_yaml(root / "colors.yaml").get("palettes", {})
    flowers = _load_yaml(root / "flowers.yaml").get("flowers", {})
    traditions = _load_yaml(root / "traditions.yaml").get("traditions", {})
    weather = _load_yaml(root / "weather.yaml").get("weather", {})
    seasons = _load_yaml(root / "season.yaml").get("seasons", {})
    wisdom = _load_yaml(root / "wisdom.yaml").get("quotes", [])
    holidays = _load_yaml(root / "holidays.yaml").get("holidays", {})
    micro_engagement = _load_yaml(root / "micro_engagement.yaml").get("prompts", [])
    patterns, banned_words = _load_patterns(root / "patterns.yaml")
    return FlowerKnowledgeBase(
        colors=colors,
        flowers=flowers,
        traditions=traditions,
        weather=weather,
        seasons=seasons,
        wisdom=wisdom,
        holidays=holidays,
        micro_engagement=micro_engagement,
        patterns=patterns,
        banned_words=banned_words,
    )


def serialize_plan(plan: Mapping[str, Any]) -> str:
    return json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True)
