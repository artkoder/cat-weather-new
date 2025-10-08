from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable

DEFAULT_CITY_NAME = "Калининград"
DEFAULT_COAST_NAME = "Балтийское побережье"


class FlowersWeatherPlanner:
    """Builds a structured weather snapshot for the flowers rubric."""

    def __init__(self, db: Any, *, emoji: Callable[[int, int | None], str]):
        self._db = db
        self._emoji = emoji

    def build(self) -> dict[str, Any]:
        city = self._kaliningrad_weather()
        sea = self._coast_weather()
        plan_lines: list[str] = []
        city_plan = city.get("plan_line")
        if city_plan:
            plan_lines.append(city_plan)
        sea_plan = sea.get("plan_line")
        if sea_plan:
            plan_lines.append(sea_plan)
        change_summary = city.get("change_summary") or "Погода остаётся приятной."
        plan_lines.append(f"Вывод о переменах относительно вчера: {change_summary}")
        positive_phrases = self._collect_positive_phrases(city, sea)
        if not positive_phrases:
            positive_phrases = ["Погода дарит приятное настроение."]
        plan_lines.append(
            "Позитивные формулировки: " + " ".join(positive_phrases)
        )
        positive_summary = positive_phrases[0].rstrip(".") + "."
        locations = [
            city.get("location") or city.get("name"),
            sea.get("location") or sea.get("name"),
        ]
        clean_locations = [str(loc) for loc in locations if loc]
        if clean_locations:
            caption_line = f"{positive_summary.rstrip('.')} • {', '.join(clean_locations)}"
        else:
            caption_line = positive_summary
        return {
            "city": city,
            "sea": sea,
            "change_summary": change_summary,
            "positive_phrases": positive_phrases,
            "positive_summary": positive_summary,
            "plan_lines": plan_lines,
            "caption_line": caption_line,
            "locations": clean_locations,
        }

    def _kaliningrad_weather(self) -> dict[str, Any]:
        row = self._db.execute(
            "SELECT id, name FROM cities WHERE lower(name)=lower(?)",
            ("Kaliningrad",),
        ).fetchone()
        city_id = row["id"] if row else None
        city_name = row["name"] if row and row["name"] else DEFAULT_CITY_NAME
        info: dict[str, Any] = {"name": city_name, "location": city_name}
        if not city_id:
            return self._city_fallback(info)
        weather_row = self._db.execute(
            """
            SELECT temperature, weather_code, wind_speed, is_day
            FROM weather_cache_hour
            WHERE city_id=?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (city_id,),
        ).fetchone()
        if not weather_row:
            return self._city_fallback(info)
        temp = weather_row["temperature"]
        wind = weather_row["wind_speed"]
        code = weather_row["weather_code"]
        try:
            is_day = weather_row["is_day"]
        except (KeyError, TypeError):
            is_day = None
        summary = self._compose_city_summary(temp, wind)
        emoji = self._emoji(int(code), is_day) if code is not None else ""
        display_parts: list[str] = []
        if emoji:
            display_parts.append(emoji)
        if summary:
            display_parts.append(f"{city_name}: {summary}")
        else:
            display_parts.append(city_name)
        info["summary"] = summary
        info["display"] = " ".join(display_parts).strip()
        previous = self._previous_day(city_id)
        prev_temp = previous.get("temperature") if previous else None
        prev_wind = previous.get("wind_speed") if previous else None
        temp_phrase = self._positive_temperature_phrase(city_name, temp, prev_temp)
        wind_phrase = self._positive_wind_phrase(wind, prev_wind)
        info["positive_phrases"] = [temp_phrase, wind_phrase]
        info["change_summary"] = self._change_summary(
            city_name, temp, prev_temp, wind, prev_wind
        )
        if summary:
            info["plan_line"] = f"Текущая погода в {city_name}: {summary}."
        else:
            info["plan_line"] = (
                f"Текущая погода в {city_name}: данные появятся позже, но утро всё равно приятное."
            )
        return info

    def _city_fallback(self, info: dict[str, Any]) -> dict[str, Any]:
        city_name = info["name"]
        info["summary"] = None
        info["display"] = f"{city_name}: данные уточняются"
        info["positive_phrases"] = [f"Погода в {city_name} остаётся уютной."]
        info["change_summary"] = f"В {city_name} погода остаётся приятной."
        info["plan_line"] = (
            f"Текущая погода в {city_name}: данных нет, но утро обещает быть приятным."
        )
        return info

    def _coast_weather(self) -> dict[str, Any]:
        row = self._db.execute(
            """
            SELECT s.id, s.name, c.current, c.wave
            FROM seas AS s
            LEFT JOIN sea_cache AS c ON c.sea_id = s.id
            ORDER BY COALESCE(c.updated, '') DESC, s.id ASC
            LIMIT 1
            """
        ).fetchone()
        name = row["name"] if row and row["name"] else DEFAULT_COAST_NAME
        temp = row["current"] if row else None
        wave = row["wave"] if row else None
        temp_str = self._format_temperature(temp, decimals=1)
        wave_desc = self._describe_wave_height(wave)
        parts: list[str] = []
        if temp_str:
            parts.append(f"вода {temp_str}")
        if wave_desc:
            parts.append(wave_desc)
        summary = ", ".join(parts) if parts else None
        info: dict[str, Any] = {
            "name": name,
            "location": name,
            "summary": summary,
            "display": f"{name}: {summary}" if summary else f"{name}: данные уточняются",
            "positive_phrases": [self._sea_positive_phrase(name, wave)],
        }
        if summary:
            info["plan_line"] = f"Море у {name}: {summary}."
        else:
            info["plan_line"] = (
                f"Море у {name}: данные появятся позже, но берег остаётся уютным."
            )
        return info

    def _collect_positive_phrases(self, *segments: dict[str, Any]) -> list[str]:
        phrases: list[str] = []
        for segment in segments:
            values = segment.get("positive_phrases") or []
            for phrase in values:
                text = str(phrase or "").strip()
                if text:
                    phrases.append(text if text.endswith(".") else f"{text}.")
        return phrases

    def _compose_city_summary(self, temp: Any, wind: Any) -> str | None:
        temp_str = self._format_temperature(temp)
        wind_str = self._format_wind(wind)
        parts: list[str] = []
        if temp_str:
            parts.append(temp_str)
        if wind_str:
            parts.append(f"ветер {wind_str}")
        return ", ".join(parts) if parts else None

    def _previous_day(self, city_id: Any) -> dict[str, Any] | None:
        yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
        row = self._db.execute(
            """
            SELECT temperature, wind_speed
            FROM weather_cache_day
            WHERE city_id=? AND day=?
            """,
            (city_id, yesterday.isoformat()),
        ).fetchone()
        if not row:
            return None
        return dict(row)

    def _format_temperature(self, value: Any, *, decimals: int = 0) -> str | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if decimals:
            return f"{number:.{decimals}f}\u00B0C"
        return f"{int(round(number))}\u00B0C"

    def _format_wind(self, value: Any) -> str | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return f"{int(round(number))} м/с"

    def _positive_temperature_phrase(
        self,
        city_name: str,
        current: Any,
        previous: Any,
    ) -> str:
        try:
            current_val = float(current)
        except (TypeError, ValueError):
            current_val = None
        try:
            previous_val = float(previous)
        except (TypeError, ValueError):
            previous_val = None
        if current_val is None:
            return f"Температура в {city_name} остаётся комфортной."
        if previous_val is None:
            return (
                f"Температура в {city_name} около {int(round(current_val))}\u00B0 — хорошее утро."
            )
        diff = current_val - previous_val
        amount = max(1, abs(int(round(diff))))
        if diff > 1:
            return f"В {city_name} теплее на {amount}\u00B0 — отличный повод прогуляться."
        if diff < -1:
            return (
                f"Свежий воздух в {city_name} бодрит (на {amount}\u00B0 прохладнее)."
            )
        return f"Температура в {city_name} держится комфортной."

    def _positive_wind_phrase(self, current: Any, previous: Any) -> str:
        try:
            current_val = float(current)
        except (TypeError, ValueError):
            current_val = None
        try:
            previous_val = float(previous)
        except (TypeError, ValueError):
            previous_val = None
        if current_val is None:
            return "Ветер остаётся спокойным."
        if previous_val is None:
            return "Ветер мягкий и не мешает прогулкам."
        diff = current_val - previous_val
        if diff < -1:
            return "Ветер стал мягче и уютнее."
        if diff > 1:
            return "Лёгкий ветерок добавляет энергии."
        return "Ветер остаётся спокойным."

    def _change_summary(
        self,
        city_name: str,
        current_temp: Any,
        previous_temp: Any,
        current_wind: Any,
        previous_wind: Any,
    ) -> str:
        try:
            temp_now = float(current_temp)
        except (TypeError, ValueError):
            temp_now = None
        try:
            temp_prev = float(previous_temp)
        except (TypeError, ValueError):
            temp_prev = None
        try:
            wind_now = float(current_wind)
        except (TypeError, ValueError):
            wind_now = None
        try:
            wind_prev = float(previous_wind)
        except (TypeError, ValueError):
            wind_prev = None
        parts: list[str] = []
        if temp_now is not None and temp_prev is not None:
            diff_temp = temp_now - temp_prev
            amount = max(1, abs(int(round(diff_temp))))
            if diff_temp > 1:
                parts.append(f"теплее на {amount}\u00B0")
            elif diff_temp < -1:
                parts.append(f"свежее на {amount}\u00B0 — бодрит")
            else:
                parts.append("температура осталась комфортной")
        else:
            parts.append("температура ощущается комфортной")
        if wind_now is not None and wind_prev is not None:
            diff_wind = wind_now - wind_prev
            if diff_wind < -1:
                parts.append("ветер заметно спокойнее")
            elif diff_wind > 1:
                parts.append("ветер чуть бодрее и освежает")
            else:
                parts.append("ветер почти не изменился")
        else:
            parts.append("ветер мягкий")
        joined = ", ".join(parts)
        return f"В {city_name} {joined}."

    def _describe_wave_height(self, value: Any) -> str | None:
        if value is None:
            return None
        try:
            wave = float(value)
        except (TypeError, ValueError):
            return None
        if wave < 0.5:
            return "волны тихие"
        if wave < 1.5:
            return "волны мягкие"
        return "волны мощные"

    def _sea_positive_phrase(self, name: str, wave: Any) -> str:
        try:
            wave_val = float(wave)
        except (TypeError, ValueError):
            wave_val = None
        if wave_val is None:
            return f"Побережье {name} радует спокойствием."
        if wave_val < 0.5:
            return f"Побережье {name} встречает спокойными волнами."
        if wave_val < 1.5:
            return f"У {name} лёгкие волны — приятно прогуляться вдоль берега."
        return f"Море у {name} мощное и завораживающее."
