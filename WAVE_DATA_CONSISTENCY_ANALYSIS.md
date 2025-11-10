# Диагностика: Консистентность данных о волнах между ingestion и sea rubric

**Дата анализа:** 2024  
**Цель:** Проверить, что данные о волнах (wave_score, wave_conf) корректно передаются от процесса ingestion к рубрике "Море/Закат на море"

---

## ✅ РЕЗУЛЬТАТ: Данные о волнах КОНСИСТЕНТНЫ

Система использует надёжную цепь обработки данных с несколькими уровнями резервирования и обратной совместимости.

---

## 1. ПРОЦЕСС INGESTION (загрузка фото)

### Где распознаются данные о волнах

**Файл:** `ingestion.py`, строки 1212-1253

```python
wave_score_value: float | None = None
if isinstance(vision_payload, dict):
    raw_wave = vision_payload.get("sea_wave_score")
    if isinstance(raw_wave, dict):
        raw_wave = raw_wave.get("value")
    if raw_wave is not None:
        try:
            wave_score_value = float(raw_wave)
        except (TypeError, ValueError):
            wave_score_value = None
```

**Процесс:**
1. OpenAI Vision API возвращает `vision_payload` с полем `sea_wave_score`
2. Значение извлекается и конвертируется в float
3. Сохраняется в переменную `wave_score_value`

### В какие поля БД сохраняются

**Файл:** `ingestion.py`, строка 1253

```python
save_payload: SaveAssetPayload = {
    ...
    "photo_wave": wave_score_value,  # ← Здесь сохраняется wave score
    ...
}
```

**Файл:** `data_access.py`, метод `save_asset`, строки 1160-1165, 1276, 1296, 1330, 1353

```python
photo_wave_value = Asset._to_float(photo_wave)  # Нормализация значения

# При UPDATE:
UPDATE assets
   SET ...
       photo_wave=?,  # ← Сохранение в колонку photo_wave
       ...

# При INSERT:
INSERT INTO assets (..., photo_wave, ...)
VALUES (..., ?, ...)  # ← Вставка в колонку photo_wave
```

**Поля в таблице `assets`:**
- `photo_wave` — основное поле для wave score (сохраняется при ingestion)
- `vision_wave_score` — сырое значение из Vision API (может быть заполнено миграциями)
- `vision_wave_conf` — уверенность Vision API (может быть заполнено миграциями)
- `wave_score_0_10` — нормализованная оценка 0-10 (заполняется миграциями/backfills)
- `wave_conf` — нормализованная уверенность (заполняется миграциями/backfills)

---

## 2. РУБРИКА SEA: Извлечение данных

### Метод fetch_sea_candidates

**Файл:** `data_access.py`, строки 2353-2438

Система использует **приоритетную цепочку** для получения wave score с fallback'ами:

```python
# Приоритет 1: Нормализованное поле wave_score_0_10
raw_wave: Any = asset.wave_score_0_10

# Приоритет 2: Сырой результат из Vision API
if raw_wave is None:
    raw_wave = asset.vision_wave_score

# Приоритет 3: Поле photo_wave из ingestion (основной путь)
if raw_wave is None:
    raw_wave = asset.photo_wave

# Приоритет 4: Прямое чтение из vision_results JSON
if raw_wave is None:
    raw_wave = vision.get("sea_wave_score")
    if isinstance(raw_wave, dict):
        raw_wave = raw_wave.get("value")

wave_score = Asset._to_float(raw_wave)
```

**Результат:** Создаётся candidate dict со следующими полями:
```python
candidates.append({
    "asset": asset,
    "wave_score": wave_score,     # ← Основное поле
    "photo_wave": wave_score,     # ← Дубликат для обратной совместимости
    ...
})
```

### Использование в рубрике sea

**Файл:** `main.py`, метод `_publish_sea`

**Фильтрация кандидатов (строки 13745-13768):**
```python
# Calm seas guard - фильтрует фото с высокими волнами при спокойном море
for candidate in working_candidates:
    photo_wave = candidate.get("photo_wave") or candidate.get("wave_score")
    if photo_wave is not None:
        try:
            wave_val = float(photo_wave)
            if wave_val >= 5:  # Отсеивает фото с сильными волнами
                calm_guard_filtered.append(str(candidate["asset"].id))
                continue
        except (TypeError, ValueError):
            pass
```

**Оценка кандидатов (строки 13800-13819):**
```python
def evaluate_stage_candidate(candidate, stage_cfg, corridor):
    asset_obj = candidate["asset"]
    vision_wave = getattr(asset_obj, "vision_wave_score", None)
    
    # Читаем wave score с fallback'ами
    photo_wave = candidate.get("photo_wave")
    if photo_wave is None and vision_wave is not None:
        photo_wave = vision_wave
    if photo_wave is None:
        photo_wave = candidate.get("wave_score")
    
    photo_wave_val = None
    if photo_wave is not None:
        try:
            photo_wave_val = float(photo_wave)
        except (TypeError, ValueError):
            photo_wave_val = None
    
    # Используется для подсчёта очков и выбора лучшего фото
    wave_delta = abs(photo_wave_val - target_wave_value) if photo_wave_val else None
    ...
```

---

## 3. ЛОГИКА ВЫБОРА ФОТО ПО ВОЛНАМ

### Действительно ли рубрика использует wave_score?

**ДА**, система активно использует wave_score на нескольких уровнях:

#### 3.1. Calm Seas Guard (строки 13741-13778)
Когда целевой wave_score ≤ 1 (спокойное море):
- **Активируется фильтр:** Удаляет кандидаты с wave_score ≥ 5
- **Цель:** Не публиковать фото с большими волнами при спокойной погоде

#### 3.2. Система оценки кандидатов (строки 13852-13885)
Каждый кандидат получает очки по формуле:

**Бонусы:**
- `CalmWaveBonus` (+5.0): Если target ≤ 2 и photo_wave ≤ 1.0 (награда за спокойное море)
- `VisibleSkyBonus`: Если небо видно

**Штрафы:**
- `WaveDeltaPenalty`: Пропорционально разнице между target_wave и photo_wave
- `WaveCorridorPenalty`: Если разница превышает допустимый коридор (tolerance)
- `CalmWavePenalty`: Если target ≤ 2, но photo_wave > calm_wave_cap
- `CalmGuardNullWavePenalty` (-0.8): Для стадий B0/B1 при активном calm guard и отсутствии данных

#### 3.3. Многоступенчатый отбор (STAGE_CONFIGS)
Система использует 4 стадии отбора с разными tolerance'ами:
- **B0:** Самый строгий коридор
- **B1:** Умеренный коридор
- **B2:** Широкий коридор
- **AN:** Наименее строгий (Accept Nearly everything)

На каждой стадии вычисляется:
```python
corridor = (target_wave - tolerance, target_wave + tolerance)
```

Кандидаты за пределами коридора получают штраф:
```python
if wave_delta > stage_cfg.wave_tolerance:
    overshoot = wave_delta - stage_cfg.wave_tolerance
    corridor_penalty = overshoot * stage_cfg.outside_corridor_multiplier
    score -= corridor_penalty
```

---

## 4. ВЫВОД И РЕКОМЕНДАЦИИ

### ✅ Консистентность данных

**Цепь данных полностью консистентна:**

```
┌─────────────┐
│ Vision API  │
│ sea_wave_   │
│ score       │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ ingestion.py        │
│ wave_score_value    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ data_access.py      │
│ save_asset()        │
│ → photo_wave column │
└──────┬──────────────┘
       │
       ▼
┌────────────────────────┐
│ fetch_sea_candidates() │
│ Priority chain:        │
│ 1. wave_score_0_10     │
│ 2. vision_wave_score   │
│ 3. photo_wave ← main   │
│ 4. vision JSON         │
└──────┬─────────────────┘
       │
       ▼
┌──────────────────┐
│ Candidate dict:  │
│ - wave_score     │
│ - photo_wave     │
└──────┬───────────┘
       │
       ▼
┌────────────────────┐
│ _publish_sea()     │
│ Uses both:         │
│ - photo_wave       │
│ - wave_score       │
└────────────────────┘
```

### Почему могли быть проблемы в прошлом?

1. **Отсутствие данных в старых фото**
   - До внедрения Vision API wave detection
   - Миграция может заполнить `wave_score_0_10` через backfill

2. **Некорректные результаты Vision API**
   - OpenAI может ошибаться в определении волн
   - Решение: Ручная корректировка через команды бота

3. **Баги в логике calm_guard или scoring**
   - Не в data flow, а в алгоритмах выбора
   - Проверить tolerance и penalty значения в STAGE_CONFIGS

4. **NULL значения wave_score**
   - Если Vision API не вернул sea_wave_score
   - Система корректно обрабатывает через fallback chain

### Рекомендации

1. ✅ **Data flow корректен** - изменения не требуются
2. 🔍 **Мониторинг:** Добавить логирование случаев, когда photo_wave is None
3. 🔧 **Backfill:** Убедиться, что все sea фото имеют wave_score_0_10
4. 📊 **Метрики:** Отслеживать распределение wave scores в публикациях vs погода
5. 🧪 **Тесты:** Проверить, что calm_guard работает корректно при разных сценариях

---

## 5. ФАЙЛЫ И МЕТОДЫ, ТРЕБУЮЩИЕ ВНИМАНИЯ

### Если нужны исправления (их НЕТ в data flow):

**Логика выбора фото:**
- `main.py:13741-13778` — Calm seas guard logic
- `main.py:13789-13875` — Stage evaluation scoring
- `sea_selection.py` — STAGE_CONFIGS (tolerance values)

**Миграции и backfills:**
- `migrations/0027_add_wave_sky_metrics.py` — Добавление wave_score_0_10
- `utils_wave.py` — Парсинг wave score из Vision JSON

**Тесты для верификации:**
- `tests/test_calm_seas_guard_rules.py`
- `tests/test_sea_scoring_integration.py`
- `tests/test_ingestion_helper_consistency.py`

---

## ЗАКЛЮЧЕНИЕ

**✅ Данные о волнах полностью консистентны между процессом ingestion и рубрикой sea.**

Система использует надёжную архитектуру с:
- Множественными fallback'ами для обратной совместимости
- Чёткой цепью преобразования данных
- Активным использованием wave_score в логике выбора фото
- Защитными механизмами (calm_guard) против публикации неподходящих фото

Если наблюдаются проблемы с публикацией неправильных волн, причина НЕ в data flow, а в:
1. Качестве данных от Vision API
2. Настройках tolerance/penalty в алгоритмах scoring
3. Отсутствии wave данных в старых фото
