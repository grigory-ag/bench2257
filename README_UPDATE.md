# Обновление проекта: benchmark client + analytics

## Что было добавлено

- **GUI-утилита** `benchmark_client/app.py`:
  - логин в систему;
  - запуск эталонных скриптов по языкам и задачам;
  - отправка результатов на backend;
  - улучшенная обработка ошибок выполнения скриптов (без silent failure).
- **Новый API-эндпоинт** `GET /api/analytics/global` для агрегированной аналитики.
- **Страница аналитики** `frontend/analytics.html` с графиками и сравнением результатов.
- **Эталонные скрипты** в `benchmark_client/scripts/`:
  - Python, C++, CUDA, Go;
  - задачи: Sum / Multiply / Invert Matrices;
  - единый бинарный формат входных/выходных данных.

## Как запустить локально

### 1) Поднять инфраструктуру (Docker)

```bash
docker-compose -f docker-compose.dev.yml up -d --build
```

### 2) Сгенерировать тестовые данные

```bash
python benchmark_client/scripts/data_generator.py --n 4000
```

### 3) Запустить benchmark client

Установка зависимостей:

```bash
pip install -r benchmark_client/requirements.txt
```

Запуск GUI:

```bash
python benchmark_client/app.py
```

## Единый стандарт (ТЗ) для эталонных скриптов

- Скрипт должен выводить **ровно одну строку JSON**:
  - `{"execution_time_ms": X.X}`
- Время должно отражать только вычислительную часть задачи (без лишнего логирования и постороннего вывода в stdout).
