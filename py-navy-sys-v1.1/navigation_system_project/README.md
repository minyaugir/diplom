# navigation_system — INS/GNSS EKF (UAV)

Проект содержит прототип интегрированной навигационной системы БПЛА на основе ИНС/ГНСС и расширенного фильтра Калмана (EKF).

## Быстрый старт

```bash
pip install -r requirements.txt
python -m navigation_system.main
python -m pytest
```

Результаты моделирования (графики/таблицы) сохраняются в `navigation_system/data/output/`.

## Структура

Пакет находится в каталоге `navigation_system/` и включает:
- `strapdown_integration.py` — механизация ИНС
- `extended_kalman_filter.py` — EKF
- `ephemeris.py` — преобразования координат (WGS‑84)
- `visualization.py` — построение графиков
- `tests/` — модульные тесты
