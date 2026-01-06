# navigation_system

Прототип интегрированной навигации БВС (ИНС/ГНСС/баро) на базе расширенного фильтра Калмана.

## Быстрый запуск

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Запуск сценария, заданного в config.yaml
python -m navigation_system.main

# Тесты
pytest -q
```

## Результаты

Файлы сохраняются в `navigation_system/data/output/<scenario>/`:
- `log.npz` — сырые данные эксперимента
- `rmse_table.csv` — таблица RMSE
- `fig_*.png` — графики
