
"""
Простейшее логирование: сохраняем массивы в .npz и .csv.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

def save_npz(path: str, **arrays):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)

def save_csv(path: str, header: list[str], data: np.ndarray):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        for row in data:
            w.writerow(list(row))
