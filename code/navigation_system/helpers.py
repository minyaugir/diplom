
"""
Вспомогательные функции: вращения, матрицы и т.п.
В работе используется локальная навигационная система NED:
  N — север (x), E — восток (y), D — вниз (z).
Ориентация описывается углом курса (yaw) при допущении малых крена и тангажа.
"""

from __future__ import annotations
import numpy as np

G = 9.80665  # м/с^2

def rotmat_yaw(yaw: float) -> np.ndarray:
    """
    Матрица поворота из корпуса (body) в NED при нулевых roll/pitch.
    yaw — курс, рад.
    """
    c = np.cos(yaw)
    s = np.sin(yaw)
    # body -> NED
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)

def drotmat_dyaw(yaw: float) -> np.ndarray:
    """
    Производная матрицы C_bn по yaw.
    """
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([[-s, -c, 0.0],
                     [ c, -s, 0.0],
                     [0.0, 0.0, 0.0]], dtype=float)

def wrap_pi(angle: float) -> float:
    """Приведение угла к диапазону (-pi, pi]."""
    import math
    a = (angle + math.pi) % (2*math.pi) - math.pi
    return a

def rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))

def ensure_dir(path: str) -> None:
    import os
    os.makedirs(path, exist_ok=True)
