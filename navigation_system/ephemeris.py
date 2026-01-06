"""navigation_system.ephemeris

Преобразования координат (WGS‑84):
- LLA (lat, lon, h) <-> ECEF
- ECEF <-> NED (относительно опорной точки)

Модуль не использует внешние эфемериды; он предназначен для корректного описания
геометрии и для возможного расширения проекта до tight coupling.

Все углы — в радианах.
"""

from __future__ import annotations

import numpy as np

# Параметры эллипсоида WGS‑84
A_WGS84 = 6378137.0
F_WGS84 = 1.0 / 298.257223563
E2_WGS84 = F_WGS84 * (2.0 - F_WGS84)


def lla_to_ecef(lat: float, lon: float, h: float) -> np.ndarray:
    """Преобразование LLA -> ECEF.

    lat, lon — радианы, h — высота над эллипсоидом, м.
    """
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    N = A_WGS84 / np.sqrt(1.0 - E2_WGS84 * sin_lat**2)
    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = (N * (1.0 - E2_WGS84) + h) * sin_lat
    return np.array([x, y, z], dtype=float)


def ecef_to_lla(x: float, y: float, z: float, iters: int = 6) -> tuple[float, float, float]:
    """Приближенное преобразование ECEF -> LLA (итерационный метод)."""
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1.0 - E2_WGS84))

    for _ in range(iters):
        sin_lat = np.sin(lat)
        N = A_WGS84 / np.sqrt(1.0 - E2_WGS84 * sin_lat**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1.0 - E2_WGS84 * N / (N + h)))

    sin_lat = np.sin(lat)
    N = A_WGS84 / np.sqrt(1.0 - E2_WGS84 * sin_lat**2)
    h = p / np.cos(lat) - N
    return float(lat), float(lon), float(h)


def ecef_to_ned_matrix(lat0: float, lon0: float) -> np.ndarray:
    """Матрица поворота из ECEF в NED относительно опорной точки (lat0, lon0)."""
    sL = np.sin(lat0)
    cL = np.cos(lat0)
    sλ = np.sin(lon0)
    cλ = np.cos(lon0)

    # ECEF -> NED
    return np.array([
        [-sL * cλ, -sL * sλ, cL],
        [-sλ,       cλ,      0.0],
        [-cL * cλ, -cL * sλ, -sL],
    ], dtype=float)


def ecef_to_ned(ecef: np.ndarray, ref_lla: tuple[float, float, float]) -> np.ndarray:
    """ECEF -> NED относительно ref_lla=(lat0, lon0, h0)."""
    lat0, lon0, h0 = ref_lla
    ecef0 = lla_to_ecef(lat0, lon0, h0)
    R = ecef_to_ned_matrix(lat0, lon0)
    return R @ (ecef - ecef0)


def ned_to_ecef(ned: np.ndarray, ref_lla: tuple[float, float, float]) -> np.ndarray:
    """NED -> ECEF относительно ref_lla=(lat0, lon0, h0)."""
    lat0, lon0, h0 = ref_lla
    ecef0 = lla_to_ecef(lat0, lon0, h0)
    R = ecef_to_ned_matrix(lat0, lon0)
    return ecef0 + R.T @ ned
