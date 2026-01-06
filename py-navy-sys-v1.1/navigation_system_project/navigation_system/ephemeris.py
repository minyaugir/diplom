"""navigation_system.ephemeris

Преобразования координат (WGS‑84):
- LLH (lat, lon, h) <-> ECEF
- ECEF <-> NED (относительно опорной точки)

Модуль не использует внешние эфемериды; он предназначен для корректного описания
геометрии и для возможного расширения проекта до tight coupling.

Все углы — в радианах.
"""

from __future__ import annotations

import numpy as np

# WGS‑84 ellipsoid constants
_WGS84_A = 6378137.0                 # semi-major axis, meters
_WGS84_F = 1.0 / 298.257223563       # flattening
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)  # first eccentricity squared


def llh_to_ecef(lat: float, lon: float, h: float) -> np.ndarray:
    """Convert geodetic coordinates (lat, lon in rad, h in m) to ECEF XYZ (m).

    Returns:
        np.ndarray shape (3,)
    """
    s = np.sin(lat)
    c = np.cos(lat)
    sλ = np.sin(lon)
    cλ = np.cos(lon)

    # Radius of curvature in the prime vertical
    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * s * s)

    x = (N + h) * c * cλ
    y = (N + h) * c * sλ
    z = (N * (1.0 - _WGS84_E2) + h) * s
    return np.array([x, y, z], dtype=float)


def ecef_to_llh(xyz: np.ndarray, *, max_iter: int = 10) -> np.ndarray:
    """Convert ECEF XYZ (m) to geodetic (lat, lon in rad, h in m).

    Uses iterative method (Bowring-like) suitable for navigation use.

    Args:
        xyz: np.ndarray shape (3,)
        max_iter: number of iterations

    Returns:
        np.ndarray [lat, lon, h]
    """
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    lon = np.arctan2(y, x)

    p = np.hypot(x, y)
    if p < 1e-12:
        # At poles: lon undefined; set to 0.
        lon = 0.0

    # Initial latitude estimate
    lat = np.arctan2(z, p * (1.0 - _WGS84_E2))
    h = 0.0

    for _ in range(max_iter):
        s = np.sin(lat)
        N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * s * s)
        h_new = p / np.cos(lat) - N
        lat_new = np.arctan2(z, p * (1.0 - _WGS84_E2 * N / (N + h_new)))
        if abs(lat_new - lat) < 1e-12 and abs(h_new - h) < 1e-6:
            lat, h = lat_new, h_new
            break
        lat, h = lat_new, h_new

    return np.array([lat, lon, h], dtype=float)


# Backward-compatible aliases (some texts use LLA/LLH)
def lla_to_ecef(lat: float, lon: float, h: float) -> np.ndarray:
    return llh_to_ecef(lat, lon, h)


def ecef_to_lla(xyz: np.ndarray) -> np.ndarray:
    return ecef_to_llh(xyz)


def ecef_to_ned_matrix(lat0: float, lon0: float) -> np.ndarray:
    """Rotation matrix from ECEF to local NED at reference (lat0, lon0) in rad."""
    sL = np.sin(lat0)
    cL = np.cos(lat0)
    sλ = np.sin(lon0)
    cλ = np.cos(lon0)

    # ECEF -> NED
    return np.array(
        [
            [-sL * cλ, -sL * sλ, cL],
            [-sλ,       cλ,      0.0],
            [-cL * cλ, -cL * sλ, -sL],
        ],
        dtype=float,
    )


def ecef_to_ned(ecef: np.ndarray, ref_llh: tuple[float, float, float]) -> np.ndarray:
    """ECEF -> NED relative to ref_llh=(lat0, lon0, h0)."""
    lat0, lon0, h0 = ref_llh
    ecef0 = llh_to_ecef(lat0, lon0, h0)
    R = ecef_to_ned_matrix(lat0, lon0)
    return R @ (ecef - ecef0)


def ned_to_ecef(ned: np.ndarray, ref_llh: tuple[float, float, float]) -> np.ndarray:
    """NED -> ECEF relative to ref_llh=(lat0, lon0, h0)."""
    lat0, lon0, h0 = ref_llh
    ecef0 = llh_to_ecef(lat0, lon0, h0)
    R = ecef_to_ned_matrix(lat0, lon0)
    return ecef0 + R.T @ ned
