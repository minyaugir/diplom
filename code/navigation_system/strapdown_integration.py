
"""
Страпдаун-интегрирование ИНС (упрощенная 3D-модель с yaw).

Состояние:
  p = [N, E, D] (м)
  v = [Vn, Ve, Vd] (м/с)
  yaw (рад)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .helpers import G, rotmat_yaw, wrap_pi

@dataclass
class INSState:
    p: np.ndarray   # (3,)
    v: np.ndarray   # (3,)
    yaw: float

def ins_propagate(state: INSState,
                  acc_body: np.ndarray,
                  gyro_z: float,
                  dt: float) -> INSState:
    """
    Прямое счисление пути по измерениям IMU без коррекции смещений.
    acc_body — измеренная специфическая сила в body, м/с^2
    gyro_z — измеренная угловая скорость по курсу, рад/с
    """
    # Обновление курса (yaw)
    yaw_new = wrap_pi(state.yaw + gyro_z*dt)

    # Переход body -> NED
    C_bn = rotmat_yaw(yaw_new)

    # a_n = C_bn * f_b + g_n
    g_n = np.array([0.0, 0.0, G])
    a_n = C_bn @ acc_body + g_n

    # Интегрирование скорости и положения (Эйлер)
    v_new = state.v + a_n*dt
    p_new = state.p + state.v*dt

    return INSState(p=p_new, v=v_new, yaw=yaw_new)
