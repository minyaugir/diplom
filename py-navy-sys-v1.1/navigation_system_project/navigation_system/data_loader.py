\
"""
Генерация эталонной траектории БПЛА и синтетических измерений датчиков.

Сценарии:
- uav_route: типовой маршрут (взлет – полет – разворот – посадка)
- uav_orbit: полет по окружности
- uav_route_with_outage: как uav_route, но с "провалом" GNSS на заданном интервале
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .helpers import G, rotmat_yaw

@dataclass
class TruthData:
    t: np.ndarray
    p_ned: np.ndarray   # (N,3)
    v_ned: np.ndarray   # (N,3)
    a_ned: np.ndarray   # (N,3)
    yaw: np.ndarray     # (N,)

@dataclass
class SensorData:
    acc_body: np.ndarray     # (N,3) specific force in body, m/s^2
    gyro_z: np.ndarray       # (N,) yaw rate, rad/s
    gnss_pos: dict           # {k: np.array([N,E,D])}
    gnss_vel: dict           # {k: np.array([Vn,Ve,Vd])}
    baro_h: dict             # {k: height_up_m} (positive up)

def scenario_uav_route(t: np.ndarray) -> TruthData:
    """
    Маршрут: взлет до 60 м, прямолинейный полет, разворот, возвращение.
    Модель упрощенная (yaw, без roll/pitch).
    """
    N = t.size
    p = np.zeros((N,3), dtype=float)
    v = np.zeros((N,3), dtype=float)
    yaw = np.zeros(N, dtype=float)

    # Параметры маршрута
    climb_end = 20.0
    cruise_end = 140.0
    descend_end = 180.0

    # Скорости
    v_cruise = 12.0  # м/с
    climb_rate = -3.0  # D, м/с (вверх => D отриц.)
    descend_rate =  3.0

    # Сегменты
    for i, ti in enumerate(t):
        if ti <= climb_end:
            # взлет: стоим по горизонтали, набираем высоту
            v[i] = np.array([0.0, 0.0, climb_rate])
            yaw[i] = 0.0
        elif ti <= cruise_end:
            # горизонтальный полет: прямоугольник
            # 20..80: на север, 80..110: на восток, 110..140: на юг
            if ti <= 80.0:
                v[i] = np.array([v_cruise, 0.0, 0.0])
                yaw[i] = 0.0
            elif ti <= 110.0:
                v[i] = np.array([0.0, v_cruise, 0.0])
                yaw[i] = np.pi/2
            else:
                v[i] = np.array([-v_cruise, 0.0, 0.0])
                yaw[i] = np.pi
        else:
            # снижение и посадка: возвращаемся к точке старта по оси E и садимся
            v[i] = np.array([0.0, -6.0, descend_rate])
            yaw[i] = -np.pi/2

    # Интегрирование положения
    dt = t[1] - t[0]
    for k in range(1, N):
        p[k] = p[k-1] + v[k-1]*dt

    # Ограничиваем высоту: D = -height, поэтому D не меньше -60 и не больше 0
    # "подрезаем" высоту во время снижения, чтобы закончить около 0
    p[:,2] = np.clip(p[:,2], -60.0, 0.0)

    # Ускорения как численная производная скорости
    a = np.zeros_like(v)
    a[1:] = (v[1:] - v[:-1]) / dt
    a[0] = a[1]

    return TruthData(t=t, p_ned=p, v_ned=v, a_ned=a, yaw=yaw)

def scenario_uav_orbit(t: np.ndarray) -> TruthData:
    """
    Полет по окружности радиуса R на высоте 50 м с постоянной скоростью.
    """
    N = t.size
    R = 120.0
    v_mag = 14.0
    omega = v_mag / R  # рад/с

    yaw = omega*t + np.pi/2  # курс по касательной
    p = np.zeros((N,3), dtype=float)
    p[:,0] = R*np.cos(omega*t)
    p[:,1] = R*np.sin(omega*t)
    p[:,2] = -50.0  # D вниз => -50 означает 50 м вверх

    # скорость в NED
    v = np.zeros_like(p)
    v[:,0] = -R*omega*np.sin(omega*t)
    v[:,1] =  R*omega*np.cos(omega*t)
    v[:,2] = 0.0

    # ускорение
    a = np.zeros_like(p)
    a[:,0] = -R*(omega**2)*np.cos(omega*t)
    a[:,1] = -R*(omega**2)*np.sin(omega*t)
    a[:,2] = 0.0
    return TruthData(t=t, p_ned=p, v_ned=v, a_ned=a, yaw=yaw)

def generate_sensors(truth: TruthData,
                     accel_noise_std: float,
                     gyro_noise_std: float,
                     accel_bias_init: np.ndarray,
                     gyro_bias_init: float,
                     accel_bias_rw_std: float,
                     gyro_bias_rw_std: float,
                     gnss_rate_hz: float,
                     gnss_pos_noise_std: np.ndarray,
                     gnss_vel_noise_std: np.ndarray,
                     outage_enabled: bool,
                     outage_start_s: float,
                     outage_end_s: float,
                     baro_enabled: bool,
                     baro_rate_hz: float,
                     baro_h_noise_std: float,
                     rng: np.random.Generator) -> SensorData:
    """
    Синтетические измерения IMU (specific force + yaw rate),
    GNSS (позиция и скорость) и барометрическая высота.
    """
    t = truth.t
    dt = t[1] - t[0]
    N = t.size

    # Истинные специфические силы в body: f_b = C_nb * (a_n - g_n)
    # где g_n = [0,0,G] (вниз по D)
    g_n = np.array([0.0, 0.0, G])
    acc_body_true = np.zeros((N,3), dtype=float)
    gyro_z_true = np.zeros(N, dtype=float)

    # yaw rate как производная yaw
    gyro_z_true[1:] = (truth.yaw[1:] - truth.yaw[:-1]) / dt
    gyro_z_true[0] = gyro_z_true[1]

    for k in range(N):
        C_bn = rotmat_yaw(truth.yaw[k])       # body -> NED
        C_nb = C_bn.T                         # NED -> body
        acc_body_true[k] = C_nb @ (truth.a_ned[k] - g_n)

    # Смещения как случайное блуждание
    accel_bias = np.zeros((N,3), dtype=float)
    gyro_bias = np.zeros(N, dtype=float)
    accel_bias[0] = accel_bias_init
    gyro_bias[0] = gyro_bias_init
    for k in range(1, N):
        accel_bias[k] = accel_bias[k-1] + accel_bias_rw_std*np.sqrt(dt)*rng.standard_normal(3)
        gyro_bias[k] = gyro_bias[k-1] + gyro_bias_rw_std*np.sqrt(dt)*rng.standard_normal()

    # Измерения IMU
    acc_meas = acc_body_true + accel_bias + accel_noise_std*rng.standard_normal((N,3))
    gyro_meas = gyro_z_true + gyro_bias + gyro_noise_std*rng.standard_normal(N)

    # GNSS измерения (позиция+скорость) с заданной частотой
    gnss_pos = {}
    gnss_vel = {}
    step_gnss = int(round(1.0/(gnss_rate_hz*dt)))
    for k in range(0, N, step_gnss):
        ti = t[k]
        if outage_enabled and (outage_start_s <= ti <= outage_end_s):
            continue
        p_meas = truth.p_ned[k] + gnss_pos_noise_std*rng.standard_normal(3)
        v_meas = truth.v_ned[k] + gnss_vel_noise_std*rng.standard_normal(3)
        gnss_pos[k] = p_meas
        gnss_vel[k] = v_meas

    # Барометр: измеряем высоту "вверх" (Up = -D)
    baro_h = {}
    if baro_enabled:
        step_baro = int(round(1.0/(baro_rate_hz*dt)))
        for k in range(0, N, step_baro):
            h_true = -truth.p_ned[k,2]
            baro_h[k] = float(h_true + baro_h_noise_std*rng.standard_normal())

    return SensorData(
        acc_body=acc_meas,
        gyro_z=gyro_meas,
        gnss_pos=gnss_pos,
        gnss_vel=gnss_vel,
        baro_h=baro_h
    )
