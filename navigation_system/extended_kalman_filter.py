\
"""
Расширенный фильтр Калмана (EKF) для интеграции ИНС и внешних измерений (GNSS/баро).

Вектор состояния (11):
  x = [p_N, p_E, p_D,  v_N, v_E, v_D,  yaw,  b_g,  b_ax, b_ay, b_az]^T

Модель:
  yaw_{k+1} = yaw_k + (gyro_z - b_g)*dt
  a_n = C(yaw)*(acc_body - b_a) + g_n
  v_{k+1} = v_k + a_n*dt
  p_{k+1} = p_k + v_k*dt
  b_g, b_a — случайное блуждание (в Q)

Измерения:
  GNSS: z = [p, v]  (если доступно)
  Баро: z_h = h_up = -p_D
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .helpers import G, rotmat_yaw, drotmat_dyaw, wrap_pi

@dataclass
class EKFState:
    x: np.ndarray  # (11,)
    P: np.ndarray  # (11,11)

class ExtendedKalmanFilter:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q_base: np.ndarray):
        self.state = EKFState(x=x0.copy(), P=P0.copy())
        self.Q_base = Q_base.copy()

    @staticmethod
    def f(x: np.ndarray, acc_body: np.ndarray, gyro_z: float, dt: float) -> np.ndarray:
        # unpack
        p = x[0:3]
        v = x[3:6]
        yaw = x[6]
        b_g = x[7]
        b_a = x[8:11]

        yaw_new = wrap_pi(yaw + (gyro_z - b_g)*dt)
        C_bn = rotmat_yaw(yaw_new)
        g_n = np.array([0.0, 0.0, G])
        a_n = C_bn @ (acc_body - b_a) + g_n

        v_new = v + a_n*dt
        p_new = p + v*dt

        x_new = x.copy()
        x_new[0:3] = p_new
        x_new[3:6] = v_new
        x_new[6] = yaw_new
        # biases as random walk (mean stays)
        x_new[7] = b_g
        x_new[8:11] = b_a
        return x_new

    @staticmethod
    def jacobian_F(x: np.ndarray, acc_body: np.ndarray, gyro_z: float, dt: float) -> np.ndarray:
        """
        Якобиан F = df/dx для текущего шага.
        """
        F = np.eye(11, dtype=float)

        # p depends on v
        F[0:3, 3:6] = np.eye(3)*dt

        yaw = x[6]
        b_a = x[8:11]
        yaw_new = wrap_pi(yaw + (gyro_z - x[7])*dt)

        C_bn = rotmat_yaw(yaw_new)
        dC = drotmat_dyaw(yaw_new)  # dC/dyaw at yaw_new

        u = (acc_body - b_a)  # (3,)
        # v_next = v + (C_bn*u + g)*dt
        # dv/dyaw = (dC/dyaw * u) * dt
        dv_dyaw = (dC @ u) * dt
        F[3:6, 6] = dv_dyaw

        # dv/db_a = -C_bn*dt
        F[3:6, 8:11] = -C_bn*dt

        # yaw_next depends on b_g
        F[6, 7] = -dt

        return F

    def predict(self, acc_body: np.ndarray, gyro_z: float, dt: float):
        x = self.state.x
        P = self.state.P

        x_pred = self.f(x, acc_body, gyro_z, dt)
        F = self.jacobian_F(x, acc_body, gyro_z, dt)

        # Подстройка Q: базовая ковариация процесса масштабируется dt
        Q = self.Q_base * dt

        P_pred = F @ P @ F.T + Q
        self.state = EKFState(x=x_pred, P=P_pred)

    def update_gnss(self, z_pos: np.ndarray | None, z_vel: np.ndarray | None, R_pos: np.ndarray, R_vel: np.ndarray):
        """
        Коррекция по GNSS.
        z_pos: (3,) или None
        z_vel: (3,) или None
        """
        x = self.state.x
        P = self.state.P

        blocks = []
        zs = []
        Rs = []

        if z_pos is not None:
            Hpos = np.zeros((3,11), dtype=float)
            Hpos[:,0:3] = np.eye(3)
            blocks.append(Hpos)
            zs.append(z_pos)
            Rs.append(R_pos)

        if z_vel is not None:
            Hvel = np.zeros((3,11), dtype=float)
            Hvel[:,3:6] = np.eye(3)
            blocks.append(Hvel)
            zs.append(z_vel)
            Rs.append(R_vel)

        if not blocks:
            return

        H = np.vstack(blocks)
        z = np.concatenate(zs)
        R = np.block([[Rs[i] if i==j else np.zeros_like(Rs[0]) for j in range(len(Rs))] for i in range(len(Rs))])

        # Innovation
        y = z - H @ x

        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        x_upd = x + K @ y
        P_upd = (np.eye(11) - K @ H) @ P

        # Нормализация yaw
        x_upd[6] = wrap_pi(x_upd[6])

        self.state = EKFState(x=x_upd, P=P_upd)

    def update_baro(self, h_up: float, R_h: float):
        """
        Баро-высота: h_up = -p_D + noise
        """
        x = self.state.x
        P = self.state.P
        H = np.zeros((1,11), dtype=float)
        H[0,2] = -1.0  # h = -D
        z = np.array([h_up], dtype=float)
        y = z - H @ x
        S = H @ P @ H.T + np.array([[R_h]], dtype=float)
        K = P @ H.T @ np.linalg.inv(S)
        x_upd = x + (K @ y).reshape(-1)
        P_upd = (np.eye(11) - K @ H) @ P
        self.state = EKFState(x=x_upd, P=P_upd)
