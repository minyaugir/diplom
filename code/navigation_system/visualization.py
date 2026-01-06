
"""
Построение графиков результатов моделирования.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def _ensure(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def plot_trajectory(p_true: np.ndarray, p_ins: np.ndarray, p_ekf: np.ndarray, out_png: str):
    _ensure(out_png)
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(p_true[:,0], p_true[:,1], -p_true[:,2], label="Истина")
    ax.plot(p_ins[:,0], p_ins[:,1], -p_ins[:,2], label="ИНС (без корр.)")
    ax.plot(p_ekf[:,0], p_ekf[:,1], -p_ekf[:,2], label="ИНС+EKF")
    ax.set_xlabel("N, м")
    ax.set_ylabel("E, м")
    ax.set_zlabel("H, м")
    ax.set_title("Траектория полета БПЛА (3D)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_pos_errors(t: np.ndarray, e_ins: np.ndarray, e_ekf: np.ndarray, out_png: str):
    _ensure(out_png)
    fig = plt.figure(figsize=(8,5))
    plt.plot(t, e_ins[:,0], label="e_N ИНС")
    plt.plot(t, e_ins[:,1], label="e_E ИНС")
    plt.plot(t, e_ins[:,2], label="e_D ИНС")
    plt.plot(t, e_ekf[:,0], label="e_N EKF")
    plt.plot(t, e_ekf[:,1], label="e_E EKF")
    plt.plot(t, e_ekf[:,2], label="e_D EKF")
    plt.xlabel("t, с")
    plt.ylabel("ошибка, м")
    plt.title("Ошибки координат (N/E/D)")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_bias(t: np.ndarray, b_true: np.ndarray, b_est: np.ndarray, labels: list[str], out_png: str):
    _ensure(out_png)
    fig = plt.figure(figsize=(8,5))
    for i, lab in enumerate(labels):
        plt.plot(t, b_true[:,i], label=f"{lab} истина")
        plt.plot(t, b_est[:,i], linestyle="--", label=f"{lab} оценка")
    plt.xlabel("t, с")
    plt.ylabel("значение")
    plt.title("Оценка смещений датчиков")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_yaw_error(t: np.ndarray, e_ins: np.ndarray, e_ekf: np.ndarray, out_png: str):
    _ensure(out_png)
    fig = plt.figure(figsize=(8,4))
    plt.plot(t, e_ins, label="ИНС")
    plt.plot(t, e_ekf, label="EKF")
    plt.xlabel("t, с")
    plt.ylabel("ошибка курса, рад")
    plt.title("Ошибка курса (yaw)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_altitude_profile(t: np.ndarray, p_true: np.ndarray, p_ins: np.ndarray, p_ekf: np.ndarray, out_png: str):
    """Профиль высоты H (м): H = -D в системе NED."""
    _ensure(out_png)
    H_true = -p_true[:,2]
    H_ins  = -p_ins[:,2]
    H_ekf  = -p_ekf[:,2]

    plt.figure(figsize=(7,4))
    plt.plot(t, H_true, label="Истина (Truth)")
    plt.plot(t, H_ins,  label="ИНС (INS)")
    plt.plot(t, H_ekf,  label="ИНС+ЕКФ (INS+EKF)")
    plt.xlabel("t, с")
    plt.ylabel("H, м")
    plt.title("Профиль высоты (Altitude)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_error_norm(t: np.ndarray, e_p_ins: np.ndarray, e_p_ekf: np.ndarray, out_png: str, last_seconds: float | None = None):
    """Норма ошибки положения ||e_p|| (м), при необходимости — на хвостовом интервале."""
    _ensure(out_png)
    err_ins = np.linalg.norm(e_p_ins, axis=1)
    err_ekf = np.linalg.norm(e_p_ekf, axis=1)

    if last_seconds is not None and last_seconds > 0:
        t0 = t[-1] - last_seconds
        m = t >= t0
        t = t[m]; err_ins = err_ins[m]; err_ekf = err_ekf[m]

    plt.figure(figsize=(7,4))
    plt.plot(t, err_ins, label="ИНС (INS)")
    plt.plot(t, err_ekf, label="ИНС+ЕКФ (INS+EKF)")
    plt.xlabel("t, с")
    plt.ylabel("||e_p||, м")
    plt.title("Норма ошибки положения (Position error norm)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_attitude_errors(t: np.ndarray, e_yaw: np.ndarray, e_roll: np.ndarray | None, e_pitch: np.ndarray | None, out_png: str):
    """Ошибки ориентации (рыскание/крен/тангаж). В упрощённой модели roll/pitch могут быть нулевыми."""
    _ensure(out_png)
    plt.figure(figsize=(7,4))
    plt.plot(t, e_yaw, label="Рыскание (Yaw)")
    if e_roll is not None:
        plt.plot(t, e_roll, label="Крен (Roll)")
    if e_pitch is not None:
        plt.plot(t, e_pitch, label="Тангаж (Pitch)")
    plt.xlabel("t, с")
    plt.ylabel("Ошибка, рад")
    plt.title("Ошибки ориентации (Attitude errors)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
