\
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
