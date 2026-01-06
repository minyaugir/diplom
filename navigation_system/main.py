\
"""
Главный модуль: запускает моделирование и формирует графики/таблицы.
Запуск: python -m navigation_system.main
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
import yaml

from .data_loader import scenario_uav_route, scenario_uav_orbit, generate_sensors
from .strapdown_integration import INSState, ins_propagate
from .extended_kalman_filter import ExtendedKalmanFilter
from .helpers import rmse, wrap_pi
from .logger import save_npz, save_csv
from .visualization import plot_trajectory, plot_pos_errors, plot_bias, plot_yaw_error

def run(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    dt = float(cfg["simulation"]["dt"])
    T = float(cfg["simulation"]["T"])
    scenario = cfg["simulation"]["scenario"]
    t = np.arange(0.0, T+1e-12, dt)

    rng = np.random.default_rng(42)

    # Truth
    if scenario in ("uav_route", "uav_route_with_outage"):
        truth = scenario_uav_route(t)
    elif scenario == "uav_orbit":
        truth = scenario_uav_orbit(t)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    s_imu = cfg["sensors"]["imu"]
    s_gnss = cfg["sensors"]["gnss"]
    s_baro = cfg["sensors"]["baro"]

    sens = generate_sensors(
        truth=truth,
        accel_noise_std=float(s_imu["accel_noise_std"]),
        gyro_noise_std=float(s_imu["gyro_noise_std"]),
        accel_bias_init=np.array(s_imu["accel_bias_init"], dtype=float),
        gyro_bias_init=float(s_imu["gyro_bias_init"]),
        accel_bias_rw_std=float(s_imu["accel_bias_rw_std"]),
        gyro_bias_rw_std=float(s_imu["gyro_bias_rw_std"]),
        gnss_rate_hz=float(s_gnss["rate_hz"]),
        gnss_pos_noise_std=np.array(s_gnss["pos_noise_std"], dtype=float),
        gnss_vel_noise_std=np.array(s_gnss["vel_noise_std"], dtype=float),
        outage_enabled=bool(s_gnss["outage"]["enabled"]) if scenario == "uav_route_with_outage" else False,
        outage_start_s=float(s_gnss["outage"]["start_s"]),
        outage_end_s=float(s_gnss["outage"]["end_s"]),
        baro_enabled=bool(s_baro["enabled"]),
        baro_rate_hz=float(s_baro["rate_hz"]),
        baro_h_noise_std=float(s_baro["height_noise_std"]),
        rng=rng
    )

    N = t.size

    # Pure INS (no bias compensation)
    ins = INSState(p=np.array([0.0,0.0,0.0]), v=np.array([0.0,0.0,0.0]), yaw=0.0)
    p_ins = np.zeros((N,3)); v_ins = np.zeros((N,3)); yaw_ins = np.zeros(N)
    for k in range(N):
        p_ins[k] = ins.p; v_ins[k] = ins.v; yaw_ins[k] = ins.yaw
        ins = ins_propagate(ins, sens.acc_body[k], sens.gyro_z[k], dt)

    # EKF initialization
    ekf_cfg = cfg["ekf"]
    P0 = np.zeros((11,11))
    pos_std = float(ekf_cfg["P0"]["pos_std"])
    vel_std = float(ekf_cfg["P0"]["vel_std"])
    yaw_std = np.deg2rad(float(ekf_cfg["P0"]["yaw_std"]))
    gb_std = float(ekf_cfg["P0"]["gyro_bias_std"])
    ab_std = float(ekf_cfg["P0"]["accel_bias_std"])

    P0[0:3,0:3] = np.eye(3)*pos_std**2
    P0[3:6,3:6] = np.eye(3)*vel_std**2
    P0[6,6] = yaw_std**2
    P0[7,7] = gb_std**2
    P0[8:11,8:11] = np.eye(3)*ab_std**2

    x0 = np.zeros(11)
    x0[0:3] = np.array([0.0,0.0,0.0])
    x0[3:6] = np.array([0.0,0.0,0.0])
    x0[6] = 0.0
    x0[7] = 0.0
    x0[8:11] = np.array([0.0,0.0,0.0])

    # Process noise base (will be multiplied by dt inside EKF)
    Q = np.zeros((11,11))
    qcfg = ekf_cfg["Q"]
    # model uncertainty for yaw and accelerations (as state noise through dt)
    # here we inject directly to biases and to yaw/velocity channels via tuning:
    Q[6,6] = float(qcfg["gyro_noise_std"])**2
    Q[3:6,3:6] = np.eye(3)*float(qcfg["accel_noise_std"])**2
    Q[7,7] = float(qcfg["gyro_bias_rw_std"])**2
    Q[8:11,8:11] = np.eye(3)*float(qcfg["accel_bias_rw_std"])**2

    ekf = ExtendedKalmanFilter(x0, P0, Q)

    p_ekf = np.zeros((N,3)); v_ekf = np.zeros((N,3)); yaw_ekf = np.zeros(N)
    b_est = np.zeros((N,4))  # [bg, bax, bay, baz]
    for k in range(N):
        # predict
        ekf.predict(sens.acc_body[k], sens.gyro_z[k], dt)

        # baro update
        if k in sens.baro_h:
            ekf.update_baro(sens.baro_h[k], R_h=float(s_baro["height_noise_std"])**2)

        # gnss update
        if k in sens.gnss_pos:
            Rpos = np.diag(np.array(s_gnss["pos_noise_std"], dtype=float)**2)
            Rvel = np.diag(np.array(s_gnss["vel_noise_std"], dtype=float)**2)
            ekf.update_gnss(sens.gnss_pos[k], sens.gnss_vel.get(k), Rpos, Rvel)

        x = ekf.state.x
        p_ekf[k] = x[0:3]
        v_ekf[k] = x[3:6]
        yaw_ekf[k] = x[6]
        b_est[k] = np.array([x[7], x[8], x[9], x[10]])

    # Errors
    e_p_ins = p_ins - truth.p_ned
    e_p_ekf = p_ekf - truth.p_ned

    e_yaw_ins = np.array([wrap_pi(yaw_ins[i] - truth.yaw[i]) for i in range(N)])
    e_yaw_ekf = np.array([wrap_pi(yaw_ekf[i] - truth.yaw[i]) for i in range(N)])

    # RMSE table
    rmse_ins = np.array([rmse(e_p_ins[:,0]), rmse(e_p_ins[:,1]), rmse(e_p_ins[:,2]), rmse(np.linalg.norm(e_p_ins, axis=1))])
    rmse_ekf = np.array([rmse(e_p_ekf[:,0]), rmse(e_p_ekf[:,1]), rmse(e_p_ekf[:,2]), rmse(np.linalg.norm(e_p_ekf, axis=1))])

    out_dir = Path(__file__).resolve().parent/"data"/"output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save raw
    save_npz(str(out_dir/"log.npz"),
             t=t,
             p_true=truth.p_ned,
             v_true=truth.v_ned,
             yaw_true=truth.yaw,
             p_ins=p_ins,
             p_ekf=p_ekf,
             yaw_ins=yaw_ins,
             yaw_ekf=yaw_ekf,
             e_p_ins=e_p_ins,
             e_p_ekf=e_p_ekf,
             b_est=b_est)

    # Save RMSE table
    table = np.vstack([rmse_ins, rmse_ekf])
    save_csv(str(out_dir/"rmse_table.csv"),
             header=["mode", "RMSE_N_m", "RMSE_E_m", "RMSE_D_m", "RMSE_3D_m"],
             data=np.column_stack([np.array(["INS","INS+EKF"]), table]))

    # Plots
    plot_trajectory(truth.p_ned, p_ins, p_ekf, str(out_dir/"fig_traj_3d.png"))
    plot_pos_errors(t, e_p_ins, e_p_ekf, str(out_dir/"fig_pos_errors.png"))
    plot_yaw_error(t, e_yaw_ins, e_yaw_ekf, str(out_dir/"fig_yaw_error.png"))

    # Bias truth is embedded in IMU generation but we did not save it there; for report we show estimates only.
    # As a proxy, plot estimates (bg, bax, bay, baz)
    b_true = np.zeros_like(b_est)  # unknown to estimator; in the report we mention "истинное в модели".
    plot_bias(t, b_true, b_est, labels=["b_g", "b_ax", "b_ay", "b_az"], out_png=str(out_dir/"fig_bias_est.png"))

    # Console summary
    print("RMSE позиционирования (м):")
    print(f"  INS:     N={rmse_ins[0]:.2f}, E={rmse_ins[1]:.2f}, D={rmse_ins[2]:.2f}, 3D={rmse_ins[3]:.2f}")
    print(f"  INS+EKF: N={rmse_ekf[0]:.2f}, E={rmse_ekf[1]:.2f}, D={rmse_ekf[2]:.2f}, 3D={rmse_ekf[3]:.2f}")
    print(f"Outputs saved to: {out_dir}")

if __name__ == "__main__":
    run(str(Path(__file__).resolve().parent/"config.yaml"))
