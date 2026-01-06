\
import numpy as np
from navigation_system.extended_kalman_filter import ExtendedKalmanFilter

def test_ekf_shapes():
    x0 = np.zeros(11)
    P0 = np.eye(11)
    Q = np.eye(11)*1e-6
    ekf = ExtendedKalmanFilter(x0, P0, Q)
    ekf.predict(np.zeros(3), 0.0, 0.01)
    assert ekf.state.x.shape == (11,)
    assert ekf.state.P.shape == (11,11)
