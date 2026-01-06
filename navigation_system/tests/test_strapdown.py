\
import numpy as np
from navigation_system.strapdown_integration import INSState, ins_propagate

def test_ins_propagate_runs():
    st = INSState(p=np.zeros(3), v=np.zeros(3), yaw=0.0)
    st2 = ins_propagate(st, acc_body=np.zeros(3), gyro_z=0.0, dt=0.01)
    assert st2.p.shape == (3,)
