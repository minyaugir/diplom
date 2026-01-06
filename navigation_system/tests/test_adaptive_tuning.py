import numpy as np
from navigation_system.adaptive_tuning import AdaptiveRTuner


def test_adaptive_r_tuner_scales_up():
    Rpos = np.eye(3)
    Rvel = np.eye(3)
    t = AdaptiveRTuner(Rpos, Rvel)
    Rp, Rv = t.update(np.array([100.0, 0.0, 0.0]), None)
    assert float(Rp[0,0]) > 1.0
