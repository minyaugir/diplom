import numpy as np
from navigation_system.ephemeris import llh_to_ecef, ecef_to_llh

def test_llh_roundtrip():
    lat = np.deg2rad(55.75)
    lon = np.deg2rad(37.62)
    h = 200.0
    xyz = llh_to_ecef(lat, lon, h)
    llh = ecef_to_llh(xyz)
    assert abs(llh[0] - lat) < 1e-6
    assert abs(llh[1] - lon) < 1e-6
