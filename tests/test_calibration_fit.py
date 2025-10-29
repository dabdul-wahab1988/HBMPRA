import numpy as np


def test_linear_fit_positive_slope():
    # synthetic EDI (mg/kg-d) and BLL (µg/dL) roughly proportional
    edis = np.array([0.0, 0.001, 0.002, 0.005, 0.01])
    blls = np.array([0.5, 0.6, 0.7, 0.95, 1.4])
    A = np.vstack([np.ones_like(edis), edis]).T
    b0, k = np.linalg.lstsq(A, blls, rcond=None)[0]
    assert k > 0
    assert b0 >= 0
