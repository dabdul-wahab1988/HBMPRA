import math
import os
import sys
# Ensure repository root is on sys.path so tests can import modules like hbmpra
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hbmpra import dermal_dad_water
from bll_engines import compute_bll, OneCompParams


def test_dermal_numeric_pb():
    # Adult example from spec
    C = 0.01  # mg/L
    Kp = 1e-4 # cm/hr
    SA = 19652
    t = 0.71
    EV = 1
    EF = 350
    ED = 26
    BW = 70
    AT = 26*365
    dad = dermal_dad_water(C, Kp, SA, t, EV, EF, ED, BW, AT)
    # expected magnitude 1e-6 to 1e-5 mg/kg-d
    assert 1e-7 < dad < 1e-3


def test_bll_onecomp_roundtrip():
    # CDI ingestion example
    C = 0.001 # mg/L
    IR = 2.0
    BW = 70
    # compute edi mg/kg-d
    edi = (C * IR) / BW
    params = OneCompParams(f_abs=0.5, t_half_days=30.0, blood_vol_per_kg=0.07, background_ugdl=0.0)
    bll = compute_bll('onecomp', edi_mgkgd=edi, bw_kg=BW, params=params)
    assert bll >= 0.0
