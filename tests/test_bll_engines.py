from bll_engines import OneCompParams, edi_from_conc_ugL, bll_onecomp_from_water, bll_linear_from_intake


def test_units_monotonic():
    p = OneCompParams(f_abs=0.25, t_half_days=30, blood_vol_per_kg=0.07, background_ugdl=0.0)
    b0 = bll_onecomp_from_water(0, 2.0, 70.0, p)
    b1 = bll_onecomp_from_water(1, 2.0, 70.0, p)
    b2 = bll_onecomp_from_water(10, 2.0, 70.0, p)
    assert b2 > b1 > b0 >= 0.0


def test_slope_engine_positive():
    edi = edi_from_conc_ugL(5.0, 0.6, 20.0)
    b = bll_linear_from_intake(edi, 20.0, slope_ugdl_per_ugday=0.17, f_abs=0.5, background_ugdl=0.0)
    assert b > 0.0
