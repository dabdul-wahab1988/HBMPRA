import hbmpra


def test_dermal_hq_numeric():
    # Parameters for a simple hand-calculated case
    Cw_mg_L = 0.05  # 50 ug/L -> 0.05 mg/L
    Kp = 1e-3       # cm/hr
    SA = 1000.0     # cm^2
    t_event_hr = 1.0
    EV_per_day = 1.0
    EF_days_year = 365.0
    ED_years = 1.0
    BW_kg = 70.0
    AT_days = 365.0

    # Arbitrary absorbed RfD for HQ calculation (mg/kg-day)
    rfd_abs = 0.0003

    # Compute using the library function
    dad = hbmpra.dermal_dad_water(Cw_mg_L, Kp, SA, t_event_hr, EV_per_day,
                                   EF_days_year, ED_years, BW_kg, AT_days)

    # Manual calculation following RAGS Part E as implemented in the function
    Cw_mg_cm3 = Cw_mg_L * 1e-3
    DA_event_mg_cm2 = Kp * Cw_mg_cm3 * t_event_hr
    expected_dad = DA_event_mg_cm2 * SA * EV_per_day * EF_days_year * ED_years / (BW_kg * AT_days)

    assert abs(dad - expected_dad) < 1e-12

    # HQ
    HQ = dad / rfd_abs
    expected_HQ = expected_dad / rfd_abs
    assert abs(HQ - expected_HQ) < 1e-12
