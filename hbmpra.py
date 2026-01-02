"""Module shim that exposes key HBMPRA helpers for tests and CLI use."""

from __future__ import annotations

from src import hbmpra_optimized as _optimized

pm = _optimized.pm
az = _optimized.az
dill = _optimized.dill
__all__ = [
    "pm",
    "az",
    "dill",
    "main",
    "impute_censored_df",
    "build_organ_sets",
    "dermal_dad_water",
]

main = _optimized.main
impute_censored_df = _optimized.impute_censored_df
build_organ_sets = _optimized.build_organ_sets


def dermal_dad_water(
    Cw_mg_L,
    Kp_cm_per_hr,
    SA_cm2,
    t_event_hr,
    EV_per_day,
    EF_days_year,
    ED_years,
    BW_kg,
    AT_days,
) -> float:
    """Compute dermal absorbed dose (mg/kg-day) following RAGS Part E."""

    try:
        Cw_mg_L = float(Cw_mg_L)
        Kp_cm_per_hr = float(Kp_cm_per_hr)
        SA_cm2 = float(SA_cm2)
        t_event_hr = float(t_event_hr)
        EV_per_day = float(EV_per_day)
        EF_days_year = float(EF_days_year)
        ED_years = float(ED_years)
        BW_kg = float(BW_kg)
        AT_days = float(AT_days)
    except (TypeError, ValueError) as exc:
        raise ValueError("All dermal DAD inputs must be numeric") from exc

    if BW_kg <= 0 or AT_days <= 0:
        raise ValueError("Body weight (BW) and averaging time (AT) must be positive")

    Cw_mg_cm3 = Cw_mg_L * 1e-3
    DA_event_mg_cm2 = Kp_cm_per_hr * Cw_mg_cm3 * t_event_hr
    numerator = DA_event_mg_cm2 * SA_cm2 * EV_per_day * EF_days_year * ED_years
    denominator = BW_kg * AT_days
    return numerator / denominator