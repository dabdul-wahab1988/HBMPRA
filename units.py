# units.py
UG_PER_MG = 1000.0
DAYS_PER_YEAR = 365.0

# If chemistry CSV is in µg/L (typical), convert once to mg/L
CF_ugL_to_mgL = 1.0 / UG_PER_MG  # mg/L per µg/L

# Pb TK: define k_g units explicitly as µg/dL per (mg/kg-day)
# Then BLL = b0 + k_g * EDI_Pb  (no extra factor)
