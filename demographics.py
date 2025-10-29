# demographics.py

GROUP_INFO = {
    # k_wb units are explicitly declared to ensure accurate conversions in BLL calculations.
    # k_wb_units accepted values:
    #   - 'ugdl_per_mgkgd'  (µg/dL per mg/kg-day) [default]
    #   - 'ugdl_per_ugkgd'  (µg/dL per µg/kg-day)
    "Adults":   {"ED": 30*365, "AT_nc": 30*365, "AT_c": 70*365,
                 "BW": 70, "IR": 2.0,  "SA": 5700, "ET": 0.58, "AF": 0.50,
                 "k_wb": 0.5, "k_wb_units": "ugdl_per_mgkgd", "k_wb_gsd": None,
                 "bll_bg": 1.2, "bll_bg_sd": None},
    "Children": {"ED": 6*365,  "AT_nc": 6*365,  "AT_c": 70*365,
                 "BW": 15, "IR": 1.0,  "SA": 2800, "ET": 1.00, "AF": 0.90,
                 "k_wb": 1.2, "k_wb_units": "ugdl_per_mgkgd", "k_wb_gsd": None,
                 "bll_bg": 2.0, "bll_bg_sd": None},
    "Teens":    {"ED": 13*365, "AT_nc":13*365,  "AT_c": 70*365,
                 "BW": 50, "IR": 1.5,  "SA": 4500, "ET": 0.75, "AF": 0.70,
                 "k_wb": 0.8, "k_wb_units": "ugdl_per_mgkgd", "k_wb_gsd": None,
                 "bll_bg": 1.5, "bll_bg_sd": None},
    "Pregnant": {"ED": 1*365,  "AT_nc":1*365,   "AT_c": 70*365,
                 "BW": 60, "IR": 1.8,  "SA": 5000, "ET": 0.66, "AF": 0.80,
                 "k_wb": 0.6, "k_wb_units": "ugdl_per_mgkgd", "k_wb_gsd": None,
                 "bll_bg": 1.2, "bll_bg_sd": None},
}
