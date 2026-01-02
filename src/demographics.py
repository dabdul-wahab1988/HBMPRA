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

# =============================================================================
# Demographic Group Presets
# =============================================================================

GROUP_PRESETS = {
    'all': ['Adults', 'Children', 'Teens', 'Pregnant'],
    'sensitive': ['Children', 'Pregnant'],       # Most vulnerable populations
    'adults_only': ['Adults'],                   # Adults only
    'children_only': ['Children'],               # Children only (pediatric focus)
    'non_sensitive': ['Adults', 'Teens'],        # Non-sensitive adults
}


def get_available_groups():
    """Return list of all available demographic group names."""
    return list(GROUP_INFO.keys())


def get_group_info_filtered(groups: list) -> dict:
    """
    Return a filtered GROUP_INFO dictionary containing only the specified groups.
    
    Parameters
    ----------
    groups : list of str
        List of group names to include
    
    Returns
    -------
    dict
        Filtered GROUP_INFO dictionary
    """
    return {g: GROUP_INFO[g] for g in groups if g in GROUP_INFO}


def parse_group_selection(selection: str) -> list:
    """
    Parse a group selection string and return list of group names.
    
    Parameters
    ----------
    selection : str
        Either a preset name ('all', 'sensitive', etc.) or comma-separated group names
    
    Returns
    -------
    list of str
        List of validated group names
    
    Raises
    ------
    ValueError
        If unknown group name or preset is specified
    
    Examples
    --------
    >>> parse_group_selection('all')
    ['Adults', 'Children', 'Teens', 'Pregnant']
    >>> parse_group_selection('sensitive')
    ['Children', 'Pregnant']
    >>> parse_group_selection('Children,Adults')
    ['Children', 'Adults']
    """
    selection = selection.strip()
    
    # Check if it's a preset
    if selection.lower() in GROUP_PRESETS:
        return GROUP_PRESETS[selection.lower()]
    
    # Try comma-separated list
    groups = [g.strip() for g in selection.split(',')]
    
    # Also try parsing as numbers (1,2,3 style)
    all_groups = list(GROUP_INFO.keys())
    if all(g.isdigit() for g in groups):
        try:
            groups = [all_groups[int(g) - 1] for g in groups]
        except (IndexError, ValueError):
            raise ValueError(f"Invalid group number. Valid range: 1-{len(all_groups)}")
    
    # Validate all groups exist
    invalid = [g for g in groups if g not in GROUP_INFO]
    if invalid:
        valid_list = ', '.join(GROUP_INFO.keys())
        preset_list = ', '.join(GROUP_PRESETS.keys())
        raise ValueError(
            f"Unknown group(s): {', '.join(invalid)}\n"
            f"Valid groups: {valid_list}\n"
            f"Valid presets: {preset_list}"
        )
    
    return groups


def print_group_selection_help():
    """Print help for demographic group selection."""
    print("\n" + "="*60)
    print("  DEMOGRAPHIC GROUP SELECTION")
    print("="*60)
    print("\n  Available groups:")
    for i, (name, info) in enumerate(GROUP_INFO.items(), 1):
        bw = info['BW']
        ir = info['IR']
        ed_years = info['ED'] / 365
        print(f"    {i}. {name:10} (BW={bw}kg, IR={ir}L/d, {ed_years:.0f}y exposure)")
    
    print("\n  Presets:")
    for preset, groups in GROUP_PRESETS.items():
        groups_str = ', '.join(groups)
        print(f"    {preset:12} → {groups_str}")
    
    print("\n  Enter: numbers (1,2), names (Adults,Children), or preset")
    print("="*60)

