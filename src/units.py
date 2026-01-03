# units.py
"""
Unit conversion utilities for HBMPRA.

Provides conversion functions for:
- Concentration units (µg/L, mg/L)
- Nitrate basis conversion (NO₃ ⇔ NO₃–N)
"""

from typing import Optional

# =============================================================================
# Basic Constants
# =============================================================================
UG_PER_MG = 1000.0
DAYS_PER_YEAR = 365.0

# If chemistry CSV is in µg/L (typical), convert once to mg/L
CF_ugL_to_mgL = 1.0 / UG_PER_MG  # mg/L per µg/L
CF_mgL_to_ugL = UG_PER_MG        # µg/L per mg/L

# Pb TK: define k_g units explicitly as µg/dL per (mg/kg-day)
# Then BLL = b0 + k_g * EDI_Pb  (no extra factor)

# =============================================================================
# Nitrate Basis Conversion Constants
# =============================================================================
# Molecular weights (g/mol)
MW_NO3 = 62.0   # Nitrate ion (NO₃⁻)
MW_N = 14.0     # Nitrogen (N)

# Conversion factors
# C_NO3N = C_NO3 * (14/62)
# C_NO3  = C_NO3N * (62/14)
NO3_TO_NO3N = MW_N / MW_NO3       # 14/62 ≈ 0.2258
NO3N_TO_NO3 = MW_NO3 / MW_N       # 62/14 ≈ 4.4286


# =============================================================================
# Concentration Conversion Functions
# =============================================================================

def convert_mgL(value: float, from_unit: str, to_unit: str = "mg/L") -> float:
    """
    Convert concentration values between mg/L and µg/L.
    
    Parameters
    ----------
    value : float
        The concentration value to convert.
    from_unit : str
        The source unit: 'mg/L', 'µg/L', 'ug/L'
    to_unit : str
        The target unit: 'mg/L', 'µg/L', 'ug/L' (default: 'mg/L')
    
    Returns
    -------
    float
        The converted concentration value.
    
    Examples
    --------
    >>> convert_mgL(1.0, 'mg/L', 'µg/L')
    1000.0
    >>> convert_mgL(50.0, 'µg/L', 'mg/L')
    0.05
    """
    # Normalize unit strings
    from_unit = from_unit.lower().replace('μ', 'u')
    to_unit = to_unit.lower().replace('μ', 'u')
    
    if from_unit == to_unit:
        return value
    
    # Convert to mg/L first (canonical intermediate)
    if from_unit in ('ug/l', 'µg/l'):
        value_mgL = value * CF_ugL_to_mgL
    elif from_unit == 'mg/l':
        value_mgL = value
    else:
        raise ValueError(f"Unknown source unit: {from_unit}")
    
    # Convert from mg/L to target
    if to_unit in ('ug/l', 'µg/l'):
        return value_mgL * CF_mgL_to_ugL
    elif to_unit == 'mg/l':
        return value_mgL
    else:
        raise ValueError(f"Unknown target unit: {to_unit}")


def convert_nitrate_basis_mgL(value: float, 
                               from_basis: str, 
                               to_basis: str) -> float:
    """
    Convert nitrate concentration between NO₃ and NO₃–N basis.
    
    The conversion uses molecular weight ratios:
    - NO₃ to NO₃–N: multiply by (14/62) ≈ 0.2258
    - NO₃–N to NO₃: multiply by (62/14) ≈ 4.4286
    
    Parameters
    ----------
    value : float
        The concentration value to convert (mg/L).
    from_basis : str
        The source basis: 'NO3', 'NO3-N', 'NO3_N', 'nitrate', 'nitrate_N'
    to_basis : str
        The target basis: 'NO3', 'NO3-N', 'NO3_N', 'nitrate', 'nitrate_N'
    
    Returns
    -------
    float
        The converted concentration value (mg/L).
    
    Examples
    --------
    >>> convert_nitrate_basis_mgL(50.0, 'NO3', 'NO3-N')
    11.29...
    >>> convert_nitrate_basis_mgL(10.0, 'NO3-N', 'NO3')
    44.28...
    
    Notes
    -----
    WHO Guidelines:
    - 50 mg/L as NO₃ ≡ ~11.3 mg/L as NO₃–N
    - 10 mg/L as NO₃–N ≡ ~44.3 mg/L as NO₃
    
    ATSDR RfD:
    - 1.6 mg/kg-day as NO₃–N ≡ ~7 mg/kg-day as NO₃
    """
    # Normalize basis strings
    from_basis = _normalize_nitrate_basis(from_basis)
    to_basis = _normalize_nitrate_basis(to_basis)
    
    if from_basis == to_basis:
        return value
    
    if from_basis == 'NO3' and to_basis == 'NO3-N':
        return value * NO3_TO_NO3N
    elif from_basis == 'NO3-N' and to_basis == 'NO3':
        return value * NO3N_TO_NO3
    else:
        raise ValueError(f"Cannot convert from {from_basis} to {to_basis}")


def _normalize_nitrate_basis(basis: str) -> str:
    """
    Normalize nitrate basis string to canonical form.
    
    Parameters
    ----------
    basis : str
        Input basis string.
    
    Returns
    -------
    str
        Normalized basis: 'NO3' or 'NO3-N'
    """
    if basis is None:
        return 'NO3'  # Default to NO3 if not specified
    
    basis_lower = str(basis).lower().strip()
    
    # NO3-N patterns
    if any(p in basis_lower for p in ['no3-n', 'no3_n', 'nitrate_n', 'nitrate-n', 'n-n']):
        return 'NO3-N'
    
    # Plain NO3 patterns
    if any(p in basis_lower for p in ['no3', 'nitrate']):
        return 'NO3'
    
    # Return as-is if not recognized (will fail downstream)
    return basis


def detect_nitrate_basis_from_column(column_name: str) -> str:
    """
    Autodetect nitrate basis from column name.
    
    Parameters
    ----------
    column_name : str
        The column name from a CSV/dataframe.
    
    Returns
    -------
    str
        Detected basis: 'NO3' or 'NO3-N'
    
    Examples
    --------
    >>> detect_nitrate_basis_from_column('NO3_N')
    'NO3-N'
    >>> detect_nitrate_basis_from_column('NO3')
    'NO3'
    >>> detect_nitrate_basis_from_column('nitrate_N')
    'NO3-N'
    """
    col_lower = str(column_name).lower().strip()
    
    # Patterns indicating NO3-N basis
    no3n_patterns = ['_n', 'no3_n', 'nitrate_n', 'no3-n', 'nitrate-n']
    
    for pattern in no3n_patterns:
        if pattern in col_lower:
            return 'NO3-N'
    
    # Default to NO3 basis
    return 'NO3'
