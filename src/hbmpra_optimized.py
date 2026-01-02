#!/usr/bin/env python3
"""
hbmpra_optimized.py

Unified HBMPRA per guardrails:
• Single PyMC model produces organ HIs, total CR, and predictive Pb BLL.
• Ingestion + dermal-water only. No inhalation.
• Organ sets must come from external/toxref.yml (unless --allow-default-organ-sets).
• PHREEQC bio or fraction tables are used when available.
• Censoring: either impute once (MLE + truncated-lognormal expectation) or model in-graph.

Major optimizations vs original:
• Vectorized dermal HQ across (metal × site × group) instead of nested Python loops.
• Removed duplicated helpers and in-function redefinitions.
• Clear separation of “impute outside” vs “model censoring inside PyMC”.
"""

import os
import json
import argparse

def _base_key(name: str) -> str:
    """Return alphabetic base for a species (e.g., 'CrVI'->'CrVI', 'Hg(II)'->'Hg')."""
    if not name:
        return ""
    s = str(name)
    return "".join([ch for ch in s if ch.isalnum()])

def _normalize_organ_sets_to_metals(organ_sets_used, metals):
    """
    Ensure each organ's set only contains metals present in the metals list, using base key matching.
    """
    metals_base = {_base_key(m): m for m in metals}
    normalized = {}
    for organ, metal_set in organ_sets_used.items():
        filtered = set()
        for m in metal_set:
            base = _base_key(m)
            if base in metals_base:
                filtered.add(metals_base[base])
        normalized[organ] = filtered
    return normalized
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Mapping

import numpy as np
import pandas as pd

try:
    import pymc as pm
    import pytensor.tensor as at
    import arviz as az
    import dill
except Exception:
    pm = None
    at = None
    az = None
    dill = None

try:
    from scipy import optimize, stats as sps
except Exception:
    optimize = None
    sps = None

import yaml

from demographics import GROUP_INFO, GROUP_PRESETS, parse_group_selection, get_group_info_filtered
from units import (CF_ugL_to_mgL, DAYS_PER_YEAR, 
                   convert_nitrate_basis_mgL, detect_nitrate_basis_from_column)

# Get the directory where this script lives (src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Project root is parent of src/
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# External data directory
EXTERNAL_DIR = os.path.join(PROJECT_ROOT, "external")

logging.getLogger("pytensor.tensor.blas").setLevel(logging.ERROR)

# Built-in fallback organ sets (only used if user allows explicitly)
ORGAN_SETS_FALLBACK = {
    "neuro":   {"Mn"},
    "nephro":  {"Cd", "Hg", "CrVI"},
    "hepato":  {"As", "Cd", "Cu", "CrVI"},
    # "systemic" may be supplied via toxref organ_sets; if absent, remains empty.
}
CARCINOGENS = {"As", "CrVI"}

# =============================================================================
# Anion support: Fluoride and Nitrate  
# =============================================================================
# Known anion synonyms for column detection
ANION_SYNONYMS = {
    "fluoride": {"F", "F-", "fluoride", "Fluoride", "Fluoride_mgL", "F_mgL"},
    "nitrate": {"NO3", "NO3-", "nitrate", "Nitrate", "NO3_N", "nitrate_N", "NO3-N", "Nitrate_N", "NO3_mgL"},
    "nitrite": {"NO2", "NO2-", "nitrite", "Nitrite", "NO2_N", "nitrite_N", "NO2-N", "Nitrite_N", "NO2_mgL"},
}
# Canonical names for anions in toxref lookup
ANION_TOX_KEYS = {"fluoride", "nitrate", "nitrite"}
# Mixture group classification
MIXTURE_GROUPS = {
    "metals": {"As","Cd","Co","Cr","Cu","Fe","Hg","Mn","Ni","Pb","Zn","Al","V",
               "CrVI","CrIII","Hg(II)","MeHg",
               "Sb","Ba","Be","Mo","Se","Ag","Tl","U"},  # Added new metals
    "anions": {"fluoride", "nitrate", "nitrite"},
}

# Atomic weights (g/mol) to convert PHREEQC molarity -> µg/L
ATOMIC_WEIGHTS = {
    "As": 74.92, "Cd": 112.41, "Cr": 52.00, "CrIII": 52.00, "CrVI": 52.00,
    "Cu": 63.55,
    "Hg": 200.59, "MeHg": 215.59,  # MeHg ~= CH3Hg+ molar mass ~215.59 g/mol
    # PHREEQC species aliases
    "HgII": 200.59, "Hg(II)": 200.59, "Hg0": 200.59, "Hg(0)": 200.59, "HgI": 200.59, "Hg(I)": 200.59,
    "CrVI": 52.00, "Cr(6)": 52.00, "Cr(VI)": 52.00, "CrIII": 52.00, "Cr(3)": 52.00,
    "Pb": 207.2, "Co": 58.93, "Fe": 55.85,
    "Mn": 54.94, "Ni": 58.69, "Zn": 65.38,
    "Al": 26.98, "V": 50.94,
}

# ----------------- Key resolution helpers --------------------------------- #
def _base_key(name: str) -> str:
    """Return alphabetic base for a species (e.g., 'CrVI'->'CrVI', 'Hg(II)'->'Hg')."""
    if not name:
        return ""
    s = str(name)
    # prefer letters and digits and remove parentheses
    return "".join([ch for ch in s if ch.isalnum()])

def resolve_key_in_dict(key: str, d: Mapping[str, object]) -> Optional[str]:
    """Try to find a matching key inside dict d for requested key.

    Resolution order: exact -> base alphabetic match -> case-insensitive match -> None
    Returns the matching dict key (as present in d) or None if not found.
    """
    if key in d:
        return key
    base = _base_key(key)
    # try base exact
    if base in d:
        return base
    # case-insensitive search
    low = key.lower()
    for k in d.keys():
        if str(k).lower() == low:
            return k
    # try alphabetic-only match against keys
    for k in d.keys():
        if _base_key(str(k)) == base:
            return k
    return None


def validate_dermal_requirements(metals_list, C_bio_matrix, RFDS_derm_array):
    """Module-level publication-mode validator.

    Raises RuntimeError if any metal with non-zero bioavailable concentrations
    lacks an explicit dermal RfD (RfD_derm). Intended to be used as a fast
    preflight check before model construction.
    """
    has_dermal = (np.nan_to_num(C_bio_matrix, nan=0.0) > 0).any(axis=0)
    missing = [m for i, m in enumerate(metals_list)
               if has_dermal[i] and not np.isfinite(RFDS_derm_array[i])]
    if missing:
        raise RuntimeError(
            "Publication mode: missing RfD_derm for dermally exposed metals: "
            + ", ".join(missing)
            + ". Add explicit mg/kg-day values to external/toxref.yml."
        )

# ------------------------------- Helpers ------------------------------------ #

def parse_metals(df: pd.DataFrame) -> List[str]:
    """Discover metals from headers like 'C_As', 'As', or 'As(µg/L)'. """
    known = {"As","Cd","Co","Cr","Cu","Fe","Hg","Mn","Ni","Pb","Zn","Al","V"}
    metals = []
    for c in df.columns:
        cstr = str(c).strip()
        if cstr.startswith("C_"):
            m = cstr.split("_", 1)[1]
            if m: metals.append(m)
            continue
        if cstr in known:
            metals.append(cstr)
            continue
        if "(" in cstr and "ug/L" in cstr.replace("µ","u"):
            tok = cstr.split("(")[0].strip()
            if tok and tok.isalpha() and len(tok) <= 5:
                metals.append(tok)
    return sorted(list(dict.fromkeys(metals)))


def parse_anions(df: pd.DataFrame) -> Tuple[List[str], Dict[str, str]]:
    """
    Discover anions (fluoride, nitrate) from dataframe column headers.
    
    Returns
    -------
    anions : List[str]
        List of canonical anion names found (e.g., ['fluoride', 'nitrate'])
    anion_columns : Dict[str, str]
        Mapping from canonical name to original column name
    """
    # Known metal symbols to exclude from anion detection
    METAL_SYMBOLS = {"Fe", "FE", "Mn", "MN", "Ni", "NI", "Pb", "PB", "Zn", "ZN", 
                     "Cu", "CU", "Cr", "CR", "Co", "CO", "Cd", "CD", "Hg", "HG", 
                     "As", "AS", "Al", "AL", "V"}
    
    anions = []
    anion_columns = {}
    
    for c in df.columns:
        cstr = str(c).strip()
        
        # Skip if this is a known metal symbol
        if cstr in METAL_SYMBOLS or cstr.upper() in METAL_SYMBOLS:
            continue
        
        c_upper = cstr.upper().replace("-", "").replace("_", "")
        
        # Check for fluoride - use exact matching only
        for syn in ANION_SYNONYMS.get("fluoride", set()):
            syn_norm = syn.upper().replace("-", "").replace("_", "")
            # Exact match only (case insensitive for normalized forms)
            if cstr == syn or c_upper == syn_norm:
                if "fluoride" not in anions:
                    anions.append("fluoride")
                    anion_columns["fluoride"] = cstr
                break
        
        # Check for nitrate - use exact matching only
        for syn in ANION_SYNONYMS.get("nitrate", set()):
            syn_norm = syn.upper().replace("-", "").replace("_", "")
            # Exact match only (case insensitive for normalized forms)
            if cstr == syn or c_upper == syn_norm:
                if "nitrate" not in anions:
                    anions.append("nitrate")
                    anion_columns["nitrate"] = cstr
                break
    
    return anions, anion_columns


def get_analyte_mixture_group(analyte: str) -> str:
    """
    Return the mixture group ('metals', 'anions') for an analyte.
    """
    for group, members in MIXTURE_GROUPS.items():
        if analyte in members or analyte.lower() in [m.lower() for m in members]:
            return group
    return "metals"  # Default to metals for unknown

def build_organ_sets(metals: List[str],
                     toxref: Dict,
                     base_sets: Dict[str, set]) -> Tuple[Dict[str, set], np.ndarray, set, Dict[str, str]]:
    """Merge toxref organ entries onto base sets and build systemic mask.

    Extended behavior: support route-annotated target_organs entries. Each
    element in toxref[metal]["target_organs"] may be either a plain string
    (organ name) or a mapping like { organ_name: route } where route is
    'dermal' or 'ingestion'. We return an `organ_routes` mapping that maps
    lower-cased organ names to one of 'dermal'|'ingestion'|'both'. If an
    organ has conflicting route annotations, the mapping will be 'both'.
    """
    organ_sets = {k.lower(): set(v) for k, v in base_sets.items()}
    canonical = set(organ_sets.keys()) | {"systemic"}
    unknown = set()
    organ_routes: Dict[str, str] = {}

    def _norm_route(r: str) -> str:
        if not r:
            return "both"
        r0 = str(r).strip().lower()
        if r0 in ("derm", "dermal", "dermal_water", "skin"):
            return "dermal"
        if r0 in ("ingest", "ingestion", "oral", "gi"):
            return "ingestion"
        if r0 in ("both", "all", "any"):
            return "both"
        # For groundwater-focused runs, some routes are invalid.
        if r0 in ("respiratory", "inhalation", "lung"):
            raise ValueError(f"Route '{r0}' not applicable for groundwater exposure")
        # Conservative fallback for unknown routes: treat as 'both'
        return "both"

    if toxref:
        for metal_key, meta in toxref.items():
            torgs = meta.get("target_organs", []) if meta else []
            for organ in torgs:
                if not organ:
                    continue
                # Support mapping entries in the YAML where each element in the list
                # can be a single-key dict like {'hepato': 'ingestion'} or a dict with
                # multiple pairs. Also support plain string organ names.
                if isinstance(organ, dict):
                    # if single-key dict, normalize by extracting the one pair
                    if len(organ) == 1:
                        ok_raw, route_raw = next(iter(organ.items()))
                        ok = str(ok_raw).strip().lower()
                        route = _norm_route(route_raw)
                        organ_sets.setdefault(ok, set()).add(metal_key)
                        prev = organ_routes.get(ok)
                        if prev and prev != route:
                            organ_routes[ok] = "both"
                        else:
                            organ_routes[ok] = route
                        if ok not in canonical:
                            unknown.add(ok)
                    else:
                        # multi-key dict: iterate pairs
                        for ok_raw, route_raw in organ.items():
                            ok = str(ok_raw).strip().lower()
                            route = _norm_route(route_raw)
                            organ_sets.setdefault(ok, set()).add(metal_key)
                            prev = organ_routes.get(ok)
                            if prev and prev != route:
                                organ_routes[ok] = "both"
                            else:
                                organ_routes[ok] = route
                            if ok not in canonical:
                                unknown.add(ok)
                else:
                    ok = str(organ).strip().lower()
                    organ_sets.setdefault(ok, set()).add(metal_key)
                    # Do not set or override organ_routes here; explicit dict entries
                    # are required to declare route preference. Record unknown names.
                    if ok not in canonical:
                        unknown.add(ok)

    systemic_mask = np.array([1.0 if m in organ_sets.get("systemic", set()) else 0.0
                              for m in metals], dtype=float)
    return organ_sets, systemic_mask, unknown, organ_routes

def pick_total_column(df: pd.DataFrame, metal: str) -> np.ndarray:
    """Return total concentration series in µg/L for a metal name."""
    for cand in [f"C_{metal}", metal]:
        if cand in df.columns:
            return pd.to_numeric(df[cand], errors="coerce").fillna(0.0).to_numpy(float)
    # Pattern like 'As( µg/L)' or 'As (ug/L)'
    for c in df.columns:
        cstr = str(c)
        if cstr.startswith(metal) and "ug/L" in cstr.replace("µ","u"):
            return pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(float)
    return np.zeros(len(df), dtype=float)

def pick_bio_column(df: pd.DataFrame, metal: str) -> np.ndarray:
    """Return bioavailable concentration in µg/L if present, else zeros."""
    col = f"C_bio_{metal}"
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(float)
    return np.zeros(len(df), dtype=float)

def load_phreeqc_tables(results_dir: str, expected_rows: int = None):
    """Find PHREEQC bio or fraction tables, prefer results_dir then external/.
    
    If expected_rows is provided, validates that loaded tables match the row count.
    Returns None for tables that fail validation.
    """
    bio_path = os.path.join(results_dir, "table_bioavailable_concentrations.csv")
    frac_path = os.path.join(results_dir, "table_species_fractions.csv")
    if not os.path.exists(bio_path):
        alt = os.path.join(EXTERNAL_DIR, "table_bioavailable_concentrations.csv")
        if os.path.exists(alt): bio_path = alt
    if not os.path.exists(frac_path):
        alt = os.path.join(EXTERNAL_DIR, "table_species_fractions.csv")
        if os.path.exists(alt): frac_path = alt

    bio_df = None
    frac_df = None
    source = "none"
    
    if os.path.exists(bio_path):
        try:
            temp_df = pd.read_csv(bio_path)
            if expected_rows is not None and len(temp_df) != expected_rows:
                logging.warning(f"PHREEQC bio table has {len(temp_df)} rows, expected {expected_rows}. Ignoring file: {bio_path}")
                bio_df = None
            else:
                bio_df = temp_df
                source = "phreeqc_bio"
        except Exception as e:
            logging.warning(f"Failed to read PHREEQC bio table {bio_path}: {e}")
            bio_df = None
            
    if bio_df is None and os.path.exists(frac_path):
        try:
            temp_df = pd.read_csv(frac_path)
            if expected_rows is not None and len(temp_df) != expected_rows:
                logging.warning(f"PHREEQC fraction table has {len(temp_df)} rows, expected {expected_rows}. Ignoring file: {frac_path}")
                frac_df = None
            else:
                frac_df = temp_df
                source = "phreeqc_frac"
        except Exception as e:
            logging.warning(f"Failed to read PHREEQC fraction table {frac_path}: {e}")
            frac_df = None
            
    return source, bio_path if bio_df is not None else None, frac_path if frac_df is not None else None, bio_df, frac_df

# -------- Censored-data MLE + conditional expectation (vector-friendly) ------ #

def fit_left_censored_lognormal(observed: np.ndarray) -> Tuple[float, float]:
    """Fit LogNormal on detected values only. Returns (mu, sigma) on log scale."""
    y = np.log(np.asarray(observed, float) + 1e-12)
    if y.size < 2:
        mu0 = float(np.mean(y)) if y.size else -20.0
        sigma0 = float(np.std(y, ddof=1)) if y.size > 1 else 1.0
        return mu0, max(sigma0, 1e-6)

    mu0 = float(np.mean(y))
    sigma0 = float(np.std(y, ddof=1))
    if optimize is None:  # fallback
        return mu0, max(sigma0, 1e-6)

    def nll(params):
        mu, log_sig = params
        sig = np.exp(log_sig)
        ll = -0.5*np.sum(((y - mu)/sig)**2) - y.size*np.log(sig) - 0.5*y.size*np.log(2*np.pi)
        return -ll

    try:
        res = optimize.minimize(nll, x0=np.array([mu0, np.log(sigma0 + 1e-9)]), method="L-BFGS-B")
        if res.success:
            return float(res.x[0]), float(np.exp(res.x[1]))
    except Exception:
        pass
    return mu0, max(sigma0, 1e-6)

def truncated_lognormal_mean_below_lod(mu: float, sigma: float, lod: float) -> float:
    """E[X | X < L] for X~LogNormal(mu, sigma)."""
    if not np.isfinite(lod) or lod <= 0:
        return float(np.exp(mu - 0.5*sigma**2))
    if sps is None:
        return float(0.5*np.exp(mu + 0.5*sigma**2))
    t = np.log(lod + 1e-12)
    num = np.exp(mu + 0.5*sigma**2) * sps.norm.cdf((t - mu - sigma**2)/sigma)
    den = sps.norm.cdf((t - mu)/sigma)
    if den <= 1e-12:
        return float(0.5*np.exp(mu + 0.5*sigma**2))
    return float(num/den)

def impute_censored_df(df: pd.DataFrame, metals: List[str], seed: int = 42) -> pd.DataFrame:
    """Impute non-detects per metal as E[X|X<LOD] under fitted lognormal; fallback to LOD/2."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for m in metals:
        col, lodcol = f"C_{m}", f"LOD_{m}"
        if col not in out.columns or lodcol not in out.columns:
            continue
        vals = pd.to_numeric(out[col], errors="coerce")
        lods = pd.to_numeric(out[lodcol], errors="coerce").fillna(0.0).to_numpy(float)
        mask_nd = (~vals.notna()) | (vals <= 0)
        detected = vals.loc[~mask_nd].to_numpy(float)
        try:
            mu, sig = fit_left_censored_lognormal(detected)
            fill = np.array([truncated_lognormal_mean_below_lod(mu, sig, lods[i])
                             for i, nd in enumerate(mask_nd.to_numpy()) if nd], dtype=float)
            vals.loc[mask_nd] = fill if fill.size == mask_nd.sum() else lods[mask_nd.to_numpy()]/2.0
        except Exception:
            vals.loc[mask_nd] = lods[mask_nd.to_numpy()]/2.0
        out[col] = vals.astype(float)
    return out

# ------------------------------- Main --------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chemistry", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--tune", type=int, default=2000)

    p.add_argument("--bll-thresholds", default="3.5,5")
    p.add_argument("--input-units", choices=['ug/L', 'mg/L'], default='ug/L',
                   help="Concentration units in input file (default: ug/L). If mg/L, will convert to ug/L.")
    p.add_argument("--use-bioavailable", action="store_true",
                   help="Use C_bio_<metal> when provided in chemistry CSV.")
    p.add_argument("--allow-disable-dermal-if-no-bio", action="store_true",
                   help=("Allow the script to continue with the dermal_water route disabled "
                         "when no bioavailable concentrations are present. By default the script "
                         "will raise an error to force explicit handling (recommended for scientific studies)."))
    # Censoring is handled by external imputation for stability; in-graph censoring removed.
    p.add_argument("--allow-default-organ-sets", action="store_true",
                   help="Permit fallback organ sets if toxref organ_sets missing.")
    # Override now enabled by default; provide a flag to disable it explicitly.
    p.add_argument("--disable-dermal-override", action="store_true",
             help=("Disable the default behavior of including dermal contributions for organs "
                   "with dermal bioavailable concentrations. By default the override is ON; "
                   "pass this flag to restore strict toxref organ_routes behavior."))
    p.add_argument("--groups", default="all",
                   help=("Demographic groups to analyze. Can be: preset name (all, sensitive, "
                         "adults_only, children_only, non_sensitive), comma-separated group names "
                         "(Adults,Children), or numbers (1,2). Default: all"))
    p.add_argument("--save-prior-pred", action="store_true")

    args = p.parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    # Load chemistry
    raw = pd.read_csv(args.chemistry)
    
    # Extract site names from first column (if it's a site identifier)
    site_id_cols = ['site', 'Site', 'ID', 'id', 'Community', 'Location', 'Sample', 'Name']
    site_names = None
    for col in site_id_cols:
        if col in raw.columns:
            site_names = raw[col].astype(str).tolist()
            break
    if site_names is None:
        # Use first column as site name if it looks like an identifier
        first_col = raw.columns[0]
        if raw[first_col].dtype == object or not raw[first_col].apply(lambda x: isinstance(x, (int, float))).all():
            site_names = raw[first_col].astype(str).tolist()
        else:
            # Fallback to numeric indices
            site_names = [f"Site_{i}" for i in range(len(raw))]
    
    # Handle unit conversion: if input is mg/L, convert to µg/L
    if args.input_units == 'mg/L':
        print(f"Converting concentrations from mg/L to µg/L (multiply by 1000)")
        # Find metal concentration columns and convert them
        known_metals = {"As","Cd","Co","Cr","Cu","Fe","Hg","Mn","Ni","Pb","Zn","Al","V"}
        for col in raw.columns:
            if col.startswith('C_') or col in known_metals:
                raw[col] = pd.to_numeric(raw[col], errors='coerce').fillna(0.0) * 1000.0
    
    metals = parse_metals(raw)
    if not metals:
        logging.warning("No metal columns found. This run may use only anions (F/NO3).")
    
    # ==========================================================================
    # Anion detection: Fluoride and Nitrate
    # ==========================================================================
    anions, anion_columns = parse_anions(raw)
    anion_metadata = {}  # Store basis info for nitrate
    
    if anions:
        print(f"Detected anions: {anions}")
        for anion in anions:
            if anion == "nitrate":
                col_name = anion_columns.get("nitrate", "")
                detected_basis = detect_nitrate_basis_from_column(col_name)
                anion_metadata["nitrate"] = {
                    "column": col_name,
                    "detected_basis": detected_basis,
                    "target_basis": "NO3-N"  # RfD is in NO3-N
                }
                print(f"  Nitrate column '{col_name}' detected as {detected_basis} basis")
            elif anion == "fluoride":
                col_name = anion_columns.get("fluoride", "")
                anion_metadata["fluoride"] = {
                    "column": col_name,
                    "detected_basis": "F",
                    "target_basis": "F"
                }
                print(f"  Fluoride column '{col_name}' detected")
    
    # Combined analyte list: metals + anions
    # Note: anions are kept separate for unit handling but included in HI computation
    all_analytes = metals + anions
    if not all_analytes:
        raise RuntimeError("No analyte columns found. Expect metals like 'C_As' or anions like 'F', 'NO3'.")

    J = len(raw)

    # Censoring strategy: always impute non-detects outside the model for robust initialization
    df = impute_censored_df(raw, metals, seed=args.seed)

    # PHREEQC integration
    spec_src, bio_path, frac_path, bio_df, frac_df = load_phreeqc_tables(args.results_dir, expected_rows=J)

    if bio_df is not None:
        # Columns like C_bio_CrVI (mol/L). Convert to µg/L using atomic weights.
        spcols = [c for c in bio_df.columns if c.startswith("C_bio_")]
        species = [c[len("C_bio_"):] for c in spcols]
        for sp in species:
            if sp not in metals:
                metals.append(sp)
        metals = sorted(list(dict.fromkeys(metals)))
        M = len(metals)
        midx = {m:i for i,m in enumerate(metals)}

        C_ugL = np.zeros((J, M), float)
        C_bio_ugL = np.zeros((J, M), float)

        # Fill totals from df where present
        for m in metals:
            C_ugL[:, midx[m]] = pick_total_column(df, m)

        # Convert species mol/L -> µg/L
        for c, sp in zip(spcols, species):
            mol = pd.to_numeric(bio_df[c], errors="coerce").fillna(0.0).to_numpy(float)
            if mol.size != J:
                raise RuntimeError(f"PHREEQC bio table '{c}' length ({mol.size}) does not match chemistry rows ({J}). Align tables by site or provide matching rows.")
            # Resolve atomic weight: try full species, then base key
            aw_key = resolve_key_in_dict(sp, ATOMIC_WEIGHTS) or resolve_key_in_dict(_base_key(sp), ATOMIC_WEIGHTS)
            if aw_key is None:
                raise RuntimeError(f"Missing atomic weight entry for species '{sp}' or base '{_base_key(sp)}' in ATOMIC_WEIGHTS.")
            aw = ATOMIC_WEIGHTS[aw_key]
            C_bio_ugL[:, midx[sp]] = mol * aw * 1e6

        # If species (e.g., CrVI) are present, zero-out the *base metal* total
        # columns (e.g., 'Cr') to avoid double-counting totals split across
        # species. Use alphabetic-only base (so 'CrVI' -> 'Cr') and only clear
        # the total concentration column (C_ugL). Do NOT zero the species
        # bioavailable columns which we just populated from PHREEQC.
        # IMPORTANT: Only zero bases that are DIFFERENT from their species!
        # E.g., zero 'Cr' when 'CrVI'/'CrIII' exist, but NOT 'Pb' when 'Pb' exists.
        bases_to_zero = set()
        for sp in species:
            base = "".join([ch for ch in sp if ch.isalpha()])
            # Only add to zero list if base differs from the species itself
            if base and base != sp and base in midx:
                bases_to_zero.add(base)
        for base in bases_to_zero:
            C_ugL[:, midx[base]] = 0.0

        C_mgL = C_ugL * CF_ugL_to_mgL
        C_bio_mgL = C_bio_ugL * CF_ugL_to_mgL

    elif frac_df is not None:
        # Columns like frac_CrVI. Split totals by fraction.
        fcols = [c for c in frac_df.columns if c.startswith("frac_")]
        species = [c[len("frac_"):] for c in fcols]
        for sp in species:
            if sp not in metals:
                metals.append(sp)
        metals = sorted(list(dict.fromkeys(metals)))
        M = len(metals)
        midx = {m:i for i,m in enumerate(metals)}

        C_ugL = np.zeros((J, M), float)
        C_bio_ugL = np.zeros((J, M), float)

        for m in metals:
            C_ugL[:, midx[m]] = pick_total_column(df, m)

        for c, sp in zip(fcols, species):
            base = "".join([ch for ch in sp if ch.isalpha()])
            tot = pick_total_column(df, base)
            frac = pd.to_numeric(frac_df[c], errors="coerce").fillna(0.0).to_numpy(float)
            if frac.size != J:
                raise RuntimeError(f"PHREEQC fraction table '{c}' length ({frac.size}) does not match chemistry rows ({J}). Align tables by site or provide matching rows.")
            C_ugL[:, midx[sp]] = tot * frac

        # If species fractions exist for a base metal, zero-out the base totals to avoid double-counting.
        # IMPORTANT: Only zero bases that are DIFFERENT from their species!
        bases_to_zero = set()
        for sp in species:
            base = "".join([ch for ch in sp if ch.isalpha()])
            if base and base != sp and base in midx:
                bases_to_zero.add(base)
        for base in bases_to_zero:
            C_ugL[:, midx[base]] = 0.0
            C_bio_ugL[:, midx[base]] = 0.0

        C_mgL = C_ugL * CF_ugL_to_mgL
        C_bio_mgL = C_bio_ugL * CF_ugL_to_mgL

    else:
        # No PHREEQC: use provided totals (and optional C_bio_*)
        metals = sorted(list(dict.fromkeys(metals)))
        M = len(metals)
        midx = {m:i for i,m in enumerate(metals)}
        C_ugL = np.column_stack([pick_total_column(df, m) for m in metals]).astype(float)
        C_mgL = C_ugL * CF_ugL_to_mgL
        C_bio_ugL = np.column_stack([pick_bio_column(df, m) for m in metals]).astype(float) if args.use_bioavailable else np.zeros_like(C_ugL)
        C_bio_mgL = C_bio_ugL * CF_ugL_to_mgL

    # ==========================================================================
    # Anion concentration processing: Fluoride and Nitrate
    # ==========================================================================
    # Anions are in mg/L typically, need to convert to match RfD basis
    anion_C_mgL = {}  # Store anion concentrations in mg/L
    
    if anions:
        for anion in anions:
            col_name = anion_columns.get(anion, "")
            if col_name and col_name in df.columns:
                # Read anion concentration
                anion_vals = pd.to_numeric(df[col_name], errors='coerce').fillna(0.0).to_numpy(float)
                
                # Handle unit conversion if input was µg/L
                if args.input_units == 'ug/L':
                    # Anions are typically in mg/L even when metals are in µg/L
                    # Check if values look like they're already in mg/L (reasonable range)
                    if anion_vals.max() > 100:  # Values >100 suggest it might be µg/L
                        logging.warning(f"Anion '{anion}' values are high ({anion_vals.max():.2f}). "
                                       f"Assuming mg/L despite --input-units=ug/L for metals.")
                else:
                    # input-units is mg/L, values are already correct
                    pass
                
                # Apply nitrate basis conversion if needed
                if anion == "nitrate" and "nitrate" in anion_metadata:
                    from_basis = anion_metadata["nitrate"]["detected_basis"]
                    to_basis = anion_metadata["nitrate"]["target_basis"]
                    if from_basis != to_basis:
                        # Convert to RfD basis (NO3-N)
                        anion_vals = np.array([
                            convert_nitrate_basis_mgL(v, from_basis, to_basis) 
                            for v in anion_vals
                        ])
                        print(f"  Converted nitrate from {from_basis} to {to_basis} basis")
                
                anion_C_mgL[anion] = anion_vals
            else:
                logging.warning(f"Anion column '{col_name}' not found for {anion}")
                anion_C_mgL[anion] = np.zeros(J, float)
    
    # Add anions to the metals list for unified processing
    # Extend concentration matrices to include anions
    if anions:
        n_anions = len(anions)
        # Extend metals list
        metals = metals + anions
        M = len(metals)
        midx = {m:i for i,m in enumerate(metals)}
        
        # Extend concentration matrices
        C_mgL_old = C_mgL
        C_bio_mgL_old = C_bio_mgL
        C_mgL = np.zeros((J, M), float)
        C_bio_mgL = np.zeros((J, M), float)
        
        # Copy metal concentrations
        C_mgL[:, :M-n_anions] = C_mgL_old
        C_bio_mgL[:, :M-n_anions] = C_bio_mgL_old
        
        # Add anion concentrations (no dermal for anions)
        for anion in anions:
            C_mgL[:, midx[anion]] = anion_C_mgL.get(anion, np.zeros(J))
            C_bio_mgL[:, midx[anion]] = 0.0  # No dermal for anions
        
        print(f"Extended analyte list to include anions: {metals[-n_anions:]}")

    # Dermal availability check: for scientific runs we prefer an explicit error when
    # no bioavailable concentrations are present rather than silently disabling dermal.
    dermal_has_bio = bool(np.isfinite(C_bio_mgL).any() and C_bio_mgL.sum() > 0)
    if not dermal_has_bio:
        if not args.allow_disable_dermal_if_no_bio:
            logging.warning("No bioavailable concentrations found. Dermal route will be DISABLED. "
                            "To suppress this warning, use --allow-disable-dermal-if-no-bio.")
        else:
            logging.info("No bioavailable concentrations found. Dermal route disabled as requested.")

    # Tox tables and organ sets
    toxref_path = os.path.join(EXTERNAL_DIR, "toxref.yml")
    toxref = {}
    organ_sets_from_toxref = None
    if os.path.exists(toxref_path):
        try:
            loaded = yaml.safe_load(open(toxref_path)) or {}
            toxref = loaded.get("tox", {})
            organ_sets_from_toxref = loaded.get("organ_sets")
        except Exception:
            logging.warning("Failed to parse toxref YAML at %s; ignoring toxref file.", toxref_path)

    if organ_sets_from_toxref:
        base_sets = {k.lower(): set(v) for k, v in organ_sets_from_toxref.items()}
    else:
        if not args.allow_default_organ_sets:
            raise RuntimeError(
                f"organ_sets must be defined in {toxref_path}. "
                f"Pass --allow-default-organ-sets to permit fallback mapping."
            )
        base_sets = ORGAN_SETS_FALLBACK

    organ_sets_used, systemic_mask_np, unknown_orgs, organ_routes = build_organ_sets(metals, toxref, base_sets=base_sets)
    
    # Normalize organ sets to use exactly the metal names from our metal list
    organ_sets_used = _normalize_organ_sets_to_metals(organ_sets_used, metals)
    
    if unknown_orgs:
        logging.warning("Unknown organ names in toxref: %s", sorted(list(unknown_orgs)))

    # Dermal Kp table
    kp_table = {}
    kp_path = os.path.join(EXTERNAL_DIR, "dermal_water_kp.yml")
    if os.path.exists(kp_path):
        kp_table = yaml.safe_load(open(kp_path)).get("Kp_cm_per_hr", {}) or {}
    # Resolve kp keys allowing alternative names like 'Hg(II)' or 'CrVI'
    kp_key_map = {m: resolve_key_in_dict(m, kp_table) for m in metals}
    if dermal_has_bio:
        missing_kp = [m for m, k in kp_key_map.items() if k is None]
        if missing_kp:
            raise RuntimeError(
                "Missing Kp values in external/dermal_water_kp.yml for metals: " + 
                ",".join(sorted(missing_kp)) +
                ". Provide explicit Kp_cm_per_hr entries for each metal when using dermal route."
            )
        Kp_vec = np.array([float(kp_table[kp_key_map[m]]) for m in metals], float)
    else:
        Kp_vec = np.zeros(len(metals), float)

    # Toxicity: resolve toxref keys for metals and collect oral and dermal RfD/SF and ABS_GI
    RFD_map = {}
    CSF_map = {}
    RFD_derm_map = {}
    CSF_derm_map = {}
    ABS_map = {}
    missing_rfd = []
    missing_csf = []
    for m in metals:
        k = resolve_key_in_dict(m, toxref) if toxref else None
        v = toxref.get(k, {}) if (toxref and k) else {}
        rfd = v.get("RfD_oral", None)
        sf  = v.get("SF_oral",  None)
        rfd_derm = v.get("RfD_derm", None)
        sf_derm  = v.get("SF_derm", None)
        abs_gi   = v.get("ABS_GI", None)

        # Treat YAML nulls consistently
        if rfd is None or pd.isna(rfd):
            missing_rfd.append(m)
            RFD_map[m] = None
        else:
            RFD_map[m] = float(rfd)

        if sf is None or pd.isna(sf):
            missing_csf.append(m)
            CSF_map[m] = None
        else:
            CSF_map[m] = float(sf)

        RFD_derm_map[m] = float(rfd_derm) if (rfd_derm is not None and not pd.isna(rfd_derm)) else None
        CSF_derm_map[m] = float(sf_derm) if (sf_derm is not None and not pd.isna(sf_derm)) else None
        ABS_map[m] = float(abs_gi) if (abs_gi is not None and not pd.isna(abs_gi)) else None

    if missing_rfd:
        logging.warning(
            "Missing RfD_oral entries in toxref for metals: " + ",".join(sorted(missing_rfd)) +
            ". Assuming non-toxic (RfD=inf) for these metals."
        )
        for m in missing_rfd:
            RFD_map[m] = np.inf
            
    if missing_csf:
        logging.warning(
            "Missing SF_oral entries in toxref for metals: " + ",".join(sorted(missing_csf)) +
            ". Assuming non-carcinogenic (SF=0.0) for these metals."
        )
        for m in missing_csf:
            CSF_map[m] = 0.0

    RFDS = np.array([RFD_map.get(m, np.nan) for m in metals], float)  # (M,)
    CSFS = np.array([CSF_map.get(m, 0.0) for m in metals], float)  # (M,)
    RFDS_derm = np.array([RFD_derm_map.get(m, np.nan) for m in metals], float)
    CSFS_derm = np.array([CSF_derm_map.get(m, 0.0) for m in metals], float)
    ABS_VEC = np.array([ABS_map.get(m, 1.0) if (ABS_map.get(m, None) is not None) else 1.0 for m in metals], float)

    # Run the publication-mode validator (module-level) so we fail fast if RfD_derm are missing
    try:
        validate_dermal_requirements(metals, C_bio_mgL, RFDS_derm)
    except NameError:
        # If C_bio_mgL isn't defined (shouldn't happen), skip and let later checks handle it
        pass

    # If dermal route is active (bioavailable), require dermal-specific RfD entries or
    # explicit policy to fallback to oral values. We enforce strictness to avoid silent
    # scientifically-questionable fallbacks.
    if dermal_has_bio:
        # Only require dermal RfD for metals that are used dermally (per organ_routes or organ_sets)
        dermally_used = set()
        # organ_routes holds explicit preferences; organ_sets_used will be built later, so inspect toxref target_organs
        for k_m, mmeta in (toxref.items() if toxref else []):
            pass
        # Fallback: require dermal RfD for any metal if dermal is active
        missing_derm_rfd = [m for i, m in enumerate(metals) if not np.isfinite(RFDS_derm[i])]
        if missing_derm_rfd:
            logging.warning(
                "Dermal bioavailable concentrations present but missing dermal RfD (RfD_derm) for: " +
                ",".join(sorted(missing_derm_rfd)) +
                ". Assuming non-toxic dermally (RfD_derm=inf) for these metals."
            )
            for m in missing_derm_rfd:
                # Update the array directly since we've already built it
                idx = metals.index(m)
                RFDS_derm[idx] = np.inf

    # Groups - filter based on user selection
    try:
        selected_groups = parse_group_selection(args.groups)
        ACTIVE_GROUP_INFO = get_group_info_filtered(selected_groups)
    except ValueError as e:
        print(f"Error parsing --groups: {e}")
        print(f"Using all groups as fallback.")
        ACTIVE_GROUP_INFO = GROUP_INFO
    
    GROUPS = list(ACTIVE_GROUP_INFO.keys())
    G = len(GROUPS)
    print(f"\nAnalyzing {G} demographic group(s): {', '.join(GROUPS)}")
    IR_mean = np.array([ACTIVE_GROUP_INFO[g]["IR"] for g in GROUPS], float)
    BW_mean = np.array([ACTIVE_GROUP_INFO[g]["BW"] for g in GROUPS], float)
    ED_days = np.array([ACTIVE_GROUP_INFO[g]["ED"] for g in GROUPS], float)
    AT_nc  = np.array([ACTIVE_GROUP_INFO[g]["AT_nc"] for g in GROUPS], float)
    AT_ca  = np.array([ACTIVE_GROUP_INFO[g]["AT_c"]  for g in GROUPS], float)
    EF_days = np.full(G, DAYS_PER_YEAR, float)
    SA_cm2 = np.array([ACTIVE_GROUP_INFO[g].get("SA", 0.0) for g in GROUPS], float)
    ET_raw = np.array([ACTIVE_GROUP_INFO[g].get("ET", 0.0) for g in GROUPS], float)
    AF_frac= np.array([ACTIVE_GROUP_INFO[g].get("AF", 0.0) for g in GROUPS], float)

    # NOTE: dermal HQ will be computed inside the PyMC model so that group-level
    # uncertainty (e.g., BW_g) propagates correctly. We do not precompute HQ here.

    # -------------------------- Build PyMC model ------------------------------- #
    coords = {"site": np.arange(J), "metal": metals, "group": GROUPS}

    # Observed/LOD arrays and latent censoring removed; concentrations are provided by pre-imputation

    with pm.Model(coords=coords) as model:
        # BW prior per group: LogNormal with CV 0.21 around BW_mean
        # Body weight per group: non-centered LogNormal
        CV_BW = 0.21
        sigma_log_bw = np.sqrt(np.log(1.0 + CV_BW**2))
        mu_log_bw = np.log(BW_mean) - 0.5*np.log(1.0 + CV_BW**2)
        # non-centered: z ~ Normal(0,1), log_bw = mu + z*sigma
        z_log_bw = pm.Normal("z_log_bw", mu=0.0, sigma=1.0, shape=G)
        log_BW_g = pm.Deterministic("log_BW_g", mu_log_bw + z_log_bw * sigma_log_bw)
        BW_g = pm.Deterministic("BW_g", at.exp(log_BW_g))

        # Ingestion rate per kg-day prior: LogNormal anchored to demographics
        # Ingestion rate per kg-day prior: non-centered LogNormal around demographic median
        IR_perkg_med = IR_mean / BW_mean  # L/kg-day medians per group
        sigma_log_ir = 0.6
        mu_log_ir = np.log(IR_perkg_med)
        z_log_ir = pm.Normal("z_log_ir", mu=0.0, sigma=1.0, shape=G)
        log_IR_perkg_g = pm.Deterministic("log_IR_perkg_g", mu_log_ir + z_log_ir * sigma_log_ir)
        IR_perkg_g = pm.Deterministic("IR_perkg_g", at.exp(log_IR_perkg_g))

        # Concentration tensors
        C = pm.Data("C", C_mgL, dims=("site", "metal"))

        C_bio = pm.Data("C_bio", C_bio_mgL, dims=("site", "metal"))

        # Time scaling
        EF_v  = at.as_tensor_variable(EF_days / DAYS_PER_YEAR)     # fraction of year
        EDY_v = at.as_tensor_variable(ED_days / DAYS_PER_YEAR)     # years
        ATNC_v= at.as_tensor_variable(AT_nc  / DAYS_PER_YEAR)      # years
        ATCA_v= at.as_tensor_variable(AT_ca  / DAYS_PER_YEAR)      # years

        # Ingestion factors
        F_ing_nc = (IR_perkg_g * EF_v * EDY_v) / ATNC_v
        F_ing_ca = (IR_perkg_g * EF_v * EDY_v) / ATCA_v

        # Dermal factors for in-graph EDI bookkeeping
        # Treat ET as hours everywhere; convert to fraction-of-day only when needed
        ET_hr_v  = at.as_tensor_variable(ET_raw)          # hours
        ET_frac  = ET_hr_v / 24.0                         # fraction of a day
        SA_v     = at.as_tensor_variable(SA_cm2)
        # For dermal-water only, do not apply soil adherence factor (AF); set to 1
        AF_v     = at.ones_like(SA_v)
        D_der_daily = (SA_v * ET_frac * AF_v) / BW_g
        F_der_nc = (D_der_daily * EF_v * EDY_v) / ATNC_v
        F_der_ca = (D_der_daily * EF_v * EDY_v) / ATCA_v

        # EDI tensors (J,M,G)
        # Apply ABS_GI absorption fraction (oral) to ingestion concentrations/dose
        ABS_v = at.as_tensor_variable(ABS_VEC)  # (M,)
        # C is mg/L; scale ingestion EDI by ABS (per-metal fraction)
        C_ing_effective = C * ABS_v[None, :]
        EDI_ing_nc = at.tensordot(C_ing_effective,     F_ing_nc, axes=0)
        EDI_ing_ca = at.tensordot(C_ing_effective,     F_ing_ca, axes=0)
        EDI_der_nc = at.tensordot(C_bio, F_der_nc, axes=0)
        EDI_der_ca = at.tensordot(C_bio, F_der_ca, axes=0)
        EDI_nc = EDI_ing_nc + EDI_der_nc
        EDI_ca = EDI_ing_ca + EDI_der_ca

        # HQ ingestion per metal/site/group: use ingestion-only EDI for ingestion HQ.
        # Use np.inf for missing/zero RfD so those metals are effectively ignored.
        RfD_vec = at.as_tensor_variable(np.where(np.isfinite(RFDS) & (RFDS > 0), RFDS, np.inf))  # (M,)
        HQ_ing_mjg = at.transpose(EDI_ing_nc, (1,0,2)) / RfD_vec[:, None, None]

        # Dermal HQ computed in-graph so BW_g and other group uncertain parameters propagate.
        # We prepared RFDS_derm earlier and enforced presence when dermal is active
        RFDS_for_derm = RFDS_derm
        RfD_safe = at.as_tensor_variable(np.where(np.isfinite(RFDS_for_derm) & (RFDS_for_derm > 0), RFDS_for_derm, np.inf))

        # t_event in hours (ET is already in hours; remove special-casing for <=1)
        t_event_hr_v = at.as_tensor_variable(ET_raw)
        SA_v     = at.as_tensor_variable(SA_cm2)
        # Dermal-water: AF set to 1.0
        AF_v     = at.ones_like(SA_v)
        # S_group (G,) scaling term
        S_group = (t_event_hr_v * SA_v * AF_v * EF_v * EDY_v) / (ATNC_v * BW_g)

        # base_mj: (M,J) factor from bio concentrations and Kp
        base_mj = (at.transpose(C_bio, (1,0)) * at.as_tensor_variable(Kp_vec)[:, None]) * 1e-3
        HQ_der_mjg = (base_mj / RfD_safe[:, None])[:, :, None] * S_group[None, None, :]

        # Total HQ per metal/site/group
        HQ_mjg = HQ_ing_mjg + HQ_der_mjg

        # Export for diagnostics
        try:
            pm.Deterministic("HQ_ing_mjg", HQ_ing_mjg, dims=("metal","site","group"))
            pm.Deterministic("HQ_der_mjg", HQ_der_mjg, dims=("metal","site","group"))
            pm.Deterministic("HQ_mjg",     HQ_mjg,     dims=("metal","site","group"))
        except Exception:
            pass

        # Organ masks and HI for all organ sets provided by toxref/base_sets.
        def mk_mask(mset: set) -> np.ndarray:
            # Use base key matching for metals
            mset_base = set(_base_key(m) for m in mset)
            return np.array([1 if _base_key(m) in mset_base else 0 for m in metals], float)

        organ_HIs = {}
        hi_vars = []
        # Route-aware organ HIs: some organ sets (e.g., 'derm' or 'gi') should
        # only include contributions from a specific route (dermal or ingestion).
        for oname, mset in organ_sets_used.items():
            lname = str(oname).strip().lower()
            mask = at.as_tensor_variable(mk_mask(mset))
            # If the user provided an explicit organ_routes mapping, prefer it.
            route_pref = organ_routes.get(lname, None) if isinstance(organ_routes, dict) else None
            # If the user explicitly set the override flag, and dermal data exist,
            # include dermal contributions (use HQ_mjg) for this organ whenever any
            # metal in the organ mask has non-zero bioavailable concentrations.
            # override is ON by default unless user asks to disable it
            if (not getattr(args, 'disable_dermal_override', False)) and dermal_has_bio:
                # check if any metal used by this organ has a non-zero bio concentration
                try:
                    # metals and midx are plain python lists/dicts available in scope
                    organ_indices = [i for i, m in enumerate(metals) if mk_mask(mset)[i] == 1]
                    dermal_nonzero = False
                    if organ_indices:
                        # C_bio_mgL is a numpy array (J, M) where M == len(metals)
                        # if any bio concentration > 0 for any site for these metals, consider dermal present
                        dermal_nonzero = bool((C_bio_mgL[:, organ_indices].sum(axis=0) > 0).any())
                    else:
                        dermal_nonzero = False
                except Exception:
                    dermal_nonzero = dermal_has_bio
                if dermal_nonzero:
                    # include both routes
                    sel = HQ_mjg
                # else: fall through to normal routing rules below
            # If override not active (or dermal was not non-zero), use explicit mapping or heuristics
            if 'sel' not in locals():
                if route_pref == "dermal":
                    sel = HQ_der_mjg if dermal_has_bio else HQ_ing_mjg
                elif route_pref == "ingestion":
                    sel = HQ_ing_mjg
                else:
                    # Fall back to heuristic string matches for backward compatibility
                    if lname in ("derm", "dermal", "dermal_water", "skin"):
                        sel = HQ_der_mjg if dermal_has_bio else HQ_ing_mjg
                    elif lname in ("gi", "g.i.", "gastrointestinal", "gut", "ingestion"):
                        sel = HQ_ing_mjg
                    else:
                        sel = HQ_mjg if dermal_has_bio else HQ_ing_mjg

            hi = pm.Deterministic(f"HI_{oname}", at.sum(sel * mask[:, None, None], axis=0), dims=("site","group"))
            organ_HIs[oname] = hi
            hi_vars.append(hi)

        # Ensure we always include a systemic HI if present in mapping
        if systemic_mask_np.any() and "systemic" not in organ_HIs:
            systemic_mask = at.as_tensor_variable(systemic_mask_np)
            hi_sys = pm.Deterministic("HI_systemic", at.sum(HQ_mjg * systemic_mask[:, None, None], axis=0), dims=("site","group"))
            organ_HIs["systemic"] = hi_sys
            hi_vars.append(hi_sys)

        # Overall HI is the max across all computed organ HI arrays
        if len(hi_vars) == 0:
            # Fallback (shouldn't happen): overall HI as max across zero -> zeros
            HI_overall = pm.Deterministic("HI_overall", at.zeros((J, G)), dims=("site","group"))
        else:
            HI_overall = pm.Deterministic("HI_overall", at.max(at.stack(hi_vars, axis=0), axis=0), dims=("site","group"))

        # Per-group aliases and per-metal contributions
        try:
            for gi, g in enumerate(GROUPS):
                pm.Deterministic(f"HI_total_{g}", HI_overall[:, gi], dims=("site",))
                for i, m in enumerate(metals):
                    pm.Deterministic(f"HI_{g}_{m}", HQ_mjg[i, :, gi], dims=("site",))
        except Exception:
            pass
        

        # NOTE: HI_metals_screen, HI_anions_screen, HI_all_screen have been removed
        # as they are redundant with the organ-based HI approach. The organ-specific
        # HIs (HI_neuro, HI_nephro, etc.) and HI_overall provide scientifically
        # proper risk characterization that accounts for target organ specificity.


        # Cancer risk: include ingestion and dermal contributions. Use dermal-specific
        # cancer slope factor where provided (CSFS_derm); otherwise fall back to oral CSF.
        CSF_vec_ing = at.as_tensor_variable(CSFS)
        CSF_vec_derm = at.as_tensor_variable(CSFS_derm)
        CR_ing_mjg = at.transpose(EDI_ing_ca, (1,0,2)) * CSF_vec_ing[:, None, None]
        CR_der_mjg = at.transpose(EDI_der_ca, (1,0,2)) * CSF_vec_derm[:, None, None]
        CR_mjg = CR_ing_mjg + CR_der_mjg
        # Guard against any non-finite values (NaN/inf) produced by arithmetic
        # (e.g., 0 * inf) which would make CR_total unusable in summaries.
        finite_mask = (~at.isnan(CR_mjg)) & (~at.isinf(CR_mjg))
        CR_mjg_safe = at.switch(finite_mask, CR_mjg, 0.0)
        pm.Deterministic("CR_by_m", at.transpose(CR_mjg_safe, (1,0,2)), dims=("site","metal","group"))
        # Sum across all metals (non-carcinogens have SF==0)
        CR_total = pm.Deterministic("CR_total", at.sum(CR_mjg_safe, axis=0), dims=("site","group"))

        # Predictive Pb BLL (ingestion only)
        if "Pb" in metals:
            pb_idx = metals.index("Pb")
            EDI_pb = at.sum(at.transpose(EDI_ing_nc, (1,0,2)) * at.as_tensor_variable(np.eye(len(metals))[pb_idx])[:,None,None], axis=0)  # (J,G)

            # Try calibrated priors from results/calibration/priors.json
            priors_path = os.path.join(args.results_dir, "calibration", "priors.json")
            priors_data = None
            engine_metadata = {}  # Initialize as empty dict instead of None
            if os.path.exists(priors_path):
                try:
                    rawp = json.load(open(priors_path))
                    priors_data = rawp["groups"] if isinstance(rawp, dict) and "groups" in rawp else rawp
                    engine_metadata = rawp.get("engine_metadata", {}) if isinstance(rawp, dict) else {}
                except Exception:
                    priors_data = None
                    engine_metadata = {}

            # Check if we have population-specific engine information
            engines_per_group = engine_metadata.get("engines_per_group", {})
            population_classification = engine_metadata.get("population_classification", {})
            calibration_source = engine_metadata.get("calibration_source", "unknown")
            calibration_date = engine_metadata.get("calibration_date", "unknown")
            
            # Log the engine selection approach for transparency
            print("\n" + "="*60)
            print("BLL ENGINE DETAILS")
            print("="*60)
            if engines_per_group:
                print(f"Engine Mode: Population-specific")
                print(f"Calibration Source: {calibration_source}")
                print(f"Calibration Date: {calibration_date}")
                print(f"\nEngines per demographic group:")
                for group, engine in engines_per_group.items():
                    pop_class = population_classification.get(group, "unknown")
                    print(f"  • {group}: {engine} (population type: {pop_class})")
            else:
                engine_mode = engine_metadata.get("engine_selection_mode", "default (uncalibrated)")
                print(f"Engine Mode: {engine_mode}")
                if priors_data:
                    print(f"Calibration Source: {calibration_source}")
                    print(f"Calibration Date: {calibration_date}")
                    print(f"\nCalibrated prior parameters loaded for groups:")
                    for g in GROUPS:
                        if g in priors_data:
                            pd_g = priors_data[g]
                            print(f"  • {g}: k_mean={pd_g.get('k_mean', 'N/A'):.4f}, b0_mean={pd_g.get('b0_mean', 'N/A'):.4f}")
                else:
                    print("Using default uncalibrated priors (no calibration file found)")
                    print("  • b0 ~ HalfNormal(3.0)")
                    print("  • k ~ LogNormal(hierarchical)")
            print("="*60 + "\n")

            if priors_data:
                k_mean = np.array([priors_data[g]["k_mean"] for g in GROUPS], float)
                k_sd   = np.array([priors_data[g]["k_sd"]   for g in GROUPS], float)
                b0_mean= np.array([priors_data[g]["b0_mean"]for g in GROUPS], float)
                b0_sd  = np.array([priors_data[g]["b0_sd"]  for g in GROUPS], float)
                mu_logk    = np.log(k_mean / np.sqrt(1 + (k_sd/k_mean)**2))
                sigma_logk = np.sqrt(np.log(1 + (k_sd/k_mean)**2))
                # calibrated k stays as LogNormal (data-driven)
                k  = pm.LogNormal("k", mu=mu_logk, sigma=sigma_logk, shape=G)
                # non-centered for calibrated b0: z*b0_sd + b0_mean
                z_b0 = pm.Normal("z_b0", mu=0.0, sigma=1.0, shape=G)
                b0 = pm.Deterministic("b0", b0_mean + z_b0 * b0_sd)
            else:
                b0 = pm.HalfNormal("b0", 3.0, shape=G)
                # Non-centered parameterization for log_k hierarchy to improve sampler geometry
                mu_log_k = pm.Normal("mu_log_k", 0.0, 1.0)
                sigma_log_k = pm.HalfNormal("sigma_log_k", 0.7)
                # z ~ Normal(0,1) and log_k = mu + z*sigma
                z_log_k = pm.Normal("z_log_k", mu=0.0, sigma=1.0, shape=G)
                log_k = pm.Deterministic("log_k", mu_log_k + z_log_k * sigma_log_k)
                k = pm.Deterministic("k", at.exp(log_k))

            BLL = pm.Deterministic("BLL", at.clip(b0[None, :] + k[None, :] * EDI_pb, 0, np.inf), dims=("site","group"))

        if args.save_prior_pred:
            pps = pm.sample_prior_predictive(1000, random_seed=args.seed, return_inferencedata=False)
            np.save(os.path.join(args.results_dir, "prior_pred_BLL.npy"),
                    np.asarray(pps.get("BLL", np.empty((0,)))))

        trace = pm.sample(draws=args.draws, tune=args.tune, chains=4, target_accept=0.9, random_seed=args.seed)

    # Create high-precision debug outputs to see the true small values
    summary = {}
    posterior = az.extract(trace, var_names=[v.name for v in model.deterministics])

    def summarize(name):
        x = posterior[name].values  # shape: (site, group, samples) after az.extract
        # Compute statistics over the samples axis (last axis, -1)
        mean_val = np.mean(x, axis=-1)
        med = np.median(x, axis=-1)
        lo  = np.percentile(x, 3.0, axis=-1)
        hi  = np.percentile(x, 97.0, axis=-1)
        return {"mean": mean_val.tolist(), "median": med.tolist(), "p3": lo.tolist(), "p97": hi.tolist()}

    def summarize_with_samples(name):
        """Return summary stats plus raw samples for exceedance calculations."""
        x = posterior[name].values  # shape: (site, group, samples) after az.extract
        mean_val = np.mean(x, axis=-1)
        med = np.median(x, axis=-1)
        lo  = np.percentile(x, 3.0, axis=-1)
        hi  = np.percentile(x, 97.0, axis=-1)
        return {
            "mean": mean_val.tolist(), "median": med.tolist(), 
            "p3": lo.tolist(), "p97": hi.tolist(),
            "samples": x  # Keep raw samples for exceedance calculation
        }

    def exceedance_prob(samples, threshold):
        """Compute P(X > threshold) from posterior samples."""
        return np.mean(samples > threshold, axis=-1)

    # Process all HI variables
    for v in model.deterministics:
        if v.name.startswith("HI_") and "_" in v.name[3:] and any(g in v.name for g in GROUPS):
            continue  # Skip per-metal per-group aliases like HI_Adults_As
        if v.name.startswith("HI_"):
            try:
                summary[v.name] = summarize_with_samples(v.name)
            except Exception as e:
                logging.warning(f"Failed to summarize {v.name}: {e}")
                continue

    # Write high-precision debug outputs
    outdir = os.path.join(args.results_dir, "debug")
    os.makedirs(outdir, exist_ok=True)
    
    # HI_summary.json (without samples)
    summary_json = {k: {kk: vv for kk, vv in v.items() if kk != "samples"} for k, v in summary.items()}
    with open(os.path.join(outdir, "HI_summary.json"), "w") as fh:
        json.dump(summary_json, fh, indent=2)

    # Write a flat CSV with high precision for human inspection
    # Include both site name and site index for clarity
    # Added: probability of exceedance P(HI > 1)
    rows = []
    for name, d in summary.items():
        med = np.array(d["median"])
        p3  = np.array(d["p3"])
        p97 = np.array(d["p97"])
        mean_val = np.array(d["mean"])
        samples = d["samples"]
        prob_exceed_1 = exceedance_prob(samples, 1.0)
        J, G = med.shape
        for j in range(J):
            site_label = site_names[j] if j < len(site_names) else f"Site_{j}"
            for gi, g in enumerate(GROUPS):
                rows.append({
                    "organ": name.replace("HI_",""), "site_id": j, "site_name": site_label, "group": g,
                    "mean": f"{mean_val[j,gi]:.6g}", "median": f"{med[j,gi]:.6g}", 
                    "p3": f"{p3[j,gi]:.6g}", "p97": f"{p97[j,gi]:.6g}",
                    "P(HI>1)": f"{prob_exceed_1[j,gi]:.4f}"
                })
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "HI_summary.csv"), index=False)

    # ==================== TCR (CR_total) Summary ====================
    try:
        cr_data = summarize_with_samples("CR_total")
        cr_rows = []
        cr_med = np.array(cr_data["median"])
        cr_p3 = np.array(cr_data["p3"])
        cr_p97 = np.array(cr_data["p97"])
        cr_mean = np.array(cr_data["mean"])
        cr_samples = cr_data["samples"]
        
        # Exceedance probabilities for various thresholds
        prob_cr_1e4 = exceedance_prob(cr_samples, 1e-4)
        prob_cr_1e5 = exceedance_prob(cr_samples, 1e-5)
        prob_cr_1e6 = exceedance_prob(cr_samples, 1e-6)
        
        J, G = cr_med.shape
        for j in range(J):
            site_label = site_names[j] if j < len(site_names) else f"Site_{j}"
            for gi, g in enumerate(GROUPS):
                cr_rows.append({
                    "site_id": j, "site_name": site_label, "group": g,
                    "mean": f"{cr_mean[j,gi]:.6g}", "median": f"{cr_med[j,gi]:.6g}",
                    "p3": f"{cr_p3[j,gi]:.6g}", "p97": f"{cr_p97[j,gi]:.6g}",
                    "P(TCR>1e-4)": f"{prob_cr_1e4[j,gi]:.4f}",
                    "P(TCR>1e-5)": f"{prob_cr_1e5[j,gi]:.4f}",
                    "P(TCR>1e-6)": f"{prob_cr_1e6[j,gi]:.4f}"
                })
        pd.DataFrame(cr_rows).to_csv(os.path.join(outdir, "TCR_summary.csv"), index=False)
        
        # Also save JSON
        cr_json = {k: v for k, v in cr_data.items() if k != "samples"}
        cr_json["P(TCR>1e-4)"] = prob_cr_1e4.tolist()
        cr_json["P(TCR>1e-5)"] = prob_cr_1e5.tolist()
        cr_json["P(TCR>1e-6)"] = prob_cr_1e6.tolist()
        with open(os.path.join(outdir, "TCR_summary.json"), "w") as fh:
            json.dump(cr_json, fh, indent=2)
        logging.info("TCR_summary.csv and TCR_summary.json saved to debug/")
    except Exception as e:
        logging.warning(f"Could not generate TCR summary: {e}")

    # ==================== BLL Summary ====================
    try:
        bll_data = summarize_with_samples("BLL")
        bll_rows = []
        bll_med = np.array(bll_data["median"])
        bll_p3 = np.array(bll_data["p3"])
        bll_p97 = np.array(bll_data["p97"])
        bll_mean = np.array(bll_data["mean"])
        bll_samples = bll_data["samples"]
        
        # Parse BLL thresholds from args
        bll_thresholds = [float(x) for x in args.bll_thresholds.split(",")]
        
        # Compute exceedance for each threshold
        bll_exceedance = {}
        for thr in bll_thresholds:
            bll_exceedance[thr] = exceedance_prob(bll_samples, thr)
        
        J, G = bll_med.shape
        for j in range(J):
            site_label = site_names[j] if j < len(site_names) else f"Site_{j}"
            for gi, g in enumerate(GROUPS):
                row = {
                    "site_id": j, "site_name": site_label, "group": g,
                    "mean": f"{bll_mean[j,gi]:.6g}", "median": f"{bll_med[j,gi]:.6g}",
                    "p3": f"{bll_p3[j,gi]:.6g}", "p97": f"{bll_p97[j,gi]:.6g}"
                }
                # Add exceedance columns for each BLL threshold
                for thr in bll_thresholds:
                    row[f"P(BLL>{thr})"] = f"{bll_exceedance[thr][j,gi]:.4f}"
                bll_rows.append(row)
        pd.DataFrame(bll_rows).to_csv(os.path.join(outdir, "BLL_summary.csv"), index=False)
        
        # Also save JSON
        bll_json = {k: v for k, v in bll_data.items() if k != "samples"}
        for thr in bll_thresholds:
            bll_json[f"P(BLL>{thr})"] = bll_exceedance[thr].tolist()
        with open(os.path.join(outdir, "BLL_summary.json"), "w") as fh:
            json.dump(bll_json, fh, indent=2)
        logging.info("BLL_summary.csv and BLL_summary.json saved to debug/")
    except Exception as e:
        logging.warning(f"Could not generate BLL summary: {e}")

    # Log warning if values are extremely small but non-zero
    max_median = max(np.max(np.array(d["median"])) for d in summary.values())
    if max_median < 0.01:
        logging.warning(
            f"All HI values are very small (max median = {max_median:.2e}). "
            f"This is expected for low concentrations but may display as 0.0 with default formatting. "
            f"Check debug/HI_summary.csv for precise values."
        )

    # No in-graph censoring: all imputation has been done prior to model fitting.

    # Persist artifacts
    az.to_netcdf(trace, os.path.join(args.results_dir, "trace.nc"))
    with open(os.path.join(args.results_dir, "model.pkl"), "wb") as fh:
        dill.dump(model, fh)

    # Run log
    # derive carcinogens from provided CSFS (non-zero SF indicates carcinogen)
    derived_carc = [m for m, sf in zip(metals, CSFS) if np.isfinite(sf) and sf > 0]

    runlog = {
        "metals": metals,
        "organ_sets": {k: sorted(v) for k, v in organ_sets_used.items()},
        "organ_sets_source": "toxref" if organ_sets_from_toxref else ("builtin" if args.allow_default_organ_sets else "missing"),
        "organ_routes": organ_routes if organ_routes else None,
        "carcinogens": sorted(derived_carc),
        "speciation_source": spec_src,
        "phreeqc_bio_path": bio_path,
        "phreeqc_frac_path": frac_path,
        "bll_thresholds": [float(x) for x in args.bll_thresholds.split(",")],
    "use_bioavailable": args.use_bioavailable,
        "seed": args.seed, "draws": args.draws, "tune": args.tune,
    }
    json.dump(runlog, open(os.path.join(args.results_dir, "RUNLOG.json"), "w"), indent=2)

    # Assumptions for transparency
    RFD_abs_map = {}
    SF_abs_map  = {}
    for m in metals:
        entry = toxref.get(m, {}) if toxref else {}
        # Prefer dermal-specific RfD/SF if provided; otherwise fall back to oral or map.
        rfd_derm = entry.get("RfD_derm")
        rfd_oral = entry.get("RfD_oral")
        sf_derm = entry.get("SF_derm")
        sf_oral  = entry.get("SF_oral")
        if rfd_derm is not None:
            rfd_abs = float(rfd_derm)
        elif rfd_oral is not None:
            rfd_abs = float(rfd_oral)
        else:
            rfd_abs = RFD_map.get(m)

        if sf_derm is not None:
            sf_abs = float(sf_derm)
        elif sf_oral is not None:
            sf_abs = float(sf_oral)
        else:
            sf_abs = CSF_map.get(m)

        RFD_abs_map[m] = rfd_abs
        SF_abs_map[m]  = sf_abs

    assumptions = {
        "Kp_used": kp_table,
        "kp_key_map": {m: kp_key_map.get(m) for m in metals},
        "toxref_source": toxref_path if os.path.exists(toxref_path) else None,
        "toxref_used": toxref,
        "ABS_used": {m: (ABS_map.get(m) if m in ABS_map else None) for m in metals},
        "organ_sets_used": {k: sorted(v) for k, v in organ_sets_used.items()},
        "organ_sets_source": "toxref" if organ_sets_from_toxref else ("builtin" if args.allow_default_organ_sets else "missing"),
        "organ_routes_used": organ_routes if organ_routes else {},
        "RFD_abs": RFD_abs_map,
        "SF_abs":  SF_abs_map,
    # dermal_water route is enabled only if we have non-zero bioavailable concentrations
    "routes": {"ingestion": True, "dermal_water": bool(dermal_has_bio)},
        "speciation_source": spec_src,
        "phreeqc_bio_path": bio_path,
        "phreeqc_frac_path": frac_path,
    "censored": False,
        "systemic_metals": [m for m, v in zip(metals, systemic_mask_np) if v == 1.0],
        "input_units_original": args.input_units,
        "concentration_units": "µg/L",  # Internal working units (always µg/L)
        "bll_units": "µg/dL",  # BLL output units
    }
    json.dump(assumptions, open(os.path.join(args.results_dir, "ASSUMPTIONS.json"), "w"), indent=2)


if __name__ == "__main__":
    if pm is None or at is None or az is None or dill is None:
        raise RuntimeError("This script requires pymc, pytensor, arviz, and dill.")
    main()
