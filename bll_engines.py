#!/usr/bin/env python3
"""
bll_engines.py
Scientifically-defensible, transparent BLL calculators for Pb from drinking water.

Engines:
- onecomp: steady-state one-compartment PBK (primary engine for adults)
- slope: linear intake->BLL for sensitivity/triage (preferred for vulnerable populations)
- auto: automatic selection based on population group characteristics

Population-Specific Engine Selection:
- Adults: onecomp (mechanistic PK model with established parameters)
- Children/Teens/Pregnant: slope (empirical dose-response with higher sensitivity)
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class OneCompParams:
    f_abs: float = 0.5            # fraction absorbed; children ~0.5, adults ~0.2–0.3
    t_half_days: float = 30.0     # blood half-life (28–36 d reasonable)
    blood_vol_per_kg: float = 0.07 # L/kg
    background_ugdl: float = 0.0  # baseline BLL to add (µg/dL)


@dataclass
class PopulationEngineConfig:
    """Configuration for population-specific engine selection"""
    adult_groups: Optional[list] = None
    vulnerable_groups: Optional[list] = None
    default_onecomp_params: Optional[OneCompParams] = None
    default_slopes: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.adult_groups is None:
            self.adult_groups = ["adults", "adult"]
        if self.vulnerable_groups is None:
            self.vulnerable_groups = ["children", "child", "teens", "teen", "pregnant", "pregnancy", "infant", "toddler"]
        if self.default_onecomp_params is None:
            self.default_onecomp_params = OneCompParams()
        if self.default_slopes is None:
            self.default_slopes = {
                "adult": 0.08,
                "child": 0.17,
                "teen": 0.17,
                "pregnant": 0.08,
                "generic": 0.12
            }


def classify_population_group(group_name: str) -> str:
    """
    Classify population group for engine selection.
    
    Args:
        group_name: Name of the population group
        
    Returns:
        'adult' for mechanistic modeling, 'vulnerable' for empirical modeling
    """
    g = group_name.lower().strip()
    
    # Adult classification
    if any(keyword in g for keyword in ["adult"]):
        return "adult"
    
    # Vulnerable population classification
    if any(keyword in g for keyword in ["child", "infant", "toddler", "teen", "preg", "lact"]):
        return "vulnerable"
    
    # Default to vulnerable for safety (conservative approach)
    return "vulnerable"


def get_group_specific_slope(group_name: str, config: Optional[PopulationEngineConfig] = None) -> float:
    """Get population-specific slope for empirical dose-response modeling."""
    if config is None:
        config = PopulationEngineConfig()
    
    g = group_name.lower()
    if any(x in g for x in ["child", "infant", "toddler"]):
        return config.default_slopes["child"]
    elif any(x in g for x in ["teen"]):
        return config.default_slopes["teen"]
    elif any(x in g for x in ["adult", "preg", "lact"]):
        return config.default_slopes["adult"]
    else:
        return config.default_slopes["generic"]


def get_group_specific_onecomp_params(group_name: str, base_params: Optional[OneCompParams] = None) -> OneCompParams:
    """Get population-specific parameters for one-compartment modeling."""
    if base_params is None:
        base_params = OneCompParams()
    
    # Create a copy to avoid modifying the original
    params = OneCompParams(
        f_abs=base_params.f_abs,
        t_half_days=base_params.t_half_days,
        blood_vol_per_kg=base_params.blood_vol_per_kg,
        background_ugdl=base_params.background_ugdl
    )
    
    # Adult-specific adjustments: lower absorption fraction
    if "adult" in group_name.lower():
        params.f_abs = min(params.f_abs, 0.3)
    
    return params


def edi_from_conc_ugL(conc_ugL: float, ir_L_per_d: float, bw_kg: float) -> float:
    """EDI (mg/kg-day) from concentration (µg/L), ingestion rate (L/d), body weight (kg)."""
    conc_mgL = max(float(conc_ugL), 0.0) * 1e-3
    return (conc_mgL * max(float(ir_L_per_d), 0.0)) / max(float(bw_kg), 1e-12)


def bll_onecomp_from_water(conc_ugL: float, ir_L_per_d: float, bw_kg: float,
                           params: Optional[OneCompParams] = None) -> float:
    """
    Steady-state BLL (µg/dL) for water-only ingestion:
    BLL_ss = (Dose_ug/d * f_abs) / (k_el [1/d] * Vb [L] * 10) + background
    where k_el = ln(2)/t_half, Vb = 0.07 L/kg * BW.
    Units: conc_ugL (µg/L), ir_L_per_d (L/d), bw_kg (kg) -> returns µg/dL
    """
    if params is None:
        params = OneCompParams()
    k_el = np.log(2.0) / max(params.t_half_days, 1e-6)
    vb_L = max(params.blood_vol_per_kg * max(float(bw_kg), 0.0), 1e-6)
    dose_ug_d = max(float(conc_ugL), 0.0) * max(float(ir_L_per_d), 0.0)  # µg/d
    bll = (dose_ug_d * np.clip(params.f_abs, 0.0, 1.0)) / (k_el * vb_L * 10.0)
    return float(bll + params.background_ugdl)


def bll_linear_from_intake(edi_mgkgd: float, bw_kg: float,
                           slope_ugdl_per_ugday: float,
                           f_abs: float = 1.0,
                           background_ugdl: float = 0.0) -> float:
    """
    Linear intake->BLL: BLL = background + slope * (Intake_ug/d * f_abs)
    Intake_ug/d = EDI_mg/kg/d * BW_kg * 1e6
    """
    intake_ug_d = max(float(edi_mgkgd), 0.0) * max(float(bw_kg), 0.0) * 1e6
    return float(background_ugdl + slope_ugdl_per_ugday * (intake_ug_d * np.clip(f_abs, 0.0, 1.0)))


def compute_bll(method: str, *, edi_mgkgd: Optional[float] = None, conc_ugL: Optional[float] = None,
                ir_L_per_d: Optional[float] = None, bw_kg: Optional[float] = None, params: Optional[OneCompParams] = None,
                slope_ugdl_per_ugday: Optional[float] = None, background_ugdl: float = 0.0) -> float:
    """
    Compute a BLL (µg/dL) using a named engine.
    - method: 'onecomp', 'slope', or 'auto'
    - For 'onecomp' you may provide conc_ugL + ir_L_per_d + bw_kg (water-based) OR edi_mgkgd + bw_kg.
    - For 'slope' provide edi_mgkgd and slope_ugdl_per_ugday.
    - For 'auto' provide group_name and the function will select appropriate engine.
    Returns a float µg/dL
    """
    method = (method or "onecomp").lower()
    if method == "onecomp":
        # Prefer water-based inputs if present
        if conc_ugL is not None and ir_L_per_d is not None and bw_kg is not None:
            return bll_onecomp_from_water(conc_ugL, ir_L_per_d, bw_kg, params=params)
        # Fallback to edi
        if edi_mgkgd is not None and bw_kg is not None:
            # convert edi (mg/kg-d) to conc+ir proxy by computing intake µg/d and dividing by IR
            # If IR not provided, assume 2 L/d to get a water-equivalent estimate
            ir = ir_L_per_d if ir_L_per_d is not None else 2.0
            # derive conc_ugL = (edi_mgkgd * bw_kg * 1e3) / ir
            conc_ugL_est = (float(edi_mgkgd) * float(bw_kg) * 1e3) / float(ir)
            return bll_onecomp_from_water(conc_ugL_est, ir, bw_kg, params=params)
        raise ValueError("onecomp engine requires either (conc_ugL, ir_L_per_d, bw_kg) or (edi_mgkgd, bw_kg)")
    elif method == "slope":
        if edi_mgkgd is None:
            raise ValueError("slope engine requires edi_mgkgd")
        # slope_ugdl_per_ugday may be provided as ug/dL per ug/day intake; if user provided in mg/kg-d scale,
        # it's their responsibility to convert - we accept slope_ugdl_per_ugday here.
        if slope_ugdl_per_ugday is None:
            raise ValueError("slope engine requires slope_ugdl_per_ugday")
        # convert edi mg/kg-d to intake ug/d if bw provided, else use edi directly scaled per kg
        return bll_linear_from_intake(edi_mgkgd, bw_kg or 70.0, slope_ugdl_per_ugday, f_abs=1.0, background_ugdl=background_ugdl)
    else:
        raise ValueError(f"Unknown BLL engine method: {method}")


def compute_bll_auto(group_name: str, *, edi_mgkgd: Optional[float] = None, conc_ugL: Optional[float] = None,
                     ir_L_per_d: Optional[float] = None, bw_kg: Optional[float] = None, 
                     background_ugdl: float = 0.0, config: Optional[PopulationEngineConfig] = None,
                     onecomp_params: Optional[OneCompParams] = None) -> tuple[float, str, Dict[str, Any]]:
    """
    Automatically compute BLL using population-appropriate pharmacokinetic model.
    
    Args:
        group_name: Population group name (e.g., "Adults", "Children", "Pregnant")
        edi_mgkgd: Estimated daily intake (mg/kg-day)
        conc_ugL: Water concentration (µg/L) - alternative to edi_mgkgd
        ir_L_per_d: Ingestion rate (L/day) - required with conc_ugL
        bw_kg: Body weight (kg)
        background_ugdl: Background BLL (µg/dL)
        config: Population engine configuration
        onecomp_params: Base parameters for one-compartment model
        
    Returns:
        Tuple of (BLL_value, engine_used, metadata)
    """
    if config is None:
        config = PopulationEngineConfig()
    
    # Classify population group
    pop_class = classify_population_group(group_name)
    
    # Metadata for tracking
    metadata = {
        "group_name": group_name,
        "population_class": pop_class,
        "background_ugdl": background_ugdl
    }
    
    if pop_class == "adult":
        # Use mechanistic one-compartment model for adults
        engine_used = "onecomp"
        params = get_group_specific_onecomp_params(group_name, onecomp_params)
        metadata.update({
            "engine": engine_used,
            "f_abs": params.f_abs,
            "t_half_days": params.t_half_days,
            "blood_vol_per_kg": params.blood_vol_per_kg,
            "rationale": "Mechanistic PK model appropriate for established adult physiology"
        })
        
        bll_value = compute_bll(
            method="onecomp",
            edi_mgkgd=edi_mgkgd,
            conc_ugL=conc_ugL,
            ir_L_per_d=ir_L_per_d,
            bw_kg=bw_kg,
            params=params,
            background_ugdl=background_ugdl
        )
        
    else:
        # Use empirical dose-response for vulnerable populations
        engine_used = "slope"
        slope = get_group_specific_slope(group_name, config)
        metadata.update({
            "engine": engine_used,
            "slope_ugdl_per_ugday": slope,
            "rationale": "Empirical dose-response model appropriate for vulnerable populations with uncertain physiology"
        })
        
        bll_value = compute_bll(
            method="slope",
            edi_mgkgd=edi_mgkgd,
            slope_ugdl_per_ugday=slope,
            bw_kg=bw_kg,
            background_ugdl=background_ugdl
        )
    
    return bll_value, engine_used, metadata


def compute_bll_batch_auto(groups: Dict[str, Dict], edi_data: Dict[str, float], 
                          config: Optional[PopulationEngineConfig] = None) -> Dict[str, Dict[str, Any]]:
    """
    Compute BLL for multiple groups using automatic engine selection.
    
    Args:
        groups: Dictionary of group info (from demographics.GROUP_INFO)
        edi_data: Dictionary mapping group names to EDI values (mg/kg-day)
        config: Population engine configuration
        
    Returns:
        Dictionary with BLL results, engines used, and metadata per group
    """
    results = {}
    
    for group_name, group_info in groups.items():
        if group_name not in edi_data:
            continue
            
        edi = edi_data[group_name]
        bw = group_info.get("BW", 70.0)
        ir = group_info.get("IR", 2.0)
        
        bll_value, engine_used, metadata = compute_bll_auto(
            group_name=group_name,
            edi_mgkgd=edi,
            bw_kg=bw,
            ir_L_per_d=ir,
            config=config
        )
        
        results[group_name] = {
            "BLL_ugdL": bll_value,
            "engine_used": engine_used,
            "EDI_mgkgd": edi,
            "BW_kg": bw,
            "IR_Ld": ir,
            "metadata": metadata
        }
    
    return results
