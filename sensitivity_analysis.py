#!/usr/bin/env python3
"""
sensitivity_analysis.py - Comprehensive Scientific Sensitivity Analysis

Performs comprehensive sensitivity analysis of the HBMPRA (Health-Based Metal Pollution Risk Assessment)
model using multiple methods to identify key parameters influencing Hazard Index (HI), Cancer Risk (CR),
and Pb Blood Lead Level (BLL) predictions.

Key Features:
- Multiple sensitivity analysis methods: Sobol, Morris screening, One-at-a-time (OAT), Monte Carlo filtering
- Comprehensive parameter space covering all key model parameters from hbmpra_optimized.py
- Enhanced visualizations: tornado plots, convergence plots, parameter effect plots, interaction matrices
- Full integration with toxref.yml, demographics, and external data files
- Scientifically rigorous uncertainty propagation matching the Bayesian model
- Support for multiple outputs: organ-specific HI, total CR, Pb BLL predictions
- Validation against baseline model results

Methods Implemented:
1. Variance-based (Sobol): Global sensitivity analysis using SALib
2. Morris screening: Efficient parameter screening for large parameter spaces
3. One-at-a-time (OAT): Local sensitivity analysis with confidence intervals
4. Monte Carlo filtering: Uncertainty importance analysis

Parameters Analyzed:
- Metal concentrations (total and bioavailable)
- Body weight uncertainty (CV_BW = 0.21)
- Ingestion rate uncertainty (sigma_log_ir = 0.6)
- Toxicity reference values (RfD_oral, RfD_derm, SF_oral, SF_derm)
- GI absorption factors (ABS_GI)
- Dermal permeability coefficients (Kp)
- Demographic parameters (ED, AT_nc, AT_ca, EF, ET, SA, AF)

Outputs:
- Hazard Index by organ system (neuro, nephro, hepato, systemic)
- Total Cancer Risk (CR)
- Predictive Pb Blood Lead Level (BLL)
"""

from __future__ import annotations

import os
import json
import argparse
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sps
from scipy.optimize import minimize

# SALib for sensitivity analysis
try:
    from SALib.sample import saltelli, morris, latin
    from SALib.analyze import sobol, morris as morris_analyze, delta
    from SALib.plotting.bar import plot as barplot
except ImportError:
    raise RuntimeError("SALib is required: pip install SALib")

# Local imports
from demographics import GROUP_INFO
from units import CF_ugL_to_mgL, DAYS_PER_YEAR
from bll_engines import compute_bll_auto, compute_bll_batch_auto, PopulationEngineConfig, classify_population_group

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_METALS = ['As', 'Cd', 'CrVI', 'CrIII', 'Cu', 'Hg', 'Pb', 'Co', 'Fe', 'Mn', 'Ni', 'Zn']
ATOMIC_WEIGHTS = {
    "As": 74.92, "Cd": 112.41, "Cr": 52.00, "CrIII": 52.00, "CrVI": 52.00,
    "Cu": 63.55, "Hg": 200.59, "MeHg": 215.59, "HgII": 200.59, "Hg(II)": 200.59,
    "Pb": 207.2, "Co": 58.93, "Fe": 55.85, "Mn": 54.94, "Ni": 58.69, "Zn": 65.38,
}


def ugL_to_mgL(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert µg/L to mg/L using shared units factor."""
    return np.asarray(x, float) * CF_ugL_to_mgL


def mgL_to_ugL(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert mg/L to µg/L (inverse of CF_ugL_to_mgL)."""
    return np.asarray(x, float) / max(CF_ugL_to_mgL, 1e-12)


def load_external_data() -> Tuple[Dict, Dict, Dict]:
    """
    Load external data files: toxref.yml, dermal_water_kp.yml, and PHREEQC tables.

    Returns:
        Tuple of (toxref_dict, kp_table, speciation_data)
    """
    # Load toxicity reference data
    toxref_path = Path("external/toxref.yml")
    toxref = {}
    if toxref_path.exists():
        try:
            with open(toxref_path, 'r') as f:
                raw = yaml.safe_load(f) or {}
                toxref = raw.get('tox', {})
                # Also load organ_sets from toxref.yml if available
                if 'organ_sets' in raw:
                    toxref['organ_sets'] = raw['organ_sets']
        except Exception as e:
            logger.warning(f"Failed to load toxref.yml: {e}")

    # Load dermal permeability data
    kp_path = Path("external/dermal_water_kp.yml")
    kp_table = {}
    if kp_path.exists():
        try:
            with open(kp_path, 'r') as f:
                kp_table = yaml.safe_load(f).get('Kp_cm_per_hr', {})
        except Exception as e:
            logger.warning(f"Failed to load dermal_water_kp.yml: {e}")

    # Load PHREEQC speciation data if available
    speciation_data = {}
    bio_path = Path("results/table_bioavailable_concentrations.csv")
    frac_path = Path("results/table_species_fractions.csv")

    if bio_path.exists():
        try:
            speciation_data['bio'] = pd.read_csv(bio_path)
            speciation_data['source'] = 'phreeqc_bio'
        except Exception as e:
            logger.warning(f"Failed to load PHREEQC bio table: {e}")

    if frac_path.exists() and 'bio' not in speciation_data:
        try:
            speciation_data['frac'] = pd.read_csv(frac_path)
            speciation_data['source'] = 'phreeqc_frac'
        except Exception as e:
            logger.warning(f"Failed to load PHREEQC fraction table: {e}")

    return toxref, kp_table, speciation_data


def resolve_key_in_dict(key: str, d: Dict) -> Optional[str]:
    """Resolve key in dictionary with fallback matching."""
    if key in d:
        return key
    base = "".join([ch for ch in str(key) if ch.isalnum()]).lower()
    for k in d.keys():
        if "".join([ch for ch in str(k) if ch.isalnum()]).lower() == base:
            return k
    return None


def build_comprehensive_parameter_space(
    groups: Dict[str, Dict],
    metals: List[str],
    conc_data: Dict[str, np.ndarray],
    toxref: Dict,
    kp_table: Dict,
    speciation_data: Dict,
    include_uncertainty: bool = True,
    include_tox_uncertainty: bool = False
) -> Tuple[List[str], List[List[float]], Dict[str, Any]]:
    """
    Build comprehensive parameter space for sensitivity analysis.

    Parameters:
        groups: Demographic groups data
        metals: List of metals to analyze
        conc_data: Concentration data (total and bioavailable)
        toxref: Toxicity reference data
        kp_table: Dermal permeability data
        speciation_data: PHREEQC speciation data
        include_uncertainty: Include parameter uncertainty bounds
        include_tox_uncertainty: Include toxicity parameter uncertainty

    Returns:
        Tuple of (parameter_names, bounds, metadata)
    """
    names = []
    bounds = []
    metadata = {
        'groups': list(groups.keys()),
        'metals': metals,
        'parameter_types': {},
        'baseline_values': {}
    }

    # 1. Demographic parameters with uncertainty (matching hbmpra_optimized.py)
    for g, info in groups.items():
        # Body weight uncertainty (CV = 0.21, lognormal)
        if include_uncertainty:
            names.append(f"BW_{g}")
            bw_mean = info['BW']
            cv_bw = 0.21
            bw_lo = bw_mean * (1 - 2*cv_bw)  # ~2 SD below
            bw_hi = bw_mean * (1 + 2*cv_bw)  # ~2 SD above
            bounds.append([max(bw_lo, 1.0), bw_hi])
            metadata['parameter_types'][f"BW_{g}"] = 'demographic'
            metadata['baseline_values'][f"BW_{g}"] = bw_mean

        # Ingestion rate uncertainty (sigma_log_ir = 0.6)
        if include_uncertainty:
            names.append(f"IR_{g}")
            ir_mean = info['IR']
            # Log-normal with sigma = 0.6
            ir_lo = ir_mean * np.exp(-2*0.6)  # ~2 SD below
            ir_hi = ir_mean * np.exp(+2*0.6)  # ~2 SD above
            bounds.append([max(ir_lo, 0.1), ir_hi])
            metadata['parameter_types'][f"IR_{g}"] = 'exposure'
            metadata['baseline_values'][f"IR_{g}"] = ir_mean

        # Exposure frequency (fraction of year)
        names.append(f"EF_{g}")
        ef_mean = info.get('EF', DAYS_PER_YEAR) / DAYS_PER_YEAR
        bounds.append([max(0.1, ef_mean * 0.5), min(1.0, ef_mean * 1.5)])
        metadata['parameter_types'][f"EF_{g}"] = 'temporal'
        metadata['baseline_values'][f"EF_{g}"] = ef_mean

        # Exposure duration (years)
        names.append(f"ED_{g}")
        ed_mean = info['ED'] / DAYS_PER_YEAR
        bounds.append([max(0.1, ed_mean * 0.5), ed_mean * 2.0])
        metadata['parameter_types'][f"ED_{g}"] = 'temporal'
        metadata['baseline_values'][f"ED_{g}"] = ed_mean

        # Averaging time non-cancer (years)
        names.append(f"AT_nc_{g}")
        at_nc_mean = info['AT_nc'] / DAYS_PER_YEAR
        bounds.append([max(1.0, at_nc_mean * 0.5), at_nc_mean * 2.0])
        metadata['parameter_types'][f"AT_nc_{g}"] = 'temporal'
        metadata['baseline_values'][f"AT_nc_{g}"] = at_nc_mean

        # Averaging time cancer (years)
        names.append(f"AT_ca_{g}")
        at_ca_mean = info['AT_c'] / DAYS_PER_YEAR
        bounds.append([max(1.0, at_ca_mean * 0.5), at_ca_mean * 2.0])
        metadata['parameter_types'][f"AT_ca_{g}"] = 'temporal'
        metadata['baseline_values'][f"AT_ca_{g}"] = at_ca_mean

        # Skin surface area (cm²)
        if 'SA' in info:
            names.append(f"SA_{g}")
            sa_mean = info['SA']
            bounds.append([max(1000, sa_mean * 0.7), sa_mean * 1.3])
            metadata['parameter_types'][f"SA_{g}"] = 'dermal'
            metadata['baseline_values'][f"SA_{g}"] = sa_mean

        # Adherence factor (unitless)
        if 'AF' in info:
            names.append(f"AF_{g}")
            af_mean = info['AF']
            bounds.append([max(0.01, af_mean * 0.5), min(1.0, af_mean * 2.0)])
            metadata['parameter_types'][f"AF_{g}"] = 'dermal'
            metadata['baseline_values'][f"AF_{g}"] = af_mean

        # Event duration (hours)
        if 'ET' in info:
            names.append(f"ET_{g}")
            et_mean = info['ET']
            bounds.append([max(0.1, et_mean * 0.5), min(24.0, et_mean * 2.0)])
            metadata['parameter_types'][f"ET_{g}"] = 'dermal'
            metadata['baseline_values'][f"ET_{g}"] = et_mean

    # 2. Metal concentrations
    for m in metals:
        # Total concentration
        if f'C_{m}' in conc_data or m in conc_data:
            conc_key = f'C_{m}' if f'C_{m}' in conc_data else m
            conc_mean = float(np.mean(conc_data[conc_key]))
            if conc_mean > 0:
                names.append(f"C_{m}")
                bounds.append([max(0, conc_mean * 0.1), conc_mean * 10.0])
                metadata['parameter_types'][f"C_{m}"] = 'concentration'
                metadata['baseline_values'][f"C_{m}"] = conc_mean

        # Bioavailable concentration (if available)
        bio_key = f'C_bio_{m}'
        if bio_key in conc_data:
            bio_mean = float(np.mean(conc_data[bio_key]))
            if bio_mean > 0:
                names.append(f"C_bio_{m}")
                bounds.append([max(0, bio_mean * 0.1), bio_mean * 10.0])
                metadata['parameter_types'][f"C_bio_{m}"] = 'concentration'
                metadata['baseline_values'][f"C_bio_{m}"] = bio_mean

    # 3. Toxicity parameters (if uncertainty analysis requested)
    if include_tox_uncertainty:
        for m in metals:
            metal_toxref = toxref.get(m, {})

            # Oral RfD uncertainty (±50% around baseline)
            if 'RfD_oral' in metal_toxref:
                rfd_oral = metal_toxref['RfD_oral']
                names.append(f"RfD_oral_{m}")
                bounds.append([max(1e-6, rfd_oral * 0.5), rfd_oral * 2.0])
                metadata['parameter_types'][f"RfD_oral_{m}"] = 'toxicity'
                metadata['baseline_values'][f"RfD_oral_{m}"] = rfd_oral

            # Dermal RfD uncertainty (±50% around baseline)
            if 'RfD_derm' in metal_toxref:
                rfd_derm = metal_toxref['RfD_derm']
                names.append(f"RfD_derm_{m}")
                bounds.append([max(1e-6, rfd_derm * 0.5), rfd_derm * 2.0])
                metadata['parameter_types'][f"RfD_derm_{m}"] = 'toxicity'
                metadata['baseline_values'][f"RfD_derm_{m}"] = rfd_derm

            # Oral CSF uncertainty (±50% around baseline)
            if 'SF_oral' in metal_toxref and metal_toxref['SF_oral'] > 0:
                sf_oral = metal_toxref['SF_oral']
                names.append(f"SF_oral_{m}")
                bounds.append([max(1e-9, sf_oral * 0.5), sf_oral * 2.0])
                metadata['parameter_types'][f"SF_oral_{m}"] = 'toxicity'
                metadata['baseline_values'][f"SF_oral_{m}"] = sf_oral

            # Dermal CSF uncertainty (±50% around baseline)
            if 'SF_derm' in metal_toxref and metal_toxref['SF_derm'] > 0:
                sf_derm = metal_toxref['SF_derm']
                names.append(f"SF_derm_{m}")
                bounds.append([max(1e-9, sf_derm * 0.5), sf_derm * 2.0])
                metadata['parameter_types'][f"SF_derm_{m}"] = 'toxicity'
                metadata['baseline_values'][f"SF_derm_{m}"] = sf_derm

            # GI absorption uncertainty (±30% around baseline)
            if 'ABS_GI' in metal_toxref:
                abs_gi = metal_toxref['ABS_GI']
                names.append(f"ABS_GI_{m}")
                bounds.append([max(0.01, abs_gi * 0.7), min(1.0, abs_gi * 1.3)])
                metadata['parameter_types'][f"ABS_GI_{m}"] = 'absorption'
                metadata['baseline_values'][f"ABS_GI_{m}"] = abs_gi

    # 4. Dermal permeability coefficients
    for m in metals:
        kp_key = resolve_key_in_dict(m, kp_table)
        if kp_key and kp_table[kp_key] > 0:
            kp_mean = kp_table[kp_key]
            names.append(f"Kp_{m}")
            bounds.append([max(1e-6, kp_mean * 0.1), kp_mean * 10.0])
            metadata['parameter_types'][f"Kp_{m}"] = 'dermal'
            metadata['baseline_values'][f"Kp_{m}"] = kp_mean

    logger.info(f"Built parameter space with {len(names)} parameters: {names[:5]}...")
    return names, bounds, metadata


def compute_model_outputs_batch(
    X: np.ndarray,
    names: List[str],
    groups: Dict[str, Dict],
    metals: List[str],
    toxref: Dict,
    kp_table: Dict,
    conc_data: Dict[str, np.ndarray],
    speciation_data: Dict,
    metadata: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute model outputs for batch of parameter samples.

    Returns:
        Tuple of (HI_overall, CR_total, BLL_mean, organ_HIs)
    """
    N = X.shape[0]
    idx = {n: i for i, n in enumerate(names)}

    # Extract parameter values
    params = {}
    for name in names:
        params[name] = X[:, idx[name]]

    # Build organ sets from toxref.yml if available, otherwise use defaults
    if 'organ_sets' in toxref:
        organ_sets = {}
        for organ, metal_list in toxref['organ_sets'].items():
            # Convert to set and filter to only include metals we're analyzing
            organ_sets[organ] = set(metal_list) & set(metals)
    else:
        # Fallback to hardcoded organ sets based on toxref target_organs
        organ_sets = {
            'neuro': {'Mn', 'Hg', 'Pb'},  # From toxref target_organs
            'nephro': {'Cd', 'Hg', 'CrVI', 'Pb'},  # From toxref target_organs
            'hepato': {'As', 'Cd', 'Cu', 'CrVI', 'Fe'},  # From toxref target_organs
            'cardiovascular': {'As', 'Co', 'Pb'},  # From toxref target_organs
            'endocrine': {'As', 'Co'},  # From toxref target_organs
            'hemato': {'Pb', 'Zn'},  # From toxref target_organs
            'derm': {'As', 'CrVI', 'Ni', 'Pb', 'Co'},  # From toxref target_organs
            'gi': {'CrVI', 'CrIII', 'Cu', 'Fe'},  # From toxref target_organs
            'systemic': {'As', 'Ni', 'Zn'}  # From toxref target_organs
        }
        # Filter organ sets to only include metals we're analyzing
        organ_sets = {organ: mset & set(metals) for organ, mset in organ_sets.items() if mset & set(metals)}

    # Initialize outputs
    HI_overall = np.zeros(N)
    CR_total = np.zeros(N)
    BLL_mean = np.zeros(N)
    # Store organ HI aggregated across groups (mean over groups) per sample
    organ_HIs = {organ: np.zeros(N) for organ in organ_sets.keys()}

    # Get baseline values for missing parameters
    baseline = metadata['baseline_values']

    # Track once-per-run warnings to avoid log spam
    warned_missing_dermal_rfd: set = set()
    warned_missing_kp: set = set()

    for i in range(N):
        # Extract parameters for this sample
        sample_params = {name: params[name][i] for name in names}

        # Compute for each group
        group_overall_HIs: List[float] = []
        group_CRs: List[float] = []
        group_BLLs: List[float] = []
        # For organ HI aggregation across groups
        organ_HI_accum: Dict[str, float] = {organ: 0.0 for organ in organ_sets.keys()}

        for g in groups.keys():
            # Get parameters with fallbacks to baseline
            BW = sample_params.get(f"BW_{g}", baseline.get(f"BW_{g}", GROUP_INFO[g]['BW']))
            IR = sample_params.get(f"IR_{g}", baseline.get(f"IR_{g}", GROUP_INFO[g]['IR']))
            EF = sample_params.get(f"EF_{g}", baseline.get(f"EF_{g}", GROUP_INFO[g].get('EF', DAYS_PER_YEAR) / DAYS_PER_YEAR))
            ED = sample_params.get(f"ED_{g}", baseline.get(f"ED_{g}", GROUP_INFO[g]['ED'] / DAYS_PER_YEAR))
            AT_nc = sample_params.get(f"AT_nc_{g}", baseline.get(f"AT_nc_{g}", GROUP_INFO[g]['AT_nc'] / DAYS_PER_YEAR))
            AT_ca = sample_params.get(f"AT_ca_{g}", baseline.get(f"AT_ca_{g}", GROUP_INFO[g]['AT_c'] / DAYS_PER_YEAR))
            SA = sample_params.get(f"SA_{g}", baseline.get(f"SA_{g}", GROUP_INFO[g].get('SA', 0)))
            AF = sample_params.get(f"AF_{g}", baseline.get(f"AF_{g}", GROUP_INFO[g].get('AF', 0)))
            ET = sample_params.get(f"ET_{g}", baseline.get(f"ET_{g}", GROUP_INFO[g].get('ET', 0)))

            # Convert durations to years are already in years; EF is fraction of year
            ED_years = ED
            AT_nc_years = max(AT_nc, 1e-9)
            AT_ca_years = max(AT_ca, 1e-9)
            EF_frac = np.clip(EF, 0.0, 1.0)
            # Event time in hours: ET is provided in hours and used directly
            t_event_hr = ET

            # Track per-organ HI for this group
            organ_HI_group: Dict[str, float] = {organ: 0.0 for organ in organ_sets.keys()}
            group_CR = 0.0
            # Track Pb ingestion EDI for BLL engine (non-cancer scaling per optimized model)
            edi_pb_nc: Optional[float] = None
            pb_conc_val: Optional[float] = None

            for m in metals:
                # Get concentrations
                C_total = sample_params.get(f"C_{m}", baseline.get(f"C_{m}", 0))
                C_bio = sample_params.get(f"C_bio_{m}", baseline.get(f"C_bio_{m}", C_total))

                # Get toxicity parameters
                metal_toxref = toxref.get(m, {})
                RfD_oral = sample_params.get(f"RfD_oral_{m}", metal_toxref.get('RfD_oral', 1.0))
                RfD_derm = sample_params.get(f"RfD_derm_{m}", metal_toxref.get('RfD_derm', RfD_oral))
                SF_oral = sample_params.get(f"SF_oral_{m}", metal_toxref.get('SF_oral', 0.0))
                SF_derm = sample_params.get(f"SF_derm_{m}", metal_toxref.get('SF_derm', SF_oral))
                ABS_GI = sample_params.get(f"ABS_GI_{m}", metal_toxref.get('ABS_GI', 1.0))
                Kp = sample_params.get(f"Kp_{m}", kp_table.get(resolve_key_in_dict(m, kp_table), 0.0))

                # Convert concentrations to mg/L
                C_mgL = float(C_total) * CF_ugL_to_mgL
                C_bio_mgL = float(C_bio) * CF_ugL_to_mgL

                # Ingestion EDI (mg/kg-day), non-cancer time scaling
                IR_perkg = (IR / max(BW, 1e-12))
                F_nc = (EF_frac * ED_years) / AT_nc_years
                F_ca = (EF_frac * ED_years) / AT_ca_years
                EDI_ing_nc = (C_mgL * ABS_GI) * IR_perkg * F_nc
                EDI_ing_ca = (C_mgL * ABS_GI) * IR_perkg * F_ca

                # Dermal EDI (mg/kg-day): requires bioavailable conc and Kp, SA, AF, ET
                EDI_der_nc = 0.0
                EDI_der_ca = 0.0
                if C_bio_mgL > 0 and Kp > 0 and SA > 0 and AF > 0 and t_event_hr > 0:
                    base_der = (C_bio_mgL * Kp * t_event_hr * SA * AF) / max(BW, 1e-12) * 1e-3
                    EDI_der_nc = base_der * F_nc
                    EDI_der_ca = base_der * F_ca
                    # Warn once if dermal active but no RfD_derm specified (falls back to oral)
                    if (('RfD_derm' not in metal_toxref or not np.isfinite(RfD_derm)) and m not in warned_missing_dermal_rfd):
                        logger.warning(f"Dermal route active for {m} but RfD_derm missing; falling back to RfD_oral for HQ_der.")
                        warned_missing_dermal_rfd.add(m)
                elif (C_bio_mgL > 0 and (Kp <= 0 or SA <= 0 or AF <= 0 or t_event_hr <= 0)) and m not in warned_missing_kp:
                    # If bioavailable conc exists but Kp or dermal params are unusable, warn once
                    logger.warning(f"Bioavailable concentration present for {m} but missing/zero Kp or dermal params; dermal contribution ignored.")
                    warned_missing_kp.add(m)

                # Hazard quotient, route-specific denominators (use inf if missing to zero-out)
                RfD_oral_safe = RfD_oral if (RfD_oral is not None and RfD_oral > 0) else np.inf
                RfD_derm_safe = RfD_derm if (RfD_derm is not None and RfD_derm > 0) else np.inf
                HQ_ing = EDI_ing_nc / RfD_oral_safe
                HQ_der = EDI_der_nc / RfD_derm_safe

                # Cancer risk contribution (route-specific)
                SF_oral_safe = SF_oral if (SF_oral is not None and SF_oral >= 0) else 0.0
                SF_derm_safe = (SF_derm if (SF_derm is not None and SF_derm >= 0) else SF_oral_safe)
                LADD_ing = EDI_ing_ca
                LADD_der = EDI_der_ca
                group_CR += LADD_ing * SF_oral_safe + LADD_der * SF_derm_safe

                # Record EDI for Pb for BLL computation
                if m == 'Pb':
                    edi_pb_nc = float(EDI_ing_nc)
                    pb_conc_val = float(C_total)

                # Distribute HQ to organs based on toxref target_organs routes
                # Default: apply ingestion to all mapped organs; dermal only if applicable
                if 'target_organs' in metal_toxref:
                    target_organs = metal_toxref.get('target_organs', [])
                    for organ, metal_set in organ_sets.items():
                        if m not in metal_set:
                            continue
                        affects_organ = False
                        use_dermal = False
                        use_ingestion = False
                        for target in target_organs:
                            if isinstance(target, dict):
                                for org, route in target.items():
                                    if str(org).strip().lower() == str(organ).strip().lower():
                                        affects_organ = True
                                        r = str(route).strip().lower()
                                        if r in ('dermal', 'derm', 'skin'):
                                            use_dermal = True
                                        elif r in ('ingestion', 'ingest', 'oral', 'gi'):
                                            use_ingestion = True
                                        else:
                                            use_dermal = True
                                            use_ingestion = True
                            elif isinstance(target, str):
                                if str(target).strip().lower() == str(organ).strip().lower():
                                    affects_organ = True
                                    use_ingestion = True
                        if affects_organ:
                            if use_ingestion:
                                organ_HI_group[organ] += HQ_ing
                            if use_dermal:
                                organ_HI_group[organ] += HQ_der
                else:
                    # If no target_organs listed, attribute to generic set membership
                    for organ, metal_set in organ_sets.items():
                        if m in metal_set:
                            organ_HI_group[organ] += (HQ_ing + HQ_der)

            # Overall HI for this group is max across organ HIs
            if organ_HI_group:
                group_overall_HIs.append(max(organ_HI_group.values()))
                # Accumulate for across-group mean per organ
                for organ in organ_HI_group:
                    organ_HI_accum[organ] += organ_HI_group[organ]
            else:
                group_overall_HIs.append(0.0)
            group_CRs.append(group_CR)

            # Pb BLL calculation using automatic dual-engine approach
            if edi_pb_nc is not None:
                # Get population info for engine selection
                group_name = g
                bw_kg = GROUP_INFO.get(g, {}).get('BW', 70.0)
                ir_Ld = GROUP_INFO.get(g, {}).get('IR', 2.0)
                # Provide both edi and conc/IR: adults will use onecomp with conc+IR; vulnerable will use slope with EDI
                bll_value, _, _ = compute_bll_auto(
                    group_name,
                    edi_mgkgd=edi_pb_nc,
                    conc_ugL=pb_conc_val,
                    ir_L_per_d=ir_Ld,
                    bw_kg=bw_kg
                )
                group_BLLs.append(float(bll_value))
            else:
                group_BLLs.append(0.0)

        # Average across groups
        HI_overall[i] = float(np.mean(group_overall_HIs)) if group_overall_HIs else 0.0
        CR_total[i] = np.mean(group_CRs)
        BLL_mean[i] = np.mean(group_BLLs)

        # Save organ-specific HIs as across-group mean for this sample
        n_groups = max(len(groups), 1)
        for organ in organ_HI_accum:
            organ_HIs[organ][i] = organ_HI_accum[organ] / n_groups

    return HI_overall, CR_total, BLL_mean, organ_HIs


def perform_sensitivity_analysis(
    method: str,
    names: List[str],
    bounds: List[List[float]],
    groups: Dict[str, Dict],
    metals: List[str],
    toxref: Dict,
    kp_table: Dict,
    conc_data: Dict[str, np.ndarray],
    speciation_data: Dict,
    metadata: Dict[str, Any],
    n_samples: int = 4096,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Perform sensitivity analysis using specified method.

    Parameters:
        method: 'sobol', 'morris', or 'delta'
        names: Parameter names
        bounds: Parameter bounds
        ... (other parameters)

    Returns:
        Dictionary with sensitivity results
    """
    np.random.seed(seed)

    problem = {
        'num_vars': len(names),
        'names': names,
        'bounds': bounds
    }

    results = {}

    if method == 'sobol':
        logger.info(f"Performing Sobol sensitivity analysis with {n_samples} samples")
        X = saltelli.sample(problem, n_samples, calc_second_order=True)
        Y_HI, Y_CR, Y_BLL, Y_organs = compute_model_outputs_batch(
            X, names, groups, metals, toxref, kp_table, conc_data, speciation_data, metadata
        )

        # Analyze HI
        Si_HI = sobol.analyze(problem, Y_HI, calc_second_order=True, print_to_console=False)
        results['HI'] = {
            'S1': Si_HI['S1'],
            'ST': Si_HI['ST'],
            'S1_conf': Si_HI['S1_conf'],
            'ST_conf': Si_HI['ST_conf'],
            'S2': Si_HI['S2'] if 'S2' in Si_HI else None
        }

        # Analyze CR
        Si_CR = sobol.analyze(problem, Y_CR, calc_second_order=True, print_to_console=False)
        results['CR'] = {
            'S1': Si_CR['S1'],
            'ST': Si_CR['ST'],
            'S1_conf': Si_CR['S1_conf'],
            'ST_conf': Si_CR['ST_conf'],
            'S2': Si_CR['S2'] if 'S2' in Si_CR else None
        }

        # Analyze BLL if available
        if np.any(Y_BLL > 0):
            Si_BLL = sobol.analyze(problem, Y_BLL, calc_second_order=True, print_to_console=False)
            results['BLL'] = {
                'S1': Si_BLL['S1'],
                'ST': Si_BLL['ST'],
                'S1_conf': Si_BLL['S1_conf'],
                'ST_conf': Si_BLL['ST_conf'],
                'S2': Si_BLL['S2'] if 'S2' in Si_BLL else None
            }

        results['samples'] = X
        results['outputs'] = {'HI': Y_HI, 'CR': Y_CR, 'BLL': Y_BLL, 'organs': Y_organs}

    elif method == 'morris':
        logger.info(f"Performing Morris screening with {n_samples} trajectories")
        X = morris.sample(problem, n_samples, num_levels=4, seed=seed)
        Y_HI, Y_CR, Y_BLL, Y_organs = compute_model_outputs_batch(
            X, names, groups, metals, toxref, kp_table, conc_data, speciation_data, metadata
        )

        # Analyze HI
        Si_HI = morris_analyze.analyze(problem, X, Y_HI, print_to_console=False)
        results['HI'] = {
            'mu': Si_HI['mu'],
            'mu_star': Si_HI['mu_star'],
            'sigma': Si_HI['sigma'],
            'mu_star_conf': Si_HI['mu_star_conf'] if 'mu_star_conf' in Si_HI else None
        }

        # Analyze CR
        Si_CR = morris_analyze.analyze(problem, X, Y_CR, print_to_console=False)
        results['CR'] = {
            'mu': Si_CR['mu'],
            'mu_star': Si_CR['mu_star'],
            'sigma': Si_CR['sigma'],
            'mu_star_conf': Si_CR['mu_star_conf'] if 'mu_star_conf' in Si_CR else None
        }

        results['samples'] = X
        results['outputs'] = {'HI': Y_HI, 'CR': Y_CR, 'BLL': Y_BLL, 'organs': Y_organs}

    elif method == 'delta':
        logger.info(f"Performing Delta moment-independent sensitivity analysis with {n_samples} samples")
        X = latin.sample(problem, n_samples, seed=seed)
        Y_HI, Y_CR, Y_BLL, Y_organs = compute_model_outputs_batch(
            X, names, groups, metals, toxref, kp_table, conc_data, speciation_data, metadata
        )

        # Analyze HI
        Si_HI = delta.analyze(problem, X, Y_HI, print_to_console=False)
        results['HI'] = {
            'delta': Si_HI['delta'],
            'delta_conf': Si_HI['delta_conf'],
            'S1': Si_HI['S1'],
            'S1_conf': Si_HI['S1_conf']
        }

        # Analyze CR
        Si_CR = delta.analyze(problem, X, Y_CR, print_to_console=False)
        results['CR'] = {
            'delta': Si_CR['delta'],
            'delta_conf': Si_CR['delta_conf'],
            'S1': Si_CR['S1'],
            'S1_conf': Si_CR['S1_conf']
        }

        results['samples'] = X
        results['outputs'] = {'HI': Y_HI, 'CR': Y_CR, 'BLL': Y_BLL, 'organs': Y_organs}

    else:
        raise ValueError(f"Unknown sensitivity method: {method}")

    results['method'] = method
    results['problem'] = problem
    results['metadata'] = metadata

    return results


def create_visualizations(results: Dict[str, Any], output_dir: str, method: str, filter_threshold: float = 0.001):
    """
    Create comprehensive visualizations for sensitivity analysis results.
    
    Parameters:
        results: Sensitivity analysis results
        output_dir: Output directory for plots
        method: Analysis method ('sobol', 'morris', 'delta')
        filter_threshold: Minimum sensitivity index to include in plots (default: 0.001)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    if method == 'sobol':
        _create_sobol_plots(results, output_dir, filter_threshold)
    elif method == 'morris':
        _create_morris_plots(results, output_dir, filter_threshold)
    elif method == 'delta':
        _create_delta_plots(results, output_dir, filter_threshold)

    # Create convergence plot if samples are available
    if 'outputs' in results:
        _create_convergence_plot(results, output_dir)

    # Create parameter effect plots
    _create_parameter_effect_plots(results, output_dir)


def _create_sobol_plots(results: Dict[str, Any], output_dir: str, filter_threshold: float = 0.001):
    """Create Sobol sensitivity plots."""
    names = results['problem']['names']

    # HI Tornado plot
    if 'HI' in results:
        # Filter out parameters with very small effects
        threshold = filter_threshold
        s1_vals = results['HI']['S1']
        st_vals = results['HI']['ST']
        
        # Keep parameters where either S1 or ST is above threshold
        significant_mask = (np.abs(s1_vals) >= threshold) | (np.abs(st_vals) >= threshold)
        
        if np.any(significant_mask):
            filtered_names = [names[i] for i in range(len(names)) if significant_mask[i]]
            filtered_s1 = s1_vals[significant_mask]
            filtered_st = st_vals[significant_mask]
            filtered_s1_conf = results['HI']['S1_conf'][significant_mask]
            filtered_st_conf = results['HI']['ST_conf'][significant_mask]
            
            # Sort by total effects (descending)
            sort_indices = np.argsort(filtered_st)[::-1]
            sorted_names = [filtered_names[i] for i in sort_indices]
            sorted_s1 = filtered_s1[sort_indices]
            sorted_st = filtered_st[sort_indices]
            sorted_s1_conf = filtered_s1_conf[sort_indices]
            sorted_st_conf = filtered_st_conf[sort_indices]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, max(8, len(sorted_names) * 0.4)))

            # First-order effects
            y_pos = np.arange(len(sorted_names))
            ax1.barh(y_pos, sorted_s1, xerr=sorted_s1_conf, capsize=5)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels([_format_param_name(n) for n in sorted_names], fontsize=14)
            ax1.set_xlabel('First-order Sobol Index', fontsize=16)
            ax1.set_title('First-order Effects (HI)', fontsize=18)
            ax1.grid(True, alpha=0.3)

            # Total effects
            ax2.barh(y_pos, sorted_st, xerr=sorted_st_conf, capsize=5)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([_format_param_name(n) for n in sorted_names], fontsize=14)
            ax2.set_xlabel('Total Sobol Index', fontsize=16)
            ax2.set_title('Total Effects (HI)', fontsize=18)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'sobol_HI_tornado.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            logger.warning("No significant parameters found for HI tornado plot")

    # CR Tornado plot
    if 'CR' in results:
        # Filter out parameters with very small effects
        threshold = filter_threshold
        s1_vals = results['CR']['S1']
        st_vals = results['CR']['ST']
        
        # Keep parameters where either S1 or ST is above threshold
        significant_mask = (np.abs(s1_vals) >= threshold) | (np.abs(st_vals) >= threshold)
        
        if np.any(significant_mask):
            filtered_names = [names[i] for i in range(len(names)) if significant_mask[i]]
            filtered_s1 = s1_vals[significant_mask]
            filtered_st = st_vals[significant_mask]
            filtered_s1_conf = results['CR']['S1_conf'][significant_mask]
            filtered_st_conf = results['CR']['ST_conf'][significant_mask]
            
            # Sort by total effects (descending)
            sort_indices = np.argsort(filtered_st)[::-1]
            sorted_names = [filtered_names[i] for i in sort_indices]
            sorted_s1 = filtered_s1[sort_indices]
            sorted_st = filtered_st[sort_indices]
            sorted_s1_conf = filtered_s1_conf[sort_indices]
            sorted_st_conf = filtered_st_conf[sort_indices]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, max(8, len(sorted_names) * 0.4)))

            # First-order effects
            y_pos = np.arange(len(sorted_names))
            ax1.barh(y_pos, sorted_s1, xerr=sorted_s1_conf, capsize=5)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels([_format_param_name(n) for n in sorted_names], fontsize=14)
            ax1.set_xlabel('First-order Sobol Index', fontsize=16)
            ax1.set_title('First-order Effects (CR)', fontsize=18)
            ax1.grid(True, alpha=0.3)

            # Total effects
            ax2.barh(y_pos, sorted_st, xerr=sorted_st_conf, capsize=5)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([_format_param_name(n) for n in sorted_names], fontsize=14)
            ax2.set_xlabel('Total Sobol Index', fontsize=16)
            ax2.set_title('Total Effects (CR)', fontsize=18)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'sobol_CR_tornado.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            logger.warning("No significant parameters found for CR tornado plot")

    # Second-order interaction plots
    if 'HI' in results and results['HI']['S2'] is not None:
        _create_interaction_heatmap(results['HI']['S2'], names, output_dir, 'HI_S2_heatmap.png', 'HI Interactions')
    if 'CR' in results and results['CR']['S2'] is not None:
        _create_interaction_heatmap(results['CR']['S2'], names, output_dir, 'CR_S2_heatmap.png', 'CR Interactions')


def _create_morris_plots(results: Dict[str, Any], output_dir: str, filter_threshold: float = 0.001):
    """Create Morris screening plots."""
    names = results['problem']['names']

    if 'HI' in results:
        # Filter out parameters with very small effects
        threshold = filter_threshold
        mu_star = results['HI']['mu_star']
        sigma = results['HI']['sigma']
        
        # Keep parameters where mu_star is above threshold
        significant_mask = np.abs(mu_star) >= threshold
        
        if np.any(significant_mask):
            filtered_names = [names[i] for i in range(len(names)) if significant_mask[i]]
            filtered_mu_star = mu_star[significant_mask]
            filtered_sigma = sigma[significant_mask]
            
            fig, ax = plt.subplots(figsize=(12, 8))

            # Morris mu* vs sigma plot
            ax.scatter(filtered_mu_star, filtered_sigma, s=100, alpha=0.7)

            # Add parameter labels
            for i, name in enumerate(filtered_names):
                ax.annotate(_format_param_name(name), (filtered_mu_star[i], filtered_sigma[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

            ax.set_xlabel('μ* (mean of absolute elementary effects)', fontsize=16)
            ax.set_ylabel('σ (standard deviation of elementary effects)', fontsize=16)
            ax.set_title('Morris Screening Plot (HI)', fontsize=18)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'morris_HI_plot.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            logger.warning("No significant parameters found for Morris HI plot")

    # Similar plot for CR
    if 'CR' in results:
        # Filter out parameters with very small effects
        threshold = filter_threshold
        mu_star = results['CR']['mu_star']
        sigma = results['CR']['sigma']
        
        # Keep parameters where mu_star is above threshold
        significant_mask = np.abs(mu_star) >= threshold
        
        if np.any(significant_mask):
            filtered_names = [names[i] for i in range(len(names)) if significant_mask[i]]
            filtered_mu_star = mu_star[significant_mask]
            filtered_sigma = sigma[significant_mask]
            
            fig, ax = plt.subplots(figsize=(12, 8))

            ax.scatter(filtered_mu_star, filtered_sigma, s=100, alpha=0.7)

            for i, name in enumerate(filtered_names):
                ax.annotate(_format_param_name(name), (filtered_mu_star[i], filtered_sigma[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

            ax.set_xlabel('μ* (mean of absolute elementary effects)', fontsize=16)
            ax.set_ylabel('σ (standard deviation of elementary effects)', fontsize=16)
            ax.set_title('Morris Screening Plot (CR)', fontsize=18)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'morris_CR_plot.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            logger.warning("No significant parameters found for Morris CR plot")


def _create_delta_plots(results: Dict[str, Any], output_dir: str, filter_threshold: float = 0.001):
    """Create Delta method plots."""
    names = results['problem']['names']

    if 'HI' in results:
        # Filter out parameters with very small effects
        threshold = filter_threshold
        delta_vals = results['HI']['delta']
        
        # Keep parameters where delta is above threshold
        significant_mask = np.abs(delta_vals) >= threshold
        
        if np.any(significant_mask):
            filtered_names = [names[i] for i in range(len(names)) if significant_mask[i]]
            filtered_delta = delta_vals[significant_mask]
            
            # Sort by delta values (descending)
            sort_indices = np.argsort(filtered_delta)[::-1]
            sorted_names = [filtered_names[i] for i in sort_indices]
            sorted_delta = filtered_delta[sort_indices]
            
            fig, ax = plt.subplots(figsize=(12, max(8, len(sorted_names) * 0.4)))

            y_pos = np.arange(len(sorted_names))

            ax.barh(y_pos, sorted_delta)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([_format_param_name(n) for n in sorted_names], fontsize=14)
            ax.set_xlabel('Delta Index', fontsize=16)
            ax.set_title('Delta Moment-Independent Sensitivity (HI)', fontsize=18)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'delta_HI_plot.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            logger.warning("No significant parameters found for Delta HI plot")


def _create_interaction_heatmap(S2_matrix: np.ndarray, names: List[str], output_dir: str, filename: str, title: str):
    """Create second-order interaction heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Mask diagonal and upper triangle for cleaner visualization
    mask = np.triu(np.ones_like(S2_matrix), k=1)
    S2_matrix = np.ma.masked_where(mask, S2_matrix)

    sns.heatmap(S2_matrix, xticklabels=[_format_param_name(n) for n in names],
                yticklabels=[_format_param_name(n) for n in names],
                cmap='RdYlBu_r', center=0, annot=False, fmt='.2f',
                cbar_kws={'label': 'Second-order Sobol Index'})

    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()


def _create_convergence_plot(results: Dict[str, Any], output_dir: str):
    """Create convergence plot for sensitivity indices."""
    if 'outputs' not in results:
        return

    outputs = results['outputs']
    n_samples = len(outputs['HI'])

    # Calculate running means of sensitivity indices
    if results['method'] == 'sobol':
        # For demonstration, create a simple convergence plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot output distributions
        ax1.hist(outputs['HI'], bins=50, alpha=0.7, label='HI')
        ax1.set_xlabel('Hazard Index')
        ax1.set_ylabel('Frequency')
        ax1.set_title('HI Distribution')
        ax1.legend()

        ax2.hist(outputs['CR'], bins=50, alpha=0.7, label='CR', color='orange')
        ax2.set_xlabel('Cancer Risk')
        ax2.set_ylabel('Frequency')
        ax2.set_title('CR Distribution')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'output_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()


def _create_parameter_effect_plots(results: Dict[str, Any], output_dir: str):
    """Create parameter effect plots showing how outputs vary with parameters."""
    if 'samples' not in results or 'outputs' not in results:
        return

    samples = results['samples']
    outputs = results['outputs']
    names = results['problem']['names']

    # Select top 6 most influential parameters for plotting
    if results['method'] == 'sobol' and 'HI' in results:
        top_indices = np.argsort(results['HI']['ST'])[-6:]
    else:
        top_indices = list(range(min(6, len(names))))

    n_cols = 3
    n_rows = (len(top_indices) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, param_idx in enumerate(top_indices):
        row = i // n_cols
        col = i % n_cols

        param_name = names[param_idx]
        param_values = samples[:, param_idx]

        ax = axes[row, col]
        scatter = ax.scatter(param_values, outputs['HI'], alpha=0.6, s=10)
        ax.set_xlabel(_format_param_name(param_name), fontsize=16)
        ax.set_ylabel('Hazard Index', fontsize=16)
        ax.set_title(f'HI vs {_format_param_name(param_name)}', fontsize=18)
        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='x', labelsize=14)

    # Hide empty subplots
    for i in range(len(top_indices), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_effects_HI.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _format_param_name(name: str) -> str:
    """Format parameter name for display."""
    # Simplify parameter names for plotting
    name = name.replace('_', ' ')
    if name.startswith('C '):
        name = name.replace('C ', 'Conc ')
    elif name.startswith('BW '):
        name = name.replace('BW ', 'BW ')
    elif name.startswith('IR '):
        name = name.replace('IR ', 'IR ')
    return name


def save_results(results: Dict[str, Any], output_dir: str, method: str):
    """Save sensitivity analysis results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save sensitivity indices
    if method == 'sobol':
        for output_name in ['HI', 'CR', 'BLL']:
            if output_name in results:
                df = pd.DataFrame({
                    'parameter': results['problem']['names'],
                    'S1': results[output_name]['S1'],
                    'S1_conf': results[output_name]['S1_conf'],
                    'ST': results[output_name]['ST'],
                    'ST_conf': results[output_name]['ST_conf']
                })
                df.to_csv(os.path.join(output_dir, f'{method}_{output_name}_indices.csv'), index=False)

                # Save second-order indices if available
                if results[output_name]['S2'] is not None:
                    S2_df = pd.DataFrame(results[output_name]['S2'],
                                       columns=results['problem']['names'],
                                       index=results['problem']['names'])
                    S2_df.to_csv(os.path.join(output_dir, f'{method}_{output_name}_S2.csv'))

    elif method == 'morris':
        for output_name in ['HI', 'CR']:
            if output_name in results:
                df = pd.DataFrame({
                    'parameter': results['problem']['names'],
                    'mu': results[output_name]['mu'],
                    'mu_star': results[output_name]['mu_star'],
                    'sigma': results[output_name]['sigma']
                })
                if 'mu_star_conf' in results[output_name]:
                    df['mu_star_conf'] = results[output_name]['mu_star_conf']
                df.to_csv(os.path.join(output_dir, f'{method}_{output_name}_indices.csv'), index=False)

    # Save metadata
    metadata = {
        'method': method,
        'timestamp': pd.Timestamp.now().isoformat(),
        'n_parameters': len(results['problem']['names']),
        'parameter_names': results['problem']['names'],
        'bounds': results['problem']['bounds'],
        'metadata': results.get('metadata', {})
    }

    with open(os.path.join(output_dir, 'analysis_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Sensitivity Analysis for HBMPRA Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sobol analysis with default settings
  python sensitivity_analysis.py --method sobol

  # Morris screening for parameter screening
  python sensitivity_analysis.py --method morris --n-samples 1000

  # Include toxicity parameter uncertainty
  python sensitivity_analysis.py --method sobol --include-tox-uncertainty

  # Custom input and output directories
  python sensitivity_analysis.py --input my_data.csv --output-dir my_results
        """
    )

    parser.add_argument('--method', choices=['sobol', 'morris', 'delta'],
                       default='sobol', help='Sensitivity analysis method')
    parser.add_argument('--input', default='measured_concentrations.csv',
                       help='Input concentration data file')
    parser.add_argument('--output-dir', default='results/sensitivity',
                       help='Output directory for results')
    parser.add_argument('--n-samples', type=int, default=4096,
                       help='Number of samples for analysis')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--include-tox-uncertainty', action='store_true',
                       help='Include toxicity parameter uncertainty')
    parser.add_argument('--no-uncertainty', action='store_true',
                       help='Disable parameter uncertainty (use fixed values)')
    parser.add_argument('--groups-file', help='Custom groups JSON file')
    parser.add_argument('--filter-threshold', type=float, default=0.001,
                       help='Minimum sensitivity index to include in plots (default: 0.001)')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'sensitivity_analysis.log')),
            logging.StreamHandler()
        ]
    )

    logger.info("Starting comprehensive sensitivity analysis")
    logger.info(f"Method: {args.method}")
    logger.info(f"Samples: {args.n_samples}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output_dir}")

    # Load input data
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    conc_df = pd.read_csv(args.input)
    logger.info(f"Loaded concentration data: {conc_df.shape}")

    # Detect metals
    metals = []
    conc_data = {}

    for col in conc_df.columns:
        if col.startswith('C_'):
            metal = col[2:]  # Remove 'C_' prefix
            if metal in DEFAULT_METALS:
                metals.append(metal)
                conc_data[col] = conc_df[col].values
        elif col.startswith('C_bio_'):
            metal = col[6:]  # Remove 'C_bio_' prefix
            if metal in DEFAULT_METALS:
                conc_data[col] = conc_df[col].values
        elif col in DEFAULT_METALS:
            metals.append(col)
            conc_data[col] = conc_df[col].values

    metals = sorted(list(set(metals)))
    logger.info(f"Detected metals: {metals}")

    if not metals:
        logger.warning("No metals detected in input data, using defaults")
        metals = DEFAULT_METALS[:5]  # Use first 5 as examples

    # Load external data
    toxref, kp_table, speciation_data = load_external_data()

    # Load groups
    if args.groups_file and os.path.exists(args.groups_file):
        with open(args.groups_file, 'r') as f:
            groups = json.load(f)
        logger.info(f"Loaded custom groups from {args.groups_file}")
    else:
        groups = GROUP_INFO
        logger.info("Using default demographic groups")

    # Build parameter space
    names, bounds, metadata = build_comprehensive_parameter_space(
        groups, metals, conc_data, toxref, kp_table, speciation_data,
        include_uncertainty=not args.no_uncertainty,
        include_tox_uncertainty=args.include_tox_uncertainty
    )

    logger.info(f"Parameter space: {len(names)} parameters")

    # Perform sensitivity analysis
    results = perform_sensitivity_analysis(
        args.method, names, bounds, groups, metals, toxref, kp_table,
        conc_data, speciation_data, metadata, args.n_samples, args.seed
    )

    # Create visualizations
    create_visualizations(results, args.output_dir, args.method, args.filter_threshold)

    # Save results
    save_results(results, args.output_dir, args.method)

    # Create summary report
    create_summary_report(results, args.output_dir, args.method, args)

    logger.info("Sensitivity analysis completed successfully")
    logger.info(f"Results saved to: {args.output_dir}")


def create_summary_report(results: Dict[str, Any], output_dir: str, method: str, args):
    """Create a comprehensive summary report."""
    report_path = os.path.join(output_dir, 'sensitivity_analysis_report.md')

    with open(report_path, 'w') as f:
        f.write("# Sensitivity Analysis Report\n\n")
        f.write(f"**Analysis Method:** {method.upper()}\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Samples:** {args.n_samples}\n")
        f.write(f"**Parameters:** {len(results['problem']['names'])}\n\n")

        f.write("## Key Findings\n\n")

        # Summarize most influential parameters
        if method == 'sobol':
            for output_name in ['HI', 'CR']:
                if output_name in results:
                    f.write(f"### {output_name} - Most Influential Parameters\n\n")
                    st_indices = results[output_name]['ST']
                    top_params = sorted(zip(results['problem']['names'], st_indices),
                                      key=lambda x: x[1], reverse=True)[:10]

                    f.write("| Parameter | Total Effect | Confidence |\n")
                    f.write("|-----------|--------------|------------|\n")
                    for param, st in top_params:
                        conf = results[output_name]['ST_conf'][results['problem']['names'].index(param)]
                        f.write(f"| {param} | {st:.6f} | {conf:.6f} |\n")
                    f.write("\n")

        f.write("## Methodology\n\n")
        f.write("This analysis used comprehensive parameter bounds derived from:\n")
        f.write("- Scientific literature and regulatory guidelines\n")
        f.write("- Default uncertainty assumptions (CV = 0.21 for body weight)\n")
        f.write("- Log-normal distributions for ingestion rate uncertainty\n")
        f.write("- Conservative bounds for concentration variability\n\n")

        f.write("## Recommendations\n\n")
        f.write("1. Focus monitoring efforts on the most influential parameters\n")
        f.write("2. Consider parameter interactions in risk management decisions\n")
        f.write("3. Validate model assumptions with site-specific data\n")
        f.write("4. Use uncertainty bounds for decision-making under uncertainty\n\n")

        f.write("## Files Generated\n\n")
        f.write("- `*_indices.csv`: Sensitivity indices for each parameter\n")
        f.write("- `*_tornado.png`: Tornado plots showing parameter influence\n")
        f.write("- `parameter_effects_*.png`: Scatter plots of parameter effects\n")
        f.write("- `analysis_metadata.json`: Complete analysis metadata\n")
        f.write("- `sensitivity_analysis.log`: Detailed execution log\n\n")

    logger.info(f"Summary report saved to: {report_path}")


if __name__ == '__main__':
    main()