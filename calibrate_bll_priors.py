"""
Calibrate BLL priors using pure-Python engines (no AALM dependency).

Produces JSON with per-group linear fits BLL = b0 + k_wb * EDI and an engine_metadata block.
"""

import os
import json
import argparse
import csv
import numpy as np
try:
    import yaml  # for writing manifest if available
    _have_yaml = True
except Exception:
    yaml = None
    _have_yaml = False
import shutil

from bll_engines import (
    OneCompParams,
    edi_from_conc_ugL,
    bll_onecomp_from_water,
    bll_linear_from_intake,
    classify_population_group,
    get_group_specific_slope,
    get_group_specific_onecomp_params,
    PopulationEngineConfig,
    compute_bll_auto
)
import demographics


def build_conc_grid(conc_series, n, mode="logspace"):
    data = np.asarray([max(float(x), 0.0) for x in conc_series if np.isfinite(x)])
    if data.size == 0:
        return np.zeros(n)
    lo = max(np.percentile(data, 5), 1e-12)
    hi = max(np.percentile(data, 95), lo * 1.0001)
    if mode == "quantile":
        return np.quantile(np.sort(data), np.linspace(0.05, 0.95, n))
    return np.geomspace(lo, hi, num=n)


def _pick_group_slope(group_name: str, args) -> float:
    g = group_name.lower()
    if any(x in g for x in ["child", "infant", "toddler"]):
        return float(args.slope_child)
    if any(x in g for x in ["adult", "preg", "lact"]):
        return float(args.slope_adult)
    return float(args.slope_generic)


def _group_params_for_onecomp(group_name: str, args) -> OneCompParams:
    f_abs = args.f_abs
    if "adult" in group_name.lower():
        f_abs = min(f_abs, 0.3)
    return OneCompParams(
        f_abs=f_abs,
        t_half_days=args.t_half_days,
        blood_vol_per_kg=args.blood_vol_per_kg,
        background_ugdl=args.bll_background,
    )


def main():
    parser = argparse.ArgumentParser(description="Calibrate BLL priors using internal engines (no AALM).")
    parser.add_argument("--chemistry", required=True, help="Path to measured concentrations CSV with C_Pb (µg/L).")
    parser.add_argument("--templates-json", required=False,
                        help="JSON mapping group→template path (optional, only for manifest bookkeeping).")
    parser.add_argument("--duration-days", type=int, default=365, help="Exposure duration in days (>=90).")
    parser.add_argument("--out-json", required=True, help="Path to output priors JSON.")
    parser.add_argument("--groups", nargs="+", default=None, help="Optional subset of groups to calibrate.")

    # Engine options
    parser.add_argument("--bll-engine", choices=["onecomp", "slope", "auto"], default="auto",
                        help="Engine for generating (EDI,BLL) calibration pairs. 'auto' uses population-appropriate models.")
    parser.add_argument("--f-abs", type=float, default=0.5, help="Water absorption fraction (onecomp).")
    parser.add_argument("--t-half-days", type=float, default=30.0, help="Blood half-life (days, onecomp).")
    parser.add_argument("--blood-vol-per-kg", type=float, default=0.07, help="Blood volume (L/kg, onecomp).")
    parser.add_argument("--bll-background", type=float, default=0.0, help="Baseline BLL to add (µg/dL).")
    parser.add_argument("--slope-adult", type=float, default=0.08, help="Adult slope µg/dL per µg/day (slope engine).")
    parser.add_argument("--slope-child", type=float, default=0.17, help="Child slope µg/dL per µg/day (slope engine).")
    parser.add_argument("--slope-generic", type=float, default=0.12, help="Fallback slope (slope engine).")
    parser.add_argument("--grid", choices=["logspace", "quantile"], default="logspace",
                        help="Grid type for water Pb to generate pairs.")
    parser.add_argument("--n-grid", type=int, default=10, help="Number of grid points per group.")
    args = parser.parse_args()

    # load templates if provided
    templates = {}
    if args.templates_json:
        try:
            templates = json.load(open(args.templates_json))
        except Exception:
            templates = {}

    # load chemistry
    with open(args.chemistry, newline='') as f:
        reader = csv.DictReader(f)
        fnames = reader.fieldnames or []
        if 'C_Pb' in fnames:
            col = 'C_Pb'
        elif 'Pb' in fnames:
            col = 'Pb'
        else:
            parser.error("Chemistry CSV missing 'C_Pb' or 'Pb' column")
        cpb_vals = []
        for row in reader:
            try:
                cpb_vals.append(float(row[col]))
            except (ValueError, KeyError):
                continue
        cpb = np.array(cpb_vals)

    # prepare output
    out_root = os.path.dirname(args.out_json)
    os.makedirs(out_root, exist_ok=True)

    # determine groups to process
    group_keys = list(demographics.GROUP_INFO.keys()) if args.groups is None else args.groups
    missing = [g for g in group_keys if g not in demographics.GROUP_INFO]
    if missing:
        parser.error(f"Groups not in demographics.GROUP_INFO: {missing}")

    pairs_by_group = {}
    engine_used = args.bll_engine
    engines_per_group = {}  # Track which engine was used for each group

    # Setup population engine configuration if using auto mode
    config = None
    base_onecomp_params = None
    if engine_used == "auto":
        config = PopulationEngineConfig()
        config.default_slopes = {
            "adult": args.slope_adult,
            "child": args.slope_child,
            "teen": args.slope_child,  # Use child slope for teens
            "pregnant": args.slope_adult,
            "generic": args.slope_generic
        }
        base_onecomp_params = OneCompParams(
            f_abs=args.f_abs,
            t_half_days=args.t_half_days,
            blood_vol_per_kg=args.blood_vol_per_kg,
            background_ugdl=args.bll_background
        )

    for group in group_keys:
        info = demographics.GROUP_INFO[group]
        BW = float(info['BW'])
        IR = float(info['IR'])
        # build concentration grid using measured chemistry
        conc_grid = build_conc_grid(cpb, args.n_grid, mode=args.grid)
        pairs = []
        
        if engine_used == 'auto':
            # Use population-appropriate engine
            for c in conc_grid:
                edi = edi_from_conc_ugL(c, IR, BW)
                bll_value, actual_engine, metadata = compute_bll_auto(
                    group_name=group,
                    edi_mgkgd=edi,
                    bw_kg=BW,
                    ir_L_per_d=IR,
                    config=config,
                    onecomp_params=base_onecomp_params
                )
                pairs.append([edi, bll_value])
            # Store which engine was actually used for this group
            engines_per_group[group] = actual_engine
            
        elif engine_used == 'onecomp':
            params = _group_params_for_onecomp(group, args)
            for c in conc_grid:
                edi = edi_from_conc_ugL(c, IR, BW)
                bll = bll_onecomp_from_water(c, IR, BW, params)
                pairs.append([edi, bll])
            engines_per_group[group] = 'onecomp'
            
        else:  # slope
            slope = _pick_group_slope(group, args)
            for c in conc_grid:
                edi = edi_from_conc_ugL(c, IR, BW)
                bll = bll_linear_from_intake(edi, BW, slope, f_abs=args.f_abs, background_ugdl=args.bll_background)
                pairs.append([edi, bll])
            engines_per_group[group] = 'slope'
            
        pairs_by_group[group] = pairs

    # Fit linear mapping for each group
    priors = {'groups': {}, 'engine_metadata': {}}
    for g, pairs in pairs_by_group.items():
        X = np.array([p[0] for p in pairs], dtype=float)
        Y = np.array([p[1] for p in pairs], dtype=float)
        if X.size == 0:
            b0 = 0.0
            k_wb = 0.0
            sigma = 0.0
        else:
            A = np.vstack([np.ones_like(X), X]).T
            b0, k_wb = np.linalg.lstsq(A, Y, rcond=None)[0]
            resid = Y - (b0 + k_wb * X)
            sigma = float(np.std(resid, ddof=1)) if X.size >= 2 else 0.0
        b0_mu = float(b0)
        b0_sigma = float(max(sigma, 1e-6))
        k_wb_mu = float(k_wb)
        k_wb_sigma = float(max(0.1 * abs(k_wb), 1e-6))
        priors['groups'][g] = {
            # new canonical names
            'b0_mu': b0_mu,
            'b0_sigma': b0_sigma,
            'k_wb_mu': k_wb_mu,
            'k_wb_sigma': k_wb_sigma,
            # legacy names for backward compatibility
            'b0_mean': b0_mu,
            'b0_sd': b0_sigma,
            'k_mean': k_wb_mu,
            'k_sd': k_wb_sigma,
            'pairs': [[float(a), float(b)] for a, b in pairs]
        }

    # engine metadata
    priors['engine_metadata'] = {
        'engine_selection_mode': engine_used,
        'engines_per_group': engines_per_group,
        'onecomp_params': {
            'f_abs_default': args.f_abs,
            't_half_days': args.t_half_days,
            'blood_vol_per_kg': args.blood_vol_per_kg,
            'background_ugdl': args.bll_background,
        },
        'slopes': {
            'adult': args.slope_adult,
            'child': args.slope_child,
            'generic': args.slope_generic,
        },
        'group_IR_L_per_d': {g: float(demographics.GROUP_INFO[g]['IR']) for g in demographics.GROUP_INFO},
        'group_BW_kg': {g: float(demographics.GROUP_INFO[g]['BW']) for g in demographics.GROUP_INFO},
        'date': np.datetime_as_string(np.datetime64('now'), unit='s'),
        'population_classification': {
            g: classify_population_group(g) for g in group_keys
        } if engine_used == 'auto' else {}
    }
    # indicate that legacy keys were provided for compatibility
    priors['engine_metadata']['legacy_key_compat'] = True

    with open(args.out_json, 'w') as pj:
        json.dump(priors, pj, indent=2)


if __name__ == '__main__':
    main()
