#!/usr/bin/env python3
"""
summary_tables.py - Generate Publication-Ready Summary Tables from HBMPRA Results

This script generates comprehensive summary tables from HBMPRA output:
  T1: Input data summary (measured concentrations) with WHO guideline comparison
  T2: Speciation summary (bioavailable fractions) - if available
  T3: Posterior summary (3%, median, 94%, exceedance probabilities) per demographic
  T4: Risk ranking (top sites by HI_overall)
  T5: BLL summary (Blood Lead Levels) - if available

Additional analyses:
  - Spearman's correlation matrix for metals
  - PCA with varimax rotation for source apportionment
  - KMO and Bartlett's test for sampling adequacy

Usage:
    python summary_tables.py --results-dir results_myanalysis
    python summary_tables.py --results-dir results_test --input data1.csv
    
Author: Dickson Abdul-Wahab (University of Ghana) & Ebenezer Aquisman Asare
"""

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Get the directory where this script lives (src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Project root is parent of src/
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# Water data directory
WATERDATA_DIR = os.path.join(PROJECT_ROOT, "waterdata")


def load_trace(results_dir):
    """Load arviz InferenceData from trace.nc"""
    import arviz as az
    trace_path = os.path.join(results_dir, "trace.nc")
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"trace.nc not found in {results_dir}")
    return az.from_netcdf(trace_path)


def load_chemistry_data(results_dir, input_file=None):
    """Load the original chemistry input data."""
    # First try to get input file from RUNLOG
    runlog_path = os.path.join(results_dir, "RUNLOG.json")
    if os.path.exists(runlog_path):
        try:
            with open(runlog_path) as f:
                runlog = json.load(f)
            if 'chemistry_file' in runlog:
                chem_file = runlog['chemistry_file']
                if os.path.exists(chem_file):
                    return pd.read_csv(chem_file)
        except Exception:
            pass
    
    # Use provided input file
    if input_file and os.path.exists(input_file):
        return pd.read_csv(input_file)
    
    # Search for common chemistry file names in multiple locations
    search_dirs = [
        WATERDATA_DIR,                         # waterdata/ folder
        os.path.dirname(results_dir),          # Parent of results
        PROJECT_ROOT,                          # Project root
        os.getcwd(),                           # Current directory
    ]
    
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for name in ['chemistry.csv', 'data.csv', 'data1.csv', 'data2.csv']:
            path = os.path.join(d, name)
            if os.path.exists(path):
                return pd.read_csv(path)
    
    return None


def load_speciation_data(results_dir):
    """Load speciation results if available."""
    spec_path = os.path.join(results_dir, "table_species_fractions.csv")
    if os.path.exists(spec_path):
        return pd.read_csv(spec_path)
    
    # Check external folder
    ext_path = os.path.join(os.path.dirname(results_dir), "external", "table_species_fractions.csv")
    if os.path.exists(ext_path):
        return pd.read_csv(ext_path)
    
    return None


def load_standards(results_dir=None):
    """Load WHO/EPA standards."""
    # Try multiple locations
    search_paths = [
        os.path.join(results_dir, "standards.csv") if results_dir else None,
        os.path.join(PROJECT_ROOT, "standards.csv"),
        os.path.join(PROJECT_ROOT, "standards_sources.csv"),
        "standards.csv",
        "standards_sources.csv",
    ]
    for path in search_paths:
        if path and os.path.exists(path):
            return pd.read_csv(path)
    return None


def generate_t1_measured_summary(df_chem, df_standards, output_dir):
    """T1: Measured concentrations summary with WHO comparison."""
    if df_chem is None:
        print("  [WARN] No chemistry data available for T1")
        return None
    
    # Identify metal columns
    excluded = ['Site', 'site', 'Source', 'ID', 'id', 'Community', 'Location', 
                'Latitude', 'Longitude', 'Lat', 'Lon', 'Elev', 'X', 'Y']
    metals = [c for c in df_chem.columns if c not in excluded 
              and not c.startswith('frac_') and not c.startswith('bio_')]
    
    # Compute statistics
    records = []
    for col in metals:
        vals = pd.to_numeric(df_chem[col], errors='coerce').dropna()
        if len(vals) == 0:
            continue
        
        record = {
            'Parameter': col,
            'N': len(vals),
            'Min': vals.min(),
            'Max': vals.max(),
            'Mean': vals.mean(),
            'Median': vals.median(),
            'Std': vals.std(),
        }
        
        # Add WHO guideline if available
        if df_standards is not None:
            metal_symbol = col.replace('C_', '').upper()
            
            # Handle different standards file formats
            if 'analyte' in df_standards.columns:
                # standards_sources.csv format: 'analyte' column like 'Arsenic (As)'
                match = df_standards[df_standards['analyte'].str.contains(metal_symbol, case=False, na=False)]
                if len(match) > 0:
                    who_val = match.iloc[0].get('who_gv_ugL', match.iloc[0].get('who_gv', None))
                    record['WHO_Guideline'] = who_val
                    if pd.notna(who_val) and str(who_val).replace('.','').replace('-','').isdigit():
                        try:
                            record['Exceedance_%'] = (vals > float(who_val)).mean() * 100
                        except:
                            pass
            elif 'metal' in df_standards.columns:
                # standards.csv format: 'metal' column
                match = df_standards[df_standards['metal'].str.upper() == metal_symbol]
                if len(match) > 0:
                    # Check for various guideline column names
                    for gv_col in ['who_gv_ugL', 'who_gv', 'S_i', 'guideline']:
                        if gv_col in match.columns:
                            who_val = match.iloc[0].get(gv_col, None)
                            if pd.notna(who_val):
                                record['WHO_Guideline'] = who_val
                                try:
                                    record['Exceedance_%'] = (vals > float(who_val)).mean() * 100
                                except:
                                    pass
                                break
        
        # Built-in WHO guidelines for anions (not in standards file)
        col_upper = col.upper().replace('_', '').replace('-', '')
        if 'FLUORIDE' in col_upper or col_upper == 'F':
            # Fluoride: WHO guideline 1.5 mg/L
            if 'WHO_Guideline' not in record:
                record['WHO_Guideline'] = 1.5
                record['WHO_Unit'] = 'mg/L'
                try:
                    record['Exceedance_%'] = (vals > 1.5).mean() * 100
                except:
                    pass
        elif 'NO3' in col_upper or 'NITRATE' in col_upper:
            # Nitrate: WHO guideline 50 mg/L as NO3, 10 mg/L as NO3-N
            if 'WHO_Guideline' not in record:
                if '_N' in col_upper or 'N' == col_upper[-1]:
                    # NO3-N basis
                    record['WHO_Guideline'] = 10.0
                    record['WHO_Unit'] = 'mg/L as NO3-N'
                    try:
                        record['Exceedance_%'] = (vals > 10.0).mean() * 100
                    except:
                        pass
                else:
                    # NO3 basis
                    record['WHO_Guideline'] = 50.0
                    record['WHO_Unit'] = 'mg/L as NO3'
                    try:
                        record['Exceedance_%'] = (vals > 50.0).mean() * 100
                    except:
                        pass
        
        records.append(record)
    
    df_t1 = pd.DataFrame(records)
    
    # Save
    out_path = os.path.join(output_dir, "T1_measured_summary.csv")
    df_t1.to_csv(out_path, index=False)
    print(f"  [OK] Saved {out_path}")
    return df_t1


def generate_t2_speciation_summary(df_spec, output_dir):
    """T2: Speciation fractions summary."""
    if df_spec is None:
        print("  [WARN] No speciation data available for T2")
        return None
    
    # Find fraction columns
    frac_cols = [c for c in df_spec.columns if c.startswith('frac_')]
    
    if not frac_cols:
        print("  [WARN] No fraction columns found in speciation data")
        return None
    
    records = []
    for col in frac_cols:
        vals = pd.to_numeric(df_spec[col], errors='coerce').dropna()
        if len(vals) == 0:
            continue
        
        metal = col.replace('frac_', '')
        records.append({
            'Metal': metal,
            'Bioavailable_Fraction_Min': vals.min(),
            'Bioavailable_Fraction_Max': vals.max(),
            'Bioavailable_Fraction_Mean': vals.mean(),
            'Bioavailable_Fraction_Median': vals.median(),
            'Bioavailable_Fraction_Std': vals.std(),
        })
    
    df_t2 = pd.DataFrame(records)
    
    out_path = os.path.join(output_dir, "T2_speciation_summary.csv")
    df_t2.to_csv(out_path, index=False)
    print(f"  [OK] Saved {out_path}")
    return df_t2


def generate_t3_posterior_summary(idata, output_dir):
    """T3: Posterior summary (quantiles and exceedance) per demographic."""
    post = idata.posterior
    
    # Extract group names as plain strings
    groups = []
    if "group" in post.coords:
        group_coord = post.coords["group"].values
        groups = [str(g) for g in group_coord]
    if not groups:
        # Fallback to default demographic groups
        groups = ['Adults', 'Children', 'Teens', 'Pregnant']
    
    records = []
    
    # Thresholds per metric type
    def get_threshold(var_name):
        if 'CR' in var_name:
            return 1e-6
        elif 'BLL' in var_name:
            return 3.5
        else:  # HI and others
            return 1.0
    
    # Process each variable
    for var in post.data_vars:
        arr = post[var]
        dims = tuple(getattr(arr, 'dims', ()))
        
        # Skip internal/helper variables
        if var.startswith('_') or var in ['draw', 'chain']:
            continue
        
        thr = get_threshold(var)
        
        # Per demographic group
        if 'group' in dims:
            for i, grp in enumerate(groups):
                try:
                    sub = arr.isel(group=i)
                    
                    # Average over sites if present
                    if 'site' in sub.dims:
                        sub_mean = sub.mean(dim='site')
                    else:
                        sub_mean = sub
                    
                    # Compute statistics
                    vals = sub_mean.values.flatten()
                    vals = vals[np.isfinite(vals)]
                    
                    if len(vals) < 10:
                        continue
                    
                    records.append({
                        'Metric': str(var),
                        'Demographic': str(grp),
                        '3%': float(np.percentile(vals, 3)),
                        'Median': float(np.median(vals)),
                        'Mean': float(np.mean(vals)),
                        '97%': float(np.percentile(vals, 97)),
                        'P(>threshold)': float((vals > thr).mean()),
                        'Threshold': float(thr)
                    })
                except Exception:
                    continue
        
        # Site-level variables without group dimension
        elif 'site' in dims and 'group' not in dims:
            try:
                vals = arr.values.flatten()
                vals = vals[np.isfinite(vals)]
                
                if len(vals) < 10:
                    continue
                
                records.append({
                    'Metric': str(var),
                    'Demographic': 'All',
                    '3%': float(np.percentile(vals, 3)),
                    'Median': float(np.median(vals)),
                    'Mean': float(np.mean(vals)),
                    '97%': float(np.percentile(vals, 97)),
                    'P(>threshold)': float((vals > thr).mean()),
                    'Threshold': float(thr)
                })
            except Exception:
                continue
    
    df_t3 = pd.DataFrame(records)
    
    # Sort by metric and demographic
    if len(df_t3) > 0:
        df_t3 = df_t3.sort_values(['Metric', 'Demographic'])
    
    out_path = os.path.join(output_dir, "T3_posterior_summary.csv")
    df_t3.to_csv(out_path, index=False)
    print(f"  [OK] Saved {out_path}")
    
    # Also create a wide-format version (Metric as rows, demographics as column groups)
    if len(df_t3) > 0:
        try:
            pivot_cols = ['3%', 'Median', 'Mean', '97%', 'P(>threshold)']
            df_wide = df_t3.pivot(index='Metric', columns='Demographic', values=pivot_cols)
            # Flatten MultiIndex columns
            df_wide.columns = [f'{demo}_{stat}' for stat, demo in df_wide.columns]
            df_wide = df_wide.reset_index()
            
            out_wide = os.path.join(output_dir, "T3_posterior_summary_wide.csv")
            df_wide.to_csv(out_wide, index=False)
            print(f"  [OK] Saved {out_wide}")
        except Exception as e:
            print(f"  [WARN] Could not create wide format: {e}")
    
    return df_t3


def generate_t4_risk_ranking(idata, output_dir, df_chem=None, top_n=10):
    """T4: Top sites by overall hazard index."""
    post = idata.posterior
    
    # Extract coordinates as plain strings
    groups = []
    if "group" in post.coords:
        groups = [str(g) for g in post.coords["group"].values]
    if not groups:
        groups = ['Adults', 'Children', 'Teens', 'Pregnant']
    
    sites = []
    if "site" in post.coords:
        sites = [str(s) for s in post.coords["site"].values]
    
    if not sites:
        print("  [WARN] No site dimension found for risk ranking")
        return None
    
    # Try to get actual sample IDs from chemistry data
    site_id_map = {}
    if df_chem is not None:
        # Look for ID column in chemistry data
        id_col = None
        for col in ['ID', 'id', 'Sample_ID', 'sample_id', 'SampleID', 'Site', 'site']:
            if col in df_chem.columns:
                id_col = col
                break
        
        if id_col is not None:
            # Map site index to actual ID
            sample_ids = df_chem[id_col].tolist()
            for i, site_idx in enumerate(sites):
                try:
                    idx = int(site_idx)
                    if idx < len(sample_ids):
                        site_id_map[site_idx] = str(sample_ids[idx])
                except (ValueError, IndexError):
                    pass
    
    records = []
    
    # Find risk variables
    hi_var = None
    for v in ['HI_overall', 'HI_total']:
        if v in post.data_vars:
            hi_var = v
            break
    
    cr_var = None
    for v in ['CR_total', 'CR_overall', 'TCR']:
        if v in post.data_vars:
            cr_var = v
            break
    
    bll_var = 'BLL' if 'BLL' in post.data_vars else None
    
    # Find organ-specific HI variables
    organ_hi_vars = {}
    organ_names = ['neuro', 'nephro', 'hepato', 'derm', 'gi', 'cardiovascular', 
                   'endocrine', 'hemato', 'skeletal_dental', 'systemic']
    for organ in organ_names:
        var = f'HI_{organ}'
        if var in post.data_vars:
            organ_hi_vars[organ] = var
    
    if hi_var is None:
        print("  [WARN] No HI_overall variable found for risk ranking")
        return None
    
    arr_hi = post[hi_var]
    arr_cr = post[cr_var] if cr_var else None
    arr_bll = post[bll_var] if bll_var else None
    
    def get_stats(arr, site_i, group_j):
        """Extract statistics for a variable at given site/group."""
        try:
            if 'group' in arr.dims:
                sub = arr.isel(site=site_i, group=group_j)
            else:
                sub = arr.isel(site=site_i)
            vals = sub.values.flatten()
            vals = vals[np.isfinite(vals)]
            if len(vals) < 10:
                return None
            return {
                'mean': np.mean(vals),
                'median': np.median(vals),
                'p97': np.percentile(vals, 97),
            }
        except Exception:
            return None
    
    for i, site in enumerate(sites):
        for j, grp in enumerate(groups):
            # Get HI stats
            hi_stats = get_stats(arr_hi, i, j)
            if hi_stats is None:
                continue
            
            # Use mapped sample ID if available
            site_label = site_id_map.get(site, site)
            
            record = {
                'Site': site_label,
                'Demographic': grp,
                'HI_Mean': hi_stats['mean'],
                'HI_Median': hi_stats['median'],
                'HI_97%': hi_stats['p97'],
                'P(HI>1)': None,  # Will compute below
            }
            
            # Compute P(HI>1)
            try:
                if 'group' in arr_hi.dims:
                    sub = arr_hi.isel(site=i, group=j)
                else:
                    sub = arr_hi.isel(site=i)
                vals = sub.values.flatten()
                vals = vals[np.isfinite(vals)]
                record['P(HI>1)'] = (vals > 1.0).mean()
            except:
                pass
            
            # Get CR stats if available
            if arr_cr is not None:
                cr_stats = get_stats(arr_cr, i, j)
                if cr_stats:
                    record['CR_Mean'] = cr_stats['mean']
                    record['CR_Median'] = cr_stats['median']
                    record['CR_97%'] = cr_stats['p97']
                    # Compute P(CR > 1e-6)
                    try:
                        if 'group' in arr_cr.dims:
                            sub = arr_cr.isel(site=i, group=j)
                        else:
                            sub = arr_cr.isel(site=i)
                        vals = sub.values.flatten()
                        vals = vals[np.isfinite(vals)]
                        record['P(CR>1e-6)'] = (vals > 1e-6).mean()
                    except:
                        pass
            
            # Get BLL stats if available
            if arr_bll is not None:
                bll_stats = get_stats(arr_bll, i, j)
                if bll_stats:
                    record['BLL_Mean'] = bll_stats['mean']
                    record['BLL_Median'] = bll_stats['median']
                    record['BLL_97%'] = bll_stats['p97']
                    # Compute P(BLL > 3.5) for children, 5.0 for others
                    thr = 3.5 if grp in ['Children', 'Pregnant'] else 5.0
                    try:
                        if 'group' in arr_bll.dims:
                            sub = arr_bll.isel(site=i, group=j)
                        else:
                            sub = arr_bll.isel(site=i)
                        vals = sub.values.flatten()
                        vals = vals[np.isfinite(vals)]
                        record[f'P(BLL>{thr})'] = (vals > thr).mean()
                    except:
                        pass
            
            # Get dominant organ HI (organ with highest mean)
            if organ_hi_vars:
                max_organ = None
                max_hi = 0
                for organ, var in organ_hi_vars.items():
                    organ_stats = get_stats(post[var], i, j)
                    if organ_stats and organ_stats['mean'] > max_hi:
                        max_hi = organ_stats['mean']
                        max_organ = organ
                if max_organ:
                    record['Dominant_Organ'] = max_organ
                    record['Dominant_Organ_HI'] = max_hi
            
            records.append(record)
    
    df_t4 = pd.DataFrame(records)
    
    if len(df_t4) > 0:
        # Sort by mean HI descending
        df_t4 = df_t4.sort_values('HI_Mean', ascending=False)
        
        # Get top N
        df_top = df_t4.head(top_n * len(groups))
    else:
        df_top = df_t4
    
    out_path = os.path.join(output_dir, "T4_risk_ranking.csv")
    df_top.to_csv(out_path, index=False)
    print(f"  [OK] Saved {out_path}")
    return df_top


def generate_t5_bll_summary(idata, output_dir):
    """T5: Blood Lead Level summary."""
    post = idata.posterior
    
    if 'BLL' not in post.data_vars:
        print("  [WARN] No BLL variable found")
        return None
    
    # Extract group names as plain strings
    groups = []
    if "group" in post.coords:
        groups = [str(g) for g in post.coords["group"].values]
    if not groups:
        groups = ['Adults', 'Children', 'Teens', 'Pregnant']
    
    records = []
    arr = post['BLL']
    
    # CDC thresholds
    thresholds = {
        'Adults': 10.0,       # CDC occupational concern
        'Children': 3.5,      # CDC reference value (2021)
        'Teens': 5.0,         # General concern
        'Pregnant': 3.5,      # Elevated concern
    }
    
    for j, grp in enumerate(groups):
        try:
            if 'group' in arr.dims:
                sub = arr.isel(group=j)
            else:
                sub = arr
            
            # Average over sites if needed
            if 'site' in sub.dims:
                sub = sub.mean(dim='site')
            
            vals = sub.values.flatten()
            vals = vals[np.isfinite(vals)]
            
            if len(vals) < 10:
                continue
            
            thr = thresholds.get(grp, 5.0)
            
            records.append({
                'Demographic': grp,
                'BLL_Mean': np.mean(vals),
                'BLL_Median': np.median(vals),
                'BLL_3%': np.percentile(vals, 3),
                'BLL_97%': np.percentile(vals, 97),
                f'P(BLL>{thr})': (vals > thr).mean(),
                'Reference_Threshold': thr,
            })
        except Exception:
            continue
    
    df_t5 = pd.DataFrame(records)
    
    out_path = os.path.join(output_dir, "T5_BLL_summary.csv")
    df_t5.to_csv(out_path, index=False)
    print(f"  [OK] Saved {out_path}")
    return df_t5


def generate_correlation_matrix(df_chem, output_dir):
    """Generate Spearman's correlation matrix for metals."""
    if df_chem is None:
        print("  [WARN] No chemistry data for correlation analysis")
        return None
    
    # Identify numeric columns (metals)
    excluded = ['Site', 'site', 'Source', 'ID', 'id', 'Community', 'Location', 
                'Latitude', 'Longitude', 'Lat', 'Lon', 'Elev', 'X', 'Y']
    
    numeric_cols = []
    for col in df_chem.columns:
        if col in excluded:
            continue
        vals = pd.to_numeric(df_chem[col], errors='coerce')
        if vals.notna().sum() > 5:
            numeric_cols.append(col)
    
    if len(numeric_cols) < 2:
        print("  [WARN] Not enough numeric columns for correlation")
        return None
    
    df_numeric = df_chem[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Spearman correlation
    corr = df_numeric.corr(method='spearman')
    
    out_path = os.path.join(output_dir, "Correlation_matrix.csv")
    corr.to_csv(out_path)
    print(f"  [OK] Saved {out_path}")
    return corr


def generate_pca_analysis(df_chem, output_dir):
    """Generate PCA with varimax rotation for source apportionment."""
    if df_chem is None:
        print("  [WARN] No chemistry data for PCA")
        return None
    
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  [WARN] sklearn required for PCA analysis")
        return None
    
    # Try to import factor analyzer for varimax and KMO
    try:
        from factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
        from factor_analyzer.rotator import Rotator
        has_factor_analyzer = True
    except ImportError:
        has_factor_analyzer = False
        print("  [WARN] factor_analyzer not installed, skipping KMO/varimax rotation")
    
    # Get numeric metal columns
    excluded = ['Site', 'site', 'Source', 'ID', 'id', 'Community', 'Location', 
                'Latitude', 'Longitude', 'Lat', 'Lon', 'Elev', 'X', 'Y']
    
    numeric_cols = []
    for col in df_chem.columns:
        if col in excluded:
            continue
        vals = pd.to_numeric(df_chem[col], errors='coerce')
        if vals.notna().sum() > 5:
            numeric_cols.append(col)
    
    if len(numeric_cols) < 3:
        print("  [WARN] Not enough variables for PCA")
        return None
    
    df_numeric = df_chem[numeric_cols].apply(pd.to_numeric, errors='coerce').dropna()
    
    if len(df_numeric) < 10:
        print("  [WARN] Not enough samples for PCA")
        return None
    
    # Log transform (for metals which are typically log-normal)
    df_log = np.log1p(df_numeric)
    
    # Standardize
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_log), columns=numeric_cols)
    
    # KMO and Bartlett's test
    if has_factor_analyzer:
        try:
            kmo_all, kmo_model = calculate_kmo(df_scaled)
            bart_stat, bart_p = calculate_bartlett_sphericity(df_scaled)
            
            with open(os.path.join(output_dir, "KMO_Bartlett.txt"), 'w') as f:
                f.write(f"Kaiser-Meyer-Olkin (KMO) Measure of Sampling Adequacy\n")
                f.write(f"=" * 50 + "\n")
                f.write(f"Overall KMO: {kmo_model:.4f}\n")
                f.write(f"Interpretation: {'Excellent' if kmo_model > 0.9 else 'Good' if kmo_model > 0.8 else 'Mediocre' if kmo_model > 0.6 else 'Poor'}\n\n")
                f.write(f"Bartlett's Test of Sphericity\n")
                f.write(f"=" * 50 + "\n")
                f.write(f"Chi-square: {bart_stat:.2f}\n")
                f.write(f"p-value: {bart_p:.2e}\n")
                f.write(f"Interpretation: {'Significant (reject null)' if bart_p < 0.05 else 'Not significant'}\n")
            print(f"  [OK] Saved KMO_Bartlett.txt")
        except Exception as e:
            print(f"  [WARN] KMO/Bartlett failed: {e}")
    
    # PCA
    pca = PCA()
    pca.fit(df_scaled)
    
    eigenvalues = pca.explained_variance_
    explained_var = pca.explained_variance_ratio_ * 100
    cumulative_var = np.cumsum(explained_var)
    
    # Select components with eigenvalue > 1 (Kaiser criterion)
    n_components = sum(eigenvalues > 1)
    n_components = max(2, min(n_components, len(numeric_cols) - 1))
    
    loadings = pca.components_[:n_components].T  # shape: (n_features, n_components)
    
    # Varimax rotation if available
    if has_factor_analyzer:
        try:
            rotator = Rotator(method='varimax')
            rotated_loadings = rotator.fit_transform(loadings)
        except Exception:
            rotated_loadings = loadings
    else:
        rotated_loadings = loadings
    
    # Create results DataFrame
    df_loadings = pd.DataFrame(
        rotated_loadings,
        index=numeric_cols,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    df_loadings.index.name = 'Variable'
    df_loadings = df_loadings.reset_index()
    
    # Add eigenvalue and variance rows
    eigen_row = pd.DataFrame([{
        'Variable': 'Eigenvalue',
        **{f'PC{i+1}': eigenvalues[i] for i in range(n_components)}
    }])
    var_row = pd.DataFrame([{
        'Variable': '% Variance',
        **{f'PC{i+1}': explained_var[i] for i in range(n_components)}
    }])
    cum_row = pd.DataFrame([{
        'Variable': 'Cumulative %',
        **{f'PC{i+1}': cumulative_var[i] for i in range(n_components)}
    }])
    
    df_pca = pd.concat([df_loadings, eigen_row, var_row, cum_row], ignore_index=True)
    
    out_path = os.path.join(output_dir, "PCA_results.csv")
    df_pca.to_csv(out_path, index=False)
    print(f"  [OK] Saved {out_path}")
    return df_pca


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary tables from HBMPRA results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python summary_tables.py --results-dir results_test
    python summary_tables.py --results-dir results_test --input data1.csv
    python summary_tables.py --results-dir results_test --output-dir tables/
        """
    )
    
    parser.add_argument("--results-dir", "-r", required=True,
                        help="Directory containing HBMPRA results (trace.nc, etc.)")
    parser.add_argument("--input", "-i", default=None,
                        help="Original chemistry CSV input file (optional, auto-detected from RUNLOG)")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory for tables (default: results-dir/tables)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top sites to include in risk ranking (default: 10)")
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.results_dir, "tables")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("  HBMPRA Summary Tables Generator")
    print("=" * 60)
    print(f"\n  Results directory: {args.results_dir}")
    print(f"  Output directory:  {output_dir}")
    print("-" * 60)
    
    # Load data
    print("\nLoading data...")
    
    try:
        idata = load_trace(args.results_dir)
        print("  [OK] Loaded trace.nc")
    except Exception as e:
        print(f"  [ERROR] Failed to load trace: {e}")
        return 1
    
    df_chem = load_chemistry_data(args.results_dir, args.input)
    if df_chem is not None:
        print(f"  [OK] Loaded chemistry data ({len(df_chem)} samples)")
    else:
        print("  [WARN] Chemistry data not found (T1 and statistical analyses will be limited)")
    
    df_spec = load_speciation_data(args.results_dir)
    if df_spec is not None:
        print(f"  [OK] Loaded speciation data")
    
    df_standards = load_standards(args.results_dir)
    if df_standards is not None:
        print(f"  [OK] Loaded standards data")
    
    # Generate tables
    print("\n" + "-" * 60)
    print("Generating summary tables...")
    print("-" * 60)
    
    print("\n[T1] Measured concentrations summary...")
    generate_t1_measured_summary(df_chem, df_standards, output_dir)
    
    print("\n[T2] Speciation fractions summary...")
    generate_t2_speciation_summary(df_spec, output_dir)
    
    print("\n[T3] Posterior summary (risk metrics)...")
    generate_t3_posterior_summary(idata, output_dir)
    
    print("\n[T4] Risk ranking (top sites)...")
    generate_t4_risk_ranking(idata, output_dir, df_chem=df_chem, top_n=args.top_n)
    
    print("\n[T5] Blood Lead Level summary...")
    generate_t5_bll_summary(idata, output_dir)
    
    # Additional analyses
    print("\n" + "-" * 60)
    print("Statistical analyses...")
    print("-" * 60)
    
    print("\nSpearman correlation matrix...")
    generate_correlation_matrix(df_chem, output_dir)
    
    print("\nPCA with varimax rotation...")
    generate_pca_analysis(df_chem, output_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY TABLES COMPLETE!")
    print("=" * 60)
    print(f"\n  All tables saved to: {output_dir}/")
    print("\n  Generated files:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('.csv') or f.endswith('.txt'):
            print(f"    - {f}")
    print()
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

