#!/usr/bin/env python3
"""
entropy_hpi_peri_refined.py

Computes entropy‐based weights for trace‐metal criteria with toxicological refinements, 
then calculates:
  • Heavy Metal Pollution Index (HPI)
  • Potential Ecological Risk Index (PERI)
  • Risk categorization and uncertainty analysis

Key improvements:
  • Handles non-detects and missing data
  • Uses toxicologically appropriate reference values
  • Prevents negative/unrealistic values
  • Includes uncertainty quantification
  • Provides risk categorization

Usage:
    python entropy_hpi_peri_refined.py \
      --input chemistry.csv \
      --standards standards.csv \
      --toxicities toxicity.csv \
      --output-dir results \
      --bootstrap-samples 1000
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# ─── TOXICOLOGICAL CONSTANTS ────────────────────────────────────────

# Detection limits (μg/L) - use site-specific values if available
DEFAULT_DETECTION_LIMITS = {
    'As': 1.0, 'Cd': 0.5, 'Cr': 1.0, 'Cu': 2.0, 'Hg': 0.2, 'Pb': 1.0,
    'Co': 0.5, 'Fe': 5.0, 'Mn': 1.0, 'Ni': 1.0, 'Zn': 5.0
}

# Essential metals that require special handling
ESSENTIAL_METALS = {'Cu', 'Fe', 'Mn', 'Ni', 'Zn'}

# HPI risk categories (based on literature)
HPI_CATEGORIES = {
    'Low': (0, 30),
    'Moderate': (30, 60), 
    'High': (60, 100),
    'Very High': (100, float('inf'))
}

# PERI risk categories (Hakanson, 1980)
PERI_CATEGORIES = {
    'Low': (0, 150),
    'Moderate': (150, 300),
    'Considerable': (300, 600),
    'Very High': (600, float('inf'))
}

def validate_and_clean_data(df, detection_limits=None):
    """
    Validate input data and handle non-detects, negatives, and missing values
    """
    if detection_limits is None:
        detection_limits = DEFAULT_DETECTION_LIMITS
    
    df_clean = df.copy()
    cleaning_log = {}
    # Unit check: warn if values suggest mg/L instead of μg/L
    for col in [c for c in df_clean.columns if c.startswith('C_')]:
        max_val = df_clean[col].max()
        if max_val > 1000:
            warnings.warn(f"Column {col} has values > 1000. Please confirm units are μg/L (micrograms per liter). If your data is in mg/L, multiply by 1000 to convert.")
    
    # Get metal columns
    metal_cols = [col for col in df.columns if col.startswith('C_')]
    metals = [col.replace('C_', '') for col in metal_cols]
    
    # Get metal columns (with or without C_ prefix)
    metal_cols = [col for col in df.columns if col.startswith('C_')]
    # If no C_ columns, try to auto-detect and rename metal columns
    if not metal_cols:
        # List of known metals
        known_metals = set(DEFAULT_DETECTION_LIMITS.keys())
        found_metals = [col for col in df.columns if col in known_metals]
        # Rename them to C_<metal>
        if found_metals:
            rename_dict = {col: f"C_{col}" for col in found_metals}
            df_clean.rename(columns=rename_dict, inplace=True)
            metal_cols = [f"C_{col}" for col in found_metals]
            warnings.warn(f"Input columns {found_metals} were automatically renamed to C_<metal> format.")
    metals = [col.replace('C_', '') for col in metal_cols]

    for col in metal_cols:
        metal = col.replace('C_', '')
        original_data = df_clean[col].copy()
        # Handle missing values
        missing_count = original_data.isna().sum()
        # Handle negative values (measurement errors)
        negative_count = (original_data < 0).sum()
        df_clean.loc[df_clean[col] < 0, col] = np.nan
        # Handle non-detects (values below detection limit)
        if metal in detection_limits:
            dl = detection_limits[metal]
            nd_mask = (original_data > 0) & (original_data < dl)
            nd_count = nd_mask.sum()
            # Replace with DL/2 (standard practice)
            df_clean.loc[nd_mask, col] = dl / 2
        else:
            nd_count = 0
            dl = None
        # Handle remaining missing values with median
        if df_clean[col].isna().any():
            median_val = df_clean[col].median()
            if pd.isna(median_val):  # All values missing
                # Use detection limit or conservative estimate
                fill_val = dl / 2 if dl else 1.0
                warnings.warn(f"All values missing for {metal}, using {fill_val}")
            else:
                fill_val = median_val
            df_clean[col] = df_clean[col].fillna(fill_val)
        # Log cleaning actions
        cleaning_log[metal] = {
            'missing_original': missing_count,
            'negative_values': negative_count,
            'non_detects': nd_count,
            'detection_limit': dl,
            'final_median': df_clean[col].median(),
            'final_range': (df_clean[col].min(), df_clean[col].max())
        }
    return df_clean, cleaning_log

def compute_entropy_weights_robust(df, min_weight=0.01):
    """
    Compute entropy weights with robustness improvements:
    - Handles zeros and near-zeros properly
    - Prevents extreme weights
    - Includes uncertainty estimation
    """
    # Extract metal concentration columns
    metal_cols = [col for col in df.columns if col.startswith('C_')]
    if not metal_cols:
        raise ValueError("No valid metal concentration columns (C_<metal>) found in input data.")
    C = df[metal_cols].copy()
    
    # Rename columns to metal names
    C.columns = [col.replace('C_', '') for col in C.columns]
    
    # Handle zeros and very small values
    # Replace zeros with small positive value (1% of detection limit)
    for metal in C.columns:
        dl = DEFAULT_DETECTION_LIMITS.get(metal, 1.0)
        C.loc[C[metal] <= 0, metal] = dl * 0.01
    
    # Normalize concentrations
    row_sums = C.sum(axis=1)
    # Prevent division by zero
    row_sums = np.maximum(row_sums, 1e-10)
    
    P = C.div(row_sums, axis=0)
    
    # Compute entropy
    n_sites = C.shape[0]
    k = 1.0 / np.log(n_sites) if n_sites > 1 else 1.0
    
    # Handle log(0) by using small value
    P_log = P.copy()
    P_log[P_log <= 0] = 1e-10
    
    # Entropy calculation
    entropy = -k * (P * np.log(P_log)).sum(axis=0)
    
    # Information utility (divergence)
    divergence = 1 - entropy
    
    # Ensure positive divergence
    divergence = np.maximum(divergence, 0)
    
    if len(divergence) == 0:
        raise ValueError("No valid metals found for entropy weighting. Check your input data columns.")
    
    # Compute weights
    if divergence.sum() == 0:
        # If all divergences are zero, use equal weights (avoid division by zero)
        weights = pd.Series(1.0 / max(len(divergence), 1), index=divergence.index)
        warnings.warn("All divergences are zero, using equal weights")
    else:
        weights = divergence / divergence.sum()
    
    # Apply minimum weight constraint to prevent extreme values
    if min_weight > 0:
        n_metals = len(weights)
        max_weight = 1 - (n_metals - 1) * min_weight
        
        # Adjust weights that are too small or too large
        weights = np.maximum(weights, min_weight)
        weights = np.minimum(weights, max_weight)
        
        # Renormalize
        weights = weights / weights.sum()
    
    # Create results with uncertainty measures
    weight_stats = pd.DataFrame({
        'weight': weights,
        'entropy': entropy,
        'divergence': divergence,
        'cv_concentration': C.std() / C.mean(),  # Coefficient of variation
        'weight_stability': 1 - entropy  # Higher = more stable weight
    })
    
    return weight_stats

def compute_indices_robust(df_C, weight_stats, standards, toxicities):
    """
    Compute HPI and PERI with error handling and uncertainty quantification
    """
    # Extract weights
    weights = weight_stats['weight']
    
    # Get metal names
    metals = weights.index.tolist()
    
    # Extract concentration data
    metal_cols = [f"C_{metal}" for metal in metals if f"C_{metal}" in df_C.columns]
    available_metals = [col.replace('C_', '') for col in metal_cols]
    
    if not available_metals:
        raise ValueError("No metal concentration columns found")
    
    C = df_C[metal_cols].copy()
    C.columns = available_metals
    
    # Align standards and toxicities
    try:
        S = standards.loc[available_metals]
        T = toxicities.loc[available_metals]
        w = weights.loc[available_metals]
    except KeyError as e:
        missing = set(available_metals) - set(standards.index)
        if missing:
            raise ValueError(f"Missing standards for metals: {missing}")
        missing = set(available_metals) - set(toxicities.index)
        if missing:
            raise ValueError(f"Missing toxicity factors for metals: {missing}")
        raise e
    
    # Renormalize weights for available metals
    w = w / w.sum()
    
    # Calculate contamination factors (CF = C/S)
    CF = C.div(S, axis=1)
    
    # Handle infinite or very large CF values
    CF = CF.replace([np.inf, -np.inf], np.nan)
    max_cf = 1000  # Cap extremely high contamination factors
    CF = CF.clip(upper=max_cf)
    
    # Fill any remaining NaN with conservative estimate
    CF = CF.fillna(1.0)  # Assumes concentration equals standard
    
    # Calculate indices
    HPI = (w * CF).sum(axis=1)
    
    # For PERI, multiply by toxicity factors
    PERI_components = w * T * CF
    PERI = PERI_components.sum(axis=1)
    
    # Calculate individual metal contributions for analysis
    HPI_contributions = pd.DataFrame(w * CF)
    HPI_contributions.columns = [f"HPI_{metal}" for metal in available_metals]
    
    PERI_contributions = pd.DataFrame(w * T * CF)
    PERI_contributions.columns = [f"PERI_{metal}" for metal in available_metals]
    
    return HPI, PERI, HPI_contributions, PERI_contributions

def bootstrap_uncertainty(df_C, weight_stats, standards, toxicities, n_bootstrap=1000):
    """
    Estimate uncertainty in HPI and PERI using bootstrap resampling
    """
    n_sites = len(df_C)
    hpi_bootstrap = []
    peri_bootstrap = []
    
    for _ in range(n_bootstrap):
        # Resample sites with replacement
        bootstrap_indices = np.random.choice(n_sites, n_sites, replace=True)
        df_bootstrap = df_C.iloc[bootstrap_indices].reset_index(drop=True)
        
        try:
            # Recompute weights and indices
            weight_boot = compute_entropy_weights_robust(df_bootstrap)
            hpi_boot, peri_boot, _, _ = compute_indices_robust(
                df_bootstrap, weight_boot, standards, toxicities
            )
            
            hpi_bootstrap.append(hpi_boot.values)
            peri_bootstrap.append(peri_boot.values)
        except:
            # Skip failed bootstrap samples
            continue
    
    if not hpi_bootstrap:
        warnings.warn("Bootstrap failed, returning NaN confidence intervals")
        return None, None
    
    # Calculate confidence intervals
    hpi_bootstrap = np.array(hpi_bootstrap)
    peri_bootstrap = np.array(peri_bootstrap)
    
    hpi_ci = np.percentile(hpi_bootstrap, [2.5, 50, 97.5], axis=0)
    peri_ci = np.percentile(peri_bootstrap, [2.5, 50, 97.5], axis=0)
    
    return hpi_ci, peri_ci

def categorize_risk(values, categories):
    """Categorize risk levels based on threshold values"""
    risk_levels = []
    for val in values:
        for level, (low, high) in categories.items():
            if low <= val < high:
                risk_levels.append(level)
                break
        else:
            risk_levels.append('Very High')  # Default for extreme values
    return risk_levels

def save_comprehensive_results(df_C, weight_stats, HPI, PERI, 
                              HPI_contributions, PERI_contributions,
                              hpi_ci, peri_ci, cleaning_log, output_dir):
    """Save comprehensive results with uncertainty and diagnostics"""
    
    # 1. Weight analysis table
    weight_table = weight_stats.round(4)
    weight_table.to_csv(os.path.join(output_dir, "entropy_weights_analysis.csv"))
    
    # 2. Data cleaning log
    cleaning_df = pd.DataFrame(cleaning_log).T
    cleaning_df.to_csv(os.path.join(output_dir, "data_cleaning_log.csv"))
    
    # 3. Main results table
    results = pd.DataFrame({
        'HPI': HPI,
        'PERI': PERI,
        'HPI_risk_level': categorize_risk(HPI, HPI_CATEGORIES),
        'PERI_risk_level': categorize_risk(PERI, PERI_CATEGORIES)
    }, index=df_C.index)
    
    # Add confidence intervals if available
    if hpi_ci is not None:
        results['HPI_CI_lower'] = hpi_ci[0]
        results['HPI_CI_upper'] = hpi_ci[2]
        results['PERI_CI_lower'] = peri_ci[0]
        results['PERI_CI_upper'] = peri_ci[2]
    
    results.to_csv(os.path.join(output_dir, "hpi_peri_results.csv"))
    
    # 4. Metal contributions
    HPI_contributions.to_csv(os.path.join(output_dir, "hpi_metal_contributions.csv"))
    PERI_contributions.to_csv(os.path.join(output_dir, "peri_metal_contributions.csv"))
    
    # 5. Summary statistics
    summary_stats = pd.DataFrame({
        'HPI': [HPI.mean(), HPI.std(), HPI.min(), HPI.max()],
        'PERI': [PERI.mean(), PERI.std(), PERI.min(), PERI.max()]
    }, index=['Mean', 'Std', 'Min', 'Max'])
    summary_stats.to_csv(os.path.join(output_dir, "summary_statistics.csv"))
    
    print("Saved comprehensive results:")
    print("  - entropy_weights_analysis.csv")
    print("  - data_cleaning_log.csv") 
    print("  - hpi_peri_results.csv")
    print("  - hpi_metal_contributions.csv")
    print("  - peri_metal_contributions.csv")
    print("  - summary_statistics.csv")

def create_enhanced_plots(df_C, weight_stats, HPI, PERI, 
                         HPI_contributions, PERI_contributions,
                         hpi_ci, peri_ci, output_dir):
    """Create comprehensive visualization suite"""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Main HPI/PERI comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sites = HPI.index.astype(str)
    x = np.arange(len(sites))
    
    # HPI plot with risk categories
    bars1 = ax1.bar(x, HPI.values, alpha=0.7, color='steelblue')
    if hpi_ci is not None:
        hpi_err_lower = np.maximum(HPI.values - hpi_ci[0], 0)
        hpi_err_upper = np.maximum(hpi_ci[2] - HPI.values, 0)
        ax1.errorbar(x, HPI.values, 
                    yerr=[hpi_err_lower, hpi_err_upper],
                    fmt='none', color='black', capsize=3)
    
    # Add risk category lines
    for level, (low, high) in HPI_CATEGORIES.items():
        if high != float('inf'):
            ax1.axhline(high, color='red', linestyle='--', alpha=0.5)
            ax1.text(len(sites)-1, high, level, ha='right', va='bottom')
    
    ax1.set_xticks(x)
    # Rotate and reduce font size for readability
    ax1.set_xticklabels(sites, rotation=90, ha='right', fontsize=8)
    ax1.set_ylabel('HPI Value (log scale)')
    ax1.set_title('Heavy Metal Pollution Index (HPI)')
    ax1.grid(True, alpha=0.3)
    # Use logarithmic scale for y-axis to capture wide range
    ax1.set_yscale('log')
    
    # PERI plot with risk categories
    bars2 = ax2.bar(x, PERI.values, alpha=0.7, color='coral')
    if peri_ci is not None:
        peri_err_lower = np.maximum(PERI.values - peri_ci[0], 0)
        peri_err_upper = np.maximum(peri_ci[2] - PERI.values, 0)
        ax2.errorbar(x, PERI.values,
                    yerr=[peri_err_lower, peri_err_upper],
                    fmt='none', color='black', capsize=3)
    
    # Add risk category lines
    for level, (low, high) in PERI_CATEGORIES.items():
        if high != float('inf'):
            ax2.axhline(high, color='red', linestyle='--', alpha=0.5)
            ax2.text(len(sites)-1, high, level, ha='right', va='bottom')
    
    ax2.set_xticks(x)
    # Rotate and reduce font size for readability
    ax2.set_xticklabels(sites, rotation=90, ha='right', fontsize=8)
    ax2.set_ylabel('PERI Value (log scale)')
    ax2.set_title('Potential Ecological Risk Index (PERI)')
    ax2.grid(True, alpha=0.3)
    # Use logarithmic scale for y-axis to capture wide range
    ax2.set_yscale('log')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "hpi_peri_comparison.png"), dpi=300, bbox_inches='tight')
    
    # 2. Weight analysis plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    metals = weight_stats.index
    weights = weight_stats['weight']
    
    # Weights bar plot
    bars = ax1.bar(metals, weights, color='lightgreen', alpha=0.7)
    ax1.set_ylabel('Entropy Weight')
    ax1.set_title('Metal Weights from Entropy Analysis')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add weight values on bars
    for bar, weight in zip(bars, weights):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{weight:.3f}', ha='center', va='bottom')
    
    # Weight stability plot
    stability = weight_stats['weight_stability']
    ax2.scatter(weights, stability, alpha=0.7, s=100)
    for i, metal in enumerate(metals):
        ax2.annotate(metal, (weights.iloc[i], stability.iloc[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    ax2.set_xlabel('Weight')
    ax2.set_ylabel('Weight Stability')
    ax2.set_title('Weight vs Stability Analysis')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "weight_analysis.png"), dpi=300, bbox_inches='tight')
    
    # 3. Metal contributions heatmap
    # 3. Metal contributions heatmap
    # Increase width and rotate x ticks to avoid overlap
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # HPI contributions
    im1 = ax1.imshow(HPI_contributions.T, aspect='auto', cmap='YlOrRd')
    ax1.set_xticks(range(len(sites)))
    ax1.set_xticklabels(sites, rotation=90, ha='right', fontsize=8)
    ax1.set_yticks(range(len(HPI_contributions.columns)))
    ax1.set_yticklabels([col.replace('HPI_', '') for col in HPI_contributions.columns])
    ax1.set_title('HPI Metal Contributions by Site')
    plt.colorbar(im1, ax=ax1, label='Contribution')
    
    # PERI contributions
    im2 = ax2.imshow(PERI_contributions.T, aspect='auto', cmap='YlGnBu')
    ax2.set_xticks(range(len(sites)))
    ax2.set_xticklabels(sites, rotation=90, ha='right', fontsize=8)
    ax2.set_yticks(range(len(PERI_contributions.columns)))
    ax2.set_yticklabels([col.replace('PERI_', '') for col in PERI_contributions.columns])
    ax2.set_title('PERI Metal Contributions by Site')
    plt.colorbar(im2, ax=ax2, label='Contribution')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "metal_contributions_heatmap.png"), dpi=300, bbox_inches='tight')
    
    print("Saved enhanced plots:")
    print("  - hpi_peri_comparison.png")
    print("  - weight_analysis.png") 
    print("  - metal_contributions_heatmap.png")

def generate_summary_report(df_C, weight_stats, HPI, PERI, cleaning_log, output_dir):
    """Generate executive summary report"""
    
    report_lines = [
        "# ENTROPY-BASED POLLUTION INDEX ASSESSMENT REPORT",
        "=" * 55,
        "",
        "## Data Quality Summary",
        f"- Total sites analyzed: {len(df_C)}",
        f"- Metals analyzed: {len(weight_stats)}",
        ""
    ]
    
    # Data quality issues
    total_missing = sum(log['missing_original'] for log in cleaning_log.values())
    total_negatives = sum(log['negative_values'] for log in cleaning_log.values())
    total_nd = sum(log['non_detects'] for log in cleaning_log.values())
    
    report_lines.extend([
        "### Data Quality Issues Addressed:",
        f"- Missing values: {total_missing}",
        f"- Negative values: {total_negatives}",
        f"- Non-detects: {total_nd}",
        ""
    ])
    
    # Weight analysis
    report_lines.extend([
        "## Entropy Weight Analysis",
        "Metals ranked by information content (weight):",
        ""
    ])
    
    sorted_weights = weight_stats['weight'].sort_values(ascending=False)
    for metal, weight in sorted_weights.items():
        report_lines.append(f"- {metal}: {weight:.3f}")
    
    # Risk assessment results
    hpi_risk_counts = pd.Series(categorize_risk(HPI, HPI_CATEGORIES)).value_counts()
    peri_risk_counts = pd.Series(categorize_risk(PERI, PERI_CATEGORIES)).value_counts()
    
    report_lines.extend([
        "",
        "## Risk Assessment Results",
        "",
        "### Heavy Metal Pollution Index (HPI):",
        f"- Mean HPI: {HPI.mean():.1f}",
        f"- Range: {HPI.min():.1f} - {HPI.max():.1f}",
        ""
    ])
    
    for level, count in hpi_risk_counts.items():
        report_lines.append(f"- {level} risk sites: {count} ({count/len(HPI)*100:.1f}%)")
    
    report_lines.extend([
        "",
        "### Potential Ecological Risk Index (PERI):",
        f"- Mean PERI: {PERI.mean():.1f}",
        f"- Range: {PERI.min():.1f} - {PERI.max():.1f}",
        ""
    ])
    
    for level, count in peri_risk_counts.items():
        report_lines.append(f"- {level} risk sites: {count} ({count/len(PERI)*100:.1f}%)")
    
    # Recommendations
    high_risk_sites = HPI[HPI > 60].index.tolist()
    very_high_risk_sites = PERI[PERI > 600].index.tolist()
    
    report_lines.extend([
        "",
        "## Recommendations",
        ""
    ])
    
    if high_risk_sites:
        report_lines.append(f"- Priority sites for remediation (HPI > 60): {high_risk_sites}")
    
    if very_high_risk_sites:
        report_lines.append(f"- Immediate attention required (PERI > 600): {very_high_risk_sites}")
    
    if not high_risk_sites and not very_high_risk_sites:
        report_lines.append("- No sites require immediate intervention based on current thresholds")
    
    report_lines.extend([
        "",
        "- Monitor sites with moderate risk levels regularly",
        "- Validate results with additional sampling if high uncertainty",
        "- Consider site-specific background concentrations for essential metals",
        ""
    ])
    
    # Write report
    with open(os.path.join(output_dir, "assessment_summary_report.txt"), 'w') as f:
        f.write('\n'.join(report_lines))
    
    print("Saved assessment_summary_report.txt")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="CSV with C_<metal> columns")
    p.add_argument("--standards", required=True,
                   help="CSV with columns metal,S_i")
    p.add_argument("--toxicities", required=True,
                   help="CSV with columns metal,T_i")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--bootstrap-samples", type=int, default=1000,
                   help="Number of bootstrap samples for uncertainty")
    p.add_argument("--detection-limits", 
                   help="CSV with metal,DL columns (optional)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    df_C = pd.read_csv(args.input, index_col=0)
    standards = pd.read_csv(args.standards, index_col=0)["S_i"]
    toxicities = pd.read_csv(args.toxicities, index_col=0)["T_i"]
    
    # Load detection limits if provided
    detection_limits = DEFAULT_DETECTION_LIMITS
    if args.detection_limits:
        dl_df = pd.read_csv(args.detection_limits, index_col=0)
        detection_limits.update(dl_df["DL"].to_dict())

    try:
        # 1. Clean and validate data
        df_clean, cleaning_log = validate_and_clean_data(df_C, detection_limits)
        
        # 2. Compute entropy weights with robustness
        weight_stats = compute_entropy_weights_robust(df_clean)
        
        # 3. Compute HPI and PERI with error handling
        HPI, PERI, HPI_contributions, PERI_contributions = compute_indices_robust(
            df_clean, weight_stats, standards, toxicities
        )
        
        # 4. Bootstrap uncertainty analysis
        print("Computing uncertainty estimates...")
        hpi_ci, peri_ci = bootstrap_uncertainty(
            df_clean, weight_stats, standards, toxicities, args.bootstrap_samples
        )
        
        # 5. Save comprehensive results
        save_comprehensive_results(
            df_clean, weight_stats, HPI, PERI, 
            HPI_contributions, PERI_contributions,
            hpi_ci, peri_ci, cleaning_log, args.output_dir
        )
        
        # 6. Create enhanced visualizations
        create_enhanced_plots(
            df_clean, weight_stats, HPI, PERI,
            HPI_contributions, PERI_contributions,
            hpi_ci, peri_ci, args.output_dir
        )
        
        # 7. Generate summary report
        generate_summary_report(
            df_clean, weight_stats, HPI, PERI, cleaning_log, args.output_dir
        )
        
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
