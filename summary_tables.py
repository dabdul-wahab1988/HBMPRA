"""
Generates summary CSVs for:
T2: Measured conce    # T4: True posterior summary (3%, median, 94%, exceedance) from full trace
    import arviz as az
    # Load posterior draws
    idata = az.from_netcdf(hpi_fp.replace('hpi_peri_results.csv', '../run_2025/trace.nc'))
    records = []
    # Threshold for exceedance (e.g., hazard index > 1)
    thr = 1.0
    demographics = ['Adults', 'Children', 'Teens', 'Pregnant']
    for var in idata.posterior.data_vars:
        arr = idata.posterior[var]
        if 'group' in arr.dims and arr.sizes['group'] == 4:
            # Per demographic
            for i, demo in enumerate(demographics):
                sub_arr = arr.isel(group=i)
                if 'site' in sub_arr.dims:
                    sub_arr = sub_arr.mean(dim='site')  # Average over sites if per site
                q_low = float(sub_arr.quantile(0.03, dim=['chain', 'draw']))
                q_med = float(sub_arr.quantile(0.5, dim=['chain', 'draw']))
                q_high = float(sub_arr.quantile(0.94, dim=['chain', 'draw']))
                p_ex = float((sub_arr > thr).mean(dim=['chain', 'draw']))
                records.append({'Metric': var,
                                'Demographic': demo,
                                '3%': q_low,
                                'Median': q_med,
                                '94%': q_high,
                                'P(>thr)': p_ex})
    df_t4 = pd.DataFrame(records)
    # Pivot to match original format: Metric, then Adults 3%, Adults Median, etc.
    df_pivot = df_t4.pivot(index='Metric', columns='Demographic')
    # Flatten column names
    df_pivot.columns = [f"{demo} {stat}" for stat, demo in df_pivot.columns]
    df_pivot.reset_index(inplace=True)
    df_t4 = df_pivotedian, mean, min, max) and comparison to WHO guidelines.
T3: Speciation summary (bioavailable fractions) summary stats per metal.
T4: Posterior summary (quantiles and exceedance probabilities) from summary_compact.csv.
T5: Entropy-HPI/PERI summary: top N sites by HPI with uncertainty.
Additional: Spearman's correlation matrix, PCA with varimax rotation, KMO and Bartlett's sampling adequacy.
"""
import re
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from factor_analyzer.rotator import Rotator

def main():
    # Paths
    base = r"c:/Users/DicksonAbdul-Wahab/Documents/Prof Abass"
    meas_fp = f"{base}/measured_concentrations.csv"
    who_src_fp = f"{base}/standards_sources.csv"
    spec_frac_fp = f"{base}/results/table_species_fractions.csv"
    post_fp = f"{base}/results/run_2025/summary_compact.csv"
    hpi_fp = f"{base}/results/entropy/hpi_peri_results.csv"

    # T2: measured concentrations summary
    df_meas = pd.read_csv(meas_fp)
    # Identify metal columns (exclude Site, coords, and other params)
    metals = [c for c in df_meas.columns if c not in ['Site','Source','Latitude','Longitude','Elev','EC','TDS','Sal','pH','Eh','Temp']]
    df_meas_stats = df_meas[metals].agg(['median','mean','min','max']).T
    df_meas_stats.index.name = 'Metal'
    df_meas_stats.reset_index(inplace=True)

    # Also include summary for water quality params: EC, TDS, pH, Eh
    water_params = ['EC', 'TDS', 'pH', 'Eh']
    df_water_stats = df_meas[water_params].agg(['median','mean','min','max']).T
    df_water_stats.index.name = 'Metal'  # Reuse column name for consistency
    df_water_stats.reset_index(inplace=True)

    # Combine metals and water params
    df_t2 = pd.concat([df_meas_stats, df_water_stats], ignore_index=True)

    # WHO guidelines
    df_who = pd.read_csv(who_src_fp)
    # Extract metal symbol from analyte e.g. 'Arsenic (As)'
    df_who['Metal'] = df_who['analyte'].str.extract(r"\(([^)]+)\)")
    df_who = df_who[['Metal','who_gv_ugL']]
    df_t2 = pd.merge(df_t2, df_who, on='Metal', how='left')
    # Ensure who_gv_ugL is string type
    df_t2['who_gv_ugL'] = df_t2['who_gv_ugL'].astype(str)
    # Set guidelines for water params
    df_t2.loc[df_t2['Metal'] == 'pH', 'who_gv_ugL'] = '6.5-8.5'
    df_t2.loc[df_t2['Metal'] == 'TDS', 'who_gv_ugL'] = '1000'  # mg/L, WHO acceptable limit
    df_t2.loc[df_t2['Metal'] == 'EC', 'who_gv_ugL'] = '1500'   # μS/cm, common drinking water limit
    # Compute percentage of samples exceeding WHO guideline for each metal
    exceedance_values = []
    for metal, gv in zip(df_t2['Metal'], df_t2['who_gv_ugL']):
        if metal == 'pH':
            # For pH, exceedance is outside 6.5-8.5
            exceedance_values.append(((df_meas['pH'] < 6.5) | (df_meas['pH'] > 8.5)).mean() * 100)
        elif pd.isna(float(gv)) if gv != '6.5-8.5' else True or metal in water_params:
            exceedance_values.append(None)
        else:
            gv_float = float(gv)
            exceedance_values.append((df_meas[metal] > gv_float).mean() * 100)
    df_t2['exceedance_pct'] = exceedance_values
    df_t2.to_csv(f"{base}/results/T2_measured_summary.csv", index=False)

    # T3: Speciation summary (bioavailable fractions)
    df_frac = pd.read_csv(spec_frac_fp)
    frac_cols = [c for c in df_frac.columns if c.startswith('frac_')]
    df_frac_stats = df_frac[frac_cols].agg(['median','mean','min','max']).T
    df_frac_stats.index = df_frac_stats.index.str.replace('frac_','')
    df_frac_stats.index.name = 'Metal'
    df_frac_stats.reset_index(inplace=True)
    df_frac_stats.to_csv(f"{base}/results/T3_speciation_summary.csv", index=False)

    # T4: True posterior summary (3%, median, 94%, exceedance) from full trace
    import arviz as az
    # Load posterior draws
    idata = az.from_netcdf(hpi_fp.replace('hpi_peri_results.csv', '../run_2025/trace.nc'))
    records = []
    # Threshold for exceedance (e.g., hazard index > 1)
    thr = 1.0
    demographics = ['Adults', 'Children', 'Teens', 'Pregnant']
    for var in idata.posterior.data_vars:
        arr = idata.posterior[var]
        if 'group' in arr.dims and arr.sizes['group'] == 4:
            # Per demographic
            for i, demo in enumerate(demographics):
                sub_arr = arr.isel(group=i)
                if 'site' in sub_arr.dims:
                    sub_arr = sub_arr.mean(dim='site')  # Average over sites if per site
                # Set threshold based on metric
                if var.startswith('HI_'):
                    thr = 1.0
                elif var.startswith('CR_'):
                    thr = 1e-6
                elif var == 'BLL':
                    thr = 3.5
                else:
                    thr = 1.0  # default
                q_low = sub_arr.quantile(0.03, dim=sub_arr.dims).item()
                q_med = sub_arr.quantile(0.5, dim=sub_arr.dims).item()
                q_high = sub_arr.quantile(0.94, dim=sub_arr.dims).item()
                p_ex = (sub_arr > thr).mean(dim=sub_arr.dims).item()
                records.append({'Metric': var,
                                'Demographic': demo,
                                '3%': q_low,
                                'Median': q_med,
                                '94%': q_high,
                                'P(>thr)': p_ex})
    df_t4 = pd.DataFrame(records)
    # Pivot to match original format: Metric, then Adults 3%, Adults Median, etc.
    df_pivot = df_t4.pivot(index='Metric', columns='Demographic')
    # Flatten column names
    df_pivot.columns = [f"{demo} {stat}" for stat, demo in df_pivot.columns]
    df_pivot.reset_index(inplace=True)
    df_t4 = df_pivot
    try:
        df_t4.to_csv(f"{base}/results/T4_posterior_summary.csv", index=False)
    except PermissionError:
        print("T4 file is open; saving as T4_posterior_summary_new.csv")
        df_t4.to_csv(f"{base}/results/T4_posterior_summary_new.csv", index=False)

    # T5: Entropy-HPI/PERI summary top N sites
    df_hpi = pd.read_csv(hpi_fp)
    N = 5
    df_top = df_hpi.sort_values('HPI', ascending=False).head(N)
    df_top.to_csv(f"{base}/results/T5_entropy_top{N}_HPI.csv", index=False)

    # Additional analyses: Spearman's correlation, PCA with varimax, KMO, Bartlett's
    numeric_cols = metals + water_params
    df_numeric = df_meas[numeric_cols].dropna()

    # Log transform metals (assuming positive values)
    df_numeric[metals] = np.log1p(df_numeric[metals])

    # Standardize data for PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)

    # Spearman's correlation matrix (on original data)
    corr_matrix = df_numeric.corr(method='spearman')
    corr_matrix.to_csv(f"{base}/results/Correlation_matrix.csv")

    # KMO and Bartlett's test (on scaled data)
    kmo_all, kmo_model = calculate_kmo(df_scaled)
    bartlett_stat, bartlett_p = calculate_bartlett_sphericity(df_scaled)
    with open(f"{base}/results/KMO_Bartlett.txt", 'w') as f:
        f.write(f"KMO overall: {kmo_model}\n")
        f.write(f"Bartlett's test statistic: {bartlett_stat}, p-value: {bartlett_p}\n")

    # PCA with varimax rotation (on scaled data)
    pca = PCA()
    pca.fit(df_scaled)
    eigenvalues = pca.explained_variance_
    explained_var_ratio = pca.explained_variance_ratio_
    cumulative_var = (explained_var_ratio.cumsum() * 100)[:len(eigenvalues[eigenvalues > 1])]  # Only for PCs >1
    loadings = pca.components_.T  # (n_features, n_components)

    rotator = Rotator(method='varimax')
    rotated_loadings = rotator.fit_transform(loadings)

    # Select PCs with eigenvalue > 1
    n_pcs = sum(eigenvalues > 1)
    selected_loadings = rotated_loadings[:, :n_pcs]
    selected_eigenvalues = eigenvalues[:n_pcs]
    selected_cum_var = cumulative_var

    # Create DataFrame
    df_loadings = pd.DataFrame(selected_loadings, index=numeric_cols, columns=[f'PC{i+1}' for i in range(n_pcs)])
    df_loadings['Variable'] = df_loadings.index
    df_loadings = df_loadings[['Variable'] + [f'PC{i+1}' for i in range(n_pcs)]]

    # Add Eigenvalue row
    eigen_row = pd.DataFrame([['Eigenvalue'] + list(selected_eigenvalues)], columns=df_loadings.columns)
    df_loadings = pd.concat([df_loadings, eigen_row], ignore_index=True)

    # Add Cumulative %Variance row
    cum_row = pd.DataFrame([['Cumulative %Variance'] + list(selected_cum_var)], columns=df_loadings.columns)
    df_loadings = pd.concat([df_loadings, cum_row], ignore_index=True)

    df_loadings.to_csv(f"{base}/results/PCA_results.csv", index=False)

if __name__ == '__main__':
    main()
