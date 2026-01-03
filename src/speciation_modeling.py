#!/usr/bin/env python3
"""
speciation_modeling.py

Performs speciation modeling of trace metals using PHREEQC and generates:
  • Table of thermodynamic species fractions per site
  • Figure of stacked‐bar speciation profiles
Usage:
    python speciation_modeling.py --input chemistry.csv --output-dir results
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get the directory where this script lives (src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Project root is parent of src/
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# Database directory
DATABASE_DIR = os.path.join(PROJECT_ROOT, "database")

# ─── PHREEQC SETUP ────────────────────────────────────────────────────────────
# Try multiple PHREEQC Python bindings in order of preference
PHREEQC_BACKEND = None
IPhreeqc = None

try:
    # Try phreeqpython first (most modern and well-maintained)
    import phreeqpython
    PHREEQC_BACKEND = 'phreeqpython'
except ImportError:
    try:
        # Try phreeqpy with DLL interface
        from phreeqpy.iphreeqc.phreeqc_dll import IPhreeqc as IPhreeqc_dll
        IPhreeqc = IPhreeqc_dll
        PHREEQC_BACKEND = 'phreeqpy_dll'
    except ImportError:
        try:
            # Try phreeqpy with COM interface (Windows)
            from phreeqpy.iphreeqc.phreeqc_com import IPhreeqc as IPhreeqc_com
            IPhreeqc = IPhreeqc_com
            PHREEQC_BACKEND = 'phreeqpy_com'
        except ImportError:
            # No PHREEQC backend available - will use simplified speciation
            PHREEQC_BACKEND = None
            print("WARNING: No PHREEQC backend found (phreeqpython or phreeqpy).")
            print("Will use simplified speciation based on total concentrations.")
            print("For full thermodynamic speciation, install: pip install phreeqpython")

SPECIES = {
    'As':  'H2AsO4-',   # Arsenate (As V) - more toxic than arsenite, dominant in oxic water
    'Cd':  'Cd+2',      # Cadmium ion - most bioavailable form
    'Cu':  'Cu+2',      # Cupric ion - most toxic form
    'Pb':  'Pb+2',      # Lead ion - most bioavailable and toxic
}

# Chromium speciation - input total Cr, output CrVI and CrIII species
CHROMIUM_INPUT = 'Cr'
CHROMIUM_SPECIES = {
    'CrVI':  'CrO4-2',  # Hexavalent chromium (Cr VI) - highly toxic and carcinogenic
    'CrIII': 'Cr+3',    # Trivalent chromium (Cr III) - essential nutrient, less toxic
}

# Mercury speciation - input total Hg, output inorganic Hg species only
# NOTE: Methylmercury (MeHg) is NOT included because:
#   - MeHg is formed by BIOLOGICAL processes (sulfate-reducing bacteria)
#   - It cannot be predicted by thermodynamic equilibrium models
#   - MeHg must be measured directly in the field
#   - For risk assessment, apply literature MeHg fractions (typically 1-10% of total Hg)
MERCURY_INPUT = 'Hg'
MERCURY_SPECIES = {
    'Hg(II)': 'Hg+2',     # Mercuric ion (Hg II) - highly toxic, bioavailable
    'Hg(0)':  'Hg',       # Elemental mercury - volatile but less bioavailable in water
}

# Include all measured metals
MINOR_METALS = {
    'Co': 'Co+2',       # Cobalt ion - most bioavailable
    'Fe': 'Fe+2',       # Ferrous ion (Fe II) - database default, will oxidize to Fe+3 in oxic water
    'Mn': 'Mn+2',       # Manganous ion (Mn II) - most common and bioavailable
    'Ni': 'Ni+2',       # Nickel ion - most toxic aqueous form
    'Zn': 'Zn+2',       # Zinc ion - most bioavailable form
    'Al': 'Al+3',       # Aluminum ion - most toxic form (neurotoxic)
    'V': 'VO2+',        # Vanadyl (V V) - database primary species, converts to HVO4-2 in equilibrium
    # New metals added Dec 2024
    'Sb': 'Sb(OH)3',   # Antimonite - database master species (neutral Sb(III))
    'Ba': 'Ba+2',       # Barium ion - most bioavailable
    'Be': 'Be+2',       # Beryllium ion - toxic form
    'Mo': 'MoO4-2',     # Molybdate - dominant Mo species
    'Se': 'SeO4-2',     # Selenate - dominant Se species in oxic water
    'Ag': 'Ag+',        # Silver ion - most bioavailable
    'Tl': 'Tl+',        # Thallous ion - most toxic form
    'U':  'UO2+2',      # Uranyl ion - most common U species in oxic groundwater
}
SPECIES.update(MINOR_METALS)

# Atomic weights in g/mol for conversion from μg/L to mmol/L
ATOMIC_WEIGHTS = {
    'As': 74.92,
    'Cd': 112.41,
    'Cr': 52.00,   # Total chromium
    'Cu': 63.55,
    'Hg': 200.59,
    'Pb': 207.2,
    'Co': 58.93,
    'Fe': 55.85,
    'Mn': 54.94,
    'Ni': 58.69,
    'Zn': 65.38,
    'Al': 26.98,
    'V': 50.94,
    # New metals added Dec 2024
    'Sb': 121.76,
    'Ba': 137.33,
    'Be': 9.01,
    'Mo': 95.95,
    'Se': 78.97,
    'Ag': 107.87,
    'Tl': 204.38,
    'U':  238.03,
}


def has_total_column(df, metal):
    """Return True if a total concentration column for the metal exists."""
    return f"C_{metal}" in df.columns or metal in df.columns

def run_phreeqc_speciation(df, database=None):
    """
    For each row in `df`, runs PHREEQC SOLUTION and returns free‐ion molalities.
    If pH and Eh are not provided, uses default values for natural surface waters.
    
    Supports multiple PHREEQC backends:
    - phreeqpython (modern, recommended)
    - phreeqpy with DLL interface
    - phreeqpy with COM interface (Windows)
    - Simplified estimation (fallback when no PHREEQC available)
    """
    # Default database path in project database/ folder
    if database is None:
        database = os.path.join(DATABASE_DIR, "PHREEQC_ThermoddemV1.10_15Dec2020.dat")
    elif not os.path.isabs(database):
        # Convert relative path to absolute path based on database directory
        database = os.path.join(DATABASE_DIR, os.path.basename(database))
    
    # Verify database exists
    if not os.path.exists(database):
        raise FileNotFoundError(
            f"PHREEQC database file not found: {database}\n"
            f"For speciation modeling, ensure the thermodynamic database is present in:\n"
            f"  {DATABASE_DIR}"
        )
    
    if PHREEQC_BACKEND is None:
        raise RuntimeError(
            "PHREEQC is REQUIRED for speciation modeling (scientific rigor).\n"
            "No simplified fallback is available.\n\n"
            "Install PHREEQC Python binding with:\n"
            "    pip install phreeqpython\n\n"
            "Or skip speciation by selecting 'No' when prompted."
        )
    
    if PHREEQC_BACKEND == 'phreeqpython':
        return run_phreeqpython_speciation(df, database)
    else:
        # phreeqpy_dll or phreeqpy_com
        return run_phreeqpy_speciation(df, database)


def run_phreeqpython_speciation(df, database):
    """
    Run speciation using phreeqpython library.
    """
    from phreeqpython import PhreeqPython
    
    # Check if pH and Eh columns exist; use defaults for natural waters if not
    has_pH = 'pH' in df.columns
    has_Eh = 'Eh' in df.columns
    
    # Default values for natural surface waters (oxic, circumneutral)
    DEFAULT_pH = 7.0
    DEFAULT_Eh = 300  # mV
    
    if not has_pH or not has_Eh:
        print(f"WARNING: pH and/or Eh not found in input data.")
        print(f"Using default values for natural surface waters:")
        print(f"  - pH: {DEFAULT_pH} (neutral, typical for natural waters)")
        print(f"  - Eh: {DEFAULT_Eh} mV (oxic conditions)")
    
    # Initialize PhreeqPython with database
    pp = PhreeqPython(database=database)
    
    # Create bioavailable DataFrame
    all_metals = list(SPECIES.keys()) + list(CHROMIUM_SPECIES.keys()) + list(MERCURY_SPECIES.keys())
    bio = pd.DataFrame(index=df.index, columns=[f"C_bio_{m}" for m in all_metals])
    
    # Store all species data for full output
    all_species_data = {}
    
    for idx, row in df.iterrows():
        pH_val = row['pH'] if has_pH else DEFAULT_pH
        pe_val = (row['Eh'] if has_Eh else DEFAULT_Eh) / 59.16  # Convert mV to pe
        
        # Build solution dictionary - phreeqpython expects numeric values in mmol/kgw
        sol_dict = {'pH': pH_val, 'pe': pe_val, 'units': 'mmol/kgw'}
        
        # Add major ions (assuming mg/L input, convert if needed)
        for ion in ['Ca', 'Mg', 'Na', 'K', 'Cl']:
            if ion in row and not pd.isna(row[ion]):
                sol_dict[ion] = float(row[ion])  # Assumed mg/L
        
        # Handle sulfate (may be as S or SO4)
        if 'SO4' in row and not pd.isna(row['SO4']):
            sol_dict['S(6)'] = float(row['SO4'])  # mg/L as SO4
        elif 'S' in row and not pd.isna(row['S']):
            sol_dict['S(6)'] = float(row['S'])
        
        # Add metals (convert μg/L to mmol/kgw)
        # First need to know the element name for PHREEQC
        metal_element_map = {
            'As': 'As',
            'Cd': 'Cd',
            'Cu': 'Cu',
            'Pb': 'Pb',
            'Co': 'Co',
            'Fe': 'Fe',
            'Mn': 'Mn',
            'Ni': 'Ni',
            'Zn': 'Zn',
            'Al': 'Al',
            'V': 'V',
        }
        
        for m in SPECIES:
            raw = row.get(f"C_{m}", row.get(m, np.nan))
            if not pd.isna(raw) and m in ATOMIC_WEIGHTS:
                tot = raw * 1e-3 / ATOMIC_WEIGHTS[m]  # μg/L to mmol/kgw
                element = metal_element_map.get(m, m)
                sol_dict[element] = tot
        
        # Add chromium
        raw_cr = row.get("C_Cr", row.get("Cr", np.nan))
        if not pd.isna(raw_cr):
            tot_cr = raw_cr * 1e-3 / ATOMIC_WEIGHTS['Cr']
            sol_dict['Cr'] = tot_cr
        
        # Add mercury
        raw_hg = row.get("C_Hg", row.get("Hg", np.nan))
        if not pd.isna(raw_hg):
            tot_hg = raw_hg * 1e-3 / ATOMIC_WEIGHTS['Hg']
            sol_dict['Hg'] = tot_hg
        
        # Create solution and run speciation
        try:
            solution = pp.add_solution(sol_dict)
            
            # Extract species concentrations - phreeqpython stores species as dict
            species_dict = solution.species
            
            # Extract main species
            for m, sp_name in SPECIES.items():
                try:
                    molality = species_dict.get(sp_name, 0)
                    bio.loc[idx, f"C_bio_{m}"] = molality
                    key = f"m_{sp_name}"
                    if key not in all_species_data:
                        all_species_data[key] = {}
                    all_species_data[key][idx] = molality
                except:
                    bio.loc[idx, f"C_bio_{m}"] = np.nan
            
            # Extract chromium species
            for m, sp_name in CHROMIUM_SPECIES.items():
                try:
                    molality = species_dict.get(sp_name, 0)
                    bio.loc[idx, f"C_bio_{m}"] = molality
                    key = f"m_{sp_name}"
                    if key not in all_species_data:
                        all_species_data[key] = {}
                    all_species_data[key][idx] = molality
                except:
                    bio.loc[idx, f"C_bio_{m}"] = np.nan
            
            # Extract mercury species
            for m, sp_name in MERCURY_SPECIES.items():
                try:
                    molality = species_dict.get(sp_name, 0)
                    bio.loc[idx, f"C_bio_{m}"] = molality
                    key = f"m_{sp_name}"
                    if key not in all_species_data:
                        all_species_data[key] = {}
                    all_species_data[key][idx] = molality
                except:
                    bio.loc[idx, f"C_bio_{m}"] = np.nan
                    
        except Exception as e:
            print(f"Warning: PHREEQC error for sample {idx}: {e}")
            for m in all_metals:
                bio.loc[idx, f"C_bio_{m}"] = np.nan
    
    # Build full species DataFrame
    full_df = pd.DataFrame(all_species_data, index=df.index)
    
    return bio.astype(float), full_df


def run_phreeqpy_speciation(df, database):
    """
    Run speciation using phreeqpy library (original implementation).
    """
    phreeqc = IPhreeqc()
    phreeqc.load_database(database)
    
    # Check if pH and Eh columns exist; use defaults for natural waters if not
    has_pH = 'pH' in df.columns
    has_Eh = 'Eh' in df.columns
    
    # Default values for natural surface waters (oxic, circumneutral)
    DEFAULT_pH = 7.0   # Neutral pH
    DEFAULT_Eh = 300   # mV, oxic conditions (typical for surface water in equilibrium with atmosphere)
    
    if not has_pH or not has_Eh:
        print(f"WARNING: pH and/or Eh not found in input data.")
        print(f"Using default values for natural surface waters:")
        print(f"  - pH: {DEFAULT_pH} (neutral, typical for natural waters)")
        print(f"  - Eh: {DEFAULT_Eh} mV (oxic conditions, typical for oxygenated surface water)")
        print(f"For more accurate speciation modeling, provide measured pH and Eh values.")

    # Build PHREEQC input: first, SELECTED_OUTPUT block for free-ion molalities
    # Include main species, chromium species, and mercury species
    all_species = list(SPECIES.values()) + list(CHROMIUM_SPECIES.values()) + list(MERCURY_SPECIES.values())
    selected_lines = ["SELECTED_OUTPUT",
                      f"    -molalities {' '.join(all_species)}",
                      "END"]
    # Then add SOLUTION blocks for each row
    input_blocks = []
    # Use numeric solution number separate from DataFrame index
    for sol_num, (idx, row) in enumerate(df.iterrows(), start=1):
        # Use provided pH/Eh if available, otherwise use defaults
        pH_val = row['pH'] if has_pH else DEFAULT_pH
        Eh_val = row['Eh'] if has_Eh else DEFAULT_Eh
        
        lines = [f"SOLUTION {sol_num}",
                 f"    pH   {pH_val:.2f}",
                 f"    Eh   {Eh_val:.0f}"]
        for ion in ['Ca', 'Mg', 'Na', 'K', 'Cl', 'SO4']:
            if ion in row:
                lines.append(f"    {ion}   {row[ion]:.4g}")
        for m in SPECIES:
            # Try C_<metal> first, else fallback to <metal>
            raw = row.get(f"C_{m}", row.get(m, np.nan))
            # convert μg/L to mmol/L: (μg/L * 1e-6 g/μg) / atomic_weight * 1e3 mmol/mol = raw*1e-3/atomic_weight
            if not np.isnan(raw):
                tot = raw * 1e-3 / ATOMIC_WEIGHTS[m]
            else:
                tot = np.nan
            lines.append(f"    {m}   {tot:.4g}")
        
        # Add chromium input
        raw_cr = row.get("C_Cr", row.get("Cr", np.nan))
        if not np.isnan(raw_cr):
            tot_cr = raw_cr * 1e-3 / ATOMIC_WEIGHTS['Cr']
            lines.append(f"    Cr   {tot_cr:.4g}")
        
        # Add mercury input
        raw_hg = row.get("C_Hg", row.get("Hg", np.nan))
        if not np.isnan(raw_hg):
            tot_hg = raw_hg * 1e-3 / ATOMIC_WEIGHTS['Hg']
            lines.append(f"    Hg   {tot_hg:.4g}")
        
        lines.append("END")
        input_blocks.append("\n".join(lines))

    # Combine SELECTED_OUTPUT and SOLUTION blocks and run once
    full_input = "\n".join(selected_lines + input_blocks)
    phreeqc.run_string(full_input)

    # Extract results: first row of array is headings, following rows are data
    arr = phreeqc.get_selected_output_array()
    headings = arr[0]
    data_rows = arr[1:]
    
    # Create bio DataFrame with all species including chromium and mercury
    all_metals = list(SPECIES.keys()) + list(CHROMIUM_SPECIES.keys()) + list(MERCURY_SPECIES.keys())
    bio = pd.DataFrame(index=df.index, columns=[f"C_bio_{m}" for m in all_metals])
    
    # Process main species
    for i, m in enumerate(SPECIES):
        sp_name = SPECIES[m]
        matches = [j for j, h in enumerate(headings) if sp_name in h]
        if matches:
            col_idx = matches[0]
            values = np.array([row[col_idx] for row in data_rows], dtype=float)
            bio[f"C_bio_{m}"] = values
        else:
            bio[f"C_bio_{m}"] = np.nan
    
    # Process chromium species
    for m in CHROMIUM_SPECIES:
        sp_name = CHROMIUM_SPECIES[m]
        matches = [j for j, h in enumerate(headings) if sp_name in h]
        if matches:
            col_idx = matches[0]
            values = np.array([row[col_idx] for row in data_rows], dtype=float)
            bio[f"C_bio_{m}"] = values
        else:
            bio[f"C_bio_{m}"] = np.nan
    
    # Process mercury species
    for m in MERCURY_SPECIES:
        sp_name = MERCURY_SPECIES[m]
        matches = [j for j, h in enumerate(headings) if sp_name in h]
        if matches:
            col_idx = matches[0]
            values = np.array([row[col_idx] for row in data_rows], dtype=float)
            bio[f"C_bio_{m}"] = values
        else:
            bio[f"C_bio_{m}"] = np.nan
    # Also return full species molalities DataFrame
    # Build full species DataFrame from selected output array
    # Build full species DataFrame and coerce all columns to numeric (non-numeric become NaN)
    full_df = pd.DataFrame(data_rows, index=df.index, columns=headings)
    full_df = full_df.apply(pd.to_numeric, errors='coerce')
    return bio.astype(float), full_df

def compute_species_fractions(df_total, df_free, metals=None, chromium_species=None, mercury_species=None):
    """Compute speciation fractions only for metals present in the input."""
    metals = metals or list(SPECIES.keys())
    chromium_species = chromium_species or list(CHROMIUM_SPECIES.keys())
    mercury_species = mercury_species or list(MERCURY_SPECIES.keys())

    frac = pd.DataFrame(index=df_total.index)

    # Process main species that have totals
    for m in metals:
        tot_col = f"C_{m}"
        if tot_col in df_total.columns:
            raw_tot = df_total[tot_col]
        elif m in df_total.columns:
            raw_tot = df_total[m]
        else:
            # Skip metals that were not provided at all
            continue
        tot_mol = raw_tot * 1e-6 / ATOMIC_WEIGHTS[m]
        bio = df_free.get(f"C_bio_{m}", pd.Series(0, index=df_total.index))
        frac[f"frac_{m}"] = np.where(tot_mol > 0, bio / tot_mol, 0.0)

    # Chromium (only if totals exist)
    if chromium_species:
        if "C_Cr" in df_total.columns:
            raw_tot_cr = df_total["C_Cr"]
        elif "Cr" in df_total.columns:
            raw_tot_cr = df_total["Cr"]
        else:
            raw_tot_cr = None
        if raw_tot_cr is not None:
            tot_mol_cr = raw_tot_cr * 1e-6 / ATOMIC_WEIGHTS['Cr']
            for m in chromium_species:
                bio = df_free.get(f"C_bio_{m}", pd.Series(0, index=df_total.index))
                frac[f"frac_{m}"] = np.where(tot_mol_cr > 0, bio / tot_mol_cr, 0.0)

    # Mercury (only if totals exist)
    if mercury_species:
        if "C_Hg" in df_total.columns:
            raw_tot_hg = df_total["C_Hg"]
        elif "Hg" in df_total.columns:
            raw_tot_hg = df_total["Hg"]
        else:
            raw_tot_hg = None
        if raw_tot_hg is not None:
            tot_mol_hg = raw_tot_hg * 1e-6 / ATOMIC_WEIGHTS['Hg']
            for m in mercury_species:
                bio = df_free.get(f"C_bio_{m}", pd.Series(0, index=df_total.index))
                frac[f"frac_{m}"] = np.where(tot_mol_hg > 0, bio / tot_mol_hg, 0.0)

    return frac

def save_fraction_table(frac_df, output_dir):
    """Writes the species fractions table to CSV."""
    path = os.path.join(output_dir, "table_species_fractions.csv")
    frac_df.to_csv(path, float_format="%.8e")
    print(f"Saved species fractions table to {path}")

def save_bioavailable_table(df_free, output_dir):
    """Writes the bioavailable (free-ion) concentrations table to CSV."""
    path = os.path.join(output_dir, "table_bioavailable_concentrations.csv")
    df_free.to_csv(path, float_format="%.8e")
    print(f"Saved bioavailable concentrations table to {path}")

def plot_speciation_profiles(frac_df, output_dir, df_total, df_free_local, selected_species, species_tex_map):
    """Plot speciation fraction profiles with improved handling of low concentrations"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Filter out metals with very low or constant fractions for cleaner visualization
    # Keep metals that show meaningful variation (coefficient of variation > 0.1)
    metals_to_plot = []
    excluded_metals = []
    for col in frac_df.columns:
        if col.startswith('frac_'):
            metal = col[5:]  # Remove 'frac_' prefix
            values = frac_df[col].values
            # Check if values show meaningful variation
            if len(values) > 1 and not np.allclose(values, values[0], rtol=0.01):
                values_array = np.array(values)
                cv = np.std(values_array) / np.mean(values_array) if np.mean(values_array) > 0 else 0
                if cv > 0.01:  # Coefficient of variation > 1%
                    metals_to_plot.append(metal)
                else:
                    excluded_metals.append(metal)
            else:
                excluded_metals.append(metal)

    print(f"Plotting speciation profiles for metals with variable fractions: {metals_to_plot}")
    if excluded_metals:
        print(f"Excluded metals (constant/low variation fractions): {excluded_metals}")
        print("These metals typically have:")
        print("- Very low concentrations (often below detection limits)")
        print("- Non-detect values replaced with DL/2, leading to similar speciation")
        print("- Limited variation in water chemistry across sites")
        print("- This is normal behavior for trace metals in environmental samples")

    if not metals_to_plot:
        print("No metals show sufficient variation in speciation fractions")
        return metals_to_plot  # Return empty list instead of None

    n_metals = len(metals_to_plot)
    if n_metals <= 3:
        nrows, ncols = 1, n_metals
    elif n_metals <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = (n_metals + 3) // 4, 4

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if n_metals == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, metal in enumerate(metals_to_plot):
        if i >= len(axes):
            break

        ax = axes[i]
        frac_col = f"frac_{metal}"

        # Get the LaTeX label for the metal key
        label = species_tex_map.get(metal, metal)

        # Plot fraction values
        values = frac_df[frac_col].values

        # Special handling for mercury fallback
        is_mercury_fallback = metal.startswith('Hg') and 'fallback' in selected_species.get(metal, '')
        if is_mercury_fallback:
            bio_col = f"C_bio_{metal}"
            if bio_col in df_free_local.columns:
                values = df_free_local[bio_col].values
            ylabel_text = f"{label} bioavailable concentration"
            title_text = ylabel_text
        else:
            ylabel_text = f"{label} fraction"
            title_text = label

        # Use scatter plot instead of line plot for better visibility of variation
        ax.scatter(range(len(values)), values, alpha=0.7, s=30)

        # Add a reference line at the mean
        mean_val = np.mean(values)
        ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.7,
                  label=f'mean = {mean_val:.1e}')

        ax.set_ylabel(ylabel_text, fontsize=14)
        ax.set_title(title_text, fontsize=16)
        ax.set_xlabel("Sample", fontsize=14)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, alpha=0.3)

        # Set y-axis to log scale if values span several orders of magnitude
        # For mercury fallback, use bioavailable concentrations which may vary
        if np.max(values) / np.min(values[values > 0]) > 100 or is_mercury_fallback:
            ax.set_yscale('log')
            if is_mercury_fallback:
                ax.set_ylabel(f"{label} bioavailable concentration (log scale)", fontsize=14)
            else:
                ax.set_ylabel(f"{label} fraction (log scale)", fontsize=14)

    # Hide unused subplots
    for j in range(n_metals, nrows*ncols):
        fig.delaxes(axes[j])

    fig.tight_layout()
    fig_path = os.path.join(output_dir, "figure_speciation_profiles.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved improved speciation profiles figure to {fig_path}")

    return metals_to_plot  # Return the list of metals plotted

def main():
    parser = argparse.ArgumentParser(description="Speciation modeling with PHREEQC")
    parser.add_argument("--input", required=True,
                        help="CSV file with columns: site, pH, Eh, C_As, C_Cd, ..., Ca, Mg, etc.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save tables and figures")
    parser.add_argument("--database", default=None,
                        help="PHREEQC thermodynamic database file (default: database/PHREEQC_ThermoddemV1.10_15Dec2020.dat)")
    parser.add_argument("--min-concentration", type=float, default=1e-23,
                        help="Minimum mean concentration threshold (mol/L) for species selection (default: 1e-23)")
    parser.add_argument("--min-cv", type=float, default=0.01,
                        help="Minimum coefficient of variation threshold for species selection (default: 0.01)")
    parser.add_argument("--use-total-fallback", action="store_true",
                        help="Use total concentration as fallback when speciation doesn't meet quality criteria")
    parser.add_argument("--input-units", choices=['ug/L', 'mg/L'], default='ug/L',
                        help="Concentration units in input file (default: ug/L). Use mg/L if concentrations are in mg/L.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input, index_col=0)
    
    # Handle unit conversion: if input is mg/L, convert to µg/L
    if args.input_units == 'mg/L':
        print(f"Converting concentrations from mg/L to µg/L (multiply by 1000)")
        # Find metal concentration columns and convert them
        for col in df.columns:
            if col.startswith('C_') or col in ATOMIC_WEIGHTS.keys():
                df[col] = df[col] * 1000.0  # mg/L -> µg/L

    # Determine which metals are present to avoid fallback errors
    available_main = [m for m in SPECIES if has_total_column(df, m)]
    available_chromium = [m for m in CHROMIUM_SPECIES if has_total_column(df, 'Cr')]
    available_mercury = [m for m in MERCURY_SPECIES if has_total_column(df, 'Hg')]

    missing_main = sorted(set(SPECIES.keys()) - set(available_main))
    if missing_main:
        print(f"Skipping metals without total concentrations: {missing_main}")
    
    # Run PHREEQC to get free-ion molalities and all species molalities
    df_free_all, species_full = run_phreeqc_speciation(df, database=args.database)
    # Replace NaN in all species molalities
    species_full = species_full.fillna(0.0)
    # For each metal, select the species column with highest mean molality as most bioavailable
    # But only if it shows meaningful variation (CV > 1%)
    selected = {}
    df_free = pd.DataFrame(index=df.index)
    
    # Process main species
    for m in available_main:
        # find full species columns matching the free-ion code
        sp_code = SPECIES[m]
        cols = [c for c in species_full.columns if sp_code in c]
        if not cols:
            raise KeyError(f"No species columns matching '{sp_code}' found for metal '{m}' in PHREEQC output")
        # select the column (usually only one) with highest mean molality
        best = species_full[cols].mean().idxmax()
        
        # Check if this species shows meaningful variation AND has meaningful concentration
        values = species_full[best].values
        values_array = np.array(values)
        mean_conc = np.mean(values_array)
        cv = np.std(values_array) / mean_conc if mean_conc > 0 else 0

        # Use configurable thresholds for scientific rigor
        if cv > args.min_cv and mean_conc > args.min_concentration:
            selected[m] = best
            df_free[f"C_bio_{m}"] = species_full[best]
        else:
            if args.use_total_fallback:
                print(f"Warning: Species {m} ('{best}') does not meet quality criteria:")
                print(f"  - Mean concentration: {mean_conc:.2e} mol/L (threshold: > {args.min_concentration:.0e})")
                print(f"  - Coefficient of variation: {cv:.4f} (threshold: > {args.min_cv})")
                print(f"  - Using total {m} concentration as fallback")
                
                # Use total concentration as fallback
                tot_col = f"C_{m}"
                if tot_col in df.columns:
                    raw_tot = df[tot_col]
                elif m in df.columns:
                    raw_tot = df[m]
                else:
                    raise KeyError(f"Total {m} column not found for fallback")
                
                # Convert from μg/L to mol/L and use as bioavailable
                tot_mol = raw_tot * 1e-6 / ATOMIC_WEIGHTS[m]
                df_free[f"C_bio_{m}"] = tot_mol
                selected[m] = f"total_{m}_fallback"
            else:
                # For scientific rigor, raise an error rather than using fallbacks
                raise ValueError(f"Species {m} ('{best}') does not meet scientific quality criteria:\n"
                               f"  - Mean concentration: {mean_conc:.2e} mol/L (must be > {args.min_concentration:.0e})\n"
                               f"  - Coefficient of variation: {cv:.4f} (must be > {args.min_cv})\n"
                               f"  - This indicates inadequate speciation data quality\n"
                               f"  - Consider: --use-total-fallback flag, or adjust thresholds with --min-concentration or --min-cv")
    
    # Process chromium species (only if totals exist)
    for m in available_chromium:
        sp_code = CHROMIUM_SPECIES[m]
        cols = [c for c in species_full.columns if sp_code in c]
        if not cols:
            raise KeyError(f"No species columns matching '{sp_code}' found for chromium species '{m}' in PHREEQC output")
        # select the column (usually only one) with highest mean molality
        best = species_full[cols].mean().idxmax()
        
        # Check if this species shows meaningful variation AND has meaningful concentration
        values = species_full[best].values
        values_array = np.array(values)
        mean_conc = np.mean(values_array)
        cv = np.std(values_array) / mean_conc if mean_conc > 0 else 0
        
        # Use configurable thresholds for scientific rigor
        if cv > args.min_cv and mean_conc > args.min_concentration:
            selected[m] = best
            df_free[f"C_bio_{m}"] = species_full[best]
        else:
            if args.use_total_fallback:
                print(f"Warning: Chromium species {m} ('{best}') does not meet quality criteria:")
                print(f"  - Mean concentration: {mean_conc:.2e} mol/L (threshold: > {args.min_concentration:.0e})")
                print(f"  - Coefficient of variation: {cv:.4f} (threshold: > {args.min_cv})")
                print(f"  - Using total chromium concentration as fallback for {m}")
                
                # Use total chromium concentration as fallback
                tot_col = "C_Cr"
                if tot_col in df.columns:
                    raw_tot = df[tot_col]
                elif 'Cr' in df.columns:
                    raw_tot = df['Cr']
                else:
                    raise KeyError("Total chromium column not found for fallback")
                
                # Convert from μg/L to mol/L and use as bioavailable
                tot_mol = raw_tot * 1e-6 / ATOMIC_WEIGHTS['Cr']
                df_free[f"C_bio_{m}"] = tot_mol
                selected[m] = f"total_Cr_fallback"
            else:
                # For scientific rigor, raise an error rather than using fallbacks
                raise ValueError(f"Chromium species {m} ('{best}') does not meet scientific quality criteria:\n"
                               f"  - Mean concentration: {mean_conc:.2e} mol/L (must be > {args.min_concentration:.0e})\n"
                               f"  - Coefficient of variation: {cv:.4f} (must be > {args.min_cv})\n"
                               f"  - This indicates inadequate chromium speciation data quality\n"
                               f"  - Consider: --use-total-fallback flag, or adjust thresholds with --min-concentration or --min-cv\n"
                               f"  - Alternative: use total chromium concentration for both CrVI and CrIII")
    
    # Process mercury species (only if totals exist)
    for m in available_mercury:
        sp_code = MERCURY_SPECIES[m]
        cols = [c for c in species_full.columns if sp_code in c]
        if not cols:
            raise KeyError(f"No species columns matching '{sp_code}' found for mercury species '{m}' in PHREEQC output")
        # select the column (usually only one) with highest mean molality
        best = species_full[cols].mean().idxmax()
        
        # Check if this species shows meaningful variation AND has meaningful concentration
        values = species_full[best].values
        values_array = np.array(values)
        mean_conc = np.mean(values_array)
        cv = np.std(values_array) / mean_conc if mean_conc > 0 else 0
        
        # Use configurable thresholds for scientific rigor
        if cv > args.min_cv and mean_conc > args.min_concentration:
            selected[m] = best
            df_free[f"C_bio_{m}"] = species_full[best]
        else:
            if args.use_total_fallback:
                print(f"Warning: Mercury species {m} ('{best}') does not meet quality criteria:")
                print(f"  - Mean concentration: {mean_conc:.2e} mol/L (threshold: > {args.min_concentration:.0e})")
                print(f"  - Coefficient of variation: {cv:.4f} (threshold: > {args.min_cv})")
                print(f"  - Using total mercury concentration as fallback for {m}")
                
                # Use total mercury concentration as fallback
                tot_col = "C_Hg"
                if tot_col in df.columns:
                    raw_tot = df[tot_col]
                elif 'Hg' in df.columns:
                    raw_tot = df['Hg']
                else:
                    raise KeyError("Total mercury column not found for fallback")
                
                # Convert from μg/L to mol/L and use as bioavailable
                tot_mol = raw_tot * 1e-6 / ATOMIC_WEIGHTS['Hg']
                df_free[f"C_bio_{m}"] = tot_mol
                selected[m] = f"total_Hg_fallback"
            else:
                # For scientific rigor, raise an error rather than using fallbacks
                raise ValueError(f"Mercury species {m} ('{best}') does not meet scientific quality criteria:\n"
                               f"  - Mean concentration: {mean_conc:.2e} mol/L (must be > {args.min_concentration:.0e})\n"
                               f"  - Coefficient of variation: {cv:.4f} (must be > {args.min_cv})\n"
                               f"  - This indicates inadequate mercury speciation data quality\n"
                               f"  - Consider: --use-total-fallback flag, or adjust thresholds with --min-concentration or --min-cv\n"
                               f"  - Alternative: use total mercury concentration for bioavailability")
    
    print("Selected most bioavailable species per metal:", selected)
    # Map PHREEQC column names to pretty MathTeX labels for plotting
    # Build a tex map that works both by metal key (e.g. 'As','CrVI','Hg(II)')
    # and by the selected PHREEQC column name (selected[metal]).
    species_tex_map = {}

    metal_tex = {
        'As': r'$\mathrm{H_2AsO_4^{-}}$',      # Arsenate (As V)
        'Cd': r'$\mathrm{Cd^{2+}}$',           # Cadmium ion
        'Cu': r'$\mathrm{Cu^{2+}}$',           # Cupric ion
        'Pb': r'$\mathrm{Pb^{2+}}$',           # Lead ion
        'CrVI': r'$\mathrm{CrO_4^{2-}}$',      # Chromate (Cr VI)
        'CrIII': r'$\mathrm{Cr^{3+}}$',        # Chromic ion (Cr III)
        'Co': r'$\mathrm{Co^{2+}}$',           # Cobalt ion
        'Fe': r'$\mathrm{Fe^{2+}}$',           # Ferrous ion (Fe II)
        'Mn': r'$\mathrm{Mn^{2+}}$',           # Manganous ion
        'Ni': r'$\mathrm{Ni^{2+}}$',           # Nickel ion
        'Zn': r'$\mathrm{Zn^{2+}}$',           # Zinc ion
        'Al': r'$\mathrm{Al^{3+}}$',           # Aluminum ion
        'V': r'$\mathrm{VO_2^{+}}$',           # Vanadyl (V V) - equilibrates to vanadate
        'Hg(II)': r'$\mathrm{Hg^{2+}}$',       # Mercuric ion
        'Hg(0)': r'$\mathrm{Hg^0}$',           # Elemental mercury
    }

    # Populate species_tex_map with both metal key -> tex and selected column -> tex
    for metal, tex in metal_tex.items():
        species_tex_map[metal] = tex
        if metal in selected:
            species_tex_map[selected[metal]] = tex

    # Add mercury species entries (both metal key and selected column name)
    # Note: MeHg excluded - it's biologically formed, not thermodynamic
    for m in MERCURY_SPECIES:
        if m == 'Hg(II)':
            tex = r'$\mathrm{Hg^{2+}}$'
        elif m == 'Hg(0)':
            tex = r'$\mathrm{Hg^0}$'
        else:
            tex = m
        # map by metal key
        species_tex_map[m] = tex
        # also map by selected PHREEQC column name if available
        if m in selected:
            species_tex_map[selected[m]] = tex
    # Compute fractions using selected species (only for metals present)
    frac = compute_species_fractions(
        df,
        df_free,
        metals=available_main,
        chromium_species=available_chromium,
        mercury_species=available_mercury,
    )

    # Table 3: thermodynamic speciation fractions
    save_fraction_table(frac, args.output_dir)
    # Also save bioavailable (free-ion) concentrations table
    save_bioavailable_table(df_free, args.output_dir)
    # Figure 2b: speciation profiles
    metals_to_plot = plot_speciation_profiles(frac, args.output_dir, df, df_free, selected_species=selected, species_tex_map=species_tex_map)

    # Create a summary of excluded metals
    all_metals = [col[5:] for col in frac.columns if col.startswith('frac_')]
    excluded_metals = [m for m in all_metals if m not in (metals_to_plot or [])]

    if excluded_metals:
        print(f"\nExcluded metals (constant/low variation fractions): {excluded_metals}")
        print("These metals typically have:")
        print("- Very low concentrations (often below detection limits)")
        print("- Non-detect values replaced with DL/2, leading to similar speciation")
        print("- Limited variation in water chemistry across sites")
        print("- This is normal behavior for trace metals in environmental samples")

        # Save exclusion summary
        summary_path = os.path.join(args.output_dir, "speciation_exclusion_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Speciation Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            plotted_metals = metals_to_plot or []
            f.write(f"Metals with variable speciation fractions: {', '.join(plotted_metals)}\n\n")
            f.write(f"Excluded metals (constant fractions): {', '.join(excluded_metals)}\n\n")
            f.write("Exclusion reasons:\n")
            f.write("- Very low concentrations (often < DL)\n")
            f.write("- Non-detect values replaced with DL/2\n")
            f.write("- Limited variation in water chemistry\n")
            f.write("- This is expected for trace metals\n\n")
            f.write("Recommendation: Focus risk assessment on metals with variable fractions\n")
        print(f"Saved exclusion summary to {summary_path}")

if __name__ == "__main__":
    main()
