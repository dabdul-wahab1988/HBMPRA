#!/usr/bin/env python3
"""
run_hbmpra.py - User-Friendly HBMPRA Analysis Runner

This script provides a simple interface for running the complete HBMPRA
(Hierarchical Bayesian Model for Probabilistic Risk Assessment) workflow
for metals and anions in water.

For non-coders: Simply run this script and follow the prompts!

Usage:
    python run_hbmpra.py                    # Interactive mode (recommended for beginners)
    python run_hbmpra.py --input ../waterdata/data1.csv  # Quick mode with defaults
    python run_hbmpra.py --help             # Show all options

Workflow Steps:
    1. Load water chemistry data (CSV file)
    2. Run PHREEQC speciation modeling (optional but recommended)
    3. Calibrate BLL (Blood Lead Level) priors using population-appropriate engines
    4. Run Bayesian risk assessment model
    5. Generate diagnostic plots
    6. Generate result figures
    7. Generate summary tables
    8. Run sensitivity analysis (optional, advanced)
    9. Run entropy-based HPI/PERI analysis (optional, advanced)

BLL Engine Selection (automatic):
    - Adults: One-compartment pharmacokinetic model (mechanistic)
    - Children/Teens/Pregnant: Empirical slope model (higher sensitivity)

Authors:
    - Dickson Abdul-Wahab — University of Ghana, Ghana
        Email: dabdul-wahab@live.com
        ORCID: https://orcid.org/0000-0001-7446-5909
        LinkedIn: https://www.linkedin.com/in/dickson-abdul-wahab-0764a1a9
        ResearchGate: https://www.researchgate.net/profile/Dickson-Abdul-Wahab
    - Ebenezer Aquisman Asare — Organic Laboratory Research, Atomic Energy Commission (GAEC), Nuclear Chemistry and Environmental Research Centre, National Nuclear Research Institute (NNRI), Legon-Accra, Ghana
        Email: aquisman1989@gmail.com
        ORCID: https://orcid.org/0000-0003-1185-1479
        ResearchGate: https://www.researchgate.net/profile/Ebenezer-Aquisman-Asare
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Get the directory where this script lives (src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Project root is parent of src/
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# Water data directory
WATERDATA_DIR = os.path.join(PROJECT_ROOT, "waterdata")


def print_banner():
    """Print welcome banner."""
    print("\n" + "="*70)
    print("  HBMPRA - Bayesian Water Quality Risk Assessment (Metals & Anions)")
    print("="*70)
    print("  A user-friendly tool for metals and anions risk assessment in water")
    print("="*70 + "\n")


def print_step(step_num, total_steps, message):
    """Print a step indicator."""
    print(f"\n[Step {step_num}/{total_steps}] {message}")
    print("-" * 50)


def check_dependencies():
    """Check if required packages are installed and auto-install if missing."""
    import subprocess
    
    missing_required = []
    missing_optional = []
    
    required = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'yaml': 'pyyaml',
    }
    
    optional = {
        'phreeqpython': 'phreeqpython',
        'pymc': 'pymc',
        'arviz': 'arviz',
    }
    
    print("Checking dependencies...")
    
    # Check required packages
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  [OK] {package}")
        except ImportError:
            missing_required.append(package)
            print(f"  [MISSING] {package} (required)")
    
    # Check optional packages
    for module, package in optional.items():
        try:
            __import__(module)
            print(f"  [OK] {package}")
        except ImportError:
            missing_optional.append(package)
            print(f"  ○ {package} - not installed (optional)")
    
    # Auto-install missing required packages
    if missing_required:
        print(f"\n[INFO] Auto-installing missing required packages: {', '.join(missing_required)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + missing_required)
            print(f"  [OK] Successfully installed: {', '.join(missing_required)}")
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Failed to install packages: {e}")
            print(f"  Please manually run: pip install {' '.join(missing_required)}")
            return False
    
    # Offer to install optional packages
    if missing_optional:
        print(f"\n[INFO] Optional packages not installed: {', '.join(missing_optional)}")
        choice = input("  Install optional packages for full functionality? [y/N]: ").strip().lower()
        if choice == 'y':
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + missing_optional)
                print(f"  [OK] Successfully installed: {', '.join(missing_optional)}")
            except subprocess.CalledProcessError as e:
                print(f"  [WARN] Some optional packages failed to install: {e}")
                print(f"  You can manually install later: pip install {' '.join(missing_optional)}")
    
    return True


def run_bll_calibration(input_file, output_dir, input_units='µg/L'):
    """
    Run BLL (Blood Lead Level) prior calibration using population-appropriate engines.
    
    Uses 'auto' mode which selects:
    - onecomp (mechanistic PK) for Adults
    - slope (empirical dose-response) for Children, Teens, Pregnant
    
    Returns True if successful, False otherwise.
    """
    import subprocess
    
    calibration_dir = os.path.join(output_dir, "calibration")
    os.makedirs(calibration_dir, exist_ok=True)
    priors_json = os.path.join(calibration_dir, "priors.json")
    
    print("\nCalibrating BLL (Blood Lead Level) priors...")
    print("  Engine mode: auto (population-appropriate selection)")
    print("    • Adults → onecomp (mechanistic pharmacokinetic model)")
    print("    • Children/Teens → slope (empirical, higher sensitivity)")
    print("    • Pregnant → slope (empirical, conservative)")
    print(f"  Input concentration units: {input_units}")
    
    calibration_cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "calibrate_bll_priors.py"),
        "--chemistry", input_file,
        "--out-json", priors_json,
        "--bll-engine", "auto",
        "--n-grid", "15"  # More grid points for better calibration
    ]
    
    # Add unit conversion flag if needed
    if input_units in ['mg/L', 'ppm']:
        calibration_cmd.extend(["--input-units", "mg/L"])
    
    result = subprocess.run(calibration_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  [WARN] BLL calibration had issues:")
        if result.stderr:
            print(f"    {result.stderr[-300:]}")
        print("  Continuing with default uncalibrated priors...")
        return False
    
    # Show calibration summary
    try:
        import json
        with open(priors_json) as f:
            priors = json.load(f)
        
        engine_meta = priors.get('engine_metadata', {})
        engines_per_group = engine_meta.get('engines_per_group', {})
        pop_class = engine_meta.get('population_classification', {})
        calibration_date = engine_meta.get('calibration_date', 'unknown')
        calibration_source = engine_meta.get('calibration_source', 'unknown')
        
        print("  [OK] BLL calibration complete!")
        print(f"  Calibration date: {calibration_date}")
        print(f"  Calibration source: {calibration_source}")
        print("  Calibrated engines per group:")
        for group, engine in engines_per_group.items():
            pop_type = pop_class.get(group, 'unknown')
            print(f"    • {group}: {engine} ({pop_type})")
        
        # Show key parameters
        groups_data = priors.get('groups', {})
        print("  Calibrated parameters (k = intake→BLL slope):")
        for group, params in groups_data.items():
            k_mean = params.get('k_mean', params.get('k_wb_mu', 0))
            b0_mean = params.get('b0_mean', params.get('b0_mu', 0))
            print(f"    • {group}: k={k_mean:.4f}, b0={b0_mean:.4f}")
        
    except Exception as e:
        print(f"  [OK] Calibration saved to {priors_json}")
    
    return True


def run_plot_diagnostics(output_dir):
    """
    Run diagnostic plots (trace plots, convergence, prior vs posterior).
    
    Uses plot_diagnostics.py which requires model.pkl and trace.nc
    """
    import subprocess
    
    model_file = os.path.join(output_dir, "model.pkl")
    trace_file = os.path.join(output_dir, "trace.nc")
    plots_dir = os.path.join(output_dir, "diagnostics")
    
    os.makedirs(plots_dir, exist_ok=True)
    
    if not os.path.exists(trace_file):
        print("  [WARN] trace.nc not found, skipping diagnostics")
        return False
    
    print("\nGenerating diagnostic plots...")
    
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "plot_diagnostics.py"),
        "--model-file", model_file,
        "--trace-file", trace_file,
        "--output-dir", plots_dir
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("  [WARN] Some diagnostic plots may have failed")
        if result.stderr:
            # Only show last part of error
            lines = result.stderr.strip().split('\n')
            for line in lines[-5:]:
                print(f"    {line}")
    
    # Count generated files
    try:
        png_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
        if png_files:
            print(f"  [OK] Generated {len(png_files)} diagnostic plots")
            return True
        else:
            print("  [WARN] No diagnostic plots were generated")
            return False
    except Exception:
        return False


def run_plot_results(output_dir, bll_thresholds="3.5,5,10"):
    """
    Run result plots (posterior distributions, exceedance curves).
    
    Uses plot_result.py which generates publication-ready figures.
    """
    import subprocess
    
    trace_file = os.path.join(output_dir, "trace.nc")
    figures_dir = os.path.join(output_dir, "figures")
    
    os.makedirs(figures_dir, exist_ok=True)
    
    if not os.path.exists(trace_file):
        print("  [WARN] trace.nc not found, skipping result plots")
        return False
    
    print("\nGenerating result figures...")
    
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "plot_result.py"),
        "--results-dir", output_dir,
        "--output-dir", figures_dir,
        "--bll-thresholds", bll_thresholds
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("  [WARN] Some result plots may have failed")
        if result.stderr:
            lines = result.stderr.strip().split('\n')
            for line in lines[-5:]:
                print(f"    {line}")
    
    # Count generated files
    try:
        png_files = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
        csv_files = [f for f in os.listdir(figures_dir) if f.endswith('.csv')]
        if png_files or csv_files:
            print(f"  [OK] Generated {len(png_files)} figures, {len(csv_files)} summary files")
            return True
        else:
            print("  [WARN] No result figures were generated")
            return False
    except Exception:
        return False


def run_summary_tables(output_dir, input_file=None):
    """
    Generate publication-ready summary tables.
    
    Uses summary_tables.py to create CSV tables for:
    - T1: Measured concentrations with WHO comparison
    - T2: Speciation fractions
    - T3: Posterior summary (quantiles, exceedance)
    - T4: Risk ranking
    - T5: BLL summary
    - Correlation matrix, PCA results
    """
    import subprocess
    
    trace_file = os.path.join(output_dir, "trace.nc")
    tables_dir = os.path.join(output_dir, "tables")
    
    os.makedirs(tables_dir, exist_ok=True)
    
    if not os.path.exists(trace_file):
        print("  [WARN] trace.nc not found, skipping summary tables")
        return False
    
    print("\nGenerating summary tables...")
    
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "summary_tables.py"),
        "--results-dir", output_dir,
        "--output-dir", tables_dir
    ]
    
    if input_file:
        cmd.extend(["--input", input_file])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("  [WARN] Some summary tables may have failed")
        if result.stderr:
            lines = result.stderr.strip().split('\n')
            for line in lines[-5:]:
                print(f"    {line}")
    
    # Count generated files
    try:
        csv_files = [f for f in os.listdir(tables_dir) if f.endswith('.csv')]
        txt_files = [f for f in os.listdir(tables_dir) if f.endswith('.txt')]
        if csv_files or txt_files:
            print(f"  [OK] Generated {len(csv_files)} summary tables, {len(txt_files)} analysis files")
            return True
        else:
            print("  [WARN] No summary tables were generated")
            return False
    except Exception:
        return False


def run_sensitivity_analysis(input_file, output_dir, method='sobol', n_samples=512, input_units='µg/L'):
    """Run sensitivity analysis using Sobol, Morris, or Delta moment methods.
    
    This analyzes parameter sensitivity in the risk assessment model.
    
    Parameters
    ----------
    input_file : str
        Path to input water chemistry CSV file
    output_dir : str
        Path to main output directory
    method : str
        Analysis method: 'sobol', 'morris', or 'delta'
    n_samples : int
        Number of samples for analysis (default 512)
    input_units : str
        Concentration units in input file (default 'µg/L')
    
    Returns
    -------
    bool
        True if analysis succeeded
    """
    import subprocess
    
    sensitivity_dir = os.path.join(output_dir, "sensitivity")
    os.makedirs(sensitivity_dir, exist_ok=True)
    
    print(f"\nRunning {method.upper()} sensitivity analysis...")
    print(f"  This may take several minutes depending on sample size...")
    
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "sensitivity_analysis.py"),
        "--method", method,
        "--input", input_file,
        "--output-dir", sensitivity_dir,
        "--n-samples", str(n_samples)
    ]
    
    # Add unit conversion flag if needed
    if input_units in ['mg/L', 'ppm']:
        cmd.extend(["--input-units", "mg/L"])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  [WARN] Sensitivity analysis encountered issues")
        if result.stderr:
            lines = result.stderr.strip().split('\n')
            for line in lines[-5:]:
                print(f"    {line}")
        return False
    
    # Check for output files
    try:
        csv_files = [f for f in os.listdir(sensitivity_dir) if f.endswith('.csv')]
        png_files = [f for f in os.listdir(sensitivity_dir) if f.endswith('.png')]
        if csv_files or png_files:
            print(f"  [OK] Generated {len(csv_files)} indices files, {len(png_files)} plots")
            print(f"  [OK] Sensitivity results saved to: {sensitivity_dir}")
            return True
        else:
            print("  [WARN] No sensitivity analysis outputs were generated")
            return False
    except Exception as e:
        print(f"  [WARN] Could not verify outputs: {e}")
        return False


def run_entropy_hpi_peri(input_file, output_dir, standards_file=None, toxicities_file=None, bootstrap_samples=1000, input_units='µg/L'):
    """Run entropy-based HPI/PERI pollution index analysis.
    
    Computes Heavy Metal Pollution Index (HPI) and Potential Ecological Risk Index (PERI)
    using entropy-weighted calculations.
    
    Parameters
    ----------
    input_file : str
        Path to input water chemistry CSV file
    output_dir : str
        Path to main output directory
    standards_file : str, optional
        Path to standards CSV file (defaults to waterdata/standards.csv)
    toxicities_file : str, optional
        Path to toxicities CSV file (defaults to waterdata/toxicity.csv)
    bootstrap_samples : int
        Number of bootstrap samples for uncertainty (default 1000)
    input_units : str
        Concentration units in input file (default 'µg/L')
    
    Returns
    -------
    bool
        True if analysis succeeded
    """
    import subprocess
    
    # Default to waterdata folder for standards and toxicities
    if standards_file is None:
        standards_file = os.path.join(WATERDATA_DIR, "standards.csv")
        if not os.path.exists(standards_file):
            # Try project root
            standards_file = os.path.join(PROJECT_ROOT, "standards.csv")
    
    if toxicities_file is None:
        toxicities_file = os.path.join(WATERDATA_DIR, "toxicity.csv")
        if not os.path.exists(toxicities_file):
            # Try project root
            toxicities_file = os.path.join(PROJECT_ROOT, "toxicity.csv")
    
    # Verify required files exist
    if not os.path.exists(standards_file):
        print("  [WARN] Standards file not found. Skipping entropy analysis.")
        print(f"    Expected: {standards_file}")
        return False
    
    if not os.path.exists(toxicities_file):
        print("  [WARN] Toxicities file not found. Skipping entropy analysis.")
        print(f"    Expected: {toxicities_file}")
        return False
    
    entropy_dir = os.path.join(output_dir, "entropy_analysis")
    os.makedirs(entropy_dir, exist_ok=True)
    
    print("\nRunning entropy-based HPI/PERI analysis...")
    print(f"  Using standards: {standards_file}")
    print(f"  Using toxicities: {toxicities_file}")
    
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "entropy_hpi_peri.py"),
        "--input", input_file,
        "--standards", standards_file,
        "--toxicities", toxicities_file,
        "--output-dir", entropy_dir,
        "--bootstrap-samples", str(bootstrap_samples)
    ]
    
    # Add unit conversion flag if needed
    if input_units in ['mg/L', 'ppm']:
        cmd.extend(["--input-units", "mg/L"])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  [WARN] Entropy analysis encountered issues")
        if result.stderr:
            lines = result.stderr.strip().split('\n')
            for line in lines[-5:]:
                print(f"    {line}")
        return False
    
    # Check for output files
    try:
        csv_files = [f for f in os.listdir(entropy_dir) if f.endswith('.csv')]
        png_files = [f for f in os.listdir(entropy_dir) if f.endswith('.png')]
        if csv_files or png_files:
            print(f"  [OK] Generated {len(csv_files)} analysis files, {len(png_files)} plots")
            print(f"  [OK] Entropy/HPI/PERI results saved to: {entropy_dir}")
            return True
        else:
            print("  [WARN] No entropy analysis outputs were generated")
            return False
    except Exception as e:
        print(f"  [WARN] Could not verify outputs: {e}")
        return False


def find_data_files(directory=None):
    """Find CSV files that look like chemistry data.
    
    Searches in multiple locations:
    1. The specified directory (if provided)
    2. The waterdata/ folder in the project root
    3. The current working directory
    """
    if directory is None:
        directory = WATERDATA_DIR
    
    data_files = []
    
    # Search in specified directory
    search_dirs = [directory]
    
    # Also search in waterdata folder if different from specified
    if directory != WATERDATA_DIR and os.path.exists(WATERDATA_DIR):
        search_dirs.append(WATERDATA_DIR)
    
    # Search in current working directory
    cwd = os.getcwd()
    if cwd not in search_dirs:
        search_dirs.append(cwd)
    
    for d in search_dirs:
        if not os.path.exists(d):
            continue
        csv_files = list(Path(d).glob("*.csv"))
        
        for f in csv_files:
            # Skip output files and lookup tables
            if any(x in f.name.lower() for x in ['table_', 'summary', 'toxicity', 'standards', 'results']):
                continue
            if f not in data_files:
                data_files.append(f)
    
    return data_files


def validate_chemistry_file(filepath):
    """Validate that a CSV file contains expected chemistry data."""
    import pandas as pd
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return False, f"Cannot read file: {e}"
    
    # Check for metal columns
    metals_found = []
    expected_metals = ['As', 'Cd', 'Cr', 'Cu', 'Hg', 'Pb', 'Mn', 'Fe', 'Zn', 'Ni', 'Co', 'Al', 'V']
    
    for col in df.columns:
        col_upper = col.upper()
        for m in expected_metals:
            if m.upper() in col_upper or col.startswith(f'C_{m}'):
                metals_found.append(m)
                break
    
    metals_found = list(set(metals_found))
    
    # Check for anion columns (Fluoride, Nitrate) - NEW
    anions_found = []
    anion_info = {}
    
    for col in df.columns:
        col_upper = col.upper().replace('-', '').replace('_', '')
        
        # Check for fluoride
        if 'FLUORIDE' in col_upper or col_upper == 'F':
            if 'fluoride' not in anions_found:
                anions_found.append('fluoride')
                anion_info['fluoride'] = {'column': col}
        
        # Check for nitrate
        if any(pat in col_upper for pat in ['NO3', 'NITRATE']):
            if 'nitrate' not in anions_found:
                anions_found.append('nitrate')
                # Detect basis from column name
                if 'N' in col_upper.replace('NO3', '').replace('NITRATE', ''):
                    anion_info['nitrate'] = {'column': col, 'basis': 'NO3-N'}
                else:
                    anion_info['nitrate'] = {'column': col, 'basis': 'NO3'}
    
    # Allow anions-only analysis if no metals but anions present
    if not metals_found and not anions_found:
        return False, "No metal or anion concentration columns found. Expected columns like 'As', 'C_As', 'Pb', 'F', 'NO3', etc."
    
    # Check for site identifier
    if df.columns[0] in ['site', 'Site', 'ID', 'id', 'Community', 'Location']:
        has_site = True
    else:
        has_site = False
    
    info = {
        'rows': len(df),
        'metals': metals_found,
        'anions': anions_found,  # NEW
        'anion_info': anion_info,  # NEW
        'has_site_id': has_site,
        'has_pH': 'pH' in df.columns,
        'has_Eh': 'Eh' in df.columns,
        'columns': list(df.columns)
    }
    
    return True, info



class SessionContext:
    """Class to maintain session state for the interactive hub."""
    def __init__(self):
        self.input_file = None
        self.output_dir = None
        self.concentration_unit = 'µg/L'
        self.selected_groups = []
        self.idata_info = None  # Validates if risk results exist
        self.is_anion_only = False
        self.has_lead = False

    def update_info(self, info):
        self.is_anion_only = not info.get('metals') and info.get('anions')
        self.has_lead = 'Pb' in info.get('metals', [])
        self.metals = info.get('metals', [])
        self.anions = info.get('anions', [])


def configure_session(default_file=None):
    """Initial session configuration: file, units, output directory."""
    ctx = SessionContext()
    
    # Select input file
    if default_file and os.path.exists(default_file):
        filepath = default_file
    else:
        data_files = find_data_files()
        if not data_files:
            filepath = input("Enter path to your chemistry CSV file: ").strip()
        elif len(data_files) == 1:
            filepath = str(data_files[0])
            print(f"Found data file: {filepath}")
            if input("Use this file? [Y/n]: ").strip().lower() == 'n':
                filepath = input("Enter path to your chemistry CSV file: ").strip()
        else:
            print("Found multiple data files:")
            for i, f in enumerate(data_files, 1):
                print(f"  {i}. {f.name}")
            choice = input(f"Select file [1-{len(data_files)}] or enter path: ").strip()
            try:
                idx = int(choice) - 1
                filepath = str(data_files[idx])
            except (ValueError, IndexError):
                filepath = choice

    if not filepath or not os.path.exists(filepath):
        print(f"  [ERROR] File not found: {filepath}")
        return None

    # Validate file
    valid, info = validate_chemistry_file(filepath)
    if not valid:
        print(f"  [ERROR] {info}")
        return None
    
    ctx.input_file = filepath
    ctx.update_info(info)
    
    # Print status
    print(f"\n[OK] Valid chemistry file: {os.path.basename(filepath)}")
    print(f"  ✓ {info['rows']} samples detected")
    if info['metals']: print(f"  ✓ Metals: {', '.join(info['metals'])}")
    if info['anions']: print(f"  ✓ Anions: {', '.join(info['anions'])}")

    # Units selection
    print("\nSelect concentration units:")
    print("  1. µg/L (default)  2. mg/L (ppm)")
    u_choice = input("Choice [1/2, default=1]: ").strip()
    ctx.concentration_unit = 'mg/L' if u_choice == '2' else 'µg/L'
    
    # Output directory - explicitly save in project root (outside src/)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = f"results_{Path(filepath).stem}_{timestamp}"
    out_dir_input = input(f"Output directory [{default_out}]: ").strip()
    final_out = out_dir_input if out_dir_input else default_out
    
    # Force output to be in project root if relative path
    if not os.path.isabs(final_out):
        ctx.output_dir = os.path.abspath(os.path.join(PROJECT_ROOT, final_out))
    else:
        ctx.output_dir = final_out
        
    os.makedirs(ctx.output_dir, exist_ok=True)
    
    # Demographic selection
    try:
        from demographics import GROUP_INFO, parse_group_selection
        print("\nDemographic groups: (all, sensitive, non_sensitive, adults_only, children_only)")
        g_sel = input("Select groups [default=all]: ").strip()
        ctx.selected_groups = parse_group_selection(g_sel if g_sel else 'all')
    except ImportError:
        ctx.selected_groups = ["Adults", "Children", "Teens", "Pregnant"]

    return ctx


def run_risk_assessment_workflow(ctx):
    """Workflow: Speciation -> BLL Calibration -> Bayesian Model."""
    print_step(1, 4, "Starting Risk Assessment Workflow")
    
    # 1. Speciation
    run_spec = False
    if not ctx.is_anion_only:
        choice = input("Run PHREEQC speciation? (Highly Recommended) [Y/n]: ").strip().lower()
        if choice != 'n':
            print("  Running speciation...")
            spec_cmd = [sys.executable, os.path.join(SCRIPT_DIR, "speciation_modeling.py"),
                       "--input", ctx.input_file, "--output-dir", ctx.output_dir, "--use-total-fallback"]
            if ctx.concentration_unit == 'mg/L': spec_cmd.extend(["--input-units", "mg/L"])
            import subprocess
            subprocess.run(spec_cmd)
            run_spec = True

    # 2. BLL Calibration
    if not ctx.is_anion_only and ctx.has_lead:
        print_step(2, 4, "BLL Calibration")
        run_bll_calibration(ctx.input_file, ctx.output_dir, ctx.concentration_unit)

    # 3. Hamiltonian Monte Carlo
    print_step(3, 4, "Bayesian Model Sampling")
    print("  Configure MCMC: 1000 (Quick), 2000 (Standard), 4000 (Publication)")
    d_input = input("  Draws [default=2000]: ").strip()
    mcmc_draws = int(d_input) if d_input else 2000
    
    hbmp_cmd = [sys.executable, os.path.join(SCRIPT_DIR, "hbmpra_optimized.py"),
               "--chemistry", ctx.input_file, "--results-dir", ctx.output_dir,
               "--draws", str(mcmc_draws), "--tune", str(mcmc_draws),
               "--groups", ",".join(ctx.selected_groups), "--use-bioavailable"]
    if ctx.concentration_unit == 'mg/L': hbmp_cmd.extend(["--input-units", "mg/L"])
    
    import subprocess
    result = subprocess.run(hbmp_cmd)
    
    if result.returncode == 0:
        print("\n[OK] Risk Assessment Complete.")
        # Show mini preview
        try:
            import arviz as az
            trace_path = os.path.join(ctx.output_dir, "trace.nc")
            if os.path.exists(trace_path):
                idata = az.from_netcdf(trace_path)
                post = idata.posterior
                if "HI_overall" in post:
                    print(f"  → Mean Overall HI: {float(post['HI_overall'].mean()):.3f}")
        except Exception: pass
    else:
        print("\n[ERROR] Risk Assessment failed.")


def run_sensitivity_workflow(ctx):
    """Prompt for config and run sensitivity analysis."""
    print_step(1, 1, "Sensitivity Analysis")
    print("Select method: 1. sobol (best), 2. morris (fast), 3. delta")
    m_choice = input("Choice [1/2/3, default=1]: ").strip()
    method = 'morris' if m_choice == '2' else 'delta' if m_choice == '3' else 'sobol'
    
    s_choice = input("Number of samples [default=512]: ").strip()
    n_samples = int(s_choice) if s_choice else 512
    
    run_sensitivity_analysis(ctx.input_file, ctx.output_dir, method=method, n_samples=n_samples, input_units=ctx.concentration_unit)


def run_indices_workflow(ctx):
    """Prompt for config and run HPI/PERI indices analysis."""
    print_step(1, 1, "Entropy-based Pollution Indices")
    
    # Check for standards and toxicity files
    standards_file = None
    toxicities_file = None
    
    # Try to find in waterdata folder first, then project root
    for base_dir in [WATERDATA_DIR, PROJECT_ROOT]:
        std_path = os.path.join(base_dir, "standards.csv")
        tox_path = os.path.join(base_dir, "toxicity.csv")
        
        if os.path.exists(std_path) and standards_file is None:
            standards_file = std_path
        if os.path.exists(tox_path) and toxicities_file is None:
            toxicities_file = tox_path
    
    if not standards_file or not os.path.exists(standards_file):
        print("  [INFO] standards.csv not found in default locations.")
        standards_file = input("  Enter path to standards.csv: ").strip()
    
    if not toxicities_file or not os.path.exists(toxicities_file):
        print("  [INFO] toxicity.csv not found in default locations.")
        toxicities_file = input("  Enter path to toxicity.csv: ").strip()
    
    if not os.path.exists(standards_file) or not os.path.exists(toxicities_file):
        print("  [ERROR] Required files for Entropy analysis are missing. Skipping.")
        return

    b_choice = input("Bootstrap samples [default=1000]: ").strip()
    n_bootstrap = int(b_choice) if b_choice else 1000
    
    run_entropy_hpi_peri(ctx.input_file, ctx.output_dir, 
                         standards_file=standards_file, 
                         toxicities_file=toxicities_file,
                         bootstrap_samples=n_bootstrap, 
                         input_units=ctx.concentration_unit)


def check_hbmpra_results(ctx):
    """Verify if HBMPRA results exist in the output directory."""
    trace_path = os.path.join(ctx.output_dir, "trace.nc")
    if not os.path.exists(trace_path):
        print(f"\n[WARN] Model results (trace.nc) not found in: {ctx.output_dir}")
        print("Please run HBMPRA Risk Assessment first (Option 1).")
        return False
    return True


def interactive_mode():
    """Main Menu Hub for Interactive HBMPRA Analysis."""
    print_banner()
    if not check_dependencies(): return 1
    
    ctx = configure_session()
    if not ctx: return 1

    while True:
        print("\n" + "="*60)
        print("           HBMPRA INTERACTIVE HUB")
        print("="*60)
        print(f" [ CONFIGURATION ]")
        print(f"   Input File:   {os.path.basename(ctx.input_file)}")
        print(f"   Output Dir:   {ctx.output_dir}/")
        print(f"   Units:        {ctx.concentration_unit}")
        print(f"   Groups:       {', '.join(ctx.selected_groups)}")
        print("-" * 60)
        print(" [ 1. RUN ANALYSES ]")
        print("   1. Risk Assessment (Full Workflow)")
        print("   2. Sensitivity Analysis (Sobol/Morris Calculation)")
        print("   3. Pollution Indices (Entropy HPI/PERI Calculation)")
        print("-" * 60)
        print(" [ 2. HBMPRA POST-ANALYSIS ]")
        print("   4. Standard Result Figures (Posteriors/Exceedance)")
        print("   5. Interactive Custom Plot Builder")
        print("   6. HBMPRA Diagnostic Plots (Trace/Convergence)")
        print("   7. HBMPRA Summary Tables (Stats/PCA)")
        print("-" * 60)
        print(" [ 3. SENSITIVITY POST-ANALYSIS ]")
        print("   8. Sensitivity Visualization & Tables")
        print("-" * 60)
        print(" [ 4. POLLUTION INDICES POST-ANALYSIS ]")
        print("   9. Pollution Indices Visualization & Tables")
        print("-" * 60)
        print(" [ 5. SYSTEM ]")
        print("   10. Change Configuration (File/Output/Units)")
        print("   0. Exit")
        print("="*60)
        
        choice = input("\nSelect choice: ").strip()
        
        if choice == '0':
            print("Exiting HBMPRA Hub. Goodbye!")
            break
            
        elif choice == '1':
            run_risk_assessment_workflow(ctx)
            
        elif choice == '2':
            run_sensitivity_workflow(ctx)
            
        elif choice == '3':
            run_indices_workflow(ctx)
            
        elif choice == '4':
            if check_hbmpra_results(ctx):
                run_plot_results(ctx.output_dir)
                
        elif choice == '5':
            if check_hbmpra_results(ctx):
                try:
                    import arviz as az
                    from plot_panel import interactive_plot_builder
                    idata = az.from_netcdf(os.path.join(ctx.output_dir, "trace.nc"))
                    interactive_plot_builder(idata, ctx.output_dir)
                except Exception as e:
                    print(f"  [ERROR] Plot builder failed: {e}")
                    
        elif choice == '6':
            if check_hbmpra_results(ctx):
                run_plot_diagnostics(ctx.output_dir)
                
        elif choice == '7':
            if check_hbmpra_results(ctx):
                run_summary_tables(ctx.output_dir, ctx.input_file)
                
        elif choice == '8':
            # Sensitivity Post-Analysis
            sens_dir = os.path.join(ctx.output_dir, "sensitivity")
            if os.path.exists(sens_dir):
                print("  Re-generating sensitivity visualizations...")
                # We can call the script with minimal samples just to trigger plotting if results exist
                # but better to just re-run with current logic as sensitivity_analysis.py usually handles both
                run_sensitivity_workflow(ctx)
            else:
                print("  [WARN] No sensitivity results found. Run calculation first (Option 2).")
                
        elif choice == '9':
            # Indices Post-Analysis
            ent_dir = os.path.join(ctx.output_dir, "entropy_analysis")
            if os.path.exists(ent_dir):
                print("  Updating Pollution Indices Visualizations & Tables...")
                run_indices_workflow(ctx)
            else:
                print("  [WARN] No indices results found. Run calculation first (Option 3).")
                
        elif choice == '10':
            ctx = configure_session(default_file=ctx.input_file)
            if not ctx: break
            
        else:
            print("  [WARN] Invalid choice. Please try again.")

    return 0


def quick_mode(args):
    """Run in quick mode for batch/automated processing (Original Logic)."""
    print_banner()
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return 1
    
    valid, info = validate_chemistry_file(args.input)
    if not valid:
        print(f"Error: {info}")
        return 1
        
    # Set output directory - explicitly save in project root (outside src/)
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results_{Path(args.input).stem}_{timestamp}"
    
    # Force output to be in project root if relative path
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(os.path.join(PROJECT_ROOT, output_dir))
        
    os.makedirs(output_dir, exist_ok=True)
    
    input_units = 'mg/L' if args.units in ['mg/L', 'ppm'] else 'µg/L'
    
    # Standard sequential run
    if not args.skip_speciation:
        run_spec_cmd = [sys.executable, os.path.join(SCRIPT_DIR, "speciation_modeling.py"),
                       "--input", args.input, "--output-dir", output_dir, "--use-total-fallback"]
        if input_units == 'mg/L': run_spec_cmd.extend(["--input-units", "mg/L"])
        import subprocess
        subprocess.run(run_spec_cmd)
        
    run_bll_calibration(args.input, output_dir, input_units)
    
    hbmp_cmd = [sys.executable, os.path.join(SCRIPT_DIR, "hbmpra_optimized.py"),
               "--chemistry", args.input, "--results-dir", output_dir,
               "--draws", str(args.draws), "--tune", str(args.tune), "--use-bioavailable"]
    if input_units == 'mg/L': hbmp_cmd.extend(["--input-units", "mg/L"])
    import subprocess
    subprocess.run(hbmp_cmd)
    
    if not args.skip_plots:
        run_plot_diagnostics(output_dir)
        run_plot_results(output_dir)
    if not args.skip_tables:
        run_summary_tables(output_dir, args.input)
    if not args.skip_sensitivity:
        run_sensitivity_analysis(args.input, output_dir, method='sobol', n_samples=512, input_units=input_units)
    if not args.skip_entropy:
        run_entropy_hpi_peri(args.input, output_dir, input_units=input_units)
        
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="HBMPRA Interactive Analysis Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", "-i", help="Input chemistry CSV file")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--skip-speciation", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-tables", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--skip-entropy", action="store_true")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--units", choices=['ug/L', 'mg/L', 'ppb', 'ppm'], default='ug/L')
    parser.add_argument("--interactive", action="store_true")
    
    args = parser.parse_args()
    
    if args.interactive or (not args.input and len(sys.argv) == 1):
        return interactive_mode()
    elif args.input:
        return quick_mode(args)
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

