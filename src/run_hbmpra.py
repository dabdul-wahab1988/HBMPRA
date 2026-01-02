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


def interactive_mode():
    """Run in interactive mode with user prompts."""
    print_banner()
    
    TOTAL_STEPS = 11
    
    # Step 1: Check dependencies
    print_step(1, TOTAL_STEPS, "Checking system requirements")
    if not check_dependencies():
        print("\nPlease install missing packages and try again.")
        return 1
    
    # Step 2: Find and select data file
    print_step(2, TOTAL_STEPS, "Select input data file")
    
    data_files = find_data_files()
    
    if not data_files:
        print("No CSV data files found in current directory.")
        filepath = input("Enter path to your chemistry CSV file: ").strip()
        if not filepath:
            print("No file provided. Exiting.")
            return 1
    elif len(data_files) == 1:
        filepath = str(data_files[0])
        print(f"Found data file: {filepath}")
        use_it = input("Use this file? [Y/n]: ").strip().lower()
        if use_it == 'n':
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
    
    # Validate the file
    valid, info = validate_chemistry_file(filepath)
    if not valid:
        print(f"\n[WARN] Error: {info}")
        return 1
    
    print(f"\n[OK] Valid chemistry file:")
    print(f"  ✓ {info['rows']} samples/sites detected")
    if info['metals']:
        print(f"  ✓ Metals: {', '.join(info['metals'])} ({len(info['metals'])} total)")
    else:
        print(f"  ○ No metals detected")
    
    # Display anion info (NEW)
    if info.get('anions'):
        anion_display = []
        for anion in info['anions']:
            if anion == 'nitrate' and 'nitrate' in info.get('anion_info', {}):
                basis = info['anion_info']['nitrate'].get('basis', 'NO3')
                anion_display.append(f"NO₃ ({basis} basis)")
            else:
                anion_display.append('F⁻' if anion == 'fluoride' else anion)
        print(f"  ✓ Anions: {', '.join(anion_display)} ({len(info['anions'])} total)")
        
        # Show conversion note for nitrate
        if 'nitrate' in info['anions'] and 'nitrate' in info.get('anion_info', {}):
            nitrate_basis = info['anion_info']['nitrate'].get('basis', 'NO3')
            if nitrate_basis == 'NO3':
                print(f"    ⚠ NO₃ will be converted to NO₃–N basis for HQ calculation")
    else:
        print(f"  ○ No anions (F, NO₃) detected")
    
    print(f"  {'✓' if info['has_pH'] else '○'} pH data: {'available' if info['has_pH'] else 'not found (will use default pH=7.0)'}")
    print(f"  {'✓' if info['has_Eh'] else '○'} Eh data: {'available' if info['has_Eh'] else 'not found (will use default Eh=300mV)'}")
    
    # Step 2b: Select workflow preset (NEW)
    print("\n" + "="*50)
    print("  SELECT ANALYSIS MODE")
    print("="*50)
    print("  1. Quick Scan    - Fast screening (1000 draws, tables only)")
    print("  2. Standard      - Full analysis with custom plots [RECOMMENDED]")
    print("  3. Publication   - High-quality (4000 draws + sensitivity/entropy)")
    print("  4. Custom        - Configure each setting manually")
    preset_choice = input("\n  Select mode [1-4, default=2]: ").strip()
    
    if preset_choice == '1':
        preset = 'quick'
        mcmc_draws = 1000
        run_advanced = False
        skip_figures = True  # Skip figures for quick testing
        print("  → Quick Scan: Fast results, no plots")
    elif preset_choice == '3':
        preset = 'publication'
        mcmc_draws = 4000
        run_advanced = True  # Auto-run sensitivity & entropy
        skip_figures = False
        print("  → Publication: High-quality + all advanced analyses")
    elif preset_choice == '4':
        preset = 'custom'
        run_advanced = None  # Will be prompted
        skip_figures = None  # Will be prompted
        print("  → Custom: You'll configure every setting")
        
        # Prompt for draws in Custom mode
        print("\n  Configure MCMC sampling:")
        print("    Draws determine accuracy vs speed tradeoff")
        print("    1000 = fast (~5 min), 2000 = standard (~10 min), 4000 = publication (~20 min)")
        draws_input = input("  Number of draws [default=2000]: ").strip()
        try:
            mcmc_draws = int(draws_input) if draws_input else 2000
            if mcmc_draws < 500:
                print(f"  [WARN] {mcmc_draws} is very low, using 500 minimum")
                mcmc_draws = 500
        except ValueError:
            mcmc_draws = 2000
        print(f"  → Using {mcmc_draws} draws")
    else:
        preset = 'standard'
        mcmc_draws = 2000
        run_advanced = False
        skip_figures = False
        print("  → Standard: Recommended defaults with custom plots")
    
    # Step 2c: Select demographic groups (NEW)
    try:
        from demographics import (GROUP_INFO, GROUP_PRESETS, parse_group_selection, 
                                 get_group_info_filtered, print_group_selection_help)
    except ImportError:
        from .demographics import (GROUP_INFO, GROUP_PRESETS, parse_group_selection,
                                   get_group_info_filtered, print_group_selection_help)
    
    print("\n" + "="*60)
    print("  DEMOGRAPHIC GROUP SELECTION")
    print("="*60)
    print("\n  Available groups:")
    for i, (name, info_d) in enumerate(GROUP_INFO.items(), 1):
        bw = info_d['BW']
        ir = info_d['IR']
        ed_years = info_d['ED'] / 365
        print(f"    {i}. {name:10} (BW={bw}kg, IR={ir}L/d, {ed_years:.0f}y exposure)")
    
    print("\n  Presets:")
    print("    all           → Adults, Children, Teens, Pregnant [default]")
    print("    sensitive     → Children, Pregnant (vulnerable populations)")
    print("    non_sensitive → Adults, Teens")
    print("    adults_only   → Adults only")
    print("    children_only → Children only")
    
    group_selection = input("\n  Select groups [numbers, names, or preset]: ").strip()
    if not group_selection:
        group_selection = 'all'
    
    try:
        selected_groups = parse_group_selection(group_selection)
        print(f"  → Analyzing: {', '.join(selected_groups)}")
    except ValueError as e:
        print(f"  [WARN] {e}")
        print("  → Using all groups as fallback")
        selected_groups = list(GROUP_INFO.keys())
    print("="*60)
    
    # Step 3: Specify concentration units
    print("\n  Concentration units in your data:")
    print("    1. µg/L (micrograms per liter) - default")
    print("    2. mg/L (milligrams per liter)")
    print("    3. ppb (parts per billion, same as µg/L)")
    print("    4. ppm (parts per million, same as mg/L)")
    unit_choice = input("  Select units [1-4, default=1]: ").strip()
    
    if unit_choice == '2' or unit_choice.lower() == 'mg/l' or unit_choice == '4' or unit_choice.lower() == 'ppm':
        concentration_unit = 'mg/L'
        print(f"  → Using mg/L (data will be converted to µg/L internally: multiply by 1000)")
    else:
        concentration_unit = 'µg/L'
        print(f"  → Using µg/L (standard unit, no conversion needed)")
    
    # Step 3: Configure output
    print_step(3, TOTAL_STEPS, "Configure output directory")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = f"results_{Path(filepath).stem}_{timestamp}"
    
    output_dir = input(f"Output directory [{default_output}]: ").strip()
    if not output_dir:
        output_dir = default_output
    
    # Step 4: Run speciation (optional)
    print_step(4, TOTAL_STEPS, "Thermodynamic speciation modeling")
    
    try:
        import phreeqpython
        has_phreeqc = True
    except ImportError:
        has_phreeqc = False
    
    # Determine if this is an anion-only analysis (no metals)
    is_anion_only = not info.get('metals') and info.get('anions')
    has_lead = 'Pb' in info.get('metals', [])
    
    if is_anion_only:
        print("\n" + "="*50)
        print("  ANION-ONLY ANALYSIS MODE")
        print("="*50)
        print("  Your data contains only anions (F, NO₃), no metals.")
        print("  The following will be adjusted:")
        print("    • Speciation modeling: SKIPPED (not applicable)")
        print("    • BLL calibration: SKIPPED (no lead data)")
        print("    • Cancer risk: SKIPPED (anions are not carcinogens)")
        print("    • Hazard Index: COMPUTED for F and NO₃")
        print("="*50)
    
    run_speciation = False
    if not is_anion_only and has_phreeqc:
        print("PHREEQC is available for thermodynamic speciation modeling.")
        print("This calculates bioavailable metal species (recommended for accuracy).")
        choice = input("Run speciation modeling? [Y/n]: ").strip().lower()
        run_speciation = choice != 'n'
    elif is_anion_only:
        print("  [SKIP] Speciation modeling (not applicable for anions)")
    else:
        print("PHREEQC not available. Using simplified speciation estimates.")
        print("(Install phreeqpython for full thermodynamic modeling)")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Run speciation if requested
    if run_speciation:
        print("\nRunning PHREEQC speciation modeling...")
        speciation_cmd = [
            sys.executable, os.path.join(SCRIPT_DIR, "speciation_modeling.py"),
            "--input", filepath,
            "--output-dir", output_dir,
            "--use-total-fallback"
        ]
        # Add unit conversion flag if needed
        if concentration_unit == 'mg/L':
            speciation_cmd.extend(["--input-units", "mg/L"])
        
        import subprocess
        result = subprocess.run(speciation_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[WARN] Speciation modeling had issues:")
            print(result.stderr[-500:] if result.stderr else "Unknown error")
            print("Continuing with simplified speciation...")
        else:
            print("[OK] Speciation modeling complete")
    
    # Step 5: Run BLL calibration (skip for anion-only)
    print_step(5, TOTAL_STEPS, "BLL (Blood Lead Level) prior calibration")
    if is_anion_only or not has_lead:
        if is_anion_only:
            print("  [SKIP] BLL calibration (anion-only analysis, no lead data)")
        else:
            print("  [SKIP] BLL calibration (no lead (Pb) detected in data)")
    else:
        run_bll_calibration(filepath, output_dir, concentration_unit)
    
    # Step 6: Run the main analysis
    print_step(6, TOTAL_STEPS, "Running Bayesian risk assessment")
    
    # Run main HBMPRA model
    print("\nRunning Bayesian risk assessment model...")
    print("(This may take 5-15 minutes depending on your computer)")
    print(f"Concentration units: {concentration_unit}")
    
    hbmpra_cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "hbmpra_optimized.py"),
        "--chemistry", filepath,
        "--results-dir", output_dir,
        "--draws", str(mcmc_draws),  # Based on preset selection
        "--tune", str(mcmc_draws),
        "--groups", ",".join(selected_groups),  # Pass selected groups
        "--use-bioavailable",
        "--allow-default-organ-sets",
        "--allow-disable-dermal-if-no-bio"
    ]
    # Add unit conversion flag if needed
    if concentration_unit == 'mg/L':
        hbmpra_cmd.extend(["--input-units", "mg/L"])
    
    import subprocess
    result = subprocess.run(hbmpra_cmd)
    
    if result.returncode != 0:
        print(f"\n[WARN] Analysis encountered errors. Check the output above.")
        return 1
    
    # Quick Results Preview (always shown)
    print("\n" + "="*50)
    print("  QUICK RESULTS PREVIEW")
    print("="*50)
    try:
        import arviz as az
        trace_file = os.path.join(output_dir, "trace.nc")
        if os.path.exists(trace_file):
            idata = az.from_netcdf(trace_file)
            post = idata.posterior
            
            # Show HI summary
            if "HI_overall" in post:
                hi_mean = float(post["HI_overall"].mean())
                hi_max = float(post["HI_overall"].max())
                print(f"  HI_overall:  mean={hi_mean:.3f}, max={hi_max:.3f}")
                if hi_mean > 1:
                    print(f"    ⚠ WARNING: Mean HI > 1 indicates potential health risk!")
            
            # Show CR summary if available
            if "CR_total" in post:
                cr_mean = float(post["CR_total"].mean())
                print(f"  CR_total:    mean={cr_mean:.2e}")
                if cr_mean > 1e-4:
                    print(f"    ⚠ WARNING: CR > 10⁻⁴ indicates elevated cancer risk")
            
            # Show BLL summary if available
            if "BLL" in post:
                bll_mean = float(post["BLL"].mean())
                print(f"  BLL:         mean={bll_mean:.2f} µg/dL")
                if bll_mean > 3.5:
                    print(f"    ⚠ WARNING: BLL > 3.5 µg/dL exceeds CDC reference value")
        else:
            print("  [WARN] Could not load results for preview")
    except Exception as e:
        print(f"  [WARN] Could not generate preview: {e}")
    print("="*50)
    
    # Determine whether to generate figures (diagnostics vs results independently)
    if skip_figures is None:  # Custom mode: ask separately
        diag_choice = input("\nGenerate diagnostic plots? [y/N]: ").strip().lower()
        result_choice = input("Generate result plots? [y/N]: ").strip().lower()
        generate_diag = diag_choice == 'y'
        generate_results = result_choice == 'y'
    else:
        generate_diag = not skip_figures
        generate_results = not skip_figures
    
    # Step 7: Custom Diagnostic Plots
    print_step(7, TOTAL_STEPS, "Diagnostic plots")
    if generate_diag:
        from plot_diagnostics_panel import interactive_diagnostic_builder
        trace_file = os.path.join(output_dir, "trace.nc")
        model_file = os.path.join(output_dir, "model.pkl")
        diag_figs = interactive_diagnostic_builder(trace_file, model_file, output_dir)
    else:
        print("  [SKIP] Skipping diagnostic plots (user choice/preset)")
    
    # Step 8: Custom Result Plots
    print_step(8, TOTAL_STEPS, "Result plots")
    if generate_results:
        from plot_panel import interactive_plot_builder
        import arviz as az
        
        trace_file = os.path.join(output_dir, "trace.nc")
        if os.path.exists(trace_file):
            try:
                idata = az.from_netcdf(trace_file)
                
                # Get BLL thresholds from RUNLOG if available
                runlog_path = os.path.join(output_dir, "RUNLOG.json")
                bll_thresholds = "3.5,5,10"
                if os.path.exists(runlog_path):
                    import json
                    with open(runlog_path) as f:
                        runlog = json.load(f)
                    bll_list = runlog.get("bll_thresholds", [3.5, 5, 10])
                    bll_thresholds = ",".join(str(x) for x in bll_list)
                
                result_figs = interactive_plot_builder(idata, output_dir, bll_thresholds)
                    
            except Exception as e:
                print(f"  [WARN] Could not load trace for result plots: {e}")
                result_figs = 0
        else:
            print(f"  [WARN] Trace file not found: {trace_file}")
            result_figs = 0
    else:
        print("  [SKIP] Skipping result plots (user choice/preset)")
    
    # Step 9: Generate summary tables
    print_step(9, TOTAL_STEPS, "Generating summary tables")
    run_summary_tables(output_dir, filepath)
    
    # Step 10: Sensitivity Analysis (Optional, Advanced)
    print_step(10, TOTAL_STEPS, "Sensitivity Analysis (optional, advanced)")
    print("Sensitivity analysis evaluates how input parameters affect model outputs.")
    print("Methods available: Sobol (variance-based), Morris (screening), Delta (moment)")
    
    # Use preset-based decision or prompt user
    if run_advanced is True:
        print("  [AUTO] Running sensitivity analysis (Publication mode)")
        run_sens_choice = 'y'
    elif run_advanced is False:
        print("  [SKIP] Skipping sensitivity analysis (Use Publication or Custom mode to enable)")
        run_sens_choice = 'n'
    else:  # Custom mode
        print("NOTE: This can take 10-30+ minutes depending on sample size.")
        run_sens_choice = input("Run sensitivity analysis? [y/N]: ").strip().lower()
    
    if run_sens_choice == 'y':
        print("\nSelect sensitivity method:")
        print("  1. sobol  - Sobol indices (most thorough, slowest)")
        print("  2. morris - Morris method (faster screening)")
        print("  3. delta  - Delta moment (distribution-based)")
        method_choice = input("Method [1/2/3, default=1]: ").strip()
        
        if method_choice == '2':
            sens_method = 'morris'
        elif method_choice == '3':
            sens_method = 'delta'
        else:
            sens_method = 'sobol'
        
        samples_input = input("Number of samples [default=512]: ").strip()
        try:
            n_samples = int(samples_input) if samples_input else 512
        except ValueError:
            n_samples = 512
        
        run_sensitivity_analysis(filepath, output_dir, method=sens_method, n_samples=n_samples, input_units=concentration_unit)
    
    # Step 11: Entropy-based HPI/PERI Analysis (Optional, Advanced)
    print_step(11, TOTAL_STEPS, "Entropy HPI/PERI Analysis (optional, advanced)")
    print("Entropy-weighted Heavy Metal Pollution Index (HPI) and")
    print("Potential Ecological Risk Index (PERI) analysis.")
    print("Requires standards.csv and toxicity.csv files.")
    
    # Use preset-based decision or prompt user
    if run_advanced is True:
        print("  [AUTO] Running entropy analysis (Publication mode)")
        run_entropy_choice = 'y'
    elif run_advanced is False:
        print("  [SKIP] Skipping entropy analysis (Use Publication or Custom mode to enable)")
        run_entropy_choice = 'n'
    else:  # Custom mode
        run_entropy_choice = input("Run entropy HPI/PERI analysis? [y/N]: ").strip().lower()
    
    if run_entropy_choice == 'y':
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
        
        if standards_file and toxicities_file:
            print(f"  Found standards: {standards_file}")
            print(f"  Found toxicities: {toxicities_file}")
        else:
            if not standards_file:
                standards_file = input("  Path to standards.csv: ").strip()
            if not toxicities_file:
                toxicities_file = input("  Path to toxicity.csv: ").strip()
        
        bootstrap_input = input("Bootstrap samples [default=1000]: ").strip()
        try:
            bootstrap_samples = int(bootstrap_input) if bootstrap_input else 1000
        except ValueError:
            bootstrap_samples = 1000
        
        run_entropy_hpi_peri(filepath, output_dir, standards_file, toxicities_file, bootstrap_samples, input_units=concentration_unit)
    else:
        print("  [SKIP] Skipping entropy HPI/PERI analysis")
    
    # Note: Custom plot builder is now integrated into Steps 7-8
    
    # Summary
    print("\n" + "="*70)
    print("  ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print("\nKey output files:")
    print(f"  • {output_dir}/trace.nc              - Full Bayesian model results")
    print(f"  • {output_dir}/RUNLOG.json           - Analysis configuration")
    print(f"  • {output_dir}/ASSUMPTIONS.json      - Model assumptions")
    print(f"  • {output_dir}/debug/HI_summary.csv  - Hazard Index summary")
    
    if run_speciation:
        print(f"  • {output_dir}/table_species_fractions.csv")
        print(f"  • {output_dir}/table_bioavailable_concentrations.csv")
    
    print(f"\nDiagnostic plots:   {output_dir}/diagnostics/")
    print(f"Result figures:     {output_dir}/figures/")
    print(f"Summary tables:     {output_dir}/tables/")
    
    # Show advanced analysis output directories if they were run
    if run_sens_choice == 'y':
        print(f"Sensitivity analysis: {output_dir}/sensitivity/")
    if run_entropy_choice == 'y':
        print(f"Entropy HPI/PERI:   {output_dir}/entropy_analysis/")
    
    print("\nNext steps:")
    print("  1. Review HI_summary.csv for hazard indices by organ system")
    print("  2. Check figures/ for publication-ready plots")
    print("  3. Check tables/ for summary statistics and PCA results")
    if run_sens_choice == 'y':
        print("  4. Check sensitivity/ for parameter influence analysis")
    if run_entropy_choice == 'y':
        print("  5. Check entropy_analysis/ for HPI/PERI pollution indices")
    
    return 0


def quick_mode(args):
    """Run in quick mode with command-line arguments."""
    print_banner()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    valid, info = validate_chemistry_file(args.input)
    if not valid:
        print(f"Error: {info}")
        return 1
    
    print(f"Input: {args.input}")
    print(f"  - {info['rows']} samples, {len(info['metals'])} metals")
    
    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results_{Path(args.input).stem}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}/")
    
    import subprocess
    
    # Determine concentration units
    input_units = 'µg/L'  # default
    if args.units in ['mg/L', 'ppm']:
        input_units = 'mg/L'
        print(f"Input units: {args.units} (will convert to µg/L internally)")
    else:
        print(f"Input units: {args.units}")
    
    # Run speciation if not skipped
    if not args.skip_speciation:
        print("\n[1/8] Running speciation modeling...")
        speciation_cmd = [
            sys.executable, os.path.join(SCRIPT_DIR, "speciation_modeling.py"),
            "--input", args.input,
            "--output-dir", output_dir,
            "--use-total-fallback"
        ]
        if input_units == 'mg/L':
            speciation_cmd.extend(["--input-units", "mg/L"])
        result = subprocess.run(speciation_cmd)
        if result.returncode != 0:
            print("Warning: Speciation had issues, continuing...")
    
    # Run BLL calibration
    print("\n[2/8] Calibrating BLL priors...")
    run_bll_calibration(args.input, output_dir, input_units)
    
    # Run HBMPRA
    print("\n[3/8] Running Bayesian risk assessment...")
    hbmpra_cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "hbmpra_optimized.py"),
        "--chemistry", args.input,
        "--results-dir", output_dir,
        "--draws", str(args.draws),
        "--tune", str(args.tune),
        "--use-bioavailable",
        "--allow-default-organ-sets",
        "--allow-disable-dermal-if-no-bio"
    ]
    if input_units == 'mg/L':
        hbmpra_cmd.extend(["--input-units", "mg/L"])
    
    result = subprocess.run(hbmpra_cmd)
    
    if result.returncode != 0:
        print(f"\n[WARN] Errors occurred in Bayesian analysis. Check output above.")
        return result.returncode
    
    # Post-processing: Generate plots and tables
    if not args.skip_plots:
        print("\n[4/8] Generating diagnostic plots...")
        run_plot_diagnostics(output_dir)
        
        print("\n[5/8] Generating result figures...")
        run_plot_results(output_dir)
    
    if not args.skip_tables:
        print("\n[6/8] Generating summary tables...")
        run_summary_tables(output_dir, args.input)
    
    # Advanced analyses (optional)
    if not args.skip_sensitivity:
        print("\n[7/8] Running sensitivity analysis...")
        run_sensitivity_analysis(args.input, output_dir, method='sobol', n_samples=512, input_units=input_units)
    else:
        print("\n[7/8] Skipping sensitivity analysis (use --skip-sensitivity=False to include)")
    
    if not args.skip_entropy:
        print("\n[8/8] Running entropy HPI/PERI analysis...")
        # Default to project root for standards and toxicity files
        standards_file = os.path.join(PROJECT_ROOT, "standards.csv")
        toxicities_file = os.path.join(PROJECT_ROOT, "toxicity.csv")
        
        # Also check waterdata folder
        if not os.path.exists(standards_file):
            standards_file = os.path.join(WATERDATA_DIR, "standards.csv")
        if not os.path.exists(toxicities_file):
            toxicities_file = os.path.join(WATERDATA_DIR, "toxicity.csv")
        
        run_entropy_hpi_peri(args.input, output_dir, standards_file, toxicities_file, input_units=input_units)
    else:
        print("\n[8/8] Skipping entropy HPI/PERI analysis (use --skip-entropy=False to include)")
    
    # Final summary
    print("\n" + "="*70)
    print("  ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print("\nKey output directories:")
    print(f"  • {output_dir}/debug/        - HI_summary.csv")
    print(f"  • {output_dir}/diagnostics/  - Convergence and trace plots")
    print(f"  • {output_dir}/figures/      - Publication-ready figures")
    print(f"  • {output_dir}/tables/       - Summary tables and statistics")
    if not args.skip_sensitivity:
        print(f"  • {output_dir}/sensitivity/  - Sensitivity analysis results")
    if not args.skip_entropy:
        print(f"  • {output_dir}/entropy_analysis/ - HPI/PERI pollution indices")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="HBMPRA - User-Friendly Risk Assessment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_hbmpra.py                    # Interactive mode (recommended)
  python run_hbmpra.py --input data1.csv  # Quick mode with all analyses
  python run_hbmpra.py --input data1.csv --output results_myanalysis
  python run_hbmpra.py --input data1.csv --skip-plots  # Skip figure generation
  python run_hbmpra.py --input data1.csv --skip-sensitivity --skip-entropy  # Skip advanced analyses

For beginners: Just run 'python run_hbmpra.py' and follow the prompts!
        """
    )
    
    parser.add_argument("--input", "-i", 
                        help="Input chemistry CSV file")
    parser.add_argument("--output", "-o",
                        help="Output directory (default: auto-generated)")
    parser.add_argument("--skip-speciation", action="store_true",
                       help="Skip PHREEQC speciation modeling")
    parser.add_argument("--skip-plots", action="store_true",
                       help="Skip generating diagnostic plots and result figures")
    parser.add_argument("--skip-tables", action="store_true",
                       help="Skip generating summary tables")
    parser.add_argument("--skip-sensitivity", action="store_true",
                       help="Skip sensitivity analysis (advanced)")
    parser.add_argument("--skip-entropy", action="store_true",
                       help="Skip entropy-based HPI/PERI analysis (advanced)")
    parser.add_argument("--draws", type=int, default=1000,
                        help="Number of MCMC draws (default: 1000, use 2000+ for publication)")
    parser.add_argument("--tune", type=int, default=1000,
                        help="Number of tuning samples (default: 1000)")
    parser.add_argument("--units", choices=['ug/L', 'mg/L', 'ppb', 'ppm'], default='ug/L',
                        help="Concentration units in input file (default: ug/L). Use mg/L or ppm if concentrations are in mg/L.")
    parser.add_argument("--interactive", action="store_true",
                        help="Force interactive mode")
    
    args = parser.parse_args()
    
    # Determine mode
    if args.interactive or (not args.input and len(sys.argv) == 1):
        return interactive_mode()
    elif args.input:
        return quick_mode(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

