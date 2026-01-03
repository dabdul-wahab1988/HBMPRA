# Interactive Workflow Guide: run_hbmpra.py

This document provides comprehensive documentation for the **interactive workflow orchestrator** (`run_hbmpra.py`), which guides users through the complete HBMPRA analysis pipeline.

---

## Table of Contents

1. [Overview](#1-overview)
2. [The 11-Step Workflow](#2-the-11-step-workflow)
3. [User Interaction Patterns](#3-user-interaction-patterns)
4. [Decision Trees and Logic Flow](#4-decision-trees-and-logic-flow)
5. [Command-Line Options](#5-command-line-options)
6. [Error Handling and Validation](#6-error-handling-and-validation)
7. [Customization and Extension](#7-customization-and-extension)

---

## 1. Overview

### 1.1 Purpose

`run_hbmpra.py` serves as the **user-friendly entry point** to the HBMPRA framework, designed specifically for **non-coders** and researchers who need a guided, step-by-step analysis workflow.

### 1.2 Design Philosophy

- **Progressive Disclosure**: Complex options are only presented when relevant
- **Sensible Defaults**: All parameters have reasonable defaults based on EPA/WHO guidelines
- **Fail-Gracefully**: Errors provide helpful messages with recovery options
- **Transparent**: Each step explains what it does and why

### 1.3 Usage Modes

#### Interactive Mode (Recommended)

```bash
python run_hbmpra.py
```

Launches a guided session with prompts for each decision point.

#### Quick Mode (With Defaults)

```bash
python run_hbmpra.py --input waterdata/data1.csv
```

Runs the full workflow with default settings.

#### Help Mode

```bash
python run_hbmpra.py --help
```

Displays all available command-line options.

### 1.4 Workflow Presets (NEW)

When running in interactive mode, you'll be prompted to select an analysis mode:

| Mode | MCMC Draws | Advanced Analyses | Best For |
|------|------------|-------------------|----------|
| **Quick Scan** | 1,000 | Skipped | Fast screening, testing |
| **Standard** | 2,000 | Skipped (prompt in Custom) | Most analyses [RECOMMENDED] |
| **Publication** | 4,000 | Auto-run | Final results, manuscripts |
| **Custom** | User-specified | User-prompted | Full control |

```
==================================================
  SELECT ANALYSIS MODE
==================================================
  1. Quick Scan    - Fast screening (1000 draws, core outputs only)
  2. Standard      - Full analysis with plots and tables [RECOMMENDED]
  3. Publication   - High-quality (4000 draws, all advanced analyses)
  4. Custom        - Configure each step manually
```

### 1.5 Anion Support (NEW)

HBMPRA now automatically detects **Fluoride (F⁻)** and **Nitrate (NO₃⁻)** columns:

- **Fluoride**: Columns named `F`, `F-`, `Fluoride`
- **Nitrate**: Columns named `NO3`, `NO3-`, `Nitrate`, `NO3_N` (for NO₃–N basis)

Validation output now shows:

```
[OK] Valid chemistry file:
  ✓ 25 samples/sites detected
  ✓ Metals: As, Cd, Pb (3 total)
  ✓ Anions: F⁻, NO₃ (NO3 basis) (2 total)
    ⚠ NO₃ will be converted to NO₃–N basis for HQ calculation
```

### 1.6 Anion-Only Analysis Mode (NEW)

If your data contains **only anions** (F, NO₃) with no metals, HBMPRA automatically enters **Anion-Only Mode**:

```
==================================================
  ANION-ONLY ANALYSIS MODE
==================================================
  Your data contains only anions (F, NO₃), no metals.
  The following will be adjusted:
    • Speciation modeling: SKIPPED (not applicable)
    • BLL calibration: SKIPPED (no lead data)
    • Cancer risk: SKIPPED (anions are not carcinogens)
    • Hazard Index: COMPUTED for F and NO₃
==================================================
```

**What gets computed:**

- `HQ_fluoride`, `HQ_nitrate` — Hazard Quotients
- `HI_skeletal_dental` — Fluoride organ-specific HI
- `HI_hematological` — Nitrate organ-specific HI
- `HI_overall` — Maximum of organ HIs

**What is skipped:**

- PHREEQC speciation (not applicable for simple anions)
- BLL calibration (no lead data)
- Cancer risk (anions are not carcinogens)

---

## 2. The 11-Step Workflow

### Step 1: Check System Requirements

**Purpose**: Verify that all required and optional dependencies are installed.

**Logic** (`check_dependencies()`, lines 62-101):

```
Required Packages:
  • numpy, pandas, matplotlib, pyyaml
  → FATAL ERROR if missing

Optional Packages:
  • phreeqpython (thermodynamic speciation)
  • pymc (Bayesian modeling)
  • arviz (results analysis)
  → WARNING if missing, but continues
```

**User Action**: If packages are missing, install with:

```bash
pip install numpy pandas matplotlib pyyaml pymc arviz phreeqpython
```

---

### Step 2: Select Input Data File

**Purpose**: Locate and validate the water chemistry CSV file.

**Logic** (`find_data_files()`, lines 505-542):

```
Search Locations (in order):
  1. waterdata/ folder
  2. Current working directory
  3. User-specified path

File Filtering:
  • Must be .csv extension
  • Exclude: table_*, summary*, toxicity.csv, standards.csv
```

**Validation** (`validate_chemistry_file()`, lines 545-585):

```python
# Check for metal columns (case-insensitive)
Expected patterns: 'C_As', 'As', 'Cd', 'C_Pb', etc.
Minimum requirement: At least 1 metal column found

# Check for site identifier (first column)
Accepted names: 'site', 'Site', 'ID', 'Community', 'Location'

# Report summary:
  - Number of sites/samples
  - Metals detected
  - pH/Eh availability (used for speciation)
```

**Interactive Prompts**:

```
If 0 files found:
  → "Enter path to your chemistry CSV file: "

If 1 file found:
  → "Use this file? [Y/n]: "

If multiple files:
  → "Select file [1-N] or enter path: "
```

---

### Step 2b: Specify Concentration Units

**Purpose**: Handle unit conversion for international datasets.

**Options**:

```
1. µg/L (micrograms per liter) - default
2. mg/L (milligrams per liter)
3. ppb (parts per billion, same as µg/L)
4. ppm (parts per million, same as mg/L)
```

**Conversion Logic**:

- If `mg/L` selected: multiply all concentrations by 1000 internally
- All downstream calculations use µg/L as the standard unit

**Code reference**: Lines 641-654

---

### Step 3: Configure Output Directory

**Purpose**: Organize results with timestamped directories.

**Default Naming** (line 660):

```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
default_output = f"results_{Path(filepath).stem}_{timestamp}"
# Example: results_water_samples_20250122_114519
```

**User Action**:

- Press Enter to accept default
- Or type custom directory name

**Output Structure**:

```
results_*/
├── trace.nc                     # Bayesian MCMC samples
├── model.pkl                    # PyMC model object
├── RUNLOG.json                  # Analysis configuration
├── ASSUMPTIONS.json             # Model assumptions
├── calibration/
│   └── priors.json             # BLL calibrated priors
├── diagnostics/                 # MCMC diagnostics
├── figures/                     # Publication-ready plots
├── tables/                      # Summary CSV tables
├── sensitivity/                 # (optional) Sensitivity results
└── entropy_analysis/            # (optional) HPI/PERI results
```

---

### Step 4: Thermodynamic Speciation Modeling

**Purpose**: Calculate bioavailable metal species using PHREEQC.

**Decision Tree**:

```
Is phreeqpython installed?
├─ YES:
│   └─ Prompt: "Run speciation modeling? [Y/n]"
│       ├─ YES → Execute speciation_modeling.py
│       └─ NO  → Use simplified estimates (total = bioavailable)
└─ NO:
    └─ Display: "PHREEQC not available. Using simplified speciation estimates."
    └─ Suggest: "Install phreeqpython for full thermodynamic modeling"
```

**What Speciation Does** (see [Speciation_and_Metal_Chemistry.md](Speciation_and_Metal_Chemistry.md)):

1. Solves Gibbs Free Energy minimization for chemical equilibrium
2. Calculates fractions of toxic species (e.g., CrVI vs CrIII)
3. Generates:
   - `table_bioavailable_concentrations.csv`
   - `table_species_fractions.csv`
   - `figure_speciation_profiles.png`

**Fallback Behavior**:

- If PHREEQC fails: warns but continues with total concentrations as bioavailable.
- Metals without total columns are skipped cleanly to avoid false errors; present metals still run.
- Flag `--use-total-fallback` is automatically added for robustness when PHREEQC outputs low-variation species.

**Code reference**: Lines 666-708

---

### Step 5: BLL (Blood Lead Level) Prior Calibration

**Purpose**: Generate population-specific Bayesian priors for lead exposure.

**Automatic Engine Selection** (lines 104-179):

```
Population Group → Engine Selection:
  • Adults        → onecomp (mechanistic pharmacokinetic)
  • Children      → slope (empirical dose-response)
  • Teens         → slope (empirical)
  • Pregnant      → slope (conservative, empirical)
```

**Calibration Process** (`calibrate_bll_priors.py`):

1. Generate geometric grid of Pb concentrations (15 points)
2. For each concentration:
   - Calculate EDI (Estimated Daily Intake)
   - Compute BLL using population-specific engine
3. Fit linear relationship: $BLL = b_0 + k \cdot EDI$
4. Save calibrated parameters to `calibration/priors.json`

**Output Summary**:

```
Calibrated parameters (k = intake→BLL slope):
  • Adults: k=0.0234, b0=0.5000
  • Children: k=0.1700, b0=1.2000
  • Teens: k=0.1700, b0=0.8000
  • Pregnant: k=0.0800, b0=0.6000
```

**Why This Matters**: The calibrated priors ensure that the Bayesian model's lead predictions are physiologically realistic and population-appropriate.

**Code reference**: Lines 104-179, 710-712

---

### Step 6: Run Bayesian Risk Assessment

**Purpose**: Execute the main PyMC hierarchical Bayesian model.

**Command Construction** (lines 722-734):

```python
hbmpra_cmd = [
    sys.executable, "hbmpra_optimized.py",
    "--chemistry", filepath,
    "--results-dir", output_dir,
    "--draws", "1000",                          # MCMC samples
    "--tune", "1000",                           # Burn-in samples
    "--use-bioavailable",                       # Use speciation results
    "--allow-default-organ-sets",               # Permit fallback organ sets
    "--allow-disable-dermal-if-no-bio"         # Graceful dermal handling
]
```

**What This Does**:

1. Constructs PyMC model with hierarchical priors (see [HBMPRA_Mathematical_Foundations.md](HBMPRA_Mathematical_Foundations.md))
2. Runs NUTS sampler for 1000 burn-in + 1000 posterior samples
3. Computes organ-specific Hazard Index (HI), Cancer Risk (CR), and BLL
4. Saves full trace to `trace.nc` (NetCDF format)

**Expected Runtime**: 5-15 minutes depending on system and number of metals/sites.

**Code reference**: Lines 714-741

---

### Step 7: Generate Diagnostic Plots

**Purpose**: Assess MCMC convergence and model fit quality.

**Plots Generated** (`plot_diagnostics.py`):

- **Trace plots**: Time series of sampled parameters
- **Autocorrelation plots**: Check for chain mixing
- **Prior vs Posterior**: Compare learned vs assumed distributions
- **Forest plots**: Parameter uncertainty intervals

**Success Criteria**:

- Effective Sample Size (ESS) > 400
- $\hat{R}$ (Gelman-Rubin) < 1.01
- No obvious trends in trace plots

**Custom Mode Prompt**: You can now choose diagnostics independently from result plots.

**Code reference**: Lines 182-229, 743-745

---

### Step 8: Generate Result Figures

**Purpose**: Create publication-ready visualizations.

**Figures Generated** (`plot_result.py`):

- HI distributions by organ system
- CR exceedance curves
- BLL threshold exceedance (3.5, 5, 10 µg/dL)
- Site-specific risk profiles

**Output Format**: High-resolution PNG (300 DPI)

**Custom Mode Prompt**: Separate prompt lets you skip results even if you generate diagnostics (or vice versa).

**Code reference**: Lines 232-278, 747-749

---

### Step 9: Generate Summary Tables

**Purpose**: Export results to CSV for further analysis.

**Tables Generated** (`summary_tables.py`):

- **T1**: Measured concentrations with WHO comparison
- **T2**: Speciation fractions
- **T3**: Posterior summary (quantiles, exceedance probabilities)
- **T4**: Risk ranking by site (multivariate: HI, CR, BLL)
- **T5**: BLL summary statistics
- **Correlation matrix**, **PCA results**

**Code reference**: Lines 281-335, 751-753

---

### Step 10: Sensitivity Analysis (Optional, Advanced)

**Purpose**: Identify which parameters most influence model outputs.

**Interactive Prompts**:

```
Run sensitivity analysis? [y/N]: y

Select sensitivity method:
  1. sobol  - Sobol indices (most thorough, slowest)
  2. morris - Morris method (faster screening)
  3. delta  - Delta moment (distribution-based)
Method [1/2/3, default=1]: 1

Number of samples [default=512]: 1024
```

**Decision Logic**:

- **Sobol**: For comprehensive variance decomposition (10-30 min)
- **Morris**: For quick parameter screening (2-5 min)
- **Delta**: For non-normal outputs (5-15 min)

**What You Get**:

- Parameter importance rankings
- First-order and total-effect indices
- Tornado diagrams
- Convergence plots

**Code reference**: Lines 338-404, 755-784

---

### Step 11: Entropy HPI/PERI Analysis (Optional, Advanced)

**Purpose**: Calculate pollution indices using information theory.

**Interactive Prompts**:

```
Run entropy HPI/PERI analysis? [y/N]: y

Found standards: waterdata/standards.csv
Found toxicities: waterdata/toxicity.csv

Bootstrap samples [default=1000]: 1000
```

**File Requirements**:

- `standards.csv`: Regulatory limits for each metal
- `toxicity.csv`: Toxic response factors (Hakanson, 1980)

**What You Get**:

- Entropy weights (data-driven importance)
- HPI (Heavy Metal Pollution Index) with risk categories
- PERI (Potential Ecological Risk Index) with 95% CI
- Metal contribution heatmaps

**Code reference**: Lines 407-502, 786-825

---

## 3. User Interaction Patterns

### 3.1 Input Validation

All user inputs are validated with fallbacks:

```python
# Example: Numeric input with default
samples_input = input("Number of samples [default=512]: ").strip()
try:
    n_samples = int(samples_input) if samples_input else 512
except ValueError:
    n_samples = 512  # Fallback to default
```

### 3.2 Yes/No Prompts

```python
# Case-insensitive, default=YES
choice = input("Run speciation modeling? [Y/n]: ").strip().lower()
run_speciation = choice != 'n'

# Default=NO
run_sens = input("Run sensitivity analysis? [y/N]: ").strip().lower()
run_sensitivity = run_sens == 'y'
```

### 3.3 File Path Resolution

```python
# Search hierarchy: waterdata/ → current dir → user input
for base_dir in [WATERDATA_DIR, PROJECT_ROOT]:
    std_path = os.path.join(base_dir, "standards.csv")
    if os.path.exists(std_path):
        standards_file = std_path
        break
else:
    standards_file = input("  Path to standards.csv: ").strip()
```

---

## 4. Decision Trees and Logic Flow

### 4.1 Main Workflow Flowchart

```
START
  ↓
Dependencies OK?
  ├─ NO → EXIT with install instructions
  ↓
Find input file
  ├─ 0 found → Manual path entry
  ├─ 1 found → Confirm use
  ├─ N found → Select from list
  ↓
Validate file
  ├─ Invalid → EXIT with error
  ↓
Select units (µg/L or mg/L)
  ↓
Set output directory
  ↓
PHREEQC available?
  ├─ YES → Prompt for speciation
  ├─ NO  → Use simplified
  ↓
Run BLL calibration (auto-engine)
  ↓
Run Bayesian model
  ├─ Success → Continue
  ├─ Fail    → EXIT with diagnostics
  ↓
Generate diagnostics
  ↓
Generate figures
  ↓
Generate tables
  ↓
Advanced analyses?
  ├─ Sensitivity? [y/N]
  ├─ Entropy?     [y/N]
  ↓
Display summary
  ↓
END
```

### 4.2 Error Recovery Strategy

```python
# Example: Continue with warning if sub-module fails
result = subprocess.run(speciation_cmd, ...)
if result.returncode != 0:
    print("[WARN] Speciation had issues:")
    print(result.stderr[-500:])
    print("Continuing with simplified speciation...")
    # Does NOT exit; continues workflow
```

---

## 5. Command-Line Options

### 5.1 Quick Mode Arguments

```bash
python run_hbmpra.py \
    --input waterdata/data1.csv \
    --output-dir results_custom \
    --run-speciation \
    --run-sensitivity sobol \
    --sensitivity-samples 2048 \
    --run-entropy \
    --bootstrap 5000
```

### 5.2 Available Flags

*(Extend `argparse` in lines 950-1048 for more options)*

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--input` | str | None | Path to chemistry CSV |
| `--output-dir` | str | Auto-generated | Results directory |
| `--run-speciation` | bool | Prompt | Force speciation |
| `--skip-speciation` | bool | Prompt | Skip speciation |
| `--run-sensitivity` | str | Prompt | Method: sobol/morris/delta |
| `--sensitivity-samples` | int | 512 | Sample size |
| `--run-entropy` | bool | Prompt | Force entropy analysis |
| `--bootstrap` | int | 1000 | Bootstrap samples |

---

## 6. Error Handling and Validation

### 6.1 Pre-Flight Checks

```python
# File validation (lines 545-585)
def validate_chemistry_file(filepath):
    df = pd.read_csv(filepath)
    
    # Check for metal columns
    metals_found = [...]
    if not metals_found:
        return False, "No metal columns found. Expected: As, Cd, Pb, ..."
    
    # Check for site ID
    if df.columns[0] not in ['site', 'Site', 'ID', ...]:
        warn("First column may not be a site identifier")
    
    return True, {...metadata...}
```

### 6.2 Subprocess Error Handling

```python
# Capture stderr but don't crash (lines 139-146)
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"[WARN] {module} encountered issues:")
    print(result.stderr[-300:])  # Show last 300 chars only
    print("Continuing with fallback...")
    return False  # Signal failure but don't raise
```

### 6.3 User-Friendly Error Messages

```python
# Example from BLL calibration (line 142)
if result.returncode != 0:
    print("  [WARN] BLL calibration had issues:")
    print("  Continuing with default uncalibrated priors...")
    # Explains what went wrong AND what happens next
```

---

## 7. Customization and Extension

### 7.1 Adding New Analysis Steps

To add a new step (e.g., Step 12: Monte Carlo Filtering):

1. **Define the function**:

```python
def run_monte_carlo_filtering(input_file, output_dir, threshold=0.5):
    import subprocess
    cmd = [sys.executable, "monte_carlo_filter.py",
           "--input", input_file,
           "--output-dir", os.path.join(output_dir, "mc_filter"),
           "--threshold", str(threshold)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0
```

2. **Add to interactive workflow** (after line 825):

```python
# Step 12: Monte Carlo Filtering (Optional)
print_step(12, TOTAL_STEPS, "Monte Carlo Filtering (optional)")
print("Filter parameter space based on output thresholds.")
run_mc = input("Run Monte Carlo filtering? [y/N]: ").strip().lower()
if run_mc == 'y':
    threshold_input = input("Threshold value [default=0.5]: ").strip()
    threshold = float(threshold_input) if threshold_input else 0.5
    run_monte_carlo_filtering(filepath, output_dir, threshold)
```

3. **Update `TOTAL_STEPS`** (line 592):

```python
TOTAL_STEPS = 12  # Changed from 11
```

### 7.2 Customizing Defaults

Modify lines 726-730 to change MCMC settings:

```python
hbmpra_cmd = [
    ...,
    "--draws", "2000",  # Increase for publication quality
    "--tune", "2000",
    "--chains", "4",    # Add parallel chains
]
```

### 7.3 Adding Workflow Presets

```python
# Add at top of interactive_mode()
print("Select workflow preset:")
print("  1. Quick (1000 draws, no advanced)")
print("  2. Standard (2000 draws, diagnostics)")
print("  3. Publication (4000 draws, all analyses)")
preset = input("Preset [1/2/3, default=2]: ").strip()

if preset == '1':
    draws, run_sens_default, run_entropy_default = 1000, False, False
elif preset == '3':
    draws, run_sens_default, run_entropy_default = 4000, True, True
else:
    draws, run_sens_default, run_entropy_default = 2000, False, False
```

---

## 8. Best Practices for Users

### 8.1 Recommended Workflow

1. **First Run**: Use interactive mode to understand each step
2. **Subsequent Runs**: Use quick mode with validated settings
3. **Publication**: Run with 2000+ draws and all diagnostics

### 8.2 When to Skip Steps

- **Skip Speciation** if:
  - No pH/Eh data available
  - Only interested in screening-level assessment
  - PHREEQC installation issues

- **Skip Sensitivity** if:
  - Only need point estimates
  - Time-constrained
  - Small parameter space (<10 parameters)

- **Skip Entropy** if:
  - No regulatory standards available
  - Only human health (not ecological) assessment needed

### 8.3 Troubleshooting Tips

| Issue | Solution |
|-------|----------|
| "No metal columns found" | Check CSV headers: use `C_As`, `As`, etc. |
| "PHREEQC failed" | Check pH/Eh ranges (pH 4-10, Eh -200 to 800 mV) |
| "MCMC did not converge" | Increase `--tune` or `--draws` |
| "Out of memory" | Reduce number of sites or metals |

---

## 9. Technical Implementation Details

### 9.1 Subprocess Management

All analysis modules are called via `subprocess.run()` to:

- Isolate errors (module failure doesn't crash orchestrator)
- Capture stdout/stderr for logging
- Allow parallel execution (future enhancement)

### 9.2 Progress Feedback

```python
# Step indicators (line 56-59)
print(f"\n[Step {step_num}/{total_steps}] {message}")
print("-" * 50)

# Spinners for long operations (future)
# Consider adding tqdm progress bars
```

### 9.3 Configuration Persistence

All analysis settings are saved to `RUNLOG.json`:

```json
{
  "input_file": "waterdata/data1.csv",
  "concentration_units": "µg/L",
  "speciation": true,
  "mcmc_draws": 1000,
  "sensitivity_method": "sobol",
  "timestamp": "2025-01-22T11:45:19Z"
}
```

---

## 10. Future Enhancements

- [ ] **Resume capability**: Save workflow state to allow resuming from failures
- [ ] **Parallel execution**: Run diagnostics and figures simultaneously
- [ ] **Web interface**: Flask/Streamlit GUI for browser-based interaction
- [ ] **Configuration files**: YAML-based presets for batch processing
- [ ] **Docker integration**: Containerized workflow for reproducibility

---

## 👥 Authors

- **Dickson Abdul-Wahab** — University of Ghana, Ghana
  Email: [dabdul-wahab@live.com](mailto:dabdul-wahab@live.com)
  ORCID: [0000-0001-7446-5909](https://orcid.org/0000-0001-7446-5909)
  LinkedIn: [linkedin.com/in/dickson-abdul-wahab-0764a1a9](https://www.linkedin.com/in/dickson-abdul-wahab-0764a1a9)
  ResearchGate: [researchgate.net/profile/Dickson-Abdul-Wahab](https://www.researchgate.net/profile/Dickson-Abdul-Wahab)

- **Ebenezer Aquisman Asare** — Organic Laboratory Research, Atomic Energy Commission (GAEC), Nuclear Chemistry and Environmental Research Centre, National Nuclear Research Institute (NNRI), Legon-Accra, Ghana
  Email: [aquisman1989@gmail.com](mailto:aquisman1989@gmail.com)
  ORCID: [0000-0003-1185-1479](https://orcid.org/0000-0003-1185-1479)
  ResearchGate: [researchgate.net/profile/Ebenezer-Aquisman-Asare](https://www.researchgate.net/profile/Ebenezer-Aquisman-Asare)

---

*Last Updated: December 2025*
