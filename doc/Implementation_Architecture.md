# HBMPRA Implementation Architecture

This document maps the mathematical foundations to the software implementation within the `src/` directory.

---

## 1. Core Models & Risk Assessment

### `hbmpra_optimized.py`

This is the "engine" of the framework. It constructs and samples the PyMC Bayesian model.

* **Hierarchical Priors**: Implemented using `pm.LogNormal` and `pm.Normal` for parameters like `BW_g` and `IR_perkg_g`.
* **Vectorized EDI Calculation**: Utilizes `at.tensordot` for high-performance dose calculation across (sites × metals × groups).
* **Censoring Logic**: Contains `impute_censored_df()` which uses MLE to handle non-detects before model execution.
* **Organ-Route Mapping**: Uses `build_organ_sets()` to resolve toxicity preferences from `toxref.yml`.

### `bll_engines.py`

Contains the pharmacokinetic models for lead.

* **`onecomp`**: A Mechanistic Steady-State ODE solution used for adults.
* **`slope`**: An Empirical linear dose-response model used for children and vulnerable groups.
* **`auto`**: Logic for selecting the appropriate engine based on population demographics.

---

## 2. Speciation & Chemistry

### `speciation_modeling.py`

Integrates thermodynamic equilibrium calculations.

* **PHREEQC Interface**: Supports `phreeqpython` and `phreeqpy` backends.
* **Chromium/Mercury Logic**: Special handling for elements where total concentration must be split into multiple toxic species (e.g., CrVI vs CrIII).
* **Fallbacks**: Implements a "Simplified Speciation" mode to estimate bioavailable fractions if PHREEQC is unavailable.

### `units.py`

Standardizes all conversions.

* `CF_ugL_to_mgL`: Used to ensure concentration units match the $mg/kg\text{-day}$ requirements of toxicity reference values (RfDs).

---

## 3. Analysis & Visualization

### `sensitivity_analysis.py`

Implements global and local sensitivity methods.

* **Sobol Method**: Uses `saltelli.sample` and `sobol.analyze` from the `SALib` project to calculate $S_1$ and $S_T$ variance indices.
* **Morris Method**: Performs screening for high-dimensional parameter spaces.
* **Parameter Mapping**: Automatically maps toxicity, demographic, and chemical parameters to a standard $0$ to $1$ sampling space.

### `entropy_hpi_peri.py`

Calculates environmental pollution indices.

* **Weight Calculation**: Implements the entropy-based divergence equations.
* **Bootstrap**: Uses `bootstrap_uncertainty()` to calculate 95% confidence intervals for HPI and PERI indices.

---

## 4. User Interaction

### `run_hbmpra.py`

* **Interactive Orchestrator**: Wraps all sub-modules into an 11-step guided workflow.
* **Pre-flight Checks**: Validates input CSV headers and installs missing dependencies.
* **Plot Toggles (Custom Mode)**: Separate prompts for diagnostic plots vs result plots; presets retain defaults.
* **Anion-Only Path**: Skips speciation and BLL calibration if only F/NO₃ are present; still computes anion HQ/HI.

### `summary_tables.py` & `plot_result.py`

* **Post-processing**: Aggregates the Bayesian NetCDF (`.nc`) trace into publication-ready CSV tables and high-resolution PNG figures.
