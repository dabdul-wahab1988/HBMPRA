# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

HBMPRA (Hierarchical Bayesian Model for Probabilistic Risk Assessment) is a probabilistic risk assessment framework for trace metals and anions in drinking water. It combines thermodynamic speciation modeling (PHREEQC), Bayesian hierarchical modeling (PyMC), blood lead level (BLL) prediction, and sensitivity analysis.

**Key capabilities:**
- Multi-organ hazard index calculation (neuro, nephro, hepato, skeletal, hematological)
- Cancer risk estimation for carcinogenic metals
- Blood lead level prediction with CDC-aligned thresholds (3.5, 5, 10 µg/dL)
- Support for 13 heavy metals + 2 anions (Fluoride, Nitrate)
- Full uncertainty quantification via Bayesian inference

## Running the Code

### Main Entry Point

```bash
# Interactive mode (recommended for users)
cd src
python run_hbmpra.py

# Quick analysis with defaults
python run_hbmpra.py --input ../waterdata/data1.csv

# Full control
python run_hbmpra.py --input <file> --output <dir> --draws 2000 --tune 2000
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_bll_engines.py -v
pytest tests/test_phreeqc_bio_integration.py -v
pytest tests/test_dermal_hq.py -v
```

### Step-by-Step Workflow (Manual)

```bash
# 1. Speciation modeling (PHREEQC)
cd src
python speciation_modeling.py --input ../waterdata/data1.csv \
    --output-dir ../results --use-total-fallback

# 2. Risk assessment (Bayesian model)
python hbmpra_optimized.py --chemistry ../waterdata/data1.csv \
    --results-dir ../results --use-bioavailable \
    --allow-default-organ-sets

# 3. Generate summary tables
python summary_tables.py --results-dir ../results

# 4. Generate plots
python plot_result.py --results-dir ../results

# 5. (Optional) Sensitivity analysis
python sensitivity_analysis.py --results-dir ../results

# 6. (Optional) Entropy-based pollution indices
python entropy_hpi_peri.py --input ../waterdata/data1.csv \
    --output-dir ../results
```

## Architecture Overview

### Core Processing Pipeline

```
Input CSV → Speciation → Risk Assessment → Visualization → Tables
             (PHREEQC)   (PyMC Bayesian)   (Matplotlib)   (Pandas)
```

**Key architectural patterns:**

1. **Separation of Concerns**: Each module handles one scientific domain
   - `speciation_modeling.py`: Chemical equilibrium calculations
   - `hbmpra_optimized.py`: Bayesian risk model (hazard indices, cancer risk)
   - `bll_engines.py`: Blood lead level prediction (PK models + empirical slopes)
   - `sensitivity_analysis.py`: Global sensitivity methods (Sobol, Morris, Delta)
   - `entropy_hpi_peri.py`: Pollution indices (HPI, PERI)

2. **Vectorization Strategy**: Avoid nested Python loops
   - Dermal HQ calculation is vectorized across (metal × site × group)
   - PyMC models use tensor operations for batch processing
   - See `hbmpra_optimized.py:756-799` for dermal vectorization

3. **Configuration-Driven Toxicity**: All toxicity values in `external/toxref.yml`
   - RfD (Reference Dose) for non-cancer endpoints
   - SF (Slope Factor) for cancer risk
   - Organ-specific routing (`organ_sets`)
   - DO NOT hardcode toxicity values in Python code

4. **Population-Specific Engine Selection**:
   - Adults → One-compartment PK model (`bll_engines.py:onecomp`)
   - Children/Teens/Pregnant → Empirical slope model (`bll_engines.py:slope`)
   - Rationale: Vulnerable populations have higher uncertainty; empirical model is more conservative

### File Structure

```
src/
├── run_hbmpra.py              # Main interactive entry point
├── hbmpra_optimized.py        # Core Bayesian model (PRIMARY MODEL FILE)
├── bll_engines.py             # Blood lead level calculations
├── speciation_modeling.py     # PHREEQC integration
├── sensitivity_analysis.py    # Sobol/Morris/Delta methods
├── entropy_hpi_peri.py        # HPI/PERI pollution indices
├── demographics.py            # Population group parameters
├── units.py                   # Unit conversions (incl. nitrate basis)
├── plot_result.py             # Visualization
├── plot_diagnostics.py        # MCMC diagnostics
├── summary_tables.py          # Table generation (T1-T5)
└── calibrate_bll_priors.py    # BLL prior calibration

external/
├── toxref.yml                 # Toxicity reference values (CRITICAL)
├── dermal_water_kp.yml        # Dermal permeability coefficients
├── standards.csv              # Regulatory standards
└── toxicity.csv               # Legacy toxicity table

waterdata/                     # Input data location
database/                      # PHREEQC thermodynamic databases
doc/                          # Mathematical documentation
tests/                        # pytest test suite
```

## Critical Implementation Details

### 1. Anion Support (Fluoride, Nitrate)

**Fluoride:**
- Column names: `F`, `F-`, `Fluoride`
- Units: mg/L as F⁻
- Target organ: `skeletal_dental`
- RfD: 0.06 mg/kg-day (EPA IRIS)

**Nitrate:**
- Column names: `NO3`, `NO3-`, `Nitrate`, `NO3_N`
- Units: mg/L as NO₃ (or NO₃–N basis)
- Target organ: `hematological`
- RfD: 1.6 mg/kg-day (as NO₃–N basis per ATSDR)
- Basis conversion: `units.py:convert_nitrate_basis_mgL()`

**Key functions:**
- `hbmpra_optimized.py:detect_anions_in_dataframe()` - Column detection
- `hbmpra_optimized.py:_get_anion_canonical_name()` - Name resolution
- `units.py:detect_nitrate_basis_from_column()` - Basis detection

### 2. Metal Speciation

PHREEQC calculates bioavailable fractions using thermodynamic equilibrium. Critical species:

| Metal | Species Modeled | File Reference |
|-------|-----------------|----------------|
| As | H₂AsO₄⁻ (arsenate) | `speciation_modeling.py:SPECIES` |
| Cd | Cd²⁺ | `speciation_modeling.py:SPECIES` |
| CrVI/CrIII | CrO₄²⁻ / Cr³⁺ | Multi-species handling |
| Pb | Pb²⁺ | `speciation_modeling.py:SPECIES` |
| Hg | Hg²⁺ / Hg⁰ | Multi-species handling |

**Output files:**
- `table_bioavailable_concentrations.csv` - Bioavailable fractions
- `table_species_fractions.csv` - Species distribution

### 3. Censored Data Handling

Left-censored data (below detection limit) uses MLE + truncated lognormal expectation:

```python
# See hbmpra_optimized.py:323-383
def impute_censored_expectation_vectorized(concentrations, censored_mask):
    # MLE fit to uncensored data
    # Conditional expectation E[X | X < LOD]
    # Returns imputed values
```

**Important:** Always use `--use-total-fallback` flag with speciation modeling when data has censored values.

### 4. PyMC Model Structure

The Bayesian model (`hbmpra_optimized.py`) constructs a single unified model with:

- **Priors**: Non-centered parameterization for better MCMC sampling
  - Body weight: Lognormal
  - Water intake: Lognormal
  - Exposure duration: Beta
  - Exposure frequency: Beta

- **Likelihood**: Deterministic calculation of HI, CR, BLL

- **Outputs**:
  - `HI_overall` - Maximum across all organs
  - `HI_<organ>` - Organ-specific hazard indices
  - `CR_total` - Incremental lifetime cancer risk
  - `BLL` - Blood lead level (µg/dL)

**Vectorization example (dermal route):**
```python
# See hbmpra_optimized.py:756-799
# Builds (n_metals, n_sites, n_groups) tensors
# Single tensor operation instead of triple nested loop
```

### 5. Blood Lead Level Engines

**One-compartment PK model (adults):**
- Steady-state solution: `BLL = (f_abs × intake × t_half) / (ln(2) × V_blood × BW)`
- See `bll_engines.py:119-133`

**Empirical slope model (vulnerable groups):**
- Linear dose-response: `BLL = slope × (f_abs × intake × BW) + background`
- Higher slopes for children (0.17) vs adults (0.08)
- See `bll_engines.py:135-155`

**Auto-selection:**
```python
# bll_engines.py:auto_select_engine()
if group in ["adults", "adult"]:
    return "onecomp"
else:  # children, teens, pregnant
    return "slope"
```

### 6. Sensitivity Analysis Methods

Three global sensitivity methods available (`sensitivity_analysis.py`):

1. **Sobol variance decomposition** - ANOVA-based first-order and total-effect indices
2. **Morris screening** - Elementary effects for parameter ranking
3. **Delta moment-independent** - Non-parametric sensitivity measure

**Usage:**
```python
# Requires completed trace.nc file
python sensitivity_analysis.py --results-dir <dir> \
    --method sobol --n-samples 1000
```

## Common Development Tasks

### Adding a New Metal

1. **Add atomic weight** to `speciation_modeling.py:ATOMIC_WEIGHTS`
2. **Add PHREEQC species** to `speciation_modeling.py:SPECIES`
3. **Add toxicity values** to `external/toxref.yml`:
   ```yaml
   NewMetal:
     rfd_oral: 0.001  # mg/kg-day
     rfd_dermal: 0.001
     sf_oral: null  # or value if carcinogenic
     sf_dermal: null
     organ_sets:
       - hepato
       - nephro
   ```
4. **Add dermal Kp** to `external/dermal_water_kp.yml`
5. **Add to carcinogens list** (if applicable): `hbmpra_optimized.py:CARCINOGENS`

### Modifying MCMC Sampling

Default: 2000 draws, 2000 tuning steps

```python
# hbmpra_optimized.py:run_hbmpra_model()
trace = pm.sample(
    draws=draws,
    tune=tune,
    chains=4,
    return_inferencedata=True,
    random_seed=random_seed
)
```

**MCMC diagnostics:**
- R-hat < 1.01 (convergence)
- ESS > 400 (effective sample size)
- See `plot_diagnostics.py` for trace plots and posterior checks

### Input Data Format

Required columns:
- First column: Site identifier (any name)
- Metal columns: `As`, `Cd`, `Cr`, `Cu`, `Hg`, `Pb`, `Mn`, `Fe`, `Zn`, `Ni`, `Co`, `Al`, `V`
  - Alternative: `C_As`, `C_Cd`, etc.
- Optional: `pH`, `Eh` (for speciation modeling)
- Anions: `F`, `NO3`, `NO3_N`

**Units:**
- Metals: µg/L (default) or mg/L with `--units mg/L` flag
- Anions: mg/L
- pH: standard scale
- Eh: mV

## Important Constraints and Assumptions

1. **No inhalation pathway**: Only ingestion + dermal routes modeled
2. **Groundwater focus**: Dermal exposure assumes bathing/showering with contaminated water
3. **Methylmercury NOT modeled**: PHREEQC cannot predict bacterial methylation; only inorganic Hg
4. **Organ sets from toxref.yml**: Use `--allow-default-organ-sets` flag only for testing
5. **PHREEQC optional**: Model runs with simplified estimates if phreeqpython unavailable
6. **Windows compatibility**: File paths use `os.path.join()` for cross-platform support

## Output Files

### Main Results Directory

```
results_<dataset>_<timestamp>/
├── trace.nc                   # ArviZ InferenceData (NetCDF)
├── RUNLOG.json               # Execution metadata
├── ASSUMPTIONS.json          # Model parameters
├── table_bioavailable_concentrations.csv
├── table_species_fractions.csv
├── figure_speciation_profiles.png
├── debug/
│   ├── HI_summary.csv        # ← PRIMARY RESULTS TABLE
│   └── HI_summary.json
├── tables/
│   ├── T1_measured_summary.csv
│   ├── T3_posterior_summary.csv
│   ├── T4_risk_ranking.csv
│   └── T5_BLL_summary.csv
├── figures/
│   └── (publication-ready plots)
└── diagnostics/
    └── (MCMC convergence plots)
```

### Key Output Variables in trace.nc

| Variable | Description | Units |
|----------|-------------|-------|
| `HI_overall` | Maximum HI across all organs | dimensionless |
| `HI_neuro`, `HI_nephro`, etc. | Organ-specific HI | dimensionless |
| `HI_skeletal_dental` | Fluoride hazard index | dimensionless |
| `HI_hematological` | Nitrate hazard index | dimensionless |
| `CR_total` | Incremental lifetime cancer risk | probability |
| `BLL` | Blood lead level | µg/dL |
| `HQ_<metal>_<route>` | Individual hazard quotients | dimensionless |

**Interpretation:**
- HI > 1.0 → Potential health concern
- CR > 1e-6 → Regulatory screening threshold
- BLL > 3.5 µg/dL → CDC reference value exceedance (children/pregnant)

## Mathematical Documentation

Comprehensive derivations in `doc/`:
- **HBMPRA_Mathematical_Foundations.md** - Full Bayesian model derivation
- **BLL_Pharma_Foundations.md** - Blood lead PK models
- **Speciation_and_Metal_Chemistry.md** - PHREEQC integration
- **Sensitivity_Analysis_Foundations.md** - Sobol/Morris/Delta methods
- **Entropy_Pollution_Indices.md** - HPI/PERI theory

All equations traceable to peer-reviewed literature (EPA RAGS, IRIS, WHO Guidelines).

## Dependencies

**Required:**
- numpy
- pandas
- matplotlib
- pyyaml

**Optional (for full functionality):**
- pymc (Bayesian modeling)
- arviz (posterior analysis)
- phreeqpython (speciation modeling)
- scipy (optimization, statistics)

**Installation:**
```bash
pip install numpy pandas matplotlib pyyaml pymc arviz phreeqpython scipy
```

## Testing Philosophy

Tests validate:
1. **Unit calculations** - Individual functions (e.g., `edi_from_conc_ugL`)
2. **Integration** - PHREEQC → risk model pipeline
3. **Numeric stability** - Edge cases (zero concentrations, censored data)
4. **Calibration** - BLL priors match empirical data

**Critical tests:**
- `test_bll_engines.py` - BLL calculation correctness
- `test_phreeqc_bio_integration.py` - Speciation → risk workflow
- `test_dermal_hq.py` - Dermal route calculations
- `test_nitrate_units.py` - Nitrate basis conversions

## Performance Considerations

**Vectorization wins:**
- Dermal HQ: ~10x faster with NumPy broadcasting vs nested loops
- PyMC tensor operations: Essential for MCMC efficiency

**MCMC tuning:**
- Default 2000 draws usually sufficient
- Use 4000 draws for publication-quality results
- Increase `tune` if R-hat > 1.01

**Large datasets:**
- PHREEQC is bottleneck (O(n_sites × n_species))
- Consider `--skip-speciation` flag for screening runs
- Sensitivity analysis is computationally expensive (O(n_samples × n_params))

## Common Pitfalls

1. **Missing organ_sets in toxref.yml** → Use `--allow-default-organ-sets` temporarily
2. **Nitrate basis confusion** → Always specify column as `NO3` or `NO3_N`
3. **PHREEQC database errors** → Check `database/` directory exists
4. **Convergence failures** → Increase `tune` or check for extreme concentration values
5. **Path issues on Windows** → Always use `os.path.join()`, never hardcoded `'/'` or `'\'`

## Authors

- Dickson Abdul-Wahab (University of Ghana)
- Ebenezer Aquisman Asare

For questions, open an issue on the project repository.

---

*Last Updated: January 2, 2026*
