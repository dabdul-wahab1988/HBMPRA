# HBMPRA User Guide - For Non-Coders

## Quick Start (Easiest Way)

Simply double-click or run:

```
python run_hbmpra.py
```

This will guide you through everything step by step!

---

## What is HBMPRA?

HBMPRA (Hierarchical Bayesian Model for Probabilistic Risk Assessment) analyzes trace metal concentrations in water to assess health risks. It calculates:

1. **Hazard Index (HI)** - Risk of non-cancer health effects by organ system
2. **Cancer Risk (CR)** - Lifetime cancer probability from carcinogenic metals
3. **Blood Lead Level (BLL)** - Predicted lead in blood (CDC thresholds: 3.5, 5, 10 µg/dL)
4. **Anion Risk** - Fluoride (dental fluorosis) and Nitrate (methemoglobinemia) analysis

---

## Required Data Format

Your input CSV file should look like this:

| Site | pH | Eh | As | Cd | Cr | Cu | Hg | Pb | ... |
|------|----|----|----|----|----|----|----|----|-----|
| Site1 | 7.2 | 320 | 5.2 | 0.8 | 12 | 45 | 0.05 | 8.3 | ... |
| Site2 | 6.8 | 280 | 3.1 | 0.5 | 8 | 32 | 0.03 | 5.1 | ... |

### Required Columns

- **First column**: Site identifier (name or ID)
- **Metal columns**: Concentrations in µg/L
  - Can be named: `As`, `Cd`, `Cr`, `Cu`, `Hg`, `Pb`, `Mn`, `Fe`, `Zn`, `Ni`, `Co`, `Al`, `V`
  - Or: `C_As`, `C_Cd`, etc.
- **Anion columns** (optional): Concentrations in mg/L
  - Fluoride: `F`, `F-`, `Fluoride`
  - Nitrate: `NO3`, `NO3-`, `Nitrate`, `NO3_N` (for NO₃–N basis)

### Optional but Recommended

- **pH**: Water pH (default: 7.0 if missing)
- **Eh**: Redox potential in mV (default: 300 mV if missing)

---

## Understanding the Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                      YOUR INPUT DATA                            │
│                    (chemistry CSV file)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1: SPECIATION MODELING                        │
│              (speciation_modeling.py)                           │
│                                                                 │
│  What it does:                                                  │
│  • Calculates which chemical FORMS each metal exists in        │
│  • Determines BIOAVAILABLE fractions (what body absorbs)       │
│  • Uses PHREEQC thermodynamic database                         │
│                                                                 │
│  Example: Total Arsenic → H₂AsO₄⁻ (arsenate, most toxic)       │
│                                                                 │
│  Output files:                                                  │
│  • table_bioavailable_concentrations.csv                       │
│  • table_species_fractions.csv                                 │
│  • figure_speciation_profiles.png                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 2: RISK ASSESSMENT                            │
│              (hbmpra_optimized.py)                              │
│                                                                 │
│  What it does:                                                  │
│  • Calculates exposure doses for different age groups          │
│  • Computes Hazard Index by organ system                       │
│  • Estimates cancer risk from carcinogens                      │
│  • Predicts blood lead levels                                  │
│  • Uses Bayesian statistics for uncertainty                    │
│                                                                 │
│  Output files:                                                  │
│  • trace.nc (full model results)                               │
│  • RUNLOG.json (what was analyzed)                             │
│  • ASSUMPTIONS.json (model parameters)                         │
│  • debug/HI_summary.csv (hazard indices)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 3: VISUALIZATION (Optional)                   │
│              (plot_result.py, summary_tables.py)                │
│                                                                 │
│  • Generate publication-ready figures                          │
│  • Create summary tables for reports                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Three Ways to Run the Analysis

### Option 1: Interactive Mode (Recommended for Beginners)

```bash
python run_hbmpra.py
```

Just follow the prompts! The script will:

- Find your data file automatically
- Ask what you want to do
- Explain each step
- Tell you what files were created

### Option 2: Quick Command

```bash
python run_hbmpra.py --input data1.csv
```

Runs everything with sensible defaults.

### Option 3: Full Control (Advanced)

```bash
# Step 1: Run speciation
python speciation_modeling.py --input data1.csv --output-dir my_results --use-total-fallback

# Step 2: Run risk assessment  
python hbmpra_optimized.py --chemistry data1.csv --results-dir my_results --use-bioavailable --allow-default-organ-sets
```

---

## Understanding Output Files

### Main Results Directory Structure

```
my_results/
├── trace.nc                              # Full Bayesian model (technical)
├── RUNLOG.json                           # What metals/organs were analyzed
├── ASSUMPTIONS.json                      # Model parameters used
├── table_bioavailable_concentrations.csv # Bioavailable metal species
├── table_species_fractions.csv           # Fraction of each species
├── figure_speciation_profiles.png        # Speciation visualization
├── debug/
│   ├── HI_summary.csv                    # ← MOST USEFUL: Hazard indices
│   └── HI_summary.json                   # Same data in JSON format
├── tables/                               # Summary Spreadsheets
│   ├── T1_measured_summary.csv           # Input data stats
│   ├── T3_posterior_summary.csv          # Risk probabilities
│   ├── T4_risk_ranking.csv               # Top sites by HI/CR/BLL
│   └── T5_BLL_summary.csv                # Blood lead exceedances
```

### Key File: `debug/HI_summary.csv`

| organ | site | group | median | p3 | p97 |
|-------|------|-------|--------|-----|-----|
| neuro | 0 | Infants | 0.045 | 0.032 | 0.061 |
| neuro | 0 | Children | 0.038 | 0.027 | 0.052 |
| nephro | 0 | Infants | 0.12 | 0.089 | 0.16 |

**Interpreting HI values:**

- **HI < 1.0**: Acceptable risk level
- **HI > 1.0**: Potential health concern
- **p3/p97**: 94% credible interval (uncertainty range)

---

## Metal Species Explained

The speciation model determines the most toxic/bioavailable form of each metal:

| Metal | Species Modeled | Why This Form? |
|-------|-----------------|----------------|
| As | H₂AsO₄⁻ (arsenate) | Most toxic in oxic water |
| Cd | Cd²⁺ | Most bioavailable |
| Cr | CrO₄²⁻ (CrVI) + Cr³⁺ (CrIII) | CrVI is carcinogenic |
| Cu | Cu²⁺ | Most toxic aqueous form |
| Hg | Hg²⁺ + Hg⁰ | Inorganic forms only* |
| Pb | Pb²⁺ | Most bioavailable |
| Mn | Mn²⁺ | Neurotoxic form |
| Fe | Fe²⁺ | Bioavailable form |
| Zn | Zn²⁺ | Most bioavailable |
| Ni | Ni²⁺ | Toxic aqueous form |
| Co | Co²⁺ | Bioavailable form |
| Al | Al³⁺ | Neurotoxic form |
| V | VO₂⁺ | Equilibrates to vanadate |

*Note: Methylmercury (MeHg) is NOT modeled because it's formed by bacteria, not thermodynamic equilibrium. MeHg must be measured directly.

---

## Blood Lead Analysis (BLL)

HBMPRA uses different thresholds for different groups based on CDC guidelines:

- **Children & Pregnant**: 3.5 µg/dL (CDC 2021 Reference Value)
- **Teens & Others**: 5.0 µg/dL
- **Adults**: 10.0 µg/dL (Occupational action level)

The **T5_BLL_summary.csv** table will show you the probability that each group exceeds these levels: `P(BLL > 3.5)`.

---

## Fluoride and Nitrate Analysis

HBMPRA now supports **Fluoride (F⁻)** and **Nitrate (NO₃⁻)** analysis alongside heavy metals.

### Input Format

| Site | As | Pb | F | NO3 | NO3_N |
|------|----|----|---|-----|-------|
| Site1 | 5.2 | 8.3 | 1.2 | 45 | - |
| Site2 | 3.1 | 5.1 | 0.8 | - | 8.5 |

**Concentration Units:**

- Fluoride: mg/L as F⁻
- Nitrate: mg/L as NO₃ (or NO₃–N if using `NO3_N` column)

### WHO Guidelines

| Analyte | Guideline | Health Endpoint |
|---------|-----------|----------------|
| Fluoride | 1.5 mg/L | Dental/Skeletal fluorosis |
| Nitrate | 50 mg/L as NO₃ (10 mg/L as NO₃–N) | Methemoglobinemia |

### New Output Variables

| Variable | Description |
|----------|-------------|
| `HI_skeletal_dental` | Hazard Index for fluoride (dental/skeletal effects) |
| `HI_hematological` | Hazard Index for nitrate (blood effects) |
| `HI_anions_screen` | Sum of HQ for all anions |
| `HI_all_screen` | Sum of HQ for all analytes (metals + anions) |

### Nitrate Basis Conversion

If your data is in NO₃–N basis, use `NO3_N` as the column name. The model automatically converts to match the RfD basis using:

- NO₃ → NO₃–N: multiply by 14/62 (≈0.226)
- NO₃–N → NO₃: multiply by 62/14 (≈4.43)

---

## Troubleshooting

### "No metal columns found"

Your CSV column names aren't recognized. Use names like `As`, `Cd`, `Pb` or `C_As`, `C_Cd`, etc.

### "PHREEQC not available"

Install it: `pip install phreeqpython`
The model will still work using simplified estimates without it.

### "Missing RfD for metal X"

The toxicity database doesn't have reference values for that metal. Edit `external/toxref.yml` to add values.

### Very small HI values (like 1.2e-05)

This is normal for clean water with low metal concentrations. Values are displayed in scientific notation.

---

## Adding New Metals

If you have metals not currently supported:

1. **Add atomic weight** to `speciation_modeling.py`:

   ```python
   ATOMIC_WEIGHTS = {
       ...,
       'NewMetal': 63.55,  # g/mol
   }
   ```

2. **Add species** to the SPECIES dictionary:

   ```python
   SPECIES = {
       ...,
       'NewMetal': 'NewMetal+2',  # PHREEQC species name
   }
   ```

3. **Add toxicity values** to `external/toxref.yml`

---

## Need Help?

1. Run `python run_hbmpra.py` for guided interactive mode
2. Check the RUNLOG.json to see what was analyzed
3. Look at debug/HI_summary.csv for your main results

---

## Citation

If you use HBMPRA in your research, please cite:
[Add your citation here]

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
