# HBMPRA

<div align="center">

**Hierarchical Bayesian Model for Probabilistic Risk Assessment**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*A comprehensive framework for trace metal and anion risk assessment in drinking water*

</div>

---

## 🎯 What is HBMPRA?

HBMPRA is a **probabilistic risk assessment tool** that analyzes heavy metal and anion contamination in drinking water. It combines:

- **Thermodynamic speciation modeling** (PHREEQC)
- **Bayesian hierarchical modeling** (PyMC)
- **Multi-organ hazard index calculation**
- **Blood lead level prediction**
- **Sensitivity & uncertainty analysis**

### Key Features

| Feature | Description |
|---------|-------------|
| 🧪 **13 Metals** | As, Cd, Cr, Cu, Hg, Pb, Mn, Fe, Zn, Ni, Co, Al, V |
| 💧 **2 Anions** | Fluoride (F⁻), Nitrate (NO₃⁻) |
| 👥 **4 Demographics** | Adults, Children, Teens, Pregnant |
| 📊 **Organ-Specific** | Neuro, nephro, hepato, skeletal, hematological, and more |
| 🩸 **BLL Prediction** | CDC thresholds (3.5, 5, 10 µg/dL) for lead exposure |
| 🧭 **Flexible Workflow** | Independent toggles for diagnostic vs result plots; anion-only path skips speciation/BLL |
| 🎲 **Uncertainty** | Full posterior distributions, not just point estimates |

---

## 🚀 Quick Start (For Non-Coders)

### Option 1: Interactive Mode (Recommended)

Simply run the interactive workflow — it will guide you through every step:

```bash
python src/run_hbmpra.py
```

You'll see a menu to select your analysis mode:

```
==================================================
  SELECT ANALYSIS MODE
==================================================
  1. Quick Scan    - Fast screening (1000 draws)
  2. Standard      - Full analysis [RECOMMENDED]
  3. Publication   - High-quality (4000 draws)
  4. Custom        - Configure each step manually
```

### Option 2: One-Command Analysis

```bash
python run_hbmpra.py --input waterdata/your_data.csv
```

### What You Need

Your CSV file should have columns like:

```csv
Site,As,Cd,Pb,F,NO3,pH,Eh
Site_A,5.2,0.3,12.1,1.2,45,7.2,150
Site_B,3.1,0.5,8.4,0.8,32,6.9,180
```

If your file contains only fluoride/nitrate (no metals), the workflow automatically switches to an anion-only path that skips speciation and BLL calibration while still computing anion HQ/HI.

### What You Get

```
results_your_data_20251222/
├── trace.nc                 # Full Bayesian posterior
├── figures/                 # Publication-ready plots
│   ├── posterior_organs_panel.png
│   ├── posterior_core_2x2.png
│   └── exceedance_curves.png
├── tables/                  # Summary statistics
│   ├── T3_posterior_summary.csv
│   └── T4_risk_ranking.csv
└── diagnostics/             # MCMC convergence plots
```

### Plot Generation Choices

- Quick/Standard/Publication presets keep their defaults (Quick skips plots; Standard/Publication generate them).
- In Custom mode you are asked separately whether to generate diagnostic plots and whether to generate result plots.

---

## 💻 Developer Setup (For Coders)

### Installation

```bash
# Clone and setup
git clone <repo-url>
cd HBMPRA-main

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install numpy pandas scipy matplotlib pyyaml pymc arviz
pip install phreeqpython  # Optional: for thermodynamic speciation
```

### Project Structure

```
HBMPRA-main/
├── src/                        # Core source code
│   ├── run_hbmpra.py          # 🚀 Main entry point (interactive)
│   ├── hbmpra_optimized.py    # Core Bayesian model
│   ├── bll_engines.py         # Blood lead level engines
│   ├── speciation_modeling.py # PHREEQC integration
│   ├── sensitivity_analysis.py# Sobol/Morris/Delta methods
│   ├── entropy_hpi_peri.py    # HPI/PERI pollution indices
│   ├── units.py               # Unit conversion (incl. nitrate)
│   ├── plot_result.py         # Visualization
│   └── summary_tables.py      # Table generation
├── external/                   # Configuration files
│   ├── toxref.yml             # RfD and SF values
│   ├── analytes.yml           # Analyte definitions (NEW)
│   └── dermal_water_kp.yml    # Dermal permeability
├── waterdata/                  # Input data folder
├── doc/                        # Mathematical documentation
└── tests/                      # pytest test suite
```

### Running Tests

```bash
pytest tests/ -v
```

### Key APIs

```python
# Run analysis programmatically
from hbmpra_optimized import run_hbmpra_model

trace, model = run_hbmpra_model(
    chemistry_file="waterdata/data.csv",
    results_dir="my_results",
    draws=2000,
    tune=2000
)

# Access posteriors
import arviz as az
idata = az.from_netcdf("my_results/trace.nc")
hi_overall = idata.posterior["HI_overall"]
```

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [USER_GUIDE.md](USER_GUIDE.md) | Complete usage guide |
| [doc/README.md](doc/README.md) | Documentation index |
| [doc/HBMPRA_Mathematical_Foundations.md](doc/HBMPRA_Mathematical_Foundations.md) | Full mathematical derivations |
| [doc/Interactive_Workflow_Guide.md](doc/Interactive_Workflow_Guide.md) | Step-by-step workflow guide |

---

## 🧪 Supported Analytes

### Heavy Metals

| Metal | RfD (mg/kg-day) | Target Organs | Carcinogen |
|-------|-----------------|---------------|------------|
| As | 0.0003 | Neuro, Nephro, Derm | Yes |
| Cd | 0.0005 | Nephro | Yes |
| Pb | 3.5/5/10* | Neuro, Hematologic | Yes |
| Hg | 0.0003 | Neuro | No |
| Cr | 0.003 | GI, Hepato | Yes* |

### Anions (NEW)

| Anion | RfD | Target Organ | WHO Guideline |
|-------|-----|--------------|---------------|
| Fluoride (F⁻) | 0.06 mg/kg-day | Skeletal/Dental | 1.5 mg/L |
| Nitrate (NO₃⁻) | 1.6 mg/kg-day (as NO₃–N) | Hematological | 50 mg/L as NO₃ |

---

## 📊 Model Outputs

### Hazard Indices

| Variable | Description |
|----------|-------------|
| `HI_overall` | Maximum across all organs |
| `HI_neuro`, `HI_nephro`, ... | Organ-specific HIs |
| `HI_skeletal_dental` | Fluoride effects (NEW) |
| `HI_hematological` | Nitrate effects (NEW) |
| `HI_all_screen` | Sum of all HQs (backward compat) |

### Other Outputs

| Variable | Description |
|----------|-------------|
| `CR_total` | Incremental lifetime cancer risk |
| `BLL` | Blood lead level (µg/dL) |
| `HQ_*` | Individual hazard quotients |

### Summary Tables

- **T1**: Measured concentrations
- **T2**: Speciation fractions
- **T3**: Posterior summary statistics
- **T4**: Risk ranking (HI, CR, BLL)
- **T5**: Blood Lead Level summary

---

## 🔧 Configuration

### Analysis Presets

| Mode | MCMC Draws | Advanced Analyses | Use Case |
|------|------------|-------------------|----------|
| Quick Scan | 1,000 | Skipped | Testing, screening |
| **Standard** | 2,000 | Skipped | Most analyses |
| Publication | 4,000 | Auto-run | Final results |
| Custom | User choice | User choice | Full control |

### Command-Line Options

```bash
python run_hbmpra.py --help
```

Key flags:

- `--input FILE` — Input CSV file
- `--output DIR` — Output directory
- `--units {ug/L,mg/L}` — Concentration units
- `--skip-speciation` — Skip PHREEQC modeling

---

## 📜 License

MIT License — See [LICENSE](LICENSE) for details.

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

## 📧 Contact

For questions, issues, or collaboration inquiries, please open a GitHub issue or contact the authors directly.

---

<div align="center">
<i>Last updated: January 2026</i>
</div>
