---
title: 'HBMPRA: A Hierarchical Bayesian Model for Probabilistic Risk Assessment of Heavy Metals and Anions in Drinking Water'
tags:
  - Python
  - environmental health
  - risk assessment
  - Bayesian statistics
  - water quality
  - toxicology
  - PHREEQC
  - blood lead level
authors:
  - name: Dickson Abdul-Wahab
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 1
    corresponding: true
  - name: Ebenezer Aquisman Asare
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: University of Ghana, Ghana
   index: 1
date: 2 January 2026
bibliography: paper.bib
---

# Summary

Groundwater contamination by heavy metals and toxic anions poses significant public health risks worldwide, particularly in developing regions where groundwater serves as the primary drinking water source. Traditional risk assessment methods often rely on point estimates and fail to capture the full spectrum of uncertainty in exposure pathways, population variability, and chemical speciation dynamics. HBMPRA (Hierarchical Bayesian Model for Probabilistic Risk Assessment) is a comprehensive Python framework that addresses these limitations by integrating thermodynamic speciation modeling, Bayesian hierarchical inference, and population-specific pharmacokinetic modeling to provide scientifically rigorous probabilistic risk assessments for drinking water contamination.

The framework combines PHREEQC thermodynamic equilibrium calculations to determine bioavailable metal species, PyMC-based Bayesian models to quantify uncertainty in exposure parameters, and multi-organ hazard index calculations to assess organ-specific health risks. HBMPRA supports 13 heavy metals (As, Cd, Cr, Cu, Hg, Pb, Mn, Fe, Zn, Ni, Co, Al, V) and 2 toxic anions (Fluoride, Nitrate), evaluates risks across four demographic groups (adults, children, teens, pregnant women), and provides blood lead level (BLL) predictions calibrated to CDC reference values. The framework produces full posterior distributions for hazard indices, cancer risks, and blood lead levels, enabling decision-makers to quantify the probability of exceeding regulatory thresholds rather than relying on deterministic point estimates.

# Statement of Need

Environmental risk assessment is a critical tool for protecting public health from chemical exposures, yet existing approaches suffer from several fundamental limitations:

1. **Deterministic methods ignore uncertainty**: Traditional EPA Risk Assessment Guidance [@USEPA:1989] uses point estimates for exposure parameters (body weight, water intake, exposure duration), providing no quantitative measure of confidence or probability of exceedance.

2. **Total concentration overestimates risk**: Regulatory assessments typically use total dissolved metal concentrations rather than bioavailable species fractions, leading to conservative but scientifically inaccurate risk estimates [@Nordstrom:2014].

3. **Single-organ hazard indices miss critical pathways**: Summing hazard quotients across all metals and all organs obscures organ-specific vulnerabilities (e.g., manganese neurotoxicity, cadmium nephrotoxicity).

4. **Blood lead models lack population specificity**: Existing BLL prediction tools do not differentiate between mechanistic pharmacokinetic models appropriate for adults versus empirical dose-response relationships necessary for vulnerable populations (children, pregnant women) with higher lead absorption rates [@ATSDR:2020].

5. **Censored data handling**: Environmental datasets frequently contain left-censored values (below detection limit), which are often handled naively (LOD/2 substitution) rather than with statistically rigorous maximum likelihood methods.

6. **Lack of integrated sensitivity analysis**: Few tools provide global sensitivity analysis to identify which parameters most influence risk predictions, hampering targeted data collection efforts.

`HBMPRA` was designed to address all of these limitations in a single integrated workflow. The framework enables environmental researchers, public health officials, and regulatory agencies to:

- **Quantify uncertainty rigorously** using Bayesian hierarchical models that propagate uncertainty from raw measurements through to final risk metrics
- **Account for chemical speciation** by integrating PHREEQC thermodynamic calculations to estimate bioavailable fractions
- **Assess organ-specific risks** with toxicity reference values mapped to individual organ systems (neurological, renal, hepatic, skeletal, hematological)
- **Predict blood lead levels** using population-appropriate pharmacokinetic models (one-compartment PK for adults, empirical slope models for children)
- **Handle censored data scientifically** via maximum likelihood estimation with truncated lognormal imputation
- **Perform global sensitivity analysis** using Sobol variance decomposition, Morris screening, and Delta moment-independent methods

The package has already been applied to groundwater risk assessments in multiple studies and provides a transparent, reproducible, and extensible platform for advancing environmental health science.

# Mathematical Foundations

HBMPRA implements several interconnected mathematical models spanning chemical equilibrium thermodynamics, Bayesian hierarchical modeling, and pharmacokinetics. Here we present the core equations.

## Thermodynamic Speciation

Chemical speciation is modeled using PHREEQC [@Parkhurst:1999], which minimizes the total Gibbs free energy:

$$G = \sum_{i=1}^{n} n_i \left( \mu_i^0 + RT \ln a_i \right)$$

subject to mass balance constraints $\sum a_{ij} n_i = b_j$, where $a_{ij}$ is the stoichiometry of element $j$ in species $i$, $\mu_i^0$ is the standard chemical potential, and $a_i$ is the activity. Activity corrections for ionic strength use the Davies equation:

$$\log \gamma_i = -A z_i^2 \left( \frac{\sqrt{I}}{1 + \sqrt{I}} - 0.3 I \right)$$

where $A = 0.5085$ at 25°C, $z_i$ is the ionic charge, and $I = \frac{1}{2}\sum c_i z_i^2$ is the ionic strength.

## Censored Data Imputation

For left-censored data (concentrations below detection limit), HBMPRA uses maximum likelihood estimation assuming lognormal distributions. For a truncated lognormal with upper bound $L$ (the LOD), the conditional expectation is:

$$E[X | X < L] = \frac{\exp(\mu + \sigma^2/2) \cdot \Phi\left(\frac{\ln(L) - \mu - \sigma^2}{\sigma}\right)}{\Phi\left(\frac{\ln(L) - \mu}{\sigma}\right)}$$

where $\Phi(\cdot)$ is the standard normal CDF and $(\mu, \sigma^2)$ are fitted from uncensored observations via L-BFGS-B optimization.

## Bayesian Exposure Model

Population parameters use non-centered parameterization for efficient MCMC sampling. Body weight follows:

$$z_{bw} \sim \mathcal{N}(0, 1)$$
$$\log(BW_g) = \mu_{\log,bw} + z_{bw} \cdot \sigma_{\log,bw}$$

where $\sigma_{\log,bw} = \sqrt{\ln(1 + CV^2)}$ with $CV = 0.21$ representing inter-individual variability.

Ingestion rate per kg body weight:

$$z_{ir} \sim \mathcal{N}(0, 1)$$
$$\log(IR_{\text{perkg},g}) = \mu_{\log,ir} + z_{ir} \cdot \sigma_{\log,ir}$$

with $\sigma_{\log,ir} = 0.6$ reflecting higher uncertainty in water consumption patterns.

## Dose-Response Calculations

The estimated daily intake (EDI) for ingestion is:

$$EDI_{\text{ing}} = \frac{C \cdot ABS_{GI} \cdot IR \cdot EF \cdot ED}{BW \cdot AT}$$

where $C$ is the bioavailable concentration (mg/L), $ABS_{GI}$ is gastrointestinal absorption fraction, $IR$ is ingestion rate (L/day), $EF$ is exposure frequency (days/year), $ED$ is exposure duration (years), $BW$ is body weight (kg), and $AT$ is averaging time (days).

For dermal exposure via bathing/showering:

$$EDI_{\text{der}} = \frac{C \cdot K_p \cdot ET \cdot SA \cdot EF \cdot ED \cdot 10^{-3}}{BW \cdot AT}$$

where $K_p$ is the dermal permeability coefficient (cm/hr), $ET$ is event time per exposure (hr), and $SA$ is skin surface area (cm²).

## Multi-Organ Hazard Index

Instead of a single hazard index, HBMPRA calculates organ-specific hazard indices:

$$HI_{\text{organ}} = \sum_{m \in S_{\text{organ}}} \left( \frac{EDI_{m,\text{ing}}}{RfD_{m,\text{oral}}} + \frac{EDI_{m,\text{der}}}{RfD_{m,\text{derm}}} \right)$$

where $S_{\text{organ}}$ is the set of metals targeting that organ (e.g., Mn for neurological, Cd for renal).

Total cancer risk from carcinogenic metals (As, CrVI):

$$CR_{\text{total}} = \sum_{m \in \text{carcinogens}} \left( EDI_{m,\text{ing}} \cdot SF_{m,\text{oral}} + EDI_{m,\text{der}} \cdot SF_{m,\text{derm}} \right)$$

where $SF$ is the cancer slope factor (risk per mg/kg-day).

## Blood Lead Level Prediction

For adults, HBMPRA uses a steady-state one-compartment pharmacokinetic model:

$$BLL_{\text{ss}} = \frac{I \cdot f_{\text{abs}} \cdot t_{1/2}}{\ln(2) \cdot V_{\text{blood}} \cdot BW}$$

where $I$ is lead intake (µg/day), $f_{\text{abs}}$ is fractional absorption (0.2-0.3 for adults), $t_{1/2}$ is blood half-life (~30 days), and $V_{\text{blood}} = 0.07$ L/kg.

For children and pregnant women (vulnerable populations), an empirical slope model is used:

$$BLL = BLL_{\text{background}} + K \cdot I \cdot f_{\text{abs}}$$

where $K \approx 0.17$ µg/dL per µg/day for children (higher absorption and sensitivity than adults).

The model outputs posterior distributions for $P(BLL > 3.5 \text{ µg/dL})$, the CDC reference value for children [@CDC:2021].

# Implementation Architecture

HBMPRA is implemented in Python 3.9+ with a modular architecture separating scientific domains:

- **`speciation_modeling.py`**: PHREEQC integration for thermodynamic equilibrium calculations
- **`hbmpra_optimized.py`**: Core Bayesian risk model using PyMC for MCMC sampling
- **`bll_engines.py`**: Population-specific blood lead level calculators
- **`sensitivity_analysis.py`**: Global sensitivity methods (Sobol, Morris, Delta)
- **`entropy_hpi_peri.py`**: Pollution index calculations (Heavy Metal Pollution Index, Pollution Ecological Risk Index)
- **`demographics.py`**: Population group parameters (body weight, intake rates)
- **`units.py`**: Unit conversions and nitrate basis handling

All toxicity reference values (RfD, slope factors) are externalized in `external/toxref.yml` following the principle of configuration-driven toxicology, ensuring traceability to authoritative sources (EPA IRIS, ATSDR, WHO).

The framework uses vectorized NumPy operations to maximize computational efficiency. For example, dermal hazard quotient calculation is vectorized across (metal × site × demographic group) dimensions rather than using nested Python loops, achieving ~10× speedup.

MCMC sampling employs PyMC's NUTS sampler with non-centered parameterization, typically converging in 2000 tuning + 2000 draw iterations across 4 chains. Convergence diagnostics ($\hat{R} < 1.01$, effective sample size $> 400$) are automatically computed and reported.

## Software Dependencies

**Required:**
- `numpy`, `pandas` - Numerical computing and data manipulation
- `matplotlib`, `seaborn` - Visualization
- `pyyaml` - Configuration file parsing

**Optional (for full functionality):**
- `pymc` - Bayesian modeling and MCMC sampling
- `arviz` - Posterior analysis and diagnostics
- `phreeqpython` - PHREEQC thermodynamic speciation
- `scipy` - Optimization and statistical functions
- `SALib` - Sensitivity analysis methods

The package gracefully degrades when optional dependencies are unavailable (e.g., uses simplified speciation estimates if PHREEQC is not installed).

## Input Data Format

Input CSV files require:
- Site identifier column (first column)
- Metal concentration columns (µg/L or mg/L): `As`, `Cd`, `Cr`, `Cu`, `Hg`, `Pb`, `Mn`, `Fe`, `Zn`, `Ni`, `Co`, `Al`, `V`
- Optional anion columns (mg/L): `F` (fluoride), `NO3` or `NO3_N` (nitrate)
- Optional water chemistry: `pH`, `Eh` (for PHREEQC speciation)

Left-censored values can be indicated with `<` prefix (e.g., `<0.5`).

## Output Products

Each analysis generates a time-stamped results directory containing:

1. **`trace.nc`**: Full Bayesian posterior as ArviZ InferenceData object (NetCDF format)
2. **`tables/`**: Summary tables including:
   - `T3_posterior_summary.csv`: Mean, median, 95% credible intervals for all risk metrics
   - `T4_risk_ranking.csv`: Sites ranked by hazard index and cancer risk
   - `T5_BLL_summary.csv`: Blood lead level predictions with exceedance probabilities
3. **`figures/`**: Publication-ready plots (hazard index posteriors, exceedance curves, organ-specific panels)
4. **`diagnostics/`**: MCMC convergence diagnostics (trace plots, R-hat statistics)

# Validation and Testing

HBMPRA includes a comprehensive test suite (`tests/`) covering:

- **Unit calculations** (`test_bll_engines.py`): Verifies monotonicity of BLL with concentration, correct units
- **Integration tests** (`test_phreeqc_bio_integration.py`): End-to-end workflow from raw data to risk outputs
- **Numeric stability** (`test_dermal_hq_numeric.py`): Edge cases (zero concentrations, censored data)
- **Basis conversions** (`test_nitrate_units.py`): Nitrate basis transformations (NO₃ vs NO₃-N)

All numerical implementations are validated against peer-reviewed literature and regulatory guidance documents (EPA RAGS, ATSDR MRLs, WHO drinking water guidelines).

# Usage Example

Interactive mode provides a guided workflow:

```bash
cd src
python run_hbmpra.py
```

For automated pipelines:

```bash
python run_hbmpra.py --input ../waterdata/site_chemistry.csv \
    --output ../results --draws 2000 --tune 2000
```

Manual step-by-step workflow:

```bash
# 1. Speciation modeling
python speciation_modeling.py --input ../waterdata/data.csv \
    --output-dir ../results

# 2. Bayesian risk assessment
python hbmpra_optimized.py --chemistry ../waterdata/data.csv \
    --results-dir ../results --use-bioavailable

# 3. Generate summary tables and plots
python summary_tables.py --results-dir ../results
python plot_result.py --results-dir ../results

# 4. Sensitivity analysis (optional)
python sensitivity_analysis.py --results-dir ../results \
    --method sobol --n-samples 2000
```

# Research Applications

HBMPRA enables several research applications:

1. **Probabilistic risk screening**: Identify high-risk sites accounting for uncertainty
2. **Regulatory compliance assessment**: Calculate $P(HI > 1)$ or $P(CR > 10^{-6})$ for decision thresholds
3. **Population vulnerability analysis**: Compare risks across demographic groups
4. **Sensitivity-guided sampling**: Use Sobol indices to prioritize additional data collection
5. **Climate change scenarios**: Assess how changing water chemistry (pH, Eh) affects speciation and risk

# Comparison with Existing Tools

HBMPRA differs from existing risk assessment software in several key aspects:

| Feature | HBMPRA | EPA RSL Calculator | RAIS | SADA |
|---------|--------|-------------------|------|------|
| Probabilistic | Yes (full Bayesian) | No | Limited | Yes |
| Speciation modeling | Yes (PHREEQC) | No | No | No |
| Multi-organ HI | Yes | Lumped | Lumped | Lumped |
| Population-specific BLL | Yes | No | No | No |
| Open source | Yes | N/A | No | Proprietary |
| Global sensitivity | Yes (Sobol/Morris/Delta) | No | No | Limited |

# Limitations and Future Work

Current limitations include:

1. **No inhalation pathway**: Only ingestion and dermal routes are modeled
2. **Methylmercury not predicted**: PHREEQC cannot model bacterial methylation; total Hg only
3. **Steady-state assumption**: Temporal dynamics of exposure are not captured
4. **Single water source**: Does not account for multiple exposure pathways (diet, soil, air)

Planned enhancements:

- Integration with USEPA CompTox Dashboard for expanded chemical coverage
- Time-series analysis for longitudinal exposure assessment
- Multi-pathway cumulative risk assessment
- Machine learning emulators for faster sensitivity analysis

# Acknowledgements

We acknowledge the developers of PyMC, ArviZ, PHREEQC, and SALib for providing essential scientific computing infrastructure. We thank the reviewers and contributors who helped improve the documentation and test coverage. This work builds upon decades of environmental health research by EPA, ATSDR, and WHO.

# References

## Example References (paper.bib format)

```bibtex
@techreport{USEPA:1989,
  author = {{U.S. Environmental Protection Agency}},
  title = {Risk Assessment Guidance for Superfund, Volume I: Human Health Evaluation Manual (Part A)},
  institution = {U.S. EPA},
  year = {1989},
  number = {EPA/540/1-89/002},
  url = {https://www.epa.gov/risk/risk-assessment-guidance-superfund-rags-part}
}

@article{Nordstrom:2014,
  author = {Nordstrom, D. Kirk},
  title = {Improving Internal Consistency of Standard State Thermodynamic Data for Aqueous Ions and Complexes},
  journal = {Geochimica et Cosmochimica Acta},
  volume = {140},
  pages = {217--259},
  year = {2014},
  doi = {10.1016/j.gca.2014.05.013}
}

@techreport{ATSDR:2020,
  author = {{Agency for Toxic Substances and Disease Registry}},
  title = {Toxicological Profile for Lead},
  institution = {U.S. Department of Health and Human Services},
  year = {2020},
  url = {https://www.atsdr.cdc.gov/toxprofiles/tp13.pdf}
}

@article{Parkhurst:1999,
  author = {Parkhurst, David L. and Appelo, C. A. J.},
  title = {User's guide to PHREEQC (Version 2): A computer program for speciation, batch-reaction, one-dimensional transport, and inverse geochemical calculations},
  journal = {Water-Resources Investigations Report},
  volume = {99-4259},
  year = {1999},
  publisher = {U.S. Geological Survey},
  doi = {10.3133/wri994259}
}

@techreport{CDC:2021,
  author = {{Centers for Disease Control and Prevention}},
  title = {Blood Lead Reference Value},
  institution = {CDC},
  year = {2021},
  url = {https://www.cdc.gov/nceh/lead/data/blood-lead-reference-value.htm}
}

@article{Salvatier:2016,
  author = {Salvatier, John and Wiecki, Thomas V. and Fonnesbeck, Christopher},
  title = {Probabilistic programming in Python using PyMC3},
  journal = {PeerJ Computer Science},
  volume = {2},
  pages = {e55},
  year = {2016},
  doi = {10.7717/peerj-cs.55}
}

@article{Kumar:2019,
  author = {Kumar, Ravin and Carroll, Colin and Hartikainen, Ari and Martin, Osvaldo A.},
  title = {ArviZ a unified library for exploratory analysis of Bayesian models in Python},
  journal = {Journal of Open Source Software},
  volume = {4},
  number = {33},
  pages = {1143},
  year = {2019},
  doi = {10.21105/joss.01143}
}

@article{Herman:2017,
  author = {Herman, Jonathan and Usher, Will},
  title = {SALib: An open-source Python library for Sensitivity Analysis},
  journal = {Journal of Open Source Software},
  volume = {2},
  number = {9},
  pages = {97},
  year = {2017},
  doi = {10.21105/joss.00097}
}

@book{WHO:2017,
  author = {{World Health Organization}},
  title = {Guidelines for Drinking-water Quality: Fourth Edition Incorporating the First Addendum},
  publisher = {WHO},
  year = {2017},
  isbn = {978-92-4-154995-0},
  url = {https://www.who.int/publications/i/item/9789241549950}
}

@article{Sobol:2001,
  author = {Sobol', I. M.},
  title = {Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates},
  journal = {Mathematics and Computers in Simulation},
  volume = {55},
  number = {1-3},
  pages = {271--280},
  year = {2001},
  doi = {10.1016/S0378-4754(00)00270-6}
}

@article{Morris:1991,
  author = {Morris, Max D.},
  title = {Factorial Sampling Plans for Preliminary Computational Experiments},
  journal = {Technometrics},
  volume = {33},
  number = {2},
  pages = {161--174},
  year = {1991},
  doi = {10.1080/00401706.1991.10484804}
}
```
