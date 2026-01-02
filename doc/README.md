# HBMPRA Documentation Index

This directory contains comprehensive mathematical documentation for the **Hierarchical Bayesian Model for Probabilistic Risk Assessment (HBMPRA)** framework covering metals and anions in drinking water.

---

## 📚 Documentation Files

### Core Mathematical Foundations

1. **[HBMPRA_Mathematical_Foundations.md](HBMPRA_Mathematical_Foundations.md)**
   - **Purpose**: Complete mathematical derivation of the Bayesian risk assessment model
   - **Key Topics**:
     - Left-censored data imputation (MLE + conditional expectation)
     - Thermodynamic speciation equilibrium (Gibbs Free Energy minimization)
     - Hierarchical Bayesian priors (non-centered parameterization)
     - Multi-organ hazard index aggregation
     - Cancer risk calculation with route-specific slope factors
     - Dermal absorption permeability model
   - **Mathematical Depth**: ⭐⭐⭐⭐⭐ (Full proofs and derivations)

### Module-Specific Documentation

2. **[Interactive_Workflow_Guide.md](Interactive_Workflow_Guide.md)**
   - **Purpose**: Complete guide to the 11-step interactive analysis workflow
   - **Key Topics**:
     - Step-by-step walkthrough of `run_hbmpra.py`
     - User interaction patterns and decision trees
     - Input validation and error handling
     - Command-line options vs interactive mode
     - Customization and extension guide
   - **Target Audience**: ⭐⭐⭐ (Beginners and non-coders)

3. **[BLL_Pharma_Foundations.md](BLL_Pharma_Foundations.md)**
   - **Purpose**: Blood Lead Level (BLL) prediction models
   - **Key Topics**:
     - One-compartment pharmacokinetic model (adults)
     - Empirical slope model (children/vulnerable populations)
     - Grid-based Bayesian prior calibration
     - Population-specific engine selection logic
   - **Mathematical Depth**: ⭐⭐⭐⭐ (Mechanistic derivations)

3. **[Speciation_and_Metal_Chemistry.md](Speciation_and_Metal_Chemistry.md)**
   - **Purpose**: PHREEQC integration and speciation logic
   - **Key Topics**:
     - Chemical equilibrium solver integration
     - Multi-species handling (CrVI/CrIII, Hg(II)/Hg(0))
     - Bioavailability selection criteria
     - Dermal permeability coefficient ($K_p$) resolution
   - **Mathematical Depth**: ⭐⭐⭐ (Chemical equilibrium theory)

4. **[Sensitivity_Analysis_Foundations.md](Sensitivity_Analysis_Foundations.md)**
   - **Purpose**: Global and local sensitivity analysis methods
   - **Key Topics**:
     - Sobol variance decomposition (first-order and total-effect indices)
     - Morris screening (elementary effects)
     - Delta moment-independent method
     - Parameter space construction (log-normal bounds)
     - Batch vectorized model evaluation
   - **Mathematical Depth**: ⭐⭐⭐⭐⭐ (Complete sensitivity theory)

5. **[Entropy_Pollution_Indices.md](Entropy_Pollution_Indices.md)**
   - **Purpose**: Entropy-weighted pollution assessment
   - **Key Topics**:
     - Shannon entropy and information theory
     - Entropy weight calculation and normalization
     - Heavy Metal Pollution Index (HPI)
     - Potential Ecological Risk Index (PERI)
     - Bootstrap uncertainty quantification
   - **Mathematical Depth**: ⭐⭐⭐⭐ (Information theory + ecology)

### Implementation Guide

6. **[Implementation_Architecture.md](Implementation_Architecture.md)**
   - **Purpose**: Map theory to code
   - **Key Topics**:
     - Module-by-module code walkthrough
     - Data flow between components
     - PyMC model construction details
     - Vectorization strategies for performance
   - **Technical Depth**: ⭐⭐⭐⭐ (Software engineering)

---

## 🧮 Mathematical Coverage Summary

| Concept | Documented | Code Reference (module-level) | Validation |
|---------|-----------|--------------------------------|------------|
| **MLE Censored Data** | ✅ Full proof | hbmpra_optimized.py | Efron & Tibshirani (1993) |
| **Non-Centered Priors** | ✅ Full derivation | hbmpra_optimized.py | Betancourt & Girolami (2015) |
| **Gibbs Minimization** | ✅ Equilibrium equations | speciation_modeling.py | PHREEQC Manual |
| **Dermal Permeability** | ✅ Flux model + units | hbmpra_optimized.py | EPA RAGS (1989) |
| **Organ HI Routing** | ✅ Route-specific logic | hbmpra_optimized.py | IRIS Database |
| **Sobol Indices** | ✅ ANOVA decomposition | sensitivity_analysis.py | Saltelli et al. (2010) |
| **Shannon Entropy** | ✅ Information theory | entropy_hpi_peri.py | Shannon (1948) |
| **BLL Steady-State** | ✅ ODE solution | bll_engines.py | O'Flaherty (1995) |
| **Bootstrap CI** | ✅ Resampling theory | entropy_hpi_peri.py | Efron (1979) |
| **Fluoride HQ** | ✅ Oral RfD | toxref.yml, hbmpra_optimized.py | EPA IRIS |
| **Nitrate HQ** | ✅ NO₃–N basis conversion | units.py | ATSDR |
| **Anion-Only Mode** | ✅ Conditional workflow | run_hbmpra.py | (implemented) |

### Workflow Alignment Notes

- Interactive presets now allow independent toggles for diagnostic vs result plots in Custom mode.
- Anion-only datasets skip speciation and BLL calibration while computing anion HQ/HIs.
- Speciation gracefully skips metals without total concentration columns and uses total concentration fallback when requested.

---

## 📖 How to Use This Documentation

### For Researchers

1. Start with **HBMPRA_Mathematical_Foundations.md** for the overall framework
2. Dive into module-specific docs for detailed derivations
3. Cross-reference with code using provided line numbers

### For Developers

1. Read **Implementation_Architecture.md** first
2. Use mathematical docs to understand the "why" behind implementation choices
3. Refer to sensitivity and entropy docs when optimizing performance

### For Reviewers/Validators

1. Check **Mathematical Validation** sections in each document
2. All equations are traceable to peer-reviewed literature
3. Code line references enable exact verification

---

## 🔬 Notation Conventions

| Symbol | Meaning | Units |
|--------|---------|-------|
| $C_m$ | Metal concentration | µg/L or mg/L |
| $EDI$ | Estimated Daily Intake | mg/kg-day |
| $HI$ | Hazard Index | dimensionless |
| $CR$ | Cancer Risk | probability |
| $BLL$ | Blood Lead Level | µg/dL |
| $K_p$ | Dermal permeability | cm/hr |
| $RfD$ | Reference Dose | mg/kg-day |
| $SF$ | Slope Factor | (mg/kg-day)$^{-1}$ |
| $w_j$ | Entropy weight | dimensionless |
| $S_i$ | Sobol first-order index | proportion [0,1] |
| $S_{Ti}$ | Sobol total-effect index | proportion [0,1] |
| $C_{F}$ | Fluoride concentration | mg/L as F⁻ |
| $C_{NO_3}$ | Nitrate concentration | mg/L as NO₃ |
| $C_{NO_3-N}$ | Nitrate as nitrogen | mg/L as NO₃–N |

---

## 📊 Validation References

All mathematical formulations have been validated against:

- **EPA RAGS** (Risk Assessment Guidance for Superfund, 1989)
- **IRIS Database** (Integrated Risk Information System)
- **WHO Guidelines** for drinking water quality
- **Peer-reviewed literature** (see References in each document)

---

## 🚀 Future Extensions

Potential additions to the documentation suite:

- [ ] Markov Chain Monte Carlo convergence diagnostics
- [ ] Prior elicitation methodology
- [ ] Spatial autocorrelation extensions
- [ ] Machine learning surrogate models for sensitivity analysis

---

## 📝 Contributing

To add new mathematical documentation:

1. Follow the existing template structure
2. Include full derivations with proofs
3. Provide code line references
4. Add validation against literature
5. Use LaTeX for equations
6. Include worked examples where applicable

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

For questions, issues, or collaboration inquiries:

- Open an issue on the project repository
- Contact the authors via their profiles above

---

*Last Updated: 2026-01-02*
