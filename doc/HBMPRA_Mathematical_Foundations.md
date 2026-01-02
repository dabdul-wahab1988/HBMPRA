# HBMPRA Mathematical Foundations and Proofs (Comprehensive)

This document provides a complete mathematical derivation of the models and algorithms implemented in the **Hierarchical Bayesian Model for Probabilistic Risk Assessment (HBMPRA)** framework.

---

## Table of Contents

1. [Data Pre-processing: Censoring & Imputation](#1-data-pre-processing-censoring--imputation)
2. [Thermodynamic Speciation (PHREEQC)](#2-thermodynamic-speciation-phreeqc)
3. [Hierarchical Bayesian Exposure Model](#3-hierarchical-bayesian-exposure-model)
4. [Multi-Organ Risk Integration](#4-multi-organ-risk-integration)
5. [Entropy-Based Weighting (HPI/PERI)](#5-entropy-based-weighting-hpiperi)
6. [Non-Centered Parameterization](#6-non-centered-parameterization)
7. [Dermal Absorption Model](#7-dermal-absorption-model)
8. [Anion Hazard Quotient (NEW)](#8-anion-hazard-quotient-new)
9. [Risk Ranking and Summary Statistics](#9-risk-ranking-and-summary-statistics)

---

## 1. Data Pre-processing: Censoring & Imputation

Environmental datasets often contain "non-detects" (values below the Limit of Detection, LOD). HBMPRA uses a scientifically rigorous Maximum Likelihood Estimation (MLE) approach to impute these values rather than using simple $LOD/2$ substitution.

### 1.1 Log-Normal MLE for Censored Data

We assume trace metal concentrations $X$ follow a Log-Normal distribution: $\ln(X) \sim \mathcal{N}(\mu, \sigma^2)$.
For a set of observed (detected) values $\{y_1, y_2, \dots, y_n\}$, we minimize the Negative Log-Likelihood (NLL):
$$NLL(\mu, \sigma) = \frac{n}{2} \ln(2\pi\sigma^2) + \sum_{i=1}^{n} \frac{(\ln y_i - \mu)^2}{2\sigma^2}$$

**Implementation**: This is solved using L-BFGS-B optimization in `fit_left_censored_lognormal()`.

### 1.2 Conditional Expectation Imputation

For samples below the $LOD$, we impute the value as the conditional expectation $E[X | X < LOD]$:
$$E[X | X < LOD] = \frac{\exp(\mu + \sigma^2/2) \cdot \Phi\left(\frac{\ln(LOD) - \mu - \sigma^2}{\sigma}\right)}{\Phi\left(\frac{\ln(LOD) - \mu}{\sigma}\right)}$$
where $\Phi(\cdot)$ is the cumulative distribution function (CDF) of the standard normal distribution.

**Proof**: For a truncated log-normal distribution with upper bound $L = LOD$:
$$E[X | X < L] = \int_0^L x \cdot f_X(x | X < L) dx = \frac{\int_0^L x \cdot f_X(x) dx}{P(X < L)}$$
where $f_X(x) = \frac{1}{x\sigma\sqrt{2\pi}} \exp\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right)$.

Substituting $u = \ln x$, we obtain the closed-form solution above.

---

## 2. Thermodynamic Speciation (PHREEQC)

### 2.1 Chemical Equilibrium Derivation

The model assumes chemical equilibrium, minimizing the total Gibbs Free Energy ($G$):
$$G = \sum_{i=1}^{n} n_i \left( \mu_i^0 + RT \ln a_i \right)$$
subject to mass balance: $\sum a_{ij} n_i = b_j$, where $a_{ij}$ is the stoichiometry of element $j$ in species $i$.

Using Lagrange multipliers, the equilibrium condition is:
$$\mu_i^0 + RT \ln a_i + \sum_j \lambda_j a_{ij} = 0$$

**Implementation**: PHREEQC solves this iterative using Newton-Raphson methods for the ion-association model.

### 2.2 Activity Correction (Davies Equation)

For ionic species in water with moderate ionic strength ($I < 0.5$ M):
$$\log \gamma_i = -A z_i^2 \left( \frac{\sqrt{I}}{1 + \sqrt{I}} - 0.3 I \right)$$
where $A = 0.5085$ at 25°C, $z_i$ is the charge, and $I = \frac{1}{2}\sum c_i z_i^2$.

### 2.3 Bioavailability Selection Logic

HBMPRA automatically selects the "best" species for risk assessment based on:

1. **Mean Molality**: Preference for the most abundant free-ion form.
2. **Coefficient of Variation (CV)**: The species must show meaningful variation ($CV > 0.01$) across sites to be considered a robust indicator.
3. **Concentration Threshold**: Mean concentration must exceed $10^{-23}$ mol/L.

**Code reference**: `speciation_modeling.py`, lines 688-724.

---

## 3. Hierarchical Bayesian Exposure Model

### 3.1 Non-Centered Prior Distributions

To ensure biological realism and improve MCMC sampling efficiency, population parameters are modeled using **non-centered parameterization**:

#### Body Weight ($BW_g$)

Instead of directly sampling $BW_g \sim \text{LogNormal}(\mu_{bw}, \sigma_{bw}^2)$, we use:
$$z_{bw} \sim \mathcal{N}(0, 1)$$
$$\log(BW_g) = \mu_{log,bw} + z_{bw} \cdot \sigma_{log,bw}$$

where:
$$\sigma_{log,bw} = \sqrt{\ln(1 + CV^2)}, \quad CV = 0.21$$
$$\mu_{log,bw} = \ln(\text{BW}_{mean}) - \frac{1}{2}\ln(1 + CV^2)$$

**Rationale**: Non-centered parameterization decorrelates the latent variable $z$ from the hyperparameters, improving HMC sampler geometry.

#### Ingestion Rate ($IR_g$)

Similarly:
$$z_{ir} \sim \mathcal{N}(0, 1)$$
$$\log(IR_{perkg,g}) = \mu_{log,ir} + z_{ir} \cdot \sigma_{log,ir}$$
where $\sigma_{log,ir} = 0.6$ (reflecting higher uncertainty in water consumption patterns).

**Code reference**: `hbmpra_optimized.py`, lines 722-739.

### 3.2 Dose-Response Equations (Ingestion & Dermal)

#### Ingestion EDI ($EDI_{ing}$)

$$EDI_{ing} = \frac{C \cdot ABS_{GI} \cdot IR \cdot EF \cdot ED}{BW \cdot AT}$$

**Units**:

- $C$: mg/L (after conversion from $\mu$g/L using $CF_{ugL \to mgL} = 10^{-3}$)
- $IR$: L/day
- $EF$: days/year
- $ED$: years
- $BW$: kg
- $AT$: days
- Result: mg/kg-day

**Note**: $ABS_{GI}$ (Gastrointestinal Absorption) is metal-specific, ensuring only the absorbed portion contributes to systemic risk.

#### Dermal EDI ($EDI_{der}$)

The dermal route uses a permeability-based flux model:
$$EDI_{der} = \frac{C_{bio} \cdot K_p \cdot ET \cdot SA \cdot EF \cdot ED \cdot 10^{-3}}{BW \cdot AT}$$
where:

- $K_p$: Permeability coefficient (cm/hr)
- $ET$: Event time (hr/event)
- $SA$: Skin surface area (cm$^2$)
- $10^{-3}$: Conversion factor from (mg/L)(cm/hr)(hr)(cm²) to appropriate dose units

**Derivation**:
Flux $J = K_p \cdot C_{bio}$ (mg/cm²/hr)
Total daily dose = $J \cdot SA \cdot (ET/24)$ (mg/day)
Normalized to body weight and averaged over time gives the EDI.

**Code reference**: `hbmpra_optimized.py`, lines 756-777.

---

## 4. Multi-Organ Risk Integration

### 4.1 Hazard Index (HI) per Organ System

Instead of a single "total" HI, HBMPRA calculates site-specific risk for individual organ systems $O = \{\text{neuro, nephro, hepato, systemic}, \dots\}$:
$$HI_{Organ_o} = \sum_{m \in S_o} \left( \frac{EDI_{m, ing}}{RfD_{m, oral}} + \frac{EDI_{m, der}}{RfD_{m, derm}} \right)$$
where $S_o$ is the set of metals targeting organ $o$.

**Implementation Details**:

- Organ sets $S_o$ are defined in `external/toxref.yml` under the `organ_sets` key.
- The model supports route-specific contributions: some organs (e.g., "derm", "gi") only receive contributions from their respective routes.
- **Dermal Override**: By default, if bioavailable concentrations exist, dermal contributions are included for all organs unless `--disable-dermal-override` is specified.

**Code reference**: `hbmpra_optimized.py`, lines 812-866.

### 4.2 Overall Hazard Index

The overall HI is the maximum across all organ systems:
$$HI_{overall} = \max_{o \in O} HI_o$$

**Justification**: The "maximum" aggregation reflects the principle that health risk is determined by the most vulnerable target organ.

### 4.3 Incremental Lifetime Cancer Risk (CR)

Calculated as the sum of dose-potency products across carcinogenic metals:
$$CR = \sum_{m \in \text{Carcinogens}} \left( LADD_{m, ing} \cdot SF_{m, oral} + LADD_{m, der} \cdot SF_{m, derm} \right)$$

where $LADD$ (Lifetime Average Daily Dose) uses cancer averaging time $AT_c = 70$ years.

**Code reference**: `hbmpra_optimized.py`, lines 891-904.

---

## 5. Entropy-Based Weighting (HPI/PERI)

### 5.1 Shannon Entropy Weighting

Shannon Entropy ($H_j$) measures the dispersion of metal concentrations across sites:
$$H_j = -k \sum_{i=1}^n p_{ij} \ln p_{ij}, \quad p_{ij} = \frac{C_{ij}}{\sum_{i=1}^n C_{ij}}$$
where $k = 1/\ln(n)$ is the normalization constant.

The weight $w_j$ is inversely proportional to $H_j$:
$$w_j = \frac{1 - H_j}{\sum_{j=1}^m (1 - H_j)} = \frac{d_j}{\sum_{j=1}^m d_j}$$
where $d_j = 1 - H_j$ is the "divergence" or "information content".

**Theorem**: A lower entropy $H_j$ indicates that concentrations are more "concentrated" at certain sites, providing more information for pollution hotspot identification, hence a higher weight.

**Proof**:

- Maximum entropy occurs when $p_{ij} = 1/n$ (uniform distribution), giving $H_{max} = -k \cdot n \cdot (1/n) \ln(1/n) = 1$.
- Minimum entropy occurs when one site has all the concentration: $H_{min} = 0$.
- Lower $H_j$ means higher spatial heterogeneity, which is more informative for risk mapping.

**Code reference**: `entropy_hpi_peri.py`, lines 139-217.

### 5.2 Pollution Indices

- **Heavy Metal Pollution Index (HPI)**:
    $$HPI = \sum_{j=1}^m w_j \cdot Q_j, \quad Q_j = \frac{C_j}{S_j}$$
    where $S_j$ is the regulatory standard for metal $j$.

- **Potential Ecological Risk Index (PERI)**:
    $$PERI = \sum_{j=1}^m w_j \cdot T_j \cdot \frac{C_j}{S_j}$$
    where $T_j$ is the toxic-response factor (Hakanson, 1980).

---

## 6. Non-Centered Parameterization

### 6.1 Mathematical Background

Standard centered parameterization:
$$\theta \sim \text{LogNormal}(\mu, \sigma^2)$$

can have poor sampling geometry when $\sigma$ is learned from data, creating a "funnel" in parameter space.

**Solution**: Non-centered reparameterization:
$$z \sim \mathcal{N}(0, 1)$$
$$\theta = \exp(\mu + z \cdot \sigma)$$

**Advantage**: $z$ and $(\mu, \sigma)$ are now independent in the prior, improving Hamiltonian Monte Carlo (HMC) sampling efficiency.

**Code implementation**:

```python
z_log_bw = pm.Normal("z_log_bw", mu=0.0, sigma=1.0, shape=G)
log_BW_g = pm.Deterministic("log_BW_g", mu_log_bw + z_log_bw * sigma_log_bw)
BW_g = pm.Deterministic("BW_g", at.exp(log_BW_g))
```

---

## 7. Dermal Absorption Model

### 7.1 Permeability Coefficient Model

The dermal permeability coefficient $K_p$ (cm/hr) is obtained from `external/dermal_water_kp.yml` and represents the steady-state flux across the stratum corneum.

### 7.2 Dose Calculation

Daily dermal dose (mg/kg-day):
$$D_{dermal} = \frac{C_{bio} \cdot K_p \cdot t_{event} \cdot SA}{BW}$$

**Time-averaged for chronic exposure**:
$$EDI_{der} = D_{dermal} \cdot \frac{EF \cdot ED}{AT}$$

**Units verification**:

- $(mg/L) \cdot (cm/hr) \cdot (hr) \cdot (cm^2) / (kg) = (mg \cdot cm^3/L) \cdot cm^{-1} / kg$
- Using $1 L = 1000 cm^3$: $(mg) / (1000 \cdot kg) = mg/(1000 \cdot kg)$
- Hence the $10^{-3}$ factor in the code.

**Code reference**: `hbmpra_optimized.py`, lines 794-799.

---

## Mathematical Validation

All equations have been validated against:

1. **EPA RAGS** (Risk Assessment Guidance for Superfund)
2. **IRIS Database** (Integrated Risk Information System)
3. **WHO Guidelines** for drinking water quality
4. **Peer-reviewed literature** on Bayesian risk assessment and entropy-based environmental indices

---

## References

1. Hakanson, L. (1980). An ecological risk index for aquatic pollution control. *Water Research*, 14(8), 975-1001.
2. Shannon, C.E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27(3), 379-423.
3. EPA (1989). Risk Assessment Guidance for Superfund, Volume I: Human Health Evaluation Manual.
4. Bogen, K.T., & Spear, R.C. (1987). Integrating uncertainty and interindividual variability in environmental risk assessment. *Risk Analysis*, 7(4), 427-436.

---

## 8. Anion Hazard Quotient (NEW)

### 8.1 Fluoride (F⁻)

Fluoride risk is assessed via the oral Hazard Quotient:

$$HQ_{fluoride} = \frac{EDI_{F}}{RfD_{F,oral}}$$

where:

- $RfD_{F,oral} = 0.06$ mg/kg-day (EPA IRIS)
- Target organ: *skeletal_dental* (dental/skeletal fluorosis)
- Input units: mg/L as F⁻

**WHO Guideline**: 1.5 mg/L

### 8.2 Nitrate (NO₃⁻)

Nitrate requires **basis conversion** because RfD is expressed as NO₃–N:

$$C_{NO_3-N} = C_{NO_3} \times \frac{M_{N}}{M_{NO_3}} = C_{NO_3} \times \frac{14}{62}$$

Then:
$$HQ_{nitrate} = \frac{EDI_{NO_3-N}}{RfD_{NO_3-N,oral}}$$

where:

- $RfD_{NO_3-N,oral} = 1.6$ mg/kg-day (ATSDR)
- Target organ: *hematological* (methemoglobinemia)
- Conversion factor: $14/62 \approx 0.226$

**WHO Guideline**: 50 mg/L as NO₃ (≡ 10 mg/L as NO₃–N)

### 8.3 Organ-Specific HI for Anions

$$HI_{skeletal\_dental} = HQ_{fluoride}$$
$$HI_{hematological} = HQ_{nitrate}$$

### 8.4 Overall HI with Anions

$$HI_{overall} = \max\left(HI_{neuro}, HI_{nephro}, \ldots, HI_{skeletal\_dental}, HI_{hematological}\right)$$

### 8.5 Screening HIs (Backward Compatibility)

$$HI_{metals\_screen} = \sum_{m \in Metals} HQ_m$$
$$HI_{anions\_screen} = HQ_{fluoride} + HQ_{nitrate}$$
$$HI_{all\_screen} = HI_{metals\_screen} + HI_{anions\_screen}$$

**Code reference**: `hbmpra_optimized.py`, `units.py:convert_nitrate_basis_mgL`

---

## 9. Risk Ranking and Summary Statistics

To prioritize remediation efforts, HBMPRA implements a probabilistic ranking algorithm that orders sites based on multiple risk metrics (HI, Cancer Risk, BLL).

### 9.1 Risk Ranking Algorithm (T4)

Sites are ranked primarily by the **Mean Overall Hazard Index ($HI_{overall}$)**. Secondary rankings can be generated for Cancer Risk (CR) and Blood Lead Level (BLL).

For each site $i$ and demographic group $j$, statistics are computed from the posterior trace:

- **Mean HI**: $\mathbb{E}[HI_{overall}^{(i,j)}]$
- **97th Percentile**: $P_{97}(HI_{overall}^{(i,j)})$ (Upper bound of risk)

The ranking sorts sites descending by Mean HI.

### 9.2 Exceedance Probabilities

The Bayesian framework allows direct calculation of the probability that a risk metric exceeds a regulatory threshold $\tau$:

$$P(\text{Risk} > \tau) = \frac{1}{N} \sum_{k=1}^N \mathbb{I}(R_k > \tau)$$

where $R_k$ is the risk value in the $k$-th MCMC sample, and $\mathbb{I}(\cdot)$ is the indicator function.

**Thresholds used:**

- **Hazard Index (HI)**: $\tau = 1.0$ (EPA)
- **Cancer Risk (CR)**: $\tau = 10^{-6}$ (EPA de minimis risk)
- **Blood Lead Level (BLL)**:
  - Children/Pregnant: $\tau = 3.5 \mu g/dL$ (CDC 2021 Reference Value)
  - Teens/Others: $\tau = 5.0 \mu g/dL$
  - Adults: $\tau = 10.0 \mu g/dL$ (Occupational)

### 9.3 Credible Intervals

Summary tables (T3) report the **94% Highest Density Interval (HDI)** equivalent, represented by the 3rd and 97th percentiles:

- **Lower Bound**: $P_3$
- **Upper Bound**: $P_{97}$
- **Central Tendency**: Median ($P_{50}$) and Mean

**Why 94%?** This is a standard convention in the probabilistic programming community (e.g., *Statistical Rethinking*) to distinguishing it from frequentist 95% confidence intervals and acknowledges the arbitrary nature of the 5% threshold.

**Code reference**: `summary_tables.py:generate_t4_risk_ranking`
