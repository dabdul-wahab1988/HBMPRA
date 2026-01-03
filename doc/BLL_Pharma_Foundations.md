# Blood Lead Level (BLL) Engines: Mathematical Foundations

This document details the mechanistic and empirical models used to predict blood lead levels (BLL) in the `bll_engines.py` and `calibrate_bll_priors.py` modules.

---

## 1. Adult Mechanistic Model (One-Compartment)

Adult lead kinetics are modeled using a steady-state one-compartment pharmacokinetic (PK) model.

### 1.1 Differential Equation

The rate of change of lead concentration in the blood ($C_b$) is the sum of absorption (intake) and elimination:
$$\frac{dC_b}{dt} = \frac{I(t) \cdot f_{abs}}{V_{blood}} - k_{el} \cdot C_b(t)$$
where:

- $I(t)$: Intake rate ($\mu$g/day)
- $f_{abs}$: Fractional absorption (unitless, typically 0.2 for adults)
- $V_{blood}$: Total blood volume (L, approx $0.07 \cdot BW$)
- $k_{el}$: Elimination rate constant (day$^{-1}$)

### 1.2 Steady-State Solution

At steady state ($dC_b/dt = 0$):
$$C_{b,ss} = \frac{I \cdot f_{abs}}{k_{el} \cdot V_{blood}}$$
Since BLL is typically expressed in $\mu$g/dL, and $V_{blood}$ is in L, we apply a factor of 10:
$$BLL_{ss} (\mu\text{g/dL}) = \frac{\text{Intake} (\mu\text{g/day}) \times f_{abs}}{k_{el} (\text{day}^{-1}) \times V_{blood} (\text{L}) \times 10}$$

### 1.3 Relating $k_{el}$ to Half-life

The elimination constant is derived from the biological half-life ($t_{1/2}$):
$$k_{el} = \frac{\ln(2)}{t_{1/2}}$$

---

## 2. Pediatric Empirical Model (Slope Model)

For children, HBMPRA uses a linear slope model based on empirical dose-response data.

### 2.1 The Slope Equation

$$BLL = BLL_{background} + K \cdot (\text{Intake}_{Pb})$$
where:

- $K$: Slope factor ($\mu$g/dL per $\mu$g/day intake). Typical value is ~0.17 for children.
- $f_{abs}$: High absorption fraction for children (~0.5).

---

## 3. Bayesian Prior Calibration Logic (`calibrate_bll_priors.py`)

The framework "calibrates" the Bayesian model by fitting a linear transfer function to the PK results.

### 3.1 Grid-based Simulation

1. **Concentration Grid**: A geometric progression of lead concentrations ($C_w$) is generated between the 5th and 95th percentiles of the site data.
2. **Point Simulation**: For each $C_{w,i}$, a corresponding $BLL_i$ is calculated using the appropriate PK or Slope engine.
3. **Linear Mapping**: A least-squares linear fit is performed:
    $$BLL = b_0 + k_{wb} \cdot EDI + \epsilon$$
    where $EDI$ is the Estimated Daily Intake derived from $C_w$ and $b_0$ is the intercept representing background exposure.

### 3.2 Resultant Prior Parameters

The calibration produces the mean and standard deviation for the intercept ($b_0$) and slope ($k_{wb}$):

- `b0_mu`, `b0_sigma`
- `k_wb_mu`, `k_wb_sigma`

These parameters are passed into the PyMC model, where the Lead BLL is finally calculated as:
$$BLL_{site} = b_0 + k_{wb} \cdot EDI_{site,Pb}$$
This allows the model to propagate all sources of uncertainty (body weight, ingestion rate, concentration) through the PK-calibrated relationship.

---

## 4. Regulatory Thresholds and Reference Values

The analysis compares posterior BLL distributions against established regulatory and health-based benchmarks.

### 4.1 Pediatric Threshold (3.5 $\mu$g/dL)

In 2021, the **CDC** updated its Blood Lead Reference Value (BLRV) from 5.0 to **3.5 $\mu$g/dL**. This value represents the 97.5th percentile of BLLs in U.S. children ages 1–5 years (NHANES). HBMPRA uses this tight threshold for **Children** and **Pregnant** women groups to identify at-risk populations.

### 4.2 Standard Concern Threshold (5.0 $\mu$g/dL)

The previous CDC reference level of **5.0 $\mu$g/dL** is retained as a secondary benchmark for **Teens** and general population screening, providing continuity with historical studies.

### 4.3 Occupational Threshold (10.0 $\mu$g/dL)

For **Adults**, a threshold of **10.0 $\mu$g/dL** is often used as an action level in occupational settings (OSHA/NIOSH), though health effects are known to occur at lower levels.

**Implementation**: Reliability metrics such as $P(BLL > 3.5)$ are automatically computed for relevant demographics in the summary tables (`T5_BLL_summary.csv`).
