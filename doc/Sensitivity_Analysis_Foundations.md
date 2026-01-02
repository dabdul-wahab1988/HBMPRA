# Sensitivity Analysis: Mathematical Foundations (Comprehensive)

This document describes the mathematical methods used in the `sensitivity_analysis.py` module to quantify the influence of model parameters on risk predictions.

---

## Table of Contents

1. [Variance-Based Global Sensitivity Analysis (Sobol)](#1-variance-based-global-sensitivity-analysis-sobol)
2. [Morris Screening (Elementary Effects)](#2-morris-screening-elementary-effects)
3. [Delta Moment-Independent Method](#3-delta-moment-independent-method)
4. [Parameter Space Construction](#4-parameter-space-construction)
5. [Batch Model Evaluation](#5-batch-model-evaluation)

---

## 1. Variance-Based Global Sensitivity Analysis (Sobol Method)

The Sobol method decomposes the variance of the model output $V(Y)$ into contributions from individual parameters and their interactions.

### 1.1 ANOVA-like Decomposition

Given a model $Y = f(X_1, X_2, \dots, X_k)$, the variance is decomposed as:
$$V(Y) = \sum_{i} V_i + \sum_{i<j} V_{ij} + \dots + V_{12\dots k}$$
where:

- $V_i = V_{X_i}(E_{X_{\sim i}}[Y | X_i])$ (First-order effect)
- $V_{ij} = V_{X_i, X_j}(E_{X_{\sim ij}}[Y | X_i, X_j]) - V_i - V_j$ (Second-order interaction)

### 1.2 Sensitivity Indices

- **First-order Index ($S_i$):** Measures the fraction of output variance caused by parameter $X_i$ alone.
  $$S_i = \frac{V_i}{V(Y)} = \frac{V_{X_i}(E_{X_{\sim i}}[Y | X_i])}{V(Y)}$$
  
- **Total-effect Index ($S_{Ti}$):** Measures the total contribution of $X_i$ to output variance, including all interactions.
  $$S_{Ti} = 1 - \frac{V_{X_{\sim i}}(E_{X_i}[Y | X_{\sim i}])}{V(Y)} = \frac{E_{X_{\sim i}}[V_{X_i}(Y | X_{\sim i})]}{V(Y)}$$

**Mathematical Proof of the Total Effect:**
The total effect $S_{Ti}$ can be shown to equal:
$$S_{Ti} = S_i + \sum_{j \neq i} S_{ij} + \sum_{j \neq i, k \neq i, j < k} S_{ijk} + \dots$$

Thus, if $S_{Ti} \approx 0$, the parameter $X_i$ is **non-influential** and can be fixed to any value within its range without affecting the output variance.

**Convergence Property**:
$$\sum_{i=1}^k S_i \leq 1, \quad \sum_{i=1}^k S_{Ti} \geq 1$$

### 1.3 Saltelli Sampling Scheme

To estimate these indices efficiently, the Saltelli scheme generates $N(2k + 2)$ model evaluations where $N$ is the base sample size and $k$ is the number of parameters.

**Code reference**: `sensitivity_analysis.py`, lines 630-669.

---

## 2. Morris Screening (Elementary Effects)

Designed for high-dimensional models where Sobol is computationally prohibitive. Morris screening is a **one-at-a-time** (OAT) method that uses randomized trajectories.

### 2.1 Elementary Effect ($EE_i$)

For a parameter $X_i$ changed by a step $\Delta$:
$$EE_i(X) = \frac{f(X_1, \dots, X_i + \Delta, \dots, X_k) - f(X)}{\Delta}$$

### 2.2 Statistics

For $r$ random trajectories, compute:

- **$\mu_i^*$ (Mean of $|EE_i|$):** Measures the overall importance of the parameter.
  $$\mu_i^* = \frac{1}{r} \sum_{j=1}^r |EE_i^{(j)}|$$
  
- **$\sigma_i$ (Standard Deviation of $EE_i$):** Measures non-linearity and interaction effects.
  $$\sigma_i = \sqrt{\frac{1}{r-1} \sum_{j=1}^r (EE_i^{(j)} - \mu_i)^2}$$

**Interpretation**:

- High $\mu_i^*$ + Low $\sigma_i$: Parameter has linear, additive effect
- High $\mu_i^*$ + High $\sigma_i$: Parameter has non-linear effect or strong interactions
- Low $\mu_i^*$: Parameter is non-influential

**Code reference**: `sensitivity_analysis.py`, lines 671-697.

---

## 3. Delta Moment-Independent Method

Focuses on the entire output distribution rather than just the variance. Useful when the output is highly skewed or multi-modal.

### 3.1 Delta Index

The sensitivity measure $\delta_i$ is defined based on the $L_1$ distance between the unconditional and conditional probability density functions (PDFs):
$$\delta_i = \frac{1}{2} E_{X_i} \left[ \int |f_Y(y) - f_{Y|X_i}(y|x_i)| dy \right]$$

**Properties**:

- $0 \leq \delta_i \leq 1$
- $\delta_i = 0$ implies $Y$ is independent of $X_i$
- $\delta_i = 1$ implies $Y$ is fully determined by $X_i$

### 3.2 Relationship to Sobol Indices

Under certain conditions (Gaussian outputs):
$$\delta_i \approx \frac{\sqrt{2}}{\pi} S_i$$

**Code reference**: `sensitivity_analysis.py`, lines 699-725.

---

## 4. Parameter Space Construction

### 4.1 Log-Normal Parameter Bounds

Parameters like **Ingestion Rate ($IR$)** and **Body Weight ($BW$)** are sampled using log-normal priors:
$$X \sim \text{LogNormal}(\mu, \sigma^2)$$
$$\ln X \sim \text{Normal}\left(\ln(\text{Median}), \sigma_{log}\right)$$

For sensitivity analysis, these are mapped to bounded ranges representing approximate 95% confidence intervals:
$$[\text{Median} \cdot e^{-2\sigma}, \text{Median} \cdot e^{+2\sigma}]$$

**Example (Body Weight with CV=0.21)**:
$$\sigma_{log} = \sqrt{\ln(1 + CV^2)} = \sqrt{\ln(1.0441)} \approx 0.208$$
$$\text{Lower bound} = BW_{mean} \cdot (1 - 2 \cdot 0.21) = BW_{mean} \cdot 0.58$$
$$\text{Upper bound} = BW_{mean} \cdot (1 + 2 \cdot 0.21) = BW_{mean} \cdot 1.42$$

### 4.2 Concentration Parameter Bounds

Metal concentrations are varied logarithmically around the site mean:
$$[0.1 \cdot C_{mean}, 10 \cdot C_{mean}]$$

This captures both low (background) and high (contaminated) scenarios.

### 4.3 Toxicity Parameter Bounds

When `--include-tox-uncertainty` is enabled:

- **RfD (Reference Dose)**: $\pm 50\%$ around baseline
- **CSF (Cancer Slope Factor)**: $\pm 50\%$ around baseline  
- **ABS_GI (Absorption)**: $\pm 30\%$ around baseline (constrained to [0.01, 1.0])

**Code reference**: `sensitivity_analysis.py`, lines 165-354.

---

## 5. Batch Model Evaluation

### 5.1 Vectorized Dose Calculation

For computational efficiency, the sensitivity analyzer computes outputs for all parameter samples in batch:

**Ingestion EDI** (vectorized over samples):
$$EDI_{ing}^{(n)} = \frac{C^{(n)} \cdot ABS_{GI}^{(n)} \cdot IR^{(n)} \cdot EF^{(n)} \cdot ED^{(n)}}{BW^{(n)} \cdot AT^{(n)}}$$

**Dermal EDI** (vectorized):
$$EDI_{der}^{(n)} = \frac{C_{bio}^{(n)} \cdot K_p^{(n)} \cdot ET^{(n)} \cdot SA^{(n)} \cdot EF^{(n)} \cdot ED^{(n)} \cdot 10^{-3}}{BW^{(n)} \cdot AT^{(n)}}$$

where $n$ indexes the parameter sample.

### 5.2 Organ-Specific HI Aggregation

For each organ system $o$:
$$HI_o^{(n)} = \sum_{m \in S_o} \left( \frac{EDI_{m,ing}^{(n)}}{RfD_{m,oral}^{(n)}} + \frac{EDI_{m,der}^{(n)}}{RfD_{m,derm}^{(n)}} \right)$$

The overall output for sensitivity analysis is:
$$Y_{HI}^{(n)} = \text{mean across groups}\left(\max_{o} HI_o^{(n)}\right)$$

**Code reference**: `sensitivity_analysis.py`, lines 357-591.

---

## Convergence and Uncertainty

### Sobol Convergence

The standard error of Sobol indices decreases as:
$$SE(S_i) \propto \frac{1}{\sqrt{N}}$$

For publication-quality results, $N \geq 1024$ (giving $N(2k+2) \approx 200k$ evaluations for $k=100$ parameters).

### Bootstrap Confidence Intervals

All sensitivity indices are reported with 95% confidence intervals computed via bootstrap resampling.

---

## References

1. Saltelli, A., et al. (2010). Variance based sensitivity analysis of model output. *Computer Physics Communications*, 181(2), 259-270.
2. Morris, M.D. (1991). Factorial sampling plans for preliminary computational experiments. *Technometrics*, 33(2), 161-174.
3. Borgonovo, E. (2007). A new uncertainty importance measure. *Reliability Engineering & System Safety*, 92(6), 771-784.
4. Sobol', I.M. (2001). Global sensitivity indices for nonlinear mathematical models. *Mathematics and Computers in Simulation*, 55(1-3), 271-280.
