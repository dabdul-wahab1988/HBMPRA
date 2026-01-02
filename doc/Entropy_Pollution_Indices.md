# Entropy-Based Pollution Indices: Mathematical Foundations

This document provides complete mathematical derivations for the entropy-weighted Heavy Metal Pollution Index (HPI) and Potential Ecological Risk Index (PERI) implemented in `entropy_hpi_peri.py`.

---

## Table of Contents

1. [Shannon Entropy and Information Theory](#1-shannon-entropy-and-information-theory)
2. [Entropy Weight Calculation](#2-entropy-weight-calculation)
3. [Heavy Metal Pollution Index (HPI)](#3-heavy-metal-pollution-index-hpi)
4. [Potential Ecological Risk Index (PERI)](#4-potential-ecological-risk-index-peri)
5. [Uncertainty Quantification via Bootstrap](#5-uncertainty-quantification-via-bootstrap)
6. [Risk Categorization](#6-risk-categorization)

---

## 1. Shannon Entropy and Information Theory

Shannon entropy quantifies the "information content" or "disorder" in a probability distribution.

### 1.1 Definition

For a discrete probability distribution $\{p_1, p_2, \dots, p_n\}$:
$$H = -\sum_{i=1}^n p_i \ln p_i$$

**Properties**:

- $0 \leq H \leq \ln n$ (maximum when uniform: $p_i = 1/n$)
- $H = 0$ when one $p_i = 1$ and others are zero (complete certainty)

### 1.2 Normalized Entropy

To ensure $0 \leq e_j \leq 1$:
$$e_j = -\frac{1}{\ln n} \sum_{i=1}^n p_{ij} \ln p_{ij}$$

where $n$ is the number of sites and $p_{ij}$ is the probability that site $i$ contributes to metal $j$'s total load.

**Code reference**: `entropy_hpi_peri.py`, lines 176-177.

---

## 2. Entropy Weight Calculation

### 2.1 Probability Calculation

For metal $j$ across $n$ sites:
$$p_{ij} = \frac{C_{ij}}{\sum_{i=1}^n C_{ij}}$$

where $C_{ij}$ is the concentration of metal $j$ at site $i$.

**Handling zeros**: To prevent $\ln(0)$, concentrations below detection are replaced with $0.01 \times DL_j$ (1% of detection limit).

### 2.2 Divergence (Information Content)

$$d_j = 1 - e_j$$

**Interpretation**:

- High $e_j$ (near 1) means concentrations are uniformly distributed → low information → low divergence
- Low $e_j$ (near 0) means concentrations are spatially heterogeneous → high information → high divergence

### 2.3 Weight Normalization

$$w_j = \frac{d_j}{\sum_{k=1}^m d_k}$$

where $m$ is the total number of metals.

**Constraint**: $\sum_{j=1}^m w_j = 1$

**Robustness**: A minimum weight constraint ($w_{min} = 0.01$) prevents extreme weights:
$$w_j^{constrained} = \max(w_j, w_{min})$$
followed by renormalization.

**Code reference**: `entropy_hpi_peri.py`, lines 196-206.

---

## 3. Heavy Metal Pollution Index (HPI)

### 3.1 Contamination Factor

For each metal at each site:
$$CF_{ij} = \frac{C_{ij}}{S_j}$$

where $S_j$ is the regulatory standard (e.g., WHO guideline) for metal $j$.

### 3.2 HPI Calculation

$$HPI_i = \sum_{j=1}^m w_j \cdot CF_{ij}$$

**Alternative formulation** (equivalent):
$$HPI = \frac{\sum_{j=1}^m w_j \cdot Q_j}{\sum_{j=1}^m w_j}, \quad Q_j = 100 \cdot \frac{C_j}{S_j}$$

### 3.3 Risk Categories

| HPI Range | Risk Level |
|-----------|-----------|
| 0 - 30 | Low |
| 30 - 60 | Moderate |
| 60 - 100 | High |
| > 100 | Very High |

**Code reference**: `entropy_hpi_peri.py`, lines 257-268.

---

## 4. Potential Ecological Risk Index (PERI)

### 4.1 Toxic Response Factor

Each metal has a toxicity coefficient $T_j$ (Hakanson, 1980):

| Metal | $T_j$ |
|-------|-------|
| As | 10 |
| Cd | 30 |
| Cr | 2 |
| Cu | 5 |
| Hg | 40 |
| Pb | 5 |
| Zn | 1 |

### 4.2 PERI Calculation

$$PERI_i = \sum_{j=1}^m w_j \cdot T_j \cdot CF_{ij}$$

**Interpretation**: PERI weights metals not only by their spatial variability (entropy weight) but also by their inherent toxicity.

### 4.3 Ecological Risk Categories

| PERI Range | Risk Level |
|------------|-----------|
| < 150 | Low |
| 150 - 300 | Moderate |
| 300 - 600 | Considerable |
| > 600 | Very High |

**Code reference**: `entropy_hpi_peri.py`, lines 270-280.

---

## 5. Uncertainty Quantification via Bootstrap

### 5.1 Bootstrap Resampling

To estimate uncertainty in weights and indices:

1. Resample $n$ sites with replacement
2. Recompute entropy weights on the bootstrap sample
3. Recalculate HPI and PERI
4. Repeat $B$ times (typically $B = 1000$)

### 5.2 Confidence Intervals

The 95% confidence interval is computed from the bootstrap distribution:
$$CI_{95\%} = [P_{2.5}, P_{97.5}]$$

where $P_\alpha$ is the $\alpha$-th percentile of the bootstrap distribution.

**Mathematical Justification**: By the Bootstrap Theorem (Efron & Tibshirani, 1993), for sufficiently large $n$ and $B$:
$$\hat{F}^*_n \xrightarrow{d} F$$
where $\hat{F}^*_n$ is the empirical bootstrap distribution and $F$ is the true sampling distribution.

**Code reference**: `entropy_hpi_peri.py`, lines 283-320.

---

## 6. Risk Categorization

### 6.1 Categorical Assignment

For each site $i$:
$$\text{Risk Level}_i = \begin{cases}
\text{Low} & \text{if } HPI_i < 30 \\
\text{Moderate} & \text{if } 30 \leq HPI_i < 60 \\
\text{High} & \text{if } 60 \leq HPI_i < 100 \\
\text{Very High} & \text{if } HPI_i \geq 100
\end{cases}$$

### 6.2 Summary Statistics
The framework computes:
- **Mean HPI/PERI** across sites
- **Range** (min, max)
- **Fraction of sites** in each risk category
- **Priority sites** (those exceeding critical thresholds)

**Code reference**: `entropy_hpi_peri.py`, lines 322-332, 525-621.

---

## Mathematical Validation

### 6.3 Weight Sum Theorem
**Theorem**: The entropy weights always sum to unity.
$$\sum_{j=1}^m w_j = \sum_{j=1}^m \frac{d_j}{\sum_{k=1}^m d_k} = \frac{\sum_{j=1}^m d_j}{\sum_{k=1}^m d_k} = 1$$

**Proof**: Direct from definition of normalization.

### 6.4 Bounded Indices
**Theorem**: If all $CF_{ij} \geq 0$, then $HPI_i \geq 0$ and $PERI_i \geq 0$.

**Proof**:
$$HPI_i = \sum_{j=1}^m w_j \cdot CF_{ij} \geq 0$$
since $w_j \geq 0$ and $CF_{ij} \geq 0$ by construction.

---

## Comparison with Alternative Methods

| Method | Weighting | Strengths | Limitations |
|--------|-----------|-----------|-------------|
| **Entropy** | Data-driven spatial variance | Objective, site-specific | Requires sufficient samples |
| **Equal** | $w_j = 1/m$ | Simple | Ignores metal importance |
| **Expert** | Subjective scales | Incorporates knowledge | Subjective, not reproducible |
| **PCA** | Principal components | Captures correlations | Complex interpretation |

**HBMPRA Choice**: Entropy weighting provides an objective, mathematically rigorous approach that balances between equal weighting and subjective expert judgment.

---

## References

1. Hakanson, L. (1980). An ecological risk index for aquatic pollution control. *Water Research*, 14(8), 975-1001.
2. Shannon, C.E. (1948). A Mathematical Theory of Communication. *Bell System Technical Journal*, 27(3), 379-423.
3. Li, P., et al. (2014). Groundwater quality assessment based on improved water quality index. *Environmental Monitoring and Assessment*, 186(5), 3077-3091.
4. Efron, B., & Tibshirani, R.J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall/CRC.
5. Wu, J., et al. (2020). Health risk assessment of groundwater contamination using entropy weight water quality index. *Environmental Science and Pollution Research*, 27(18), 22201-22212.
