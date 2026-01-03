# Speciation and Metal Chemistry Implementation

This document describes how `speciation_modeling.py` handles complex metal chemistry through PHREEQC and empirical logic.

---

## 1. PHREEQC Integration

The module attempts to find a PHREEQC backend in the following priority:

1. **`phreeqpython`**: Preferred modern interface.
2. **`phreeqpy` (DLL/COM)**: Standard Python wrapper.
3. **Simplified Fallback**: Empirical estimates if no thermodynamic engine is installed.

### 1.1 Input Processing

Total concentrations are converted from $\mu g/L$ to $mmol/kgw$ for the PHREEQC engine using the atomic weights defined in the `ATOMIC_WEIGHTS` dictionary.

---

## 2. Advanced Multi-Species Logic

For metals where the total concentration ($C_{total}$) is split into multiple oxidation states or chemical forms, the code avoids double-counting by:

1. **Calculating Fractions**: If PHREEQC is used, it calculates the ratio of the toxic species to the total element.
2. **Zeroing the Base**: When specific species (like `CrVI` and `CrIII`) are extracted into separate columns, the original "Total Cr" column is zeroed in the risk assessment phase to ensure HIs are calculated based on the more toxic refined species only.
3. **Missing Totals Guard** (NEW): If a metal has no total concentration column, speciation for that metal is skipped gracefully; other metals continue. This avoids fallback KeyErrors and keeps output tables aligned with provided analytes.

### 2.1 Chromium (Cr)

* **CrVI (Hexavalent)**: Mapped to the `CrO4-2` species. It is linked to the Cancer Risk ($CR$) and Nephrotoxicity organ sets.
* **CrIII (Trivalent)**: Mapped to the `Cr+3` species. Generally considered non-carcinogenic and addressed for GI/systemic risk.

### 2.2 Mercury (Hg)

* **Hg(II)**: Mapped to the `Hg+2` species. Primary indicator for inorganic mercury toxicity.
* **Hg(0)**: Mapped to the elemental `Hg` species.
* **Note on MeHg**: The code explicitly excludes Methylmercury (MeHg) from thermodynamic modeling because it is formed via biological methylation (bacterial action) and cannot be predicted by abiotic thermodynamic equilibrium.

---

## 3. Calculation of Dermal Kp

The dermal permeability coefficients ($K_p$) are resolved using the `external/dermal_water_kp.yml` file. If a metal lacks a $K_p$, the code raises a validation error to prevent inaccurate "zero-dermal-risk" assumptions in scientific studies.

---

## 4. Anion Handling and Anion-Only Runs (Context)

- Fluoride and nitrate are validated alongside metals; nitrate is converted to NO₃–N basis before HQ/HI.
- If the dataset contains only anions, speciation and BLL calibration are skipped while anion HQ/HI are still computed.
