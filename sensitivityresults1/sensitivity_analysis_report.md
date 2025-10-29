# Sensitivity Analysis Report

**Analysis Method:** SOBOL
**Date:** 2025-10-02 06:57:32
**Samples:** 1000
**Parameters:** 56

## Key Findings

### HI - Most Influential Parameters

| Parameter | Total Effect | Confidence |
|-----------|--------------|------------|
| C_As | 0.579664 | 0.083788 |
| IR_Children | 0.154155 | 0.030488 |
| AT_nc_Children | 0.120387 | 0.031268 |
| ED_Children | 0.090846 | 0.016639 |
| BW_Children | 0.053075 | 0.012377 |
| IR_Teens | 0.030923 | 0.007080 |
| IR_Adults | 0.029711 | 0.006536 |
| EF_Children | 0.029234 | 0.008026 |
| AT_nc_Teens | 0.023297 | 0.005290 |
| ED_Teens | 0.020972 | 0.005069 |

### CR - Most Influential Parameters

| Parameter | Total Effect | Confidence |
|-----------|--------------|------------|
| C_As | 0.547077 | 0.077137 |
| IR_Adults | 0.149727 | 0.028696 |
| AT_ca_Adults | 0.122411 | 0.023192 |
| ED_Adults | 0.100028 | 0.022684 |
| BW_Adults | 0.056047 | 0.013800 |
| IR_Children | 0.034954 | 0.007949 |
| EF_Adults | 0.033143 | 0.011382 |
| AT_ca_Children | 0.030456 | 0.008662 |
| IR_Teens | 0.028260 | 0.005381 |
| AT_ca_Teens | 0.027803 | 0.008116 |

## Methodology

This analysis used comprehensive parameter bounds derived from:
- Scientific literature and regulatory guidelines
- Default uncertainty assumptions (CV = 0.21 for body weight)
- Log-normal distributions for ingestion rate uncertainty
- Conservative bounds for concentration variability

## Recommendations

1. Focus monitoring efforts on the most influential parameters
2. Consider parameter interactions in risk management decisions
3. Validate model assumptions with site-specific data
4. Use uncertainty bounds for decision-making under uncertainty

## Files Generated

- `*_indices.csv`: Sensitivity indices for each parameter
- `*_tornado.png`: Tornado plots showing parameter influence
- `parameter_effects_*.png`: Scatter plots of parameter effects
- `analysis_metadata.json`: Complete analysis metadata
- `sensitivity_analysis.log`: Detailed execution log

