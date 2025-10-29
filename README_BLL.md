Calibration and BLL engines

This repository has migrated away from the external AALM binary for runtime BLL calibration
and now uses internal pure-Python engines (`bll_engines.py`). The calibrator writes priors to
`results/calibration/priors.json` by default. Historical AALM logs and priors that were previously
committed in `results/aalm_calibration/` have been removed from the repository.

What's changed
- `calibrate_bll_priors.py` now uses internal BLL engines and accepts `--bll-engine` and
	engine-parameter flags (see the script for details).
- The calibrator writes `engine_metadata` into the priors JSON and emits both canonical
	(`b0_mu`, `b0_sigma`, `k_wb_mu`, `k_wb_sigma`) and legacy keys (`b0_mean`, `b0_sd`, `k_mean`, `k_sd`)
	to preserve backward compatibility.

How to generate priors (example)

python calibrate_bll_priors.py --bll-engine onecomp --f_abs 0.5 --t_half_days 30 --results-dir results

If you rely on older AALM artifacts, back them up outside this repository before cleanup.

Migration: toxref organ_sets are now authoritative (Option A)
----------------------------------------------------------

To improve accuracy and avoid silent mismatches between toxicity metadata and the model, `hbmpra.py`
now requires that the external toxref YAML (`external/toxref.yml`) includes a top-level `organ_sets` mapping.
This file is now the authoritative source of organ-to-metal groupings used when computing organ-specific HIs.

Example `external/toxref.yml` snippet:

```yaml
organ_sets:
  neuro: [Mn]
  nephro: [Cd, Hg, CrVI]
  hepato: [As, Cd, Cu, CrVI]

tox:
  As:
    RfD_oral: 0.0003
    SF_oral: 1.5
    ABS_GI: 1.0
    target_organs: [hepato, systemic]
  # ... other tox entries ...
```

If `organ_sets` is missing, `hbmpra.py` will error by default to prevent silent assumptions. To allow a
soft migration you can run with the legacy fallback flag:

```
python hbmpra.py --chemistry my.csv --results-dir results --allow-default-organ-sets
```

The `--allow-default-organ-sets` flag permits falling back to the built-in canonical mapping for
legacy or exploratory runs but logs a warning. For production analyses, add `organ_sets` to
`external/toxref.yml` and consider removing the fallback flag to enforce strict provenance.
