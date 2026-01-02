import numpy as np
import pandas as pd
import importlib.util
import sys
from pathlib import Path

# Load hbmpra module by file path to avoid import issues when sys.path differs
root = Path(__file__).resolve().parent.parent
# Ensure repo root is on sys.path so imports like `demographics` resolve
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
import hbmpra


def test_censored_imputation_simple():
    # Create synthetic lognormal data
    rng = np.random.default_rng(123)
    mu = np.log(10.0)
    sigma = 0.5
    n = 50
    vals = rng.lognormal(mean=mu, sigma=sigma, size=n)

    # Choose LODs that censor roughly half the samples
    lod = 8.0
    censored = vals.copy()
    mask = censored < lod
    censored[mask] = 0.0  # nondetects represented as zeros

    df = pd.DataFrame({
        'C_TestMetal': censored,
        'LOD_TestMetal': np.full(n, lod)
    })

    # Run imputation
    out = hbmpra.impute_censored_df(df.copy(), metals=['TestMetal'], seed=123)

    # For each imputed entry, compute expected conditional mean under original mu/sigma
    from math import log
    import scipy.stats as sps

    t = np.log(lod + 1e-12)
    num = np.exp(mu + 0.5 * sigma ** 2) * sps.norm.cdf((t - mu - sigma ** 2) / sigma)
    denom = sps.norm.cdf((t - mu) / sigma)
    expected_cond_mean = float(num / denom)

    imputed = out['C_TestMetal'].to_numpy()
    # Where original was nondetect, ensure imputed approx equals expected_cond_mean
    for i in range(n):
        if mask[i]:
            assert abs(imputed[i] - expected_cond_mean) / expected_cond_mean < 0.5
    # At least one imputation occurred
    assert mask.sum() > 0
