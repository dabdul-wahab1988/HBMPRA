import numpy as np
from hbmpra_optimized import validate_dermal_requirements
import yaml

# Load toxref and construct arrays consistent with script expectations
loaded = yaml.safe_load(open('external/toxref.yml')) or {}
tox = loaded.get('tox', {})
metals = sorted(list(tox.keys()))

# Mock C_bio_mgL: one positive entry for each metal to simulate dermal availability
C_bio_mgL = np.ones((1, len(metals)))  # shape (J=1, M)

RFDS_derm = np.array([tox.get(m, {}).get('RfD_derm', np.nan) for m in metals], float)

# This should not raise under strict mode because we expect RfD_derm for dermal metals
validate_dermal_requirements(metals, C_bio_mgL, RFDS_derm)
