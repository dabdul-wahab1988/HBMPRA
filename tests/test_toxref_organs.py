import os
import json
import numpy as np
from hbmpra import build_organ_sets


def test_systemic_mask_from_toxref():
    metals = ["As", "Cd", "Cu"]
    # toxref with As targeting 'hepato' and 'systemic', Cd targeting 'nephro'
    toxref = {
        "As": {"target_organs": ["hepato", "systemic"]},
        "Cd": {"target_organs": ["nephro"]},
    }
    organ_sets_used, systemic_mask_np, unknown_orgs = build_organ_sets(metals, toxref)
    # As should be marked systemic (first metal)
    assert systemic_mask_np[0] == 1.0
    # Cd and Cu not systemic
    assert systemic_mask_np[1] == 0.0
    assert systemic_mask_np[2] == 0.0
    # organ_sets should include 'systemic' with As
    assert 'systemic' in organ_sets_used
    assert 'As' in organ_sets_used['systemic']
    # No unknown organ names expected
    assert unknown_orgs == set()
