import os
import json
import tempfile
import pandas as pd
import numpy as np
import sys

import pytest

# Ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hbmpra


def write_csv(path, df):
    df.to_csv(path, index=False)


def test_phreeqc_frac_integration(monkeypatch, tmp_path):
    # Create minimal chemistry: one site, total Cr = 100 ug/L
    chem = pd.DataFrame({'C_Cr': [100.0]})
    chem_path = tmp_path / 'chem.csv'
    write_csv(chem_path, chem)

    # Create minimal fractions table: 70% CrVI, 30% CrIII
    frac = pd.DataFrame({'frac_CrVI': [0.7], 'frac_CrIII': [0.3]})
    results_dir = tmp_path / 'results'
    results_dir.mkdir()
    frac_path = results_dir / 'table_species_fractions.csv'
    write_csv(frac_path, frac)

    # Monkeypatch heavy operations
    monkeypatch.setattr(hbmpra.pm, 'sample', lambda *a, **k: {'trace': 'dummy'})
    monkeypatch.setattr(hbmpra.az, 'to_netcdf', lambda *a, **k: None)
    monkeypatch.setattr(hbmpra.dill, 'dump', lambda *a, **k: None)

    # Run main
    monkeypatch.setattr(sys, 'argv', ['hbmpra.py', '--chemistry', str(chem_path), '--results-dir', str(results_dir)])
    hbmpra.main()

    # Check ASSUMPTIONS.json and RUNLOG
    assumptions_path = results_dir / 'ASSUMPTIONS.json'
    assert assumptions_path.exists()
    assumptions = json.loads(assumptions_path.read_text())
    assert assumptions.get('speciation_source') == 'phreeqc_frac'

    runlog = json.loads((results_dir / 'RUNLOG.json').read_text())
    assert runlog.get('speciation_source') == 'phreeqc_frac'


if __name__ == '__main__':
    pytest.main([__file__])
