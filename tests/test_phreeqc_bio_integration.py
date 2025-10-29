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


def test_phreeqc_bio_integration_monkeypatch(monkeypatch, tmp_path):
    # Create minimal chemistry: one site, total Cr = 100 ug/L
    chem = pd.DataFrame({'C_Cr': [100.0], 'pH': [7.5], 'Eh': [100]})
    chem_path = tmp_path / 'chem.csv'
    write_csv(chem_path, chem)

    # Create minimal PHREEQC bioavailable table: mol/L values for CrVI and CrIII
    # Use small mol/L values such that conversion is easy to check
    # For Cr (atomic weight 52), 1e-6 mol/L -> 52e-6 g/L -> 52e-6 * 1e6 = 52 ug/L
    bio = pd.DataFrame({'C_bio_CrVI': [1e-6], 'C_bio_CrIII': [1e-6]})
    results_dir = tmp_path / 'results'
    results_dir.mkdir()
    bio_path = results_dir / 'table_bioavailable_concentrations.csv'
    write_csv(bio_path, bio)

    # Monkeypatch heavy operations: pm.sample, az.to_netcdf, dill.dump
    monkeypatch.setattr(hbmpra.pm, 'sample', lambda *a, **k: {'trace': 'dummy'})
    monkeypatch.setattr(hbmpra.az, 'to_netcdf', lambda *a, **k: None)
    monkeypatch.setattr(hbmpra.dill, 'dump', lambda *a, **k: None)

    # Run main with args pointing to our temp files
    monkeypatch.setattr(sys, 'argv', ['hbmpra.py', '--chemistry', str(chem_path), '--results-dir', str(results_dir)])
    hbmpra.main()

    # Check ASSUMPTIONS.json in results_dir
    assumptions_path = results_dir / 'ASSUMPTIONS.json'
    assert assumptions_path.exists()
    assumptions = json.loads(assumptions_path.read_text())
    assert assumptions.get('speciation_source') == 'phreeqc_bio'

    # Check that RUNLOG.json records the phreeqc_bio path
    runlog = json.loads((results_dir / 'RUNLOG.json').read_text())
    assert runlog.get('speciation_source') == 'phreeqc_bio'


if __name__ == '__main__':
    pytest.main([__file__])
