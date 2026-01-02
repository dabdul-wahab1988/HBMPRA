import sys
import json
from pathlib import Path
import pandas as pd

import hbmpra


def write_csv(p: Path, df: pd.DataFrame):
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def test_dermal_hq_basic(monkeypatch, tmp_path):
    chem = pd.DataFrame({'C_As': [50.0], 'C_Cr': [100.0]})
    chem_path = tmp_path / 'chem.csv'
    write_csv(chem_path, chem)

    results_dir = tmp_path / 'results'
    results_dir.mkdir()

    # Ensure no heavy sampling
    monkeypatch.setattr(hbmpra.pm, 'sample', lambda *a, **k: {'trace': 'dummy'})
    monkeypatch.setattr(hbmpra.az, 'to_netcdf', lambda *a, **k: None)
    monkeypatch.setattr(hbmpra.dill, 'dump', lambda *a, **k: None)

    monkeypatch.setattr(sys, 'argv', ['hbmpra.py', '--chemistry', str(chem_path), '--results-dir', str(results_dir)])
    hbmpra.main()

    assumptions_path = results_dir / 'ASSUMPTIONS.json'
    assert assumptions_path.exists()
    ass = json.loads(assumptions_path.read_text())
    assert ass.get('routes', {}).get('dermal_water', False) is True
