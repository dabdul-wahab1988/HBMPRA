# HBMPRA

Human Bioavailability Model / Probabilistic Risk Assessment (HBMPRA)

This repository contains code, data, and results for the HBMPRA project. The project contains scripts for exposure/speciation modeling, sensitivity analysis, plotting, and result summaries. Key scripts are Python files in the repository root and helper modules in the same tree.

## Highlights
- Core scripts: `hbmpra_optimized.py`, `bll_engines.py`, `speciation_modeling.py`, `sensitivity_analysis.py`
- Data and lookup tables: `database/`, `external/`
- Results and figures: `results/` (includes precomputed outputs used for reporting)
- Tests: `tests/` (pytest-compatible test files)

## Quick start

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies. This project does not contain a pinned `requirements.txt` in the repo; identify and install the packages you need (typical packages used by scientific Python projects include `numpy`, `pandas`, `scipy`, `matplotlib`, `pytest`). Example:

```powershell
pip install numpy pandas scipy matplotlib pytest
```

3. Run a script, for example:

```powershell
python hbmpra_optimized.py
```

Outputs will be written into the `results/` directory.

## Tests

Run the test suite with pytest:

```powershell
pip install pytest
pytest -q
```

## Repository notes & recommended housekeeping

- Large binary and data files are included in this repository (files under `database/`, `results/`, and several images and model artifacts). That made the initial push large. Consider using Git LFS for large files to avoid bloating the Git history:

  - Install Git LFS locally and configure this repo (example):

    ```powershell
    git lfs install
    git lfs track "*.dat" "*.pkl" "*.png" "*.jpg"
    git add .gitattributes
    git commit -m "Track large binary files with Git LFS"
    ```

- Add a `.gitignore` to stop committing transient files (here's a recommended starter):

```
# Python
__pycache__/
*.py[cod]
*.so

# Environments
.venv/
env/

# Jupyter
*.ipynb_checkpoints

# OS
.DS_Store
Thumbs.db

# Data/results that you may want to keep out of the repo
results/
database/*.dat
```

If you'd like, I can add a `.gitignore` file and remove `__pycache__`/`.pyc` files from the repository in a follow-up commit.

## Data licensing & confidentiality

Some data files may contain licensed or sensitive data. Before sharing or publishing, confirm licensing and privacy constraints for the files in `database/`, `external/`, and `results/`.

## License

Please add a `LICENSE` file to indicate a license for this project (for example, MIT, Apache-2.0). If you want, I can add an `MIT` license file in a follow-up change.

## Contact / next steps

If you'd like, I can also:

- Add a `.gitignore` and remove committed `__pycache__` / `.pyc` files.
- Add/enable Git LFS and migrate large files.
- Add a `requirements.txt` or `environment.yml` with pinned package versions.

Open an issue or reply here with which of those follow-ups you'd like me to perform next.

---
_Generated and added to the repository by an assistant on request._
