"""
configs/paths.py
-----------------
Central path config. Import BASE_PATH-derived constants everywhere
instead of hardcoding paths. Matches the Colab/local detection pattern
used in the GDSC2 repo.

NOTE: this file does the detection automatically when imported, but
notebooks should ALSO run the explicit Colab-setup cell at the top
(mount Drive, pip install, etc.) — see notebooks/nb01_qc.ipynb.
"""
from pathlib import Path

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    BASE_PATH = Path('/content/drive/MyDrive/multiomics-relationship-modeling')
else:
    BASE_PATH = Path(__file__).resolve().parent.parent

# ── Raw data ──────────────────────────────────────────────────
DATA_DIR = BASE_PATH / "data" / "benchmark"

CITE_H5AD     = DATA_DIR / "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad"
MULTIOME_H5AD = DATA_DIR / "GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad"

# ── Outputs ───────────────────────────────────────────────────
RESULTS_DIR = BASE_PATH / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR  = RESULTS_DIR / "tables"

for d in (FIGURES_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)
