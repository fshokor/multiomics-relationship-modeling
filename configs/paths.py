"""
Central path config — edit DATA_DIR to match where your h5ad files live.
Import this everywhere instead of hardcoding paths.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Raw data ──────────────────────────────────────────────────
DATA_DIR = REPO_ROOT / "data" / "benchmark"

CITE_H5AD     = DATA_DIR / "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad"
MULTIOME_H5AD = DATA_DIR / "GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad"

# ── Outputs ───────────────────────────────────────────────────
RESULTS_DIR  = REPO_ROOT / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"
TABLES_DIR   = RESULTS_DIR / "tables"
