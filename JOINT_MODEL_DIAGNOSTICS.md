# nb14: diagnose the saved joint model

Open `notebooks/nb14_joint_model_diagnostics.ipynb` and run all cells in the existing
scientific Python environment. The notebook supports local execution and the same
Google Drive repository layout as nb13. It never trains a model or modifies nb01–13.
Required packages are NumPy, pandas, SciPy, scikit-learn, h5py, anndata, Scanpy,
Matplotlib and Jupyter. scvi-tools is not needed to execute these diagnostics.

The reusable implementation is `src/joint_model_diagnostics.py`. The notebook can
be rebuilt with `scripts/build_nb14.py` (this clears **nb14** outputs) and executed
with `scripts/execute_nb14.py`. Execution saves completed cells even if it fails.

## Inputs and missing-data policy

The required nb13 folder is `results/joint_integration/nb13_run01`, including
`model_input.h5ad` with saved `obsm/X_joint`, evaluation cell IDs, baseline metric
tables and saved UMAP. No weights are loaded. The exact 19,670 labelled evaluation
cells are reused. PCA uses all 141,039 saved cells and evaluates the fixed subset.

The matched broader RNA test requires
`results/shared_rna_integration/run03/shared_rna.h5ad` with raw `layers/counts` and
original composite IDs. The code verifies coverage, ordering and equality of the
994-gene counts before comparing PCA spaces. Do not substitute an old integrated
embedding or infer feature effects from nb10 summary tables: those are not the
controlled feature-panel comparison.

Frozen modules use `results/single_donor/gsea/used_collection.gmt`, its enrichment
table and `results/cross_donor/nb09_run01/protocol.json`. The GMT hash is checked.
Optional original CITE metadata and nb09 cell QC are used when present. Missing
original Multiome labels and full-library QC are explicitly reported; selected
RNA/ATAC counts are labelled as such. Protein/ATAC storage placeholders are never
used as measured zeros in correlations or residualization.

## Outputs

`results/joint_model_diagnostics/` contains baseline verification, cell and group
metrics, coordinate associations, diagnostic representations, UMAP coordinates,
classifier folds, RNA-panel availability, gene retention, local/cross-capture
module gradients, failure classifications, all PNG/PDF figures, a final comparison
table, a hypothesis table, a recommendation and a provenance manifest.

Generated results follow the repository's existing `.gitignore` policy and stay
local. The executed notebook embeds figures, tables and the numerical conclusion.
Outputs in the nb14 folder are replaced when rerunning; upstream artifacts are
read-only. A `partial` status means locally available analyses ran but missing
inputs prevent some requested conclusions. It is not a successful full-RNA test.

## Interpretation

Coordinates are zero-based. The largest CITE protein-depth |rho| chooses the
single deletion; up to three dimensions with |rho|≥0.5 may be removed in a separate
sensitivity test. Residualization subtracts a centered log-depth term only in
CITE donor/type strata with at least 50 cells; group means and all Multiome values
remain unchanged. Neither operation is a validated integration method.

Assay prediction uses leave-one-donor-out logistic regression. Standardization
is learned inside folds; coordinate selection is exploratory on the diagnostic
cohort. Coefficient column numbers in deleted spaces refer to retained-column
order, not original latent-coordinate numbers. No classifier is fitted to the
label-informed residualized space.

Protein depth dependence in the final table is the maximum within-CITE |rho|
across that representation's coordinates. PCA axes and model axes are different
bases; this summary is not a basis-invariant measure of all encoded depth.
Restricted-module proxies cannot establish preservation of full validated programs.
Biology checks use RNA already contributing to the representations and are
descriptive, not independent validation. No conclusion establishes matched cells,
protein–ATAC pairing, validated imputation, or molecular causality.

Run numerical guardrails with:

```text
python -m unittest discover -s tests -p test_joint_model_diagnostics.py
```
