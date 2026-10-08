# nb10 — shared RNA bridge evaluation

Copy the new notebook and three modules to the same locations in Drive:

- `notebooks/nb10_shared_rna_integration.ipynb`
- `src/rna_integration.py`
- `src/rna_integration_metrics.py`
- `src/rna_integration_plots.py`

nb01–09 are preserved. Existing shared helper modules remain dependencies. Run in
a fresh Colab runtime, preferably with GPU and enough RAM for full shared RNA plus
the HVG training subset. The first cell installs the integration dependencies;
the notebook then copies the two input h5ad files to local Colab disk. Training and
input hashing can take time. All results go to Drive.

## Required inputs

Both benchmark h5ad files; completed `results/cross_donor/nb09_run01/` metadata
(`status.json`, `manifest.json`, `protocol.json`, `donor_eligibility.csv`,
`mapping_audit.csv`); and the discovery GMT in `results/single_donor/gsea/`.
The nb09 manifests authenticate the cohort, mapping and source hashes. No fallback
silently invents a cohort or redefines cell-type harmonization. The validated eight
donors are loaded; unresolved labels stay in training but not labelled evaluation.

## Analysis choices

- GEX only; shared gene intersection before CP10k/log1p. Source .X is sampled for
  diagnostics and never assumed comparable. Counts are validated across all cells.
- Up to 3,000 HVGs, equal-size assay samples, Scanpy assay-aware dispersion selection.
- Baseline: centered PCA on log RNA, neighbors and UMAP. No gene-wise variance scaling.
- Primary integration: unsupervised scVI with raw HVG counts, assay batch only,
  NB likelihood, 30 latent dimensions, two layers, maximum 100 epochs, early stopping.
- No ADT/ATAC, donor/site correction or cell-type labels enter model training.
- Biological labels evaluate output only. Same evaluation cells in both spaces:
  <=20,000 donor/assay quota-sampled cells, k=30, silhouette subset <=3,000.
- Per-type and donor/type alignment uses group-restricted neighbors, a
  composition-adjusted mixing ratio and centroid distance normalized by within-assay
  RMS radius. Raw latent distances must not be directly compared across models.
- Full-population mean RNA profiles supply donor/type correlations. Site confounding
  remains explicit; donor/site entropy is a diagnostic, not an optimization target.
- Frozen CD14 gene sets are scored on expression using common equal-group moments.
  Different shared-panel normalization/scaling means values need not match nb09.
  Gradient preservation is assessed by opposite-assay neighbor score prediction
  within CD14 donor groups, not merely by coloring an unchanged score on UMAP.

Thresholds in the notebook are exploratory and predeclared, not universal biological
standards. The criterion table preserves concern/partial support. Never proceed
because assay mixing alone improved. No inference treats cells as independent people.

## Outputs

`results/shared_rna_integration/run01/` contains one shared RNA h5ad (normalized X
and one counts layer), HVG/shared gene lists, metadata, PCA/scVI/UMAP coordinates,
model and training histories, module scores/membership, input audits, per-cell,
per-type and donor/type metrics, gradient checks, decision criteria and 16 PNG/PDF
figures. The model does not save another copy of AnnData. Reconstruct its input by
subsetting the h5ad to `var.highly_variable` in original order before model loading.

An existing status file blocks overwrites, even after failure. Choose a new output
folder for a fresh run. There is no automatic resume. Baseline embeddings are saved
before training. A completed computational run still requires biological review.
No totalVI, MultiVI or cross-assay cell pairing is performed.

## Validation and API references

`python -m unittest discover -s tests -p test_rna_integration.py -v` checks synthetic
eight-donor input loading, raw-vs-processed handling, barcode collisions, mapping
fingerprints, missing groups, self-neighbor removal, alignment vs biological collapse,
gradient diagnostics, PCA/UMAP, figures, h5ad writing and a short scVI training/save
smoke test when installed. Synthetic execution is not real biological validation.

Implementation follows [Scanpy HVG documentation](https://scanpy.readthedocs.io/en/latest/api/generated/scanpy.pp.highly_variable_genes.html)
and the [scVI API](https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCVI.html).
