# nb06: paired Multiome RNA–ATAC support

Copy these files to the existing `multiomics-relationship-modeling` Drive project:

- `notebooks/nb06_pathway_atac_support.ipynb`
- `src/atac_programs.py`
- `src/atac_plots.py`

Keep the updated shared `src/` modules from nb04/05. Run nb06 in Colab from the top,
using the same Drive mount and `configs.paths.MULTIOME_H5AD` as the earlier notebooks.
Restart an existing runtime if it has cached older module versions. There is no need
to rerun nb04/05 if their saved outputs remain consistent. Nb06 validates those
artifacts and never rewrites their results or notebooks.

## Panel

The default five provisional programs follow the reviewed nb05 shortlist:

| Population | Program |
|---|---|
| CD14 monocytes | Inflammatory Response |
| CD14 monocytes | TNF-alpha Signaling via NF-kB |
| CD16 monocytes | Interferon Alpha Response |
| Lymphoid progenitors | E2F Targets |
| MK/E progenitors | Myc Targets V1 |

The donor is read from nb04. Each requested entry must exist and be selected in the
current nb05 results, with positive supported enrichment in both RNA datasets and
positive direction in its balanced draw. The code checks enrichment values against
the saved nb04 source tables. These provisional choices do not establish cross-site
RNA pathway replication; that outstanding check remains explicit in nb06.

## Analysis

Only paired Multiome cells are used for association. ATAC rows are aligned to cached
RNA **by cell ID**, with source donor/site/original cell-type metadata checked. Dense,
CSR and CSC gene-activity arrays are read in chunks; only selected pathway columns
are retained. Zero-total ATAC cells are excluded from both layers and recorded.

RNA remains nb04's log1p(CP10k). ATAC activity is used as stored because its original
transformation is not established; it is not assumed to be raw counts. The notebook
reports a separate log1p sensitivity, without replacing the primary representation.

Each module uses the same genes in both layers: the pathway's intersection with
nb04's shared eligible genes and available ATAC genes, excluding genes constant in
either layer. Minimum coverage is 10 genes and 20% of the original GMT set. Missing
coverage remains missing evidence. Coverage lists and all exclusions are saved.

Each gene is standardized separately within each layer, using an equally weighted
mixture of retained cell types as the reference. Module score is mean gene z-score,
not an absolute activity measurement or a module-level z-score. The notebook reports:

- Within-target Pearson and Spearman correlations, with every group's cell count.
- Per-site associations and population means; groups below 50 cells are untested.
- Depth/site-adjusted residual associations using log RNA depth, log ATAC peak counts
  when available and site indicators. A processed-activity-total fallback is named
  explicitly when raw depth metadata is unavailable.
- Full-pathway versus shared-leading-edge and ATAC-transform sensitivities.
- Shared leading-edge RNA/ATAC levels, nonzero fractions and gene associations.
- Relative population states with an intermediate category and threshold sensitivity.

High/low cutoffs are exploratory (+/-0.5 mean gene z-score, with +/-0.25 and +/-0.75
also reported). Low is not biologically inactive. High ATAC with low RNA is at most
potentially permissive, not proof of temporal priming. No cell-level p-values are
used to claim replication across individuals. Pooled correlations are labelled as
composition-confounded diagnostics, not evidence of within-type coupling.

## Outputs and review

`results/single_donor/atac/` contains:

- `coverage.csv`, `gene_membership.csv`, `feature_standardization.csv`
- `cell_qc.csv`, `cell_counts.csv`, `selected_rna_programs.csv`
- `cell_scores.csv`, `population_summary.csv`, `target_program_summary.csv`
- `associations.csv`, `leading_edge_genes.csv`, `state_threshold_sensitivity.csv`
- `plotted_genes.csv` when covered leading-edge genes exist
- `manifest.json`, `status.json`, and separate PNG/PDF files in `figures/`

Use the figure names in the current manifest to distinguish generated plots from any
older run's figures. A run with insufficient coverage may produce no plots; its
coverage audit, rather than an old figure, determines what was tested.

Review coverage first, then target activity, within-type correlations, site/depth
sensitivity and individual genes. Report population concordance and cell-level
association separately. Do not transfer per-cell ATAC to CITE or infer causality.
Protein follow-up, targeted peaks and full integration remain deferred.

## Software validation

From a scientific Python environment in the repository root:

```bash
python -m unittest discover -s tests -v
```

The ATAC tests cover reordered cell IDs, missing/mismatched cells, dense/CSR/CSC
storage, negative values, equal-type reference moments, constant genes, missing
coverage, intermediate states, simulated depth confounding, and an end-to-end
synthetic nb04-to-nb06 run. Synthetic tests are not biological validation; the real
nb06 benchmark outputs must be generated in Colab.

`scripts/build_nb06.py` generates nb06 only and refuses to overwrite it. Do not run
the nb04/05 generator to create nb06: that would replace their executed notebooks.
