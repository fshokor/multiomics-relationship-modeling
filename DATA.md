# Data Description

## Source

**Dataset:** NeurIPS 2021 Multimodal Single-Cell Integration benchmark
**GEO accession:** [GSE194122](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122)
**Competition page:** https://openproblems.bio/events/2021-09_neurips
**Organism:** Human, bone marrow mononuclear cells (BMMC)
**Donors:** 10 healthy donors, processed across 4 sites

Files used (downloaded via `scripts/download_benchmark.py`, stored in `data/benchmark/`, git-ignored):
- `GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad`
- `GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad`

These are the **processed** benchmark files (not raw FASTQ/counts) — normalization, QC metrics, and dimensionality reductions are already computed by the benchmark organizers.

---

## CITE-seq file (RNA + protein, co-measured per cell)

`n_obs × n_vars = 90,261 × 14,087`

### Structure
Both modalities live in a **single combined matrix**, distinguished by `var['feature_types']`:

| feature_types | count | meaning |
|---|---|---|
| GEX | 13,953 | gene expression |
| ADT | 134 | surface protein (antibody-derived tags) |

To split into separate RNA/protein AnnData objects:
```python
gex_mask = cite.var['feature_types'] == 'GEX'
adt_mask = cite.var['feature_types'] == 'ADT'
cite_gex = cite[:, gex_mask].copy()
cite_adt = cite[:, adt_mask].copy()
```

### `.X` representation
Not raw counts — float32, min 0 / max ~423,672, non-integer values. Likely library-size or CLR-normalized.
**Raw counts preserved separately** in `layers['counts']`.
`cite.raw` is `None` — no second copy of unnormalized data there.

### `obs` columns (cell metadata)
QC metrics (`GEX_n_genes_by_counts`, `GEX_pct_counts_mt`, `GEX_phase`, `ADT_n_antibodies_by_counts`, `ADT_total_counts`, `ADT_iso_count`), annotations (`cell_type` — 45 fine-grained immune types, `batch`, `Site`), donor metadata (`DonorID`, `DonorAge`, `DonorBMI`, `DonorBloodType`, `DonorRace`, `Ethnicity`, `DonorGender`, `QCMeds`, `DonorSmoker`), and a benchmark split flag `is_train`.

- `batch`: 12 sample-donor batches (e.g. `s1d1`...`s4d9`)
- `Site`: 4 sites (site1–site4)
- `cell_type`: 45 categories, e.g. CD14+ Mono (21,693 cells), CD4+ T activated, NK, HSC, pDC, etc. — already annotated, no clustering needed.

### `obsm` keys
`ADT_X_pca` (50d), `ADT_X_umap` (2d), `ADT_isotype_controls` (6d), `GEX_X_pca` (50d), `GEX_X_umap` (2d) — precomputed embeddings, not used for per-feature correlation analysis.

### `var` columns
`feature_types`, `gene_id` (Ensembl ID).

---

## Multiome file (RNA + ATAC, co-measured per cell)

`n_obs × n_vars = 69,249 × 129,921`

### Structure
Same combined-matrix pattern as CITE-seq, split via `feature_types`:

| feature_types | count | meaning |
|---|---|---|
| ATAC | 116,490 | chromatin accessibility peaks |
| GEX | 13,431 | gene expression |

```python
gex_mask  = multiome.var['feature_types'] == 'GEX'
atac_mask = multiome.var['feature_types'] == 'ATAC'
multi_gex  = multiome[:, gex_mask].copy()
multi_atac = multiome[:, atac_mask].copy()
```

### Gene activity scores (pre-computed — key shortcut)
`obsm['ATAC_gene_activity']`, shape `(69249, 19039)`, with feature names in `uns['ATAC_gene_activity_var_names']` (clean gene symbols, e.g. `OR4F5`, `SAMD11`, `NOC2L`...).

This is ATAC peaks **already aggregated to gene-level accessibility scores** by the benchmark organizers. It removes the need for manual peak-to-gene linkage (originally planned as a fallback in `NEXT_SESSION.md`) — nb03 can correlate GEX directly against `ATAC_gene_activity` per gene, the same way nb02 correlates GEX against ADT.

### `.X` representation
Float32, min 0 / max ~9,319, mean 0.056 — much sparser than CITE-seq `.X` (mean 3.05), consistent with combined GEX+ATAC peak data. Raw counts in `layers['counts']`.

### `obs` columns
QC metrics (`GEX_pct_counts_mt`, `GEX_n_counts`, `GEX_n_genes`, `GEX_phase`, `ATAC_nCount_peaks`, `ATAC_atac_fragments`, `ATAC_reads_in_peaks_frac`, `ATAC_blacklist_fraction`, `ATAC_nucleosome_signal`), annotations (`cell_type` — 22 coarser categories than CITE-seq, `batch`, `Site`), same donor metadata fields as CITE-seq. No `is_train` column in this file.

- `batch`: 13 sample-donor batches
- `Site`: 4 sites
- `cell_type`: 22 categories (coarser granularity than CITE-seq's 45) — e.g. CD8+ T (11,589), CD14+ Mono (10,843), NK, Erythroblast, HSC, etc.

### `obsm` keys
`ATAC_gene_activity` (19,039d, see above), `ATAC_lsi_full` (50d), `ATAC_lsi_red` (39d), `ATAC_umap` (2d), `GEX_X_pca` (50d), `GEX_X_umap` (2d).

---

## Relationship between the two files

CITE-seq and Multiome are **disjoint cell populations** — confirmed 0 shared barcodes between the 90,261 CITE-seq cells and 69,249 Multiome cells. They draw from the same donor/site pool but are separate single-cell captures. This means:
- RNA↔protein analysis (nb02) and RNA↔ATAC analysis (nb03) are run **independently**, each within its own file.
- Cross-comparison between the two relationship structures (nb03 Step 4) happens at the **gene level** (which genes show coupling in one axis vs the other), not the cell level.

---

## Open questions / not yet checked

- Whether `is_train` in the CITE-seq file should be respected as a train/holdout split for any model evaluation, or ignored since this project isn't doing prediction.
- Cell type label harmonization between CITE-seq (45 types) and Multiome (22 types) if a joint comparison across both is ever needed.
