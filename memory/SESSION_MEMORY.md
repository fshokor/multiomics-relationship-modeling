# SESSION MEMORY
## multiomics-relationship-modeling

**Last updated:** 2026-06-30

## Project state
nb01_qc complete — both h5ad files loaded, inspected, and structure fully confirmed. See `DATA.md` for full data description. Ready to start nb02 (RNA-protein, CITE-seq).

## Dataset
NeurIPS 2021 Multimodal Single-Cell Integration (GSE194122) — see `DATA.md` for full details.
- CITE-seq: 90,261 cells, RNA (13,953 genes) + protein (134 ADT) combined in one AnnData, split via `var['feature_types']`
- Multiome: 69,249 cells, RNA (13,431 genes) + ATAC (116,490 peaks) combined in one AnnData, same split pattern
- CITE-seq and Multiome are disjoint cell populations (0 shared barcodes) — same donor/site pool, separate captures

## Scientific question
What is the structure of the relationship between omics layers?
- Where does RNA→protein coupling hold vs break down?
- Is breakdown cell-type-specific or universal?
- Same question for RNA→ATAC (chromatin regulation axis)

## Notebooks completed
- **nb01_qc** — done. Key findings:
  - `.X` is normalized (not raw) in both files; raw counts in `layers['counts']`
  - `cite.raw` is None — no secondary raw store
  - Cell type annotations already present: CITE-seq has 45 fine-grained types, Multiome has 22 coarser types (no clustering needed for either)
  - **Multiome has a major shortcut**: `obsm['ATAC_gene_activity']` (69249 × 19039) is pre-computed gene-level accessibility, with gene symbols in `uns['ATAC_gene_activity_var_names']`. This removes the need for manual peak-to-gene linkage originally planned for nb03 — can correlate GEX directly against gene activity scores.
  - Rich donor metadata available in both files (age, BMI, blood type, race, ethnicity, gender, smoker status) if donor-effect analysis is wanted later
  - CITE-seq has an `is_train` column (benchmark's own split) — Multiome does not. Not yet decided whether to respect it.

## Key decisions made
- No prediction task — this is relationship characterization
- CITE-seq track (nb02) and Multiome track (nb03) are parallel, run independently (disjoint cell populations)
- VAE (nb04) deferred until nb02/nb03 are complete
- nb03 revised: use `ATAC_gene_activity` obsm directly instead of building peak-to-gene linkage from scratch
- Colab/local environment switching uses the `IN_COLAB` try/except pattern (matches GDSC repo convention), `configs/paths.py` auto-detects

## Open questions for later
- Whether to respect `is_train` in CITE-seq for any evaluation split
- Cell type label harmonization between CITE-seq (45 types) and Multiome (22 types) if cross-comparison is needed
