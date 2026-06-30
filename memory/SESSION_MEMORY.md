# SESSION MEMORY
## multiomics-relationship-modeling

**Last updated:** 2026-06-30

## Project state
nb01_qc and nb02_rna_protein both complete. nb02 results are strong and
interpretable. Ready to start nb03 (RNA-ATAC, Multiome).

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

### nb01_qc — done
See `DATA.md` for full structure findings (X normalization, cell type annotations,
donor/batch columns, ADT/ATAC location, no cell overlap between files).

### nb02_rna_protein — done
**ADT-to-gene matching:** Built `src/analysis/cd_gene_mapping.py`, a curated
CD-nomenclature -> HGNC gene symbol dictionary (handles cases like CD11a->ITGAL,
CD16->FCGR3A that a naive name-match misses). Matched 112/134 ADT proteins to
GEX genes. 22 unmatched: either gene not present in this GEX panel (CD142, CD146,
CD169, CD224, CD274, CD279, CD41, CD49b, CD57, CD62P, Podoplanin) or deliberately
excluded as ambiguous (CD45RA/CD45RO are PTPRC isoforms not separate genes;
CD158/CD158b/CD158e1 are clone-dependent KIR family; TCR/HLA-A-B-C/HLA-DR/
TCRVa7.2/TCRVd2 are multi-gene complexes).

**Results (112 genes, all 90,261 cells):**
- Median per-gene Pearson r = 0.194, mean = 0.230
- Top coupled: CD3D (0.645), IL7R (0.614), MS4A1/CD20 (0.611), KLRB1 (0.589),
  IL3RA (0.584) — canonical lineage-defining markers
- Most independent (lowest r): TNFRSF4/OX40 (0.034), PVR (0.028), TNFRSF14
  (0.027), TNFRSF9/4-1BB (0.022), CD69 (0.016), CTLA4 (-0.019), LAG3 (-0.037),
  CSF1R (-0.087), ITGAE (-0.095)
- **Key finding: decoupled genes are almost entirely activation/checkpoint
  receptors** (OX40, 4-1BB, CTLA4, LAG3, CD69) — consistent with known biology
  of intracellular protein stores rapidly trafficked to the surface upon
  activation, independent of transcription timing. This is a genuinely
  interpretable, pitch-worthy result.
- Cell-type-specific coupling (high variance across 42 cell types): IGHD,
  KLRB1, IL3RA, FCGR3A, CD22, ITGAX, CD8A, CD63, IL7R
- Universal (low-variance, consistently weak) coupling: FCER2, CD40LG, CSF1R,
  NCR1, SLAMF6, CD83, TNFRSF4, CXCR5, CTLA4 — note "universal" here means
  *stable*, not *strong*; these checkpoint/co-stim genes are consistently
  weakly coupled everywhere, not consistently tightly coupled.
- Donor consistency: mean pairwise r = 0.758 across 9 donors (1 donor dropped
  below MIN_CELLS_PER_TYPE=50) — strong agreement, supports biological (not
  technical) origin of coupling patterns
- Deviation by cell type: CD14+ Mono lowest (0.266, most RNA-protein
  concordant), Reticulocyte and dnT highest (~0.41-0.42, most divergent)
- All tables saved to `results/tables/nb02_*.csv`, figures to `results/figures/nb02_*.png`

## Key decisions made
- No prediction task — this is relationship characterization
- CITE-seq track (nb02) and Multiome track (nb03) are parallel, run independently (disjoint cell populations)
- VAE (nb04) deferred until nb02/nb03 are complete
- nb03 revised: use `ATAC_gene_activity` obsm directly instead of building peak-to-gene linkage from scratch
- Colab/local environment switching uses the `IN_COLAB` try/except pattern (matches GDSC repo convention), `configs/paths.py` auto-detects
- Analysis uses `.X` (normalized), not raw counts — avoids library-size confound, consistent with benchmark's intended analysis-ready representation
- ADT-gene matching requires curated CD->gene mapping, not name-string matching — most CD nomenclature differs from HGNC gene symbols

## Colab/Drive sync notes
- `src/` and `configs/` must be manually copied to Google Drive (`MyDrive/multiomics-relationship-modeling/`) for Colab imports to work — they are NOT auto-synced from the WSL2 git repo
- If `ModuleNotFoundError: No module named 'src'` occurs even after copying files and `!ls` shows them: this is a Drive FUSE mount caching issue. Fix: `drive.flush_and_unmount()`, remount with `force_remount=True`, then restart the Colab runtime and re-run all cells from the top
- Remember to re-sync `src/`/`configs/` to Drive after any local edits — no auto-sync

## Open questions for later
- Whether to respect `is_train` in CITE-seq for any evaluation split
- Cell type label harmonization between CITE-seq (45 types) and Multiome (22 types) if cross-comparison is needed
- Consider longer-term: clone GitHub repo directly in Colab instead of manual Drive copy, for cleaner src/ sync

- **Note for nb04 (VAE):** This analysis (correlation, deviation scores) only
  used the 112 genes with a clear RNA<->protein name match, since pairing
  requires aligning each protein to its corresponding gene. The future DL
  model (nb04) is not constrained this way — it will take the full RNA
  vector (13,953 genes) and the full ADT vector (134 proteins) as two
  separate inputs, with no requirement that individual features be
  matched by name. The model learns the joint structure across all
  features in each modality, not just the name-matched subset used here
  for interpretable per-gene correlation.