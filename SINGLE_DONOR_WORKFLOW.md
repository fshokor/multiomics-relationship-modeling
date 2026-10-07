# Single-donor RNA workflow (first milestone)

The first increment implements donor selection, conservative cell-type harmonization,
independent RNA program characterization, ranked enrichment, RNA concordance and a
candidate shortlist. Real-data execution is pending in the existing Colab/Drive
environment. No benchmark biological findings are claimed from local synthetic tests.

## Run in your existing Colab project

1. Copy the updated `src/` directory, `configs/`, and the two new notebooks into
   `/content/drive/MyDrive/multiomics-relationship-modeling` on Drive. Restart an
   already-running runtime after replacing imported Python modules.
2. Run `notebooks/nb04_single_donor_rna_characterization.ipynb` from the top.
3. Inspect the donor ranking, mapping decisions and selected-donor original subtype
   counts. Unknown labels remain unresolved. If necessary, make a reviewed mapping
   CSV with explicit reasons, set `MAPPING_OVERRIDE`, and rerun nb04.
4. Run `notebooks/nb05_rna_concordance.ipynb` from the top. It verifies nb04's saved
   artifacts rather than relying on variables from another notebook session.
5. Review the RNA findings and leading edges before the next ATAC implementation.

Both notebooks follow nb01's Drive mount and project root. `configs.paths` resolves:

```
data/benchmark/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad
data/benchmark/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad
```

Only metadata and selected-donor count rows are read, in chunks. The full ATAC peak
matrix and all-cell gene-activity matrix are not loaded into memory. Original h5ad
files and nb01–nb03 are never modified. Colab installs AnnData if needed; the code
otherwise uses NumPy, pandas, SciPy, h5py and matplotlib. No GPU or heavy model is needed.

## Scientific choices

- **Selection:** require at least three shared harmonized populations with at least
  50 cells each per capture and maximum dominance <=70%. Eligible donors are sorted
  by supported shared populations, sum of per-population minimum counts, lower
  dominance, then ID. Thresholds and all candidates are saved; failure is explicit.
- **Mapping:** visible conservative rules; preserve original labels. Pool clear
  CD4/CD8, B and erythroid subtypes; keep progenitor lineages and DC classes separate.
  Identical original labels are marked as such. Exact labels do not imply equal
  subtype composition. Review the selected donor's original counts.
- **RNA scale:** start from integer `layers['counts']`, select GEX before library
  normalization, scale to 10,000 and log1p independently. `.X` is not assumed to have
  comparable processing. Existing files remain unchanged. GEX panel differences
  remain a limitation; this is not a preprocessing replacement for nb01–nb03.
- **Specificity:** mean log-normalized RNA minus the equally weighted mean of other
  retained shared cell types. Keep high-expression ranking and detection separately.
  This is descriptive marker/program characterization, not disease/control DE.
- **Gene flags:** mitochondrial, ribosomal and selected housekeeping genes remain
  annotated in complete tables. Only technical spike-ins are categorically excluded
  from enrichment. B2M, for example, is not discarded as housekeeping.
- **Shared universe:** genes present and detected in >=1% of cells of at least one
  retained type in each capture; one universe across all compared rankings. Same
  cell types, normalization, collection and output schema are used independently.
- **Enrichment:** weighted preranked running sum and gene-set permutation nulls.
  Same-sign null normalization gives NES; plus-one p-values never report zero.
  BH `q_value` is corrected within dataset/type. `q_value_global` additionally adjusts
  all tests in `all_enrichment.csv`. Neither is Broad's pooled GSEA FDR. These tests
  ignore gene correlation and provide no inference across donors. The implementation
  is an explicit dependency-light alternative, not a call to GSEApy.
- **Database:** Enrichr's named `MSigDB_Hallmark_2020` human-symbol snapshot, downloaded
  once and cached. Exact GMT bytes, checksum, coverage and seed are saved. Supply a
  reviewed local human-symbol GMT when the endpoint is unavailable. No automatic
  fallback or invented pathway membership. See the
  [Enrichr library endpoint](https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020)
  and [GSEA method documentation](https://docs.gsea-msigdb.org/GSEA/GSEA_User_Guide/).
- **Concordance:** shared-gene Pearson/Spearman, specificity-rank Spearman, positive
  top-50/100/250 Jaccard, pathway NES/support classes, five independent balanced
  sampling repeats, one balanced pathway rerun, and common-site profile comparisons.
  Capture-specific significance does not establish a between-capture difference;
  negative specificity is not biological inactivity. Unsupported is not absent.
- **Candidates:** up to ten positively replicated programs, effect-ranked with
  cell-type diversity, shared leading edges and redundancy reduction. Positive
  balanced-run direction is required. Biological review remains explicit; no forced
  quota, automatic claims of immune activation or automatic downstream release.
- **RNA bridge:** uncorrected PCA of shared-gene population means is exploratory.
  No cross-capture matched cells, model fitting, batch correction or information
  transfer. One donor cannot support assessment of donor effects.

## Saved results

All results live under `results/single_donor/`:

| Directory | Main outputs |
|---|---|
| `donor_selection/` | All-donor/site/original-type counts, ranked donors, long and paired mapping tables, selected donor subtype counts, supported type counts |
| `rna_cite/`, `rna_multiome/` | Sparse normalized RNA cache, original/harmonized cell metadata with depth, complete profiles and two gene rankings |
| `gsea/` | Exact GMT, coverage audit, full enrichment tables including leading edges, global BH table |
| `rna_concordance/` | Shared gene universe, expression/rank/overlap results, paired profiles, pathway classes, balanced and site sensitivities, shortlist, centroid PCA, quantitative JSON |
| `atac/`, `protein/`, `multilayer/` | Reserved for subsequent validated milestones; no fabricated results |

Figures are separate 300-dpi PNG and vector PDF files inside each stage's `figures/`.
`manifest.json` records donor, parameters, input sizes/timestamps, package versions,
mapping/collection/output hashes and normalization. `status.json` tracks stage
completion. Full multi-GB source files are not hashed; their paths, sizes and
timestamps are recorded. A failed rerun invalidates completion state. Use a new
output root for parameter experiments that should be preserved side by side.

## Validation and remaining milestones

Run software tests from the repository root in a scientific Python environment:

```bash
python -m unittest discover -s tests -v
```

Tests cover count validation and normalization, equal-type specificity, conservative
mapping, donor selection, independently balanced samples, enrichment against a dense
reference calculation, deterministic permutations, negative/unsupported states, empty
results, notebook structure, and a synthetic h5ad-to-concordance run with stale-output
detection. Synthetic outputs use temporary directories and do not populate biological
results. Regenerate only the new notebooks with
`python scripts/build_single_donor_notebooks.py`.

The real data are on Drive, not available in this local checkout. The first milestone
must still be run and interpreted there. In accordance with the requested sequencing,
nb06 (ATAC gene activity) is now implemented following review of the executed RNA
notebooks; see [the ATAC run guide](ATAC_WORKFLOW.md). Its real-data execution remains
pending in Colab. nb07 (direct versus phenotypic protein support) and nb08
(multilayer summaries) are **not implemented yet**. Their future implementation must
preserve within-capture pairing only, distinguish unmeasured ADTs from negative
evidence, standardize layer scores, and avoid causal or temporal-priming claims.
Targeted peak follow-up and multi-donor pseudobulk are also deferred.
