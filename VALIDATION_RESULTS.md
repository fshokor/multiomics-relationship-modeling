# Completed single-donor validation — donor 15078

Executed locally on 2026-10-07 with `scripts/validate_single_donor.py`. All
manifest-listed input SHA256 fingerprints matched. The copied CITE source matched
the expected size and cell metadata; its SHA256 is recorded in the validation
manifest. Upstream results were not replaced.

## Shared-site RNA pathway consistency

Recomputed cell-type specificity independently in each capture within sites 1, 2
and 4. Within each site, both captures used exactly the same reference cell types,
each with at least 50 cells in both captures, and the frozen 12,000-gene universe.
Reference types can differ between sites. Used 1,000 gene-set permutations and BH
adjustment across all eligible Hallmark sets per target/capture/site, then inspected
the five preselected programs. No cells were paired across captures.

| Program | Testable shared sites | Positive NES and q <= 0.05 in both captures |
|---|---|---|
| CD14 inflammatory response | 1, 2, 4 | 3/3 |
| CD14 TNF/NF-kB | 1, 2, 4 | 3/3 |
| CD16 interferon-alpha | 2, 4 | 2/2 |
| Lymphoid progenitor E2F | 1, 2, 4 | 3/3 |
| MK/E progenitor Myc V1 | None | Untested |

All 11 testable comparisons agreed positively (maximum q approximately 0.021).
CD16 site 1 has only 45 CITE cells. MK/E CITE counts are 14, 43 and 31 in sites
1, 2 and 4; Multiome site 2 also has only 33. These are insufficient groups, not
failed replication. CITE site 3 has no corresponding Multiome site for this donor.
Thus pooled Myc RNA agreement remains without a sufficiently powered site check
under the specified cell-count rule. Site-level agreement is within-donor
consistency, not independent biological replication. No balanced-cell resampling
or donor-level significance is implied by this check.

## Seven unavailable nb05 NES values

All seven had finite negative observed enrichment scores but **zero negative
same-sign null draws among the original 1,000 permutations**. The implemented
same-sign normalization and conditional p-value are consequently undefined.
This is not missing gene coverage or evidence of zero enrichment.

One affected CITE cDC2 heme metabolism. Six affected Multiome G/M progenitors:
allograft rejection, complement, inflammatory response, interferon-gamma response,
TNF/NF-kB and heme metabolism. None is one of our five selected cell-type/pathway
pairs, so this numerical issue does not invalidate their recorded enrichment scores.

An independent 10,000-draw diagnostic yielded only 0–14 same-sign draws per affected
test; G/M interferon-gamma still had zero. Exploratory finite NES estimates based
on 1–14 draws are unstable and must not replace the original missing values or be
used for new significance claims. The cause is established, but reliable normalized
scores for these seven tests remain unavailable. A future change to the null model
or enrichment method must be evaluated systematically, not used selectively to
recover preferred results.

## Leave-one-gene-out protein sensitivity

Reproduced the original nb07 full-module adjusted correlations to tolerance 1e-8
for all three antigens, overall and in each site. Removed only the matched RNA gene
from the 128-gene inflammatory module, leaving 127 genes with the same per-gene
reference standardization. ADT transformation, cells and adjustment stayed fixed.

| ADT | Removed RNA gene | Full adjusted Pearson | Leave-one-out adjusted Pearson | Leave-one-out site range |
|---|---|---:|---:|---:|
| CD14 | CD14 | 0.279 | 0.268 | 0.198–0.475 |
| CD88 | C5AR1 | 0.236 | 0.217 | 0.139–0.364 |
| CD54 | ICAM1 | 0.171 | 0.169 | 0.112–0.278 |

Every association remains positive in all four CITE sites. Inclusion of the matched
gene alone does not explain these module–protein relationships. Other correlated
genes remain in the score, so this does not establish independent mechanistic or
causal evidence. ICAM1/CD54 same-gene agreement remains weak despite its broader
module association.

## Updated interpretation and next step

The CD14 inflammatory program retains the clearest measured-subset support across
RNA, ATAC and protein. TNF/NF-kB has strong RNA–ATAC and weaker protein support.
Interferon and E2F RNA enrichment now show within-site cross-capture consistency,
but this does not strengthen their weak downstream associations automatically.
Myc protein coverage and within-site RNA validation remain unavailable.

The three requested checks have been executed; their coverage and numerical limits
remain explicit. The next substantive step is testing the leading programs across
additional donors with donor-aware summaries, before full latent integration.

Detailed tables and reproducibility metadata are in
`results/single_donor/validation/`: `site_coverage.csv`,
`site_enrichment_all_sets.csv`, `site_pathway_comparison.csv`,
`missing_nes_audit.csv`, `protein_leave_one_gene_out.csv`, and `manifest.json`.
