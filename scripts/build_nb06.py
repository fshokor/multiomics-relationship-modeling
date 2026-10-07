"""Generate nb06 only. Never regenerate executed nb04/nb05."""
import json
from pathlib import Path
from build_single_donor_notebooks import code, markdown, SETUP

ROOT = Path(__file__).resolve().parents[1]

cells = [markdown('''
# nb06 — ATAC support for reproducible RNA programs

## Scientific question
For the provisional RNA programs recovered in both captures, is related gene-level
chromatin accessibility elevated in the same population, and do RNA and ATAC module
scores covary **within genuinely paired Multiome cells**?

## Analysis strategy
Load nb04's cached Multiome RNA and nb05's shortlist. Read corresponding
`obsm['ATAC_gene_activity']` rows by **cell ID**, with gene names from
`uns['ATAC_gene_activity_var_names']`. Audit coverage, score the same covered genes
in both layers, and compare within-type, site-specific and depth-adjusted
associations. Inspect shared RNA leading-edge genes and relative population states.

The five programs below are provisional choices from the reviewed donor-15078 run.
The donor itself is read from nb04, not hard-coded. Changing the donor may require
changing this list to supported programs from its own nb05 run.

**No CITE–Multiome cells are paired.** CITE provides population-level RNA replication
only. No peak-to-gene inference, protein analysis, pseudobulk or latent integration
is performed here. Existing notebooks and RNA results are left intact.
'''), SETUP, markdown('''
## Inputs and provisional program panel
Copy `src/atac_programs.py`, `src/atac_plots.py`, and this notebook to your existing
Drive project. Existing shared modules must be the updated versions used for nb04/05.
Restart a reused Colab runtime after copying Python modules.

The loader checks nb04 artifact hashes, nb05 completion and whether each requested
program still has positive supported RNA in both captures and positive direction in
nb05's balanced draw. It checks shortlist NES/q-values against nb04's source tables.
The shortlist is not proof of replication across people or robustness to site.
'''), code('''
from src.atac_programs import load_inputs, run_atac, INITIAL_PROGRAMS

PROGRAMS = list(INITIAL_PROGRAMS)
MIN_MODULE_GENES = 10
MIN_ORIGINAL_SET_COVERAGE = 0.20
MIN_CELLS_FOR_ASSOCIATION = 50
RELATIVE_STATE_THRESHOLD = 0.50
RANDOM_SEED = 42

manifest, selected, rna_cache, rna_obs, rna_genes, universe = load_inputs(OUTPUT_ROOT, PROGRAMS)
print('Selected donor:', manifest['donor'])
with pd.option_context('display.max_rows', None, 'display.max_colwidth', None):
    display(selected[['cell_type', 'pathway', 'NES_cite', 'NES_multiome',
                      'q_value_cite', 'q_value_multiome', 'shared_leading_edge']])
display(rna_obs.groupby(['cell_type_harmonized', 'Site'], observed=True).size().rename('n_cells').reset_index())
del rna_cache, rna_obs, rna_genes
'''), markdown('''
## Representation, scoring and safeguards
- **RNA:** retain nb04's independent log1p(CP10k) RNA; do not normalize it again.
- **ATAC:** use gene activity **as stored**. nb03 recorded nonnegative processed
  scores (maximum about 7 in its inspected overlap). This does not establish the
  original transformation. Do not pretend these are counts or automatically log
  them again. A log1p sensitivity score is reported separately, not substituted.
- **Coverage:** use the intersection of the GMT pathway, nb04's shared eligible RNA
  universe and available ATAC symbols. Exclude features constant in either layer.
  Both module scores use exactly this same gene list. Require at least 10 genes and
  20% of the original GMT membership; save excluded genes and coverage either way.
  Absent or inadequate coverage is **missing evidence**, never negative support.
- **Reference:** within each layer, calculate per-gene means and variances with equal
  weight for every retained cell type, and equal weight for cells within a type.
  Module score is the mean of these gene z-scores. RNA and ATAC are standardized
  separately. Module scores themselves do not have unit variance and cannot be
  interpreted as equal biological effect sizes across layers or pathways.
- **Cell QC:** exclude zero-total ATAC-activity cells from both layers, audit counts,
  and stop on negative/nonfinite activity, mismatched cell IDs or duplicate symbols.
  No new organizer-QC thresholds are silently applied.
- **Associations:** Pearson/Spearman are effect-size summaries, not donor inference.
  Primary results are within the requested population. Pooled correlations are
  labelled separately because cell-type structure can drive them. Minimum 50 cells
  applies to target and site associations; insufficient groups remain visible.
- **Confound sensitivity:** residual Pearson adjusts both scores for log RNA count
  depth, log ATAC peak-count depth when available, and site indicators. Residual
  Spearman is correlation of the OLS residual ranks, not a formal partial-Spearman
  test. If raw depth metadata is missing, the processed activity total is explicitly
  identified as a fallback proxy. Adjustment is a sensitivity check, not causal proof.

No cell-level p-values are used to promote programs merely because they have many cells.
The two CD14 programs may have overlapping genes and are not independent confirmations.
'''), code('''
tables, target_summary, atac_manifest = run_atac(
    OUTPUT_ROOT, MULTIOME_H5AD, requested=PROGRAMS,
    min_genes=MIN_MODULE_GENES, min_coverage=MIN_ORIGINAL_SET_COVERAGE,
    min_cells=MIN_CELLS_FOR_ASSOCIATION, threshold=RELATIVE_STATE_THRESHOLD,
    seed=RANDOM_SEED)
print('ATAC depth covariate:', atac_manifest['depth_covariate'])
display(pd.read_csv(DIRS['atac'] / 'cell_counts.csv'))
display(tables['coverage'].drop(columns=['used_genes', 'missing_atac_or_rna']))
print('Complete gene coverage and membership are saved under results/single_donor/atac/.')
'''), markdown('''
## 7A — Within-population RNA–ATAC support
Compare the unadjusted and adjusted associations below. Similar elevated population
means and a positive within-population correlation are different observations; one
does not require the other. A narrow range within a type can weaken correlation even
when its mean activity is elevated. Conversely, pooled association can reflect
population composition rather than within-type coordination.
'''), code('''
assoc = tables['associations']
primary = assoc[(assoc.scope == 'cell_type') & (assoc.group == assoc.cell_type)]
display(primary)
display(target_summary)
display(assoc[assoc.scope == 'target_by_site'])
for name in ['rna_atac_module_scatter.png', 'target_activity_heatmap.png',
             'population_activity_heatmap.png', 'site_association_heatmap.png']:
    if name in atac_manifest['figures']:
        display(Image(filename=str(DIRS['atac'] / 'figures' / name)))
'''), markdown('''
## 7B — Shared RNA leading-edge genes
For each target population, report all covered, variable genes in the shared RNA
leading edge: RNA level, stored ATAC activity level, nonzero fractions and unadjusted/
depth-site-adjusted associations. ATAC nonzero fraction is a property of the processed
score, not a calibrated probability of promoter accessibility.

The figure shows up to three genes per program chosen by joint nonzero detection
coverage, not by strongest correlation. Statistics use all eligible cells; scatter
plots subsample solely for readability. Negative associations remain visible.
'''), code('''
gene_results = tables['leading_edge_genes']
display(gene_results)
if 'leading_edge_gene_scatter.png' in atac_manifest['figures']:
    display(pd.read_csv(DIRS['atac'] / 'plotted_genes.csv'))
    display(Image(filename=str(DIRS['atac'] / 'figures/leading_edge_gene_scatter.png')))
'''), markdown('''
## 7C — Relative population states and sensitivity
High means target population mean module score >= +0.5; low means <= -0.5.
Intermediate values remain **intermediate/mixed** rather than being forced into a
binary class. These are exploratory reference-relative cutoffs, not significance
thresholds or universal biological activation thresholds. The table also evaluates
0.25 and 0.75 to expose threshold dependence.

| Relative ATAC | Relative RNA | Cautious description |
|---|---|---|
| High | High | Elevated in both layers; assess within-type association separately |
| High | Low | Potentially permissive; not demonstrated temporal priming |
| Low | High | Candidate RNA/accessibility decoupling; consider score limitations |
| Low | Low | Relatively low in both; not proof of inactivity |

Accessibility preceding or permitting transcription cannot be established from these
cross-sectional data. Low gene activity is not proof of closed regulatory chromatin.
'''), code('''
display(tables['state_threshold_sensitivity'])
if not primary.empty:
    display(primary[['cell_type', 'pathway', 'n_cells', 'spearman', 'adjusted_pearson',
                     'log1p_atac_spearman', 'leading_edge_spearman',
                     'rna_score_vs_rna_depth', 'atac_score_vs_atac_depth']])
'''), markdown('''
## Main findings
No biological result is pre-filled. Use the quantitative outputs below to distinguish
coverage, relative population support, within-type covariation, depth/site sensitivity
and gene-level exceptions. Site-specific ATAC associations do not replace the still
outstanding site-specific **cross-capture RNA pathway** replication check from nb05.
'''), code('''
print('Programs analyzed:', atac_manifest['n_programs_analyzed'], '/', atac_manifest['n_programs_requested'])
print('Paired Multiome cells in the reference:', atac_manifest['n_cells'])
if not target_summary.empty:
    display(target_summary[['cell_type', 'pathway', 'n_cells', 'rna_mean', 'atac_mean', 'state']])
if not primary.empty:
    display(primary[['cell_type', 'pathway', 'n_cells', 'tested', 'pearson', 'spearman', 'adjusted_pearson']])
print('Saved tables, PNG/PDF figures and provenance:', DIRS['atac'])
print('Review these outputs before protein follow-up. No causal or individual-level inference.')
'''), markdown('''
## Limitations
- One donor; cells are not independent biological donor replicates.
- The reference uses retained harmonized populations, and broad labels can mix states.
- RNA and ATAC share Multiome cells, but CITE cells are separate captures.
- Gene activity is a processed approximation, not promoter-specific or peak-level
  regulatory evidence. The original activity transformation has not been established.
- Z-scores are relative to this donor/reference composition, not absolute activity.
- Full-program and leading-edge scores remain RNA-selected exploratory analyses.
- Site/depth adjustment can remove biological as well as technical variation; inspect
  both results. Cell cycle and other unmeasured covariates are not fully controlled.
- Within-site small groups may be untestable; missing coverage is not negative evidence.
- Balanced RNA direction is supported, but site-specific RNA pathway replication and
  the seven unavailable nb05 NES values remain separate outstanding checks.
- No protein evidence is available here. Later unmeasured ADTs must not be coded as low.
- Cross-sectional associations cannot establish ATAC → RNA causality or temporal priming.
''')]

if __name__ == '__main__':
    for i, cell in enumerate(cells):
        cell['id'] = f'nb06-{i:03d}'
    notebook = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                    'language_info': {'name': 'python', 'version': '3.11'}}, nbformat=4, nbformat_minor=5)
    path = ROOT / 'notebooks/nb06_pathway_atac_support.ipynb'
    if path.exists():
        raise FileExistsError('Refusing to overwrite nb06; preserve executed results before regenerating')
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
