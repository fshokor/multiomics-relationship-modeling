"""Regenerate only the two new notebooks; never touches nb00–nb03."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": textwrap.dedent(text).strip().splitlines(keepends=True)}


SETUP = code('''
from pathlib import Path
import sys
import subprocess

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    # Same Drive root as nb01. No changes to the original benchmark files.
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'anndata>=0.11,<0.13'])
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_PATH = Path('/content/drive/MyDrive/multiomics-relationship-modeling')
else:
    BASE_PATH = Path.cwd().resolve()
    if not (BASE_PATH / 'src').is_dir():
        BASE_PATH = BASE_PATH.parent

if not (BASE_PATH / 'src/single_donor_workflow.py').is_file():
    raise FileNotFoundError('Copy the updated src/, configs/ and new notebooks to the project on Drive first.')
sys.path.insert(0, str(BASE_PATH))
import pandas as pd
from IPython.display import display, Image
from configs.paths import CITE_H5AD, MULTIOME_H5AD
from src.single_donor_io import output_dirs

OUTPUT_ROOT = BASE_PATH / 'results/single_donor'
DIRS = output_dirs(OUTPUT_ROOT)
print('Colab:', IN_COLAB, '| Project:', BASE_PATH)
''')

nb04 = [markdown('''
# nb04 — Single-donor RNA program characterization

## Scientific question
Which donor and biologically comparable cell populations provide a useful test case,
and which transcriptional programs characterize those populations in each capture?

## Analysis strategy
Inspect the benchmark metadata and count layer without loading the full ATAC matrix.
Select a common donor by an explicit coverage/diversity rule; retain original labels
and audit every broad lineage mapping. Normalize RNA counts independently with the
same log1p(CP10k) convention. Characterize mean expression, detection and specificity
against equally weighted other shared cell types. Rank-enrich one cached human-symbol
Hallmark collection against the same shared gene universe.

This is marker/program characterization in healthy cells, **not condition DE**.
CITE and Multiome are separate captures. Cells must never be cross-matched.

**Run location:** your existing Colab/Drive project, as in nb01. Copy the updated
`src/` directory and these notebooks to Drive before running. Local synthetic tests
do not constitute biological validation. nb01–nb03 are preserved.
'''), SETUP,
markdown('''
## Parameters and input inspection
Defaults require at least 50 cells per type in each capture, at least three supported
shared types, and no population above 70% of either donor capture. These development
thresholds are explicit, configurable and saved. No donor is hard-coded.

The count layer is used because `.X` has uncertain/incompatible processed scales in
the repository's DATA.md. RNA library size is calculated on **GEX only**, never ADT or
ATAC. Each capture retains its own GEX panel during normalization; panel differences
are a limitation. Existing preprocessing and input files are not overwritten.
'''), code('''
from src.single_donor_io import inspect_h5ad

PATHS = {'cite': CITE_H5AD, 'multiome': MULTIOME_H5AD}
MIN_CELLS = 50
MIN_SHARED_TYPES = 3
MAX_DOMINANCE = 0.70
RANDOM_SEED = 42
PERMUTATIONS = 1000
MIN_DETECTION = 0.01
COMPUTE_MEDIANS = False  # optional blockwise sparse->dense calculation
MAPPING_OVERRIDE = None  # path to a reviewed copy of celltype_mapping.csv

for dataset, path in PATHS.items():
    if not path.is_file():
        raise FileNotFoundError(f'{dataset}: missing {path}; use the same data/benchmark location as nb01.')
    obs, var, audit = inspect_h5ad(path)
    print(dataset, audit)
    display(obs.groupby(['DonorID', 'Site'], observed=True).size().rename('n_cells').reset_index())
    print('Original labels:', sorted(obs.cell_type.unique()))
del obs, var
'''), markdown('''
## Donor selection and explicit cell-type harmonization
Candidates are ranked by eligible shared-type coverage, sum of minimum per-type cell
counts across captures, lower maximum dominance, then donor ID. Dominance includes
unresolved cells. The full donor ranking records the criteria and selection.

Mappings are explicit code rules, not fuzzy label matching. CD4/CD8 subtypes, B-cell
subtypes and erythroid stages are pooled as documented. HSC, lymphoid, G/M and MK/E
progenitors remain separate; pDC and cDC populations remain separate. Labels such as
T reg, MAIT, gamma-delta T, ILC1 and plasmablasts are not silently assigned to broader
comparators. Exact otherwise-unmapped labels can be retained if present in both.

**Review the mapping and subtype composition before interpreting results.** To revise
it, copy `celltype_mapping.csv`, edit the harmonized labels/confidence/reasons, set
`MAPPING_OVERRIDE` above, and rerun. Do not edit a saved result in place and continue
with stale downstream outputs. Unresolved or undersized populations remain in audits
but are excluded from the shared RNA reference.
'''), code('''
from src.single_donor_workflow import prepare_donor

donor, donor_ranking, cell_counts, rna_data = prepare_donor(
    PATHS, OUTPUT_ROOT, min_cells=MIN_CELLS, min_shared=MIN_SHARED_TYPES,
    max_dominance=MAX_DOMINANCE, mapping_override=MAPPING_OVERRIDE)
print('Automatically selected donor:', donor)
display(donor_ranking)
display(pd.read_csv(DIRS['donor_selection'] / 'celltype_mapping_paired.csv'))
display(pd.read_csv(DIRS['donor_selection'] / 'selected_donor_mapping.csv'))
display(cell_counts)
display(Image(filename=str(DIRS['donor_selection'] / 'figures/cell_counts.png')))
'''), markdown('''
## Independent RNA profiles and ranked enrichment
Mean expression is the mean of log1p(CP10k), not log of an aggregated count total.
Specificity is target-type mean minus the equally weighted mean of other retained
types. All types use the same reference populations in both datasets.

Mitochondrial, ribosomal and selected housekeeping genes are **flagged**, retained in
complete rankings, and excluded only from the compact marker figure where indicated.
Technical spike-ins are excluded from enrichment. Shared genes must be detected in
at least 1% of cells in at least one retained type in **each** capture; this creates
one fixed, symmetric universe, not separate cell-type-specific gene filters.

The default collection is Enrichr's `MSigDB_Hallmark_2020` human-symbol library (50
sets). The first run downloads it explicitly; subsequent runs reuse the exact GMT.
Its SHA256 and bytes are saved. A locally supplied human-symbol Hallmark, Reactome or
GO-BP GMT may be substituted, but both datasets must use the same file. No fabricated
gene sets or automatic database substitution are used on network failure.

The dependency-light GSEA-style implementation uses a weighted running-sum statistic
(p=1), same-sign gene-set permutation nulls, NES, and plus-one nominal p-values.
`q_value` is **Benjamini–Hochberg within dataset × cell type**, not Broad's pooled GSEA
FDR. An additional pooled BH column is saved in `all_enrichment.csv`. Null gene sets
ignore gene correlations, so significance is exploratory. No individual-level
replication is implied. Stable gene-symbol ordering breaks rank ties; tie fractions
and gene-set coverage are reported. Sets need 15–500 overlapping genes and fewer than
the full ranking. Negative enrichment means lower **relative specificity**, not
absence/inactivity of a biological pathway.

Method reference: [GSEA User Guide](https://docs.gsea-msigdb.org/GSEA/GSEA_User_Guide/).
'''), code('''
from src.pathway_enrichment import cache_hallmark
from src.single_donor_workflow import characterize_donor

GMT_PATH = DIRS['gsea'] / 'MSigDB_Hallmark_2020.gmt'
# For a custom GMT, replace GMT_PATH and omit cache_hallmark below.
print('Cached collection SHA256:', cache_hallmark(GMT_PATH))
programs, enrichment, universe = characterize_donor(
    rna_data, OUTPUT_ROOT, GMT_PATH, permutations=PERMUTATIONS,
    seed=RANDOM_SEED, detection=MIN_DETECTION, median=COMPUTE_MEDIANS)
print('Shared eligible genes:', len(universe))
display(pd.read_csv(DIRS['gsea'] / 'gene_set_coverage.csv'))
for dataset in ['cite', 'multiome']:
    print(dataset)
    display(programs[dataset].sort_values('specificity', ascending=False).groupby('cell_type').head(5))
    display(enrichment[dataset].sort_values(['cell_type', 'q_value']).groupby('cell_type').head(5))
    display(Image(filename=str(DIRS[f'rna_{dataset}'] / 'figures/characteristic_genes.png')))
    figure = DIRS[f'rna_{dataset}'] / 'figures/pathway_enrichment.png'
    if not enrichment[dataset].empty and figure.exists():
        display(Image(filename=str(figure)))
'''), markdown('''
## Main findings
The next cell reports observed counts and test coverage only. Inspect the saved
rankings and leading-edge genes before writing a biological interpretation. No
biological finding is pre-filled in this unexecuted notebook.
'''), code('''
print(f'Donor {donor}: {int(cell_counts.included.sum())} retained shared populations; {len(universe)} comparison genes.')
for dataset, result in enrichment.items():
    print(dataset, ':', len(result), 'pathway/type tests;', int((result.q_value <= 0.05).sum()),
          'with within-ranking BH q <= 0.05 (exploratory).')
print('Next: run nb05 to test concordance and sampling/site sensitivity before any multimodal follow-up.')
'''), markdown('''
## Limitations
- One healthy donor: no disease/control contrast and no inference across individuals.
- Separate captures: broad cell types may contain different state/subtype mixtures.
- Unequal capture sizes are reported; nb05 performs independent balanced subsampling.
- Site composition may confound pooled donor summaries; nb05 checks shared sites.
- Library normalization uses slightly different measured GEX panels; equal processing
  cannot remove capture, depth, ambient RNA or annotation differences.
- This workflow uses organizer-processed cells with no new QC threshold imposed;
  examine nb01's QC findings and depth summaries before interpretation.
- Gene-set p-values are competitive, exploratory statistics, not donor-level tests.
- Later ADT evidence will cover a limited panel; unmeasured proteins are not negatives.
- Later ATAC gene activity is an approximation, and cross-sectional data cannot prove
  ATAC → RNA → protein causality or temporal priming.
''')]

nb05 = [markdown('''
# nb05 — RNA concordance across independent captures

## Scientific question
For one donor, do CITE-seq and Multiome recover the same cell-type-specific genes
and biological pathways, sufficiently consistently to motivate multimodal follow-up?

## Analysis strategy
Compare independent cell-type mean profiles on one shared gene universe, inspect
specificity-rank and top-gene agreement, and compare pathway NES with statistical
support. Repeat expression comparisons with equal cell counts and inspect shared
sites. Shortlist replicated positive programs by effect, shared leading edge and
cell-type diversity. Explore a shared uncorrected PCA of population means only.

Run nb04 first. This notebook reads its saved outputs and verifies hashes to reject
mixed/stale runs. It never aligns CITE and Multiome cells by row or barcode.
'''), SETUP, code('''
from src.single_donor_workflow import load_characterization, run_concordance
manifest, programs, enrichment, universe, rna_data = load_characterization(OUTPUT_ROOT)
display(manifest)
print('Donor:', manifest['donor'], '| Shared genes:', len(universe))
'''), markdown('''
## Expression, ranked-gene and pathway concordance
Pearson/Spearman correlations use independent within-cell-type average RNA profiles.
Specificity-rank Spearman is computed over the complete shared eligible gene list.
Top-50/100/250 overlap uses only positive-specificity genes, with actual list lengths
reported when fewer genes qualify. No gene-correlation p-values are interpreted as
evidence across individuals.

Pathway classes use q ≤ 0.05 in both captures for concordant active/negative or
opposite-direction discordant results. A pathway significant in only one capture is
labelled capture-specific **descriptively**; a difference in significance does not
establish a significant difference. Non-significant results are unsupported, never
automatically inactive. Negative enrichment denotes relative underrepresentation.

Five balanced sampling repeats use independent draws in each capture and the same
cell count per cell type (minimum of both available counts and 1000). This is a
sensitivity analysis, not five biological replicates. One balanced draw also repeats
enrichment with the original shared gene universe and collection. Site-specific
comparisons require at least two adequately represented shared cell types.
'''), code('''
BALANCED_REPEATS = 5
BALANCED_CAP = 1000
expression_comparison, pathway_comparison, followup, findings = run_concordance(
    OUTPUT_ROOT, repeats=BALANCED_REPEATS, cap=BALANCED_CAP)
display(expression_comparison)
display(pd.read_csv(DIRS['rna_concordance'] / 'top_gene_overlap.csv'))
display(pathway_comparison[['cell_type', 'pathway', 'NES_cite', 'NES_multiome', 'q_value_cite', 'q_value_multiome', 'concordance']])
display(pd.read_csv(DIRS['rna_concordance'] / 'pathway_correlations.csv'))
for name in ['expression_scatter', 'pathway_nes_scatter', 'pathway_concordance_heatmap']:
    figure = DIRS['rna_concordance'] / 'figures' / f'{name}.png'
    if figure.exists() and (name == 'expression_scatter' or not pathway_comparison.empty):
        display(Image(filename=str(figure)))
'''), markdown('''
## Unequal cell numbers and donor/site composition
Compare the balanced ranges below to the complete-profile estimates above. A range
across draws describes sampling sensitivity, not confidence across donors. Shared
site analyses change the comparator pool when some populations are absent; inspect
their coverage table. If no shared site is testable, site confounding remains
unresolved. With one selected donor, donor effects cannot be estimated.
'''), code('''
sensitivity = pd.read_csv(DIRS['rna_concordance'] / 'balanced_expression_sensitivity.csv')
display(sensitivity.groupby('cell_type')[['pearson', 'spearman', 'specificity_rank_spearman']].agg(['min', 'median', 'max']))
display(pd.read_csv(DIRS['rna_concordance'] / 'balanced_pathway_sensitivity.csv'))
display(pd.read_csv(DIRS['rna_concordance'] / 'site_sensitivity_coverage.csv'))
display(pd.read_csv(DIRS['rna_concordance'] / 'site_expression_sensitivity.csv'))
'''), markdown('''
## Candidate RNA programs for follow-up
Require replicated positive enrichment, at least two shared leading-edge genes and
positive NES in the balanced sensitivity run. A round-robin across cell types ranks
by the weaker NES, limits redundant leading edges and yields **up to ten** candidates;
it does not force five when evidence is insufficient. No candidate is chosen solely
by p-value. Inspect biological relevance and pathway redundancy explicitly: a list
of statistically supported pathways is not automatically a diverse biological panel.

The saved `biological_review` field remains pending. Use the table and leading edges
to choose interpretable programs before implementing nb06. Scarce Hallmark coverage
of antigen presentation, cytotoxicity or stemness may justify a later, separately
documented Reactome/GO run; do not invent these findings or combine mismatched runs.
'''), code('''
display(followup[['cell_type', 'pathway', 'NES_cite', 'NES_multiome', 'shared_leading_edge',
                  'balanced_direction_supported', 'selected_for_followup', 'reason', 'biological_review']])
print('Candidate programs:', int(followup.selected_for_followup.sum()))
'''), markdown('''
## Exploratory RNA bridge assessment
This PCA uses a common gene basis and uncorrected population mean RNA, with equal
weight for every dataset × cell-type centroid. It does not compare unrelated PCA axes
from the original objects. Inspect whether equivalent cell types are near each other
and whether capture identity separates them. Centroid agreement does not establish
cell-level latent alignment. No aggressive batch correction, latent-space transfer,
multi-donor pseudobulk or full integration is implemented.
'''), code('''
display(pd.read_csv(DIRS['rna_concordance'] / 'shared_centroid_pca.csv'))
display(Image(filename=str(DIRS['rna_concordance'] / 'figures/shared_centroid_pca.png')))
'''), markdown('''
## Main findings
The quantitative report below is computed from this run. Use correlations, rank
overlap, pathway coverage, balanced draws, site checks and leading edges jointly to
assess the RNA bridge. No universal pass threshold or biological conclusion is
pre-filled. **Review these results before implementing ATAC and protein analyses.**
'''), code('''
display(findings)
print('Saved tables, figures and provenance:', OUTPUT_ROOT)
'''), markdown('''
## Limitations
- One donor cannot establish reproducibility across individuals or donor effects.
- Separate CITE/Multiome cells prohibit cross-capture cell-level association claims.
- Broad lineage agreement can conceal subtype composition and capture differences.
- Correlation can be driven by abundant/housekeeping genes: inspect specificity and
  pathway agreement alongside mean expression, and check the technical gene flags.
- Gene-set permutation p-values ignore gene correlation; q-values describe the
  chosen exploratory test family, not condition or individual-level inference.
- Balanced subsampling cannot correct unmeasured technical or site confounders.
- Negative specificity is not biological inactivity; missing evidence is not absence.
- Subsequent protein coverage is limited, and ATAC gene activity approximates
  regulatory accessibility. Neither snapshot establishes causality or temporal priming.
- nb06–nb08 are deliberately deferred until this RNA milestone runs and is reviewed
  on the real benchmark files in Drive, following the requested incremental sequence.
''')]


def write(name, cells):
    for i, cell in enumerate(cells):
        cell['id'] = f'{name[:4]}-{i:03d}'
    payload = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                'language_info': {'name': 'python', 'version': '3.11'}}, 'nbformat': 4, 'nbformat_minor': 5}
    (ROOT / 'notebooks' / name).write_text(json.dumps(payload, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    write('nb04_single_donor_rna_characterization.ipynb', nb04)
    write('nb05_rna_concordance.ipynb', nb05)
