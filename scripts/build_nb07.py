"""Generate only nb07; preserve all executed notebooks."""
import json
from pathlib import Path
from build_single_donor_notebooks import code, markdown, SETUP

ROOT = Path(__file__).resolve().parents[1]
cells = [markdown('''
# nb07 — CITE protein support for reproducible RNA programs

## Scientific question
Do individual measured surface proteins covary with the same five RNA programs
reviewed in nb06, within paired CITE cells of the selected donor?

## Analysis strategy
Reuse nb04's CITE RNA, nb05's supported shortlist and nb06's selected panel.
Audit antibody labels before distinguishing **direct molecular matches** from
**exploratory phenotypic context**. No CITE cells are matched to Multiome cells.
Missing ADTs provide no evidence about protein activity. Intracellular E2F/Myc
programs may have no useful surface-panel coverage.

Copy this notebook, `src/protein_programs.py` and `src/protein_plots.py` into the
existing Drive project. Keep the shared modules used for nb04–06. Restart a reused
Colab runtime after copying modules. Run cells in order.
'''), SETUP, code('''
import shutil
from src.protein_programs import run_protein

MIN_CELLS_FOR_ASSOCIATION = 50
CITE_INPUT = CITE_H5AD
# Reading HDF5 repeatedly over Drive can be slow. Copy once per notebook run.
if str(BASE_PATH).startswith('/content/drive/'):
    CITE_INPUT = Path('/content/nb07_cite_processed.h5ad')
    required = CITE_H5AD.stat().st_size
    if shutil.disk_usage('/content').free < required:
        raise RuntimeError('Insufficient Colab disk space for local CITE copy')
    print(f'Copying {required / 1e6:.0f} MB from Drive to Colab local disk...', flush=True)
    shutil.copyfile(CITE_H5AD, CITE_INPUT)
    assert CITE_INPUT.stat().st_size == required
    print('Local copy ready.', flush=True)
'''), markdown('''
## Representation and safeguards
RNA retains nb04's log1p(CP10k) values. Each module uses variable pathway genes
in the shared RNA universe, with equal cell-type weight in the reference mean and
variance. Require at least 10 genes. This score has a different gene coverage from
nb06 and must not be compared numerically across assays as an absolute effect.

ADT comes from raw `layers['counts']`, aligned by CITE cell ID and checked against
donor, site and original cell type. Exclude labelled isotype controls and zero-total
biological-ADT cells. Primary transform is explicitly **log(1+count) minus the
cell's mean log(1+count) over biological ADTs**. This relative panel transformation
is not background correction. Uncentered log1p ADT is a sensitivity comparison.

CD3, CD8, CD16/CD32, CD57, CD45RA/RO and HLA family labels are not assumed to
measure one unique gene. Consult the saved mapping audit and clone metadata.
Direct matches mean mapped genes belonging to the GMT pathway, not validation of
the whole pathway. Individual ADTs are reported without a forced aggregate score.

CD86 and HLA-DR are exploratory surface context for the three immune programs.
Their functions are described by [UniProt CD86](https://www.uniprot.org/uniprotkb/P42081/entry)
and [UniProt HLA-DRA](https://www.uniprot.org/entry/P01903); these references do not
establish a specific relationship or expected correlation direction with our RNA
scores. HLA-DR is not a direct HLA-DRA molecular match. No forced E2F/Myc proxies.

Report target-population and within-site Pearson/Spearman correlations. Residual
Pearson adjusts both measurements for log RNA/ADT count totals and site indicators;
residual Spearman is correlation of OLS residual ranks. These are sensitivity
analyses, not causal estimates. No cell-level p-values or donor inference.
'''), code('''
tables, protein_manifest = run_protein(
    OUTPUT_ROOT, CITE_INPUT, min_cells=MIN_CELLS_FOR_ASSOCIATION)
PROTEIN_DIR = OUTPUT_ROOT / 'protein'
print('Donor:', protein_manifest['donor'])
display(tables['coverage'].drop(columns='rna_genes'))
display(tables['cell_qc'].groupby(['cell_type_harmonized', 'included_protein'], observed=True).size().rename('n_cells').reset_index())
'''), markdown('''
## Panel audit and evidence tiers
Review unresolved antigens and absent candidates before interpreting correlations.
Measured direct ADTs with constant or insufficient data remain untested, not negative.
'''), code('''
with pd.option_context('display.max_rows', None, 'display.max_colwidth', 100):
    display(tables['mapping'])
    display(tables['evidence'])
display(tables['population_summary'])
'''), markdown('''
## Paired within-population relationships
Compare module–ADT associations to the direct same-gene RNA–ADT associations.
Inspect site heterogeneity, depth adjustment and normalization sensitivity. Small
site groups remain visible but untested. Scatter plots display at most 1,500 cells
per panel; statistics use all eligible cells. Up to three ADTs per program are
plotted in mapping order, not selected by strongest correlation.
'''), code('''
assoc = tables['associations']
display(assoc[assoc.scope == 'target'])
display(assoc[assoc.scope == 'target_by_site'])
display(tables['direct_gene_associations'])
for name in protein_manifest['figures']:
    display(Image(filename=str(PROTEIN_DIR / 'figures' / name)))
'''), markdown('''
## Main findings
No findings are pre-filled. Review direct coverage first, then effect sizes and
sensitivity. Positive measured-subset association, phenotypic context and no direct
coverage are distinct outcomes. Lack of correlation is not evidence of absent
translation or RNA–protein decoupling without considering measurement limitations.
'''), code('''
display(tables['coverage'][['program_id', 'cell_type', 'pathway', 'n_direct_genes', 'protein_coverage_status']])
print('Saved CSVs, PNG/PDF figures and provenance:', PROTEIN_DIR)
print('Elapsed minutes:', round(protein_manifest['elapsed_minutes'], 1))
print('Review these outputs before building nb08 multilayer summaries.')
'''), markdown('''
## Limitations
One donor; no independent donor replication. Limited surface panel, clone ambiguity,
no empty-droplet/background correction, and compositional ADT normalization constrain
interpretation. Adjusting library depth may remove biology as well as technical
variation. Broad cell labels may mix states. No whole-pathway protein validation,
cross-capture cell pairing, causal inference or temporal-priming claim is supported.
Site-specific cross-capture RNA pathway replication remains a separate check.
''')]

if __name__ == '__main__':
    for i, cell in enumerate(cells):
        cell['id'] = f'nb07-{i:03d}'
    path = ROOT / 'notebooks/nb07_pathway_protein_support.ipynb'
    if path.exists():
        raise FileExistsError('Refusing to overwrite nb07; preserve executed results')
    notebook = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.11'}}, nbformat=4, nbformat_minor=5)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
