"""Create nb08 only; never overwrite executed notebooks."""
import json
from pathlib import Path
from build_single_donor_notebooks import code, markdown

cells = [markdown('''
# nb08 — Multilayer pathway states

## Scientific question
Which selected programs have concordant RNA, accessibility and measured protein
evidence, and which have inconsistent or missing support?

## Analysis strategy
Summarize saved nb04–07 tables for the same donor and selected cell-type/pathway
panel. Pairing is retained only within each capture. RNA enrichment, relative
population activity and within-type associations are different measurements and
are displayed separately. No combined activation score or forced binary label.

Copy this notebook plus `src/multimodal_summary.py` and `src/multimodal_plots.py`
to your existing Drive project. Earlier shared modules and outputs must remain
available. No raw h5ad reads or reruns of nb04–07 are required.
'''), code('''
from pathlib import Path
import sys
try:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_PATH = Path('/content/drive/MyDrive/multiomics-relationship-modeling')
except ImportError:
    BASE_PATH = Path.cwd().resolve()
    if not (BASE_PATH / 'src').is_dir():
        BASE_PATH = BASE_PATH.parent
sys.path.insert(0, str(BASE_PATH))
import pandas as pd
from IPython.display import display, Image
from src.multimodal_summary import run_summary
OUTPUT_ROOT = BASE_PATH / 'results/single_donor'
RELATIVE_PROTEIN_THRESHOLD = 0.5
summary, proteins, provenance = run_summary(OUTPUT_ROOT, threshold=RELATIVE_PROTEIN_THRESHOLD)
print('Donor:', provenance['donor'])
print('Saved results:', OUTPUT_ROOT / 'multilayer')
'''), markdown('''
## Program evidence across captures
RNA NES/q-values describe cell-type enrichment in each capture. Multiome RNA/ATAC
means use the same covered genes, each standardized within its own layer. Protein
coverage is the fraction of original GMT genes with directly mapped ADTs, not the
fraction of pathway function measured. Missing protein coverage cannot be low activity.
'''), code('''
with pd.option_context('display.max_columns', None, 'display.max_colwidth', 100):
    display(summary[['cell_type', 'pathway', 'NES_cite', 'q_value_cite', 'NES_multiome', 'q_value_multiome',
                     'activity_rna_mean', 'activity_atac_mean', 'activity_state', 'atac_coverage',
                     'atac_adjusted_pearson', 'atac_site_values', 'protein_n_direct_genes',
                     'protein_direct_gene_coverage', 'protein_interpretation']])
'''), markdown('''
## Protein states and site sensitivity
Show each ADT independently. Its mean z-score is relative to the donor's equal-type
reference: high >= +0.5, low <= -0.5, otherwise intermediate. These are exploratory
cutoffs, not significance or absolute protein activity. CITE RNA module means have
different coverage from nb06 and are not directly comparable numerical effect sizes.

Site sign consistency has no minimum effect threshold and is **not** a statistical
support label: inspect magnitudes. Untested sites are excluded from sign counts and
remain visible. Same-gene and module associations answer different questions.
'''), code('''
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(proteins[['cell_type', 'pathway', 'adt', 'evidence_type', 'rna_module_mean',
                      'adt_mean_z', 'relative_protein_state', 'pearson', 'adjusted_pearson',
                      'same_gene_adjusted_pearson', 'n_testable_sites', 'n_sites', 'site_values']])
    display(pd.read_csv(OUTPUT_ROOT / 'multilayer/same_gene_site_sensitivity.csv'))
for name in provenance['figures']:
    display(Image(filename=str(OUTPUT_ROOT / 'multilayer/figures' / name)))
'''), markdown('''
## State-threshold sensitivity and interpretation
Keep nb06's intermediate states. Elevated RNA and ATAC means do not imply positive
within-type covariation, and positive covariation does not imply elevated means.
High accessibility with low RNA is at most a potentially permissive relative state;
temporal priming cannot be established. RNA/protein disagreement is a candidate
observation subject to coverage, normalization and confounding, not proven decoupling.
'''), code('''
display(pd.read_csv(OUTPUT_ROOT / 'multilayer/atac_state_sensitivity.csv'))
for threshold in [0.25, 0.5, 0.75]:
    view = proteins[['cell_type', 'pathway', 'adt', 'adt_mean_z']].copy()
    view['threshold'] = threshold
    view['state'] = view.adt_mean_z.map(lambda x: 'unmeasured/constant' if pd.isna(x) else 'high' if x >= threshold else 'low' if x <= -threshold else 'intermediate')
    display(view)
'''), markdown('''
## Main findings
The reviewed donor-15078 nb07 run identified CD14 inflammatory response as the
clearest measured-subset protein result: CD14/CD88/CD54 module associations remained
positive across four sites, with clearer same-gene agreement for CD14 and CD88.
TNF/NF-kB showed weaker overlapping support. Interferon/E2F lacked consistent positive
adjusted protein support, and Myc had no direct ADT coverage. These are historical
review notes; verify the generated tables if inputs or donor change.

Do not automatically label the five programs coordinated multimodal activation:
population means, paired associations and limited surface measurements must be
considered together. Save the generated evidence tables for review before advancing.

## RNA as a future bridge and unresolved checks
The earlier nb05 review found median mean-expression Pearson about 0.628 and
specificity Spearman about 0.629; its first PC explained 49.8% with capture separation.
These historical results motivate cautious exploration, not readiness for aggressive
integration. Cross-capture site-specific pathway replication and the seven unavailable
nb05 NES values remain unresolved; nb08 does not silently mark them complete.
No latent integration, cell matching, or multi-donor pseudobulk is implemented.

## Limitations
One donor, RNA-selected programs, overlapping gene sets, processed ATAC activity,
limited surface panel and relative scaling constrain conclusions. Depth adjustment
can remove biology; site ranges are not confidence intervals. Older stages lack
hashes of their own output tables, so nb08 records input fingerprints but cannot
prove those tables were never edited. No causal, temporal or donor-generalization
claim follows from this descriptive synthesis.
''')]

if __name__ == '__main__':
    path = Path(__file__).resolve().parents[1] / 'notebooks/nb08_multilayer_pathway_states.ipynb'
    if path.exists():
        raise FileExistsError('Preserve executed nb08 before regenerating')
    for i, cell in enumerate(cells):
        cell['id'] = f'nb08-{i:03d}'
    path.write_text(json.dumps(dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=5), indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
