"""Build the follow-up investigation without editing nb01–14."""
from pathlib import Path
import nbformat as nbf
root=Path(__file__).resolve().parents[1]
cells=[]
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))
def section(title,question,why,source):
    md(f'# {title}\n\n## Scientific question\n\n{question}\n\n## Why this diagnostic matters\n\n{why}')
    code(source)
md('''# nb14b — Investigate persistent capture structure

Follow-up to nb14: separate donor/site composition, within-population capture
structure, shared versus heterogeneous capture directions, and recurring RNA
differences. **No generative model retraining and no paired-cell assumptions.**

Use the exact nb14 evaluation sample. Comparisons require ≥50 cells per capture
within donor × site × harmonized population. Balance up to 250 per capture within
these strata using seeds 42–44. Population classifiers require at least two training
donors, and hold out the entire test donor. Standardization and projection directions
use only training donors. Both captures remain represented at matched sites.

Sequential linear classifier directions are removed only as a diagnostic. Compare
rank 1, rank 3 and a random rank-3 negative control. Each held-out donor is evaluated
separately: **fold-specific coordinates are never combined into a corrected space**.
Refit classifiers after projection to detect remaining linearly predictable capture
information. Global mixing and purity use all cells of that held-out donor; group
alignment uses adequately sized site/population strata. Do not compare those global
fractions directly with nb14's all-donor neighborhoods.

Also report removed within-stratum variance and an orientation-free AUROC diagnostic:
max(AUROC, 1−AUROC) distinguishes reversed held-donor relations from loss of
separability. This uses test labels post hoc and is **not** predictive accuracy.
An RNA-QC-only classifier uses shared RNA depth, detected genes, mitochondrial
fraction and ribosomal fraction. These features are not assumed purely technical.

Biological checks use full-RNA CD14 module scores, within donor/site cross-capture
neighbors, and 100 reference-score permutations. Average sites/directions within
donors before summarizing changes. Gene comparisons use pseudobulk log2(CPM+1),
equal-site averaging within donor/population, ≥4 donors and descriptive sign
consistency. They do not establish technical causation or cell-level significance.

Exploratory practical screens: mixing gain ≥0.02, purity loss ≤0.02, module rho
loss ≤0.05. A learned rank-3 effect must exceed the random control by ≥0.01 to
support a useful geometric contribution. A single random basis is a sensitivity
control, not a complete null distribution. Report paired donor bootstrap intervals
(2,000 resamples); eight donors give limited precision. No thresholds are optimized.
''')
code('''from pathlib import Path
import sys
from IPython.display import display, Markdown, Image
try:
    from google.colab import drive
except ImportError:
    BASE=Path.cwd().resolve()
    if not (BASE/'src').exists(): BASE=BASE.parent
else:
    drive.mount('/content/drive')
    BASE=Path('/content/drive/MyDrive/multiomics-relationship-modeling')
sys.path.insert(0,str(BASE))
from src.capture_structure_diagnostics import CaptureInvestigation
run=CaptureInvestigation(BASE)
print('Results:',run.output)''')
section('1. Assay/site overlap','Which cells can be compared at the same donor, site and population?',
        'Recover full RNA metadata and frozen nb14 scores. Explicitly count excluded and one-capture strata.',
        '''coverage=run.load()
display(coverage)
print('Matched cells:',run.eligible.sum(),'of',len(run.obs))
display(run.obs.groupby(['DonorID','Site','assay'],observed=True).size().unstack(fill_value=0))''')
section('2. Composition-controlled neighborhoods','Does site matching and equal capture abundance resolve alignment concerns?',
        'Use group-restricted neighbors, inherited nb14 thresholds and three balanced samples. Summaries are descriptive; groups share donors.',
        'display(run.composition())')
section('3. Held-donor capture prediction','Can capture still be predicted within the same population on unseen donors?',
        'Balanced logistic regression on joint, restricted RNA PCA and broad RNA PCA. Each donor/site/population/capture receives equal training weight; type-specific models exclude other populations.',
        'display(run.predict())')
section('4. Common versus heterogeneous capture geometry','Are capture shifts consistent across donors and populations?',
        'Compute partial R² after matched-stratum intercepts, with common or stratum-specific capture effects. Compare centroid-offset cosines against references excluding the current donor. These are associations, not unique variance causally explained.',
        '''display(run.geometry())
display(run.offset_geometry.groupby('reference').cosine.agg(['median','min','max']))''')
section('5. Held-donor geometric perturbations','Does removing a small learned capture subspace improve neighborhoods and preserve biology?',
        'Learn rank-1/rank-3 classifier normals only on other donors. Project the held-out donor without using its capture labels for transformation. Refit capture classifiers on projected training data. Evaluate CD14 cross-capture gradients within the same donor/site, with repeated reference permutations.',
        '''display(run.perturb())
display(run.gradients.groupby(['arm','module'])[['local_rho','cross_rho','excess_over_null']].median())''')
section('6. Recurrent RNA capture differences','Which relative RNA expression differences recur across independent donors?',
        'Aggregate raw RNA counts per donor/site/population/capture, compare log2(CPM+1), and average sites within donors. Report genes with ≥75% directional agreement and |median difference|≥1 across ≥4 donors. Audit full RNA depth, detected genes, original annotation composition and validated CD14 program overlap. Differences may reflect capture technology or population/state sampling.',
        '''display(run.rna())
display(run.rna_effects[run.rna_effects.cell_type.eq('CD14 monocytes')].head(30))
display(run.rna_geometry.groupby('reference').cosine.agg(['median','min','max']))''')
section('7. Evidence and modeling decision','What now explains the failure, and what should be tested before nb15?',
        'Conclusions follow the executed matched-group, held-donor, RNA and full-module analyses. Avoid turning a diagnostic projection into an approved correction.',
        '''deltas,decisions,report=run.finish()
display(deltas)
display(decisions)
for path in sorted((run.output/'figures').glob('*.png')):
    display(Image(filename=str(path)))
display(Markdown(report))''')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}})
path=root/'notebooks/nb14b_capture_structure_investigation.ipynb'
nbf.write(nb,path)
print(path)
