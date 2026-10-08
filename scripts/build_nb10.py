"""Generate nb10 only; never modify nb01–09."""
import json
from pathlib import Path
from build_single_donor_notebooks import code,markdown

cells=[markdown('''
# nb10 — Shared RNA integration

## Scientific question
Can CITE and Multiome cells occupy a common RNA-defined space while preserving
biological identity and reducing assay structure? This is an RNA bridge evaluation,
not final multimodal integration. CITE and Multiome cells are different cells.
Neighbors indicate similar measured states, never one-to-one cell matches.

## Analysis strategy
Use nb09's eight eligible donors and saved harmonization mapping; retain all RNA
cells in those donors, including labelled unresolved populations. Use only shared
GEX genes and raw counts. Build uncorrected PCA/neighbors/UMAP, then one primary
assay-only scVI model with the same HVGs. No cell-type supervision, donor covariates,
site covariates, ADT or ATAC features enter training. Evaluate both spaces on the
same sampled cells, including population/donor agreement and CD14 score gradients.

Copy this notebook and `src/rna_integration.py`, `src/rna_integration_metrics.py`,
`src/rna_integration_plots.py` to the existing Drive project. nb01–09 stay unchanged.
A Colab GPU runtime is recommended for scVI; CPU is allowed but may be slow.
'''),code('''
from pathlib import Path
import sys, subprocess, json
try:
    from google.colab import drive
    IN_COLAB=True
except ImportError:
    IN_COLAB=False
if IN_COLAB:
    subprocess.check_call([sys.executable,'-m','pip','install','-q','scanpy>=1.11,<1.12','scvi-tools>=1.3,<1.5','anndata>=0.11,<0.13'])
    drive.mount('/content/drive')
    BASE_PATH=Path('/content/drive/MyDrive/multiomics-relationship-modeling')
else:
    BASE_PATH=Path.cwd().resolve()
    if not (BASE_PATH/'src').is_dir(): BASE_PATH=BASE_PATH.parent
sys.path.insert(0,str(BASE_PATH))
import pandas as pd
from IPython.display import display,Image
from configs.paths import CITE_H5AD,MULTIOME_H5AD
from src.rna_integration import run_integration
COHORT=BASE_PATH/'results/cross_donor/nb09_run01'
DISCOVERY=BASE_PATH/'results/single_donor'
OUTPUT=BASE_PATH/'results/shared_rna_integration/run03'
PATHS={'cite':CITE_H5AD,'multiome':MULTIOME_H5AD}
SEED=42
N_HVGS=3000
N_LATENT=30
MAX_EPOCHS=300
KL_WARMUP_EPOCHS=50
MIN_EPOCHS=100
EARLY_STOPPING_PATIENCE=30
ACCELERATOR="auto"
BATCH_SIZE=128
display(pd.read_csv(COHORT/'donor_eligibility.csv'))
print('Model: raw RNA counts; batch_key=assay only; no cell-type labels supplied.')
'''),markdown('''
## Q1 — Do RNA profiles already occupy similar biological spaces?
First inspect the source representation and cohort counts. The loader samples RNA
entries from .X and layers/counts, recording integer fraction, range and row sums.
This cannot reconstruct preprocessing history; .X is never assumed comparable.
All selected raw GEX counts are checked for finite nonnegative integer values.

Both assays are subset to the same shared genes **before** per-cell normalization
to 10,000 and log1p. Zero shared-count cells are audited and excluded. This differs
from earlier full-GEX-panel normalization and is documented rather than concealed.
All metadata are retained; cell IDs are prefixed by assay to avoid barcode collision.

HVGs use Scanpy's log-normalized Seurat dispersion method, batch_key=assay, on equal
random samples of up to 10,000 cells per assay. No cell-type labels select HVGs.
Unequal biological composition can still influence selection. PCA uses centered
log-expression on those HVGs without per-gene unit-variance scaling.

References: [Scanpy HVGs](https://scanpy.readthedocs.io/en/latest/api/generated/scanpy.pp.highly_variable_genes.html),
[scVI model API](https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCVI.html).
'''),code('''
# Optional fast local copies: originals stay on Drive, results are written to OUTPUT.
import shutil
if IN_COLAB:
    local=Path('/content/nb10_inputs'); local.mkdir(exist_ok=True)
    needed=sum(p.stat().st_size for p in PATHS.values())
    if shutil.disk_usage(local).free < needed:
        raise RuntimeError('Not enough local disk for input copies; choose a larger runtime or skip copying')
    for assay,source in list(PATHS.items()):
        print('Copying',assay,flush=True)
        dest=local/source.name
        shutil.copyfile(source,dest)
        PATHS[assay]=dest
# Fresh folder required: partial/completed runs are preserved, no automatic resume.
comparison,types,donors,gradients,criteria=run_integration(
    PATHS,COHORT,DISCOVERY,OUTPUT,n_hvgs=N_HVGS,n_latent=N_LATENT,
    max_epochs=MAX_EPOCHS,seed=SEED,accelerator=ACCELERATOR,batch_size=BATCH_SIZE,
    kl_warmup_epochs=KL_WARMUP_EPOCHS,min_epochs=MIN_EPOCHS,
    early_stopping_patience=EARLY_STOPPING_PATIENCE)
display(json.loads((OUTPUT/'training_summary.json').read_text()))
'''),code('''
import json
audit=json.loads((OUTPUT/'input_audit.json').read_text())
display(audit)
for field in ['assay','DonorID','Site','cell_type_harmonized']:
    display(pd.read_csv(OUTPUT/f'counts_by_{field}.csv'))
display(Image(filename=str(OUTPUT/'figures/cell_counts.png')))
'''),markdown('''
## Q2 — How much structure is associated with assay, donor, site and cell type?
Inspect the uncorrected panels first. Quantify silhouette, local label purity,
majority-neighbor agreement and normalized entropy for each metadata field. Cell-type
purity and kNN label agreement are related, not independent validations. Metrics are
computed in PCA/latent space, not UMAP. Donor/site associations are diagnostic: they
do not establish which variable caused separation, and high mixing is not always good.

Metrics use the same fixed sample with equal maximum donor/assay quotas, up to 20,000
cells; silhouette uses a fixed subset up to 3,000. Rare groups may have insufficient
evaluation cells despite sufficient full-cohort counts. Unresolved labels stay in
training/plots but are excluded from labelled biological evaluation. No cell-level
p-values or claims that cells are independent human replicates.
'''),code('''
for field in ['cell_type_harmonized','assay','DonorID','Site']:
    display(Image(filename=str(OUTPUT/f'figures/umap_{field}.png')))
display(comparison)
display(Image(filename=str(OUTPUT/'figures/metric_comparison.png')))
'''),markdown('''
## Q3 — Does correction improve mixing while preserving cell identity?
The scVI model uses raw HVG counts, assay as its only batch variable, negative-binomial
likelihood, two hidden layers and 30 latent dimensions. Training has a 90% training
split, KL warm-up of 50 epochs, at least 100 and at most 300 epochs. Early stopping
monitors validation ELBO only after warm-up, with patience 30 and validation every
epoch. The final model weights are saved. Record actual package
versions, training history and saved model. No hyperparameter search is performed.
scVI and PCA differ in model as well as correction, so this is a practical before/
after comparison, not causal isolation of an assay-correction effect.

Global mixing may improve by merging unrelated types. Require preserved cell-type
purity (loss no more than 0.02) and silhouette (loss no more than 0.05), then inspect
per-type results. These tolerances are exploratory choices fixed before results,
not universal validation standards. Inspect training histories for convergence.
'''),markdown('''
## Q4 — Does alignment work across populations?
Within each type, rebuild neighbors and give CITE/Multiome query cells equal weight.
Divide cross-assay neighbor fraction by its expectation under random mixing at the
observed assay proportions. A ratio near one reflects composition-adjusted mixing.
The centroid distance is divided by pooled within-assay RMS radius, because raw
distances in PCA and scVI have incompatible scales. Different distributions can
share centroids, so use both diagnostics.

Exploratory population labels: supported if mixing ratio >=0.8 and scaled distance
<=0.5; concern if ratio <0.5 or distance >1; otherwise partially supported. Require
50 evaluation cells per assay. Retain insufficient groups and inspect minority types.
'''),code('''
with pd.option_context('display.max_rows',None,'display.max_columns',None):
    display(types.sort_values(['space','mixing_ratio'],ascending=[True,False]))
for name in ['celltype_assay_mixing','celltype_centroid_distance']:
    display(Image(filename=str(OUTPUT/f'figures/{name}.png')))
'''),markdown('''
## Q5 — Do equivalent donor × cell-type populations agree?
Evaluate centroid separation and cross-assay neighbors within each donor/type on the
fixed evaluation sample. RNA profile Pearson correlation uses full-group mean
normalized shared-gene expression and is unchanged by integration: it is independent
context, not an improvement caused by the model. Full and evaluation counts are both
reported. Most donors occur at one site; donor/site biology is partly confounded.
'''),code('''
with pd.option_context('display.max_rows',None,'display.max_columns',None):
    display(donors)
'''),markdown('''
## Q6 — Are the validated CD14 gradients preserved?
Use the original two GMT sets and frozen nb09 universe. Scores average variable
gene z-scores from normalized shared RNA, using common moments with equal weight
per donor × assay × nb09-reference-type group. This preserves assay offsets rather
than forcing them to vanish. It changes score scaling from nb09, not membership.
The modules are evaluated after unsupervised fitting, not supplied as training labels.

An unchanged expression score colored on UMAP alone cannot prove preservation.
Within CD14 donor groups, predict each cell's score by averaging its opposite-assay
neighbors' scores, then compare with its observed score using Spearman correlation
before and after integration. Report both directions and a one-shuffle diagnostic
(not a calibrated null test). Correlated gene expression is also used to train the
embedding, so this is internal consistency, not independent biological validation.
'''),code('''
for number in [1,2]:
    for suffix in ['umap','distribution_assay','distribution_DonorID','distribution_cell_type_harmonized']:
        display(Image(filename=str(OUTPUT/f'figures/module_{number}_{suffix}.png')))
with pd.option_context('display.max_rows',None):
    display(gradients)
'''),markdown('''
## Q7 — Is RNA supported as a bridge for the next phase?
The criteria table combines global mixing, identity preservation, population-specific
failures, donor agreement and module-neighbor gradient consistency. It can report
supported, partially supported or concern; there is no forced overall success flag.
Donor support uses descriptive median scaled distance <=0.5 and RNA correlation >=0.5.
CD14 gradient support uses median rho >=0.2 and median decline no worse than 0.05.
Mixing support uses global increase >=0.05, always conditional on preserved biology.
Inspect individual groups; good medians cannot excuse a collapsed population.
'''),code('''
with pd.option_context('display.max_colwidth',None):
    display(criteria)
print('Concern populations:')
display(types[(types.space=='integrated') & (types.alignment_status=='concern')])
print('Results and model:',OUTPUT)
'''),markdown('''
## Previous run and new evaluation
Local run02 completed 100 epochs, but KL weight reached only 0.2475 and validation
ELBO was best at the final epoch. Assay mixing improved while CD14 pathway-neighbor
agreement declined in all 32 paired comparisons. These are run02 findings, not
predictions for run03. This run changes the training schedule only; the cohort,
seed, HVGs, model architecture and biological evaluation remain fixed.

Review `training_summary.json` and training histories: verify full KL weight,
inspect post-warm-up validation ELBO, and flag runs reaching the epoch limit.
Compare run03 with run02 using the same metrics; better mixing alone is insufficient.

## Main findings
No biological conclusion for run03 is pre-written. Fill this section after reviewing the
executed metrics, training histories, population failures and donor/CD14 checks.
Distinguish cell visualization, population agreement and donor reproducibility.

## Decision for next phase
The table informs a review; it does not automatically authorize stronger correction.
If evidence is mixed, identify populations to restrict or investigate. If supported,
the next possible work is CITE RNA+protein with totalVI and Multiome RNA+ATAC with
MultiVI, followed by carefully evaluated mapping through RNA. Gene-activity scores
alone are not the raw peak-count input needed for a future MultiVI workflow.
Neither model is implemented here, and neighboring cells remain different cells.

## Saved outputs and limitations
`shared_rna.h5ad` stores all shared normalized RNA, a single counts layer, metadata,
HVG flags and embeddings. Separate CSV/NPY files provide gene lists, metrics, scores
and coordinates. `scvi_model/` saves model weights/registry without another AnnData
copy; reconstruct its input by subsetting shared_rna.h5ad to highly_variable genes
in stored order. Training CSVs and manifest document settings and versions.

No donor/site regression, causal inference or claim of biological equivalence.
The eight-donor cohort was selected for earlier CD14 validation, so generalization
to rare types or excluded donors requires separate assessment. Stronger mixing is
not the goal if cell identity or meaningful biological variation is lost.
''')]

if __name__=='__main__':
    p=Path(__file__).resolve().parents[1]/'notebooks/nb10_shared_rna_integration.ipynb'
    if p.exists(): raise FileExistsError('Refusing to overwrite nb10')
    for i,c in enumerate(cells): c['id']=f'nb10-{i:03d}'
    p.write_text(json.dumps(dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},nbformat=4,nbformat_minor=5),indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
