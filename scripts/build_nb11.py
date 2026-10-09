"""Generate nb11 only; preserve previous notebooks."""
import json
from pathlib import Path
from build_single_donor_notebooks import code, markdown

cells = [markdown("""
# nb11 — CITE RNA + protein integration with totalVI

Jointly model measured RNA and ADT counts in the eight nb09 donors, across all
cell populations. No pathway selection, supervised labels or cell matching.
This is within-CITE multimodal integration, not a shared CITE/Multiome space.

**First training:** choose TRAIN and a fresh output folder; GPU recommended.
**Return tomorrow:** leave LOAD selected and Run all. No raw inputs, GPU or
training packages are required to view saved tables and figures.
**Interrupted evaluation:** choose EVALUATE to rebuild analysis from model_input.h5ad
and the saved model, without training. Never switch to TRAIN to recover analysis.

Copy this notebook and src/totalvi_workflow.py to Drive. Existing src helpers from
nb04–10 remain dependencies. Previous notebooks are unchanged.
"""), code("""
from pathlib import Path
import json, sys, subprocess
MODE = 'LOAD'  # LOAD, TRAIN, or EVALUATE; training is always explicit
assert MODE in {'LOAD', 'TRAIN', 'EVALUATE'}
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if IN_COLAB:
    drive.mount('/content/drive')
    BASE = Path('/content/drive/MyDrive/multiomics-relationship-modeling')
else:
    BASE = Path.cwd().resolve()
    if not (BASE/'src').is_dir(): BASE = BASE.parent
if MODE != 'LOAD':
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                           'scanpy>=1.11,<1.12', 'scvi-tools>=1.3,<1.5', 'anndata>=0.11,<0.13'])
sys.path.insert(0, str(BASE))
import pandas as pd
from IPython.display import display, Image
OUTPUT = BASE/'results/totalvi/nb11_run01'
COHORT = BASE/'results/cross_donor/nb09_run01'
SEED = 42
N_HVGS = 3000
TRAINING = dict(max_epochs=300, min_epochs=100, warmup_epochs=50,
                patience=30, accelerator='auto', batch_size=128, n_latent=30)
print('Mode:', MODE, 'Output:', OUTPUT)
"""), markdown("""
## Inputs and model choices
Verify nb09 cohort/source fingerprints. Read paired RNA and ADT by cell ID and
feature position, preventing RNA/protein name collisions. Exclude control-labelled
or all-zero proteins and cells with zero RNA or biological ADT totals; audit all
exclusions. This uses existing QC and does not invent new donor-specific filters.

Select up to 3,000 RNA HVGs from normalized expression with equal donor quotas.
Train on **raw RNA and raw ADT**, not normalized or CLR values. All remaining
proteins enter training. Labels are used only for descriptive evaluation.

No donor/site nuisance covariates: their biological and technical effects are
confounded. This first model learns a joint representation without explicitly
removing those effects. Donor/site mixing alone is not a success criterion.

50-epoch KL warm-up, minimum 100 / maximum 300 epochs, validation ELBO early
stopping after warm-up with patience 30; no learning-rate scheduler. Save final
weights and histories. A final-epoch minimum does not alone prove nonconvergence;
inspect the size of recent changes.

Reference: [totalVI API](https://docs.scvi-tools.org/en/1.4.3/api/reference/scvi.model.TOTALVI.html).
"""), code("""
if MODE == 'TRAIN':
    from configs.paths import CITE_H5AD
    from src.totalvi_workflow import run_totalvi
    # One CITE file only; no Multiome input or copying of raw files for LOAD.
    run_totalvi(CITE_H5AD, COHORT, OUTPUT, seed=SEED, n_hvgs=N_HVGS, **TRAINING)
elif MODE == 'EVALUATE':
    from src.totalvi_workflow import evaluate_saved
    evaluate_saved(OUTPUT, seed=SEED)
else:
    print('Training and evaluation skipped; loading saved outputs below.')
"""), code("""
required = ['metric_comparison.csv', 'celltype_metrics.csv', 'depth_associations.csv',
            'protein_neighbor_agreement.csv', 'observed_protein_means_by_celltype.csv']
missing = [name for name in required if not (OUTPUT/name).exists()]
if missing:
    raise FileNotFoundError(f'Missing saved outputs: {missing}. For a first run choose TRAIN. '
                            'If model/model.pt and model_input.h5ad already exist, choose EVALUATE instead.')
comparison, types, depths, proteins, marker_means = [pd.read_csv(OUTPUT/name) for name in required]
for name in ['status.json', 'evaluation_status.json', 'input_audit.json', 'training_summary.json']:
    if (OUTPUT/name).exists():
        print(name); display(json.loads((OUTPUT/name).read_text()))
"""), markdown("""
## Training and broad cell identity
Compare RNA-only PCA with totalVI on exactly the same fixed donor-quota sample
(up to 20,000 resolved cells). No matching to nb10's cross-assay sample is claimed.
Silhouette uses at most 3,000 cells. Rare types can have unstable estimates: inspect
n_evaluation_cells. Unresolved cells train the model but are excluded from these
label-based evaluations. Purity/label agreement are related internal diagnostics.
"""), code("""
display(Image(filename=str(OUTPUT/'figures/training.png')))
display(comparison)
with pd.option_context('display.max_rows', None): display(types)
for field in ['cell_type_harmonized', 'DonorID', 'Site']:
    for space in ['rna_pca', 'totalVI']:
        display(Image(filename=str(OUTPUT/f'figures/{space}_{field}.png')))
"""), markdown("""
## Protein consistency and sequencing depth
Protein neighbor agreement compares observed per-cell CLR protein abundance with
mean abundance among embedding neighbors (self excluded). CLR is diagnostic only.
An increase is expected when proteins enter the model: it is not held-out validation.
The observed protein means table allows inspection of broad lineage markers (for
example CD3/CD4/CD8, CD19, CD14, CD56 where measured). Neither means nor neighbor
correlations prove accurate protein denoising or cell-type annotation.

Inspect depth associations and population-specific losses, not only a visually
mixed UMAP. Maximum coordinate/depth correlations are descriptive and coordinate
system dependent; they do not identify a technical cause. Donor/site correlations
may include biology and should not automatically trigger stronger correction.
"""), code("""
display(depths)
with pd.option_context('display.max_rows', None): display(proteins)
display(marker_means)
"""), markdown("""
## Findings and decision after execution
Record training convergence, broad cell identity, rare-type limitations, observed
protein marker patterns, and donor/site/depth associations. No result is assumed
before execution. These are descriptive internal checks, not donor-independent
validation or a causal analysis. Do not treat cells as independent human replicates.

The next separate workflow can fit MultiVI to paired Multiome RNA and ATAC peak
counts. Separate totalVI/MultiVI latent coordinates cannot simply be concatenated
or compared dimension-by-dimension; cross-capture alignment requires evaluation.

## Saved outputs
model_input.h5ad stores normalized HVG X for PCA, raw HVG counts, raw ADT, metadata
and totalVI coordinates. model/ stores weights and registry. The input is saved
before training and the model before evaluation, allowing recovery without retraining.
CSV tables, UMAP coordinates, PNG figures, training histories, input audits and
manifest remain in OUTPUT. LOAD mode reads only saved outputs. To load a trained
model manually use TOTALVI.load with model_input.h5ad in its saved gene/protein order.
""")]

if __name__ == '__main__':
    path = Path(__file__).resolve().parents[1]/'notebooks/nb11_cite_totalvi.ipynb'
    if path.exists(): raise FileExistsError(path)
    for i, cell in enumerate(cells): cell['id'] = f'nb11-{i:03d}'
    path.write_text(json.dumps(dict(cells=cells, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}, nbformat=4, nbformat_minor=5), indent=1)+'\n', encoding='utf-8')
