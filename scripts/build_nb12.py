"""Generate nb12 only; independent of the running nb11."""
import json
from pathlib import Path
from build_single_donor_notebooks import code, markdown
cells = [markdown("""
# nb12 — Paired Multiome RNA + ATAC integration with MultiVI

Jointly model RNA and genomic peak counts from the eight eligible nb09 donors.
No pathway selection, labels as supervision, donor/site correction, or CITE input.
This learns a within-Multiome representation. It does not align to totalVI coordinates.

Use TRAIN once in a fresh GPU runtime and output folder. Return with LOAD (default)
to view saved results without training or raw data. Use EVALUATE if analysis was
interrupted after model saving. Run independently of the active nb11 runtime.
Copy this notebook and src/multivi_workflow.py to the existing Drive repository;
existing src helpers remain dependencies. nb01–11 are unchanged.
"""), code("""
from pathlib import Path
import json, sys, subprocess
MODE = 'LOAD'  # TRAIN for first run; EVALUATE to rebuild analysis without training
assert MODE in {'LOAD','TRAIN','EVALUATE'}
try:
    from google.colab import drive
    IN_COLAB=True
except ImportError:
    IN_COLAB=False
if IN_COLAB:
    drive.mount('/content/drive')
    BASE=Path('/content/drive/MyDrive/multiomics-relationship-modeling')
else:
    BASE=Path.cwd().resolve()
    if not (BASE/'src').is_dir(): BASE=BASE.parent
if MODE != 'LOAD':
    subprocess.check_call([sys.executable,'-m','pip','install','-q',
        'scanpy>=1.11,<1.12','scvi-tools>=1.3,<1.5','anndata>=0.11,<0.13'])
sys.path.insert(0,str(BASE))
import pandas as pd
from IPython.display import display,Image
OUTPUT=BASE/'results/multivi/nb12_run01'
COHORT=BASE/'results/cross_donor/nb09_run01'
SEED=42
N_HVGS=3000
MAX_PEAKS=20000
MIN_PEAK_CELLS=50
TRAINING=dict(max_epochs=300,min_epochs=100,warmup_epochs=50,patience=30,
              accelerator='auto',batch_size=128,n_latent=30)
print('Mode:',MODE,'Output:',OUTPUT)
"""), markdown("""
## Input checks and feature selection
Authenticate the Multiome input and nb09 cohort/mapping by hashes. Read raw
layers/counts in bounded row chunks. RNA and ATAC stay paired by source cell ID.
Cells need positive RNA and peak totals; exclusions are audited. Unique genomic
peaks are required. Existing gene-activity matrices are not used.

RNA: up to 3,000 HVGs from log1p CP10k expression with equal donor quotas.
ATAC: peaks detected in at least 50 eligible cells; retain up to 20,000 with highest
pooled detection prevalence (stable ties). This computational starting choice may
underrepresent rare cell states and favor abundant populations. Inspect peak_audit.csv.
Cells with no counts in selected peaks are excluded and logged. Counts remain sparse.

The combined model input stores raw RNA first and raw peaks second, with explicit
n_genes/n_regions. At training/reload it is reconstructed as paired RNA/ATAC MuData
using setup_mudata, with cell order checked. MultiVI models peak detection/accessibility, not a quantitative
fragment-count likelihood. RNA uses NB. No labels, donor/site nuisance covariates
or adversarial batch mixing. Hidden width 128, latent size 30. The cohort is fully
paired; no missing modality is invented. Separate modality encoders use MultiVI's
default alignment penalty.

50-epoch KL warm-up, minimum 100 / maximum 300 epochs. A callback monitors validation
ELBO after warm-up with patience 30. Final weights are saved, not restored best weights.
No convergence or biological success is assumed from reaching the epoch limit.

[MultiVI documentation](https://docs.scvi-tools.org/en/1.4.3/tutorials/notebooks/multimodal/MultiVI_tutorial.html)
"""), code("""
if MODE == 'TRAIN':
    from configs.paths import MULTIOME_H5AD
    from src.multivi_workflow import run_multivi
    run_multivi(MULTIOME_H5AD,COHORT,OUTPUT,seed=SEED,n_hvgs=N_HVGS,
                max_peaks=MAX_PEAKS,min_peak_cells=MIN_PEAK_CELLS,**TRAINING)
elif MODE == 'EVALUATE':
    from src.multivi_workflow import evaluate_saved
    evaluate_saved(OUTPUT,seed=SEED)
else:
    print('Training skipped; reading saved results next.')
"""), code("""
files=['metric_comparison.csv','celltype_metrics.csv','depth_associations.csv']
missing=[name for name in files if not (OUTPUT/name).exists()]
if missing:
    raise FileNotFoundError(f'Missing {missing}. First run: choose TRAIN. If model/model.pt '
                            'and model_input.h5ad exist, choose EVALUATE, not TRAIN.')
comparison,types,depths=[pd.read_csv(OUTPUT/name) for name in files]
for name in ['status.json','evaluation_status.json','input_audit.json','training_summary.json']:
    if (OUTPUT/name).exists():
        print(name);display(json.loads((OUTPUT/name).read_text()))
"""), markdown("""
## Training and three-way biological comparison
Compare RNA PCA, ATAC LSI and MultiVI on the same fixed donor-quota sample (up to
20,000 resolved cells). Labels evaluate but never train the model. Rare-type sample
sizes are reported; unresolved labels remain in training and are excluded here.
Silhouette uses at most 3,000 cells. No cell-level inferential p-values.

RNA PCA uses log1p CP10k normalized across all GEX before HVG selection. ATAC LSI
uses binary selected peaks, row term frequency and log1p(N/detection count) inverse
document frequency, then truncated SVD. LSI1 is dropped a priori; remaining depth
associations are still inspected. Neither baseline is fed to MultiVI training.
"""), code("""
display(Image(filename=str(OUTPUT/'figures/training.png')))
display(comparison)
with pd.option_context('display.max_rows',None):display(types)
for field in ['cell_type_harmonized','DonorID','Site']:
    for space in ['rna_pca','atac_lsi','MultiVI']:
        display(Image(filename=str(OUTPUT/f'figures/{space}_{field}.png')))
display(depths)
"""), markdown("""
## Interpretation after execution
Record convergence, broad cell identity and rare-population behavior relative to
both baselines. Check donor/site structure and depth associations; mixing is not
an optimization target. Coordinate/depth correlations are descriptive and depend
on the coordinate system. UMAP agreement is not independent biological validation.
No pathway-specific conclusion, held-out accessibility accuracy or causal claim
is inferred. Generalization beyond these donors remains untested.

## Saved results and next step
model_input.h5ad stores one sparse raw RNA+peak matrix, ordered features, metadata,
RNA PCA, ATAC LSI and joint coordinates. model/ stores weights/registry; histories,
audits, CSVs and ten PNG figures are saved separately. The input is written before
training and weights before analysis. EVALUATE can recover latent coordinates
from weights if training finished before coordinates were saved. LOAD reads only
saved outputs; no dependence on previous session variables.

Review totalVI and MultiVI separately before considering cross-capture integration.
Their latent axes are unrelated and cannot be directly concatenated. Neither
notebook identifies matching CITE/Multiome cells.
""")]
if __name__=='__main__':
    path=Path(__file__).resolve().parents[1]/'notebooks/nb12_multiome_multivi.ipynb'
    if path.exists():raise FileExistsError(path)
    for i,c in enumerate(cells):c['id']=f'nb12-{i:03d}'
    path.write_text(json.dumps(dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},nbformat=4,nbformat_minor=5),indent=1)+'\n',encoding='utf-8')
