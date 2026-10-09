"""Generate the joint RNA/ATAC/protein notebook without touching prior notebooks."""
import json
from pathlib import Path
from build_single_donor_notebooks import code,markdown
cells=[markdown("""
# nb13 — Joint RNA, ATAC and protein integration

Fit a **new three-modality MultiVI model** across CITE (RNA+protein) and Multiome
(RNA+ATAC), connected by common RNA. Previous totalVI/MultiVI latent axes are not
merged. Cells remain distinct, including repeated barcodes across captures.

This pilot reuses raw counts from nb11 and nb12 model_input.h5ad files. RNA features
are the **intersection of their selected HVG panels**, not the complete shared RNA
universe. Peaks and proteins remain frozen to nb12/nb11 respectively. The input audit
reports the actual intersection size; feature restrictions limit interpretation.

TRAIN once in a fresh GPU runtime and folder. Return with LOAD (default). Use
EVALUATE to rebuild analysis from saved weights without retraining. Copy this
notebook and src/joint_integration.py into the existing Drive project; the existing
src helper files remain dependencies. nb01–12 stay unchanged.
"""),code("""
from pathlib import Path
import sys,json,subprocess
MODE='LOAD'  # TRAIN for first run; EVALUATE only to rebuild saved analysis
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
    if not (BASE/'src').is_dir():BASE=BASE.parent
if MODE!='LOAD':
    subprocess.check_call([sys.executable,'-m','pip','install','-q',
        'scvi-tools==1.4.3','scanpy>=1.11,<1.12','anndata>=0.11,<0.13'])
sys.path.insert(0,str(BASE))
import pandas as pd
from IPython.display import display,Image
CITE_RUN=BASE/'results/totalvi/nb11_run01'
MULTIOME_RUN=BASE/'results/multivi/nb12_run01'
OUTPUT=BASE/'results/joint_integration/nb13_run01'
SEED=42
TRAINING=dict(max_epochs=300,min_epochs=100,warmup_epochs=50,patience=30,
              accelerator='auto',batch_size=128,n_latent=30)
print('Mode:',MODE,'Output:',OUTPUT)
"""),markdown("""
## Input and missing-modality contract
Each upstream folder must contain model_input.h5ad, manifest.json and status.json.
The cohort fingerprints must agree. Source input hashes, ordered feature names,
cell IDs and counts are recorded. Raw RNA/protein/peak counts are checked.
Cells with zero selected RNA or zero measured modality counts are excluded and audited.

CITE ATAC rows and Multiome protein rows contain **storage placeholders**. Explicit
observed-modality flags are saved; MultiVI 1.4.3 detects whole missing modalities
from zero row totals and masks their likelihood contributions. Individual zeros
within measured modalities remain observations. We test this behavior in synthetic
mixed-capture minibatches; no absent modality is presented as a measured zero.

Assay is the batch variable, with adversarial mixing enabled. Donor/site and cell-type
labels are not nuisance covariates or training supervision. The default modality
alignment penalty remains active. RNA likelihood NB, hidden width 128, latent size 30.
Train on raw counts with 50-epoch KL warm-up, 100–300 epochs, and validation ELBO
patience 30 after warm-up. Final weights are saved. This correction choice can remove
real capture-associated biology, so evaluate preservation as well as mixing.

No missing-modality predictions are exported in this first notebook. A shared
embedding does not validate protein–ATAC relationships or provide paired cells.

[MultiVI three-modality API](https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.MULTIVI.html)
"""),code("""
if MODE=='TRAIN':
    from src.joint_integration import run_joint
    run_joint(CITE_RUN,MULTIOME_RUN,OUTPUT,seed=SEED,**TRAINING)
elif MODE=='EVALUATE':
    from src.joint_integration import evaluate_saved
    evaluate_saved(OUTPUT,seed=SEED)
else:
    print('Training skipped; loading saved results below.')
"""),code("""
files=['metric_comparison.csv','celltype_alignment.csv','donor_celltype_alignment.csv','depth_associations.csv']
missing=[n for n in files if not (OUTPUT/n).exists()]
if missing:
    raise FileNotFoundError(f'Missing {missing}. First run: TRAIN. If weights and model_input.h5ad '
                            'already exist here, choose EVALUATE rather than retraining.')
comparison,types,donors,depths=[pd.read_csv(OUTPUT/n) for n in files]
for name in ['status.json','input_audit.json','training_summary.json','evaluation_status.json']:
    if (OUTPUT/name).exists():
        print(name);display(json.loads((OUTPUT/name).read_text()))
"""),markdown("""
## Does a common space preserve cell identity?
Compare the joint model with RNA PCA on the exact same shared RNA panel and fixed
assay/donor-quota sample (up to 20,000 resolved cells). RNA PCA is a baseline, not a
matched-complexity ablation: changes cannot be uniquely attributed to ATAC/protein.
Silhouette uses at most 3,000 cells. Unresolved labels train but do not evaluate.

Within-type and donor/type mixing use equal assay query weights and normalize
by random-mixing expectation. Require 50 evaluation cells per assay, retain
insufficient groups. Exploratory labels: supported when mixing ratio >=0.8 and
scaled centroid distance <=0.5; concern when ratio <0.5 or distance >1; otherwise
partial. These labels are diagnostics, not an automatic go/no-go rule.

Inspect cell-type purity and silhouette loss, minority populations, donor/site
structure and depth associations **within each assay**, excluding unmeasured-depth
placeholders. RNA-profile correlations use normalized observed RNA and cannot
improve because of embedding integration. No cell-level inferential p-values.
"""),code("""
display(Image(filename=str(OUTPUT/'figures/training.png')))
display(comparison)
with pd.option_context('display.max_rows',None):
    display(types)
    display(donors)
display(depths)
for field in ['assay','cell_type_harmonized','DonorID','Site']:
    for space in ['rna_pca','joint']:
        display(Image(filename=str(OUTPUT/f'figures/{space}_{field}.png')))
"""),markdown("""
## Findings and decision after execution
Record training convergence, assay mixing conditional on cell identity, unsupported
populations, donor agreement and depth structure. No success is assumed. A good
UMAP cannot validate unobserved protein/ATAC or demonstrate causal regulation.

This is a feature-restricted integration pilot. The nb12 prevalent-peak selection
may underrepresent rare states, and the RNA HVG intersection may omit useful genes.
If evidence is mixed, review those limitations before increasing correction.
Cross-modal predictions would require held-out observed-modality validation and
explicit labeling as predictions; they are not produced here.

## Saved artifacts
model_input.h5ad contains raw shared RNA and peaks, sparse protein counts,
observed-modality flags and joint coordinates. model/ contains weights/registry.
Input is saved before training, weights before evaluation; EVALUATE can recover
joint coordinates if needed. Hashes, manifests, CSV metrics, per-cell diagnostics,
UMAP coordinates and nine PNGs support a review. LOAD requires only the saved
analysis files. Keep the large inputs and weights on Drive for further diagnostics.
""")]
if __name__=='__main__':
    p=Path(__file__).resolve().parents[1]/'notebooks/nb13_joint_rna_atac_protein.ipynb'
    if p.exists():raise FileExistsError(p)
    for i,c in enumerate(cells):c['id']=f'nb13-{i:03d}'
    p.write_text(json.dumps(dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},nbformat=4,nbformat_minor=5),indent=1)+'\n')
