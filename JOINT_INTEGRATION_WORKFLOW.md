# nb13: joint RNA, ATAC and protein integration

Copy notebooks/nb13_joint_rna_atac_protein.ipynb and src/joint_integration.py to the
existing Drive repository. Keep the existing src helper modules. Previous notebooks
and result folders are not changed.

## Inputs and execution

From both results/totalvi/nb11_run01 and results/multivi/nb12_run01 retain
model_input.h5ad, manifest.json and status.json on Drive. Matching cohort fingerprints
are required. This workflow uses their raw counts, not their latent coordinates.
Set MODE='TRAIN' in a fresh GPU Colab runtime, preferably with ample host RAM, and
Run all. Results go to the new results/joint_integration/nb13_run01 folder. Occupied
output folders are refused. Full training is new work; nb11/12 weights are not reused.

The pilot RNA panel is the intersection of nb11 and nb12 HVGs. It is not the full
shared gene universe or the nb10 panel. ATAC and protein panels are frozen upstream.
The actual intersection and exclusions are recorded. Peak prevalence selection and
HVG intersection may limit rare-state resolution. No new pathway selection occurs.

## Model and missingness

A three-modality MultiVI model (scvi-tools 1.4.3) receives shared RNA for every cell,
ATAC only for Multiome and proteins only for CITE. MuData storage placeholders for
missing whole modalities are masked by MultiVI's row-sum missingness logic, verified
in the synthetic test at the reconstruction-loss level. Retained cells must have
positive selected RNA and positive measured modality counts. Measured per-feature
zeros remain observations. Modality flags are saved alongside distinct cell IDs.

Assay is the batch variable and adversarial mixing is enabled; donor/site and
cell-type labels are not nuisance variables or supervision. Default modality
alignment penalty, RNA NB likelihood, hidden128, latent30. Warm-up50, min100/max300
epochs, validation-ELBO stopping patience30 after warm-up, float32, batch128. The
final weights are saved, not restored best weights. Better mixing may sacrifice
biology: inspect both sides of that tradeoff.

## Reopen and review

MODE='LOAD' reads only saved tables/figures. MODE='EVALUATE' reconstructs diagnostics
from saved data/weights without training. Model weights and input are saved before
evaluation. Input fingerprints, CSV checksums and explicit completion states are
recorded. Nine figures show training and RNA-baseline/joint UMAPs by assay, cell type,
donor and site. Tables cover cell identity, within-type and within-donor/type assay
alignment, observed RNA profile correlations and depth associations within capture.

The RNA PCA baseline uses the same RNA panel and evaluation cells. It is not a
matched-model ablation, so differences cannot be uniquely credited to added ATAC
or protein. Results are descriptive, not independent donor validation. No missing
modality predictions are exported. A joint embedding does not establish matching
cells, direct protein-ATAC measurements, validated imputation or causal regulation.

Local validation uses synthetic mixed-capture data for actual training, explicit
missing-likelihood checks, saved-model recovery, output creation and notebook
syntax. Full cohort training and biological validation remain Colab tasks.

[MultiVI API](https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.MULTIVI.html)
