# nb12: paired Multiome RNA and ATAC integration

Copy notebooks/nb12_multiome_multivi.ipynb and src/multivi_workflow.py to their
matching locations in the existing Drive repository. Existing helper modules
remain dependencies. This does not change nb11 or use totalVI outputs.

## Run

Use a separate Colab GPU runtime from the running totalVI notebook, preferably
with ample host RAM. Set MODE='TRAIN' and use the fresh results/multivi/nb12_run01
folder. Inputs are the original Multiome h5ad and completed nb09 cohort metadata.
Read raw RNA and genomic peak counts, not gene activity or processed X.

Default features: up to 3,000 RNA HVGs using equal donor quotas and up to 20,000
peaks detected in at least 50 eligible cells, ranked by pooled detection frequency.
The peak cap limits model size; it can underrepresent rare-state accessibility.
All selections and zero-modality exclusions are audited. Both modalities remain
paired by cell ID; every retained cell has positive counts in each selected modality.

A single sparse combined input is saved; paired MuData is reconstructed in memory
for MultiVI using setup_mudata. Gene/peak order and cell identity are preserved.
No labels, donor/site nuisance covariates or adversarial batch mixing are used.
The default modality alignment penalty remains active. RNA likelihood is NB and
ATAC models detection. Hidden width 128, latent size 30, batch size 128, full float32.

Training uses 50 warm-up epochs, 100–300 epochs and a validation-ELBO early-stopping
callback after warm-up with patience 30. Explicit callback avoids version-specific
MultiVI defaults that otherwise monitor reconstruction loss. Final weights are saved.

## Reopen or recover

MODE='LOAD' (default) displays saved tables, training summary and figures without
model packages, raw files or a GPU. MODE='EVALUATE' rebuilds analysis from saved
model_input.h5ad and, if necessary, model weights; it never calls train. An occupied
training output directory is refused. A separate evaluation_status.json records
completion and CSV hashes, including after recovering an interrupted evaluation.

## Review

Compare RNA-only PCA, ATAC-only TF-IDF/LSI and joint MultiVI on the same fixed
resolved-cell donor-quota sample. UMAP panels show cell type, donor and site. Tables
show cell identity, per-type counts, donor/site structure and depth associations.
LSI1 is excluded a priori; remaining depth correlations are still reported.
These are descriptive internal checks, not held-out modality prediction accuracy.
Donor/site mixing is not a success target, and cells are not independent donors.

Weights, input, ordered features, baselines, joint coordinates, training histories,
audits, tables and ten PNGs are saved. Full data/model fitting happens in Colab;
local tests use a small synthetic cohort to exercise training and reload recovery.
Separate totalVI and MultiVI latent axes cannot be directly combined or interpreted
as matched CITE/Multiome cells. Cross-capture alignment is a later evaluated step.

[MultiVI tutorial](https://docs.scvi-tools.org/en/1.4.3/tutorials/notebooks/multimodal/MultiVI_tutorial.html)
