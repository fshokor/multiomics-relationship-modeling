# nb11: paired CITE RNA and protein integration

Copy `notebooks/nb11_cite_totalvi.ipynb` and `src/totalvi_workflow.py` to their matching
locations in the existing Drive repository. Existing repository helper modules are
dependencies. No nb01–10 files are changed.

## First run in Colab

Choose a GPU runtime, set `MODE = 'TRAIN'` in the setup cell and keep the fresh
`results/totalvi/nb11_run01` output folder. Run all. Inputs are the CITE benchmark
file and nb09's completed cohort, mapping, protocol and manifest files. The source
hash must agree with nb09. Multiome data and pathway collections are not required.

Defaults: eight donors, up to 3,000 RNA HVGs selected using equal donor quotas,
all non-control, nonzero ADTs, 30 latent dimensions, raw RNA NB likelihood and
totalVI protein likelihood. No donor/site or cell-type covariates are supplied.
This learns a joint RNA/protein representation within CITE; it does not promise
donor batch removal or align CITE to Multiome. Donor/site effects are diagnostic.

Training uses 50 warm-up epochs, at least 100 and at most 300 epochs, early stopping
on validation ELBO after warm-up with patience 30, batch size 128 and full float32
precision. Final weights are saved; no best-checkpoint restoration is claimed.
Hyperparameters are starting choices, not guarantees of convergence.

## Reopen without training

Set `MODE = 'LOAD'` (the notebook default), select the completed output folder and
Run all. This mounts Drive and reads CSV/JSON/PNG outputs without loading raw data,
installing model packages or requiring a GPU. Every displayed table is loaded
explicitly so a fresh runtime does not depend on yesterday's variables.

If evaluation was interrupted, choose `MODE = 'EVALUATE'`. It uses the saved
`model_input.h5ad` and totalVI coordinates, or recovers coordinates from `model/`
if needed. It reruns diagnostics and figures only, leaving model weights intact.
The separate `evaluation_status.json` records evaluation completion and hashes;
`status.json` may retain an earlier training-stage status after recovery.

## Review

Inspect training curves and summary, RNA-PCA versus totalVI cell-type purity,
silhouette and neighbor agreement, per-type sample counts, donor/site structure,
depth associations, and observed protein means and neighborhood consistency.
Comparisons use the same resolved donor-quota sample, up to 20,000 cells. UMAP is
for visualization. Protein consistency is internal: those proteins also trained
the model. No held-out protein accuracy, donor replication p-values, pathway
preservation, automatic multimodal success, or cell-to-cell matching is claimed.

Synthetic tests cover feature collisions, exclusion of controls and zero-RNA
cells, actual CPU totalVI training, saving, model recovery without retraining,
evaluation figures, notebook validity and overwrite protection. The real-data
training and biological review must still be performed in Colab.

[Official totalVI API](https://docs.scvi-tools.org/en/1.4.3/api/reference/scvi.model.TOTALVI.html)
