# NEXT SESSION

## Goal
nb01_qc — load both h5ad files, run QC, confirm structure.

## Steps
1. Symlink or copy h5ad files into data/benchmark/
2. Copy relevant loading/QC scripts from OmicsAge
3. Load CITE-seq: inspect .obs (donor_id, cell_type, batch, site), .var, .X shape
4. Load Multiome: same inspection
5. QC per dataset: cell counts per donor, % mito, count distributions
6. Confirm cell barcode alignment within each paired dataset
7. Check whether cell type annotations exist in obs or need clustering

## Output
- QC plots saved to results/figures/nb01_*
- Confirmed obs column names for use in nb02/nb03
