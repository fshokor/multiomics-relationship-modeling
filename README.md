# Multiomics Relationship Modeling

**Question:** What is the structure of the relationship between omics layers
at single-cell resolution — and where does it break down?

## Dataset
NeurIPS 2021 Multimodal Single-Cell Integration benchmark (GSE194122)
- **CITE-seq:** RNA + surface protein (ADT), ~70k human bone marrow cells, 10 donors
- **Multiome:** RNA + ATAC (chromatin accessibility), ~69k cells, same donors

## Scientific tracks
| Track | Modalities | Notebook |
|---|---|---|
| RNA ↔ Protein | CITE-seq GEX + ADT | nb02_rna_protein |
| RNA ↔ ATAC | Multiome GEX + ATAC | nb03_rna_atac |

## Key questions
1. Is RNA→protein coupling cell-type-specific or universal?
2. Which genes are post-transcriptionally regulated (low RNA-protein r)?
3. Which genes are regulated at the chromatin level (high ATAC-RNA r)?
4. Do the same genes appear in both axes?

## Setup
```bash
conda env create -f environment.yml
conda activate multiomics-sc

# Symlink your h5ad files (see data/README.md)
# Then run notebooks in order: nb01 → nb02 → nb03
```

## Context
Genopole Shaker application deadline: July 15, 2026.
GDSC/ProCan drug response project parked at nb17 — see separate repo.

## Single-donor program characterization

New notebooks `nb04_single_donor_rna_characterization.ipynb` and
`nb05_rna_concordance.ipynb` implement the first incremental RNA milestone using
the existing Colab/Drive data paths. See [the workflow guide](SINGLE_DONOR_WORKFLOW.md)
for setup, scientific choices, saved outputs, tests and the real-data validation
gate before ATAC/protein follow-up. The original notebooks are preserved.

After reviewing nb05, run `nb06_pathway_atac_support.ipynb` for paired Multiome
RNA–ATAC program support. Copy `src/atac_programs.py` and `src/atac_plots.py` to the
same Drive project first. See [the ATAC run guide](ATAC_WORKFLOW.md).

After reviewing nb06, run `nb07_pathway_protein_support.ipynb` for paired CITE
RNA–ADT support. See [the protein run guide](PROTEIN_WORKFLOW.md) for the three
files to copy, evidence tiers, progress reporting and saved outputs.

After nb07 review, `nb08_multilayer_pathway_states.ipynb` combines saved RNA, ATAC
and protein evidence without loading raw matrices. See [the multilayer guide](MULTILAYER_WORKFLOW.md).

`nb09_cross_donor_validation.ipynb` extends the fixed CD14 programs to eligible
donors and exports site-resolved raw RNA pseudobulk. See [the cross-donor guide](CROSS_DONOR_WORKFLOW.md).

`nb10_shared_rna_integration.ipynb` evaluates an unsupervised assay-only scVI RNA
bridge against an uncorrected baseline, with cell-type, donor and CD14-gradient
safeguards. See [the shared RNA integration guide](SHARED_RNA_INTEGRATION.md).
