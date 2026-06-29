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
