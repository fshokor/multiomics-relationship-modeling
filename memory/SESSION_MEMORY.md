# SESSION MEMORY
## multiomics-relationship-modeling

**Last updated:** [date]

## Project state
Repo just created. No notebooks run yet.
h5ad files available locally — path to be symlinked into data/benchmark/.

## Dataset
NeurIPS 2021 Multimodal Single-Cell Integration (GSE194122)
- CITE-seq: RNA + protein (ADT), ~70k bone marrow cells, 10 donors
- Multiome: RNA + ATAC, ~69k cells, same donors

## Scientific question
What is the structure of the relationship between omics layers?
- Where does RNA→protein coupling hold vs break down?
- Is breakdown cell-type-specific or universal?
- Same question for RNA→ATAC (chromatin regulation axis)

## Notebooks completed
- None yet

## Key decisions made
- No prediction task — this is relationship characterization
- CITE-seq track (nb02) and Multiome track (nb03) are parallel
- VAE (nb04) deferred until nb02/nb03 are complete
- OmicsAge scripts available to copy for QC and loading logic
