# nb07: paired CITE RNA–protein support

Copy these three files into the same folders in your Drive project:

- `notebooks/nb07_pathway_protein_support.ipynb`
- `src/protein_programs.py`
- `src/protein_plots.py`

Open nb07 in Colab, restart a reused runtime, and run all cells. Keep the updated
shared modules and completed nb04–06 outputs. No rerun of earlier stages is needed.
The notebook copies the CITE h5ad to Colab local disk before reading ADT counts;
results are saved to Drive under `results/single_donor/protein/`. Progress messages
identify artifact checks, ADT reading, scoring and figure generation.

The selected panel is read from nb06, with donor and upstream consistency checks.
Only genuine CITE RNA–ADT pairs are used. Mapping overrides prevent ambiguous
CD/complex/isoform labels from silently becoming unique gene–protein pairs.

Review `mapping.csv` and `coverage.csv` first. `evidence.csv` separates direct
pathway-gene measurements from exploratory CD86/HLA-DR surface context.
`associations.csv` reports module–ADT correlations within the target and each site;
`direct_gene_associations.csv` reports same-gene RNA–ADT effects. Depth/site
adjustment and uncentered log1p sensitivity accompany the centered-log ADT analysis.
All tables, cell QC, RNA scores, figures in PNG/PDF, and provenance are saved.

Zero coverage means unmeasured. A few surface proteins cannot validate an entire
pathway; E2F/Myc have no forced proxies. Phenotypic markers have no prespecified
direction. No background correction, causal conclusion or donor replication is
claimed. Review the real outputs before proceeding to nb08.

Software checks: `python -m unittest discover -s tests -v`. Synthetic tests cover
cell-ID alignment across dense/CSR/CSC storage, ambiguous mapping, missing evidence,
depth confounding, notebook structure and the complete nb04–07 saved-output path.
