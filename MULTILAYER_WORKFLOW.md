# nb08: multilayer pathway evidence

Copy `notebooks/nb08_multilayer_pathway_states.ipynb`, `src/multimodal_summary.py`
and `src/multimodal_plots.py` to the matching folders in your Drive project.
Open nb08 in Colab, restart a reused runtime, and run all cells. This stage reads
saved summary tables from nb04–07; it does not load raw h5ad files or rerun scoring.

Outputs under `results/single_donor/multilayer/`:

- `program_evidence.csv`: RNA NES/q-values, Multiome RNA/ATAC population means,
  ATAC coverage and paired associations, protein coverage and cautious interpretation.
- `protein_evidence.csv`: every measured ADT, evidence tier, relative population
  state, module and same-gene associations, and within-site ranges/sign counts.
- `same_gene_site_sensitivity.csv`, `atac_state_sensitivity.csv`, and complete
  copied association tables preserve missing, negative and intermediate outcomes.
- PNG/PDF figures keep different estimands separate; site ranges are not CIs.
- `manifest.json` records donor, inputs and SHA256 fingerprints; `status.json`
  records completion pending review.

No composite activation score is created. Sign consistency alone is not statistical
support, no ADT coverage is not low protein activity, and CITE/Multiome cells are
never paired. The notebook shows relative protein-state sensitivity at 0.25/0.5/0.75.
RNA site replication and unavailable nb05 NES auditing remain explicit pending
checks. Full integration and multi-donor pseudobulk remain outside this milestone.

Local software validation includes a synthetic nb04–08 run, donor mismatch rejection,
missing-site handling, duplicate-key rejection and notebook syntax/schema checks.
The full real-data summary must run in Drive, where upstream outputs are stored.
