# nb14b: investigate persistent capture structure

Run `notebooks/nb14b_capture_structure_investigation.ipynb` after the complete nb14
analysis. It reads frozen joint/RNA coordinates, the restored nb10 shared RNA file,
and full-RNA module scores. It does not train a generative model or change earlier
notebooks. The reusable implementation is `src/capture_structure_diagnostics.py`.

Results are saved under `results/joint_model_diagnostics/capture_structure/`,
including `investigation_report.md`, decision tables, paired donor comparisons,
RNA gene effects, observed counts/QC, training-only projection directions, and
PNG/PDF figures. Generated results follow the repository's existing ignore policy;
the executed notebook embeds the report, tables, and figures.

## Questions and controls

- **Composition:** require at least 50 cells of each capture within the same donor,
  site and harmonized population. Repeat equal-count sampling using three seeds.
  Comparisons refer only to eligible strata; no conclusions are extrapolated to
  excluded or very small populations.
- **Generalization:** hold out entire donors for logistic capture classifiers,
  both across and within populations. Give each training donor/site/type/capture
  group equal weight. RNA PCA scores were fitted unsupervised to the full cohort
  in nb14; these are exploratory representation diagnostics, not an independent
  validation of the unsupervised embedding selection.
- **Geometry:** contrast common and stratum-specific capture effects, and compare
  capture offset directions to references from other donors. Remove one/three
  sequential classifier directions learned on training donors only, versus a
  random rank-three control. Evaluate each held-out donor separately; never merge
  differently transformed folds into a shared embedding.
- **Predictability versus distance:** report the removed within-stratum variance,
  refitted classifier performance, and actual neighbor mixing. A low-variance
  direction can perfectly predict capture while barely affecting Euclidean
  neighborhoods. Below-chance held-out AUROC may reflect a reversed relationship,
  not removal of capture information. Orientation-free AUROC is explicitly
  descriptive and must not be called predictive accuracy.
- **RNA differences:** aggregate raw counts within donor/site/type/capture, compare
  log2(CPM+1), then average sites within donors. Require four donors for gene-level
  summaries and report sign agreement, effect size, expression abundance and
  detection. Thresholded gene lists are descriptive, not differential-expression
  significance tests. Relative abundance and sparse detection can influence them.
- **Biology:** use full-RNA CD14 inflammatory and TNF module scores. Compare local
  and same-site cross-capture gradients against 100 shuffled-reference controls.
  Average sites/directions within donors before comparing diagnostic perturbations.
  RNA-derived checks are not independent validation of an RNA-informed space.

The notebook records the prespecified exploratory thresholds. Bootstrap intervals
resample eight donors rather than cells and should be interpreted cautiously.
Mitochondrial/ribosomal fractions, depth, and detection can reflect biological as
well as technical differences; none is automatically a safe correction target.

To rebuild and execute:

```text
python scripts/build_nb14b.py
python scripts/execute_nb14b.py
python -m unittest discover -s tests -p test_capture_structure_diagnostics.py
```

Rebuilding clears only nb14b's notebook outputs. Execution saves completed cells
on errors. Source checksums and coverage are recorded in the result manifest.
