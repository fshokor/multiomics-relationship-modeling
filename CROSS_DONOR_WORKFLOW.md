# nb09 — cross-donor validation and RNA pseudobulk

Copy these files into the matching folders of the existing Drive project:

- `notebooks/nb09_cross_donor_validation.ipynb`
- `src/cross_donor.py`
- `src/cross_donor_plots.py`

Existing shared modules from nb04–08 are dependencies. Both original CITE and
Multiome h5ad files must be at the configured benchmark paths. Discovery inputs
are `results/single_donor/manifest.json`, the frozen shared gene universe and GMT.
The discovery mapping audit is checked when available; custom mapping overrides
must be explicitly ported if they differ from the standard mapping policy.

Start with the metadata-only coverage cell. Donors need >=50 CD14 cells and a
comparator in both captures. A common set of discovery-listed reference types is
fixed across all eligible donors/captures before examining outcomes. A changed
reference means discovery results are recomputed, not assumed numerically identical.

Continue through the optional Colab-local file copies and run cell. The full run
reads donor RNA, exports raw pseudobulk, runs site-specific enrichment across the
Hallmark collection, and tests paired ATAC/protein associations for the two predefined
CD14 programs. It includes same-gene protein associations and leave-one-out checks
for CD14/CD88/CD54 where their mapped genes belong to the program. Full direct-ADT
tables are exploratory; the primary readouts are fixed in advance.

Outputs live under `results/cross_donor/nb09_run01/`. Raw pseudobulk is separate for
each donor/capture and retains site × cell type rows with eligibility metadata.
Use `raw_count_sums.npz` with `samples.csv` and `genes.csv` for later count models.
Normalized summed counts and averaged normalized cell expression are separately
labelled descriptive outputs. Small groups remain flagged rather than discarded.
No condition differential expression or formal donor-level test is implemented.

The final summaries exclude the discovery donor from validation-donor medians and
sign counts. Sites are not independent donors, and consistent positive signs alone
are not significance. Per-donor tables retain negative and untested associations.
Inspect site coverage and magnitudes before judging replication.

Use a new output directory for a rerun. An existing status file blocks overwriting,
including partial runs; automatic resume is not implemented. The manifest records
input hashes, protocol and result CSV fingerprints. Large source hashes are computed
at the end, which can add time. No previous notebooks or single-donor results change.

Tests: `python -m unittest discover -s tests -v`. The synthetic two-donor pipeline
checks raw-count sum conservation, distinct mean/sum transformations, saved outputs,
discovery exclusion and overwrite protection. Real-data validation remains in Colab.
