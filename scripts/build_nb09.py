"""Build only nb09, preserving every earlier executed notebook."""
import json
from pathlib import Path
from build_single_donor_notebooks import code, markdown

cells = [markdown('''
# nb09 — Cross-donor validation and RNA pseudobulk

## Scientific question
Do the CD14 inflammatory and TNF/NF-kB findings recur in additional donors?
How can raw RNA counts be aggregated for later donor-aware population analysis?

## Analysis strategy
Freeze the two discovery-selected programs and the discovery RNA universe/GMT.
Audit donor/site coverage before examining outcomes. Use one common cell-type
reference across eligible donors and captures. Repeat RNA enrichment and paired
RNA–ATAC/RNA–ADT analyses, with site/depth sensitivity and matched-gene removal.
Donor 15078 is discovery, not independent validation. No cross-capture cell matching.

Separately export summed RNA counts by donor × site × cell type × capture, with
cell counts and descriptive normalized means. No condition DE or latent integration.

Copy this notebook, `src/cross_donor.py`, and `src/cross_donor_plots.py` into Drive.
Keep the existing nb04–08 modules and discovery results. Restart a reused runtime.
'''), code('''
from pathlib import Path
import sys, subprocess
try:
    from google.colab import drive
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'anndata>=0.11,<0.13'])
    drive.mount('/content/drive')
    BASE_PATH = Path('/content/drive/MyDrive/multiomics-relationship-modeling')
except ImportError:
    BASE_PATH = Path.cwd().resolve()
    if not (BASE_PATH / 'src').is_dir():
        BASE_PATH = BASE_PATH.parent
sys.path.insert(0, str(BASE_PATH))
import pandas as pd
from IPython.display import display, Image
from configs.paths import CITE_H5AD, MULTIOME_H5AD
from src.cross_donor import audit_inputs, run_cross_donor
PATHS = {'cite': CITE_H5AD, 'multiome': MULTIOME_H5AD}
DISCOVERY = BASE_PATH / 'results/single_donor'
OUTPUT = BASE_PATH / 'results/cross_donor/nb09_run01'
MIN_CELLS = 50
PERMUTATIONS = 1000
SEED = 42
for path in PATHS.values():
    if not path.is_file():
        raise FileNotFoundError(path)
'''), markdown('''
## Coverage first
Eligibility requires at least 50 CD14 cells and an adequately represented comparator
type in both captures. Reference types are the intersection of discovery-listed
types meeting that threshold in every eligible donor/capture. This is fixed before
outcomes and may differ from nb04; the discovery donor is recomputed consistently.
If fewer than two reference types remain, stop and review rather than silently adapt.
Unresolved labels are excluded and audited. Site-specific RNA tests require the
whole fixed reference to meet the threshold; missing tests remain explicit.
'''), code('''
_, _, donors, protocol = audit_inputs(PATHS, DISCOVERY, OUTPUT / 'coverage_audit', MIN_CELLS)
display(donors)
print('Fixed reference types:', protocol['reference_types'])
display(pd.read_csv(OUTPUT / 'coverage_audit/donor_site_coverage.csv'))
print('Independent eligible validation donors:', int(((donors.role == 'validation') & donors.eligible).sum()))
'''), markdown('''
## Optional local Colab copy
Reading HDF5 repeatedly through Drive can be slow. Copy both inputs once to Colab
local storage when disk space permits. The final outputs still go to Drive.
Copying can take time, especially for Multiome. No files are deleted to make room.
'''), code('''
import shutil
COPY_TO_COLAB = str(BASE_PATH).startswith('/content/drive/')
if COPY_TO_COLAB:
    local = Path('/content/nb09_inputs')
    local.mkdir(exist_ok=True)
    required = sum(p.stat().st_size for p in PATHS.values())
    if shutil.disk_usage(local).free < required:
        raise RuntimeError('Insufficient local disk space; use a larger runtime or set COPY_TO_COLAB=False')
    copied = {}
    for capture, source in PATHS.items():
        print('Copying', capture, round(source.stat().st_size / 1e9, 2), 'GB', flush=True)
        destination = local / source.name
        shutil.copyfile(source, destination)
        assert destination.stat().st_size == source.stat().st_size
        copied[capture] = destination
    PATHS = copied
'''), markdown('''
## Run the fixed protocol
This is heavier than nb08: it reads all eligible donor cells, produces pseudobulk,
and performs site-level permutations. Progress is printed by donor/capture.
Raw RNA is selected by feature position, then normalized independently per cell.
ATAC gene activity stays on its stored scale. Protein uses the audited centered-log
panel transform from nb07. Site/depth adjustment is sensitivity analysis, not causality.

All eligible Hallmark sets contribute to BH adjustment in each RNA ranking; only the
two predefined programs are used as primary validation targets. No outcome-based
donor selection. Results for all direct pathway ADTs are saved as exploratory tables;
CD14/CD88/CD54 are the predefined readouts, when their genes belong to the pathway.
Missing membership/coverage is not a zero effect. No new shared leading-edge panel
is selected. RNA–protein leave-one-out removes only the matched gene.

The runner refuses to overwrite a completed or partial run. For a new run, choose
a new OUTPUT folder; there is no automatic resume. Previous results remain available.
'''), code('''
evidence, validation_summary = run_cross_donor(
    PATHS, DISCOVERY, OUTPUT, min_cells=MIN_CELLS,
    permutations=PERMUTATIONS, seed=SEED)
with pd.option_context('display.max_columns', None):
    display(evidence)
    display(validation_summary)
for path in sorted(OUTPUT.glob('donor_evidence_*.png')):
    display(Image(filename=str(path)))
'''), markdown('''
## Inspect site sensitivity and matched-gene removal
Compare RNA only between matching donor/site/capture groups; sites unique to one
capture cannot establish cross-capture replication. Site strata are not extra donors.
An effect consistent in sign may still be negligible: inspect magnitudes and missing
groups. The table below includes all per-donor site results for the two targets.
'''), code('''
from src.cross_donor import PROGRAMS
site_tables, loo_tables = [], []
for folder in sorted(OUTPUT.glob('donor_*')):
    if not folder.is_dir():
        continue
    donor = folder.name.removeprefix('donor_')
    for capture in ['cite', 'multiome']:
        t = pd.read_csv(folder / f'{capture}_enrichment.csv')
        t = t[t.pathway.isin(PROGRAMS)].assign(donor=donor, capture=capture)
        site_tables.append(t)
        display(pd.read_csv(folder / f'{capture}_rna_site_coverage.csv').assign(donor=donor, capture=capture))
    loo_tables.append(pd.read_csv(folder / 'protein_leave_one_out.csv').assign(donor=donor))
sites = pd.concat(site_tables, ignore_index=True)
loo = pd.concat(loo_tables, ignore_index=True)
sites.to_csv(OUTPUT / 'selected_rna_site_results.csv', index=False)
loo.to_csv(OUTPUT / 'all_donor_leave_one_out.csv', index=False)
with pd.option_context('display.max_rows', None):
    display(sites[['donor', 'capture', 'site', 'pathway', 'NES', 'q_value', 'n_null_same_sign']])
    display(loo)
'''), markdown('''
## Pseudobulk: sum versus mean
Each donor folder has `pseudobulk_cite/` and `pseudobulk_multiome/`:

- `raw_count_sums.npz`: integer **sums of raw GEX counts**, rows indexed by samples.csv,
  columns by genes.csv. Groups preserve donor, site and harmonized cell type.
- `samples.csv`: cell count, total library count and >=50-cell eligibility flag.
  Small groups remain exported but must be filtered before downstream inference.
- `sum_log1p_cp10k.npz`: normalized summed counts, descriptive only.
- `mean_cell_log1p_cp10k.npz`: average normalized single-cell expression, a distinct
  descriptive output. It is not count-based pseudobulk.

Pseudobulk includes all resolved types within eligible donors, not just CD14 or the
common validation reference. Unresolved cells are excluded. Matrices remain separate
by capture because panels differ. For later count models, use raw_count_sums with
an appropriate design and library offsets/normalization. Do not treat donor-site
rows as independent people. No DESeq2/edgeR hypothesis test is performed here.
Processed ATAC activity and normalized ADTs are not exported as RNA-like count sums.
'''), code('''
samples = []
for folder in sorted(OUTPUT.glob('donor_*')):
    if folder.is_dir():
        for capture in ['cite', 'multiome']:
            samples.append(pd.read_csv(folder / f'pseudobulk_{capture}/samples.csv').assign(capture=capture))
display(pd.concat(samples, ignore_index=True).groupby(['donor', 'capture', 'eligible_min_cells']).size().rename('n_groups').reset_index())
'''), markdown('''
## Main findings
No cross-donor result is pre-filled. First count eligible validation donors excluding
15078. Then assess positive RNA enrichment in both captures, adjusted associations,
site variation, leave-one-out persistence and unmeasured groups. Report donor effects
individually; medians and sign counts weight donors equally and are descriptive.
If only discovery is eligible, this run supplies no independent replication.

## Limitations
The programs and readouts were selected in donor 15078. Frozen discovery gene filtering
can limit sensitivity elsewhere. A smaller common reference changes the estimand,
so nb04 values need not reproduce numerically. Cell-level correlations are not
donor-level tests; the number and site structure of donors determine later inference.
Unknown ATAC preprocessing, limited ADT panel, possible background and correlated
module genes remain limitations. No causal, disease/control or latent integration
claim is made. Inspect results before planning a pooled statistical model.
''')]

if __name__ == '__main__':
    path = Path(__file__).resolve().parents[1] / 'notebooks/nb09_cross_donor_validation.ipynb'
    if path.exists():
        raise FileExistsError('Refusing to overwrite nb09')
    for i, c in enumerate(cells):
        c['id'] = f'nb09-{i:03d}'
    path.write_text(json.dumps(dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, nbformat=4, nbformat_minor=5), indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
