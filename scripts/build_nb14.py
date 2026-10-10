"""Build only nb14; analysis lives in src/joint_model_diagnostics.py."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


def section(number, title, question, why, source):
    md(f'# {number}. {title}\n\n## Scientific question\n\n{question}\n\n'
       f'## Why this diagnostic matters\n\n{why}')
    code(source)


md('''# nb14 — Why does the joint model preserve identity but align captures poorly?

This notebook diagnoses **saved nb13 outputs without retraining**. CITE and Multiome
contain distinct cells. Shared donors, population labels and RNA provide a bridge;
protein and ATAC are never treated as directly paired measurements.

All neighborhood comparisons use the exact saved nb13 evaluation cells, Euclidean
distances without coordinate standardization, k=30 and ≥50 cells **per assay** for
alignment groups. Original units matter for coordinate perturbations. Report both
unweighted global mixing and composition-adjusted within-group mixing. Concern:
mixing ratio <0.5 **or** scaled centroid distance >1; supported: ratio ≥0.8 and
distance ≤0.5. These inherited thresholds are exploratory, not inferential tests.

Protein depth is observed only in CITE and ATAC depth only in Multiome. Missing
values remain missing throughout. RNA associations are assay-specific. The sample
is donor/assay quota sampled; it is not a population prevalence estimate.

Predeclared practical interpretation: mixing changes of 0.02 and purity losses of
0.02 are material; strong global depth association means |rho|≥0.5. Additional
deletion uses at most the three top coordinates meeting this threshold. These
choices are sensitivity diagnostics, not optimized correction rules. Module local
rho≥0.2 with loss≤0.05 is an exploratory preservation screen, not biological validation.

**Input limitation policy:** missing broad RNA counts or full modules produce explicit
unresolved results, never estimates inferred from earlier summary figures. The
broader panel comparison runs automatically when nb10 `shared_rna.h5ad` is restored.
The full restricted RNA PCA is still executed. Previously validated biology must
not be declared preserved on the basis of restricted-panel proxies alone.
''')
code('''from pathlib import Path
import sys
import pandas as pd
from IPython.display import display, Image, Markdown
try:
    from google.colab import drive
except ImportError:
    BASE = Path.cwd().resolve()
    if not (BASE/'src').exists():
        BASE = BASE.parent
else:
    drive.mount('/content/drive')
    BASE = Path('/content/drive/MyDrive/multiomics-relationship-modeling')
assert (BASE/'src/joint_model_diagnostics.py').exists(), 'Set BASE to the repository root'
sys.path.insert(0, str(BASE))
from src.joint_model_diagnostics import DiagnosticRun
run = DiagnosticRun(BASE)
def show_figure(name):
    display(Image(filename=str(run.output/'figures'/f'{name}.png')))
print('Output:', run.output)
print('No model training, matching, imputation, or modification of notebooks 01–13.')''')
section(1, 'Saved representation and metadata',
        'What representation and metadata are available for diagnosing the current joint model?',
        'Recover exact cell identities and feature scope before interpreting depth. Selected-peak counts are not fragment counts; selected-gene counts are not full RNA library sizes.',
        '''display(run.load())
display(run.bundle['availability'])
display(run.obs.groupby(['assay', 'DonorID', 'Site']).size().rename('n_cells').to_frame())''')
section(2, 'Reproduce the nb13 baseline',
        'Can we reproduce the reported identity preservation and alignment concerns?',
        'Stop on major discrepancies before any perturbation. Recompute purity, mixing, per-population and donor/type centroid metrics. Expected references: purity ≈90.6%, 11/13 and 39/42 concerns.',
        '''display(run.baseline())
display(pd.DataFrame([run.results['joint_full']['summary']]))
show_figure('05_full_latent_assay')
show_figure('07_full_latent_celltype')''')
section(3, 'Every latent dimension: technical and biological associations',
        'Is technical depth concentrated in one coordinate or distributed across the representation?',
        'Spearman and log-depth Pearson correlations quantify continuous effects; unadjusted eta-squared quantifies categorical associations. These effects overlap and cannot be added as independent variance components.',
        '''display(run.associations())
print('Strongest zero-based coordinate:', run.top)
print('Coordinates meeting |rho| >= 0.5:', run.selected)''')
section(4, 'Depth within biological strata',
        'Could the protein-depth association simply reflect population composition, or general sequencing depth?',
        'Use ≥50 measured cells per assay × donor × harmonized cell type. Compare protein, RNA and ATAC separately; report signed and absolute rho, IQR, and effect thresholds. Strata sharing donors are not independent biological replicates.',
        '''display(run.strata())
for name in ['01_latent_protein_depth', '02_latent_RNA_depth', '03_latent_ATAC_depth', '04_stratified_protein_depth']:
    show_figure(name)''')
section(5, 'Coordinate deletion: neighborhood sensitivity',
        'Does the strongest protein-depth-associated coordinate materially affect cross-capture alignment?',
        'Recompute neighbors, UMAP and all alignment metrics after deletion; if justified also drop up to three strong coordinates. Deletion removes biological and technical information together and is not a valid final correction.',
        '''display(run.perturb())
show_figure('06_minus_top1_assay')
show_figure('08_minus_top1_celltype')''')
section(6, 'Centered depth residualization',
        'Is neighborhood separation sensitive specifically to the within-stratum depth component?',
        'Within each adequately sized CITE donor/type stratum fit latent coordinate against log1p ADT total; subtract only the centered fitted depth term. Preserve stratum means and all Multiome values. This isolates within-stratum depth variation, but cannot remove between-capture offsets or infer unobserved Multiome protein depth. Labels are used: this is not a deployment method.',
        '''display(run.residualize())
display(pd.read_csv(run.output/'depth_coordinate/residualization_coefficients.csv'))''')
section(7, 'Neighborhood dependence on ADT depth',
        'Are local neighborhoods organized by protein depth, and does deletion weaken that organization?',
        'Compare query depth with the median of observed CITE neighbors; report observed-neighbor coverage and a same-donor/type random-neighbor reference of matched size. Multiome depth is not measurable. Compare type, assay, donor and site similarity on the same neighborhoods.',
        '''display(run.neighborhoods())''')
section(8, 'Capture predictability on held-out donors',
        'Do depth-associated coordinates contribute to predicting assay identity?',
        'Fit balanced logistic regression with training-fold-only standardization, holding out each donor in turn. Use identical folds for full and deleted spaces; report balanced accuracy, AUROC and standardized coefficients. The depth-coordinate selection is exploratory on the full diagnostic set, so this is not unbiased model selection. Label-informed residualization is excluded.',
        '''display(run.predictability())
display(run.predictions)''')
section(9, 'Restricted versus broader RNA information',
        'Does restricting RNA to the 994 shared genes limit alignment?',
        'Fit identical 30-PC RNA-only representations on all saved cohort cells, evaluate on the same nb13 subset. Each panel uses raw counts → panel-wise CP10k → log1p → centered PCA, matching nb13’s RNA baseline. Broad RNA uses the complete common feature set saved by nb10 (12,059 genes in its manifest), without selecting new features. This measures panel restriction including its normalization denominator; it is not a pure gene-deletion experiment. An old integrated nb10 embedding is not a controlled substitute.',
        '''display(run.rna_panels())
display(pd.DataFrame({k:v['summary'] for k,v in run.results.items() if k.startswith('rna_')}).T)''')
section(10, 'Validated program gene retention',
        'Are the CD14 inflammatory and TNF/NF-κB programs represented in the restricted panel?',
        'Verify the frozen GMT fingerprint and use the original validated universe. Report original and retained genes, and discovery CD14 leading edges (union across assays). Module scoring reuses the project’s donor/assay/type-balanced reference moments. Compare full and restricted scores by donor/assay when full RNA is available. Restricted scores alone are explicitly marked as proxies.',
        '''display(run.modules())
print(run.module_status)''')
section(11, 'Population-specific failure modes',
        'Are problematic populations failing in the same way?',
        'Allow multiple flags: large centroid shift, capture-specific neighborhoods, donor/site enrichment, purity below 0.8, or insufficient cells. Donor/site excess same-label fractions >0.15 are descriptive flags relative to within-type composition. Capture-specific neighborhoods suggest subclustering; they do not establish a distinct cluster mechanism.',
        '''display(run.failures())''')
section(12, 'Donor × cell-type alignment groups',
        'What technical and sampling patterns accompany the 39/42 concern groups?',
        'Save both capture centroids, counts, sites, technical medians and local mixing. Report group-level descriptive associations and donor-median correlations. Do not use cell counts as independent biological replicates. The donor/site contingency table exposes confounding.',
        '''display(run.donor_groups())
display(run.donor_table.drop(columns=[c for c in run.donor_table if '_centroid_' in c]))
display(pd.crosstab(run.obs.DonorID, run.obs.Site))''')
section(13, 'CD14 gradient preservation across spaces',
        'Would improved mixing erase inflammatory or TNF/NF-κB gradients?',
        'Quantify score versus neighbor-mean correlation within donor/assay CD14 cells, with 20 shuffled-score references. Also quantify cross-assay directional neighborhood score prediction within donors using the existing repository function. These descriptive RNA-based checks have RNA-input circularity and are not independent validation. Full RNA scores are required for a definitive preservation claim.',
        '''display(run.gradients())
print(run.module_status)
display(run.cross.groupby(['representation','module']).spearman.agg(['median','count']))''')
section(14, 'Protein preprocessing and feature-depth dependence',
        'Could the current protein input be unusually sensitive to library depth?',
        'Audit nb11/nb13 raw-count preparation and compare raw/log1p ADT with explicitly defined within-cell centered log1p CLR. A transformation reducing feature-depth correlations is not evidence it would improve the joint model. Do not feed CLR into a count likelihood. No denoising or retraining is performed.',
        '''display(run.proteins())
display(Markdown((run.output/'depth_coordinate/protein_preprocessing.md').read_text()))''')
section(15, 'Evidence table and final quantitative comparison',
        'Which hypotheses are supported by the executed results?',
        'Keep unresolved hypotheses unresolved. The final protein-depth column is maximum within-CITE |rho| over dimensions; CD14 preservation is never declared from proxy scores alone. Global mixing improvement must be assessed alongside population alignment and biology.',
        '''comparison, decisions, recommendation = run.finish()
display(comparison)
display(decisions)
for name in ['09_mixing_comparison','10_purity_comparison','11_celltype_alignment',
             '12_RNA_panel_comparison','13_module_retention','14_CD14_gradient',
             '15_CD14_gradient','16_alignment_vs_depth','17_neighbor_depth_dependence']:
    show_figure(name)''')
section(16, 'Decision before nb15',
        'What should change—or remain unchanged—before the next model?',
        'The generated recommendation uses only executed diagnostics. No training-duration extension, coordinate deletion as a final correction, paired-cell interpretation, imputation validation, or ATAC→RNA→protein causality follows from these results.',
        '''display(Markdown(recommendation))
display(pd.read_json(run.output/'status.json', typ='series'))''')
nb = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11'}})
target = ROOT/'notebooks/nb14_joint_model_diagnostics.ipynb'
nbf.write(nb, target)
print(target)
