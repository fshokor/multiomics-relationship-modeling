"""Re-render saved nb14 tables/figures after presentation edits; no metric refit."""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
import json
import base64
from pathlib import Path
import pandas as pd
import nbformat

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from src.joint_model_diagnostics import comparison_figures, decision_tables, plot_umap

out = root/'results/joint_model_diagnostics'
read = lambda name: pd.read_csv(out/name)
comparison = read('final_comparison.csv')
retention = read('module_preservation/gene_retention.csv')
status = json.loads((out/'module_preservation/status.json').read_text())['status']
decisions, report = decision_tables(comparison, read('latent_associations/all_dimensions.csv'),
    read('latent_associations/strata_summary.csv'), read('depth_coordinate/assay_prediction_folds.csv'),
    read('celltype_failures/classification.csv'), status, retention)
decisions.to_csv(out/'hypothesis_decision_table.csv', index=False)
(out/'final_recommendation.md').write_text(report, encoding='utf-8')
results = {}
for name in comparison.representation:
    folder = 'baseline' if name == 'joint_full' else 'rna_panel' if name.startswith('rna_') else 'depth_coordinate'
    path = out/folder/f'{name}_types.csv'
    if path.exists():
        results[name] = {'types': pd.read_csv(path)}
depth = {name: read(f'depth_coordinate/{name}_neighbor_depth.csv') for name in ['joint_full', 'minus_top1']}
comparison_figures(results, comparison, depth, read('donor_celltype/groups.csv'), retention,
                   read('module_preservation/local_gradients.csv'), out, status)
obs = pd.read_csv(out/'baseline/cell_metadata.csv', index_col=0)
for name, folder, coordfile, numbers in [
        ('full_latent', 'baseline', 'recomputed_umap.csv', [5, 7]),
        ('minus_top1', 'depth_coordinate', 'minus_top1_umap.csv', [6, 8])]:
    coords = pd.read_csv(out/folder/coordfile, index_col=0).reindex(obs.index).to_numpy()
    for number, field, label in zip(numbers, ['assay', 'cell_type_harmonized'], ['assay', 'celltype']):
        plot_umap(coords, obs, field, out, f'{number:02d}_{name}_{label}')
nbpath = root/'notebooks/nb14_joint_model_diagnostics.ipynb'
nb = nbformat.read(nbpath, as_version=4)
figures = ['09_mixing_comparison','10_purity_comparison','11_celltype_alignment',
           '12_RNA_panel_comparison','13_module_retention','14_CD14_gradient',
           '15_CD14_gradient','16_alignment_vs_depth','17_neighbor_depth_dependence']
for cell in nb.cells:
    if cell.cell_type != 'code':
        continue
    if cell.source.startswith('comparison, decisions, recommendation = run.finish()'):
        images = [o for o in cell.outputs if 'image/png' in o.get('data', {})]
        assert len(images) == len(figures)
        for output, name in zip(images, figures):
            output.data['image/png'] = base64.b64encode((out/'figures'/f'{name}.png').read_bytes()).decode()
    for prefix, names in [('display(run.baseline())', ['05_full_latent_assay', '07_full_latent_celltype']),
                           ('display(run.perturb())', ['06_minus_top1_assay', '08_minus_top1_celltype'])]:
        if cell.source.startswith(prefix):
            images = [o for o in cell.outputs if 'image/png' in o.get('data', {})]
            assert len(images) == len(names)
            for output, name in zip(images, names):
                output.data['image/png'] = base64.b64encode((out/'figures'/f'{name}.png').read_bytes()).decode()
    if cell.source.startswith('display(Markdown(recommendation))'):
        for output in cell.outputs:
            if 'text/markdown' in output.get('data', {}):
                output.data['text/markdown'] = report
nb.cells = [c for c in nb.cells if not (c.cell_type == 'markdown' and c.source.startswith('# Main findings\n'))]
nbformat.validate(nb)
nbformat.write(nb, nbpath)
print('Refreshed report and figures from saved executed metrics.')
