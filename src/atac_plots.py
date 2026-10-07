"""Saved RNA–ATAC figures. Plot sampling never changes reported statistics."""
import numpy as np
import matplotlib.pyplot as plt
from src.single_donor_plots import save, heatmap


def make_figures(tables, rna, atac, genes, obs, folder, seed=42):
    rng = np.random.default_rng(seed)
    files = []
    scores = tables['cell_scores']
    if scores.empty:
        return files
    ids = list(scores.program_id.unique())
    fig, axes = plt.subplots(len(ids), 1, figsize=(7, 4 * len(ids)), squeeze=False)
    for ax, pid in zip(axes.flat, ids):
        part = scores[(scores.program_id == pid) & (scores.cell_type_harmonized == scores.target_cell_type)]
        subset = part.iloc[rng.choice(len(part), min(2000, len(part)), replace=False)]
        for site in sorted(subset.Site.unique()):
            points = subset[subset.Site == site]
            ax.scatter(points.rna_score, points.atac_score, s=8, alpha=.3, label=str(site), rasterized=True)
        stats = tables['associations']
        stats = stats[(stats.program_id == pid) & (stats.scope == 'cell_type') & (stats.group == stats.cell_type)].iloc[0]
        ax.set(xlabel='RNA module: mean gene z-score', ylabel='ATAC module: mean gene z-score',
               title=f'{pid}: {stats.cell_type} | {stats.pathway}\nn={len(part)}; rho={stats.spearman:.2f}; adjusted r={stats.adjusted_pearson:.2f}')
        ax.legend(title='Site', fontsize=7)
    save(fig, folder / 'rna_atac_module_scatter')
    files.append('rna_atac_module_scatter.png')
    summary = tables['population_summary'].copy()
    summary['label'] = summary.program_id + ' | ' + summary.cell_type + ' | ' + summary.pathway
    matrix = summary.pivot(index='label', columns='evaluated_cell_type', values=['rna_mean', 'atac_mean'])
    matrix.columns = [f'{ct} | {layer.split("_")[0].upper()}' for layer, ct in matrix.columns]
    heatmap(matrix, folder / 'population_activity_heatmap', 'Relative program activity: same covered genes; separate layer standardization')
    files.append('population_activity_heatmap.png')
    target = summary[summary.cell_type == summary.evaluated_cell_type].set_index('label')[['rna_mean', 'atac_mean']]
    target.columns = ['Multiome RNA', 'ATAC gene activity']
    heatmap(target, folder / 'target_activity_heatmap', 'Target populations: mean gene z-score (not absolute activity)')
    files.append('target_activity_heatmap.png')
    gene_stats = tables['leading_edge_genes'].copy()
    if not gene_stats.empty:
        # Predefined display rule: detection coverage, never strongest correlation.
        gene_stats['joint_detection'] = gene_stats.rna_detection * gene_stats.atac_nonzero_fraction
        chosen = gene_stats.sort_values(['joint_detection', 'gene'], ascending=[False, True]).groupby('program_id', sort=True).head(3)
        chosen.to_csv(folder.parent / 'plotted_genes.csv', index=False)
        fig, axes = plt.subplots(int(np.ceil(len(chosen) / 3)), 3, squeeze=False,
                                 figsize=(15, 4 * int(np.ceil(len(chosen) / 3))))
        for ax, row in zip(axes.flat, chosen.itertuples()):
            j = list(genes).index(row.gene)
            cells = np.flatnonzero(obs.cell_type_harmonized.to_numpy() == row.cell_type)
            cells = rng.choice(cells, min(1500, len(cells)), replace=False)
            ax.scatter(rna[cells, j], atac[cells, j], s=6, alpha=.25, rasterized=True)
            ax.set(xlabel='RNA log1p(CP10k)', ylabel='ATAC gene activity as stored',
                   title=f'{row.program_id} | {row.cell_type}\n{row.gene}: rho={row.spearman:.2f}; n={row.n_cells}')
        for ax in list(axes.flat)[len(chosen):]:
            ax.set_visible(False)
        save(fig, folder / 'leading_edge_gene_scatter')
        files.append('leading_edge_gene_scatter.png')
    stats = tables['associations']
    site = stats[stats.scope == 'target_by_site'].copy()
    site['label'] = site.program_id + ' | ' + site.cell_type
    if site.spearman.notna().any():
        heatmap(site.pivot(index='label', columns='group', values='spearman'), folder / 'site_association_heatmap',
                'Within-target RNA–ATAC Spearman by site; untested cells blank')
        files.append('site_association_heatmap.png')
    return files
