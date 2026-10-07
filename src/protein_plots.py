"""Descriptive paired CITE figures, with explicit evidence tiers."""
import numpy as np
import matplotlib.pyplot as plt
from src.single_donor_plots import save


def make_figures(tables, pairs, folder):
    names = []
    coverage = tables['coverage']
    fig, ax = plt.subplots(figsize=(10, max(3, len(coverage) * .7)))
    labels = coverage.program_id + ' | ' + coverage.cell_type + ' | ' + coverage.pathway
    ax.barh(labels, coverage.n_direct_genes)
    ax.set_xlabel('Unique pathway genes measured by a directly mapped ADT')
    ax.set_title('Panel coverage: zero means unmeasured, not inactive')
    save(fig, folder / 'direct_coverage')
    names.append('direct_coverage.png')
    for kind in ['module', 'gene']:
        chosen = [p for p in pairs if kind == 'module' or p['gene'] is not None]
        if not chosen:
            continue
        fig, axes = plt.subplots(len(chosen), 1, figsize=(8, 3.5 * len(chosen)), squeeze=False)
        for ax, pair in zip(axes.ravel(), chosen):
            x = pair['rna'] if kind == 'module' else pair['gene_rna']
            y = pair['adt_values']
            valid = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
            sampled = np.random.default_rng(42).choice(valid, min(1500, len(valid)), replace=False)
            ax.scatter(x[sampled], y[sampled], s=5, alpha=.25, rasterized=True)
            ax.set_xlabel('RNA module score' if kind == 'module' else pair['gene'] + ' RNA log1p(CP10k)')
            ax.set_ylabel(pair['adt'] + ' centered log1p ADT')
            ax.set_title(f"{pair['program_id']} | {pair['cell_type']} | {pair['adt']}\n{pair['evidence_type']} (finite cells: {len(valid):,})")
            if not len(valid):
                ax.text(.5, .5, 'Insufficient measurements', transform=ax.transAxes, ha='center')
        name = f'rna_adt_{kind}_scatter'
        save(fig, folder / name)
        names.append(name + '.png')
    assoc = tables['associations']
    target = assoc[assoc.scope == 'target']
    if not target.empty:
        fig, ax = plt.subplots(figsize=(10, max(4, len(target) * .45)))
        labels = target.program_id + ' | ' + target.adt + ' | ' + target.evidence_type
        positions = np.arange(len(target))
        ax.scatter(target.spearman, positions, label='Raw Spearman')
        ax.scatter(target.adjusted_pearson, positions, marker='x', label='Depth/site-adjusted Pearson')
        for pos, row in enumerate(target.itertuples()):
            if not np.isfinite(row.spearman):
                ax.text(0, pos, 'untested', va='center', color='gray')
        ax.set_yticks(positions, labels)
        ax.axvline(0, color='gray', lw=.8)
        ax.set_xlim(-1, 1)
        ax.set_xlabel('Within-target-population correlation (different estimands)')
        ax.set_title('RNA module versus individual ADTs; descriptive associations')
        ax.legend()
        save(fig, folder / 'target_associations')
        names.append('target_associations.png')
    return names
