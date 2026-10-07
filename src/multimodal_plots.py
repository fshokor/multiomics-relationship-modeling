"""Separate panels for distinct estimands; missing values never become zeros."""
import numpy as np
import matplotlib.pyplot as plt
from src.single_donor_plots import save


def make_figures(summary, detail, folder):
    names = []
    labels = summary.cell_type + ' | ' + summary.pathway
    fig, axes = plt.subplots(1, 3, figsize=(16, max(4, len(summary) * .65)), sharey=True)
    y = np.arange(len(summary))
    axes[0].scatter(summary.NES_cite, y, label='CITE RNA')
    axes[0].scatter(summary.NES_multiome, y, marker='x', label='Multiome RNA')
    axes[0].set_xlabel('RNA enrichment NES')
    axes[1].scatter(summary.activity_rna_mean, y, label='Multiome RNA')
    axes[1].scatter(summary.activity_atac_mean, y, marker='x', label='ATAC gene activity')
    axes[1].set_xlabel('Relative population mean module score')
    axes[2].scatter(summary.atac_pearson, y, label='Raw Pearson')
    axes[2].scatter(summary.atac_adjusted_pearson, y, marker='x', label='Adjusted Pearson')
    axes[2].set_xlabel('Paired Multiome within-type correlation')
    axes[2].set_xlim(-1, 1)
    axes[0].set_yticks(y, labels)
    for ax in axes:
        ax.axvline(0, color='gray', lw=.7)
        ax.legend(fontsize=8)
    fig.suptitle('Different measurements shown separately; no combined activation scale')
    save(fig, folder / 'rna_atac_evidence')
    names.append('rna_atac_evidence.png')
    if not detail.empty:
        labels = detail.cell_type + ' | ' + detail.pathway + ' | ' + detail.adt + ' | ' + detail.evidence_type
        fig, ax = plt.subplots(figsize=(12, max(4, len(detail) * .4)))
        y = np.arange(len(detail))
        ax.hlines(y, detail.site_adjusted_min, detail.site_adjusted_max, color='gray', label='Within-site range (not CI)')
        ax.scatter(detail.adjusted_pearson, y, label='Module–ADT adjusted Pearson')
        ax.scatter(detail.same_gene_adjusted_pearson, y, marker='x', label='Same-gene adjusted Pearson')
        ax.set_yticks(y, labels)
        ax.axvline(0, color='gray', lw=.7)
        ax.set_xlim(-1, 1)
        ax.legend(fontsize=8)
        ax.set_title('CITE evidence: direct matches and phenotypic context remain distinct')
        save(fig, folder / 'protein_site_evidence')
        names.append('protein_site_evidence.png')
    return names
