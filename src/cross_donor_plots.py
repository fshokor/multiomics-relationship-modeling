"""Show donors individually; no cell-count weighting or inferred significance."""
import matplotlib.pyplot as plt
import numpy as np
from src.single_donor_plots import save


def plot_evidence(result, output):
    for number, (pathway, part) in enumerate(result.groupby('pathway'), 1):
        fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(part) * .6)))
        y = np.arange(len(part))
        axes[0].scatter(part.cite_NES, y, label='CITE RNA NES')
        axes[0].scatter(part.multiome_NES, y, marker='x', label='Multiome RNA NES')
        for field, label in [('atac_adjusted', 'RNA–ATAC'), ('CD14_module_adjusted', 'Module–CD14'), ('CD88_module_adjusted', 'Module–CD88'), ('CD54_module_adjusted', 'Module–CD54')]:
            axes[1].scatter(part[field], y, label=label)
        for ax in axes:
            ax.set_yticks(y, part.donor.astype(str) + ' (' + part.role + ')')
            ax.axvline(0, color='gray', lw=.7)
            ax.legend(fontsize=8)
        axes[0].set_xlabel('RNA enrichment; see table for q-values')
        axes[1].set_xlabel('Depth/site-adjusted Pearson; missing values omitted')
        axes[1].set_xlim(-1, 1)
        fig.suptitle(pathway + ' in CD14 monocytes')
        save(fig, output / f'donor_evidence_{number}')
