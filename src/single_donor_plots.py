"""Simple saved figures; no biological claims inferred by plotting code."""
import numpy as np
import matplotlib.pyplot as plt


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def heatmap(frame, path, title, center=True):
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(max(7, len(frame.columns) * .55), max(4, len(frame) * .3)))
    limit = np.nanmax(np.abs(frame.to_numpy())) if center else None
    im = ax.imshow(np.ma.masked_invalid(frame.to_numpy(float)), aspect="auto", cmap="RdBu_r" if center else "viridis",
                   vmin=-limit if center and limit else None, vmax=limit if center and limit else None)
    ax.set_xticks(range(len(frame.columns)), frame.columns, rotation=65, ha="right")
    ax.set_yticks(range(len(frame)), frame.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=.65)
    save(fig, path)


def cell_counts(counts, path):
    fig, ax = plt.subplots(figsize=(10, max(4, len(counts) * .4)))
    counts.plot.barh(ax=ax)
    ax.set_xlabel("Cells retained for RNA comparison")
    ax.set_title("Selected donor: independent captures")
    save(fig, path)


def gene_heatmap(programs, path):
    # Ribosomal/mitochondrial genes remain in full tables and enrichment;
    # omit only from this compact marker display, not from biological analysis.
    view = programs[~programs.mitochondrial & ~programs.ribosomal & ~programs.technical_flag]
    genes = view.sort_values(["specificity", "gene"], ascending=[False, True]).groupby("cell_type").head(3).gene.unique()
    matrix = programs[programs.gene.isin(genes)].pivot(index="gene", columns="cell_type", values="mean_expression")
    z = matrix.sub(matrix.mean(axis=1), axis=0).div(matrix.std(axis=1).replace(0, np.nan), axis=0)
    heatmap(z, path, "Characteristic genes: row-standardized mean RNA")


def concordance_scatter(paired, path, x="mean_expression_cite", y="mean_expression_multiome", annotate="gene"):
    types = sorted(paired.cell_type.unique())
    if not types:
        return
    cols = min(3, len(types))
    fig, axes = plt.subplots(int(np.ceil(len(types) / cols)), cols, squeeze=False,
                             figsize=(5 * cols, 4 * int(np.ceil(len(types) / cols))))
    from src.rna_concordance import correlations
    markers = {"CD3D", "IL7R", "CD8A", "NKG7", "MS4A1", "CD14", "FCGR3A", "CD34", "HBB", "GZMB"}
    for ax, ct in zip(axes.flat, types):
        part = paired[paired.cell_type == ct].dropna(subset=[x, y])
        ax.scatter(part[x], part[y], s=7, alpha=.35, rasterized=True)
        r, rho = correlations(part[x], part[y])
        ax.set_title(f"{ct}\nr={r:.2f}, rho={rho:.2f}; n={len(part)}")
        ax.set_xlabel(x.replace("_", " "))
        ax.set_ylabel(y.replace("_", " "))
        if annotate == "gene":
            for row in part[part.gene.isin(markers)].itertuples():
                ax.annotate(row.gene, (getattr(row, x), getattr(row, y)), fontsize=6)
    for ax in list(axes.flat)[len(types):]:
        ax.set_visible(False)
    save(fig, path)


def pathway_heatmap(table, path, limit=20):
    if table.empty:
        return
    terms = table.groupby("pathway").NES.apply(lambda x: x.abs().max()).nlargest(limit).index
    part = table[table.pathway.isin(terms)].copy()
    part["column"] = part.dataset + " | " + part.cell_type
    heatmap(part.pivot(index="pathway", columns="column", values="NES"), path,
            "Pathway NES: relative cell-type specificity (not absolute activity)")
