"""Independent sparse RNA normalization and balanced cell-type specificity."""
import numpy as np
import pandas as pd
from scipy import sparse


def normalize_counts(counts, target_sum=10000):
    x = sparse.csr_matrix(counts, dtype=np.float64, copy=True)
    x.sum_duplicates()
    if not np.isfinite(x.data).all() or (x.data < 0).any() or not np.allclose(x.data, np.rint(x.data), atol=1e-6, rtol=0):
        raise ValueError("RNA counts must be finite, nonnegative integers; do not normalize unknown .X")
    totals = np.asarray(x.sum(axis=1)).ravel()
    if (totals == 0).any():
        raise ValueError("Zero-library RNA cells found; inspect QC before continuing")
    x = sparse.diags(target_sum / totals).dot(x).tocsr()
    x.data = np.log1p(x.data)
    return x


def gene_flags(genes):
    genes = pd.Index(genes).astype(str)
    frame = pd.DataFrame({"gene": genes})
    frame["mitochondrial"] = genes.str.upper().str.startswith("MT-")
    frame["ribosomal"] = genes.str.match(r"^RP[SL]\d")
    frame["housekeeping_flag"] = genes.isin(["ACTB", "GAPDH", "B2M", "MALAT1", "EEF1A1", "TUBB"])
    # Flag, do not remove: B2M and other abundant genes can be biologically meaningful.
    frame["technical_flag"] = genes.str.upper().str.startswith(("ERCC-", "SPIKE"))
    frame["enrichment_eligible"] = ~frame.technical_flag
    return frame


def characterize(x, genes, labels, min_cells=50, median=False):
    """Mean log1p(CP10k), detection and difference from equally weighted other types.

    Equal type weights avoid the largest population defining the comparator.
    The common retained cell-type reference must be supplied to both datasets.
    """
    x = sparse.csr_matrix(x)
    genes, labels = np.asarray(genes, dtype=str), np.asarray(labels, dtype=str)
    if x.shape != (len(labels), len(genes)) or len(set(genes)) != len(genes):
        raise ValueError("Matrix dimensions or unique gene identifiers invalid")
    types, counts = np.unique(labels, return_counts=True)
    keep = types[counts >= min_cells]
    if len(keep) < 2:
        raise ValueError("Specificity requires at least two adequately represented cell types")
    means = np.vstack([np.asarray(x[labels == ct].mean(axis=0)).ravel() for ct in keep])
    rows = []
    for i, ct in enumerate(keep):
        part = x[labels == ct]
        specificity = means[i] - (means.sum(axis=0) - means[i]) / (len(keep) - 1)
        frame = gene_flags(genes)
        frame["cell_type"] = ct
        frame["n_cells"] = part.shape[0]
        frame["mean_expression"] = means[i]
        frame["fraction_expressing"] = np.asarray((part > 0).mean(axis=0)).ravel()
        frame["specificity"] = specificity
        frame["mean_rank"] = pd.Series(means[i]).rank(ascending=False, method="min").to_numpy()
        frame["specificity_rank"] = pd.Series(specificity).rank(ascending=False, method="min").to_numpy()
        if median:
            # Bound memory by gene blocks, never densify the complete donor matrix.
            frame["median_expression"] = np.concatenate([
                np.median(part[:, j:j + 128].toarray(), axis=0) for j in range(0, part.shape[1], 128)])
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def balanced_indices(labels_a, labels_b, seed=42, cap=1000):
    """Independent draws, identical n per cell type; never cell matching."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(labels_a), np.asarray(labels_b)
    ia, ib = [], []
    for ct in sorted(set(a) & set(b)):
        ca, cb = np.flatnonzero(a == ct), np.flatnonzero(b == ct)
        n = min(len(ca), len(cb), cap)
        ia.extend(rng.choice(ca, n, replace=False))
        ib.extend(rng.choice(cb, n, replace=False))
    return np.sort(ia), np.sort(ib)
