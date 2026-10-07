"""Dependency-light weighted preranked enrichment with gene-set permutations.

This is a documented GSEA-style alternative, not Broad's pooled GSEA FDR.
BH q-values adjust nominal same-sign permutation p-values within each ranking.
Gene-set permutations do not preserve gene correlations or support donor inference.
"""
import hashlib
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

HALLMARK_URL = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020"


def read_gmt(path):
    sets = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 3 or fields[0] in sets:
            raise ValueError("Malformed GMT or duplicate pathway name")
        sets[fields[0]] = sorted({g.strip() for g in fields[2:] if g.strip()})
    if not sets:
        raise ValueError("Empty GMT")
    return sets


def cache_hallmark(path):
    """Explicit download, once; retain exact bytes and SHA256 for both datasets."""
    path = Path(path)
    if not path.exists():
        with urlopen(HALLMARK_URL, timeout=120) as response:
            data = response.read()
        text = data.decode("utf-8")
        if not text.startswith("HALLMARK_") or len(text.splitlines()) != 50:
            raise ValueError("Unexpected Hallmark response; supply a reviewed human-symbol GMT")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    read_gmt(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    if len(order):
        out[order] = np.minimum(1, np.minimum.accumulate(
            (p[order] * len(order) / np.arange(1, len(order) + 1))[::-1])[::-1])
    return out


def enrichment_score(scores, hits):
    """Weighted (p=1) running sum, evaluated only at hit boundaries."""
    hits = np.sort(np.asarray(hits, dtype=int))
    n, k = len(scores), len(hits)
    if k == 0 or k == n:
        raise ValueError("Gene set must contain between 1 and N-1 ranked genes")
    weights = np.abs(np.asarray(scores)[hits])
    weights = weights / weights.sum() if weights.sum() else np.full(k, 1 / k)
    after = np.cumsum(weights) - (hits - np.arange(k)) / (n - k)
    before = after - weights
    pos, neg = int(np.argmax(after)), int(np.argmin(before))
    if after[pos] >= abs(before[neg]):
        return float(after[pos]), hits[:pos + 1]
    return float(before[neg]), hits[neg:]


def preranked(ranking, gene_sets, permutations=1000, min_size=15, max_size=500, seed=42):
    if permutations < 100:
        raise ValueError("Use at least 100 permutations (1000 or more for analysis)")
    ranking = ranking.astype(float)
    if ranking.index.duplicated().any() or not np.isfinite(ranking).all():
        raise ValueError("Ranking needs unique gene symbols and finite scores")
    # Stable gene-symbol tie breaker; tie fraction is reported, not jittered away.
    ranking = ranking.sort_index().sort_values(ascending=False, kind="stable")
    genes, scores = ranking.index.to_numpy(), ranking.to_numpy()
    lookup = {g: i for i, g in enumerate(genes)}
    rng = np.random.default_rng(seed)
    nulls, rows = {}, []
    for pathway, members in sorted(gene_sets.items()):
        hits = np.array(sorted({lookup[g] for g in members if g in lookup}), dtype=int)
        k = len(hits)
        if not min_size <= k <= min(max_size, len(scores) - 1):
            continue
        es, edge = enrichment_score(scores, hits)
        if k not in nulls:
            nulls[k] = np.array([enrichment_score(scores, rng.choice(len(scores), k, replace=False))[0]
                                for _ in range(permutations)])
        null = nulls[k]
        side = null[null >= 0] if es >= 0 else null[null < 0]
        denom = np.abs(side).mean() if len(side) else np.nan
        p = (1 + (np.abs(side) >= abs(es)).sum()) / (1 + len(side)) if len(side) else np.nan
        rows.append(dict(pathway=pathway, ES=es, NES=es / denom if denom > 0 else np.nan,
                         p_value=p, direction="positive" if es >= 0 else "negative",
                         leading_edge=";".join(genes[edge]), n_genes=k,
                         n_original_genes=len(set(members)), coverage=k / len(set(members)),
                         permutations=permutations, n_null_same_sign=len(side),
                         tied_fraction=float(ranking.duplicated(keep=False).mean())))
    out = pd.DataFrame(rows, columns=["pathway", "ES", "NES", "p_value", "direction", "leading_edge",
                                      "n_genes", "n_original_genes", "coverage", "permutations",
                                      "n_null_same_sign", "tied_fraction"])
    out["q_value"] = bh_adjust(out.p_value)
    return out


def enrich_programs(programs, gene_sets, universe, **kwargs):
    rows = []
    for ct, part in programs.groupby("cell_type", sort=True):
        rank = part.set_index("gene").loc[list(universe), "specificity"]
        result = preranked(rank, gene_sets, **kwargs)
        result["cell_type"] = ct
        result["n_cells"] = int(part.n_cells.iloc[0])
        rows.append(result)
    return pd.concat(rows, ignore_index=True)
