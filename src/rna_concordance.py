"""Population comparisons only: CITE and Multiome cells are separate captures."""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def correlations(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan, np.nan
    return float(pearsonr(a, b).statistic), float(spearmanr(a, b).statistic)


def compare_expression(cite, multiome, universe, top_ns=(50, 100, 250)):
    paired = cite.merge(multiome, on=["cell_type", "gene"], suffixes=("_cite", "_multiome"), validate="one_to_one")
    paired = paired[paired.gene.isin(universe)].copy()
    rows, overlaps = [], []
    for ct, part in paired.groupby("cell_type", sort=True):
        r, rho = correlations(part.mean_expression_cite, part.mean_expression_multiome)
        _, rank_rho = correlations(part.specificity_cite, part.specificity_multiome)
        rows.append(dict(cell_type=ct, n_genes=len(part), n_cite=int(part.n_cells_cite.iloc[0]),
                         n_multiome=int(part.n_cells_multiome.iloc[0]), pearson=r, spearman=rho,
                         specificity_rank_spearman=rank_rho))
        for n in top_ns:
            sets = []
            for d in ("cite", "multiome"):
                # Characteristic genes must have positive specificity, not just top rank.
                top = part[part[f"specificity_{d}"] > 0].sort_values(
                    [f"specificity_{d}", "gene"], ascending=[False, True]).head(n)
                sets.append(set(top.gene))
            a, b = sets
            overlaps.append(dict(cell_type=ct, top_n=n, n_cite=len(a), n_multiome=len(b),
                                 intersection=len(a & b), union=len(a | b),
                                 jaccard=len(a & b) / len(a | b) if a | b else np.nan,
                                 overlapping_genes=";".join(sorted(a & b))))
    return pd.DataFrame(rows), pd.DataFrame(overlaps), paired


def compare_pathways(cite, multiome, alpha=0.05):
    out = cite.merge(multiome, on=["cell_type", "pathway"], how="outer",
                     suffixes=("_cite", "_multiome"), validate="one_to_one")
    def classify(row):
        if pd.isna(row.NES_cite) or pd.isna(row.NES_multiome):
            return "not tested in both"
        a, b = row.q_value_cite <= alpha, row.q_value_multiome <= alpha
        if a and b:
            if row.NES_cite * row.NES_multiome < 0:
                return "discordant"
            return "concordant active" if row.NES_cite > 0 else "concordant negative"
        if a:
            return "CITE-specific"
        if b:
            return "Multiome-specific"
        return "unsupported"
    out["concordance"] = out.apply(classify, axis=1)
    out["classification_note"] = "Relative specificity; negative is not inactivity; specific is not proof of absence"
    return out


def select_programs(concordance, limit=10, min_edge_overlap=2):
    """Candidate shortlist: effect strength, shared leading edge, redundancy, diversity.

    Positive replicated specificity is required. Greedy round-robin across cell
    types avoids one lineage filling the list; redundant edges are suppressed.
    A shortlist can be empty. Biological review remains explicit in the table.
    """
    out = concordance.copy()
    out["selected_for_followup"] = False
    out["reason"] = "Not positively supported in both RNA datasets"
    out["shared_leading_edge"] = [";".join(sorted(set(str(a).split(";")) & set(str(b).split(";")) - {"", "nan"}))
                                  for a, b in zip(out.leading_edge_cite, out.leading_edge_multiome)]
    out["biological_review"] = "Pending: inspect pathway and leading-edge genes"
    out["effect_strength"] = out[["NES_cite", "NES_multiome"]].min(axis=1)
    candidates = out[out.concordance == "concordant active"].sort_values(
        ["effect_strength", "cell_type", "pathway"], ascending=[False, True, True])
    queues = {ct: list(part.index) for ct, part in candidates.groupby("cell_type", sort=True)}
    chosen = []
    while queues and len(chosen) < limit:
        for ct in list(queues):
            if len(chosen) >= limit:
                break
            i = queues[ct].pop(0)
            if not queues[ct]:
                del queues[ct]
            edge = set(out.at[i, "shared_leading_edge"].split(";")) - {""}
            if len(edge) < min_edge_overlap:
                out.at[i, "reason"] = "Insufficient shared leading-edge genes"
                continue
            redundant = False
            for j in chosen:
                other = set(out.at[j, "shared_leading_edge"].split(";")) - {""}
                if out.at[j, "cell_type"] == ct and len(edge & other) / len(edge | other) > 0.7:
                    redundant = True
            if redundant:
                out.at[i, "reason"] = "Redundant shared leading edge (>0.7 Jaccard)"
                continue
            chosen.append(i)
            out.at[i, "selected_for_followup"] = True
            out.at[i, "reason"] = "Replicated positive specificity; shared edge; effect-ranked with cell-type diversity; review required"
    remainder = (out.concordance == "concordant active") & (out.reason == "Not positively supported in both RNA datasets")
    out.loc[remainder, "reason"] = "Replicated candidate outside initial shortlist"
    return out
