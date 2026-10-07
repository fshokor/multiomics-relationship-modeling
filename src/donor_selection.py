"""Select a donor by joint cell-type coverage, representation and diversity."""
import numpy as np
import pandas as pd


def summarize_counts(metadata):
    return pd.concat([
        obs.groupby(["DonorID", "Site", "cell_type", "cell_type_harmonized"],
                    observed=True, dropna=False).size().rename("n_cells").reset_index().assign(dataset=d)
        for d, obs in metadata.items()], ignore_index=True)


def select_donor(metadata, min_cells=50, min_shared=3, max_dominance=0.70):
    """Lexicographic ranking, never a hidden weighted score.

    Require >=min_shared populations with >=min_cells in each capture, and
    <=max_dominance in either full (including unresolved) donor population.
    Rank eligible donors by supported shared types, sum of per-type minimum
    cell counts, lower maximum dominance, then donor ID for reproducibility.
    """
    donors = [set(x.DonorID.astype(str)) for x in metadata.values()]
    rows = []
    for donor in sorted(set.intersection(*donors)):
        row = {"DonorID": donor}
        counts = {}
        sites = {}
        for d, obs in metadata.items():
            part = obs[obs.DonorID.astype(str) == donor]
            counts[d] = part.cell_type_harmonized.value_counts()
            # Unresolved cells remain separate original labels for dominance.
            labels = part.cell_type_harmonized.fillna("unresolved:" + part.cell_type.astype(str))
            freq = labels.value_counts() / len(part)
            sites[d] = set(part.Site.astype(str))
            row.update({f"n_{d}": len(part), f"n_original_types_{d}": part.cell_type.nunique(),
                        f"n_harmonized_types_{d}": len(counts[d]),
                        f"dominance_{d}": float(freq.max()),
                        f"effective_types_{d}": float(np.exp(-(freq * np.log(freq)).sum())),
                        f"n_unresolved_{d}": int(part.cell_type_harmonized.isna().sum()),
                        f"sites_{d}": ";".join(sorted(sites[d]))})
        joint = pd.concat(counts, axis=1).fillna(0)
        supported = joint.min(axis=1) >= min_cells
        row["n_shared_types"] = int((joint.min(axis=1) > 0).sum())
        row["n_supported_shared_types"] = int(supported.sum())
        row["joint_cells"] = int(joint.loc[supported].min(axis=1).sum())
        row["max_dominance"] = max(row["dominance_cite"], row["dominance_multiome"])
        row["shared_sites"] = ";".join(sorted(sites["cite"] & sites["multiome"]))
        row["eligible"] = supported.sum() >= min_shared and row["max_dominance"] <= max_dominance
        rows.append(row)
    if not rows:
        raise ValueError("No donor occurs in both datasets")
    table = pd.DataFrame(rows).sort_values(
        ["eligible", "n_supported_shared_types", "joint_cells", "max_dominance", "DonorID"],
        ascending=[False, False, False, True, True]).reset_index(drop=True)
    table["selected"] = False
    table["selection_rule"] = (f">={min_shared} shared types with >={min_cells} cells per capture; "
                               f"dominance <= {max_dominance}; sort by coverage, joint cells, diversity, ID")
    if table.eligible.any():
        table.loc[0, "selected"] = True
    return table
