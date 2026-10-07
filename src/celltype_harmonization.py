"""Conservative, auditable label harmonization for GSE194122.

Rules are proposed biological hierarchies, not inferred equivalences of states.
Unknown labels remain unresolved. Original annotations are never overwritten.
"""
import pandas as pd


def label_rule(label):
    label = str(label)
    for prefix, target in [("CD4+ T ", "CD4 T cells"), ("CD8+ T", "CD8 T cells"),
                           ("Naive CD20+ B", "B cells"), ("B1 B", "B cells")]:
        if label.startswith(prefix):
            return target, "hierarchical", "Broad lineage; subtype composition may differ"
    if label == "NK" or label.startswith("NK CD"):
        return "NK cells", "hierarchical", "NK lineage; receptor subtypes pooled"
    exact = {
        "Transitional B": "B cells", "CD14+ Mono": "CD14 monocytes",
        "CD16+ Mono": "CD16 monocytes", "HSC": "HSC",
        "Lymph prog": "Lymphoid progenitors", "G/M prog": "G/M progenitors",
        "MK/E prog": "MK/E progenitors", "pDC": "pDC", "cDC2": "cDC2",
        "cDC1": "cDC1", "Erythroblast": "Erythroid populations",
        "Proerythroblast": "Erythroid populations", "Normoblast": "Erythroid populations",
        "Reticulocyte": "Erythroid populations", "ILC": "ILC",
    }
    if label in exact:
        return exact[label], "hierarchical", "Explicit lineage rule; inspect original subtype counts"
    if label == "Plasma cell" or label.startswith("Plasma cell IGKC"):
        return "Plasma cells", "hierarchical", "Plasma lineage; light-chain subsets pooled"
    # T reg, ILC1, MAIT, gdT, dnT, cycling progenitors and plasmablasts are
    # intentionally not assumed equivalent to broader comparator labels.
    return None, "unresolved", "No confident rule; excluded unless explicitly reviewed"


def mapping_table(cite_labels, multiome_labels):
    """Long audit table, with explicit exact-label handling before lineage rules."""
    labels = {"cite": set(map(str, cite_labels)), "multiome": set(map(str, multiome_labels))}
    common = labels["cite"] & labels["multiome"]
    rows = []
    for dataset, values in labels.items():
        for original in sorted(values):
            target, confidence, reason = label_rule(original)
            if target is None and original in common:
                target, confidence, reason = original, "exact", "Same original label in both datasets"
            mapping_type = confidence
            confidence = ("unresolved" if target is None else
                          "high" if mapping_type == "exact" or target in
                          {"HSC", "Lymphoid progenitors", "G/M progenitors", "MK/E progenitors",
                           "pDC", "cDC1", "cDC2", "CD14 monocytes", "CD16 monocytes", "ILC"}
                          else "moderate: broad lineage only")
            rows.append(dict(dataset=dataset, original_label=original,
                             harmonized_label=target, mapping_confidence=confidence,
                             mapping_type=mapping_type, exact_label_in_both=original in common, reason=reason))
    return pd.DataFrame(rows)


def apply_mapping(obs, mapping, dataset):
    part = mapping[mapping.dataset == dataset]
    if part.original_label.duplicated().any():
        raise ValueError("Mapping contains duplicate dataset/label rows")
    out = obs.copy()
    out["cell_type_original"] = out["cell_type"].astype(str)
    out["cell_type_harmonized"] = out.cell_type_original.map(part.set_index("original_label").harmonized_label)
    return out


def paired_mapping_table(mapping):
    rows = []
    for target, part in mapping.dropna(subset=["harmonized_label"]).groupby("harmonized_label"):
        labels = {d: "; ".join(sorted(part.loc[part.dataset == d, "original_label"]))
                  for d in ("cite", "multiome")}
        rows.append({"CITE label": labels["cite"], "Multiome label": labels["multiome"],
                     "Harmonized label": target,
                     "Mapping confidence": "; ".join(sorted(set(part.mapping_confidence))),
                     "Reason": "; ".join(sorted(set(part.reason)))})
    for row in mapping[mapping.harmonized_label.isna()].itertuples():
        rows.append({"CITE label": row.original_label if row.dataset == "cite" else "",
                     "Multiome label": row.original_label if row.dataset == "multiome" else "",
                     "Harmonized label": None, "Mapping confidence": "unresolved", "Reason": row.reason})
    return pd.DataFrame(rows)
