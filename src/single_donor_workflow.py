"""Two incremental RNA milestones. Later molecular layers require real-data review."""
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from src.celltype_harmonization import mapping_table, apply_mapping, paired_mapping_table
from src.donor_selection import select_donor, summarize_counts
from src.expression_programs import normalize_counts, characterize, balanced_indices, gene_flags
from src.pathway_enrichment import read_gmt, enrich_programs, bh_adjust
from src.rna_concordance import compare_expression, compare_pathways, select_programs, correlations
from src.single_donor_io import inspect_h5ad, read_rna_counts, output_dirs, write_json, sha256, save_rna, load_rna
from src import single_donor_plots as plots


def prepare_donor(paths, output_root, min_cells=50, min_shared=3, max_dominance=.7, mapping_override=None):
    dirs = output_dirs(output_root)
    # Invalidate completion state before any rerun, so a failed run cannot pass a gate.
    write_json(Path(output_root) / "status.json", {"rna_characterization": "in_progress", "rna_concordance": "pending"})
    inspected = {d: inspect_h5ad(p) for d, p in paths.items()}
    metadata = {d: result[0] for d, result in inspected.items()}
    mapping = mapping_table(metadata["cite"].cell_type.unique(), metadata["multiome"].cell_type.unique())
    if mapping_override is not None:
        reviewed = pd.read_csv(mapping_override)
        required = set(mapping.columns)
        if not required <= set(reviewed.columns):
            raise ValueError(f"Mapping override needs columns {sorted(required)}")
        if set(zip(reviewed.dataset, reviewed.original_label)) != set(zip(mapping.dataset, mapping.original_label)):
            raise ValueError("Reviewed mapping must cover exactly the observed dataset/label pairs")
        if reviewed.reason.isna().any():
            raise ValueError("Every mapping decision needs a reason")
        mapping = reviewed
    mapping.to_csv(dirs["donor_selection"] / "celltype_mapping.csv", index=False)
    paired_mapping_table(mapping).to_csv(dirs["donor_selection"] / "celltype_mapping_paired.csv", index=False)
    metadata = {d: apply_mapping(obs, mapping, d) for d, obs in metadata.items()}
    summarize_counts(metadata).to_csv(dirs["donor_selection"] / "all_donor_site_celltype_counts.csv", index=False)
    donors = select_donor(metadata, min_cells, min_shared, max_dominance)
    donors.to_csv(dirs["donor_selection"] / "donor_ranking.csv", index=False)
    if not donors.selected.any():
        raise ValueError("No eligible donor: review saved ranking and thresholds; no silent fallback")
    donor = donors.loc[donors.selected, "DonorID"].iloc[0]
    selected = {d: obs[obs.DonorID == donor].copy() for d, obs in metadata.items()}
    summarize_counts(selected).to_csv(dirs["donor_selection"] / "selected_donor_original_counts.csv", index=False)
    count_table = pd.concat({d: obs.cell_type_harmonized.value_counts() for d, obs in selected.items()}, axis=1).fillna(0).astype(int)
    count_table["included"] = count_table.min(axis=1) >= min_cells
    count_table.to_csv(dirs["donor_selection"] / "shared_celltype_counts.csv", index_label="cell_type")
    retained = count_table.index[count_table.included].tolist()
    selected_mapping = pd.concat([mapping[(mapping.dataset == d) & mapping.original_label.isin(obs.cell_type_original)]
                                  for d, obs in selected.items()])
    selected_mapping.to_csv(dirs["donor_selection"] / "selected_donor_mapping.csv", index=False)
    plots.cell_counts(count_table.loc[retained, ["cite", "multiome"]], dirs["donor_selection"] / "figures/cell_counts")
    data = {}
    for d, path in paths.items():
        full = metadata[d]
        mask = (full.DonorID == donor) & full.cell_type_harmonized.isin(retained)
        rows = np.flatnonzero(mask.to_numpy())
        counts, genes = read_rna_counts(path, rows, inspected[d][1])
        obs = full.iloc[rows].copy()
        x = normalize_counts(counts)
        obs["rna_library_counts"] = np.asarray(counts.sum(axis=1)).ravel()
        save_rna(dirs[f"rna_{d}"], x, obs, genes)
        data[d] = (x, obs, genes)
        del counts
    manifest = {"donor": donor, "min_cells": min_cells, "min_shared": min_shared,
                "max_dominance": max_dominance, "retained_cell_types": retained,
                "inputs": {d: result[2] for d, result in inspected.items()},
                "normalization": "RNA-only raw counts -> total 10000 over each capture's GEX panel -> log1p",
                "specificity": "mean log1p CP10k minus equally weighted other retained cell-type means",
                "mapping_sha256": sha256(dirs["donor_selection"] / "celltype_mapping.csv"),
                "python": platform.python_version(), "limitations": ["single donor", "separate captures", "no condition DE"]}
    manifest["package_versions"] = {}
    for package in ("numpy", "pandas", "scipy", "anndata", "h5py", "matplotlib"):
        try:
            manifest["package_versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            manifest["package_versions"][package] = "unavailable"
    write_json(Path(output_root) / "manifest.json", manifest)
    return donor, donors, count_table, data


def characterize_donor(data, output_root, gmt_path, permutations=1000, seed=42, detection=.01, median=False):
    dirs = output_dirs(output_root)
    write_json(Path(output_root) / "status.json", {"rna_characterization": "in_progress", "rna_concordance": "pending"})
    manifest = json.loads((Path(output_root) / "manifest.json").read_text())
    programs = {d: characterize(x, genes, obs.cell_type_harmonized, manifest["min_cells"], median)
                for d, (x, obs, genes) in data.items()}
    shared = sorted(set(data["cite"][2]) & set(data["multiome"][2]))
    # One symmetric universe across every cell type and both RNA captures.
    # Gene must be detected in >=detection of at least one retained type in EACH capture.
    eligible = []
    for d, table in programs.items():
        detected = table.groupby("gene").fraction_expressing.max()
        eligible.append(set(detected[detected >= detection].index))
    flags = gene_flags(shared)
    universe = sorted(set(shared) & eligible[0] & eligible[1] & set(flags.loc[flags.enrichment_eligible, "gene"]))
    if len(universe) < 20:
        raise ValueError("Too few shared detected genes for pathway comparisons")
    pd.DataFrame({"gene": shared, "included_in_comparison": [g in set(universe) for g in shared]}).to_csv(
        dirs["rna_concordance"] / "shared_gene_universe.csv", index=False)
    sets = read_gmt(gmt_path)
    gene_set_audit = pd.DataFrame([dict(pathway=p, n_original=len(g), n_overlap=len(set(g) & set(universe)),
                                      included=15 <= len(set(g) & set(universe)) <= min(500, len(universe) - 1))
                                  for p, g in sets.items()])
    gene_set_audit.to_csv(dirs["gsea"] / "gene_set_coverage.csv", index=False)
    enrichment = {}
    for d, table in programs.items():
        table["comparison_eligible"] = table.gene.isin(universe)
        table.to_csv(dirs[f"rna_{d}"] / "gene_programs.csv", index=False)
        for ranking in ("mean", "specificity"):
            table.sort_values(["cell_type", ranking + "_rank", "gene"]).to_csv(
                dirs[f"rna_{d}"] / f"{ranking}_ranking.csv", index=False)
        plots.gene_heatmap(table, dirs[f"rna_{d}"] / "figures/characteristic_genes")
        enrichment[d] = enrich_programs(table, sets, universe, permutations=permutations, seed=seed)
        enrichment[d]["dataset"] = d
        enrichment[d]["donor"] = manifest["donor"]
        enrichment[d].to_csv(dirs["gsea"] / f"{d}_enrichment.csv", index=False)
        plots.pathway_heatmap(enrichment[d], dirs[f"rna_{d}"] / "figures/pathway_enrichment")
    all_enrichment = pd.concat(enrichment.values(), ignore_index=True)
    all_enrichment["q_value_global"] = bh_adjust(all_enrichment.p_value)
    all_enrichment.to_csv(dirs["gsea"] / "all_enrichment.csv", index=False)
    # Preserve the exact collection alongside results, even if supplied externally.
    snapshot = dirs["gsea"] / "used_collection.gmt"
    if Path(gmt_path).resolve() != snapshot.resolve():
        snapshot.write_bytes(Path(gmt_path).read_bytes())
    manifest.update({"gmt_sha256": sha256(snapshot), "gmt_source_path": str(gmt_path),
                     "permutations": permutations, "seed": seed, "detection_threshold": detection,
                     "shared_gene_count": len(shared), "comparison_gene_count": len(universe),
                     "FDR": "BH within dataset/cell type; additional pooled BH in all_enrichment.csv",
                     "median_computed": median})
    # Hash artifacts consumed by nb05 to detect mixed/stale notebook outputs.
    manifest["artifact_sha256"] = {str(p.relative_to(output_root)): sha256(p)
        for p in [dirs["rna_concordance"] / "shared_gene_universe.csv", snapshot] +
        [dirs[f"rna_{d}"] / name for d in data for name in
         ("gene_programs.csv", "rna_log1p_cp10k.npz", "cells.csv", "genes.csv")] +
        [dirs["gsea"] / f"{d}_enrichment.csv" for d in data]}
    write_json(Path(output_root) / "manifest.json", manifest)
    write_json(Path(output_root) / "status.json", {"rna_characterization": "complete", "rna_concordance": "pending",
                                                     "biological_review": "required in Colab"})
    return programs, enrichment, universe


def load_characterization(output_root):
    root = Path(output_root)
    status = json.loads((root / "status.json").read_text())
    if status.get("rna_characterization") != "complete":
        raise ValueError("Run nb04 successfully before nb05")
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest["artifact_sha256"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Changed/stale artifact {name}: rerun nb04")
    programs = {d: pd.read_csv(root / f"rna_{d}/gene_programs.csv") for d in ("cite", "multiome")}
    enrichment = {d: pd.read_csv(root / f"gsea/{d}_enrichment.csv") for d in programs}
    universe_table = pd.read_csv(root / "rna_concordance/shared_gene_universe.csv")
    universe = universe_table.loc[universe_table.included_in_comparison, "gene"].tolist()
    data = {d: load_rna(root / f"rna_{d}") for d in programs}
    return manifest, programs, enrichment, universe, data


def run_concordance(output_root, repeats=5, cap=1000):
    if repeats < 1 or cap < 2:
        raise ValueError("Sensitivity analysis requires repeats >=1 and cap >=2")
    root = Path(output_root)
    manifest, programs, enrichment, universe, data = load_characterization(root)
    write_json(root / "status.json", {"rna_characterization": "complete", "rna_concordance": "in_progress"})
    dirs = output_dirs(root)
    out = dirs["rna_concordance"]
    correlations_table, overlap, paired = compare_expression(programs["cite"], programs["multiome"], universe)
    correlations_table.to_csv(out / "expression_correlations.csv", index=False)
    overlap.to_csv(out / "top_gene_overlap.csv", index=False)
    paired.to_csv(out / "paired_gene_profiles.csv", index=False)
    plots.concordance_scatter(paired, out / "figures/expression_scatter")
    pathways = compare_pathways(enrichment["cite"], enrichment["multiome"])
    pathways.to_csv(out / "pathway_concordance.csv", index=False)
    plots.concordance_scatter(pathways, out / "figures/pathway_nes_scatter", "NES_cite", "NES_multiome", None)
    plots.pathway_heatmap(pd.concat(enrichment.values()), out / "figures/pathway_concordance_heatmap")
    pd.DataFrame([dict(cell_type=ct, n_pathways=len(part), pearson=correlations(part.NES_cite, part.NES_multiome)[0],
                       spearman=correlations(part.NES_cite, part.NES_multiome)[1])
                  for ct, part in pathways.groupby("cell_type")],
                 columns=["cell_type", "n_pathways", "pearson", "spearman"]).to_csv(out / "pathway_correlations.csv", index=False)
    sensitivity, overlap_sensitivity, balanced_enrichment = [], [], {}
    for repeat in range(repeats):
        indices = balanced_indices(data["cite"][1].cell_type_harmonized, data["multiome"][1].cell_type_harmonized,
                                   seed=manifest["seed"] + repeat, cap=cap)
        balanced = {}
        for d, idx in zip(("cite", "multiome"), indices):
            x, obs, genes = data[d]
            balanced[d] = characterize(x[idx], genes, obs.cell_type_harmonized.iloc[idx], min_cells=min(cap, manifest["min_cells"]))
            if repeat == 0:
                balanced_enrichment[d] = enrich_programs(balanced[d], read_gmt(dirs["gsea"] / "used_collection.gmt"), universe,
                    permutations=manifest["permutations"], seed=manifest["seed"])
        corr, top, _ = compare_expression(balanced["cite"], balanced["multiome"], universe)
        sensitivity.append(corr.assign(repeat=repeat, seed=manifest["seed"] + repeat))
        overlap_sensitivity.append(top.assign(repeat=repeat))
    pd.concat(sensitivity).to_csv(out / "balanced_expression_sensitivity.csv", index=False)
    pd.concat(overlap_sensitivity).to_csv(out / "balanced_overlap_sensitivity.csv", index=False)
    balanced_pathways = compare_pathways(balanced_enrichment["cite"], balanced_enrichment["multiome"])
    balanced_pathways.to_csv(out / "balanced_pathway_sensitivity.csv", index=False)
    selection = select_programs(pathways)
    stable = balanced_pathways.set_index(["cell_type", "pathway"])
    selection["balanced_direction_supported"] = [
        bool((ct, p) in stable.index and stable.loc[(ct, p), "NES_cite"] > 0 and stable.loc[(ct, p), "NES_multiome"] > 0)
        for ct, p in zip(selection.cell_type, selection.pathway)]
    unstable = selection.selected_for_followup & ~selection.balanced_direction_supported
    selection.loc[unstable, "selected_for_followup"] = False
    selection.loc[unstable, "reason"] = "Positive direction did not survive equal-cell-count sensitivity"
    selection.to_csv(out / "followup_programs.csv", index=False)
    site_sensitivity(data, universe, manifest["min_cells"], out)
    centroid_pca(paired, out)
    report = {"donor": manifest["donor"], "n_shared_types": len(correlations_table),
              "median_expression_pearson": correlations_table.pearson.median(),
              "median_specificity_spearman": correlations_table.specificity_rank_spearman.median(),
              "pathway_classes": pathways.concordance.value_counts().to_dict(),
              "n_followup_candidates": int(selection.selected_for_followup.sum()),
              "balanced_repeats": repeats, "balanced_cap": cap,
              "interpretation": "Descriptive within-donor concordance; biological review required. No matched cross-capture cells.",
              "next_gate": "Inspect mapping, site sensitivity, pathway coverage, effect sizes and leading edges before ATAC."}
    write_json(out / "quantitative_findings.json", report)
    write_json(root / "status.json", {"rna_characterization": "complete", "rna_concordance": "complete",
                                      "biological_review": "required", "atac": "not implemented: RNA validation gate",
                                      "protein": "not implemented: ATAC validation gate"})
    return correlations_table, pathways, selection, report


def site_sensitivity(data, universe, min_cells, out):
    """Compare shared sites separately where >=2 shared cell types remain supported."""
    sites = sorted(set(data["cite"][1].Site) & set(data["multiome"][1].Site))
    results, audit = [], []
    for site in sites:
        counts = {d: obs.loc[obs.Site == site, "cell_type_harmonized"].value_counts()
                  for d, (_, obs, _) in data.items()}
        joint = pd.concat(counts, axis=1).fillna(0)
        types = joint.index[joint.min(axis=1) >= min_cells]
        audit.append(dict(Site=site, n_supported_types=len(types), tested=len(types) >= 2))
        if len(types) < 2:
            continue
        profiles = {}
        for d, (x, obs, genes) in data.items():
            mask = (obs.Site == site) & obs.cell_type_harmonized.isin(types)
            profiles[d] = characterize(x[mask.to_numpy()], genes, obs.loc[mask, "cell_type_harmonized"], min_cells)
        corr, _, _ = compare_expression(profiles["cite"], profiles["multiome"], universe)
        results.append(corr.assign(Site=site))
    pd.DataFrame(audit, columns=["Site", "n_supported_types", "tested"]).to_csv(out / "site_sensitivity_coverage.csv", index=False)
    (pd.concat(results) if results else pd.DataFrame(columns=["cell_type", "Site", "pearson", "spearman"])).to_csv(
        out / "site_expression_sensitivity.csv", index=False)


def centroid_pca(paired, out):
    """Shared uncorrected PCA of population means; not a per-cell integration."""
    frames = []
    for d in ("cite", "multiome"):
        frame = paired.pivot(index="cell_type", columns="gene", values=f"mean_expression_{d}")
        frame.index = d + " | " + frame.index
        frames.append(frame)
    matrix = pd.concat(frames).dropna(axis=1)
    x = matrix.to_numpy()
    x -= x.mean(axis=0)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    if len(s) < 2:
        return
    coords = pd.DataFrame(u[:, :2] * s[:2], index=matrix.index, columns=["PC1", "PC2"])
    coords.to_csv(out / "shared_centroid_pca.csv", index_label="dataset_cell_type")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 7))
    for label, row in coords.iterrows():
        d, ct = label.split(" | ", 1)
        ax.scatter(row.PC1, row.PC2, marker="o" if d == "cite" else "^", color="tab:blue" if d == "cite" else "tab:orange")
        ax.annotate(label, (row.PC1, row.PC2), fontsize=6)
    variance = s ** 2 / (s ** 2).sum() if s.any() else s
    ax.set(xlabel=f"PC1 ({variance[0]:.1%})", ylabel=f"PC2 ({variance[1]:.1%})",
           title="Shared RNA population means: uncorrected exploratory PCA")
    plots.save(fig, out / "figures/shared_centroid_pca")
